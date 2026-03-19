from typing import Any

import torch
import torch.nn as nn
import numpy as np
 
from shared.registry import register

from .types import DreamerV3Config
from .world_model import DreamerWorldModel, WorldModelOutputs, ImaginedTrajectory
from .actor_critic import DreamerActorCritic

class DreamerV3Agent(nn.Module):
    def __init__(
        self,
        obs_space: dict[str, Any],
        action_dim: int,
        config: DreamerV3Config
    ):
        super().__init__()
        
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.config = config
        
        # --- world model ---
        wm_kwargs = config.wm_kwargs or {}
        self.world_model = DreamerWorldModel(
            obs_space=obs_space,
            action_dim=action_dim,
            horizon=config.horizon,
            contdisc=config.contdisc,
            **wm_kwargs
        )
        
        # --- actor-critic ---
        ac_kwargs = config.ac_kwargs or {}
        self.actor_critic = DreamerActorCritic(
            feat_dim=self.world_model.feat_dim,
            action_dim=action_dim,
            discrete=config.discrete,
            horizon=config.horizon,
            contdisc=config.contdisc,
            **ac_kwargs
        )
        
        # --- loss scale ---
        
        self.scales = self._build_loss_scales(config, obs_space)
        
        # --- optimizer ---
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )
        
    def _build_loss_scales(
        self,
        config: DreamerV3Config,
        obs_space: dict[str, Any]
    ) -> dict[str, float]:
        raw = config.loss_scales or {
            'dyn': 1.0, 'rep': 0.1, 'rec': 1.0,
            'rew': 1.0, 'con': 1.0,
            'policy': 1.0, 'value': 1.0, 'repval': 0.3
        }
        scales = dict(raw)
        
        rec_scale = scales.pop('rec')
        exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
        for key in obs_space:
            if key not in exclude:
                scales[key] = rec_scale
                
        if not config.repval_loss and 'repval' in scales:
            scales.pop('repval')
            
        return scales
    
    def initial_state(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> dict[str, torch.Tensor]:
        return self.world_model.initial_state(batch_size, device)
    
    def initial_prevact(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ):
        # one-hot action → 全零 (no action selected)
        return torch.zeros(batch_size, self.action_dim, device=device)
    
    @torch.no_grad()
    def policy(
        self,
        obs: dict[str, torch.Tensor],
        state: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        is_first: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        # --- encoder 加工成 token ---
        obs_seq = {k: v.unsqueeze(1) for k, v in obs.items()
                   if k not in ('is_first', 'is_last', 'is_terminal', 'reward')}
        tokens = self.world_model.encoder(obs_seq)
        
        # --- 走一步 > 更新 state ---
        reset = is_first
        new_state, rssm_out = self.world_model.rssm.observe_step(
            state, tokens.squeeze(1), prev_action, reset
        )
        
        # --- 建 feat 然後 sample action ---
        feat = self.world_model.rssm.get_feat(new_state)
        action = self.actor_critic.policy_head(feat).sample()
        
        return action, new_state
    
    def train_step(
        self,
        data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        
        # --- 建構 prevact ---
        obs = {k: data[k] for k in self.obs_space if k in data}
        obs['reward'] = data['reward']
        obs['is_first'] = data['is_first']
        obs['is_last'] = data['is_last']
        obs['is_terminal'] = data['is_terminal']
        
        action = data['action']                 # (B, T, action_dim)
        prevact = self._make_prevact(action)    # (B, T, action_dim)
        
        # --- forward ---
        total_loss, losses, metrics = self._compute_loss(obs, prevact)
        
        # --- backward + optimizer step ---
        self.optimizer.zero_grad()
        total_loss.backward()
        
        if self.config.agc > 0:
            self._agc_clip(self.config.agc)
            
        self.optimizer.step()
        
        # --- 更新 EMA ---
        self.actor_critic.update_slow_target()
        
        return metrics
    
    def _compute_loss(
        self,
        obs: dict[str, torch.Tensor],
        prevact: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        
        B, T = obs['is_first'].shape
        config = self.config
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        
        # --- world model forward ---
        wm_out: WorldModelOutputs = self.world_model.observe(
            obs=obs,
            action=prevact,
            reset=obs['is_first']
        )
        
        wm_total, wm_losses, wm_metrics = self.world_model.compute_loss(
            obs=obs,
            wm_out=wm_out
        )
        
        losses.update(wm_losses)
        metrics.update(wm_metrics)
        
        # --- imagination ---
        K = min(config.imag_last or T, T)
        H = config.imag_horizon
        
        start_states, start_feat = self.world_model.get_start_states(
            wm_out.rssm_outputs, K
        )
        N = B * K
        
        policy_fn = self.actor_critic.get_policy_fn()   # 給 imagination 呼叫用

        # world model imagine 產出軌跡
        traj: ImaginedTrajectory = self.world_model.imagine(
            wm_out=wm_out,
            policy_fn=policy_fn,
            horizon=H,
            K=K,
            ac_grads=config.ac_grads
        )
        
        # Actor 去跟 imagine 出來的軌跡玩
        imag_total, imag_losses, imag_metrics, imag_ret = \
            self.actor_critic.compute_imag_loss(traj, update_norm=True)
            
        # raw loss: (B*K, H)
        # 取 mean 變成 (B*K, )
        # reshape 成 (B, K)
        for k, v in imag_losses.items():
            losses[k] = v.mean(dim=1).reshape(B, K)
            
        metrics.update(imag_metrics)
        
        if config.repval_loss:
            feat = wm_out.feat
            if not config.repval_grad:
                feat = feat.detach()
                
            bootstrap = imag_ret[:, 0].reshape(B, K)

            repl_losses, repl_metrics = self.actor_critic.compute_repl_loss(
                replay_feat=feat,
                is_last=obs['is_last'],
                is_terminal=obs['is_terminal'],
                reward=obs['reward'],
                bootstrap=bootstrap,
                K=K,
                update_norms=True
            )
            
            losses.update(repl_losses)
            metrics.update(repl_metrics)
            
        assert set(losses.keys()) == set(self.scales.keys()), (
            f'Loss/scale key mismatch:\n'
            f'  losses: {sorted(losses.keys())}\n'
            f'  scales: {sorted(self.scales.keys())}'
        )
        
        total_loss = sum(
            losses[k].mean() * self.scales[k]
            for k in self.scales
        )
        
        for k, v in losses.items():
            metrics[f'loss/{k}'] = v.mean().detach()
        metrics['loss/total'] = total_loss.detach()
        
        return total_loss, losses, metrics
    
    def _make_prevact(
        self,
        action: torch.Tensor
    ) -> torch.Tensor:
        B, T, D = action.shape
        zero = torch.zeros(B, 1, D, device=action.device, dtype=action.dtype)
        prevact = torch.cat([zero, action[:, :-1]], dim=1)
        return prevact
    
    def _agc_clip(self, clip_factor: float, eps: float=1e-3):
        for param in self.parameters():
            if param.grad is None:
                continue
            
            grad_norm = param.grad.data.norm(2)
            param_norm = param.data.norm(2)
            
            max_norm = clip_factor * param_norm.clamp(min=eps)
            
            if grad_norm > max_norm:
                param.grad.data.mul_(max_norm / grad_norm.clamp(min=1e-6))

    ''' 要用到再說 '''
    # @torch.no_grad()
    # def report(
    #     self,
    #     data: dict[str, torch.Tensor],
    #     num_videos: int=6
    # ) -> dict[str, torch.Tensor]:
    #     obs = {k: data[k] for k in self.obs_space if k in data}
    #     obs['reward'] = data['reward']
    #     obs['is_first'] = data['is_first']
    #     obs['is_last'] = data['is_last']
    #     obs['is_terminal'] = data['is_terminal']
        
    #     action = data['action']
    #     prevact = self._make_prevact(action)
        
    #     B, T = obs['is_first'].shape
    #     half = T // 2
    #     num_videos = min(num_videos, B)
    
register('agent', 'dreamerv3')(DreamerV3Agent)