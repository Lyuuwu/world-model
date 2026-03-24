from typing import Any
 
import torch
import torch.nn as nn
import numpy as np
 
from shared.registry import register
from shared.optimizer import LaProp
 
from .types import DreamerV3Config, WorldModelOutputs, ImaginedTrajectory
from .world_model import DreamerWorldModel
from .actor_critic import DreamerActorCritic
 
 
@register('agent', 'dreamerv3')
class DreamerV3Agent(nn.Module):
    def __init__(
        self,
        obs_space: dict[str, Any],
        action_dim: int,
        config: DreamerV3Config,
    ):
        super().__init__()
 
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.config = config
 
        # --- World Model ---
        wm_kwargs = config.wm_kwargs or {}
 
        wm_kwargs.setdefault('reward_grad', config.reward_grad)
 
        self.world_model = DreamerWorldModel(
            obs_space=obs_space,
            action_dim=action_dim,
            horizon=config.horizon,
            contdisc=config.contdisc,
            **wm_kwargs,
        )

        if config.use_compile:
            try:
                self.world_model = torch.compile(
                    self.world_model,
                    mode=config.compile_mode,
                    fullgraph=False,
                )
                print('[Agent] torch.compile enabled '
                      f'(mode={config.compile_mode})')
            except Exception as e:
                print(f'[Agent] torch.compile failed: {e}, proceeding without')
 
        # --- Actor-Critic ---
        ac_kwargs = config.ac_kwargs or {}
 
        self.actor_critic = DreamerActorCritic(
            feat_dim=self.world_model.feat_dim,
            action_dim=action_dim,
            discrete=config.discrete,
            horizon=config.horizon,
            contdisc=config.contdisc,
            **ac_kwargs,
        )
 
        # --- Loss scales ---
        self.scales = self._build_loss_scales(config, obs_space)
 
        # --- Optimizer ---
        self.optimizer = LaProp(
            self.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            agc=config.agc,
        )
 
    # --- Loss scales ---
 
    def _build_loss_scales(
        self,
        config: DreamerV3Config,
        obs_space: dict[str, Any],
    ) -> dict[str, float]:
        raw = config.loss_scales or {
            'dyn': 1.0, 'rep': 0.1, 'rec': 1.0,
            'rew': 1.0, 'con': 1.0,
            'policy': 1.0, 'value': 1.0, 'repval': 0.3,
        }
        scales = dict(raw)
 
        # rec -> 每個 obs key
        rec_scale = scales.pop('rec', 1.0)
        exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
        for key in obs_space:
            if key not in exclude:
                scales[key] = rec_scale
 
        # repval_loss 關閉時移除
        if not config.repval_loss and 'repval' in scales:
            scales.pop('repval')
 
        return scales
 
    # --- State init ---
 
    def initial_state(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ) -> dict[str, torch.Tensor]:
        return self.world_model.initial_state(batch_size, device)
 
    def initial_prevact(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.action_dim, device=device)
 
    # --- Policy (inference) ---
 
    @torch.no_grad()
    def policy(
        self,
        obs: dict[str, torch.Tensor],
        state: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # encoder → tokens
        obs_seq = {
            k: v.unsqueeze(1) for k, v in obs.items()
            if k not in ('is_first', 'is_last', 'is_terminal', 'reward')
        }
        tokens = self.world_model.encoder(obs_seq)
 
        # RSSM observe step
        new_state, _ = self.world_model.rssm.observe_step(
            state, tokens.squeeze(1), prev_action, is_first
        )
 
        # feat → sample action
        feat = self.world_model.rssm.get_feat(new_state)
        action = self.actor_critic.policy_head(feat).sample()
 
        return action, new_state
 
    # --- Train step ---
 
    def train_step(
        self,
        data: dict[str, torch.Tensor],
        device_type: str = 'cuda',
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, float]:
        '''
        單步 training: forward + backward + optimizer step.
        '''
        # build obs & prevact
        obs = {k: data[k] for k in self.obs_space if k in data}
        obs['reward'] = data['reward']
        obs['is_first'] = data['is_first']
        obs['is_last'] = data['is_last']
        obs['is_terminal'] = data['is_terminal']
 
        action = data['action']                  # (B, T, action_dim)
        prevact = self._make_prevact(action)     # (B, T, action_dim)
 
        # forward
        with torch.autocast(device_type=device_type, dtype=compute_dtype):
            total_loss, losses, metrics = self._compute_loss(obs, prevact)
 
        # backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
 
        # EMA update
        self.actor_critic.update_slow_target()
 
        return metrics
 
    # ── Core loss computation ──────────────────────────────
 
    def _compute_loss(
        self,
        obs: dict[str, torch.Tensor],
        prevact: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:
        B, T = obs['is_first'].shape
        config = self.config
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, float] = {}
 
        # --- world model forward ---
        wm_out: WorldModelOutputs = self.world_model.observe(
            obs=obs, action=prevact, reset=obs['is_first'],
        )
        wm_total, wm_losses, wm_metrics = self.world_model.compute_loss(
            obs=obs, wm_out=wm_out,
        )
        losses.update(wm_losses)
        metrics.update(wm_metrics)
 
        # --- imagination ---
        K = min(config.imag_last or T, T)
        H = config.imag_horizon
 
        policy_fn = self.actor_critic.get_policy_fn()
        traj: ImaginedTrajectory = self.world_model.imagine(
            wm_out=wm_out,
            policy_fn=policy_fn,
            horizon=H,
            K=K,
            ac_grads=config.ac_grads,
        )
 
        imag_total, imag_losses, imag_metrics, imag_ret = \
            self.actor_critic.compute_imag_loss(traj, update_norm=True)
 
        for k, v in imag_losses.items():
            losses[k] = v.mean(dim=1).reshape(B, K)
        metrics.update(imag_metrics)
 
        # --- replay value loss ---
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
                update_norms=True,
            )
            losses.update(repl_losses)
            metrics.update(repl_metrics)
 
        # --- aggregate ---
        assert set(losses.keys()) == set(self.scales.keys()), (
            f'Loss/scale key mismatch: '
            f'losses={sorted(losses.keys())} vs '
            f'scales={sorted(self.scales.keys())}'
        )
 
        total = torch.zeros(1, device=prevact.device)
        for k, v in losses.items():
            s = self.scales[k]
            if s > 0:
                total = total + s * v.mean()
                metrics[f'loss/{k}'] = v.mean().detach().item()
                metrics[f'scale/{k}'] = s
 
        metrics['loss/total'] = total.detach().item()
        return total, losses, metrics
 
    def _make_prevact(self, action: torch.Tensor) -> torch.Tensor:
        '''action[:, 1:] = prevact[:, 1:], prevact[:, 0] = 0'''
        B, T, A = action.shape
        zero = torch.zeros(B, 1, A, device=action.device, dtype=action.dtype)
        return torch.cat([zero, action[:, :-1]], dim=1)
 