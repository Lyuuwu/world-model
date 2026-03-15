from typing import Callable, NamedTuple

import torch
import torch.nn as nn
import numpy as np

from shared.obs_spec import ObsSpec
from shared.math_utils import symlog, symexp
from shared.networks.mlp import MLP, LinearHead, MLPHead
from shared.networks.distributions import BinaryDist, TwoHotCategorical, build_symexp_bins
from shared.networks.losses import MSE, Agg
from shared.registry import register

from .types import WorldModelOutputs, ImaginedTrajectory
from .encoder import DreamerEncoder
from .decoder import DreamerDecoder
from .rssm import RSSM
    
class RewardHead(nn.Module):
    ''' feat > MLP > TwoHot '''
    
    def __init__(
        self,
        feat_dim: int,
        units: int=1024,
        layers: int=5,
        bins: int=255,
        norm: str='rms',
        act: str='silu',
        outscale: float=1.0
    ):
        super().__init__()
        
        self.mlp = MLP(feat_dim, units, layers, norm, act)
        self.head = LinearHead(units, bins, outscale)
        self._bins = build_symexp_bins(bins)
        
    def forward(self, feat: torch.Tensor) -> TwoHotCategorical:
        '''
        feat: (..., feat_dim)
        
        return: TwoHot Dist
        '''
        
        x = self.head(self.mlp(feat))
        bins = self._bins.to(x.device)
        return TwoHotCategorical(x, bins)
    
class ContinueHead(nn.Module):
    ''' feat > MLP > Bernoulli '''
    
    def __init__(
        self,
        feat_dim: int,
        units: int=1024,
        layers: int=5,
        norm: str='rms',
        act: str='silu',
        outscale: float=1.0
    ):
        super().__init__()
        
        self.mlp = MLP(feat_dim, units, layers, norm, act)
        self.head = LinearHead(units, 1, outscale)
        
    def forward(self, feat: torch.Tensor) -> BinaryDist:
        logit = self.head(self.mlp(feat)).squeeze(-1)
        return BinaryDist(logit)
    
class DreamerWorldModel(nn.Module):
    
    def __init__(
        self,
        obs_space: dict[str, ObsSpec],
        action_dim: int,
        
        # --- RSSM ---
        h_dim: int=4096,
        stoch: int=32,
        classes: int=32,
        
        rssm_blocks: int=8,
        rssm_hidden: int=2048,
        
        # --- Encoder ---
        enc_depth: int=64,
        enc_mults: tuple[int, ...]=(2, 3, 4, 4),
        enc_kernel: int=5,
        img_size: tuple[int, int]=(64, 64),
        
        # --- Decoder ---
        dec_depth: int=64,
        dec_mults: tuple[int, ...]=(2, 3, 4, 4),
        dec_kernel: int=5,
        
        # --- MLP ---
        units: int=1024,
        
        # --- Head ---
        head_layers: int=5,
        reward_bins: int=255,
        
        # --- Loss ---
        dyn_scale: float=1.0,
        rep_scale: float=0.1,
        free_nats: float=1.0,
        reward_grad: bool=False,
        
        # --- Continue ---
        contdisc: bool=True,
        horizon: int=333,
        
        # --- Network ---
        norm: str='rms',
        act: str='silu',
        outscale: float=1.0,
        
        **kwargs
    ):
        super().__init__()
        
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.free_nats = free_nats
        self.reward_grad = reward_grad
        self.contdisc = contdisc
        self.horizon = horizon
        
        exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
        self.enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
        self.dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
        
        self.encoder = DreamerEncoder(
            obs_space=self.enc_space,
            depth=enc_depth,
            mults=enc_mults,
            img_size=img_size,
            kernel=enc_kernel,
            units=units,
            norm=norm,
            act=act
        )
        
        self.rssm = RSSM(
            action_dim=action_dim,
            h_dim=h_dim,
            hidden=rssm_hidden,
            stoch=stoch,
            classes=classes,
            blocks=rssm_blocks,
            token_dim=self.encoder.token_dim,
            norm=norm,
            act=act,
            outscale=outscale
        )
        
        feat_dim = self.rssm.feat_dim
        
        self.decoder = DreamerDecoder(
            obs_space=self.dec_space,
            h_dim=h_dim,
            stoch=stoch,
            classes=classes,
            img_size=img_size,
            depth=dec_depth,
            mults=dec_mults,
            kernel=dec_kernel,
            units=units,
            norm=norm,
            act=act,
            outscale=outscale
        )
        
        self.reward_head = RewardHead(
            feat_dim=feat_dim,
            units=units,
            layers=head_layers,
            bins=reward_bins,
            norm=norm,
            act=act,
            outscale=outscale
        )
        
        self.continue_head = ContinueHead(
            feat_dim=feat_dim,
            units=units,
            layers=head_layers,
            norm=norm,
            act=act,
            outscale=outscale
        )
        
        # --- TEMP ---
        # FIX LATER
        self._loss_scales = {
            'dyn': dyn_scale,
            'rep': rep_scale,
            'rew': 1.0,
            'con': 1.0
        }
        
        for key in self.dec_space:
            self._loss_scales[key] = 1.0
            
    @property
    def feat_dim(self) -> int:
        return self.rssm.feat_dim
    
    def _preprocess_target(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        targets = {}
        for key in self.dec_space:
            space = self.dec_space[key]
            value = obs[key]
            
            if space.is_image:
                # FIX: 看之後 wrapper
                targets[key] = value.float() / 255.0
            else:
                targets[key] = value
                
        return targets
    
    def _make_continue_target(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        con = (~obs['is_terminal']).float()
        
        if self.contdisc:
            con = con * (1.0 - 1.0 / self.horizon)
            
        return con
    
    def _get_feat(
        self,
        rssm_outputs: dict[str, torch.Tensor],
        detach: bool=False
    ) -> torch.Tensor:
        
        feat = self.rssm.get_feat(rssm_outputs)
        if detach:
            feat = feat.detach()
        return feat
    
    def observe(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reset: torch.Tensor,
        state: dict[str, torch.Tensor] | None=None
    ) -> WorldModelOutputs:
        '''
        1. Encoder: obs > tokens (B, T, token_dim)
        2. RSSM observe: tokens + actions > state sequence
        3. feat extraction
        '''
        
        tokens = self.encoder(obs)
        
        final_state, rssm_outputs = self.rssm.observe(
            tokens, action, reset, state
        )
        
        feat = self._get_feat(rssm_outputs)
        
        return WorldModelOutputs(
            state=final_state,
            feat=feat,
            rssm_outputs=rssm_outputs
        )
        
    def compute_loss(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reset: torch.Tensor,
        state: dict[str, torch.Tensor] | None=None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        B, T = reset.shape
        losses:  dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        
        # observe
        wm_out = self.observe(obs, action, reset, state)
        
        # KL losses
        dyn_loss, rep_loss, kl_metrics = self.rssm.kl_loss(
            wm_out.rssm_outputs, free_nats=self.free_nats
        )
        losses['dyn'] = dyn_loss
        losses['rep'] = rep_loss
        metrics.update(kl_metrics)
        
        # decoder reconstruction losses
        feat = wm_out.feat
        recons = self.decoder(wm_out.rssm_outputs)
        targets = self._preprocess_target(obs)
        
        for key, dist in recons.items():
            target = targets[key].detach()
            losses[key] = dist.loss(target)
            
        # reward loss
        rew_feat = feat if self.reward_grad else feat.detach()
        rew_dist = self.reward_head(rew_feat)
        losses['rew'] = -rew_dist.log_prob(obs['reward'])   # (B, T)
        
        con_target = self._make_continue_target(obs)
        con_dist = self.continue_head(feat)
        losses['con'] = con_dist.loss(con_target)
        
        # shape assertions
        shapes = {k: v.shape for k, v in losses.items()}
        assert all(s == (B, T) for s in shapes.values()), \
            f'Loss shape mismatch: expected ({B}, {T}), got {shapes}'
            
        # scale and sum
        assert set(losses.keys()) == set(self._loss_scales.keys()), \
            f'Scale/loss key mismatch: {sorted(losses.keys())} vs {sorted(self._loss_scales.keys())}'
            
        total_loss = sum(
            losses[k].mean() * self._loss_scales[k]
            for k in losses
        )
        
        for k, v in losses.items():
            metrics[f'loss/{k}'] = v.mean().detach()
        metrics['loss/total'] = total_loss.detach()
        metrics['reward/pred_mean'] = rew_dist.mean.mean().detach()
        metrics['continue/prob_mean'] = con_dist.prob(1.0).mean().detach()
        
        return total_loss, losses, metrics
    
    def imagine(
        self,
        wm_out: WorldModelOutputs,
        policy_fn,
        horizon: int,
        K: int | None=None,
        ac_grads: bool=False
    ) -> ImaginedTrajectory:
        
        # --- extraction ---
        start_states, start_feat = self.get_start_states(wm_out.rssm_outputs, K)
        N = start_states['deter'].shape[0]
        
        # --- rssm imagine ---
        img_feats_dict, img_actions = self.rssm.imagine(start_states, policy_fn, horizon)
        # img_feats_dict:   {deter: (N, H), stoch: (N, H), logit: (N, H)}
        # img_actions:      (N, H, action_dim)
        
        # --- feature tensor ---
        img_feat = self.rssm.get_feat(img_feats_dict)
        
        if not ac_grads:
            start_feat = start_feat.detach()
            img_feat = img_feat.detach()
        full_feat = torch.cat([start_feat, img_feat], dim=1)    # (N, H+1, feat_dim)
        
        # --- Last action ---
        with torch.no_grad():
            last_act = policy_fn(full_feat[:, -1])
            if last_act.dim() == 1:
                last_act = last_act.unsqueeze(-1)
            last_act = last_act.unsqueeze(1)
        full_action = torch.cat([img_actions, last_act], dim=1) # (N, H+1, act_dim)
        
        # --- reward / continue heads ---
        with torch.no_grad():
            reward = self.reward_head(full_feat).mean           # (N, H+1)
            cont   = self.continue_head(full_feat).prob(1.0)    # (N, H+1)
        
        return ImaginedTrajectory(
            feat=full_feat,
            reward=reward,
            cont=cont
        )
        
    
    def initial_state(self, batch_size: int, device: torch.device='cpu'):
        return self.rssm.initial_state(batch_size, device)
    
    def get_start_states(
        self,
        rssm_outputs: dict[str, torch.Tensor],
        K: int | None=None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        B, T = rssm_outputs['deter'].shape[:2]
        K = min(K or T, T)
            
        states = {}
        for key in ('deter', 'stoch'):
            val = rssm_outputs[key][:, -K:]
            states[key] = val.reshape(B*K, *val.shape[2:])
        
        feat = self.rssm.get_feat(states).squeeze(1)   # (B*K, feat_dim) -> (B*K, 1, feat_dim)
        
        return states, feat
    
register('world_model', 'dreamerv3')(DreamerWorldModel)