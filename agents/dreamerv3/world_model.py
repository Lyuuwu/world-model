from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from shared.networks.mlp import MLP, LinearHead
from shared.distributions import BinaryDist, TwoHotCategorical, build_symexp_bins
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
        self.register_buffer('_bins', build_symexp_bins(bins), persistent=False)
        
    def forward(self, feat: torch.Tensor) -> TwoHotCategorical:        
        x = self.head(self.mlp(feat))
        bins = self._bins.to(x.device)
        return TwoHotCategorical(x, bins)
    
    if TYPE_CHECKING:
        def __call__(self, feat: torch.Tensor) -> TwoHotCategorical:
            '''
            forward in:
                - feat: (..., feat_dim)
            
            return:
                - TwoHot Dist
            '''
            ...
    
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
    
    if TYPE_CHECKING:
        def __call__(self, feat: torch.Tensor) -> BinaryDist:
            '''
            forward in:
                - feat: (..., feat_dim)
            
            return:
                - BinaryDist
            '''
            ...
    
@register('world_model', 'dreamerv3')
class DreamerWorldModel(nn.Module):
    
    def __init__(
        self,        
        encoder: nn.Module,
        decoder: nn.Module,
        rssm: nn.Module,
        reward_head: nn.Module,
        continue_head: nn.Module,
        
        # --- Loss ---
        free_nats: float=1.0,
        reward_grad: bool=False,
        
        # --- Continue ---
        contdisc: bool=True,
        horizon: int=333
    ):
        super().__init__()
        
        self.free_nats = free_nats
        self.reward_grad = reward_grad
        self.contdisc = contdisc
        self.horizon = horizon
        
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.reward_head = reward_head
        self.continue_head = continue_head
            
    @property
    def feat_dim(self) -> int:
        return self.rssm.feat_dim
    
    def _preprocess_target(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        targets = {}
        for key in self.decoder.obs_space:
            space = self.decoder.obs_space_space[key]
            value = obs[key]
            
            if space.is_image:
                targets[key] = value.float() / 255.0
            else:
                targets[key] = value
                
        return targets
    
    def _make_continue_target(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        con = (~obs['is_terminal']).float()
        
        if self.contdisc:
            con *= 1 - 1 / self.horizon
        
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
        params:
            - action: (B, T, action_dim)
            - reset: (B, T, )
            - state: {deter, stoch} 

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
        wm_out: WorldModelOutputs,
        state: dict[str, torch.Tensor] | None=None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        B, T = obs['is_first'].shape
        losses:  dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        
        # KL losses
        dyn_loss, rep_loss, kl_metrics = self.rssm.kl_loss(
            wm_out.rssm_outputs, free_nats=self.free_nats
        )
        losses['dyn'] = dyn_loss
        losses['rep'] = rep_loss
        metrics.update(kl_metrics)
        
        # decoder reconstruction losses
        recons = self.decoder(wm_out.rssm_outputs)
        targets = self._preprocess_target(obs)
        
        for key, dist in recons.items():
            losses[key] = dist.loss(targets[key].detach())
            
        # reward loss
        feat = wm_out.feat
        rew_feat = feat if self.reward_grad else feat.detach()
        rew_dist = self.reward_head(rew_feat)
        losses['rew'] = rew_dist.loss(obs['reward'])   # (B, T)
        
        con_target = self._make_continue_target(obs)
        con_dist = self.continue_head(feat)
        losses['con'] = con_dist.loss(con_target)
        
        # shape assertions
        assert all(v.shape == (B, T) for v in losses.values())
        
        for k, v in losses.items():
            metrics[f'loss/{k}'] = v.mean().detach()
        metrics['reward/pred_mean'] = rew_dist.mean.mean().detach()
        metrics['continue/prob_mean'] = con_dist.prob(1.0).mean().detach()
        
        return losses, metrics
    
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
            cont=cont,
            action=full_action
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
        
        feat = self.rssm.get_feat(states).unsqueeze(1)   # (B*K, feat_dim) -> (B*K, 1, feat_dim)
        
        return states, feat
