from typing import Callable, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from shared.math_utils import Normalizer
from shared.networks.mlp import MLP, LinearHead
from shared.distributions import (
    StraightThroughCategorical, TwoHotCategorical, NormalDist, build_symexp_bins
)
from shared.losses import Agg
from shared.registry import register

from .types import ImaginedTrajectory

class PolicyHead(nn.Module):
    '''
    feat > action distribution
    
    discrete:   MLP > Categorical logits + unimix \n
    continuous: MLP > (mean, raw_std) > Normalize
    
    '''
    
    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        discrete: bool=True,
        units: int=1024,
        layers: int=5,
        norm: str='rms',
        act: str='silu',
        outscale: float=0.01,
        unimix: float=0.01,
        minstd: float=0.1,
        maxstd: float=1.0
    ):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        self.unimix = unimix
        self.minstd = minstd
        self.maxstd = maxstd
        
        self.mlp = MLP(feat_dim, units, layers, norm, act)
        
        if discrete:
            self.head = LinearHead(units, action_dim, outscale)
        else:
            self.mean_head = LinearHead(units, action_dim, outscale)
            self.stddev_head = LinearHead(units, action_dim, outscale)
        
    def forward(self, feat: torch.Tensor) -> StraightThroughCategorical | Agg:
        '''
        feat: (..., feat_dim)
        
        discrete:   return stc \n
        continuous: return Agg(NormalDist)
        '''
        
        x = self.mlp(feat)
        
        if self.discrete:
            logits = self.head(x)
            return StraightThroughCategorical(logits, unimix_ratio=self.unimix)
        else:
            raw_mean    = self.mean_head(x)
            raw_std     = self.stddev_head(x)
            
            mean = torch.tanh(raw_mean)
            stddev = ((self.maxstd - self.minstd)
                      * torch.sigmoid(raw_std + 2.0)
                      + self.minstd)
            
            dist = NormalDist(mean, stddev)
            dist.minent = NormalDist(
                torch.zeros_like(mean),
                torch.full_like(mean, self.minstd)
            ).entropy()
            dist.maxent = NormalDist(
                torch.zeros_like(mean),
                torch.full_like(mean, self.maxstd)
            ).entropy()
            
            return Agg(dist, agg_dims=1)
        
    if TYPE_CHECKING:
        def __call__(self,  feat: torch.Tensor) -> StraightThroughCategorical | Agg:
            '''
            forward in:
                - feat (..., feat_dim)
            
            forward out:
                - discrete: STC
                - continuous: Agg(Normal)
            '''
            ...
            
    def sample(self, feat: torch.Tensor) -> torch.Tensor:        
        dist = self.forward(feat)    
        return dist.sample()

class ValueHead(nn.Module):
    '''
    feat > twohot dist
    '''
    
    def __init__(
        self,
        feat_dim: int,
        units: int=1024,
        layers: int=5,
        bins: int=255,
        norm: str='rms',
        act: str='silu',
        outscale: float=0.0
    ):
        super().__init__()
        self.mlp = MLP(feat_dim, units, layers, norm, act)
        self.head = LinearHead(units, bins, outscale)
        self.register_buffer('_bins', build_symexp_bins(bins), persistent=False)
        
    def forward(self, feat: torch.Tensor):
        x = self.head(self.mlp(feat))
        bins = self._bins.to(x.device)
        return TwoHotCategorical(x, bins)
    
    if TYPE_CHECKING:
        def __call__(self, feat: torch.Tensor):
            '''
            forward in:
                - feat: (..., feat_dim)

            forward out:
                - TwoHot
            '''
            ...
    
class SlowValueTarget(nn.Module):
    '''
    ValueHead 的 EMA
    '''
    
    def __init__(self, value_head: ValueHead, decay: float=0.98):
        super().__init__()
        self.decay = decay
        
        self.target = copy.deepcopy(value_head)
        for p in self.target.parameters():
            p.requires_grad_(False)
            
    @torch.no_grad()
    def update(self, source: ValueHead) -> None:
        for tp, sp in zip(self.target.parameters(), source.parameters()):
            tp.data.mul_(self.decay).add_(sp.data, alpha=1-self.decay)
    
    def forward(self, feat: torch.Tensor) -> TwoHotCategorical:
        return self.target(feat)
    
    if TYPE_CHECKING:
        def __call__(self, feat: torch.Tensor) -> TwoHotCategorical:
            '''
            forward in:
                - (..., feat_dim)

            forward out:
                - TwoHot
            '''
            ...

def lambda_return(
    last: torch.Tensor,
    term: torch.Tensor,
    rew: torch.Tensor,
    val: torch.Tensor,
    boot: torch.Tensor,
    disc: float,
    lam: float=0.95
) -> torch.Tensor:
    rets = [boot[:, -1]]
    live = (1 - term.float())[:, 1:] * disc
    cont = (1 - last.float())[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * val[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    
    return torch.stack(list(reversed(rets))[:-1], dim=1)
@register('actor_critic', 'dreamerv3')
class DreamerActorCritic(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        discrete: bool=True,
        
        # --- network ---
        units: int=1024,
        layers: int=5,
        bins: int=255,
        norm: str='rms',
        act: str='silu',
        
        # --- policy ---
        policy_outscale: float = 1.0,
        policy_unimix: float=0.01,
        actent: float=3e-4,
        
        minstd: float=0.1,
        maxstd: float=1.0,
        
        # --- value ---
        value_outscale: float=0.0,
        slow_decay: float=0.98,
        slowreg: float=1.0,
        slowtar: bool=True,
        
        # --- returns ---
        horizon: int=333,
        contdisc: bool=True,
        lam: float=0.95,
        
        # --- normalization ---
        retnorm_decay: float=0.99,
        retnorm_limit: float=1.0,
        retnorm_plow: float=5.0,
        retnorm_phigh: float=95.0,
        valnorm_decay: float=0.99,
        advnorm_decay: float=0.99,
        valnorm_enable: bool=False,
        advnorm_enable: bool=False,
        
        # --- loss scales ---
        policy_scale: float=1.0,
        value_scale: float=1.0,
    ):
        super().__init__()
        
        self.actent = actent
        self.slowreg = slowreg
        self.slowtar = slowtar
        self.horizon = horizon
        self.contdisc = contdisc
        self.lam = lam
        self.policy_scale = policy_scale
        self.value_scale = value_scale
        self.discrete = discrete
        
        # --- heads ---
        
        self.policy_head = PolicyHead(
            feat_dim, action_dim, discrete,
            units, layers, norm, act,
            unimix=policy_unimix, minstd=minstd, maxstd=maxstd, outscale=policy_outscale
        )
        
        self.value_head = ValueHead(
            feat_dim, units, layers, bins, norm, act,
            outscale=value_outscale
        )
        
        self.slow_value = SlowValueTarget(self.value_head, slow_decay)
        
        # --- normalizers ---
        
        self.retnorm = Normalizer(
            decay=retnorm_decay, use_percentile=True,
            percentile_low=retnorm_plow, percentile_high=retnorm_phigh,
            max_limit=retnorm_limit
        )
        
        self.valnorm = Normalizer(decay=valnorm_decay, use_percentile=False, enable=valnorm_enable)
        self.advnorm = Normalizer(decay=advnorm_decay, use_percentile=False, enable=advnorm_enable)
    
    def forward(self, feat: torch.Tensor, train: bool=True) -> torch.Tensor:
        if train:
            return self.policy_head(feat).sample()
        else:
            return self.policy_head(feat).mode
    
    if TYPE_CHECKING:
        def __call__(self, feat: torch.Tensor) -> torch.Tensor:
            '''
            根據 policy 與 feature 來決定 action

            forward in:
                - feat: (..., feat_dim)

            forward out:
                - action (..., )
            '''

    def get_policy_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def policy_fn(feat: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.policy_head(feat).sample()
        return policy_fn
    
    def compute_imag_loss(
            self,
            traj: ImaginedTrajectory,
            update_norm: bool=True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        '''
        return (total loss, losses, metrics, ret)
        '''
        
        # --- value prediction ---
        
        voffset, vscale = self.valnorm.stats()
        val_dist: TwoHotCategorical = self.value_head(traj.feat)
        slow_val_dist: TwoHotCategorical = self.slow_value(traj.feat)
        val = val_dist.mean * vscale + voffset
        slowval = slow_val_dist.mean * vscale + voffset
        tarval = slowval if self.slowtar else val

        # --- discount weight ---
        
        disc = 1.0 if self.contdisc else 1.0 - 1.0 / self.horizon
        weight = torch.cumprod(disc * traj.cont.float(), dim=1) / disc  # (N, H+1)

        # --- lambda return ---
        
        last = torch.zeros_like(traj.cont)
        term = 1 - traj.cont
        ret = lambda_return(last=last,
                            term=term,
                            rew=traj.reward,
                            val=tarval,
                            boot=tarval,
                            disc=disc,
                            lam=self.lam) # (N, H)

        # --- Advantage ---

        roffset, rscale = self.retnorm(ret, update_norm)
        adv = (ret - tarval[:, :-1]) / rscale
        aoffset, ascale = self.advnorm(adv, update_norm)
        adv_normed = (adv - aoffset) / ascale

        # --- Policy Loss ---

        policy_dist = self.policy_head(traj.feat)
        logpi = policy_dist.log_prob(traj.action.detach())[:, :-1]
        ent = policy_dist.entropy()[:, :-1]

        policy_loss = weight[:, :-1].detach() * -(
            logpi * adv_normed.detach() + self.actent * ent
        )

        # --- Value Loss ---

        voffset, vscale = self.valnorm(ret, update_norm)
        tar_normed = (ret - voffset) / vscale
        tar_padded = torch.cat([tar_normed, 0 * tar_normed[:, -1:]], dim=1)

        value_loss = weight[:, :-1].detach() * (
            val_dist.loss(tar_padded.detach()) +
            self.slowreg * val_dist.loss(slow_val_dist.mean.detach())
        )[:, :-1]

        # --- Aggregate ---

        total = policy_loss.mean() * self.policy_scale + value_loss.mean() * self.value_scale

        losses = {
            'policy': policy_loss,
            'value':  value_loss
        }

        metrics = self._build_metrics(
            adv, traj.reward, traj.cont,
            ret, val, tar_normed, weight, slowval,
            policy_dist, ent, traj.action[:, :-1], roffset, rscale
        )
        metrics['loss/policy'] = policy_loss.mean().detach()
        metrics['loss/value']  = value_loss.mean().detach()
        metrics['loss/total']  = total.detach()

        return total, losses, metrics, ret
    
    def compute_repl_loss(
            self,
            replay_feat: torch.Tensor,
            is_last: torch.Tensor,
            is_terminal: torch.Tensor,
            reward: torch.Tensor,
            bootstrap: torch.Tensor,
            K: int,
            update_norms: bool=True
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        
        feat = replay_feat[:, -K:]
        last = is_last[:, -K:]
        term = is_terminal[:, -K:]
        rew = reward[:, -K:]
        boot = bootstrap
        
        voffset, vscale = self.valnorm.stats()
        val_dist = self.value_head(feat)
        slow_val_dist = self.slow_value(feat)
        val = val_dist.mean * vscale + voffset
        slowval = slow_val_dist.mean * vscale + voffset
        tarval = slowval if self.slowtar else val

        disc = 1.0 - 1.0 / self.horizon

        weight = (~last).float()

        ret = lambda_return(last, term, rew, tarval, boot, disc, self.lam)   # (B, K-1)

        voffset, vscale = self.valnorm(ret, update=update_norms)
        ret_normed = (ret - voffset) / vscale
        ret_padded = torch.cat([ret_normed, 0 * ret_normed[:, -1:]], dim=1)         # (B, K)
        
        repval_loss = weight[:, :-1] * (
            val_dist.loss(ret_padded.detach()) +
            self.slowreg * val_dist.loss(slow_val_dist.mean.detach())
        )[:, :-1]
        
        return {'repval': repval_loss}, {}

    def update_slow_target(self) -> None:
        self.slow_value.update(self.value_head)

    def _build_metrics(
            self,
            adv: torch.Tensor,
            rew: torch.Tensor,
            con: torch.Tensor,
            ret: torch.Tensor,
            val: torch.Tensor,
            tar_normed: torch.Tensor,
            weight: torch.Tensor,
            slowval: torch.Tensor,
            policy_dist: Agg,
            entropy: torch.Tensor,
            action: torch.Tensor,
            roffset: torch.Tensor,
            rscale: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        
        ret_normed = (ret - roffset) / rscale
        
        metrics = {
            'imag/adv':       adv.mean().detach(),
            'imag/adv_std':   adv.std().detach(),
            'imag/adv_mag':   adv.abs().mean().detach(),
            'imag/rew':       rew.mean().detach(),
            'imag/rew_std':   rew.std().detach(),
            'imag/con':       con.mean().detach(),
            'imag/ret':       ret_normed.mean().detach(),
            'imag/ret_std':   ret_normed.std().detach(),
            'imag/val':       val.mean().detach(),
            'imag/tar':       tar_normed.mean().detach(),
            'imag/weight':    weight.mean().detach(),
            'imag/slowval':   slowval.mean().detach(),
            'imag/ret_min':   ret_normed.min().detach(),
            'imag/ret_max':   ret_normed.max().detach(),
            'imag/ret_rate':  (ret_normed.abs() >= 1.0).float().mean().detach(),
            'imag/entropy':   entropy.mean().detach(),
            'imag/rscale':    rscale.detach(),
        }

        if hasattr(policy_dist, 'minent'):
            lo = policy_dist.minent
            hi = policy_dist.maxent
            metrics['imag/rand'] = ((entropy.mean() - lo) / (hi - lo)).detach()
        elif hasattr(policy_dist, 'logits'):
            logits = policy_dist.logits[:, :-1]
            probs = torch.softmax(logits, dim=-1)
            maxent = torch.log(torch.as_tensor(
                probs.shape[-1], device=probs.device, dtype=probs.dtype
            ))
            metrics.update({
                'imag/rand':            (entropy.mean() / maxent).detach(),
                'imag/policy_logit_std': logits.std(dim=-1).mean().detach(),
                'imag/policy_prob_std':  probs.std(dim=-1).mean().detach(),
                'imag/policy_max_prob':  probs.max(dim=-1).values.mean().detach(),
            })
            if action.ndim >= 3 and action.shape[-1] == probs.shape[-1]:
                action_freq = action.float().mean(dim=tuple(range(action.ndim - 1)))
                for i, freq in enumerate(action_freq):
                    metrics[f'imag/action_{i}_frac'] = freq.detach()
            
        return metrics
