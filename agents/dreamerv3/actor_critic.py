import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from shared.math_utils import Normalizer
from shared.networks.mlp import MLP, LinearHead
from shared.networks.distributions import (
    StraightThroughCategorical, TwoHotCategorical, NormalDist, build_symexp_bins
)
from shared.networks.losses import Agg
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
        outscale: float=1.0,
        unimix: float=0.01,
        minstd: float=1.0,
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
        self._bins = build_symexp_bins(bins)
        
    def forward(self, feat: torch.Tensor):
        x = self.head(self.mlp(feat))
        bins = self._bins.to(x.device)
        return TwoHotCategorical(x, bins)
    
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
    
def lambda_return(
    rew: torch.Tensor,
    val: torch.Tensor,
    cont: torch.Tensor,
    disc: float,
    lam: float=0.95
) -> torch.Tensor:
    bootstrap = val[:, -1]
    live = cont[:, 1:] * disc
    lam_w = lam
    interm = rew[:, 1:] + (1 - lam_w) * live * val[:, 1:]
    
    rets = [bootstrap]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * lam_w * rets[-1])
        
    rets = list(reversed(rets))[:-1]
    return torch.stack(rets, dim=1)

def lambda_return_replay(
    last: torch.Tensor,
    term: torch.Tensor,
    rew: torch.Tensor,
    val: torch.Tensor,
    boot: torch.Tensor,
    disc: float,
    lam: float=0.95
) -> torch.Tensor:
    raise NotImplementedError

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
        policy_unimix: float=0.01,
        actent: float=3e-4,
        
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
        
        # --- loss scales ---
        policy_scale: float=1.0,
        value_scale: float=1.0,
        
        **kwargs
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
            unimix=policy_unimix
        )
        
        self.value_head = ValueHead(
            feat_dim, units, layers, bins, norm, act,
            outscale=value_outscale
        )
        
        self.slow_values = SlowValueTarget(self.value_head, slow_decay)
        
        # --- normalizers ---
        
        self.retnorm = Normalizer(
            decay=retnorm_decay, use_percentile=True,
            percentile_low=retnorm_plow, percentile_high=retnorm_phigh,
            max_limit=retnorm_limit
        )
        
        self.valnorm = Normalizer(decay=valnorm_decay, use_percentile=False)
        self.advnorm = Normalizer(decay=advnorm_decay, use_percentile=False)
        
    