from dataclasses import dataclass
from typing import Any

import torch

@dataclass
class WorldModelOutputs:
    ''' return of observe() '''
    
    state: dict[str, torch.Tensor]          # final RSSM state {deter, stoch}
    feat: torch.Tensor                      # (B, T, feat_dim) = concat(deter, stoch_flat)
    rssm_outputs: dict[str, torch.Tensor]   # RSSM observe outputs (deter, stoch, logit)
    
@dataclass
class ImaginedTrajectory:
    ''' return of imagine() '''
    
    feat: torch.Tensor      # (B*K, H+1, feat_dim)
    reward: torch.Tensor    # (B*K, H+1)            predicted reward (symexp)
    cont: torch.Tensor      # (B*K, H+1)            predicted continuation prob
    action: torch.Tensor    # (B*K, H+1, action_dim)

@dataclass
class DreamerV3Config:
    # --- env ---
    action_dim: int = 18
    discrete: bool = True
    
    # --- training ---
    batch_size: int = 16
    batch_length: int = 64
    imag_horizon: int = 15
    imag_last: int | None = None
    
    # --- optimizer ---
    lr: float = 4e-5
    agc: float = 0.3
    eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.99
    max_grad_norm: float | None = None
    
    # --- loss scales ---
    loss_scales: dict[str, float] | None = None
    
    # --- gradient flow flags ---
    ac_grads: bool = False
    reward_grad: bool = False
    
    # --- continue target ---
    contdisc: bool = True
    horizon: int = 333
    
    # --- replay value loss ---
    repval_loss: bool = True
    repval_grad: bool = False
    
    # --- world model kwargs ---
    wm_kwargs: dict[str, Any] | None = None
    
    # --- actor-critic kwargs ---
    ac_kwargs: dict[str, Any] | None = None
