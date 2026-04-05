from dataclasses import dataclass, field
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
    '''
    return of imagine()
    
    - feat: (B*K, H+1, feat_dim)
    - reward: (B*K, H+1)
    - cont: (B*K, H+1)
    - action: (B*K, H+1, action_dim)
    '''
    
    feat: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor
    action: torch.Tensor

@dataclass
class DreamerConfig:
    # --- optimizer ---
    lr: float = 4e-5
    agc: float = 0.3
    eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999
 
    # --- env ---
    discrete: bool = True
 
    # --- continue target ---
    contdisc: bool = True
    horizon: int = 333
 
    # --- imagination ---
    imag_horizon: int = 15
    imag_last: int | None = None
 
    # --- gradient flow ---
    ac_grads: bool = False
    reward_grad: bool = False
 
    # --- replay value loss ---
    repval_loss: bool = False
    repval_grad: bool = False
 
    # --- compile ---
    use_compile: bool = False
    compile_mode: str = 'reduce-overhead'
 
    # --- loss scales ---
    loss_scales: dict[str, float] = field(default_factory=lambda: {
        'dyn': 1.0, 'rep': 0.1, 'rec': 1.0,
        'rew': 1.0, 'con': 1.0,
        'policy': 1.0, 'value': 1.0, 'repval': 0.3,
    })
 
    # --- sub-module kwargs ---
    wm_kwargs: dict[str, Any] = field(default_factory=dict)
    ac_kwargs: dict[str, Any] = field(default_factory=dict)