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
