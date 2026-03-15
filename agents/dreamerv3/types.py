from dataclasses import dataclass
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