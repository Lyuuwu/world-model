import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.obs_spec import ObsSpec
from shared.math_utils import symlog
from shared.networks.cnn import CNNEncoder, compute_cnn_out_dim
from shared.networks.mlp import MLP

class DreamerEncoder(nn.Module):
    '''
    - image keys  -> CNNEncoder -> flatten
    - vector keys -> symlog (optional) -> MLP
    
    forward in:
        - images:  (B, T, C, H, W)
        - vectors: (B, T, *obs_shape)
    
    forward out:
        - (B, T, token_dim) | token_dim = cnn_flat + mlp_units
    '''

    def __init__(
        self,
        obs_space: dict[str, ObsSpec],

        # --- CNN path ---
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        img_size: tuple[int, int] = (64, 64),   # assumed same for all image keys
        kernel: int = 5,
        downsample: str = 'maxpool',            # 'maxpool' | 'stride'

        # --- MLP path ---
        units: int = 1024,
        mlp_layers: int = 3,
        symlog_vecs: bool = True,               # apply symlog to continuous vector obs

        # --- shared ---
        norm: str = 'rms',
        act: str = 'silu',
    ):
        super().__init__()

        self.obs_space = obs_space
        self.symlog_vecs = symlog_vecs
        self.img_size = img_size

        self.img_keys: list[str] = sorted([k for k, s in obs_space.items() if s.is_image])
        self.vec_keys: list[str] = sorted([k for k, s in obs_space.items() if not s.is_image])

        cnn_flat_dim = 0
        if self.img_keys:
            channels = sum(obs_space[key].shape[0] for key in self.img_keys)
            self.cnn = CNNEncoder(channels, depth, mults, kernel, downsample, norm, act)
            cnn_flat_dim = compute_cnn_out_dim(img_size, depth, mults)

        vec_out_dim = 0
        if self.vec_keys:
            vec_in = self._cal_vec_in()
            self.vec_mlp = MLP(vec_in, units, mlp_layers, norm, act)
            vec_out_dim = units

        self._token_dim = cnn_flat_dim + vec_out_dim

        assert self._token_dim > 0, \
            'obs_space must contain at least one image or vector key'

    @property
    def token_dim(self) -> int:
        return self._token_dim
    
    def _cal_vec_in(self):
        total = 0
        
        for key in self.vec_keys:
            space = self.obs_space[key]
            if space.discrete:
                total += math.prod(space.shape) * space.classes
            else:
                total += math.prod(space.shape)
                
        return total

    def _encode_images(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Stack image keys along channel dim -> CNNEncoder -> (prod(bshape), cnn_flat_dim)
        '''
        imgs = [obs[k] for k in self.img_keys]
        
        # img: (B, C, H, W)
        x = torch.cat(imgs, dim=1).float() / 255.0 - 0.5
        x = self.cnn(x)
        
        return x
        

    def _encode_vectors(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Concat vector keys -> optional symlog -> MLP -> (prod(bshape), units)
        '''
        
        vecs = self._vec_cat(obs)
        param_dtype = next(self.vec_mlp.parameters()).dtype
        x = self.vec_mlp(vecs.to(param_dtype))
        return x
        
    
    def _vec_cat(self, obs: dict[str, torch.Tensor]):
        vspace = [self.obs_space[k] for k in self.vec_keys]
        vecs = [obs[k] for k in self.vec_keys]
        
        res = []
        for space, vec in zip(vspace, vecs):
            if space.discrete:
                x = F.one_hot(vec, space.classes).float()
            else:
                x = symlog(vec.float()) if self.symlog_vecs else vec.float()
            
            x = x.flatten(1)
            res.append(x)
        
        return torch.cat(res, dim=-1)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        bdims: int = 2,                   # leading batch dims: 2=(B,T,...), 1=(B,...)
    ) -> torch.Tensor:
        '''
        obs: dict of (B, T, *shape) or (B, *shape) tensors
        
        return: tokens (B, T, token_dim) or (B, token_dim)
        '''
        first_key = next(iter(obs))
        bshape = obs[first_key].shape[:bdims]
        
        if bdims == 2:
            obs_flat = {k: v.flatten(0, 1) for k, v in obs.items()}
        else:
            obs_flat = obs
        
        outs = []
        
        if self.vec_keys:
            outs.append(self._encode_vectors(obs_flat))
        
        if self.img_keys:
            outs.append(self._encode_images(obs_flat))
        
        tokens = torch.cat(outs, dim=-1)
        tokens = tokens.reshape(*bshape, -1)
        
        return tokens

    if TYPE_CHECKING:
        def __call__(
            self,
            obs: dict[str, torch.Tensor],
            bdims: int=2
        ):
            '''
            forward in:
                - images:  (B, T, C, H, W)
                - vectors: (B, T, *obs_shape)
            
            forward out:
                - (B, T, token_dim) | token_dim = cnn_flat + mlp_units
            '''
            ...