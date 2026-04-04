from typing import TYPE_CHECKING

import math

import torch
import torch.nn as nn

from shared.obs_spec import ObsSpec
from shared.math_utils import symlog
from shared.networks.mlp import (
    MLP, NormedLinear, LinearHead, BlockLinear,
    get_act
)
from shared.networks.cnn import CNNUpsampleStack, SpatialNorm
from shared.distributions import CategoricalDist
from shared.losses import MSE, Agg

class ImageSpatialProjection(nn.Module):
    '''
    feat 的 deter/stoch 分別投影到 spatial feature map，再相加
    '''

    def __init__(
        self,
        h_dim: int,           # deter 維度
        stoch_flat: int,      # stoch * classes (已 flatten)
        units: int,           # MLP hidden units (用於 stoch branch)
        spatial_shape: tuple[int, int, int],  # (depths[-1], H_min, W_min)
        bspace: int = 8,      # BlockLinear blocks 數
        norm: str = 'rms',
        act: str = 'silu',
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.bspace = bspace
        
        spatial_flat = math.prod(spatial_shape)
        self.deter_proj = BlockLinear(h_dim, spatial_flat, blocks=bspace)
    
        self.stoch_proj = nn.Sequential(
            NormedLinear(stoch_flat, 2 * units, norm, act),
            LinearHead(2 * units, spatial_flat)
        )
        
        self.merge_norm = SpatialNorm(norm, spatial_shape[0])
        self.merge_act= get_act(act)

    def _rearrange(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        g = self.bspace
        C, H, W = self.spatial_shape
        c = C // g
        
        # x: (B, g, H * W * c)
        x = x.reshape(B, g, H, W, c)    # (B, g, H, W, c)
        x = x.permute(0, 2, 3, 1, 4)    # (B, H, W, g, c)
        x = x.reshape(B, H, W, C)       # (B, H, W, g * c)
        x = x.permute(0, 3, 1, 2)       # (B, C, H, W)
        
        return x

    def forward(
        self,
        deter: torch.Tensor,   # (B, h_dim)
        stoch: torch.Tensor,   # (B, stoch_flat)  已 flatten
    ) -> torch.Tensor:
        '''
        return: (B, C_spatial, H_min, W_min)
        '''
        B = deter.shape[0]
        C, H, W = self.spatial_shape
        
        x0 = self.deter_proj(deter)
        x0 = self._rearrange(x0)                    # (B, C, H, W)
        
        x1 = self.stoch_proj(stoch)                 # (B, 2*uints)
        x1 = x1.reshape(B, C, H, W)                 # (B, C, H, W)
        
        out = self.merge_act(self.merge_norm(x0 + x1))
        
        return out
    
    if TYPE_CHECKING:
        def __call__(
            self,
            deter: torch.Tensor,
            stoch: torch.Tensor) -> torch.Tensor:
            '''
            forward in:
                - deter: (B, h_dim)
                - stoch: (B, stoch_flat)

            forward out:
                - (B, C, H, W)
            '''
            ...

class VectorDecoderHead(nn.Module):
    '''
    feat -> shared MLP body -> per-key LinearHead -> Distribution
    '''

    def __init__(
        self,
        feat_dim: int,
        obs_space: dict[str, ObsSpec],
        units: int = 1024,
        layers: int = 3,
        norm: str = 'rms',
        act: str = 'silu',
        symlog: bool = True,
        outscale: float = 1.0,
    ):
        '''
        feat_dim = h_dim + stoch * classes

        obs_space 只包含 vector keys
        '''
        super().__init__()
        
        self._symlog = symlog
        self.mlp = MLP(feat_dim, units, layers, norm, act)
        
        self.heads = nn.ModuleDict()
        self.key_info = {}
        
        for key, spec in obs_space.items():
            if spec.discrete:
                out_dim = math.prod(spec.shape) * spec.classes
            else:
                out_dim = math.prod(spec.shape)
            
            self.heads[key] = LinearHead(units, out_dim, outscale)
            self.key_info[key] = {
                'discrete': spec.discrete,
                'shape': spec.shape,
                'classes': spec.classes
            }

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:    
        hidden = self.mlp(feat)    
        return {key: head(hidden) for key, head in self.heads.items()}
    
    if TYPE_CHECKING:
        def __call__(
            self,
            feat: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            '''
            forward in:
                - feat: (B, feat_dim)
            
            forward out:
                - {key: (B, units)}
            '''
            ...

class ImageDecoderHead(nn.Module):
    '''
    feat > ImageSpatialProjection > CNN upsample > sigmoid > MSE per pixel
    '''

    def __init__(
        self,
        h_dim: int,
        stoch_flat: int,
        obs_space: dict[str, ObsSpec],
        img_size: tuple[int, int] = (64, 64),
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        upsample: str = 'upsample',
        units: int = 1024,
        bspace: int = 8,
        norm: str = 'rms',
        act: str = 'silu',
        outscale: float = 1.0,
    ):
        '''
        obs_sapce 只包含 image keys
        '''
        super().__init__()
    
        n_stages = len(mults)
        H_min = img_size[0] // (2 ** n_stages)
        W_min = img_size[1] // (2 ** n_stages)
        depths = [depth * m for m in mults]
        C_spatial = depths[-1]
        out_channels = sum(spec.shape[0] for key, spec in obs_space.items())
        
        assert 3 <= H_min <= 16 and 3 <= W_min <= 16
        
        self.spatial_proj = ImageSpatialProjection(
            h_dim, stoch_flat, units,
            (C_spatial, H_min, W_min),
            bspace, norm, act
        )

        self.cnn_up = CNNUpsampleStack(
            depths=depths,
            out_channels=out_channels,
            kernel_size=kernel,
            upsample=upsample,
            norm=norm,
            act=act,
            out_act='none',
            outscale=outscale
        )
        
        self._img_channels = {key: spec.shape[0] for key, spec in obs_space.items()}

    def forward(
        self,
        deter: torch.Tensor,   # (B, h_dim)
        stoch: torch.Tensor,   # (B, stoch_flat)
    ) -> dict[str, torch.Tensor]:
        '''
        return: {key: ImageMSEDist}
        '''
        
        x = self.spatial_proj(deter, stoch)
        x = self.cnn_up(x)
        x = torch.sigmoid(x)
        
        ch_list = list(self._img_channels.values())
        splits = torch.split(x, ch_list, dim=1)
        
        return {key: img for (key, _), img in zip(self._img_channels.items(), splits)}
    
    if TYPE_CHECKING:
        def __call__(
            self,
            deter: torch.Tensor,
            sstoch: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            '''
            forward in:
                - deter: (B, h_dim)
                - stoch: (B, stoch_flat)

            forward out:
                - {key: (B, C, H, W)}
            '''
            ...

# ═══════════════════════════════════════════════════════════════
#  DreamerDecoder: 組裝 Image + Vector paths
# ═══════════════════════════════════════════════════════════════

class DreamerDecoder(nn.Module):
    '''
    Decoder of Dreamerv3
    '''

    def __init__(
        self,
        obs_space: dict[str, ObsSpec],

        # --- 維度 (從 RSSM 傳入) ---
        h_dim: int = 4096,
        stoch: int = 32,
        classes: int = 32,

        # --- Image decoder ---
        img_size: tuple[int, int] = (64, 64),
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        upsample: str = 'upsample',
        bspace: int = 8,

        # --- Vector decoder ---
        units: int = 1024,
        mlp_layers: int = 3,
        symlog_vecs: bool = True,

        # --- Shared ---
        norm: str = 'rms',
        act: str = 'silu',
        outscale: float = 1.0,

        **kwargs,
    ):
        super().__init__()

        # --- 分離 obs keys ---
        self.img_keys = sorted([k for k, s in obs_space.items() if s.is_image])
        self.vec_keys = sorted([k for k, s in obs_space.items() if not s.is_image])
        self.obs_space = obs_space

        # feat_dim = h_dim + stoch * classes
        # stoch_flat = stoch * classes
        feat_dim = h_dim + stoch * classes
        stoch_flat = stoch * classes
        self._h_dim = h_dim
        self._feat_dim = feat_dim

        if self.img_keys:
            img_space = {k: obs_space[k] for k in self.img_keys}
            self.img_decoder = ImageDecoderHead(h_dim, stoch_flat, img_space, img_size, depth, mults, kernel, upsample, units, bspace, norm, act, outscale)

        if self.vec_keys:
            vec_space = {k: obs_space[k] for k in self.vec_keys}
            self.vec_decoder = VectorDecoderHead(feat_dim, vec_space, units, mlp_layers, norm, act, symlog_vecs, outscale)

    def forward(
        self,
        feat: torch.Tensor | dict[str, torch.Tensor],
        bdims: int = 2,
    ) -> dict[str, Agg]:
        
        if isinstance(feat, dict):
            deter = feat['deter']
            stoch = feat['stoch'].flatten(-2)
            feat_flat = torch.cat([deter, stoch], dim=-1)
        else:
            feat_flat = feat
            deter = feat[..., :self._h_dim]
            stoch = feat[..., self._h_dim:]
            
        bshape = feat_flat.shape[:bdims]
        B_flat = math.prod(bshape)
        deter_2d = deter.reshape(B_flat, -1)
        stoch_2d = stoch.reshape(B_flat, -1)
        feat_2d  = feat_flat.reshape(B_flat, -1)
        
        recons = {}
        
        if self.img_keys:
            raw_imgs = self.img_decoder(deter_2d, stoch_2d)
            
            # raw_imgs: {key: (B_flat, C_k, H, W)}
            for key, img in raw_imgs.items():
                img = img.reshape(*bshape, *img.shape[1:])  # (B, T, C, H, W)
                recons[key] = Agg(MSE(img), agg_dims=3)     # (C, H, W)
        
        if self.vec_keys:
            raw_vecs = self.vec_decoder(feat_2d)
            
            # raw_vecs: {key: (B_flat, out_dim)}
            for key, raw in raw_vecs.items():
                info = self.vec_decoder.key_info[key]
                
                if info['discrete']:
                    raw = raw.reshape(*bshape, *info['shape'], info['classes'])
                    recons[key] = Agg(CategoricalDist(raw), agg_dims=len(info['shape']))
                else:
                    raw = raw.reshape(*bshape, *info['shape'])
                    squash = symlog if self.vec_decoder._symlog else None
                    recons[key] = Agg(MSE(raw, squash), agg_dims=len(info['shape']))
                                
        return recons

    if TYPE_CHECKING:
        def __call__(
            self,
            feat: torch.Tensor | dict[str, torch.Tensor],
            bdims: int = 2,
        ) -> dict[str, Agg]:
            '''
            forward in:
                - feat: (B, T, feat_dim) | {deter, stoch}
            
            forward out:
                - {key: Agg(Categorical | MSE), agg_dims = len(shape)}
                (discrete: Categorical | continuous: MSE)
            '''
            ...