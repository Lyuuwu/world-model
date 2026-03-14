from __future__ import annotations

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
from shared.networks.distributions import CategoricalDist
from shared.networks.losses import MSE, Agg


# ═══════════════════════════════════════════════════════════════
#  Image Spatial Projection (BlockLinear variant)
# ═══════════════════════════════════════════════════════════════

class ImageSpatialProjection(nn.Module):
    """
    feat 的 deter/stoch 分別投影到 spatial feature map，再相加
    
    forward: (B, C, H, W)
    """

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
        """
        return: (B, C_spatial, H_min, W_min)  可以直接丟進 CNN upsample
        """
        B = deter.shape[0]
        C, H, W = self.spatial_shape
        
        x0 = self.deter_proj(deter)
        x0 = self._rearrange(x0)                    # (B, C, H, W)
        
        x1 = self.stoch_proj(stoch)
        x1 = x1.reshape(B, C, H, W)                 # (B, C, H, W)
        
        out = self.merge_act(self.merge_norm(x0 + x1))
        
        return out


# ═══════════════════════════════════════════════════════════════
#  Vector Decoder Head
# ═══════════════════════════════════════════════════════════════

class VectorDecoderHead(nn.Module):
    """
    feat → shared MLP body → per-key LinearHead → Distribution
    
    每個 obs key 根據其 ObsSpec 選擇不同的 output distribution:
    
      discrete key  → LinearHead(units → prod(shape) * classes)
                    → reshape (B, *shape, classes) → Categorical
                    
      continuous key (symlog=True)  → LinearHead(units → prod(shape))
                                    → Symlog
                                    
      continuous key (symlog=False) → LinearHead(units → prod(shape))
                                    → MSE
    """

    def __init__(
        self,
        feat_dim: int,          # input feature dim = h_dim + stoch*classes
        obs_space: dict[str, ObsSpec],   # 只包含 vector keys
        units: int = 1024,
        layers: int = 3,
        norm: str = 'rms',
        act: str = 'silu',
        symlog: bool = True,
        outscale: float = 1.0,
    ):
        super().__init__()
        
        self._symlog = symlog
        self.mlp = MLP(feat_dim, units, layers, norm, act)
        
        self.heads = nn.ModuleDict()
        self._key_info = {}
        
        for key, spec in obs_space.items():
            if spec.discrete:
                out_dim = math.prod(spec.shape) * spec.classes
            else:
                out_dim = math.prod(spec.shape)
            
            self.heads[key] = LinearHead(units, out_dim, outscale)
            self._key_info[key] = {
                'discrete': spec.discrete,
                'shape': spec.shape,
                'classes': spec.classes
            }

    def forward(
        self,
        feat: torch.Tensor,    # (B, feat_dim) 或 (B, T, feat_dim)
    ) -> dict[str, Agg]:
        """
        return: {key: Distribution} 
        
        每個 Distribution 物件支援:
          - .loss(target)  → (B,) 或 (B,T)  scalar loss per sample
          - .mode          → 重建值 (eval 時用)
        """
        
        leading = feat.shape[:-1]
        hidden = self.mlp(feat)
        
        recons = {}
        for key, head in self.heads.items():
            info = self._key_info[key]
            raw = head(hidden)
            
            if info['discrete']:
                raw = raw.reshape(*leading, *info['shape'], info['classes'])
                inner = CategoricalDist(raw)
                recons[key] = Agg(inner, agg_dims=len(info['shape']))
            else:
                raw = raw.reshape(*leading, *info['shape'])
                squash = symlog if self._symlog else None
                inner = MSE(raw, squash)
                recons[key] = Agg(inner, agg_dims=len(info['shape']))
            
        return recons


# ═══════════════════════════════════════════════════════════════
#  Image Decoder Head  
# ═══════════════════════════════════════════════════════════════

class ImageDecoderHead(nn.Module):
    """
    feat → ImageSpatialProjection → CNN upsample → sigmoid → MSE per pixel
    
    CNN upsampling 結構（與 Encoder 鏡像對稱）:
    
      (B, depths[-1], H_min, W_min)
        → ConvTransposeBlock(depths[-1] → depths[-2])  + upsample ×2
        → ConvTransposeBlock(depths[-2] → depths[-3])  + upsample ×2
        → ...
        → ConvTransposeBlock(depths[1] → depths[0])    + upsample ×2
        → final upsample ×2 + Conv2d(depths[0] → out_channels) + sigmoid
    
    Output: (B, C_out, H, W) ∈ [0, 1]
    
    ⚠️ 注意原作在 loss 計算時：
       - target image 要從 uint8 [0,255] 轉成 float [0,1]
       - loss = MSE(pred, target)，然後 SUM over (C, H, W)（不是 mean！）
       - 這個 sum-over-spatial 的行為透過 Agg(MSE, dims=3) 實現
       - 我們在 Distribution wrapper 裡處理
       
    ⚠️ 如果有多個 image keys (e.g., rgb + depth)，channel concat 後一起 decode
       最後 split 回各自的 channel 數
    """

    def __init__(
        self,
        h_dim: int,
        stoch_flat: int,
        obs_space: dict[str, ObsSpec],   # 只包含 image keys
        img_size: tuple[int, int] = (64, 64),
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        upsample: str = 'upsample',     # 'upsample' (nearest+conv) | 'stride' (transposed conv)
        units: int = 1024,               # stoch projection hidden dim
        bspace: int = 8,
        norm: str = 'rms',
        act: str = 'silu',
        outscale: float = 1.0,
    ):
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
    ) -> dict[str, Agg]:
        """
        return: {key: ImageMSEDist}
        
        ImageMSEDist 包裝:
          - .pred()        → (B, C, H, W) ∈ [0, 1]
          - .loss(target)  → (B,) 其中 target 已是 float [0,1]
                             loss = sum over (C, H, W) of (pred - target)²
        
        ⚠️ target 的 uint8→float 轉換在 world_model.py 做，不在 decoder 裡
        """
        
        x = self.spatial_proj(deter, stoch)
        x = self.cnn_up(x)
        x = torch.sigmoid(x)
        
        ch_list = list(self._img_channels.values())
        splits = torch.split(x, ch_list, dim=1)
        
        recons = {}
        for (key, _), img in zip(self._img_channels.items(), splits):
            recons[key] = Agg(MSE(img), agg_dims=3)
        
        return recons

# ═══════════════════════════════════════════════════════════════
#  DreamerDecoder: 組裝 Image + Vector paths
# ═══════════════════════════════════════════════════════════════

class DreamerDecoder(nn.Module):
    """
    DreamerV3 的完整 Decoder
    
    組裝邏輯:
      1. 從 RSSM feat dict 取出 deter 和 stoch
      2. 如果有 image keys → ImageDecoderHead
      3. 如果有 vector keys → VectorDecoderHead
      4. 合併 recons dict 回傳
    
    ════════════════════════════════════════════
    呼叫流程 (world_model.py 中)
    ════════════════════════════════════════════
    
      # 1. RSSM observe → feat dict
      state, outputs = rssm.observe(tokens, actions, resets)
      feat = rssm.get_feat(state)          # (B, T, feat_dim)
      
      # 2. Decoder → per-key distributions
      recons = decoder(feat)               # dict[str, Distribution]
      
      # 3. Prediction loss
      for key, dist in recons.items():
          target = preprocess(obs[key])     # image: uint8→float/255, vec: as-is
          losses[key] = dist.loss(target)   # (B, T) per-sample loss
    
    ════════════════════════════════════════════
    config 驅動：與其他 agent 的可替換性
    ════════════════════════════════════════════
    
    未來 R2I / EDELINE 可以：
      - 直接複用 VectorDecoderHead（它們的 vector 重建邏輯一樣）
      - 替換 ImageDecoderHead（DIAMOND 用 diffusion decoder）
      - 或整個替換 DreamerDecoder（IRIS 根本沒有 pixel-level decoder）
    
    因此這個 class 應該註冊到 Registry 方便 config 切換
    """

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

        # --- Registry 相容 ---
        **kwargs,
    ):
        super().__init__()

        # ── 分離 obs keys ──
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
            recons.update(self.img_decoder(deter_2d, stoch_2d))
        
        if self.vec_keys:
            recons.update(self.vec_decoder(feat_2d))
            
        return recons

# ═══════════════════════════════════════════════════════════════
#  Registry registration
# ═══════════════════════════════════════════════════════════════

from shared.registry import register
register('decoder', 'dreamerv3')(DreamerDecoder)

#
# YAML config 範例:
#
# decoder:
#   type: dreamerv3
#   h_dim: 4096
#   stoch: 32
#   classes: 32
#   img_size: [64, 64]
#   depth: 64
#   mults: [2, 3, 4, 4]
#   kernel: 5
#   upsample: upsample
#   bspace: 8
#   units: 1024
#   mlp_layers: 3
#   symlog_vecs: true
#   norm: rms
#   act: silu
#   outscale: 1.0