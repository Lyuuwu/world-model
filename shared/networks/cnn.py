from __future__ import annotations

import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import get_norm, get_act, trunc_normal_init_

class SpatialNorm(nn.Module):
    
    def __init__(self, norm: str, channels: int):
        super().__init__()
        self.norm = get_norm(norm, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (B, C, H, W) -> (B, H, W, C) -> norm -> (B, C, H, W)
        '''
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvBlock(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=5,
                 downsample: str='maxpool',
                 norm: str='rms',
                 act: str='silu'):
        super().__init__()
        self.downsample = downsample
        
        stride = 2 if downsample == 'stride' else 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = SpatialNorm(norm, out_channels)
        self.act = get_act(act)
        
        trunc_normal_init_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
        if downsample == 'maxpool':
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.pool is not None:
            x = self.pool(x)
        
        x = self.act(self.norm(x))
        
        return x
    
class ConvTransposeBlock(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=5,
                 upsample: str='upsample',
                 norm: str='rms',
                 act: str='silu'):
        super().__init__()
        self.upsample_mode = upsample
        padding = kernel_size // 2
        
        if upsample == 'stride':
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size,
                stride=2, padding=padding, output_padding=1
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=1, padding=padding
            )
            
        self.norm = SpatialNorm(norm, out_channels)
        self.act = get_act(act)
        
        trunc_normal_init_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def forward(self, x: torch.Tensor):
        if self.upsample_mode == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x

class CNNUpsampleStack(nn.Module):
    """
    純 CNN upsample: (B, C_in, H_min, W_min) -> (B, out_channels, H, W)
    """
    
    def __init__(
        self,
        depths: Sequence[int],       # 原始順序 [128, 192, 256, 256]
        out_channels: int,
        kernel_size: int = 5,
        upsample: str = 'upsample',
        norm: str = 'rms',
        act: str = 'silu',
        out_act: str = 'sigmoid',
        outscale: float = 1.0,
    ):
        # rev_depths = list(reversed(depths))  # [256, 256, 192, 128]
        # self.blocks = nn.ModuleList([
        #     ConvTransposeBlock(rev_depths[i], rev_depths[i+1], ...)
        #     for i in range(len(rev_depths) - 1)
        # ])
        # self.out_conv = ...  (rev_depths[-1] -> out_channels, outscale init)
        # self.upsample_mode = upsample
        # self._out_act = out_act
        
        super().__init__()
        
        padding = kernel_size // 2
        rev_depths = list(reversed(depths))
        
        self.blocks = nn.ModuleList([
            ConvTransposeBlock(rev_depths[i], rev_depths[i+1], kernel_size, upsample, norm, act)
            for i in range(len(rev_depths) - 1)
        ])
        
        if upsample == 'stride':
            self.out_conv = nn.ConvTranspose2d(
                rev_depths[-1], out_channels, kernel_size,
                stride=2, padding=padding, output_padding=1
            )
        else:
            self.out_conv = nn.Conv2d(
                rev_depths[-1], out_channels, kernel_size,
                stride=1, padding=padding
            )
            
        trunc_normal_init_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
            
        if outscale != 1.0:
            with torch.no_grad():
                self.out_conv.weight.mul_(outscale)
        
        self.upsample_mode = upsample
        
        self._out_act = out_act
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, depths[-1], H_min, W_min)  ← 已是 spatial tensor
        return: (B, out_channels, H, W)
        """
        
        for block in self.blocks:
            x = block(x)
        
        if self.upsample_mode == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.out_conv(x)
        
        if self._out_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self._out_act == 'tanh':
            x = torch.tanh(x)
        elif self._out_act == 'none':
            pass
        else:
            raise ValueError(f'Unknown out_act: {self._out_act!r}')
        
        return x

class CNNEncoder(nn.Module):
    
    def __init__(self,
                 in_channels: int=3,
                 depth: int=64,
                 mults: Sequence[int]=(2, 3, 4, 4),
                 kernel_size: int=5,
                 downsample: str='maxpool',
                 norm: str='rms',
                 act: str='silu',
                 flatten_output: bool=True,
                 return_intermediates: bool=False):
        super().__init__()
        self.depths = [depth * m for m in mults]
        self.flatten_output = flatten_output
        self.return_intermediates = return_intermediates
        
        channels = [in_channels] + self.depths
        self.blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i+1], kernel_size, downsample, norm, act)
            for i in range(len(self.depths))
        ])
        
    @property
    def out_dim(self) -> int | None:
        '''
        最後一層的 channel 的數量
        '''
        return self.depths[-1]
    
    def compute_out_dim(self, img_size: tuple[int, int]) -> tuple[int, int ,int]:
        '''
        根據 img_size 回傳 (depths[-1], h, w) \\
        '''
        h, w = img_size
        for _ in self.blocks:
            h, w = h // 2, w // 2
        return (self.depths[-1], h, w)
    
    def forward(self, x: torch.Tensor):
        '''
        x: (B, C, H, W), float, [0, 1] or [-0.5, 0.5]
        
        flatten=True:  return tensor (B, D) \\
        flatten=False: return tensor (B, C', H', W') \\
        intermidiates=True: return tuple (上述, [skip1, skip2, ...])
        '''
        intermediates = []
        
        for block in self.blocks:
            x = block(x)
            
            if self.return_intermediates:
                intermediates.append(x)
        
        if x.shape[2] < 2 or x.shape[3] < 2:
            warnings.warn(f'Spatial size after encoding is very small: {x.shape[2:]}')
        
        if self.flatten_output:
            x = x.flatten(start_dim=1)
            
        if self.return_intermediates:
            return x, intermediates
        
        return x
    
''' CNN Decoder '''

class CNNDecoder(nn.Module):
    
    def __init__(self,
                 in_dim: int,
                 out_channels: int=3,
                 img_size: tuple[int, int]=(64, 64),
                 depth: int=64,
                 mults: Sequence[int]=(2, 3, 4, 4),
                 kernel_size: int=5,
                 upsample: str='upsample',
                 norm: str='rms',
                 act: str='silu',
                 out_act: str='sigmoid',
                 outscale: float=1.0):
        super().__init__()
        self.img_size = img_size
        self._out_act = out_act
        depths = [depth * m for m in mults]
        
        n_down = len(mults)
        self.min_h = img_size[0] // (2 ** n_down)
        self.min_w = img_size[1] // (2 ** n_down)
        assert 3 <= self.min_h <= 16 and 3 <= self.min_w <= 16, \
            f'Bottleneck spatial size {self.min_h}*{self.min_w} out of expected range'
            
        self.spatial_channels = depths[-1]
        spatial_dim = self.spatial_channels * self.min_h * self.min_w
        
        self.fc = nn.Linear(in_dim, spatial_dim)
        self.fc_norm = SpatialNorm(norm, self.spatial_channels)
        self.fc_act = get_act(act)
        
        self.cnn_up = CNNUpsampleStack(
            depths, out_channels, kernel_size,
            upsample, norm, act, out_act, outscale
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (B, in_dim)
        
        return (B, C, H, W) reconstructed image in [0, 1]
        '''
        
        x = self.fc(x)
        x = x.reshape(-1, self.spatial_channels, self.min_h, self.min_w)
        x = self.fc_act(self.fc_norm(x))
        
        return self.cnn_up(x)
    
''' Convenience '''

def compute_cnn_out_dim(img_size: tuple[int, int],
                        depth: int=64,
                        mults: Sequence[int]=(2, 3, 4, 4))->int:
    h, w = img_size
    for _ in mults:
        h, w = h // 2, w // 2
    return depth * mults[-1] * h * w

if __name__ == '__main__':
    # --- config ---
    B = 2
    C = 3
    H, W = 64, 64
    DEPTH = 64
    MULTS = (2, 3, 4, 4)

    # --- encoder ---
    encoder = CNNEncoder(in_channels=C, depth=DEPTH, mults=MULTS)
    x = torch.rand(B, C, H, W)
    z = encoder(x)

    expected_dim = compute_cnn_out_dim((H, W), DEPTH, MULTS)
    assert z.shape == (B, expected_dim), f"Encoder output shape wrong: {z.shape}"
    print(f"✅ Encoder OK: {x.shape} -> {z.shape}")

    # --- decoder ---
    decoder = CNNDecoder(in_dim=expected_dim, out_channels=C, img_size=(H, W), depth=DEPTH, mults=MULTS)
    recon = decoder(z)

    assert recon.shape == (B, C, H, W), f"Decoder output shape wrong: {recon.shape}"
    assert recon.min() >= 0.0 and recon.max() <= 1.0, "Decoder output out of [0,1]"
    print(f"✅ Decoder OK: {z.shape} -> {recon.shape}")

    # --- encode -> decode roundtrip ---
    print(f"✅ Roundtrip OK, recon range: [{recon.min():.3f}, {recon.max():.3f}]")