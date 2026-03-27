import math

import torch
import torch.nn as nn

''' Normalization '''

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-4):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype)
    
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-4):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype)
    
def get_norm(norm: str, dim: int, **kwargs) -> nn.Module:
    '''
    norm: rms | layer | none
    '''
    
    if norm == 'rms':
        return RMSNorm(dim)
    elif norm == 'layer':
        return LayerNorm(dim)
    elif norm == 'group':
        num_groups = kwargs.get('num_groups', min(8, dim))
        return nn.GroupNorm(num_groups, dim)
    elif norm == 'none':
        return nn.Identity()
    
    raise ValueError(f'Unkown norm: {norm!r}')

''' Activation '''

def get_act(name: str) -> nn.Module:
    '''
    name: silu | relu | gelu | mish | elu | tanh | none
    '''
    
    activations = {
        'silu': nn.SiLU,
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'mish': nn.Mish,
        'elu':  nn.ELU,
        'tanh': nn.Tanh,
        'none': nn.Identity
    }
    
    if name not in activations:
        raise ValueError(f'Unkown activation: {name!r}')
    
    return activations[name]()

''' Initialization '''

def trunc_normal_init_(tensor: torch.Tensor, fan: str='in', scale: float=1.0):
    '''
    fan: in | out | avg
    '''
    
    if tensor.ndim < 2:
        return nn.init.zeros_(tensor)
    
    if scale == 0.0:
        return nn.init.zeros_(tensor)
    
    if fan == 'in':
        fan_val = tensor.shape[1] * (math.prod(tensor.shape[2:]) if tensor.ndim > 2 else 1)
    elif fan == 'out':
        fan_val = tensor.shape[0] * (math.prod(tensor.shape[2:]) if tensor.ndim > 2 else 1)
    elif fan == 'avg':
        fan_in = tensor.shape[1] * (math.prod(tensor.shape[2:]) if tensor.ndim > 2 else 1)
        fan_out = tensor.shape[0] * (math.prod(tensor.shape[2:]) if tensor.ndim > 2 else 1)
        fan_val = (fan_in + fan_out) / 2
    else:
        raise ValueError(f'Unknown fan mode: {fan!r}')
    
    std = 1.1368 * math.sqrt(1.0 / fan_val) * scale
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2*std, b=2*std)
    return tensor

def init_linear_(linear: nn.Linear, outscale: float=1.0, fan: str='in'):
    '''
    fan: in | out | avg
    '''
    
    trunc_normal_init_(linear.weight, fan=fan, scale=outscale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
        
''' MLP '''

class NormedLinear(nn.Module):
    '''
    Single Linear > Norm > Act block
    '''
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 norm: str='rms',
                 act: str='silu',
                 outscale: float=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = get_norm(norm, out_dim)
        self.act = get_act(act)
        init_linear_(self.linear, outscale=outscale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.linear(x)))
    
class MLP(nn.Module):
    '''
    multi layer for: linear > norm > act
    '''
    
    def __init__(self,
                 in_dim: int,
                 units: int=1024,
                 layers: int=3,
                 norm: str='rms',
                 act: str='silu'):
        super().__init__()
        self.out_dim = units
        
        dims = [in_dim] + [units] * layers
        self.nets = nn.Sequential(*[
            NormedLinear(dims[i], dims[i+1], norm=norm, act=act)
            for i in range(layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nets(x)
    
''' Linear projection head '''

class LinearHead(nn.Module):
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 outscale: float=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        init_linear_(self.linear, outscale=outscale)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
''' MLP + Head '''

class MLPHead(nn.Module):
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 units: int=1024,
                 layers: int=3,
                 norm: str='rms',
                 act: str='silu',
                 outscale: float=1.0):
        '''
        norm: rms | layer | none
        
        act: silu | relu | gelu | mish | elu | tanh | none
        '''
        super().__init__()
        self.mlp = MLP(in_dim, units, layers, norm, act)
        self.head = LinearHead(units, out_dim, outscale)
        
    @property
    def out_dim(self) -> int:
        return self.head.linear.out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(x))
    
''' block diagonal linear '''

class BlockLinear(nn.Module):
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 blocks: int=8,
                 bias: bool=True):
        super().__init__()
        assert in_dim % blocks == 0 and out_dim % blocks == 0, \
            f'in_dim={in_dim}, out_dim={out_dim} 都要被 blocks={blocks} 整除'
        
        self.blocks = blocks
        self.in_dim = in_dim
        self.out_dim = out_dim
        bsize_in = in_dim // blocks
        bsize_out = out_dim // blocks
        
        # shape: (blocks, in_per_block, out_per_block)
        self.weight = nn.Parameter(torch.empty(blocks, bsize_in, bsize_out))

        for i in range(blocks):
            trunc_normal_init_(self.weight.data[i].T)
        
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (..., in_dim) -> (..., out_dim)
        '''
        
        leading = x.shape[:-1]
        
        # reshape to (..., blocks, in_per_block)
        x = x.reshape(*leading, self.blocks, self.in_dim // self.blocks)
        
        x = torch.einsum('...bi,bio->...bo', x, self.weight)
        
        x = x.reshape(*leading, self.out_dim)
        
        if self.bias is not None:
            x = x + self.bias
        
        return x

class NormedBlockLinear(nn.Module):
    '''
    Block -> Norm -> Act
    '''
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 blocks: int,
                 norm: str='rms',
                 act: str='silu',
                 bias: bool=True):
        super().__init__()
        
        self.block = BlockLinear(in_dim, out_dim, blocks, bias)
        self.norm = get_norm(norm, out_dim)
        self.act = get_act(act)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.block(x)))

