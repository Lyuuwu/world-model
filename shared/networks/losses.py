from ..registry import register
from .base import Output
from .distributions import Dist

import torch

@register('loss', 'mse')
class MSE(Output):
    def __init__(self, mean: torch.Tensor, squash=None):
        super().__init__()
        
        self._mean = mean
        self._squash = squash or (lambda x: x)
    
    @property
    def mode(self) -> torch.Tensor:
        return self._mean
    
    def loss(self, target: torch.Tensor):
        assert self._mean.shape == target.shape, (self._mean.shape, target.shape)
        return torch.square(self._mean - self._squash(target).detach())

@register('loss', 'huber')
class Huber(Output):
    def __init__(self, mean: torch.Tensor, eps: float=1.0):
        super().__init__()
        
        self._mean = mean
        self.eps = eps
    
    @property
    def mode(self):
        return self._mean
    
    def loss(self, target: torch.Tensor):
        assert self._mean.shape == target.shape, (self._mean.shape, target.shape)
        
        dist = self._mean - target.detach()
        return torch.sqrt(torch.square(dist) + torch.square(self.eps)) - self.eps
    
class Agg(Output):
    
    def __init__(self, inner, agg_dims: int, agg_fn=torch.sum):
        super().__init__()
        
        assert isinstance(inner, Output), (
            f'Agg expects an Output subclass, got {type(inner).__name__}'
        )
        assert agg_dims >= 1, f'agg_dims must be >= 1, got {agg_dims}'
        
        self._inner = inner
        self._dims = list(range(-agg_dims, 0))
        self._agg_fn = agg_fn
        
    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError
        try:
            return getattr(self._inner, name)
        except:
            raise AttributeError(
                f'"{type(self).__name__}" wrapping "{type(self._inner).__name__}"'
                f'has no attribute {name}'
            )
    
    @property
    def mode(self) -> torch.Tensor:
        return self._inner.mode
    
    def loss(self, target: torch.Tensor) -> torch.Tensor:
        per_elem = self._inner.loss(target)
        return self._agg_fn(per_elem, dim=self._dims)
    
    ''' for Dist '''
    def _requre_dist(self, method_name: str) -> None:
        ''' Error Message if inner is not a Dist '''
        
        if not isinstance(self._inner, Dist):
            raise TypeError(
                f'Agg.{method_name}() requires a Dist inner, '
                f'got {type(self._inner).__name__}'
            )
            
    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        self._requre_dist('log_prob')
        return self._agg_fn(self._inner.log_prob(target), dim=self._dims)
    
    def entropy(self) -> torch.Tensor:
        self._requre_dist('entropy')
        return self._agg_fn(self._inner.entropy(), dim=self._dims)
    
    def kl(self, other: 'Agg') -> torch.Tensor:
        self._requre_dist('kl')
        assert isinstance(other, Agg), (
            f'Agg.kl() expects another Agg, got {type(other).__name__}'
        )
        assert isinstance(other._inner, Dist), (
            f'Agg.kl() requires other.inner to be a Dist, '
            f'got {type(other._inner).__name__}'
        )
        return self._agg_fn(self._inner.kl(other._inner), dim=self._dims)
        
    def sample(self) -> torch.Tensor:
        self._requre_dist('sample')
        return self._inner.sample()
    
    ''' Debug '''
    def __repr__(self) -> str:
        return (
            f"Agg({type(self._inner).__name__}, "
            f"mode={self.mode.shape}, "
            f"agg_dims={self._dims})"
        )