import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Output
from ..registry import register
from ..math_utils import symlog, symexp, twohot_symlog_encode

class Dist(Output):
    '''
    for every distribution
    '''

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        ''' log likelihood '''
        raise NotImplementedError(f'{type(self).__name__} does not implement log_prob')
    
    def loss(self, target: torch.Tensor) -> torch.Tensor:
        ''' pre-element loss (default -log_prob(target.detach())) '''
        return -self.log_prob(target.detach())
    
    def sample(self) -> torch.Tensor:
        raise NotImplementedError(f'{type(self).__name__} does not support sampling')
    
    def entropy(self) -> torch.Tensor:
        raise NotImplementedError(f'{type(self).__name__} does not implement entropy')
    
    def kl(self, other: 'Dist') -> torch.Tensor:
        raise NotImplementedError(f'{type(self).__name__} does not implement kl')

@register('dist', 'categorical')
@register('dist', 'cat')
class CategoricalDist(Dist):
    '''
    一般整數 index 的 categorical distribution
    
    回傳 interger index
    '''
    
    def __init__(self, logits: torch.Tensor, unimix: float=0.0):
        self._logits = logits
        self._num_classes = logits.shape[-1]
        
        if unimix > 0:
            raw_probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(raw_probs) / self._num_classes
            probs = (1 - unimix) * raw_probs + unimix * uniform
            self._logits = torch.log(probs)
        
    @property
    def mode(self) -> torch.Tensor:
        return torch.argmax(self._logits, dim=-1)
    
    def sample(self) -> torch.Tensor:
        return torch.distributions.Categorical(logits=self._logits).sample()
    
    def log_prob(self, event: torch.Tensor | int) -> torch.Tensor:
        onehot = F.one_hot(event, self._num_classes)
        return (torch.log_softmax(self._logits, dim=-1) * onehot).sum(-1)
    
    def entropy(self):
        logprob = torch.log_softmax(self._logits, dim=-1)
        prob = torch.softmax(self._logits, -1)
        return -(prob * logprob).sum(-1)
    
    def kl(self, other: 'CategoricalDist'):
        logprob = torch.log_softmax(self._logits, -1)
        logother = torch.log_softmax(other._logits, -1)
        prob = torch.softmax(self._logits, -1)
        
        return (prob * (logprob - logother)).sum(-1)

@register('dist', 'straight_through_categorical')
@register('dist', 'stc')
class StraightThroughCategorical(Dist):
    '''
    離散 latent state z 的 categorical distribution

    - forward pass: sample one-hot（不可微）
    - backward pass: straight-through，梯度直接穿過 one-hot 回到 softmax probs
    - unimix: 混合 (1-u)*softmax(logits) + u*uniform，防止概率歸零
    '''

    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.01):
        '''
        logits: (..., num_classes)  未經 softmax 的 raw logits
        unimix_ratio: uniform mixture 比例，default 0.01 (DreamerV3)
        '''
        self.dist = CategoricalDist(logits, unimix_ratio)
    
    @property
    def logits(self) -> torch.Tensor:
        return self.dist._logits

    @property
    def mode(self) -> torch.Tensor:
        onehot = F.one_hot(self.dist.mode, self.dist._num_classes).detach()
        return onehot.float()

    def sample(self) -> torch.Tensor:
        '''
        回傳 one-hot sample with straight-through gradient
        '''
        
        # 抽樣
        idx = self.dist.sample()
        
        # 中獎的地方設成1 (one-hot)
        one_hot = F.one_hot(idx, num_classes=self.dist._num_classes).float()
        
        # 讓梯度能夠從 probs 流過去
        probs = torch.softmax(self.dist._logits, dim=-1)
        res = probs + (one_hot - probs).detach()
        
        return res
        

    def log_prob(self, event: torch.Tensor) -> torch.Tensor:
        '''
        計算 one-hot event 的 log probability

        event: (..., num_classes) one-hot encoded

        return: (...) scalar log prob per sample
        '''
        log_probs = torch.log_softmax(self.dist._logits, dim=-1)
        return (event * log_probs).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def kl(self, other: 'StraightThroughCategorical'):
        return self.dist.kl(other.dist)

''' TwoHotCategorical '''

def build_symexp_bins(num_bins: int = 255,
                      lower: float = -20.0,
                      upper: float = 20.0) -> torch.Tensor:
    '''
    建立 symexp-spaced bins

    return: (num_bins,) 1-D tensor
    '''
    linear_bins = torch.linspace(lower, upper, num_bins)
    return symexp(linear_bins)

@register('dist', 'twohot')
class TwoHotCategorical(Dist):
    '''
    用於 reward / event 預測的離散化分布
    '''

    def __init__(self, logits: torch.Tensor, bins: torch.Tensor | None = None):
        '''
        logits: (..., num_bins)  raw logits \\
        bins:   (num_bins,) bin 位置, None 時自動用 build_symexp_bins()
        '''
        self._logits = logits

        if bins is None:
            bins = build_symexp_bins()

        if bins.device != logits.device:
            bins = bins.to(device=logits.device)
        
        self._bins = bins
        self._probs = F.softmax(logits, dim=-1)

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        return self._probs

    @property
    def bins(self) -> torch.Tensor:
        return self._bins

    @property
    def mean(self) -> torch.Tensor:
        # bins are in real space, convert to symlog for averaging
        bins64 = symlog(self._bins).double()
        probs64 = self._probs.double()
        
        # symmetric summation in symlog space
        n = bins64.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1, p2, p3 = probs64[..., :m], probs64[..., m:m+1], probs64[..., m+1:]
            b1, b2, b3 = bins64[..., :m], bins64[..., m:m+1], bins64[..., m+1:]
            wavg = (p2 * b2).sum(-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(-1)
        else:
            p1, p2 = probs64[..., :n//2], probs64[..., n//2:]
            b1, b2 = bins64[..., :n//2], bins64[..., n//2:]
            wavg = ((p1 * b1).flip(-1) + (p2 * b2)).sum(-1)
        
        return symexp(wavg.float())

    @property
    def mode(self) -> torch.Tensor:
        '''
        回傳概率最高的 bin 對應的值
        '''
        idx = torch.argmax(self._probs, dim=-1)
        return self._bins[idx]

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        '''
        計算 twohot cross-entropy log probability

        target: (...) 連續值

        return: (...) log prob (負的 cross-entropy，越大越好)
        '''
        
        target = symlog(target)
        below = (self.bins <= target[..., None]).int().sum(-1) - 1
        above = len(self.bins) - (
            self.bins > target[..., None]).int().sum(-1)
        below = below.clamp(0, len(self.bins) - 1)
        above = above.clamp(0, len(self.bins) - 1)
        equal = (below == above)
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - target))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None] +
            F.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        
        return (target * log_pred).sum(-1)

@register('dist', 'normal')
class NormalDist(Dist):
    def __init__(self, mean: torch.Tensor, stddev: torch.Tensor):
        self._mean = mean
        self._stddev = stddev.broadcast_to(mean.shape)
        
        self.minent = None
        self.maxent = None
    
    @property
    def mode(self) -> torch.Tensor:
        return self._mean
    
    def sample(self) -> torch.Tensor:
        return self._mean + torch.randn_like(self._mean) * self._stddev
    
    def log_prob(self, event: torch.Tensor) -> torch.Tensor:
        return (
            -0.5 * math.log(2 * math.pi)
            - torch.log(self._stddev)
            - 0.5 * ((event - self._mean) / self._stddev).square()
        )
        
    def entropy(self):
        return 0.5 * torch.log(2 * math.pi * self._stddev.square()) + 0.5
    
    def kl(self, other: 'NormalDist') -> torch.Tensor:
        return 0.5 * (
            (self._stddev / other._stddev).square()
            + ((other._mean - self._mean) / other._stddev).square()
            + 2 * (torch.log(other._stddev) - torch.log(self._stddev))
            - 1.0
        )

@register('dist', 'binary')
class BinaryDist(Dist):
    
    def __init__(self, logit: torch.Tensor):
        self._logit = logit
        
    @property
    def mode(self) -> torch.Tensor:
        return (self._logit > 0).float()
    
    def log_prob(self, event: float | torch.Tensor):
        event = torch.as_tensor(event, dtype=self._logit.dtype, device=self._logit.device)
        return(
            event * F.logsigmoid(self._logit)
            + (1.0 - event) * F.logsigmoid(-self._logit)
        )
    
    def prob(self, event: float | torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(event))
    
    def sample(self):
        prob = torch.sigmoid(self._logit)
        return torch.bernoulli(prob)
    
    def entropy(self):
        p = torch.sigmoid(self._logit)
        return -(
            p * F.logsigmoid(self._logit)
            + (1.0 - p) * F.logsigmoid(-self._logit)
        )