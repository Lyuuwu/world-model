from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from ..math_utils import symlog, symexp, twohot_symlog_encode

@runtime_checkable
class Distribution(Protocol):
    '''
    所有 distribution 都要有的
    '''
    
    @property
    def mode(slef) -> torch.Tensor: ...

    def log_prob(self, target: torch.Tensor) -> torch.Tensor: ...

class Categorical:
    '''
    一般的分布，回傳 interger index
    '''
    
    def __init__(self, logits: torch.Tensor, unimix_ratio: float=0.0):
        self._logits = logits
        self._num_classes = logits.shape[-1]
        
        if unimix_ratio > 0:
            raw_probs = F.softmax(logits, dim=-1)
            uniform = 1.0 / self._num_classes
            probs = (1 - unimix_ratio) * raw_probs + unimix_ratio * uniform
            self._logits = torch.log(probs)
        
        self._probs = F.softmax(self._logits, dim=-1)
        
    @property
    def mode(self) -> torch.Tensor:
        return torch.argmax(self._logits, dim=-1)
    
    def sample(self) -> torch.Tensor:
        return torch.distributions.Categorical(logits=self._logits).sample()
    
    def log_prob(self, event: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(event.long(), self._num_classes)
        return (torch.log_softmax(self._logits, dim=-1) * onehot).sum(-1)
    
    def loss(self, target: torch.Tensor):
        return -self.log_prob(target.detach())
    
    def entropy(self, eps: float=1e-8):
        logprob = torch.log(self._probs + eps)
        return -(self._probs * logprob).sum(-1)
    
    def kl(self, other: 'Categorical'):
        logprob = torch.log_softmax(self._logits, -1)
        logother = torch.log_softmax(other._logits, -1)
        prob = torch.softmax(self._logits, -1)
        
        return (prob * (logprob - logother)).sum(-1)

''' StraightThroughCategorical '''

class StraightThroughCategorical:
    '''
    RSSM 離散 latent state z 的 categorical distribution

    - forward pass: sample one-hot（不可微）
    - backward pass: straight-through，梯度直接穿過 one-hot 回到 softmax probs
    - unimix: 混合 (1-u)*softmax(logits) + u*uniform，防止概率歸零
    '''

    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.01):
        '''
        logits: (..., num_classes)  未經 softmax 的 raw logits
        unimix_ratio: uniform mixture 比例，default 0.01 (DreamerV3)
        '''
        self.dist = Categorical(logits, unimix_ratio)
    
    @property
    def logits(self) -> torch.Tensor:
        return self.dist._logits

    @property
    def probs(self) -> torch.Tensor:
        return self.dist._probs

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
        
        # 讓梯度能夠從 self._probs 流過去
        res = self.dist._probs + (one_hot - self.dist._probs).detach()
        
        return res
        

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        '''
        計算 one-hot value 的 log probability

        value: (..., num_classes) one-hot encoded

        return: (...) scalar log prob per sample
        '''
        log_probs = torch.log_softmax(self.dist._logits, dim=-1)
        return (value * log_probs).sum(dim=-1)

    def entropy(self, eps: float=1e-8) -> torch.Tensor:
        return self.dist.entropy(eps=eps)

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

class TwoHotCategorical:
    '''
    用於 reward / value 預測的離散化分布
    '''

    def __init__(self, logits: torch.Tensor, bins: torch.Tensor | None = None):
        '''
        logits: (..., num_bins)  raw logits \\
        bins:   (num_bins,) bin 位置，None 時自動用 build_symexp_bins()
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
        '''
        加權平均 = sum(probs * bins)

        return: (...) 連續預測值
        '''
    
        probs64 = self._probs.double()
        bins64 = self._bins.double()
        weighted = probs64 * bins64
        pos = torch.where(bins64 >= 0, weighted, torch.zeros_like(weighted)).sum(-1)
        neg = torch.where(bins64 < 0, weighted, torch.zeros_like(weighted)).sum(-1)
        return (pos + neg).float()

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
        
        target_encoded = twohot_symlog_encode(target, self.bins)
        log_probs = F.log_softmax(self._logits, dim=-1)
        return (target_encoded * log_probs).sum(dim=-1)


''' SymlogGaussian '''

class SymlogGaussian:
    '''
    symlog 空間的 Gaussian (var=1) \n
    等價 symlog MSE loss
    '''

    def __init__(self, mean: torch.Tensor):
        self._mean = mean

    @property
    def mode(self) -> torch.Tensor:
        return symexp(self._mean)

    @property
    def mean_symlog(self) -> torch.Tensor:
        return self._mean

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        target_symlog = symlog(target)
        return -0.5 * ((self._mean - target_symlog) ** 2).sum(dim=-1)

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(target)
    
class MSEGaussian:
    '''
    原始空間的 Gaussian (var=1)
    '''
    
    def __init__(self, mean: torch.Tensor):
        self._mean = mean
    
    @property
    def mode(self) -> torch.Tensor:
        return self._mean
    
    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((self._mean - target) ** 2).sum(dim=-1)
    
    def loss(self, target: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(target)

class MSEDist:
    """
    Per-element MSE (不做 sum)
    對應 JAX outs.MSE — loss 回傳與 input 同 shape 的 tensor
    
    可選 squash: 用於 symlog_mse 模式
    """
    def __init__(self, mean: torch.Tensor, squash=None):
        self._mean = mean
        self._squash = squash or (lambda x: x)
    
    @property
    def mode(self) -> torch.Tensor:
        return self._mean
    
    def loss(self, target: torch.Tensor) -> torch.Tensor:
        """return: same shape as mean — per-element squared error"""
        return (self._mean - self._squash(target.detach())) ** 2


class AggDist:
    """
    Wraps any inner distribution, aggregates loss over trailing dims
    
    對應 JAX outs.Agg(inner, dims, agg_fn)
    
    用法:
      image:     AggDist(MSEDist(pred),            dims=3)  # sum over C,H,W
      symlog:    AggDist(MSEDist(pred, symlog),     dims=1)  # sum over features
      mse:       AggDist(MSEDist(pred),             dims=1)
      categorical: AggDist(CategoricalDist(logits), dims=len(shape))
    """
    def __init__(self, inner, agg_dims: int, agg_fn=torch.sum):
        self._inner = inner
        self._agg_dims = agg_dims
        self._agg_fn = agg_fn
    
    @property
    def mode(self) -> torch.Tensor:
        return self._inner.mode
    
    def loss(self, target: torch.Tensor) -> torch.Tensor:
        per_elem = self._inner.loss(target)
        dims = list(range(-self._agg_dims, 0))
        return self._agg_fn(per_elem, dim=dims)

class BernoulliDist:
    '''
    continue flag 的預測 \n
    接收 raw logits
    '''
    
    def __init__(self, logits: torch.Tensor):
        self._logits = logits
    
    @property
    def mode(self) -> torch.Tensor:
        return (self._logits > 0).float()
    
    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        return -F.binary_cross_entropy_with_logits(
            self._logits, target, reduction='none'
        ).sum(dim=-1)

    def loss(self) -> torch.Tensor:
        return torch.sigmoid(self._logits)

''' Convenience '''

def get_dist(name: str, logits_or_mean: torch.Tensor, **kwargs) -> Distribution:
    '''
    統一介面，方便 config 驅動

    name:
      'categorical'                  | 'cat'     → Catgorical
      'straight_through_categorical' | 'stc'     → StraightThroughCategorical
      'twohot_categorical'           | 'twohot'  → TwoHotCategorical
      'symlog_gaussian'              | 'symlog'  → SymlogGaussian
      'mse_gaussian'                 | 'mse'     → MSEGaussian
      'bernoulli'                                → BernoulliDist
    '''
    _map = {
        'catgorical': Categorical,
        'cat': Categorical,
        'straight_through_categorical': StraightThroughCategorical,
        'stc': StraightThroughCategorical,
        'twohot_categorical': TwoHotCategorical,
        'twohot': TwoHotCategorical,
        'symlog_gaussian': SymlogGaussian,
        'symlog': SymlogGaussian,
        'mse_gaussian': MSEGaussian,
        'mse': MSEGaussian,
        'bernoulli': BernoulliDist,
    }
    if name not in _map:
        raise ValueError(f'Unknown distribution: {name!r}. Available: {list(_map)}')
    return _map[name](logits_or_mean, **kwargs)

if __name__ == '__main__':
    B, C = 4, 32
    NUM_BINS = 255

    def test(name, fn):
        try:
            out = fn()
            if isinstance(out, torch.Tensor):
                print(f'PASS  {name}  -> {out.shape}  range=[{out.min():.4f}, {out.max():.4f}]')
            else:
                print(f'PASS  {name}  -> {out}')
        except Exception as e:
            print(f'FAIL  {name}: {e}')

    # --- StraightThroughCategorical ---
    logits = torch.randn(B, 32, C)  # 32 latents × 32 classes
    dist_stc = StraightThroughCategorical(logits, unimix_ratio=0.01)
    test('STC sample',   lambda: dist_stc.sample())
    test('STC mode',     lambda: dist_stc.mode)
    test('STC log_prob', lambda: dist_stc.log_prob(dist_stc.sample()))
    test('STC entropy',  lambda: dist_stc.entropy())

    # gradient 測試
    logits_g = torch.randn(B, C, requires_grad=True)
    dist_g = StraightThroughCategorical(logits_g)
    z = dist_g.sample()
    loss = z.sum()
    loss.backward()
    test('STC grad flows', lambda: logits_g.grad is not None and logits_g.grad.abs().sum() > 0)

    # --- TwoHotCategorical ---
    bins = build_symexp_bins(NUM_BINS)
    logits_th = torch.randn(B, NUM_BINS)
    dist_th = TwoHotCategorical(logits_th, bins)
    test('TwoHot mean',     lambda: dist_th.mean)
    test('TwoHot mode',     lambda: dist_th.mode)
    target = torch.tensor([0.0, 1.5, -100.0, 500.0])
    test('TwoHot log_prob', lambda: dist_th.log_prob(target))

    # --- SymlogGaussian ---
    mean = torch.randn(B, 64)
    dist_sg = SymlogGaussian(mean)
    test('SymlogGauss mode',     lambda: dist_sg.mode)
    test('SymlogGauss log_prob', lambda: dist_sg.log_prob(torch.randn(B, 64)))
    test('SymlogGauss loss',     lambda: dist_sg.loss(torch.randn(B, 64)))

    # --- get_dist factory ---
    test('get_dist stc',    lambda: get_dist('stc', torch.randn(B, C)).sample())
    test('get_dist twohot', lambda: get_dist('twohot', torch.randn(B, NUM_BINS), bins=bins).mean)
    test('get_dist symlog', lambda: get_dist('symlog', torch.randn(B, 64)).mode)