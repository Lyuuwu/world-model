import torch
import torch.nn as nn

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))

def rms(x: torch.Tensor) -> torch.Tensor:
    return (x.square().mean(dim=-1, keepdim=True)).sqrt().to(x.dtype)

class ReturnNorm(nn.Module):
    
    def __init__(self,
                 decay: float=0.99,
                 percentile_low: float=5.0,
                 percentile_high: float=95.0,
                 limit: float=1.0,
                 eps: float=1e-8):
        super().__init__()
        self.decay = decay
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.limit = limit
        self.eps = eps
        
        self.register_buffer('low', torch.zeros(1))
        self.register_buffer('high', torch.ones(1))
        self.register_buffer('initialized', torch.tensor(False))
        
    @torch.no_grad()
    def update(self, returns: torch.Tensor):
        ''' EMA 更新 percentile '''
        low  = torch.quantile(returns.detach().float(), self.percentile_low/100)
        high = torch.quantile(returns.detach().float(), self.percentile_high / 100)
        
        if not self.initialized:
            self.low.copy_(low)
            self.high.copy_(high)
            self.initialized.copy_(torch.tensor(True))
        else:
            self.low.mul_(self.decay).add_(low * (1 - self.decay))
            self.high.mul_(self.decay).add_(high * (1 - self.decay))
            
    
    
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        ''' update + normalize '''
        self.update(returns)
        return self.normalize(returns)
    
class Normalizer(nn.Module):
    
    def __init__(
        self,
        decay: float=0.99,
        limit: float=1e-8,
        use_percentile: bool=False,
        percentile_low: float=5.0,
        percentile_high: float=95.0,
        max_limit: float=1.0,
        enable: bool=True
    ):
        super().__init__()
        
        self.decay = decay
        self.limit = limit
        self.use_percentile = use_percentile
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.max_limit = max_limit
        self.enable = enable
        
        self.register_buffer('_offset', torch.zeros(1))
        self.register_buffer('_scale', torch.ones(1))
        self.register_buffer('_initialized', torch.tensor(False))
    
    @torch.no_grad()
    def update(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        update EMA
        
        return (offset, scale)
        '''
        
        if not self.enable:
            return (0, 1)
        
        if self.use_percentile:
            low = torch.quantile(data.float(), self.percentile_low / 100)
            high = torch.quantile(data.float(), self.percentile_high / 100)
            new_offset = torch.zeros(1, device=data.device)
            new_scale = torch.max(high - low, torch.tensor(self.max_limit))
        else:
            new_offset = data.float().mean()
            new_scale = data.float().std().clamp(min=self.limit)
        
        if not self._initialized:
            self._offset.copy_(new_offset)
            self._scale.copy_(new_scale)
            self._initialized.fill_(True)
        else:
            self._offset.mul_(self.decay).add_(new_offset * (1 - self.decay))
            self._scale.mul_(self.decay).add_(new_scale * (1 - self.decay))
            
        return (self._offset.clone(), self._scale.clone())
    
    def stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self._offset.clone(), self._scale.clone())
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self._offset) / self._scale
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        return data * self._scale + self._offset
    
    def forward(self, data: torch.Tensor, update: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
        if update:
            return self.update(data)
        return self.stats()