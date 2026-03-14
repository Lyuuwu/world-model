import torch
import torch.nn as nn

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))

def rms(x: torch.Tensor) -> torch.Tensor:
    return (x.square().mean(dim=-1, keepdim=True)).sqrt().to(x.dtype)

def twohot_symlog_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    '''
    將連續 scalar 編碼為 twohot vector

    x:    (...) 任意 shape 的連續值 \n
    bins: (B,) 1-D sorted bin 位置

    return: (..., B) twohot encoded vector
    '''

    # 轉換到 symlog 的空間
    symlog_x = symlog(x)
    symlog_bins = symlog(bins)
    
    # 找 k 的位置 (binL 的位置)
    k = torch.searchsorted(symlog_bins, symlog_x) - 1
    k = k.clamp(0, len(bins) - 2)
    
    # 計算 weight
    w_upper = (symlog_x - symlog_bins[k]) / (symlog_bins[k+1] - symlog_bins[k])
    w_upper = w_upper.clamp(0, 1)
    w_lower = 1 - w_upper
    
    # 建立 result = zeros(..., B)
    res = torch.zeros(*x.shape, len(bins), device=x.device, dtype=x.dtype)
    
    # scatter 填入 w_lower 到 index k, w_upper 到 index k+1
    res.scatter_(-1, k.unsqueeze(-1), w_lower.unsqueeze(-1))
    res.scatter_(-1, (k+1).unsqueeze(-1), w_upper.unsqueeze(-1))
    
    return res

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
            
    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        scale = (self.high - self.low).clamp(min=self.eps)
        return returns / torch.max(scale, torch.tensor(self.limit))
    
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        ''' update + normalize '''
        self.update(returns)
        return self.normalize(returns)