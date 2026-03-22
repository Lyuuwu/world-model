import torch
from torch.optim.optimizer import Optimizer

class LaProp(Optimizer):
    ''' RMS normalize > momentum '''
    
    def __init__(
        self,
        params,
        lr: float=4e-5,
        betas: tuple[float, float]=(0.9, 0.99),
        eps: float=1e-20,
        agc: float=0.0,
        agc_eps: float=1e-3
        ):
        defaults = dict(lr=lr, betas=betas, eps=eps, agc=agc, agc_eps=agc_eps)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            agc = group['agc']
            agc_eps = group['agc_eps']
            
            # --- collect tensors ---
            params, grads, v_list, m_list, steps = [], [], [], [], []
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p)
                    state['m'] = torch.zeros_like(p)
                    
                state['step'] += 1
                
                params.append(p)
                grads.append(p.grad)
                v_list.append(state['v'])
                m_list.append(state['m'])
                steps.append(state['step'])
                
            if not params:
                continue
            
            # --- AGC ---
            if agc > 0:
                _foreach_agc(params, grads, agc, agc_eps)
                
            torch._foreach_mul_(v_list, beta2)
            torch._foreach_addcmul_(v_list, grads, grads, value=1.0 - beta2)
            
            bias_corr = [1.0 / (1.0 - beta2 ** s) for s in steps]
            v_hat = torch._foreach_mul(v_list, bias_corr)
            denom = torch._foreach_sqrt(v_hat)
            torch._foreach_add_(denom, eps)
            
            g_normed = torch._foreach_div(grads, denom)
            
            torch._foreach_mul_(m_list, beta1)
            torch._foreach_add_(m_list, g_normed, alpha=1.0 - beta1)
            
            torch._foreach_add_(params, m_list, alpha=-lr)
        
        return loss
        
            
                
def _foreach_agc(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    clip: float,
    eps: float
):
    if not params:
        return
    
    p_norms = torch._foreach_norm(params)
    g_norms = torch._foreach_norm(grads)
    scales = []
    
    for p_norm, g_norm in zip(p_norms, g_norms):
        max_norm = clip * p_norm.clamp(min=eps)
        
        scale = (max_norm / g_norm.clamp_min(1e-6)).clamp_max(1.0)
        scales.append(scale)
        
    torch._foreach_mul_(grads, scales)