from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import get_norm, NormedLinear, BlockLinear, NormedBlockLinear
from ..registry import register

''' Single-step GRU Cell with normalization '''

@register('sequence_model', 'gru')
class NormedGRUCell(nn.Module):
    '''
    單步 GRU：(h_prev, x) -> h_next
    輸出的 h_next 會經過 norm，讓 hidden state 的 scale 保持穩定
    '''

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 norm: str = 'rms',
                 **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cell = nn.GRUCell(input_dim, hidden_dim)
        
        self.norm = get_norm(norm, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        '''
        x: (B, input_dim)
        
        h: (B, hidden_dim)

        return h_next: (B, hidden_dim)
        '''

        dtype = x.dtype
        h_next = self.norm(self.cell(x, h)).to(dtype)
        
        return h_next
    
    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        '''回傳 (B, hidden_dim) 的全零初始 state'''
        return torch.zeros(batch_size, self.hidden_dim, device=device)

@register('sequence_model', 'block_gru')
class NormedBlockGRUCell(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 blocks: int = 8,
                 hidden_layers: int = 1,
                 norm: str = 'rms',
                 act: str = 'silu',
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.blocks = blocks

        assert hidden_dim % blocks == 0, \
            f'hidden_dim={hidden_dim} 必須能被 blocks={blocks} 整除'

        # 展開後：(B, hidden_dim + input_dim * blocks)
        block_in = hidden_dim + input_dim * blocks

        self.hidden = nn.Sequential(*[
            NormedBlockLinear(block_in if i == 0 else hidden_dim,
                              hidden_dim, blocks, norm, act)
            for i in range(hidden_layers)
        ])

        self.gate_proj = BlockLinear(hidden_dim, 3 * hidden_dim, blocks)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        '''
        x: (B, input_dim)  — 已經 embed 過的輸入
        h: (B, hidden_dim)  — 上一步的 deterministic state

        return: h_next (B, hidden_dim)
        '''
        g = self.blocks

        # --- Block expansion ---
        x = x.unsqueeze(-2).expand(-1, g, -1)          # (B, g, input_dim)
        h_grouped = h.unflatten(-1, (g, -1))           # (B, g, hidden_dim // g)
        x = torch.cat([h_grouped, x], dim=-1).flatten(-2)
        # -> (B, hidden_dim + input_dim * g)

        # --- Hidden layers -> gate projection ---
        x = self.hidden(x)         # (B, hidden_dim)
        x = self.gate_proj(x)      # (B, 3 * hidden_dim)

        # --- GRU gate computation ---
        x = x.unflatten(-1, (g, -1))
        reset, cand, update = [gate.flatten(-2) for gate in x.chunk(3, dim=-1)]

        reset  = torch.sigmoid(reset)
        cand   = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)

        h_next = update * cand + (1 - update) * h
        return h_next

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

''' Multi-layer GRU over a sequence '''

class GRUSequence(nn.Module):
    '''
    多步、多層 GRU: 整段序列進去，所有時步的 hidden state 出來
    '''

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 layers: int = 1,
                 norm: str = 'rms'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers

        dims = [input_dim] + [hidden_dim] * layers
        
        self.cells = nn.ModuleList([
            NormedGRUCell(dims[i], dims[i+1], norm)
            for i in range(layers)
        ])

    def forward(self,
                x: torch.Tensor,
                h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        x:  (B, T, input_dim)
        h0: (num_layers, B, hidden_dim) or None -> 自動補零

        return:
            h_seq:  (B, T, hidden_dim)  每個時步最後一層的 hidden state
            h_last: (num_layers, B, hidden_dim)  最後時步每層的 hidden state
        '''

        B, T, _ = x.shape

        if h0 is None:
            h0 = torch.zeros(self.layers, B, self.hidden_dim, device=x.device)
        
        # 把 h0 拆成 list，長度 = layers，每個元素 shape (B, hidden_dim)
        h = [element for element in h0]

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t, h[i])
                x_t = h[i]
            
            outputs.append(h[-1])

        # 把所有時步輸出 stack 成 (B, T, hidden_dim)
        h_seq = torch.stack(outputs, dim=1)
        
        # 把最後時步所有層的 hidden state stack 成 (num_layers, B, hidden_dim)
        h_last = torch.stack(h, dim=0)

        return h_seq, h_last

''' Convience '''

def get_initial_state(batch_size: int,
                      h_dim: int,
                      device: torch.device | str = 'cpu',
                      layers: int = 1) -> torch.Tensor:
    '''
    回傳全零 hidden state

    return: (layers, B, h_dim)
    '''
    
    return torch.zeros(layers, batch_size, h_dim, device=device)
