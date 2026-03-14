from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

@runtime_checkable
class SequenceModelCell(Protocol):
    '''
    單步 recurrent model
    '''
    hidden_dim: int
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        '''
        x: (B, input_dim) \n
        h: (B, hidden_dim) \n
        return: h_next(B, hidden_dim)
        '''
        ...
        
    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ''' return (B, hidden_dim) 的初始 hidden state '''
        ...
        
@runtime_checkable
class SequenceModelSeq(Protocol):
    '''
    多步 sequence model
    '''
    
    hidden_dim: int
    
    def forward(self,
                x: torch.Tensor,
                h0: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        x: (B, T, input_dim) \n
        h0: (B, hidden_dim) or None \n
        return: (h_seq: (B, T, hidden_dim), h_last: (B, hidden_dim))
        '''
        ...
        
''' wrapper '''

class Cell2SeqWrapper(nn.Module):
    '''
    把 SequenceModelCell 包成 SequenceModelSeq
    '''
    
    def __init__(self, cell: SequenceModelCell):
        super().__init__()
        
        if isinstance(cell, nn.Module):
            self.cell = cell
        else:
            self._cell = cell
            
        self.hidden_dim = cell.hidden_dim
        
    @property
    def _cell_module(self) -> SequenceModelCell:
        if hasattr(self, 'cell'):
            return self.cell
        return self._cell

    def forward(self, x, h0=None):
        B, T, _ = x.shape
        cell = self._cell_module

        if h0 is None:
            h0 = cell.initial_state(B, x.device)

        h = h0
        outputs = []
        for t in range(T):
            h = cell(x[:, t], h)
            outputs.append(h)

        return torch.stack(outputs, dim=1), h