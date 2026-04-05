from abc import ABC, abstractmethod
from typing import Protocol
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

class Output(ABC):
    '''
    封裝 output tensor 的純數學物件
    
    提供 預測 (mode) 與 per-element loss 計算
    
    不參與 optimizer
    '''
    
    @property
    @abstractmethod
    def mode(self):
        return NotImplementedError(f'{type(self).__name__} does not implement mode')
    
    @abstractmethod
    def loss(self, target):
        return NotImplementedError(f'{type(self).__name__} does not implement loss')

class BufferBase(ABC):
    @abstractmethod
    def add_step(
        self,
        obs: dict,
        action: np.ndarray,
        reward: float,
        is_terminal: bool,
        is_first: bool,
        is_last: bool):
        ...
        
    @abstractmethod
    def sample(
        self,
        batch_size: int,
        seq_len: int
    ) -> dict[str, torch.Tensor]:
        ...
        
    @property
    @abstractmethod
    def total_steps(self) -> int:
        ...

class AgentBase(ABC, nn.Module):
    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor: ...
    
    @abstractmethod
    def initial_prevact(self, batch_size: int, device: torch.device) -> torch.Tensor: ...
    
    @abstractmethod
    def policy(self, obs, state, prev_action, is_first, train=True) -> tuple[torch.Tensor, dict]: ...
    
    @abstractmethod
    def train_step(self, data, device_type='cuda', compute_dtype=torch.bfloat16) -> dict[str, float]: ...
    
class BaseConfig:
    def to_dict(self) -> dict:
        return asdict(self)
