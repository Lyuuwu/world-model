from abc import ABC, abstractmethod
from typing import Protocol

'''
所有 networks/ 中需要被繼承的底層類別
'''

class Output(ABC):
    '''
    封裝 network output tensor 的純數學物件
    
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