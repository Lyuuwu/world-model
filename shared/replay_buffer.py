from typing import Optional

import numpy as np
import torch

class Episode:
    def __init__(self):
        self._steps: list[dict[str, np.ndarray]] = []
        self._finalized: Optional[dict[str, np.ndarray]] = False
        
    def add(self, obs: dict, action: np.ndarray, reward: float, is_first: bool, is_last: bool, is_terminal: bool):
        assert not self._finalized, 'Cannot add to finalized episode'
        
        step = {}
        for key, val in obs.items():
            step[key] = np.asarray(val)
        
        step['action'] = np.asarray(action, dtype=np.float32)
        step['reward'] = np.float32(reward)
        step['is_first'] = np.bool_(is_first)
        step['is_last'] = np.bool_(is_last)
        step['is_terminal'] = np.bool_(is_terminal)
        self._steps.append(step)
        
    def finalize(self) -> dict[str, np.ndarray]:
        assert len(self._steps) > 0, 'Cannot finalize empty episode'
        assert not self._finalized, 'Already finalized'
        
        keys = self._steps[0].keys()
        data = {}
        for key in keys:
            data[key] = np.stack([step[key] for step in self._steps], axis=0)
        
        self._steps = []        # free memory
        self._finalized = True
        
        return data
    
    def __len__(self):
        if self._finalized:
            raise RuntimeError('Use len on finalized data dict, not Episode')
        return len(self._steps)
    
class EpisodeReplayBuffer:
    def __init__(
        self,
        capacity: int=5000000,
        min_episode_len: int=2,
        device: str='cuda'
    ):
        self._capacity = capacity
        self._min_episode_len = min_episode_len
        self._device = device
        
        self._episodes: list[dict[str, np.ndarray]] = []
        self._episode_lenghts: list[int] = []
        self._total_steps: int = 0
        
        self._ongoing: Optional[Episode] = None
        
    def add_step(
        self,
        obs: dict,
        action: np.ndarray,
        reward: float,
        is_terminal: bool,
        is_first: bool,
        is_last: bool
    ):
        if is_first:
            if self._ongoing is not None and len(self._ongoing) > 0:
                self._commit_ongoing()
            self._ongoing = Episode()
        
        assert self._ongoing is not None, 'Must call add_step with is_first=True first'
        self._ongoing.add(obs, action, reward, is_first=is_first, is_last=is_last, is_terminal=is_terminal)
        
        if is_terminal:
            self._commit_ongoing()
            
    def add_episode(self, episode_data: dict[str, np.ndarray]):
        ep_len = next(iter(episode_data.values())).shape[0]
        if ep_len < self._min_episode_len:
            return
        self._episodes.append(episode_data)
        self._episode_lenghts.append(ep_len)
        self._total_steps += ep_len
        self._evict()
                
    def _commit_ongoing(self):
        if self._ongoing is None:
            return
        
        data = self._ongoing.finalize()
        self._ongoing = None
        ep_len = data['reward'].shape[0]
        if ep_len < self._min_episode_len:
            return
        self._episodes.append(data)
        self._episode_lenghts.append(ep_len)
        self._total_steps += ep_len
        self._evict()
        
    def sample(
        self,
        batch_size: int,
        seq_len: int
    ) -> dict[str, torch.Tensor]:
        assert len(self._episodes) > 0, 'Replay buffer is empty'
        
        lenghts = np.array(self._episode_lenghts)
        weights = np.maximum(lenghts - seq_len + 1, 0).astype(np.float64)
        total_weight = weights.sum()
        assert total_weight > 0, (
            f'No episode long enough for seq_len={seq_len}. '
            f'Max episode lenght: {lenghts.max()}'
        )
        probs = weights / total_weight
        
        ep_indices = np.random.choice(len(self._episodes), size=batch_size, p=probs)
        
        sequences = []
        for ep_idx in ep_indices:
            ep = self._episodes[ep_idx]
            ep_len = self._episode_lenghts[ep_idx]
            max_start = ep_len - seq_len
            start = np.random.randint(0, max_start + 1)
            seq = {key: val[start : start + seq_len] for key, val in ep.items()}
            sequences.append(seq)
            
        batch = {}
        keys = sequences[0].keys()
        for key in keys:
            stacked = np.stack([seq[key] for seq in sequences], axis=0)
            
            if stacked.dtype == np.bool_:
                tensor = torch.from_numpy(stacked).to(device=self._device)
            elif stacked.dtype == np.uint8:
                tensor = torch.from_numpy(stacked).to(device=self._device)
            else:
                tensor = torch.from_numpy(stacked).to(
                    dtype=torch.float32, device=self._device
                )
            
            batch[key] = tensor
            
        return batch
    
    def _evict(self):
        while self._total_steps > self._capacity and len(self._episodes) > 1:
            removed = self._episodes.pop(0)
            removed_len = self._episode_lenghts.pop(0)
            self._total_steps -= removed_len
    
    @property
    def total_steps(self) -> int:
        return self._total_steps
    
    @property
    def num_episodes(self) -> int:
        return len(self._episodes)     
    
    @property
    def stats(self) -> dict:
        if len(self._episode_lenghts) == 0:
            return {'total_steps': 0, 'num_episodes': 0}
        lengths = np.array(self._episode_lenghts)
        return{
            'total_steps': self._total_steps,
            'num_episodes': len(self._episodes),
            'mean_episode_len': float(lengths.mean()),
            'min_episode_len': int(lengths.min()),
            'max_episode_len': int(lengths.max()),
            'capacity_usage': self._total_steps / self._capacity
        }
        
    def __len__(self) -> int:
        return self._total_steps
    
    def __repr__(self) -> str:
        return (
            f'EpisodeReplayBuffer('
            f'episodes={self.num_episodes}, '
            f'steps={self.total_steps}/{self._capacity}'
        )