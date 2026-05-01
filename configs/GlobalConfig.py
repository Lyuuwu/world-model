from dataclasses import dataclass, field, asdict
import copy

from shared.base import BaseConfig

@dataclass
class GlobalConfig(BaseConfig):
    ''' shared config '''
    
    # --- meta data ---
    agent: str = ''
    task: str = ''
    profile: str = ''
    
    # --- basic ---
    seed: int=0
    device: str='auto'
    total_env_steps: int=100_000
    seed_steps: int=1024
    batch_size: int=16
    seq_len: int=64
    train_ratio: int=256
    num_envs: int=1
    buffer_capacity: int=5_000_000
    min_episode_len: int=2
    
    # --- logging ---
    log_every: int=1000
    eval_every: int=1000
    eval_episodes: int=10
    use_checkpoint: bool=True
    checkpoint_every: int=25_000
    log_dir: str='runs'
    
    # --- compute ---
    compute_dtype: str='bfloat16'
    compile_mode: str='reduce-overhead'
    trainer: str='interleaved'
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def override(self, **kwargs) -> 'GlobalConfig':
        c = copy.deepcopy(self)
        for k, v in kwargs.items():
            if not hasattr(c, k):
                raise ValueError(f'Unkown config key: {k}')
            setattr(c, k, v)
        return c
