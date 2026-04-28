from dataclasses import dataclass, field
from shared.base import BaseConfig

@dataclass
class AtariEnvConfig(BaseConfig):
    action_repeat: int = 4
    max_noop: int = 0
    sticky_prob: float = 0.0
    fire_reset: bool = False
    life_loss_terminal: bool = False
    reward_clip: str | None = None
    reward_scale: float = 1.0
    grayscale: bool = True
    resize: list[int] = field(default_factory=lambda: [64, 64])
    frame_stack: int = 1
    max_episode_steps: int = 108_000
    obs_key: str = 'image'

def Atari_override(task: str):
    table = {
            'atari': {
                'total_env_steps': 200_000_000,
                'eval_every': 1_000_000
            },
            'atari100k': {
                'total_env_steps': 100_000
            }
        }
    
    if task not in table.keys():
        raise ValueError(f'Atari Override table does not has key: {task}')
    
    return table[task]