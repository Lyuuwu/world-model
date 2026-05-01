from typing import Literal

import gymnasium as gym

from .wrapper import (
    StickyActionWrapper, AtariActionSetWrapper, MaxNoopWrapper, FireResetWrapper,
    EpisodicLifeWrapper, MaxAndSkipWrapper, RewardClipWrapper,
    GrayscaleWrapper, ResizeWrapper, FrameStackWrapper,
    TimeLimitWrapper, DictObsWrapper
)

def make_atari(
    env: gym.Env,
    *,
    action_repeat: int=4,
    actions: Literal['needed', 'all']='needed',
    max_noop: int=0,
    sticky_prob: float=0.0,
    fire_reset: bool=False,
    life_loss_terminal: bool=False,
    reward_clip: Literal['sign', 'tanh', 'scale'] | None=None,
    reward_scale: float=1.0,
    grayscale: bool=True,
    resize: tuple[int, int]=(64, 64),
    resize_method: Literal['opencv', 'pillow']='pillow',
    frame_stack: int=1,
    max_episode_steps: int=108000,
    obs_key: str='image'
) -> gym.Env:
    env = AtariActionSetWrapper(env, actions)

    if sticky_prob > 0:
        env = StickyActionWrapper(env, sticky_prob)
        
    if max_noop > 0:
        env = MaxNoopWrapper(env, max_noop)
        
    if fire_reset:
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetWrapper(env)
            
    if life_loss_terminal:
        env = EpisodicLifeWrapper(env)
        
    assert action_repeat >= 1
    if action_repeat > 1:
        env = MaxAndSkipWrapper(env, repeat=action_repeat)
        
    if reward_clip is not None:
        env = RewardClipWrapper(env, mode=reward_clip, scale=reward_scale)
        
    if grayscale:
        env = GrayscaleWrapper(env, keep_dim=True)
        
    env = ResizeWrapper(env, size=resize, method=resize_method)
    
    if frame_stack > 1:
        env = FrameStackWrapper(env, num_stack=frame_stack)
        
    agent_steps = max_episode_steps // action_repeat
    env = TimeLimitWrapper(env, max_steps=agent_steps)
    
    env = DictObsWrapper(env, obs_key=obs_key)
    
    return env
