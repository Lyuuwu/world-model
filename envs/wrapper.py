from collections import deque
from typing import Literal

import gymnasium as gym
import numpy as np
import cv2

class StickyActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, sticky_prob: float=0.25):
        super().__init__(env)
        assert 0.0 <= sticky_prob < 1.0, f'sticky_prob should set in [0, 1), sticky_prob={sticky_prob}'
        self._sticky_prob = sticky_prob
        self._last_action: int | None = None
    
    def reset(self, *, seed: int | None=None, options: dict | None=None) -> tuple[np.ndarray, dict]:
        self._last_action = None
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: int):
        
        if self._last_action is not None:
            if np.random.random() < self._sticky_prob:
                action = self._last_action
        
        self._last_action = action
        return self.env.step(action)
    
class MaxNoopWrapper(gym.Wrapper):
    '''
    Noop action = No-op action = 發呆
    '''
    
    def __init__(self, env: gym.Env, max_noop: int=30, noop_action: int=0):
        super.__init__(env)
        assert max_noop >= 0, f'max_noop cannot be negative: max_noop={max_noop}'
        self._max_noop = max_noop
        self._noop_action = noop_action
        
    def reset(self, *, seed: int | None, options: dict | None=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        noop_n = np.random.randint(0, self._max_noop + 1)
        
        for _ in range(noop_n):
            obs, _, term, trun, info = self.env.step(self._noop_action)
            
            if term or trun:
                return self.reset()
        
        return obs, info
    
    def step(self, action: int):
        return self.env.step(action)
    
class FireResetWrapper(gym.Wrapper):
    '''
    reset 後需要 FIRE 的
    '''
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        assert hasattr(env.unwrapped, 'get_action_meanings')
        action_meanings = env.unwrapped.get_action_meanings()
        assert 'FIRE' in action_meanings, (
            f'FireResetWrapper requires FIRE action, got: {action_meanings}'
        )
        self._fire_action = action_meanings.index('FIRE')
        
    def reset(self, *, seed: int | None=None, options: dict | None=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs, _, term, trun, info = self.env.step(self._fire_action)
        
        if term or trun:
            return self.reset()
            
        return obs, info
    
    def step(self, action: int):
        return self.env.step(action)
    
class EpisodicLifeWrapper(gym.Wrapper):
    '''life loss 時 讓 agent 認為 ep 結束，但環境繼續'''
    
    def __init__(self, env):
        super().__init__(env)
        self._lives: int = 0
        self._real_done: bool = True
        
    def reset(self, *, seed: int | None=None, options: dict | None=None) -> tuple[np.ndarray, dict]:
        if self._real_done:
            obs, info = self.env.reset()
        else:
            obs, _, term, trun, info = self.env.step(0) # NOOP
        
        self._lives = self.env.unwrapped.ale.lives()
        return obs, info
    
    def step(self, action: int):
        obs, rew, term, trun, info = self.env.step(action)
        self._real_done = term or trun
        lives = self.env.unwrapped.ale.lives()
        
        life_loss = False
        if lives < self._lives and not term:
            life_loss = True
            term = True
        
        self._lives = lives
        info['real_terminated'] = self._real_done
        
        return obs, rew, (term or life_loss), trun, info
    
class ActionRepeatWrapper(gym.Wrapper):
    ''' 同一個動作重複做 N 次， reward 累加 '''
    
    def __init__(self, env: gym.Env, repeat: int=4):
        super().__init__(env)
        assert repeat >=1, f'repeat should >= 1, repeat={repeat}'
        self._repeat = repeat
        
    def step(self, action: int):
        sum = 0.0
        
        for _ in range(self._repeat):
            obs, rew, term, trun, info = self.env.step(action)
            sum += rew
            
            if term or trun:
                break
            
        return obs, sum, term, trun, info
    
class MaxAndSkipWrapper(gym.Wrapper):
    ''' 取最後2幀, 然後做 Action repeat '''

    def __init__(self, env: gym.Env, repeat: int=4):
        super().__init__(env)
        assert repeat >=1, f'repeat should >= 1, repeat={repeat}'
        self._repeat = repeat
        self._obs_buffer: list[np.ndarray] = []

    def step(self, action: int):
        self._obs_buffer = []
        sum = 0.0

        for _ in self._repeat:
            obs, rew, term, trun, info = self.env.step(action)
            sum += rew
            self._obs_buffer.append(obs)

            if term or trun:
                break
        
        max_obs = np.maximum(self._obs_buffer[-1], self._obs_buffer[-2])

        return max_obs, sum, term, trun, info
    
    def reset(self, *, seed: int | None=None, options: dict | None=None) -> tuple[np.ndarray, dict]:
        self._obs_buffer = []
        return self.env.reset(seed=seed, options=options)

class GrayscaleWrapper(gym.Wrapper):
    ''' RGB to gray '''

    def __init__(self, env: gym.Env, keep_dim: bool=True):
        super().__init__(env)
        self._keep_dim = keep_dim

        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)
        assert len(old_space.shape) == 3 and old_space.shape[-1] == 3

        # old_space: Box(0, 255, (H, W, 3), uint8)

        h, w, _ = old_space.shape
        new_shape = (h, w, 1) if keep_dim else (h, w)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        R = obs[..., 0]
        G = obs[..., 1]
        B = obs[..., 2]

        gray = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)
        
        if self._keep_dim:
            gray = gray[..., np.newaxis]
        
        return gray
    
class ResizeWrapper(gym.ObservationWrapper):
    ''' 把圖片 resize 到指定大小 '''

    def __init__(self, env: gym.Wrapper, size: tuple[int, int]=(64, 64)):
        super().__init__(env)
        self._size = size

        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)

        if len(old_space.shape) == 3:
            channels = old_space.shape[-1]
            new_shape = (*size, channels)
        elif len(old_space) == 2:
            new_shape = size
        else:
            raise ValueError(f'Expected 2D or 3D obs, got shape {old_space.shape}')

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        resized = cv2.resize(obs, self._size[::-1], interpolation=cv2.INTER_AREA)
        if obs.shape == 3:
            resized = resized[..., np.newaxis]
        return resized.astype(np.uint8)
    
class FrameStackWrapper(gym.Wrapper):
    ''' 沿著 channel axis 堆疊 N 幀 '''

    def __init__(self, env: gym.Env, num_stack: int=4):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames: deque[np.array] = deque(maxlen=num_stack)

        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)
        assert len(old_space.shape) == 3, 'FrameStack requires (H, W, C) obs'

        h, w, c = old_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c * num_stack), dtype=np.uint8
        )

    def reset(self, *, seed: int | None=None, options: dict | None=None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action: int):
        obs, rew, term, trun, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), rew, term, trun, info
    
    def _get_obs(self) -> np.ndarray:
        return np.concat(list(self._frames), axis=-1)  # (H, W, C * N)

class RewardClipWrapper(gym.Wrapper):
    ''' Clip reward '''

    def __init__(self, env: gym.Env, mode: Literal['sign', 'tanh', 'scale'] = 'sign', scale: float=1.0):
        super().__init__(env)
        self._mode = mode
        self._scale = scale

    def step(self, action: int):
        obs, rew, term, trun, info = self.env.step(action)
        info['reward'] = rew

        if self._mode == 'sign':
            rew = np.sign(rew)
        elif self._mode == 'tanh':
            rew = np.tanh(rew)
        elif self._mode == 'scale':
            rew = rew / self._scale
        else:
            raise ValueError(f'there is no mode name = {self._mode}')
        
        return obs, rew, term, trun, info
