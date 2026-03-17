import gymnasium as gym
import numpy as np

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
    '''
    同一個動作重複做 N 次， reward 累加
    '''
    
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