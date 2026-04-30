'''
所有 agent 共用的 env 構建入口。
task 格式: "{domain}_{game}", e.g. "atari_pong", "crafter_crafter"

擴充方式:
  1. 新增 envs/{domain}.py (e.g. envs/crafter.py)
  2. 在 _DOMAIN_BUILDERS 裡加一行
'''

from typing import Callable, Any
from functools import partial

import ale_py
import gymnasium as gym
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Domain builders
# ═══════════════════════════════════════════════════════════════

def _build_atari(game: str, env_config: dict, seed: int | None = None) -> gym.Env:
    '''
    構建一個 wrapped Atari env
    '''
    from .atari import make_atari

    # game name -> ALE id
    ale_name = ''.join(w.capitalize() for w in game.split('_'))
    env_id = f'ALE/{ale_name}-v5'
    raw = gym.make(env_id, render_mode=None)
    if seed is not None:
        raw.action_space.seed(seed)
        raw.observation_space.seed(seed)
        raw.reset(seed=seed)
    return make_atari(raw, **env_config)

_DOMAIN_BUILDERS: dict[str, Callable] = {
    'atari': _build_atari,
    'atari100k': _build_atari
}

def parse_task(task: str) -> tuple[str, str]:
    '''
    "atari_pong" -> ("atari", "pong")
    '''
    parts = task.split('_', 1)
    if len(parts) != 2:
        raise ValueError(
            f'Task must be "domain_game" format, got: {task}. '
            f'Available domains: {list(_DOMAIN_BUILDERS.keys())}'
        )
    return parts[0], parts[1]

def make_env(
    task: str,
    env_config: dict | None = None,
    seed: int | None = None,
) -> gym.Env:
    '''
    構建單個 wrapped env。

    Args:
        task: "atari_pong", "crafter_crafter", etc.
        env_config: forwarded to domain builder (e.g. make_atari kwargs)
        seed: env seed
    '''
    domain, game = parse_task(task)
    if domain not in _DOMAIN_BUILDERS:
        raise ValueError(
            f'Unknown domain: {domain}. '
            f'Available: {list(_DOMAIN_BUILDERS.keys())}'
        )
    env_config = env_config or {}
    return _DOMAIN_BUILDERS[domain](game, env_config, seed)

def make_env_fn(
    task: str,
    env_config: dict | None = None,
    seed: int | None = None,
) -> Callable[[], gym.Env]:
    ''' 回傳一個 callable，呼叫時才構建 env（用於 VecEnv） '''
    return partial(make_env, task=task, env_config=env_config, seed=seed)

def make_vec_env(
    task: str,
    num_envs: int,
    env_config: dict | None = None,
    base_seed: int = 0,
) -> Any:
    '''
    構建 SyncVectorEnvWrapper (多個 env instance)
    每個 env 用不同的 seed: base_seed, base_seed+1, ...
    '''
    from .wrapper import SyncVectorEnvWrapper

    env_fns = [
        make_env_fn(task, env_config, seed=base_seed + i)
        for i in range(num_envs)
    ]
    return SyncVectorEnvWrapper(env_fns)

def get_spaces(task: str, env_config: dict | None = None):
    '''
    構建一個 temp env 來取得 obs_space / action_space, 然後關掉
    回傳 (obs_space_dict, num_actions)
    '''
    env = make_env(task, env_config, seed=0)

    obs, _ = env.reset()

    # 從 obs dict 推 obs_space
    from shared.obs_spec import ObsSpec
    obs_space = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs_space[key] = ObsSpec(
                shape=val.shape,
                dtype=val.dtype,
            )

    num_actions = env.unwrapped.action_space.n

    env.close()
    return obs_space, num_actions

gym.register_envs(ale_py)
