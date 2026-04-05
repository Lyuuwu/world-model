from typing import Any

import torch.nn as nn

from shared.config import Config


def build_agent(
    agent_name: str,
    obs_space: dict,
    num_actions: int,
    config: Config,
) -> nn.Module:
    _BUILDERS = {
        'dreamerv3': _build_dreamerv3,
    }
    if agent_name not in _BUILDERS:
        raise ValueError(
            f'Unknown agent: {agent_name}. '
            f'Available: {list(_BUILDERS.keys())}'
        )
    return _BUILDERS[agent_name](obs_space, num_actions, config)

def _build_dreamerv3(
    obs_space: dict,
    action_dim: int,
    config,
) -> nn.Module:
    from agents.dreamerv3.builder import build
    return build(obs_space, action_dim, config)
