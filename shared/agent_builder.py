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
    num_actions: int,
    config: Config,
) -> nn.Module:
    from agents.dreamerv3.agent import DreamerV3Agent
    from agents.dreamerv3.types import DreamerV3Config

    wm_kwargs = _to_dict(config.get('wm_kwargs', {}))
    ac_kwargs = _to_dict(config.get('ac_kwargs', {}))
    loss_scales = _to_dict(config.get('loss_scales', {}))

    _BAD_AC_KEYS = {
        'policy_layers', 'value_layers', 'policy_units', 'value_units',
        'imag_horizon', 'gamma', 'lambda_', 'entropy_scale',
        'slow_target_fraction', 'slow_target_update',
    }
    bad_found = set(ac_kwargs.keys()) & _BAD_AC_KEYS
    if bad_found:
        print(f'[agent_builder]  ac_kwargs include incorrect key: {bad_found}')
        print(f'  These keys wont do anything ! please check the key name'
              f'DreamerActorCritic.__init__ signature。')

        _KEY_FIXES = {
            'policy_layers': 'layers',
            'value_layers': 'layers',
            'policy_units': 'units',
            'value_units': 'units',
            'lambda_': 'lam',
            'entropy_scale': 'actent',
            'slow_target_fraction': 'slow_decay',
        }
        for bad_key in bad_found:
            if bad_key in _KEY_FIXES:
                print(f'  {bad_key} -> should be {_KEY_FIXES[bad_key]}')

    agent_config = DreamerV3Config(
        # optimizer
        lr=config.get('lr', 4e-5),
        beta1=config.get('beta1', 0.9),
        beta2=config.get('beta2', 0.999),
        eps=config.get('eps', 1e-20),
        agc=config.get('agc', 0.3),

        # architecture
        horizon=config.get('horizon', 333),
        contdisc=config.get('contdisc', True),
        discrete=config.get('discrete', True),

        # gradient flow
        ac_grads=config.get('ac_grads', False),
        reward_grad=config.get('reward_grad', False),
        repval_grad=config.get('repval_grad', False),
        repval_loss=config.get('repval_loss', False),

        # imagination
        imag_horizon=ac_kwargs.get('imag_horizon',
                                   config.get('imag_horizon', 15)),
        imag_last=config.get('imag_last', None),

        # compile
        use_compile=config.get('use_compile', False),
        compile_mode=config.get('compile_mode', 'reduce-overhead'),

        # sub-module kwargs
        wm_kwargs=wm_kwargs,
        ac_kwargs=ac_kwargs,
        loss_scales=loss_scales,
    )

    return DreamerV3Agent(
        obs_space=obs_space,
        action_dim=num_actions,
        config=agent_config,
    )


def _to_dict(val: Any) -> dict:
    if hasattr(val, 'to_dict'):
        return val.to_dict()
    if hasattr(val, '_data'):
        return dict(val._data)
    if isinstance(val, dict):
        return val
    return {}