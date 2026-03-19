from typing import Any
 
import torch.nn as nn
 
from shared.config import Config
 
 
def build_agent(
    agent_name: str,
    obs_space: dict,
    num_actions: int,
    config: Config,
) -> nn.Module:
    '''
    根據 agent_name dispatch 到對應的 builder。
    '''
    
    if agent_name == 'dreamerv3':
        return _build_dreamerv3(obs_space, num_actions, config)
    else:
        raise ValueError(f'Unknown agent: {agent_name}')
 
 
# ═══════════════════════════════════════════════════════════════
#  DreamerV3
# ═══════════════════════════════════════════════════════════════
 
def _build_dreamerv3(
    obs_space: dict,
    num_actions: int,
    config: Config,
) -> nn.Module:
    '''
    把 generic Config → DreamerV3Config → DreamerV3Agent。
    '''
    
    from agents.dreamerv3.agent import DreamerV3Agent
 
    # --- 嘗試用現有的 DreamerV3Config ---
    try:
        from agents.dreamerv3.types import DreamerV3Config
 
        # 從 generic config 提取 DreamerV3Config 的 fields
        agent_config = DreamerV3Config(
            # optimizer
            lr=config.get('lr', 1e-4),
            beta1=config.get('beta1', 0.9),
            beta2=config.get('beta2', 0.999),
            eps=config.get('eps', 1e-20),
            agc=config.get('agc', 0.3),
            # architecture
            horizon=config.get('horizon', 333),
            contdisc=config.get('contdisc', True),
            discrete=config.get('discrete', True),
            # sub-module kwargs
            wm_kwargs=config.get('wm_kwargs', Config({})).to_dict()
                if hasattr(config.get('wm_kwargs', {}), 'to_dict')
                else config.get('wm_kwargs', {}),
            ac_kwargs=config.get('ac_kwargs', Config({})).to_dict()
                if hasattr(config.get('ac_kwargs', {}), 'to_dict')
                else config.get('ac_kwargs', {}),
            # loss scales
            loss_scales=config.get('loss_scales', Config({})).to_dict()
                if hasattr(config.get('loss_scales', {}), 'to_dict')
                else config.get('loss_scales', {}),
        )
    except (ImportError, TypeError) as e:
        # 如果 DreamerV3Config 不存在或 fields 不 match，
        # fallback: 直接把 config 當 namespace 傳入
        print(f'[agent_builder] DreamerV3Config construction failed ({e}), '
              f'passing Config directly')
        agent_config = config
 
    return DreamerV3Agent(
        obs_space=obs_space,
        action_dim=num_actions,
        config=agent_config,
    )