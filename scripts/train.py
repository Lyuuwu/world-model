import sys
import argparse
from pathlib import Path

import torch

from shared.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='World Model RL Training')
    p.add_argument('--agent', type=str, required=True,
                   help='Agent name (e.g. dreamerv3, diamond, iris)')
    p.add_argument('--task', type=str, required=True,
                   help='Task in "domain_game" format (e.g. atari_pong)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='auto',
                   help='"auto", "cpu", "cuda", "cuda:0", etc.')
    p.add_argument('--override', type=str, default=None,
                   help='Comma-separated key=value overrides (e.g. "lr=3e-4,batch_size=32")')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint directory for resume')
    return p.parse_args()
 
 
def resolve_device(device_str: str) -> 'torch.device':
    import torch
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def get_trainer_class(trainer_type: str):
    from shared.trainer_interleaved import InterleavedTrainer
 
    _TRAINER_MAP = {
        'interleaved': InterleavedTrainer,
    }
 
    if trainer_type not in _TRAINER_MAP:
        raise ValueError(
            f'Unknown trainer type: {trainer_type}. '
            f'Available: {list(_TRAINER_MAP.keys())}'
        )
    return _TRAINER_MAP[trainer_type]

def _import_agent(agent_name: str):
    '''
    觸發 agent module 的 import，讓 @register 生效。
    '''
    if agent_name == 'dreamerv3':
        import agents.dreamerv3.agent  # noqa: F401
    # elif agent_name == 'edeline':
    #     import agents.edeline.agent
    # elif agent_name == 'iris':

    else:
        raise ValueError(
            f'Unknown agent: {agent_name}. '
            f'Add import in scripts/train.py::_import_agent()'
        )
        
def bootstrap(config: 'Config', device: 'torch.device') -> dict:
    '''
    構建所有 training 元件:
      env, eval_env, agent, buffer, logger, trainer
    '''
    import torch
    from shared.trainer_base import seed_everything, SimpleLogger
    from shared.agent_builder import build_agent
    from shared.replay_buffer import EpisodeReplayBuffer
    from envs import make_vec_env, make_env, get_spaces
 
    agent_name = config.agent
    task = config.task
    seed = config.seed
 
    # --- 1. Seed ---
    seed_everything(seed)
    print(f'[Bootstrap] seed={seed}, device={device}')
 
    # --- 2. Env ---
    env_config = config.get('env', Config({}))
    if hasattr(env_config, 'to_dict'):
        env_config_dict = env_config.to_dict()
    elif hasattr(env_config, '_data'):
        env_config_dict = dict(env_config._data)
    else:
        env_config_dict = env_config if isinstance(env_config, dict) else {}
 
    num_envs = config.get('num_envs', 1)
    vec_env = make_vec_env(task, num_envs, env_config_dict, base_seed=seed)
    eval_env = make_env(task, env_config_dict, seed=seed + 1000)
 
    print(f'[Bootstrap] task={task}, num_envs={num_envs}')
 
    # --- 3. Spaces ---
    obs_space, num_actions = get_spaces(task, env_config_dict)
    print(f'[Bootstrap] obs_space keys={list(obs_space.keys())}, num_actions={num_actions}')
 
    # --- 4. Agent ---
    _import_agent(agent_name)
    agent = build_agent(agent_name, obs_space, num_actions, config)
    agent = agent.to(device)
 
    param_count = sum(p.numel() for p in agent.parameters())
    print(f'[Bootstrap] Agent: {agent_name}, params={param_count:,}')
 
    # --- 5. Buffer ---
    buffer = EpisodeReplayBuffer(
        capacity=config.get('buffer_capacity', 1000000),
        min_episode_len=config.get('min_episode_len', 2),
        device=str(device),
    )
 
    # --- 6. Logger ---
    run_name = f'{agent_name}_{task}_s{seed}'
    log_dir = Path(config.get('log_dir', 'runs')) / run_name
    logger = SimpleLogger(str(log_dir), use_tb=True)
    print(f'[Bootstrap] Logging to {log_dir}')
 
    # --- 7. Trainer ---
    trainer_type = config.get('trainer', 'interleaved')
    TrainerClass = get_trainer_class(trainer_type)
    trainer = TrainerClass(
        agent=agent,
        vec_env=vec_env,
        eval_env=eval_env,
        buffer=buffer,
        logger=logger,
        config=config,
        device=device,
    )
 
    return {
        'trainer': trainer,
        'agent': agent,
        'vec_env': vec_env,
        'eval_env': eval_env,
        'buffer': buffer,
        'logger': logger,
    }
    
def main():
    args = parse_args()
 
    # --- Config composition ---
    from shared.config import compose_config, Config
 
    config_dict = compose_config(
        agent=args.agent,
        task=args.task,
        override_str=args.override,
        project_root=PROJECT_ROOT,
    )
 
    # CLI args 優先
    config_dict['seed'] = args.seed
    if args.resume:
        config_dict['resume'] = args.resume
    if args.device != 'auto':
        config_dict['device'] = args.device
 
    config = Config(config_dict)
 
    # --- Device ---
    device = resolve_device(config.get('device', 'auto'))
 
    # --- Print banner ---
    print('=' * 60)
    print(f'  Agent: {args.agent}')
    print(f'  Task:  {args.task}')
    print(f'  Seed:  {args.seed}')
    print(f'  Device: {device}')
    print('=' * 60)
 
    # --- Bootstrap + Run ---
    components = bootstrap(config, device)
    try:
        components['trainer'].run()
    except KeyboardInterrupt:
        print('\n[Interrupted] Saving checkpoint...')
        components['trainer']._save_checkpoint(tag='interrupted')
    finally:
        components['vec_env'].close()
        components['eval_env'].close()
 
 
if __name__ == '__main__':
    main()