import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def bless_banner():
    bless = r'''
                                  /===-_---~~~~~~~~~------____
                                 |===-~___                _,-'
                  -==\\                         `//~\\   ~~~~`---.___.-~~
              ______-==|                         | |  \\           _-~`
        __--~~~  ,-/-==\\                        | |   `\        ,'
     _-~       /'    |  \\                      / /      \      /
   .'        /       |   \\                   /' /        \   /'
  /  ____  /         |    \`\.__/-~~ ~ \ _ _/'  /          \/'
 /-'~    ~~~~~---__  |     ~-/~         ( )   /'        _--~`
                   \_|      /        _)   ;  ),   __--~~
                     '~~--_/      _-~/-  / \   '-~ \
                    {\__--_/}    / \\_>- )<__\      \
                    /'   (_/  _-~  | |__>--<__|      |
                   |   _/) )-~     | |__>--<__|      |
                   / /~ ,_/       / /__>---<__/      |
                  o-o _//        /-~_>---<__-~      /
                  (^(~          /~_>---<__-      _-~
                 ,/|           /__>--<__/     _-~
              ,//('(          |__>--<__|     /                  .----_
             ( ( '))          |__>--<__|    |                 /' _---_~\
          `-)) )) (           |__>--<__|    |               /'  /     ~\`\
         ,/,'//( (             \__>--<__\    \            /'  //        ||
       ,( ( ((, ))              ~-__>--<_~-_  ~--____---~' _/'/        /'
     `~/  )` ) ,/|                 ~-_~>--<_/-__       __-~ _/
   ._-~//( )/ )) `                    ~~-'_/_/ /~~~~~~~__--~
    ;'( ')/ ,)(                              ~~~~~~~~~~
   ' ') '( (/
     '   '  `
    '''
    
    print(bless)
    print(f'[=== THE GOD DRAGON BLESS THIS PROGRAM ===]')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='World Model RL Training')
    p.add_argument('--agent', type=str, required=True,
                   help='Agent name (dreamerv3, diamond, iris, ...)')
    p.add_argument('--task', type=str, required=True,
                   help='Task in "domain_game" format (e.g. atari_pong)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='auto',
                   help='"auto", "cpu", "cuda", "cuda:0"')
    p.add_argument('--override', type=str, default=None,
                   help='Comma-separated key=value overrides. '
                        'Supports dotpath: "wm_kwargs.act=gelu,lr=3e-4"')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint .pt file for resume')
    p.add_argument('--profile', type=str, default=None,
                   help='Model size profile (e.g. m12m, m50m, m100m)')
    return p.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def _import_agent(agent_name: str) -> None:
    _AGENT_IMPORTS = {
        'dreamerv3': 'agents.dreamerv3.agent',
        # 'iris':      'agents.iris.agent',
        # 'diamond':   'agents.diamond.agent',
        # 'r2i':       'agents.r2i.agent',
        # 'edeline':   'agents.edeline.agent',
    }
    if agent_name not in _AGENT_IMPORTS:
        raise ValueError(
            f'Unknown agent: {agent_name}. '
            f'Available: {list(_AGENT_IMPORTS.keys())}. '
            f'Add import in scripts/train.py::_import_agent()'
        )
    __import__(_AGENT_IMPORTS[agent_name])

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

def bootstrap(config, device: torch.device) -> dict:
    from shared.trainer_base import seed_everything
    from shared.logger import JSONLLogger
    from shared.agent_builder import build_agent
    from shared.replay_buffer import EpisodeReplayBuffer
    from envs import make_vec_env, make_env, get_spaces

    agent_name = config.agent
    task = config.task
    seed = config.seed

    # 1. Seed
    seed_everything(seed)
    print(f'[Bootstrap] seed={seed}, device={device}')

    # 2. Environment
    env_config = _extract_dict(config.get('env', {}))
    num_envs = config.get('num_envs', 1)
    vec_env = make_vec_env(task, num_envs, env_config, base_seed=seed)
    eval_env = make_env(task, env_config, seed=seed + 1000)
    print(f'[Bootstrap] task={task}, num_envs={num_envs}')

    # 3. Spaces
    obs_space, num_actions = get_spaces(task, env_config)
    print(f'[Bootstrap] obs_space keys={list(obs_space.keys())}, '
          f'num_actions={num_actions}')

    # 4. Agent
    _import_agent(agent_name)
    agent = build_agent(agent_name, obs_space, num_actions, config)
    agent = agent.to(device)
    param_count = sum(p.numel() for p in agent.parameters())
    print(f'[Bootstrap] Agent: {agent_name}, params={param_count:,}')

    # 5. Buffer
    buffer = EpisodeReplayBuffer(
        capacity=config.get('buffer_capacity', 1000000),
        min_episode_len=config.get('min_episode_len', 2),
        device=str(device),
    )

    # 6. Logger
    time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run_name = f'{agent_name}_{task}_s{seed}_{time}'
    log_dir = Path(config.get('log_dir', 'runs')) / run_name
    logger = JSONLLogger(str(log_dir))
    logger.save_config(config.to_dict())
    print(f'[Bootstrap] Logging to {log_dir}')

    # 7. Trainer
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


def _extract_dict(val) -> dict:
    ''' Config object -> plain dict) '''
    if hasattr(val, 'to_dict'):
        return val.to_dict()
    if hasattr(val, '_data'):
        return dict(val._data)
    if isinstance(val, dict):
        return val
    return {}

def main():
    args = parse_args()

    from shared.config import compose_config, Config

    config_dict = compose_config(
        agent=args.agent,
        task=args.task,
        override_str=args.override,
        project_root=PROJECT_ROOT,
        profile=args.profile,
    )

    # CLI args first
    config_dict['seed'] = args.seed
    if args.resume:
        config_dict['resume'] = args.resume
    if args.device != 'auto':
        config_dict['device'] = args.device

    config = Config(config_dict)
    device = resolve_device(config.get('device', 'auto'))

    # Print banner
    bless_banner()
    print('=' * 60)
    print(f'  Agent:   {args.agent}')
    print(f'  Task:    {args.task}')
    print(f'  Profile: {args.profile or "default"}')
    print(f'  Seed:    {args.seed}')
    print(f'  Device:  {device}')
    print(f'  Compute: {config.get("compute_dtype", "bfloat16")}')
    print('=' * 60)

    import json
    print('[Config] Effective config:')
    print(json.dumps(config.to_dict(), indent=2, default=str)[:2000])
    print('...')

    # Bootstrap + Run
    components = bootstrap(config, device)
    try:
        components['trainer'].run()
    except KeyboardInterrupt:
        print('\n[Interrupted] Saving checkpoint ...')
        components['trainer']._save_checkpoint(tag='interrupted')
    finally:
        components['vec_env'].close()
        components['eval_env'].close()

if __name__ == '__main__':
    main()