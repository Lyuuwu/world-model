import random
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
 
import numpy as np
import torch
 
from shared.config import Config

def stack_dict_list(
    dict_list: list[dict[str, np.ndarray]],
    device: torch.device = torch.device('cpu'),
) -> dict[str, torch.Tensor]:
    '''
    list of dict obs > batched dict of tensors
    
    [{key: (H,W,C)}, ...] > {key: (B, H, W, C)}
 
    image 保持 uint8
          
    scalar (reward, is_first, ...) 轉 float32 or bool
    '''
    keys = dict_list[0].keys()
    result = {}
    for k in keys:
        vals = [d[k] for d in dict_list]
        stacked = np.stack(vals, axis=0)
        if stacked.dtype == np.bool_:
            result[k] = torch.from_numpy(stacked).to(device)
        elif stacked.dtype == np.uint8:
            result[k] = torch.from_numpy(stacked).to(device)
        else:
            result[k] = torch.from_numpy(stacked).to(dtype=torch.float32, device=device)
    return result

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
class SimpleLogger:
    '''
    最小 logger: console print + optional TensorBoard
    '''
 
    def __init__(self, log_dir: str, use_tb: bool = True):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(str(self._log_dir))
            except ImportError:
                print('[Logger] tensorboard not installed, falling back to console only')
        
        self._jsonl_path = self._log_dir / 'metrics.jsonl'
        self._jsonl_file = open(self._jsonl_path, 'a', buffering=1, encoding='utf-8')
 
    def log(self, metrics: dict[str, float], step: int, prefix: str = ''):
        '''
        寫入 TensorBoard + JSONL
        '''
        record: dict[str, Any] = {'step': step}
 
        for k, v in metrics.items():
            tag = f'{prefix}/{k}' if prefix else k
 
            # TensorBoard
            if self._writer:
                scalar = self._to_scalar(v)
                if scalar is not None:
                    self._writer.add_scalar(tag, scalar, step)
 
            # JSONL: 只寫數值型
            scalar = self._to_scalar(v)
            if scalar is not None:
                record[tag] = scalar
 
        # step 之外至少有一個數值才寫行，避免空行污染
        if len(record) > 1:
            self._jsonl_file.write(json.dumps(record) + '\n')
 
    def log_print(self, metrics: dict[str, float], step: int, prefix: str = ''):
        self.log(metrics, step, prefix)
        parts = [f'{prefix} step={step}'] if prefix else [f'step={step}']
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f'{k}={v:.4f}')
            else:
                parts.append(f'{k}={v}')
        print(' | '.join(parts))
 
    def close(self):
        if self._writer:
            self._writer.close()
            
class TrainerBase(ABC):
    '''
    run() → _setup() → _prefill() → _main_loop() → _final_eval()
    '''
 
    def __init__(
        self,
        agent: torch.nn.Module,
        vec_env: Any,                   # SyncVectorEnvWrapper
        eval_env: Any,                  # single gym.Env (wrapped)
        buffer: Any,                    # EpisodeReplayBuffer
        logger: SimpleLogger,
        config: Config,
        device: torch.device,
    ):
        self.agent = agent
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger
        self.config = config
        self.device = device
 
        self.num_envs = vec_env.num_envs
        self._global_env_step = 0       # 追蹤 total env steps (across resume)
 
    # --- Template method ---
 
    def run(self):
        ''' 完整 training 流程 '''
        self._setup()
        self._prefill()
        self._main_loop()
        self._final_eval()
        self._save_checkpoint(tag='final')
        self.logger.close()
 
    @abstractmethod
    def _main_loop(self):
        raise NotImplementedError
 
    # --- Setup / Resume ---
 
    def _setup(self):
        ''' 載入 checkpoint (如果有) '''
        resume_path = self.config.get('resume', None)
        if resume_path is not None:
            self._load_checkpoint(resume_path)
            print(f'[Trainer] Resumed from {resume_path}, env_step={self._global_env_step}')
        else:
            print(f'[Trainer] Starting fresh training')
 
    # --- Prefill ---
 
    def _prefill(self):
        '''
        用 random policy 填充 buffer seed_steps
        
        確保第一次 train 就有足夠的 data
        '''
        seed_steps = self.config.get('seed_steps', 1024)
 
        if self.buffer.total_steps >= seed_steps:
            print(f'[Prefill] Buffer already has {self.buffer.total_steps} steps, skipping')
            return
 
        target = seed_steps - self.buffer.total_steps
        print(f'[Prefill] Collecting {target} random steps...')
 
        obs_list, _ = self.vec_env.reset()
 
        collected = 0
        while collected < target:
            # random actions
            actions = np.array([
                self.eval_env.unwrapped.action_space.sample()
                for _ in range(self.num_envs)
            ])
 
            next_obs_list, rews, terms, truns, infos = self.vec_env.step(actions)
 
            for i in range(self.num_envs):
                # 把 action 轉成 one-hot (Atari discrete)
                action_onehot = np.zeros(
                    self.eval_env.unwrapped.action_space.n, dtype=np.float32
                )
                action_onehot[actions[i]] = 1.0
 
                done = terms[i] or truns[i]
 
                if done:
                    # done step: 存 final obs
                    final_obs = infos[i].get('final_observation', obs_list[i])
                    self.buffer.add_step(
                        obs=final_obs,
                        action=action_onehot,
                        reward=float(rews[i]),
                        is_first=bool(obs_list[i].get('is_first', False)),
                        is_last=True,
                        is_terminal=bool(infos[i].get('real_terminated', terms[i])),
                    )
                else:
                    self.buffer.add_step(
                        obs=obs_list[i],
                        action=action_onehot,
                        reward=float(rews[i]),
                        is_first=bool(obs_list[i].get('is_first', False)),
                        is_last=False,
                        is_terminal=False,
                    )
                collected += 1
 
            obs_list = next_obs_list
 
        print(f'[Prefill] Done. Buffer has {self.buffer.total_steps} steps')
 
    # --- Collect step ---
 
    @torch.no_grad()
    def collect_step(
        self,
        obs_list: list[dict],
        agent_state: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
    ) -> tuple[list[dict], dict[str, torch.Tensor], torch.Tensor, dict]:
        '''
        單步 data collection:
          1. batch obs -> agent.policy() -> action
          2. vec_env.step(action) -> next obs
          3. buffer.add_step()
          4. 回傳 (next_obs_list, new_state, action, metrics)
        '''
        # --- batch obs for policy ---
        batched_obs = stack_dict_list(obs_list, self.device)
        is_first = batched_obs['is_first']
 
        # --- agent policy ---
        action, new_state = self.agent.policy(
            obs=batched_obs,
            state=agent_state,
            prev_action=prev_action,
            is_first=is_first,
        )
        # action: (B, action_dim) tensor — one-hot for discrete
 
        # --- env step ---
        # 轉成 int actions for Atari
        action_np = action.cpu().numpy()
        if self.config.get('discrete', True):
            env_actions = action_np.argmax(axis=-1)  # (B,) int
        else:
            env_actions = action_np
 
        next_obs_list, rews, terms, truns, infos = self.vec_env.step(env_actions)
 
        # --- store transitions ---
        for i in range(self.num_envs):
            done = terms[i] or truns[i]
            if done:
                final_obs = infos[i].get('final_observation', next_obs_list[i])
                self.buffer.add_step(
                    obs=final_obs,
                    action=action_np[i],
                    reward=float(rews[i]),
                    is_first=bool(obs_list[i].get('is_first', False)),
                    is_last=True,
                    is_terminal=bool(infos[i].get('real_terminated', terms[i])),
                )
            else:
                self.buffer.add_step(
                    obs=obs_list[i],
                    action=action_np[i],
                    reward=float(rews[i]),
                    is_first=bool(obs_list[i].get('is_first', False)),
                    is_last=False,
                    is_terminal=False,
                )
 
        self._global_env_step += self.num_envs
 
        return next_obs_list, new_state, action, {}
 
    # ── Eval ─────────────────────────────────────────────
 
    @torch.no_grad()
    def _eval_episodes(self, n_episodes: int) -> dict[str, float]:
        '''
        跑 n_episodes 個完整 eval episode，回傳 metrics
        
        用 eval_env (single env, no auto-reset)
        '''
        self.agent.eval()
        returns = []
        lengths = []
 
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            state = self.agent.initial_state(1, self.device)
            prevact = self.agent.initial_prevact(1, self.device)
            ep_return = 0.0
            ep_len = 0
 
            done = False
            while not done:
                # 單步 obs → tensor
                obs_t = stack_dict_list([obs], self.device)
                is_first = obs_t['is_first']
 
                action, state = self.agent.policy(
                    obs=obs_t, state=state,
                    prev_action=prevact, is_first=is_first,
                )
 
                action_np = action.cpu().numpy()[0]
                if self.config.get('discrete', True):
                    env_action = int(action_np.argmax())
                else:
                    env_action = action_np
 
                obs, rew, term, trun, info = self.eval_env.step(env_action)
                prevact = action
 
                ep_return += rew
                ep_len += 1
                done = term or trun
 
            returns.append(ep_return)
            lengths.append(ep_len)
 
        self.agent.train()
 
        return {
            'eval/return_mean': float(np.mean(returns)),
            'eval/return_std': float(np.std(returns)),
            'eval/length_mean': float(np.mean(lengths)),
            'eval/num_episodes': n_episodes,
        }
 
    def _final_eval(self):
        n = self.config.get('eval_episodes', 10)
        print(f'[Final eval] Running {n} episodes...')
        metrics = self._eval_episodes(n)
        self.logger.log_print(metrics, self._global_env_step, prefix='final_eval')
 
    # --- Checkpoint ---
 
    def _save_checkpoint(self, tag: str | int = 'latest'):
        ckpt_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        run_name = f"{self.config['agent']}_{self.config['task']}_s{self.config['seed']}"
        save_dir = ckpt_dir / run_name / str(tag)
        save_dir.mkdir(parents=True, exist_ok=True)
 
        torch.save({
            'agent': self.agent.state_dict(),
            'env_step': self._global_env_step,
            'rng_torch': torch.random.get_rng_state(),
            'rng_numpy': np.random.get_state(),
            'rng_python': random.getstate(),
        }, save_dir / 'checkpoint.pt')
 
        # 存 config snapshot for reproducibility
        import yaml
        with open(save_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
 
        print(f'[Checkpoint] Saved to {save_dir}')
 
    def _load_checkpoint(self, path: str):
        ckpt = torch.load(Path(path) / 'checkpoint.pt', map_location=self.device)
        self.agent.load_state_dict(ckpt['agent'])
        self._global_env_step = ckpt['env_step']
        torch.random.set_rng_state(ckpt['rng_torch'])
        np.random.set_state(ckpt['rng_numpy'])
        random.setstate(ckpt['rng_python'])