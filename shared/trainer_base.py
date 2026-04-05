import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import AgentBase, BufferBase
from .logger import JSONLLogger
from .config import Config

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class TrainerBase(ABC):
    def __init__(
        self,
        agent: AgentBase,
        vec_env: Any,
        eval_env: Any,
        buffer: BufferBase,
        logger: JSONLLogger,
        config: Config,
        device: torch.device,
        use_checkpoint: bool=False,
    ):
        self.agent = agent
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger
        self.config = config
        self.device = device
        self.use_checkpoint = use_checkpoint
        
        self.num_envs = vec_env.num_envs
        self._global_env_step = 0
        self._num_actions = self._infer_num_actions()
        
    def _infer_num_actions(self) -> int:
        space = self.eval_env.action_space
        if hasattr(space, 'n'):
            return space.n
        return space.shape[0]
    
    def run(self) -> None:
        self._setup()
        self._prefill()
        self._main_loop()
        self._final_eval()
        self._save_checkpoint(tag='final')
        self.logger.close()
    
    @abstractmethod
    def _main_loop(self) -> None:
        return NotImplementedError
    
    def _setup(self) -> None:
        resume_path = self.config.get('resume', None)
        if resume_path is not None:
            self._load_checkpoint(resume_path)
            print(f'[Trainer] Resumed from {resume_path}, '
                  f'env_step={self._global_env_step}')
        else:
            print(f'[Trainer] Starting fresh training')
    
    def _prefill(self) -> None:
        seed_steps = self.config.get('seed_steps', 1024)
        
        if self.buffer.total_steps >= seed_steps:
            print(f'[Prefill] Buffer already has {self.buffer.total_steps} steps',
                  f'skipping')
            return
        
        target = seed_steps - self.buffer.total_steps
        print(f'[Prefill] Cllecting {target} random steps ...')
        
        obs_list, _ = self.vec_env.reset()
        collected = 0
        
        while collected < target:
            actions = np.array([
                # self.vec_env.single_action_space.sample()
                self.eval_env.unwrapped.action_space.sample()
                for _ in range(self.num_envs)
            ])
            
            next_obs_list, rews, terms, truns, infos = self.vec_env.step(actions)
            
            for i in range(self.num_envs):
                action_vec = self._action_to_vector(actions[i])
                done = terms[i] or truns[i]
                
                if done:
                    final_obs = infos[i].get('final_observation', obs_list[i])
                    self.buffer.add_step(
                        obs = final_obs,
                        action = action_vec,
                        reward = float(rews[i]),
                        is_first = bool(obs_list[i].get('is_first', False)),
                        is_last = True,
                        is_terminal = bool(infos[i].get('real_terminated', terms[i]))
                    )
                else:
                    self.buffer.add_step(
                        obs = obs_list[i],
                        action = action_vec,
                        reward = float(rews[i]),
                        is_first = bool(obs_list[i].get('is_first', False)),
                        is_last = False,
                        is_terminal = False,
                    )
                
                collected += 1
            
            obs_list = next_obs_list
        
        print(f'[Prefill] Done. Buffer has {self.buffer.total_steps} steps')
    
    @torch.no_grad()
    def collect_step(
        self,
        obs_list: list[dict],
        agent_state: dict[str, torch.Tensor],
        prev_action: torch.Tensor
    ) -> tuple[list[dict], dict[str, torch.Tensor], torch.Tensor, dict]:
        obs_batch = self._batch_obs(obs_list)
        is_first = torch.tensor(
            [o.get('is_first', False) for o in obs_list],
            dtype=torch.bool, device=self.device
        )
        
        action, new_state = self.agent.policy(
            obs_batch, agent_state, prev_action, is_first
        )
        
        if self.config.get('discrete', True):
            act_np = action.argmax(dim=-1).cpu().numpy()
        else:
            act_np = action.cpu().numpy()
        
        next_obs_list, rews, terms, truns, infos = self.vec_env.step(act_np)
        
        for i in range(self.num_envs):
            action_vec = action[i].cpu().numpy()
            done = terms[i] or truns[i]
            
            if done:
                final_obs = infos[i].get('final_observation', obs_list[i])
                self.buffer.add_step(
                    obs = final_obs,
                    action = action_vec,
                    reward = float(rews[i]),
                    is_first = bool(obs_list[i].get('is_first', False)),
                    is_last = True,
                    is_terminal = bool(infos[i].get('real_terminated', terms[i]))
                )
            else:
                self.buffer.add_step(
                    obs = obs_list[i],
                    action = action_vec,
                    reward = float(rews[i]),
                    is_first = bool(obs_list[i].get('is_first', False)),
                    is_last = False,
                    is_terminal = False
                )
            
        self._global_env_step += self.num_envs
        
        info = {'reward': rews.tolist(), 'done': (terms | truns).tolist()}
        
        return next_obs_list, new_state, action, info
    
    def _eval_episodes(self, n_episodes: int) -> dict[str, float]:
        self.agent.eval()
        returns = []
        lengths = []
 
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            state = self.agent.initial_state(1, self.device)
            prev_act = self.agent.initial_prevact(1, self.device)
            done = False
            ep_return = 0.0
            ep_len = 0
 
            while not done:
                obs_t = self._single_obs_to_tensor(obs)
                is_first = torch.tensor([ep_len == 0],
                                        dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    action, state = self.agent.policy(
                        obs_t, state, prev_act, is_first, train=False
                    )
 
                if self.config.get('discrete', True):
                    act_np = action.argmax(dim=-1).cpu().numpy()[0]
                else:
                    act_np = action.cpu().numpy()[0]
 
                obs, rew, term, trun, _ = self.eval_env.step(act_np)
                prev_act = action
                ep_return += float(rew)
                ep_len += 1
                done = term or trun
 
            returns.append(ep_return)
            lengths.append(ep_len)
 
        self.agent.train()
 
        return {
            'return_mean': float(np.mean(returns)),
            'return_std': float(np.std(returns)),
            'return_min': float(np.min(returns)),
            'return_max': float(np.max(returns)),
            'length_mean': float(np.mean(lengths)),
        }
 
    def _final_eval(self) -> None:
        n = self.config.get('eval_episodes', 10)
        metrics = self._eval_episodes(n)
        self.logger.log_print(metrics, self._global_env_step, prefix='final_eval')
        
    def _save_checkpoint(self, tag: str | int) -> None:
        if not self.use_checkpoint:
            return
        
        ckpt_dir = self.logger.log_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f'ckpt_{tag}.pt'
 
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'global_env_step': self._global_env_step,
            'buffer_stats': self.buffer.stats,
        }, path)
        print(f'[Checkpoint] Saved -> {path}')
 
    def _load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(ckpt['agent_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self._global_env_step = ckpt.get('global_env_step', 0)
        
    def _action_to_vector(self, action_int: int | np.ndarray) -> np.ndarray:
        '''把 discrete action int -> one-hot vector'''
        if self.config.get('discrete', True):
            vec = np.zeros(self._num_actions, dtype=np.float32)
            vec[int(action_int)] = 1.0
            return vec
        return np.asarray(action_int, dtype=np.float32)
 
    def _batch_obs(self, obs_list: list[dict]) -> dict[str, torch.Tensor]:
        '''list[dict] > dict[tensor(B, ...)]'''
        keys = [k for k in obs_list[0] if k not in
                ('is_first', 'is_last', 'is_terminal', 'reward')]
        batch = {}
        for k in keys:
            arr = np.stack([o[k] for o in obs_list], axis=0)
            if arr.dtype == np.uint8:
                batch[k] = torch.from_numpy(arr).to(self.device)
            else:
                batch[k] = torch.from_numpy(arr).float().to(self.device)
        return batch
 
    def _single_obs_to_tensor(self, obs: dict) -> dict[str, torch.Tensor]:
        '''single dict obs > dict[tensor(1, ...)]'''
        result = {}
        for k, v in obs.items():
            if k in ('is_first', 'is_last', 'is_terminal', 'reward'):
                continue
            arr = np.asarray(v)
            if arr.dtype == np.uint8:
                result[k] = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            else:
                result[k] = (torch.from_numpy(arr).float()
                             .unsqueeze(0).to(self.device))
        return result