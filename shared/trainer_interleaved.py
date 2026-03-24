import time
 
import numpy as np
import torch
 
from shared.trainer_base import TrainerBase
 
 
class InterleavedTrainer(TrainerBase):
    """
    Interleaved collect-train loop:
 
        for step in total_env_steps:
            collect_step()
            if buffer.ready:
                for _ in train_steps:
                    batch = buffer.sample()
                    metrics = agent.train_step(batch)
    """
 
    def _main_loop(self) -> None:
        cfg = self.config
 
        total_steps    = cfg.total_env_steps
        seed_steps     = cfg.get('seed_steps', 1024)
        batch_size     = cfg.batch_size
        seq_len        = cfg.seq_len
        train_ratio    = cfg.get('train_ratio', 1)
        log_every      = cfg.get('log_every', 1000)
        eval_every     = cfg.get('eval_every', 10000)
        checkpoint_every = cfg.get('checkpoint_every', 25000)
 
        # 推斷 autocast device type
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        compute_dtype = self._resolve_dtype(cfg.get('compute_dtype', 'bfloat16'))
 
        # --- Init collection state ---
        obs_list, _ = self.vec_env.reset()
        agent_state = self.agent.initial_state(self.num_envs, self.device)
        prev_action = self.agent.initial_prevact(self.num_envs, self.device)
 
        # --- Tracking ---
        train_metrics_acc: dict[str, list[float]] = {}
        ep_returns: list[float] = []
        ep_return_running = [0.0] * self.num_envs
        grad_steps = 0
        t_start = time.time()
 
        tokens_per_grad_step = batch_size * seq_len
        train_credit = 0.0
 
        print(f'[Train] Starting interleaved loop: {total_steps} env steps, '
              f'train_ratio={train_ratio}, B={batch_size}, T={seq_len}')
 
        step = 0
        while step < total_steps:
            # --- Phase 1: Collect ---
            obs_list, agent_state, prev_action, info = self.collect_step(
                obs_list, agent_state, prev_action,
            )
 
            # track episode returns
            for i in range(self.num_envs):
                rew = float(obs_list[i].get('reward', 0.0))
                ep_return_running[i] += rew
                if obs_list[i].get('is_first', False):
                    if ep_return_running[i] != 0.0 or step > 0:
                        ep_returns.append(ep_return_running[i])
                    ep_return_running[i] = 0.0
 
            step += self.num_envs
 
            # --- Phase 2: Train ---
            if self.buffer.total_steps >= seed_steps:
                train_credit += train_ratio * self.num_envs
                while train_credit >= tokens_per_grad_step:
                    batch = self.buffer.sample(batch_size, seq_len)
                    metrics = self.agent.train_step(
                        batch,
                        device_type=device_type,
                        compute_dtype=compute_dtype,
                    )
                    train_credit -= tokens_per_grad_step
                    grad_steps += 1
 
                    # accumulate metrics for smoothing
                    for k, v in metrics.items():
                        s = self._metric_to_float(v)
                        if s is not None:
                            train_metrics_acc.setdefault(k, []).append(s)
 
            # --- Phase 3: Log ---
            if step % log_every < self.num_envs and train_metrics_acc:
                averaged = {
                    k: sum(v) / len(v)
                    for k, v in train_metrics_acc.items()
                }
                elapsed = time.time() - t_start
                averaged['fps'] = step / max(elapsed, 1e-6)
                averaged['buffer_steps'] = self.buffer.total_steps
                averaged['grad_steps'] = grad_steps
 
                if ep_returns:
                    averaged['ep_return_mean'] = float(
                        np.mean(ep_returns[-20:])
                    )
 
                self.logger.log_print(
                    averaged, self._global_env_step, prefix='train'
                )
                train_metrics_acc.clear()
 
            # --- Phase 4: Eval ---
            if step % eval_every < self.num_envs:
                eval_n = cfg.get('eval_episodes', 10)
                eval_metrics = self._eval_episodes(eval_n)
                self.logger.log_print(
                    eval_metrics, self._global_env_step, prefix='eval'
                )
 
            # --- Phase 5: Checkpoint ---
            if step % checkpoint_every < self.num_envs:
                self._save_checkpoint(tag=self._global_env_step)
 
        print(f'[Train] Done. Total env steps: {self._global_env_step}, '
              f'grad steps: {grad_steps}')
 
    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        _map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        if name not in _map:
            raise ValueError(
                f'Unknown compute_dtype: {name!r}. '
                f'Choose from {list(_map.keys())}'
            )
        return _map[name]
 
    @staticmethod
    def _metric_to_float(v) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            return v.item()
        return None