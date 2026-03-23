import time
 
import numpy as np
import torch
 
from shared.trainer_base import TrainerBase

class InterleavedTrainer(TrainerBase):
    """
    Interleaved collect-train loop:
 
        for step in total_env_steps:
            obs, state, act = collect_step(obs, state, act)
            if buffer.ready:
                batch = buffer.sample(B, T)
                metrics = agent.train_step(batch)
    """
 
    def _main_loop(self):
        config = self.config
        total_steps = config.total_env_steps
        seed_steps = config.get('seed_steps', 1024)
        batch_size = config.batch_size
        seq_len = config.seq_len
        train_ratio = config.get('train_ratio', 1)
        log_every = config.get('log_every', 1000)
        eval_every = config.get('eval_every', 10000)
        checkpoint_every = config.get('checkpoint_every', 25000)
 
        # --- Init collection state ---
        obs_list, _ = self.vec_env.reset()
        agent_state = self.agent.initial_state(self.num_envs, self.device)
        prev_action = self.agent.initial_prevact(self.num_envs, self.device)
 
        # --- Tracking ---
        train_metrics_acc: dict[str, list[float]] = {}
        ep_returns: list[float] = []
        ep_return_running = [0.0] * self.num_envs
        t_start = time.time()
 
        print(f'[Train] Starting interleaved loop: {total_steps} env steps')
 
        step = 0
        tokens_per_grad_step = batch_size * seq_len
        train_credit = 0.0
        while step < total_steps:
            #  --- Phase 1: Collect ---
            obs_list, agent_state, prev_action, _ = self.collect_step(
                obs_list, agent_state, prev_action,
            )
 
            # 追蹤 episode returns
            for i in range(self.num_envs):
                rew = float(obs_list[i].get('reward', 0.0))
                ep_return_running[i] += rew
                if obs_list[i].get('is_first', False):
                    # 新 episode 開始 -> 上一個 episode 結束
                    if ep_return_running[i] != 0.0 or step > 0:
                        ep_returns.append(ep_return_running[i])
                    ep_return_running[i] = 0.0
 
            step += self.num_envs
 
            #  --- Phase 2: Train ---
            if self.buffer.total_steps >= seed_steps:
                train_credit += train_ratio * self.num_envs
                while train_credit >= tokens_per_grad_step:
                    batch = self.buffer.sample(batch_size, seq_len)
                    metrics = self.agent.train_step(batch)
                    train_credit -= tokens_per_grad_step
 
                    # 累計 metrics 做 smoothing
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            train_metrics_acc.setdefault(k, []).append(float(v))
                        elif isinstance(v, torch.Tensor) and v.numel() == 1:
                            train_metrics_acc.setdefault(k, []).append(v.item())
 
            #  --- Phase 3: Log ---
            if step % log_every < self.num_envs and train_metrics_acc:
                averaged = {
                    k: sum(v) / len(v) for k, v in train_metrics_acc.items()
                }
                elapsed = time.time() - t_start
                averaged['fps'] = step / max(elapsed, 1e-6)
                averaged['buffer_steps'] = self.buffer.total_steps
 
                if ep_returns:
                    averaged['train/return_mean'] = float(np.mean(ep_returns[-20:]))
 
                self.logger.log_print(averaged, self._global_env_step, prefix='train')
                train_metrics_acc.clear()
 
            #  --- Phase 4: Eval ---
            if step % eval_every < self.num_envs:
                eval_n = config.get('eval_episodes', 10)
                eval_metrics = self._eval_episodes(eval_n)
                self.logger.log_print(eval_metrics, self._global_env_step, prefix='eval')
 
            #  --- Phase 5: Checkpoint ---
            if step % checkpoint_every < self.num_envs:
                self._save_checkpoint(tag=self._global_env_step)
 
        print(f'[Train] Done. Total env steps: {self._global_env_step}')
