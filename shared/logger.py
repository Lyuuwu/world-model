import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

class JSONLLogger:
    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        self._jsonl_path = self._log_dir / 'metrics.jsonl'
        self._jsonl_file = open(
            self._jsonl_path, 'a', buffering=1, encoding='utf-8'
        )
        self._start_time = time.time()
        
    def log(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str=''
    ) -> None:
        record: dict[str, Any] = {
            'step': step,
            'wall_time': round(time.time() - self._start_time, 2)
        }
        
        for k, v in metrics.items():
            tag = f'{prefix}/{k}' if prefix else k
            scalar = _to_scalar(v)
            if scalar is not None:
                record[tag] = scalar
        if len(record) > 2:
            self._jsonl_file.write(json.dumps(record) + '\n')
    
    def log_print(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str=''
    ) -> None:
        self.log(metrics, step, prefix)
        parts = [f'{prefix} step={step}'] if prefix else [f'step={step}']
        
        for k, v in metrics.items():
            s = _to_scalar(v)
            if s is not None:
                parts.append(f'{k}={s:.4f}' if isinstance(s, float) else f'{k}={s}')
        print(' | '.join(parts))
        
    def save_config(self, config_dict: dict) -> None:
        path = self._log_dir / 'config.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
            
    def close(self) -> None:
        self._jsonl_file.close()
    
    @property
    def log_dir(self) -> Path:
        return self._log_dir
    
    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path
    
def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
    
def plot_from_jsonl(
    paths: list[str | Path] | str | Path,
    keys: list[str] | None = None,
    smooth: int = 10,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 4)
) -> None:
    import matplotlib as plt
    
    if isinstance(paths, (str, Path)):
        paths = [paths]
    
    all_runs = list[tuple[str, list[dict]]] = []
    for p in paths:
        p = Path(p)
        label = p.parent.name
        records = load_jsonl(p)
        if records:
            all_runs.append(label, records)
            
    if not all_runs:
        print('No data found.')
        return
    
    if keys is None:
        all_keys = set()
        for _, records in all_runs:
            for r in records:
                all_keys.update(k for k in r if k not in ('step', 'wall_time'))
        keys = sorted(all_keys)
        
    n_keys = len(keys)
    if n_keys == 0:
        print('No metrics keys found.')
        return
    
    cols = min(n_keys, 4)
    rows = (n_keys + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows))
    if n_keys == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
    for idx, key in enumerate(keys):
        ax = axes[idx]
        for label, records in all_runs:
            steps, vals = [], []
            for r in records:
                if key in r and 'step' in r:
                    steps.append(r['step'])
                    vals.append(r[key])
            if not vals:
                continue
            steps, vals = np.arange(steps), np.array(vals)
            if smooth > 1 and len(vals) / smooth:
                kernel = np.ones(smooth) / smooth
                vals_smooth = np.convolve(vals, kernel, mode='valid')
                steps_smooth = steps[smooth - 1:]
                ax.plot(steps_smooth, vals_smooth, label=label)
            else:
                ax.plot(steps, vals, label=label)
                
        ax.set_title(key, fontsize=9)
        ax.set_xlabel('env steps', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        
    for idx in range(n_keys, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    else:
        plt.show()
    
    plt.close(fig)            
            
def _to_scalar(v: Any) -> float | int | None:
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.item()
    if isinstance(v, np.ndarray) and v.size == 1:
        return float(v.flat[0])
    return None