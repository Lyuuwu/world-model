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
        self._phase_files: dict[str, Any] = {}
        self._start_time = time.time()
        
    def log(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str=''
    ) -> None:
        wall_time = round(time.time() - self._start_time, 2)
        phase = prefix or 'metrics'
        record: dict[str, Any] = {
            'phase': phase,
            'env_step': step,
            'step': step,
            'wall_time': wall_time,
        }
        phase_record: dict[str, Any] = dict(record)
        metrics = {**metrics, **self._system_metrics()}
        
        for k, v in metrics.items():
            tag = f'{prefix}/{k}' if prefix else k
            scalar = _to_scalar(v)
            if scalar is not None:
                record[tag] = scalar
                phase_record[k] = scalar
        if len(record) > 5:
            self._write_jsonl(self._jsonl_file, record)
            if prefix in ('train', 'eval', 'final_eval'):
                file_key = 'eval' if prefix in ('eval', 'final_eval') else prefix
                self._write_jsonl(self._get_phase_file(file_key), phase_record)
    
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
        print('=' * 60)
        print(' | '.join(parts))
        print('Max VRAM allocate:', torch.cuda.max_memory_allocated() / 1024**3, 'GB')
        
    def save_config(self, config_dict: dict) -> None:
        path = self._log_dir / 'config.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def save_run_metadata(self, metadata: dict[str, Any]) -> None:
        path = self._log_dir / 'run.json'
        record = {
            'created_wall_time': self._start_time,
            'log_dir': str(self._log_dir),
            **metadata,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, default=str)
            
    def close(self) -> None:
        self._jsonl_file.close()
        for f in self._phase_files.values():
            f.close()
    
    @property
    def log_dir(self) -> Path:
        return self._log_dir
    
    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    def _get_phase_file(self, phase: str):
        if phase not in self._phase_files:
            path = self._log_dir / f'{phase}.jsonl'
            self._phase_files[phase] = open(
                path, 'a', buffering=1, encoding='utf-8'
            )
        return self._phase_files[phase]

    @staticmethod
    def _write_jsonl(file, record: dict[str, Any]) -> None:
        file.write(json.dumps(record, sort_keys=True) + '\n')

    @staticmethod
    def _system_metrics() -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            'system/cuda_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'system/cuda_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'system/cuda_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }
    
def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

def discover_metric_files(paths: list[str | Path] | str | Path) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    found: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = Path(path)
        candidates = []
        if path.is_dir():
            direct = path / 'metrics.jsonl'
            candidates = [direct] if direct.exists() else sorted(path.rglob('metrics.jsonl'))
        elif path.exists():
            candidates = [path]

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                found.append(candidate)
                seen.add(resolved)
    return found

def run_label_for_jsonl(path: str | Path) -> str:
    path = Path(path)
    metadata_path = path.parent / 'run.json'
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding='utf-8') as f:
                metadata = json.load(f)
            if metadata.get('plot_label'):
                return str(metadata['plot_label'])
            seq = metadata.get('sequence_model_label') or metadata.get('sequence_model')
            profile = metadata.get('profile')
            seed = metadata.get('seed')
            parts = [str(x) for x in (seq, profile, f's{seed}' if seed is not None else None) if x]
            if parts:
                return '/'.join(parts)
            if metadata.get('run_name'):
                return str(metadata['run_name'])
        except (OSError, json.JSONDecodeError):
            pass
    return path.parent.name
    
def plot_from_jsonl(
    paths: list[str | Path] | str | Path,
    keys: list[str] | None = None,
    smooth: int = 10,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 4)
) -> None:
    import matplotlib.pyplot as plt
    
    metric_paths = discover_metric_files(paths)

    all_runs: list[tuple[str, list[dict]]] = []
    for p in metric_paths:
        label = run_label_for_jsonl(p)
        records = load_jsonl(p)
        if records:
            all_runs.append((label, records))
            
    if not all_runs:
        print('No data found.')
        return
    
    if keys is None:
        all_keys = set()
        for _, records in all_runs:
            for r in records:
                all_keys.update(
                    k for k, v in r.items()
                    if k not in ('phase', 'step', 'env_step', 'wall_time')
                    and _is_number(v)
                )
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
                if key in r and 'step' in r and _is_number(r[key]):
                    steps.append(r['step'])
                    vals.append(r[key])
            if not vals:
                continue
            steps, vals = np.array(steps), np.array(vals)
            if smooth > 1 and len(vals) >= smooth:
                kernel = np.ones(smooth) / smooth
                vals_smooth = np.convolve(vals, kernel, mode='valid')
                steps_smooth = steps[smooth - 1:]
                ax.plot(steps_smooth, vals_smooth, label=label)
            else:
                ax.plot(steps, vals, label=label)
                
        ax.set_title(key, fontsize=9)
        ax.set_xlabel('env steps', fontsize=8)
        if ax.get_lines():
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

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
