import copy
import yaml
from pathlib import Path
from typing import Any

from . import tool

def deep_update(base: dict, override: dict) -> dict:
    ''' 遞迴合併 override 到 base '''
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _parse_value(s: str) -> Any:
    ''' 把 CLI override string 轉成 Python type '''
    if s.lower() in ('true', 'yes'):
        return True
    if s.lower() in ('false', 'no', 'null', 'none'):
        if s.lower() in ('null', 'none'):
            return None
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # list: "[1,2,3]"
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(x.strip()) for x in inner.split(',')]
    return s

def parse_overrides(override_str: str | None) -> dict:
    '''
    解析 CLI override string: "lr=3e-4,batch_size=32,wm_kwargs.h_dim=2048"
    dot-separated keys -> nested dict
    '''
    if not override_str:
        return {}
    result = {}
    for pair in override_str.split(','):
        pair = pair.strip()
        if not pair or '=' not in pair:
            continue
        key, val = pair.split('=', 1)
        keys = key.strip().split('.')
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = _parse_value(val.strip())
    return result

# def _load_config(path: Path, module_name: str, mode: str) -> dict:
#     if not path.exists():
#         raise FileNotFoundError(f'config file does not exist: {path}')
    
#     module = tool.import_module(module_name, path)
    
#     try:
#         func = getattr(module, mode)
#     except AttributeError:
#         raise ValueError(f'module: "{path.name}" does not include: "{mode}"')
    
#     if not callable(func):
#         raise TypeError(f'"{mode}" is in the module, but not callable')
    
#     cfg = func()
#     if not hasattr(cfg, 'to_dict'):
#         raise ValueError(f'Config does not has attribute: to_dict()')
#     if not callable(cfg.to_dict):
#         raise TypeError(f'Config to_dict is not callable')
    
#     return cfg.to_dict()

def _load_config(module, mode):
    func = getattr(module, mode)
    cfg = func()
    return cfg

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data or {}

def compose_config(
    agent: str,
    task: str,
    override_str: str | None = None,
    project_root: Path | None = None,
    profile: str | None = None
) -> dict:
    '''
    組合 config
 
    task 格式: "atari_pong" -> domain="atari", game="pong"
    '''
    if project_root is None:
        # 假設從 scripts/ 執行，project root 是上一層
        project_root = Path(__file__).resolve().parent.parent
 
    config = {}
    
    # --- agent ---
    agent_dir = project_root / 'agents' / agent
    profile_path = agent_dir / 'profiles.py'
    if not profile_path.exists():
        raise FileNotFoundError(f'profile file does not exist: {profile_path}')
    
    module_name = f'agents.{agent}.profiles'
    profile_module = tool.import_module(module_name)
    profile_mode = profile if profile else 'default'
    agent_cfg = _load_config(profile_module, profile_mode)
    config['agent_config'] = agent_cfg
 
    # --- task domain ---
    domain = task.split('_', 1)[0]  # "atari_pong" -> "atari"
    deep_update(config, _load_yaml(project_root / 'configs' / f'{domain}.yaml'))
 
    # --- CLI overrides ---
    deep_update(config, parse_overrides(override_str))
 
    # --- 注入 meta fields ---
    config['agent'] = agent
    config['task'] = task
 
    return config

class Config:
    '''
    dict wrapper: 支援 config.lr 和 config['lr'] 兩種 access
    
    方便 agent code 讀取
    '''
 
    def __init__(self, d: dict):
        self._data = d
 
    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            return super().__getattribute__(key)
        try:
            val = self._data[key]
        except KeyError:
            raise AttributeError(f'Config has no key: {key}')
        if isinstance(val, dict):
            return Config(val)
        return val
 
    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        if isinstance(val, dict):
            return Config(val)
        return val
 
    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return Config(val)
        return val
 
    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)
 
    def __contains__(self, key: str) -> bool:
        return key in self._data
 
    def __repr__(self) -> str:
        return f'Config({self._data})'