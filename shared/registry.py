from typing import Any

_REGISTRY: dict[str, dict[str, type]] = {}

def register(cat: str, name: str):
    ''' 把 class 註冊到 _REGISTRY[cat][name] 裡面 '''
    def decorator(cls):
        _REGISTRY.setdefault(cat, {})[name] = cls
        return cls
    return decorator

def build(cat: str, name: str, **kwargs) -> Any:
    ''' 從 _REGISTRY 查表，參數用 kwargs '''
    if cat not in _REGISTRY or name not in _REGISTRY[cat]:
        avail = list(_REGISTRY.get(cat, {}).keys())
        raise ValueError(
            f'[Registry] Unknown {cat}/{name}. Available: {avail}'
        )
    
    return _REGISTRY[cat][name](**kwargs)

def list_registered(cat: str) -> list[str]:
    ''' 列出 cat 底下註冊的名字 '''
    return list(_REGISTRY.get(cat, {}).keys())