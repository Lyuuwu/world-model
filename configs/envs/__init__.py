from .atari import AtariEnvConfig, Atari_override

_ENV_CONFIGS = {
    'atari': (AtariEnvConfig, Atari_override),
    'atari100k': (AtariEnvConfig, Atari_override),
}

def get_env_config(domain: str):
    if domain not in _ENV_CONFIGS:
        raise ValueError(f'Unknown domain: {domain}. Available: {list(_ENV_CONFIGS.keys())}')
    
    cfg_cls, override = _ENV_CONFIGS[domain]
    
    return cfg_cls(), override(domain)