from .atari import AtariEnvConfig, Atari_override

_ENV_CONFIGS = {
    'atari': (AtariEnvConfig, Atari_override),
    'atari100k': (AtariEnvConfig, Atari_override),
}

def get_env_config(domain: str):
    if domain not in _ENV_CONFIGS:
        raise ValueError(f'Unknown domain: {domain}. Available: {list(_ENV_CONFIGS.keys())}')
    
    cfg_cls, override = _ENV_CONFIGS[domain]
    cfg = cfg_cls()
    if domain == 'atari100k':
        cfg.max_noop = 30
        cfg.actions = 'needed'
        cfg.grayscale = False
        cfg.resize = [64, 64]
        cfg.resize_method = 'pillow'
    elif domain == 'atari':
        cfg.max_noop = 30
        cfg.actions = 'all'
        cfg.sticky_prob = 0.25
        cfg.resize = [96, 96]
        cfg.resize_method = 'pillow'
        cfg.grayscale = True
    
    return cfg, override(domain)
