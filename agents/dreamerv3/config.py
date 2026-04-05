from dataclasses import dataclass, field

from shared.base import BaseConfig
from configs.GlobalConfig import GlobalConfig

@dataclass
class EncConfig(BaseConfig):
    mults: list = field(default_factory=lambda: [2, 3, 4, 4])
    kernel: int = 5
    downsample: str = 'maxpool'
    
    apply_symlog: bool = True
    
    act: str = 'silu'
    norm: str = 'rms'

@dataclass
class DecConfig(BaseConfig):
    mults: list = field(default_factory=lambda: [2, 3, 4, 4])
    kernel: int = 5
    upsample: str = 'upsample'
    bspace: int = 8

    apply_symlog: bool = True
    
    act: str = 'silu'
    norm: str = 'rms'
    outscale: float = 1.0

@dataclass
class RSSMConfig(BaseConfig):
    # tmp: 之後可能會修改
    seq_type: str = 'block_gru'
    blocks: int = 8
    dyn_layers: int = 1
    
    prior_layers: int = 2
    post_layers: int = 1
    
    act: str = 'silu'
    norm: str = 'rms'
    outscale: float = 1.0
    
    use_compile: bool = False
    compile_mode: str = 'reduce-overhead'

@dataclass
class RewHeadConfig(BaseConfig):
    act: str = 'silu'
    norm: str = 'rms'
    outscale: float = 0.0
    
@dataclass
class ContHeadConfig(BaseConfig):
    act: str = 'silu'
    norm: str = 'rms'
    outscale: float = 1.0

@dataclass
class WMConfig(BaseConfig):
    # --- world model shared ---
    h_dim: int = 8192
    stoch: int = 32
    classes: int = 64
    blocks: int = 8
    hidden: int = 1024
    units: int = 1024
    layers: int = 3
    
    # --- Encoder&Decoder shared ---
    depth: int = 64
    
    # --- Encoder ---
    enc: EncConfig = field(default_factory=EncConfig)
    
    # --- Decoder ---
    dec: DecConfig = field(default_factory=DecConfig)
    
    # --- RSSM ---
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    
    # --- WorldModel ---
    head_layers: int = 1
    bins: int = 255
    
    # --- RewardHead ---
    rewhead: RewHeadConfig = field(default_factory=RewHeadConfig)
    
    # --- ContinueHead ---
    conthead: ContHeadConfig = field(default_factory=ContHeadConfig)
    
    act: str = 'silu'
    norm: str = 'rms'
    outscale: float = 1.0
    
    free_nats: float = 1.0
    reward_grad: bool = True
    
@dataclass
class ACConfig(BaseConfig):
    # --- Networks ---
    units: int = 1024
    layers: int = 3
    bins: int = 255
    act: str = 'silu'
    norm: str = 'rms'
    
    # --- Policy ---
    policy_outscale: float = 0.01
    policy_unimix: float = 0.01
    actent: float = 3e-4
    
    minstd: float = 0.1
    maxstd: float = 1.0
    
    # --- Value ---
    value_outscale: float = 0.0
    slow_decay: float = 0.98
    slowreg: float = 1.0
    slowtar: bool = False

    # --- Returns ---
    lam: float = 0.95
    
    # --- Normalization ---
    retnorm_decay: float = 0.99
    retnorm_limit: float = 1.0
    retnorm_plow: float = 5.0
    retnorm_phigh: float = 95.0
    valnorm_decay: float = 0.99
    advnorm_decay: float = 0.99
    valnorm_enable: bool = False
    advnorm_enable: bool = False
    
    policy_scale: float = 1.0
    value_scale: float = 1.0

@dataclass
class LossScales(BaseConfig):
    dyn: float = 1.0
    rep: float = 0.1
    rec: float = 1.0
    rew: float = 1.0
    con: float = 1.0
    policy: float = 1.0
    value: float = 1.0
    repval: float = 0.3

@dataclass
class DreamerConfig(GlobalConfig):    
    # --- optimizer ---
    lr: float = 4e-5
    agc: float = 0.3
    eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999
    
    # --- env ---
    discrete: bool = True
    
    # --- continue target ---
    contdisc: bool = True
    horizon: int = 333
    
    # --- imagination ---
    imag_horizon: int = 15
    imag_last: int | None = None
    
    # --- gradient flow ---
    ac_grads: bool = False
    
    # --- replay value loss ---
    repval_loss: bool = True
    repval_grad: bool = True
    
    wm: WMConfig = field(default_factory=WMConfig)
    ac: ACConfig = field(default_factory=ACConfig)
    loss_scales: LossScales = field(default_factory=LossScales)