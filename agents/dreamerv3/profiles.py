from .config import WMConfig, ACConfig, RSSMConfig, DreamerConfig

def m12m() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(h_dim=2048, classes=16, hidden=256, depth=16, units=256),
        ac = ACConfig(units=256)
    )

def gru_small() -> DreamerConfig:
    return m12m()
    
def m25m() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(h_dim=3072, classes=24, hidden=384, depth=24, units=384),
        ac = ACConfig(units=384)
    )
    
def m50m() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(h_dim=4096, classes=32, hidden=512, depth=32, units=512),
        ac = ACConfig(units=512)
    )
    
def m100m() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(h_dim=6144, classes=48, hidden=768, depth=48, units=768),
        ac = ACConfig(units=768)
    )
    
def m200m() -> DreamerConfig:
    return DreamerConfig()

def mamba_small() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(
            h_dim=2048, classes=16, hidden=256, depth=16, units=256,
            rssm=RSSMConfig(
                seq_type="mamba2",
                mamba_dim=256,
                mamba_d_state=128,
                mamba_d_conv=4,
                mamba_expand=2,
                mamba_headdim=64,
                mamba_ngroups=1,
            )),
        ac = ACConfig(units=256),
    )

def mamba() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(
            h_dim=2048, classes=16, hidden=256, depth=16, units=256,
            rssm=RSSMConfig(
                seq_type="mamba2",
                mamba_dim=576,
                mamba_d_state=128,
                mamba_d_conv=4,
                mamba_expand=2,
                mamba_headdim=64,
                mamba_ngroups=1,
            )),
        ac = ACConfig(units=256),
    )

def default():
    return DreamerConfig()
