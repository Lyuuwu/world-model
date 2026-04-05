from .config import WMConfig, ACConfig, DreamerConfig

def m12m() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(h_dim=1024, classes=16, hidden=256, depth=4),
        ac = ACConfig(units=256)
    )
    
def m25m() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(h_dim=3072, classes=24, hidden=384, depth=24),
        ac = ACConfig(units=384)
    )
    
def m50m() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(h_dim=4096, classes=32, hidden=512, depth=32),
        ac = ACConfig(units=512)
    )
    
def m100m() -> DreamerConfig:
    return DreamerConfig(
        wm = WMConfig(h_dim=6144, classes=768, hidden=1024, depth=64),
        ac = ACConfig(units=768)
    )
    
def m200m() -> DreamerConfig:
    return DreamerConfig()

def default():
    return DreamerConfig()