from .config import DreamerConfig
from .encoder import DreamerEncoder
from .decoder import DreamerDecoder
from .rssm import RSSM
from .world_model import RewardHead, ContinueHead, DreamerWorldModel
from .actor_critic import DreamerActorCritic
from .agent import DreamerAgent

def build(obs_space: dict, action_dim: int, cfg: DreamerConfig):
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    coder_space = {k: v for k, v in obs_space.items() if k not in exclude}
    image_spaces = [v for v in coder_space.values() if v.is_image]
    img_size = tuple(image_spaces[0].shape[-2:]) if image_spaces else (64, 64)
    
    encoder = DreamerEncoder(
        obs_space = coder_space,
        img_size = img_size,
        depth = cfg.wm.depth,
        units = cfg.wm.units,
        layers = cfg.wm.layers,
        **cfg.wm.enc.to_dict()
    )
    
    decoder = DreamerDecoder(
        obs_space = coder_space,
        img_size = img_size,
        h_dim = cfg.wm.h_dim,
        stoch = cfg.wm.stoch,
        classes = cfg.wm.classes,
        depth = cfg.wm.depth,
        units = cfg.wm.units,
        layers = cfg.wm.layers,
        **cfg.wm.dec.to_dict()
    )
    
    rssm = RSSM(
        action_dim = action_dim,
        h_dim = cfg.wm.h_dim,
        hidden = cfg.wm.hidden,
        stoch = cfg.wm.stoch,
        classes = cfg.wm.classes,
        token_dim = encoder.token_dim,
        **cfg.wm.rssm.to_dict()
    )
    
    feat_dim = rssm.feat_dim
    rewhead = RewardHead(
        feat_dim = feat_dim,
        units = cfg.wm.units,
        layers = cfg.wm.head_layers,
        bins = cfg.wm.bins,
        **cfg.wm.rewhead.to_dict()
    )
    
    conthead = ContinueHead(
        feat_dim = feat_dim,
        units = cfg.wm.units,
        layers = cfg.wm.head_layers,
        **cfg.wm.conthead.to_dict()
    )
    
    world_model = DreamerWorldModel(
        encoder = encoder,
        decoder = decoder,
        rssm = rssm,
        reward_head = rewhead,
        continue_head = conthead,
        
        free_nats = cfg.wm.free_nats,
        reward_grad = cfg.wm.reward_grad,
        
        contdisc = cfg.contdisc,
        horizon = cfg.horizon
    )
    
    actor_critic = DreamerActorCritic(
        feat_dim = feat_dim,
        action_dim = action_dim,
        contdisc = cfg.contdisc,
        discrete = cfg.discrete,
        **cfg.ac.to_dict()
    )
    
    return DreamerAgent(
        obs_space = obs_space,
        action_dim = action_dim,
        world_model = world_model,
        actor_critic = actor_critic,
        config=cfg
    )
