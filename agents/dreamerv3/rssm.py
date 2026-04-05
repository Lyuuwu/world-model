from typing import Callable

import torch
import torch.nn as nn

from shared.registry import register
from shared.networks.mlp import NormedLinear, LinearHead
from shared.networks.gru import NormedBlockGRUCell
from shared.distributions import StraightThroughCategorical

@register('rssm', 'dreamerv3')
class RSSM(nn.Module):
    '''
    DreamerV3 Recurrent State-Space Model
    '''

    def __init__(
        self,
                 
        # --- action space ---
        action_dim: int,
        
        # --- 維度 ---
        h_dim: int=4096,           # deterministic state dim (= 8 * model_dim)
        hidden: int=2048,          # MLP hidden units
        stoch: int=32,             # category
        classes: int=32,           # class
        
        # --- Block GRU ---
        seq_type: str = 'block_gru',    # tmp: 暫時用不到
        blocks: int=8,
        dyn_layers: int=1,
        
        # --- prior / posterior MLP ---
        prior_layers: int=2,
        post_layers: int=1,
        token_dim: int=1024,

        # --- network config ---
        norm: str='rms',
        act: str='silu',

        # --- distribution ---
        unimix: float=0.01,

        # --- init ---
        outscale: float=1.0,
        
        # --- compile ---
        use_compile: bool=False,
        compile_mode: str = 'reduce-overhead'
    ):
        super().__init__()
        
        self.h_dim = h_dim
        self.hidden = hidden
        self.stoch = stoch
        self.classes = classes
        self.blocks = blocks
        self.unimix = unimix
        
        self.feat_dim = h_dim + stoch * classes   # downstream heads 用的維度
        
        # --- Block GRU core ---
        # 三條 input branch：deter, stoch, action 各自先投影到 hidden
        # 串起來扔GRU
        self.dynin0, self.dynin1, self.dynin2, \
        self.cell = self._build_core(action_dim, h_dim, hidden, blocks, dyn_layers, norm, act)
        
        # --- Prior network: h -> logits ---
        self.prior = self._build_prior(h_dim, hidden, stoch, classes, prior_layers, norm, act, outscale)
        
        # --- Posterior network ---
        self.posterior = self._build_posterior(h_dim, hidden, stoch, classes, post_layers, token_dim, norm, act, outscale)
        
        # --- compile ---
        self.use_compile = use_compile
        if use_compile:
            self._compiled_observe = torch.compile(self._observe_fused, mode=compile_mode, fullgraph=True)
            self._compiled_imagine = torch.compile(self._imagine_fused, mode=compile_mode, fullgraph=True)
    
    def _build_core(self, action_dim, h_dim, hidden, blocks, dyn_layers, norm, act):
        '''
        三條 input branch + NormedBlockGRUCell
        '''

        dynin0 = NormedLinear(h_dim, hidden, norm, act)
        dynin1 = NormedLinear(self.stoch * self.classes, hidden, norm, act)
        dynin2 = NormedLinear(action_dim, hidden, norm, act)
        
        cell = NormedBlockGRUCell(
            input_dim=hidden * 3,       # 三條 branch concat 後的維度
            hidden_dim=h_dim,
            blocks=blocks,
            hidden_layers=dyn_layers,
            norm=norm,
            act=act,
        )
        
        return dynin0, dynin1, dynin2, cell

    def _build_prior(self, h_dim, hidden, stoch, classes, layers, norm, act, outscale):
        '''
        Prior MLP: deter -> stoch logits
        
        結構:
        1. layers 個 NormedLinear(hidden -> hidden)
        2. Linear(hidden -> stoch * classes)，用 outscale init
        '''
        
        prior = nn.Sequential(*[
            NormedLinear(h_dim if i == 0 else hidden, hidden, norm, act)
            for i in range(layers)
        ], LinearHead(hidden, stoch * classes, outscale))
        
        return prior
    
    def _build_posterior(self, h_dim, hidden, stoch, classes, layers, token_dim, norm, act, outscale):
        '''
        Posterior MLP: concat(deter, tokens) -> stoch logits
        '''
        posterior = nn.Sequential(*[
            NormedLinear((h_dim + token_dim) if i == 0 else hidden, hidden, norm, act)
            for i in range(layers)
        ], LinearHead(hidden, stoch * classes, outscale))
        
        return posterior
    
    @property
    def state_keys(self) -> tuple[str, ...]:
        return ('deter', 'stoch')
    
    def initial_state(self, batch_size: int, device: torch.device = 'cpu') -> dict[str, torch.Tensor]:
        '''回傳全零初始 state dict'''
        return {
            'deter': torch.zeros(batch_size, self.h_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch, self.classes, device=device),
        }
    
    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        state -> feature vector for downstream heads (decoder, reward, continue)
        
        return: (..., feat_dim) where feat_dim = h_dim + stoch * classes
        '''
        stoch = state['stoch'].flatten(-2)
        return torch.cat([state['deter'], stoch], dim=-1)

    def _core(self,
              deter: torch.Tensor,     # (B, h_dim)
              stoch: torch.Tensor,     # (B, stoch, classes)
              action: torch.Tensor,    # (B, action_dim)
              ) -> torch.Tensor:       # (B, h_dim)
        '''
        Block GRU deterministic state transition

        三條 input branch -> concat -> NormedBlockGRUCell
        '''
        action = action / torch.maximum(torch.ones_like(action), torch.abs(action)).detach()
        
        x0 = self.dynin0(deter)                # (B, hidden)
        x1 = self.dynin1(stoch.flatten(-2))    # (B, hidden)
        x2 = self.dynin2(action)               # (B, hidden)
        x  = torch.cat([x0, x1, x2], dim=-1)   # (B, 3 * hidden)
        
        # Block expansion + hidden layers + gate 全部委託給 cell
        return self.cell(x, deter)
    
    def _prior_logits(self, deter: torch.Tensor) -> torch.Tensor:
        '''
        deter -> prior logits

        return: (..., stoch, classes)
        '''
        x = self.prior(deter)
        return x.reshape(x.shape[:-1] + (self.stoch, self.classes))
    
    def _posterior_logits(self, deter: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        '''
        (deter, token) -> posterior logits
        
        return: (..., stoch, classes)
        '''
        
        x = torch.cat([deter, token], dim=-1)
        x = self.posterior(x)
        return x.reshape(x.shape[:-1] + (self.stoch, self.classes))
    
    def _observe_fused(self, deter, stoch, action, token):
        h = self._core(deter, stoch, action)
        x = torch.cat([h, token], dim=-1)
        logit = self.posterior(x)
        logit = logit.reshape(logit.shape[:-1] + (self.stoch, self.classes))
        return h, logit
    
    def _imagine_fused(self, deter, stoch, action):
        h = self._core(deter, stoch, action)
        logit = self.prior(h)
        logit = logit.reshape(logit.shape[:-1] + (self.stoch, self.classes))
        return h, logit
    
    def _make_dist(self, logits: torch.Tensor) -> StraightThroughCategorical:
        return StraightThroughCategorical(logits, unimix_ratio=self.unimix)
    
    def observe_step(
        self,
        state: dict[str, torch.Tensor],
        tokens: torch.Tensor,
        action: torch.Tensor,
        reset: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        '''
        單步 posterior transition

        in:
            - state: {deter, stoch}
            - tokens: (B, token_dim)
            - action: (B, action_dim)
            - reset: (B, )

        return:
            - next_state: {deter, stoch}
            - output: {deter: (B, h_dim), stoch: (B, stoch, classes), logits: (B, stoch, classes)}
        '''
        
        mask = (~reset).float().unsqueeze(-1)   # (B, 1)
        
        deter = state['deter'] * mask
        stoch = state['stoch'] * mask.unsqueeze(-1)
        action = action * mask
        
        observe = self._compiled_observe if self.use_compile else self._observe_fused
        h, logit = observe(deter, stoch, action, tokens)
        
        z = self._make_dist(logit).sample()
        
        next_state = {'deter': h, 'stoch': z}
        output = {'deter': h, 'stoch': z, 'logit': logit}
        
        return (next_state, output)

    def observe(
        self,
        tokens: torch.Tensor,      # (B, T, token_dim)
        action: torch.Tensor,      # (B, T, action_dim)
        reset: torch.Tensor,       # (B, T) bool
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        '''
        整段序列的 posterior rollout

        in:
            - tokens: (B, T, token_dim)
            - action: (B, T, action_dim)
            - reset:  (B, T)

        return:
            - final_state: {deter, stoch}
            - outputs: {deter: (B, T, h_dim), stoch: (B, T, stoch, classes), logit: (B, T, stoch, classes)}
        '''
        
        if state is None:
            state = self.initial_state(tokens.shape[0], tokens.device)
        
        outputs = []
        for t in range(tokens.shape[1]):
            state, out_t = self.observe_step(state, tokens[:, t], action[:, t], reset[:, t])
            outputs.append(out_t)
            
        outputs = {k: torch.stack([o[k] for o in outputs], dim=1) for k in outputs[0]}
        
        return (state, outputs)
    
    def imagine_step(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        '''
        單步 prior transition

        in:
            - state: {deter, stoch}
            - action: (B, action_dim)

        return:
            - next_state: {deter, stoch}
            - feat: {deter: (B, h_dim), stoch: (B, stoch, classes), logit: (B, stoch, classes)}
        '''
        
        imagine = self._compiled_imagine if self.use_compile else self._imagine_fused
        h, logit = imagine(state['deter'], state['stoch'], action)
        
        z = self._make_dist(logit).sample()
        
        next_state = {'deter': h, 'stoch': z}
        feat = {'deter': h, 'stoch': z, 'logit': logit}
        
        return (next_state, feat)
    
    def imagine(
        self,
        state: dict[str, torch.Tensor],                                 # 起始 state (B, ...)
        policy: Callable[[torch.Tensor], torch.Tensor] | None = None,   # feat -> action
        horizon: int = 15,
        action_seq: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        '''
        多步 imagination rollout

        in:
            - state: {deter, stoch}

        return:
            - feats = {deter: (B, H, ...), stoch: (B, H, ...), logit: (B, H, ...)}
            - actions = (B, H, action_dim)
        '''
        
        assert (policy is None) != (action_seq is None), \
            'Exactly one of policy_fn or action_sequence must be provided'
        
        if action_seq is not None:
            horizon = action_seq.shape[1]
        
        feats = []
        actions = []
        for t in range(horizon):
            if policy is not None:
                feat = self.get_feat(state)
                action = policy(feat)
            else:
                action = action_seq[:, t]
            
            state, feat_t = self.imagine_step(state, action)

            feats.append(feat_t)
            actions.append(action)
        
        feats = {k: torch.stack([o[k] for o in feats], dim=1) for k in feats[0]}
        actions = torch.stack(actions, dim=1)
        
        return (feats, actions)

    def kl_loss(
        self,
        outputs: dict[str, torch.Tensor],   # observe() 的 outputs
        free_nats: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        '''
        計算 dynamics loss 和 representation loss
        '''
        
        posterior_logits = outputs['logit']
        prior_logits = self._prior_logits(outputs['deter'])
        
        post_dist = self._make_dist(posterior_logits)
        prior_dist = self._make_dist(prior_logits)
        
        # KL shape: (B, T, stoch) > .sum(-1) > (B, T)
        dyn_loss = self._make_dist(posterior_logits.detach()).kl(prior_dist).sum(-1)  
        rep_loss = post_dist.kl(self._make_dist(prior_logits.detach())).sum(-1)
        
        dyn_loss = dyn_loss.clamp_min(free_nats)
        rep_loss = rep_loss.clamp_min(free_nats)
        
        metrics = {
            'prior_entropy': prior_dist.entropy().mean(),
            'posterior_entropy': post_dist.entropy().mean()
        }
        
        return (dyn_loss, rep_loss, metrics)
