from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import NormedLinear, get_norm

try:
    import mamba_ssm.modules.mamba2 as mamba2_module
    Mamba2 = mamba2_module.Mamba2
except ImportError:  # pragma: no cover - depends on the training environment.
    try:
        import mamba_ssm.modules.mamba2 as mamba2_module
        Mamba2 = mamba2_module.Mamba2
    except ImportError:
        mamba2_module = None
        Mamba2 = None


class Mamba2StepCore(nn.Module):
    '''
    Single-step adapter around mamba_ssm.Mamba2.

    Mamba2 keeps recurrent information in convolution and SSM caches, so the
    caller must store those tensors in the RSSM state alongside deter/stoch.
    '''

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mamba_dim: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rmsnorm: bool = False,
        use_triton_step: bool = False,
        norm: str = 'rms',
        act: str = 'silu',
        **kwargs,
    ):
        super().__init__()
        if Mamba2 is None:
            raise ImportError(
                'mamba_ssm is required for seq_type="mamba2". '
                'Install it in the Python environment used to run training.'
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mamba_dim = mamba_dim
        self.use_triton_step = use_triton_step

        if not use_triton_step:
            if ngroups != 1:
                raise ValueError('Mamba2 torch step fallback requires ngroups=1')
            mamba2_module.selective_state_update = None

        self.in_proj = NormedLinear(input_dim, mamba_dim, norm, act)
        self.mamba = Mamba2(
            d_model=mamba_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            rmsnorm=rmsnorm,
            **kwargs,
        )
        self.out_proj = nn.Linear(mamba_dim, output_dim)
        self.out_norm = get_norm(norm, output_dim)

    def initial_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        if dtype is None:
            dtype = next(self.parameters()).dtype

        conv_state, ssm_state = self.mamba.allocate_inference_cache(
            batch_size,
            max_seqlen=1,
            dtype=dtype,
        )
        return {
            'mamba_conv': conv_state.to(device=device, dtype=dtype),
            'mamba_ssm': ssm_state.to(device=device, dtype=dtype),
        }

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        x = self.in_proj(x).unsqueeze(1)
        conv_state = conv_state.to(device=x.device, dtype=x.dtype)
        ssm_state = ssm_state.to(device=x.device, dtype=x.dtype)
        if not self.use_triton_step:
            y, conv_state, ssm_state = self._torch_step(x, conv_state, ssm_state)
            y = self.out_proj(y.squeeze(1))
            y = self.out_norm(y).to(dtype)
            return y, conv_state, ssm_state

        y, conv_state, ssm_state = self.mamba.step(x, conv_state, ssm_state)
        y = self.out_proj(y.squeeze(1))
        y = self.out_norm(y).to(dtype)
        return y, conv_state, ssm_state

    def _torch_step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mamba = self.mamba
        dtype = hidden_states.dtype
        zxbcdt = mamba.in_proj(hidden_states.squeeze(1))
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * mamba.d_ssm
            - 2 * mamba.ngroups * mamba.d_state
            - mamba.nheads
        ) // 2
        z0, x0, z, xbc, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                mamba.d_ssm,
                mamba.d_ssm + 2 * mamba.ngroups * mamba.d_state,
                mamba.nheads,
            ],
            dim=-1,
        )

        conv_state = torch.cat([conv_state[:, :, 1:], xbc.unsqueeze(-1)], dim=-1)
        conv_weight = mamba.conv1d.weight.squeeze(1)
        xbc = torch.sum(conv_state * conv_weight.unsqueeze(0), dim=-1)
        if mamba.conv1d.bias is not None:
            xbc = xbc + mamba.conv1d.bias
        xbc = mamba.act(xbc).to(dtype=dtype)

        x, b, c = torch.split(
            xbc,
            [mamba.d_ssm, mamba.ngroups * mamba.d_state, mamba.ngroups * mamba.d_state],
            dim=-1,
        )
        if mamba.ngroups != 1:
            raise ValueError('Mamba2 torch step fallback requires ngroups=1')

        a = -torch.exp(mamba.A_log.float())
        dt = F.softplus(dt + mamba.dt_bias.to(dtype=dt.dtype))
        da = torch.exp(dt * a)
        x = x.reshape(x.shape[0], mamba.nheads, mamba.headdim)
        dbx = torch.einsum('bh,bn,bhp->bhpn', dt, b, x)
        ssm_state = ssm_state * da[:, :, None, None] + dbx

        y = torch.einsum('bhpn,bn->bhp', ssm_state.to(dtype), c)
        y = y + mamba.D.to(dtype).reshape(1, mamba.nheads, 1) * x
        y = y.reshape(y.shape[0], -1)
        y = y * mamba.act(z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        y = mamba.out_proj(y)
        return y.unsqueeze(1), conv_state, ssm_state.to(dtype=dtype)
