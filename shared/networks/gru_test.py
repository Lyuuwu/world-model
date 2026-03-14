"""
GRU 模組訓練測試
驗證各 class 的 forward / backward 是否正常

測試任務：
  1. NormedGRUCell  — 單步，手動展開 T 步，學 sine wave 下一步
  2. GRUSequence    — 直接餵整段序列，同樣學 sine wave
  3. RecurrentStateModel — 模擬 world model 流程：
                           給 (z, action) 序列，預測下一個 z
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import math

from gru import NormedGRUCell, GRUSequence, RecurrentStateModel, get_initial_state


# ── 共用超參數 ───────────────────────────────────────────────
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH      = 32
SEQ_LEN    = 20
STEPS      = 300        # 訓練步數
LR         = 3e-3
HIDDEN_DIM = 64


# ── 資料：多相位的 sine wave ──────────────────────────────────
def make_sine_batch(batch: int, seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    每個 batch item 是不同相位的 sine wave

    return:
        x:      (B, T, 1)   輸入序列（t=0..T-1）
        target: (B, T, 1)   目標序列（t=1..T），即預測下一步
    """
    phase = torch.rand(batch, 1) * 2 * math.pi            # (B, 1) 隨機相位
    t     = torch.arange(seq_len + 1).float().unsqueeze(0) # (1, T+1)
    wave  = torch.sin(t * 0.5 + phase)                    # (B, T+1)
    x      = wave[:, :-1].unsqueeze(-1).to(device)        # (B, T, 1)
    target = wave[:, 1:].unsqueeze(-1).to(device)         # (B, T, 1)
    return x, target


def print_header(title: str):
    print(f'\n{"="*50}')
    print(f'  {title}')
    print(f'{"="*50}')

def should_decrease(losses: list[float], name: str):
    """ 檢查 loss 是否有下降趨勢（最後 50 步均值 < 最初 50 步均值） """
    early = sum(losses[:50]) / 50
    late  = sum(losses[-50:]) / 50
    ok    = late < early * 0.5      # 至少下降一半
    status = '✅ PASS' if ok else '❌ FAIL'
    print(f'  {status}  early_loss={early:.4f}  late_loss={late:.4f}')


# ── Test 1：NormedGRUCell ────────────────────────────────────
def test_normed_gru_cell():
    print_header('Test 1 — NormedGRUCell')

    cell = NormedGRUCell(input_dim=1, hidden_dim=HIDDEN_DIM).to(DEVICE)
    head = nn.Linear(HIDDEN_DIM, 1).to(DEVICE)
    optimizer = optim.Adam(list(cell.parameters()) + list(head.parameters()), lr=LR)

    losses = []
    for step in range(STEPS):
        x, target = make_sine_batch(BATCH, SEQ_LEN, DEVICE)

        # 手動展開：初始 hidden state 全零
        h = torch.zeros(BATCH, HIDDEN_DIM, device=DEVICE)
        preds = []
        for t in range(SEQ_LEN):
            h = cell(x[:, t, :], h)    # (B, HIDDEN_DIM)
            preds.append(head(h))       # (B, 1)

        pred = torch.stack(preds, dim=1)   # (B, T, 1)
        loss = nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(f'  step {step+1:3d}  loss={loss.item():.4f}')

    should_decrease(losses, 'NormedGRUCell')


# ── Test 2：GRUSequence ──────────────────────────────────────
def test_gru_sequence():
    print_header('Test 2 — GRUSequence')

    gru  = GRUSequence(input_dim=1, hidden_dim=HIDDEN_DIM, layers=2).to(DEVICE)
    head = nn.Linear(HIDDEN_DIM, 1).to(DEVICE)
    optimizer = optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=LR)

    losses = []
    for step in range(STEPS):
        x, target = make_sine_batch(BATCH, SEQ_LEN, DEVICE)

        h_seq, _ = gru(x)              # (B, T, HIDDEN_DIM)
        pred = head(h_seq)             # (B, T, 1)
        loss = nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(f'  step {step+1:3d}  loss={loss.item():.4f}')

    should_decrease(losses, 'GRUSequence')


# ── Test 3：RecurrentStateModel ──────────────────────────────
def test_recurrent_state_model():
    """
    模擬簡單的 world model 流程：

        for t in 0..T-1:
            h_t = RSM(h_{t-1}, z_{t-1}, a_{t-1})
            pred_z_t = head(h_t)

    目標：pred_z_t ≈ z_t（下一個 latent state）
    z 用 sine wave 當作 dummy latent，action 隨機
    """
    print_header('Test 3 — RecurrentStateModel')

    Z_DIM      = 16
    ACTION_DIM = 4

    rsm  = RecurrentStateModel(
        h_dim=HIDDEN_DIM, z_dim=Z_DIM, action_dim=ACTION_DIM
    ).to(DEVICE)
    head = nn.Linear(HIDDEN_DIM, Z_DIM).to(DEVICE)
    optimizer = optim.Adam(list(rsm.parameters()) + list(head.parameters()), lr=LR)

    losses = []
    for step in range(STEPS):
        # dummy z sequence：每個維度是不同相位的 sine（當作假的 latent）
        phase  = torch.rand(BATCH, Z_DIM, device=DEVICE) * 2 * math.pi
        t_idx  = torch.arange(SEQ_LEN + 1, device=DEVICE).float()
        z_seq  = torch.sin(t_idx[None, :, None] * 0.5 + phase[:, None, :])  # (B, T+1, Z_DIM)

        actions = torch.randn(BATCH, SEQ_LEN, ACTION_DIM, device=DEVICE)

        # 手動展開序列
        h    = get_initial_state(BATCH, HIDDEN_DIM, device=DEVICE).squeeze(0)  # (B, H)
        preds = []
        for t in range(SEQ_LEN):
            h = rsm(h, z_seq[:, t, :], actions[:, t, :])
            preds.append(head(h))   # 預測 z_{t+1}

        pred   = torch.stack(preds, dim=1)      # (B, T, Z_DIM)
        target = z_seq[:, 1:, :]                # (B, T, Z_DIM)
        loss   = nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(f'  step {step+1:3d}  loss={loss.item():.4f}')

    should_decrease(losses, 'RecurrentStateModel')


# ── 梯度健康度檢查（共用）───────────────────────────────────
def check_gradients(model: nn.Module, name: str):
    """確認所有參數都有梯度、且沒有 NaN"""
    issues = []
    for pname, p in model.named_parameters():
        if p.grad is None:
            issues.append(f'{pname}: no grad')
        elif torch.isnan(p.grad).any():
            issues.append(f'{pname}: NaN grad')
    if issues:
        print(f'  ❌ {name} grad issues:')
        for i in issues:
            print(f'     {i}')
    else:
        print(f'  ✅ {name}: all gradients OK')


def test_gradient_flow():
    print_header('Gradient Flow Check')

    x, target = make_sine_batch(BATCH, SEQ_LEN, DEVICE)

    # GRUSequence
    gru  = GRUSequence(input_dim=1, hidden_dim=HIDDEN_DIM, layers=2).to(DEVICE)
    head = nn.Linear(HIDDEN_DIM, 1).to(DEVICE)
    h_seq, _ = gru(x)
    loss = nn.functional.mse_loss(head(h_seq), target)
    loss.backward()
    check_gradients(gru, 'GRUSequence')

    # RecurrentStateModel
    Z_DIM, ACTION_DIM = 16, 4
    rsm  = RecurrentStateModel(HIDDEN_DIM, Z_DIM, ACTION_DIM).to(DEVICE)
    head2 = nn.Linear(HIDDEN_DIM, Z_DIM).to(DEVICE)
    h     = get_initial_state(BATCH, HIDDEN_DIM, device=DEVICE).squeeze(0)
    phase = torch.rand(BATCH, Z_DIM, device=DEVICE) * 2 * math.pi
    t_idx = torch.arange(SEQ_LEN + 1, device=DEVICE).float()
    z_seq = torch.sin(t_idx[None, :, None] * 0.5 + phase[:, None, :])
    actions = torch.randn(BATCH, SEQ_LEN, ACTION_DIM, device=DEVICE)
    preds = []
    for t in range(SEQ_LEN):
        h = rsm(h, z_seq[:, t, :], actions[:, t, :])
        preds.append(head2(h))
    loss2 = nn.functional.mse_loss(torch.stack(preds, 1), z_seq[:, 1:])
    loss2.backward()
    check_gradients(rsm, 'RecurrentStateModel')


# ── 入口 ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'device: {DEVICE}')

    test_normed_gru_cell()
    test_gru_sequence()
    test_recurrent_state_model()
    test_gradient_flow()

    print('\n🎉 All tests done.')