"""
SimpleVelocityNet: a deliberately tiny axial-attention velocity network,
kept only as the minimal plumbing reference (`--model simple`, flow
objective) — not the paper model. `timestep_embedding` also lives here and
IS on the paper path (velocity-head flow-time conditioning). Layout
(B, K, L); no learned parameter depends on K or L.
"""
import torch
import torch.nn as nn


def timestep_embedding(t_cont: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding of continuous flow time t_cont (B,) in [0, 1].

    Frequencies span 10^0..10^4 like the source repo's DiffusionEmbedding table,
    but evaluated continuously instead of via a discrete-step lookup.
    Returns (B, dim), dim must be even (dim/2 sin + dim/2 cos).
    """
    assert dim % 2 == 0, "embedding dim must be even"
    half = dim // 2
    frequencies = 10.0 ** (
        torch.arange(half, device=t_cont.device, dtype=t_cont.dtype)
        / max(half - 1, 1) * 4.0
    ).unsqueeze(0)  # (1, half)
    table = t_cont.unsqueeze(1) * frequencies  # (B, half)
    return torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (B, dim)


class AxialBlock(nn.Module):
    """
    Pre-LN residual block on (B, K, L, C):
    temporal self-attention over L (K folded into batch) ->
    feature self-attention over K (L folded into batch) -> pointwise FFN.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm_time = nn.LayerNorm(d_model)
        self.attn_time = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_feat = nn.LayerNorm(d_model)
        self.attn_feat = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, K, L, C = h.shape

        # Temporal attention: (B, K, L, C) -> (B*K, L, C), mirrors CSDI forward_time
        y = self.norm_time(h).reshape(B * K, L, C)
        y, _ = self.attn_time(y, y, y, need_weights=False)
        h = h + y.reshape(B, K, L, C)

        # Feature attention: (B, K, L, C) -> (B*L, K, C), mirrors CSDI forward_feature
        y = self.norm_feat(h).permute(0, 2, 1, 3).reshape(B * L, K, C)
        y, _ = self.attn_feat(y, y, y, need_weights=False)
        h = h + y.reshape(B, L, K, C).permute(0, 2, 1, 3)

        # Pointwise FFN
        h = h + self.ffn(self.norm_ffn(h))
        return h


class SimpleVelocityNet(nn.Module):
    """
    Minimal flow-matching velocity network for W1 smoke tests.

    forward(x_t, cond_pinned, cond_mask, t_cont) -> velocity (B, K, L)
      x_t:         (B, K, L) current flow state
      cond_pinned: (B, K, L) observed values (zeros where unobserved)
      cond_mask:   (B, K, L) 1 = observed / pinned, 0 = to reconstruct
      t_cont:      (B,) flow time in [0, 1]
    """
    def __init__(self, d_model: int = 64, n_blocks: int = 2, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model

        # 3 input channels [x_t, cond_pinned, cond_mask] -> d_model, 1x1 conv over K*L
        self.input_proj = nn.Conv1d(3, d_model, kernel_size=1)

        self.t_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList(
            [AxialBlock(d_model, n_heads) for _ in range(n_blocks)]
        )

        self.norm_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Conv1d(d_model, 1, kernel_size=1)

    def forward(
        self,
        x_t: torch.Tensor,
        cond_pinned: torch.Tensor,
        cond_mask: torch.Tensor,
        t_cont: torch.Tensor,
    ) -> torch.Tensor:
        B, K, L = x_t.shape

        # (B, 3, K, L) -> (B, 3, K*L) -> 1x1 conv -> (B, d_model, K*L)
        h = torch.stack([x_t, cond_pinned, cond_mask], dim=1)
        h = self.input_proj(h.reshape(B, 3, K * L))

        # Sinusoidal time embedding, linear-projected, broadcast over all positions
        t_emb = self.t_proj(timestep_embedding(t_cont, self.d_model))  # (B, d_model)
        h = h + t_emb.unsqueeze(-1)

        # (B, d_model, K*L) -> channels-last (B, K, L, d_model) for the blocks
        h = h.reshape(B, self.d_model, K, L).permute(0, 2, 3, 1)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)

        # (B, K, L, d_model) -> (B, d_model, K*L) -> 1x1 conv -> (B, K, L)
        h = h.permute(0, 3, 1, 2).reshape(B, self.d_model, K * L)
        velocity = self.output_proj(h).reshape(B, K, L)
        return velocity


if __name__ == '__main__':
    torch.manual_seed(0)
    net = SimpleVelocityNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"SimpleVelocityNet params: {n_params:,} (target: well under 1M)")
    assert n_params < 1_000_000

    for K, L in [(6, 12), (9, 7)]:
        B = 2
        x_t = torch.randn(B, K, L)
        cond_pinned = torch.randn(B, K, L)
        cond_mask = (torch.rand(B, K, L) < 0.5).float()
        t_cont = torch.rand(B)
        v = net(x_t, cond_pinned, cond_mask, t_cont)
        assert v.shape == (B, K, L)
        v.pow(2).mean().backward()
        print(f"(K={K}, L={L}) -> velocity {tuple(v.shape)}  OK")
    print("trunk: all tests passed!")
