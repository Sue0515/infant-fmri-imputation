"""
Balanced k-means patcher for (V, 3) coordinate point clouds: seeded
farthest-point init -> Lloyd k-means -> greedy rebalance until
max_size <= 2 * min_size. Torch-only and deterministic given (coords,
n_patches, seed), so a patcher can be rebuilt bit-identically from config.
`gather` maps (B, V, T) to padded patch tokens with a validity mask;
`scatter` is the exact inverse.
"""
from typing import Tuple

import torch


class SurfacePatcher(torch.nn.Module):
    """Deterministic balanced k-means partition of a vertex cloud.

    Attributes (buffers, so .to(device) moves them):
      patch_index     (V,) int64   patch id of every vertex
      patch_centroids (P, 3)       mean coordinate of each final patch
      patch_sizes     (P,) int64   member count per patch (sum == V)
      slot_mask       (P, max_size) bool  validity mask used by gather/scatter
    """

    def __init__(self, coords: torch.Tensor, n_patches: int, seed: int = 0):
        super().__init__()
        coords = torch.as_tensor(coords, dtype=torch.float32)
        assert coords.ndim == 2 and coords.shape[1] == 3, \
            f"coords must be (V, 3), got {tuple(coords.shape)}"
        V = coords.shape[0]
        assert 1 <= n_patches <= V, f"need 1 <= n_patches <= V, got {n_patches} vs V={V}"
        self.n_vertices = V
        self.n_patches = int(n_patches)

        centroids = self._farthest_point_init(coords, self.n_patches, seed)
        assign, centroids = self._lloyd(coords, centroids)
        assign = self._rebalance(coords, assign, centroids, self.n_patches)

        sizes = torch.bincount(assign, minlength=self.n_patches)
        # Final centroids from the *rebalanced* assignment (rebalance guarantees
        # no empty patch: min_size >= 1 once max <= 2*min holds).
        cent = torch.zeros(self.n_patches, 3).index_add_(0, assign, coords)
        cent = cent / sizes.clamp(min=1).unsqueeze(1).float()

        # Fixed slot layout for gather/scatter: patch p's members (ascending
        # vertex id) occupy slots [0, sizes[p]); the rest are padding.
        self.max_size = int(sizes.max())
        slot_index = torch.zeros(self.n_patches, self.max_size, dtype=torch.int64)
        slot_mask = torch.zeros(self.n_patches, self.max_size, dtype=torch.bool)
        for p in range(self.n_patches):
            members = (assign == p).nonzero(as_tuple=True)[0]
            slot_index[p, :members.numel()] = members
            slot_mask[p, :members.numel()] = True

        self.register_buffer('patch_index', assign)
        self.register_buffer('patch_centroids', cent)
        self.register_buffer('patch_sizes', sizes)
        self.register_buffer('slot_index', slot_index)
        self.register_buffer('slot_mask', slot_mask)

    # ------------------------------------------------------------------ build
    @staticmethod
    def _farthest_point_init(coords: torch.Tensor, k: int, seed: int) -> torch.Tensor:
        """Seeded first center, then greedy farthest-point selection."""
        g = torch.Generator().manual_seed(seed)
        idx = torch.empty(k, dtype=torch.int64)
        idx[0] = torch.randint(coords.shape[0], (1,), generator=g)
        min_d2 = ((coords - coords[idx[0]]) ** 2).sum(-1)
        for i in range(1, k):
            idx[i] = torch.argmax(min_d2)
            min_d2 = torch.minimum(min_d2, ((coords - coords[idx[i]]) ** 2).sum(-1))
        return coords[idx].clone()

    @staticmethod
    def _lloyd(coords: torch.Tensor, centroids: torch.Tensor,
               n_iters: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plain Lloyd iterations; stop early once assignments are stable."""
        assign = None
        for _ in range(n_iters):
            new_assign = torch.cdist(coords, centroids).argmin(dim=1)
            if assign is not None and torch.equal(new_assign, assign):
                break
            assign = new_assign
            counts = torch.bincount(assign, minlength=centroids.shape[0])
            sums = torch.zeros_like(centroids).index_add_(0, assign, coords)
            # empty clusters keep their previous centroid
            centroids = torch.where((counts > 0).unsqueeze(1),
                                    sums / counts.clamp(min=1).unsqueeze(1).float(),
                                    centroids)
        return assign, centroids

    @staticmethod
    def _rebalance(coords: torch.Tensor, assign: torch.Tensor,
                   centroids: torch.Tensor, n_patches: int) -> torch.Tensor:
        """Move boundary points of the largest patch to the nearest undersized
        patch until max_size <= 2 * min_size (or a bounded number of moves).

        A receiver must have size < ceil(max_size / 2): each move then strictly
        decreases sum(sizes^2), so termination is guaranteed; the bound is a
        safety net. Empty clusters count as undersized and get filled first.
        Centroids stay fixed during rebalance (recomputed by the caller).
        """
        assign = assign.clone()
        sizes = torch.bincount(assign, minlength=n_patches)
        for _ in range(8 * coords.shape[0]):
            mx = int(sizes.max())
            if mx <= 2 * int(sizes.min()):
                break
            donor = int(sizes.argmax())
            receivers = (sizes < (mx + 1) // 2).nonzero(as_tuple=True)[0]
            members = (assign == donor).nonzero(as_tuple=True)[0]
            d = torch.cdist(coords[members], centroids[receivers])
            flat = int(d.argmin())
            point = members[flat // receivers.numel()]
            dest = receivers[flat % receivers.numel()]
            assign[point] = dest
            sizes[donor] -= 1
            sizes[dest] += 1
        return assign

    # -------------------------------------------------------------- (de)token
    def gather(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, V, T) -> tokens (B, P, max_size, T), validity mask (P, max_size).

        Valid slots carry x values bit-identically; padded slots are zero.
        """
        B, V, T = x.shape
        assert V == self.n_vertices, f"expected V={self.n_vertices}, got {V}"
        tokens = x[:, self.slot_index.reshape(-1), :] \
            .reshape(B, self.n_patches, self.max_size, T)
        tokens = tokens.masked_fill(~self.slot_mask[None, :, :, None], 0.0)
        return tokens, self.slot_mask.clone()

    def scatter(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, P, max_size, T) -> (B, V, T). Exact inverse of gather:
        valid-slot vertex ids form a permutation of arange(V)."""
        B, P, S, T = tokens.shape
        assert P == self.n_patches and S == self.max_size, \
            f"expected (B, {self.n_patches}, {self.max_size}, T), got {tuple(tokens.shape)}"
        valid = self.slot_mask.reshape(-1)
        out = tokens.new_zeros(B, self.n_vertices, T)
        out[:, self.slot_index.reshape(-1)[valid], :] = \
            tokens.reshape(B, P * S, T)[:, valid, :]
        return out
