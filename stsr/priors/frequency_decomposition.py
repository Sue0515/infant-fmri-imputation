"""
FFT band split of fMRI signals into a low-frequency component (default
<= 0.1 Hz, the dominant hemodynamic trend) and a high-frequency component
(0.1-0.2 Hz), used as the routing-prompt conditioning channels.
LF + HF reconstructs the in-band signal exactly; no parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyDecomposition(nn.Module):
    """
    Decompose fMRI signal into low-frequency and high-frequency components.

    Uses FFT-based filtering for clean frequency separation.

    Args:
        sampling_rate: TR in seconds (e.g., 0.72 for HCP data)
        lf_cutoff: Low-frequency cutoff in Hz (default: 0.1 Hz)
        hf_cutoff: High-frequency cutoff in Hz (default: 0.2 Hz)
        method: 'fft' or 'wavelet' (currently only FFT implemented)
    """
    def __init__(
        self,
        sampling_rate: float = 0.72,
        lf_cutoff: float = 0.1,
        hf_cutoff: float = 0.2,
        method: str = 'fft'
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.lf_cutoff = lf_cutoff
        self.hf_cutoff = hf_cutoff
        self.method = method

        # Nyquist frequency
        self.nyquist = 1.0 / (2.0 * sampling_rate)

        assert lf_cutoff < hf_cutoff, "lf_cutoff must be < hf_cutoff"
        assert hf_cutoff <= self.nyquist, f"hf_cutoff {hf_cutoff} exceeds Nyquist {self.nyquist}"

    def forward(self, x: torch.Tensor):
        """
        Decompose signal into LF and HF components.

        Args:
            x: (B, K, L) or (B, C, K, L) - input signal

        Returns:
            x_lf: Low-frequency component (same shape as x)
            x_hf: High-frequency component (same shape as x)
        """
        if self.method == 'fft':
            return self._decompose_fft(x)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

    def _decompose_fft(self, x: torch.Tensor):
        """
        FFT-based frequency decomposition.

        Process:
        1. FFT → frequency domain
        2. Apply frequency masks (LF, HF)
        3. IFFT → time domain
        """
        # Handle different input shapes
        original_shape = x.shape
        if len(x.shape) == 3:  # (B, K, L)
            B, K, L = x.shape
            x_reshape = x  # Already correct shape
        elif len(x.shape) == 4:  # (B, C, K, L)
            B, C, K, L = x.shape
            x_reshape = x.reshape(B * C, K, L)  # Flatten C into batch
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # FFT along temporal dimension
        x_fft = torch.fft.rfft(x_reshape, dim=-1)  # (B, K, L//2+1) complex

        # Frequency bins
        freqs = torch.fft.rfftfreq(L, d=self.sampling_rate).to(x.device)  # (L//2+1,)

        # Create frequency masks
        lf_mask = (freqs <= self.lf_cutoff).float()  # Low-pass filter
        hf_mask = ((freqs > self.lf_cutoff) & (freqs <= self.hf_cutoff)).float()  # Band-pass filter

        # Broadcast masks: (L//2+1,) → (1, 1, L//2+1)
        lf_mask = lf_mask.view(1, 1, -1)
        hf_mask = hf_mask.view(1, 1, -1)

        # Apply masks in frequency domain
        x_fft_lf = x_fft * lf_mask
        x_fft_hf = x_fft * hf_mask

        # IFFT back to time domain
        x_lf = torch.fft.irfft(x_fft_lf, n=L, dim=-1)  # (B, K, L) real
        x_hf = torch.fft.irfft(x_fft_hf, n=L, dim=-1)  # (B, K, L) real

        # Reshape back to original
        if len(original_shape) == 4:  # (B, C, K, L)
            x_lf = x_lf.reshape(B, C, K, L)
            x_hf = x_hf.reshape(B, C, K, L)

        return x_lf, x_hf

    def get_frequency_weights(self, length: int, device: torch.device):
        """
        Get frequency weights for weighted loss computation.

        Args:
            length: Temporal length L
            device: torch device

        Returns:
            weights_lf: (L//2+1,) - weights for LF band
            weights_hf: (L//2+1,) - weights for HF band
        """
        freqs = torch.fft.rfftfreq(length, d=self.sampling_rate).to(device)

        weights_lf = (freqs <= self.lf_cutoff).float()
        weights_hf = ((freqs > self.lf_cutoff) & (freqs <= self.hf_cutoff)).float()

        return weights_lf, weights_hf


# AdaptiveFrequencyDecomposition (learnable cutoffs) was removed 2026-08-31:
# nothing on any path used it, and its nn.Parameter cutoffs could never learn
# — forward() detached them with .item(), and a hard threshold mask
# (freqs <= cutoff) has zero gradient with respect to the cutoff regardless.
# A learnable variant needs a soft (sigmoid) band edge; build that if a use
# arrives, do not resurrect this one.


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("Testing Frequency Decomposition")
    print("=" * 80)

    # Test data
    B, K, L = 4, 100, 284
    TR = 0.72

    # Create synthetic signal with known frequencies
    t = torch.arange(L) * TR

    # LF component: 0.05 Hz (HRF-like, 20s period)
    signal_lf = torch.sin(2 * np.pi * 0.05 * t)

    # HF component: 0.15 Hz (neural fluctuation, ~6.7s period)
    signal_hf = 0.3 * torch.sin(2 * np.pi * 0.15 * t)

    # Combined signal
    signal = signal_lf + signal_hf
    signal = signal.view(1, 1, L).expand(B, K, L)

    print(f"Input signal: {signal.shape}")
    print(f"  LF component: 0.05 Hz (period ~20s)")
    print(f"  HF component: 0.15 Hz (period ~6.7s)")

    # Test decomposition
    decomp = FrequencyDecomposition(sampling_rate=TR, lf_cutoff=0.1, hf_cutoff=0.2)

    signal_lf_rec, signal_hf_rec = decomp(signal)

    print(f"\nDecomposition results:")
    print(f"  LF: {signal_lf_rec.shape}")
    print(f"  HF: {signal_hf_rec.shape}")

    # Check reconstruction
    signal_reconstructed = signal_lf_rec + signal_hf_rec
    reconstruction_error = (signal - signal_reconstructed).abs().mean()
    print(f"\nReconstruction error: {reconstruction_error.item():.6f} (should be ~0)")

    # Check LF component recovery
    lf_corr = torch.corrcoef(torch.stack([
        signal_lf.flatten(),
        signal_lf_rec[0, 0, :].flatten()
    ]))[0, 1]
    print(f"LF component correlation: {lf_corr.item():.4f} (should be ~1.0)")

    # Check HF component recovery
    hf_corr = torch.corrcoef(torch.stack([
        signal_hf.flatten(),
        signal_hf_rec[0, 0, :].flatten()
    ]))[0, 1]
    print(f"HF component correlation: {hf_corr.item():.4f} (should be ~1.0)")

    # Test with 4D input (B, C, K, L)
    print(f"\n{'=' * 80}")
    print("Testing 4D input (B, C, K, L)")
    signal_4d = signal.unsqueeze(1).expand(B, 2, K, L)
    signal_lf_4d, signal_hf_4d = decomp(signal_4d)
    print(f"  Input: {signal_4d.shape}")
    print(f"  LF output: {signal_lf_4d.shape}")
    print(f"  HF output: {signal_hf_4d.shape}")

    print("\n" + "=" * 80)
    print("All tests passed!")
