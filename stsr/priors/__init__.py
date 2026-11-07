"""
Signal priors. `FrequencyDecomposition` is the one module on the paper
path: SpaceTimeAttentionNet splits the gap-filled observed signal into low/high
frequency components and injects them as the routing prompt tokens.
"""
from .frequency_decomposition import FrequencyDecomposition

__all__ = ['FrequencyDecomposition']
