
import math

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

from cs336_basics.nn_utils import softmax


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with NVTX annotations."""

    with nvtx.range("scaled dot product attention"):

        # 1. Compute attention scores
        with nvtx.range("computing attention scores"):
            d_k = K.shape[-1]

            attention_scores = (
                einsum(
                    Q,
                    K,
                    "... query d_k, ... key d_k -> ... query key",
                )
                / math.sqrt(d_k)
            )

            if mask is not None:
                attention_scores = torch.where(
                    mask,
                    attention_scores,
                    float("-inf"),
                )

        # 2. Compute softmax
        with nvtx.range("computing softmax"):
            attention_weights = softmax(
                attention_scores,
                dim=-1,
            )

        # 3. Final matrix multiplication
        with nvtx.range("final matmul"):
            output = einsum(
                attention_weights,
                V,
                "... query key, ... key d_v -> ... query d_v",
            )

    return output
