"""Independent CUDA implementation of the pinned vLLM scoring entry point."""

from functools import cache
from pathlib import Path

import torch


@cache
def load_extension():
    """Build once, before timing or CUDA graph capture; uses PyTorch's toolchain."""
    from torch.utils.cpp_extension import load

    return load(
        name="compute_topk_scores_cuda",
        sources=[str(Path(__file__).resolve().parents[1] / "csrc" / "scores.cu")],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        is_python_module=False,
    )


def compute_topk_scores_cuda(
    logits,
    num_logprobs,
    sampled_token_ids,
    cu_num_logits=None,
    logprob_token_ids_state=None,
    expanded_idx_mapping=None,
    max_per_req_token_ids=0,
    logits_mode=False,
):
    """Same arguments/container as vLLM; equal logits use ascending token ID.

    Requires CUDA FP16/BF16/FP32 [B,V], unit inner stride, contiguous sampled
    IDs (int32/int64), 0 <= K <= V. Row padding is supported. Custom state
    buffers and mapping must be contiguous within each row, on the same GPU.
    Rank deliberately preserves upstream's >= comparison and -inf padding.
    Token indices/counts must be valid; invalid values trigger device asserts.

    Outputs: LogprobsTensors(ids, scores, ranks, boundaries, device_boundaries).
    IDs/scores have shape [B, 1 + max(K, max_per_req_token_ids)]; column zero
    is the sampled token. Scores are FP32; ranks are int64 [B]. IDs are int64
    for ordinary K>0, otherwise sampled_token_ids.dtype. Custom lists override
    the top-K columns per row, preserve duplicates/order, and pad with ID 0 and
    score -inf. logits_mode emits gathered raw logits instead of logprobs.
    cu_num_logits (list/Tensor/None) is passed through without copying.

    Original execution: torch.topk -> cat/custom Triton assembly -> Triton
    log-softmax gather (or torch.gather) -> optional masked_fill -> Triton rank.
    CUDA execution: tile selection -> merge/assembly -> scores -> rank.
    The pinned original function is neither patched nor replaced globally.
    """
    from vllm.v1.outputs import LogprobsTensors

    load_extension()
    custom = max_per_req_token_ids > 0
    if custom and (logprob_token_ids_state is None or expanded_idx_mapping is None):
        raise ValueError("custom token columns require state and expanded mapping")
    ids, scores, ranks = torch.ops.topk_scores_cuda.compute(
        logits,
        sampled_token_ids,
        num_logprobs,
        max_per_req_token_ids,
        logits_mode,
        expanded_idx_mapping if custom else None,
        logprob_token_ids_state.num_token_ids.gpu if custom else None,
        logprob_token_ids_state.token_ids.gpu if custom else None,
    )
    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        ids,
        scores,
        ranks,
        None if is_tensor else cu_num_logits,
        cu_num_logits if is_tensor else None,
    )
