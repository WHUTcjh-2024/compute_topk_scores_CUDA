"""Shared original-source verification and tie-aware output comparison."""

import hashlib
import importlib
import inspect
import json
import subprocess
from pathlib import Path

import torch


def original_function():
    module = importlib.import_module("vllm.v1.worker.gpu.sample.logprob")
    root = Path(inspect.getfile(module)).resolve().parents[5]
    lock = json.loads((Path(__file__).parents[1] / "baseline.lock.json").read_text())
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != lock["commit"]:
        raise RuntimeError(f"baseline commit mismatch: {commit}")
    if subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=False
    ).returncode:
        raise RuntimeError("baseline checkout has tracked modifications")
    for relative, expected in lock["files"].items():
        digest = hashlib.sha256(
            (root / relative).read_text(encoding="utf-8").encode()
        ).hexdigest()
        if digest != expected:
            raise RuntimeError(f"baseline source modified: {relative}")
    return module.compute_topk_scores


def compare_outputs(
    logits,
    reference,
    actual,
    *,
    exact_ids=False,
    custom_counts=None,
    num_logprobs=None,
):
    """Validate IDs as well as scores; only permit IDs differing at equal logits.

    Reference scores remain authoritative. Comparing raw gathered logits also
    prevents all-NaN logprobs from concealing incorrect Top-K membership.
    Custom lists and sampled IDs always require exact equality. custom_counts
    contains the expanded per-row custom list lengths (zero means Top-K).
    Supply num_logprobs with custom_counts to distinguish Top-K from padding.
    """
    a, b = reference.logprob_token_ids, actual.logprob_token_ids
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    assert actual.logprobs.dtype == torch.float32
    assert actual.selected_token_ranks.dtype == torch.int64
    torch.testing.assert_close(a[:, 0], b[:, 0], rtol=0, atol=0)
    if custom_counts is None:
        counts = torch.zeros(a.shape[0], dtype=torch.int64, device=a.device)
        k = a.shape[1] - 1
    else:
        counts = torch.as_tensor(custom_counts, dtype=torch.int64, device=a.device)
        if num_logprobs is None:
            raise ValueError("num_logprobs is required with custom_counts")
        k = num_logprobs
        if counts.shape != (a.shape[0],) or not 0 <= k < a.shape[1]:
            raise ValueError("invalid custom_counts shape or num_logprobs")
        if ((counts < 0) | (counts >= a.shape[1])).any().item():
            raise ValueError("custom_counts exceed output width")
    custom_rows = counts > 0
    lengths = torch.where(custom_rows, counts, k)
    padding = torch.arange(a.shape[1], device=a.device)[None, :] > lengths[:, None]
    # Padding is not a Top-K candidate: repeated zero IDs there are valid.
    torch.testing.assert_close(b[padding], torch.zeros_like(b[padding]), rtol=0, atol=0)
    torch.testing.assert_close(a[padding], b[padding], rtol=0, atol=0)
    torch.testing.assert_close(
        actual.logprobs[padding],
        torch.full_like(actual.logprobs[padding], -torch.inf),
        rtol=0,
        atol=0,
    )
    if exact_ids:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        torch.testing.assert_close(a[custom_rows], b[custom_rows], rtol=0, atol=0)
        ordinary = ~custom_rows
        top_a, top_b = a[ordinary, 1 : k + 1], b[ordinary, 1 : k + 1]
        torch.testing.assert_close(
            logits[ordinary].gather(1, top_a.long()),
            logits[ordinary].gather(1, top_b.long()),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        if k > 1:
            sorted_ids = top_b.sort(dim=1).values
            assert not (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item()
    torch.testing.assert_close(
        reference.logprobs, actual.logprobs, rtol=2e-5, atol=2e-5, equal_nan=True
    )
    torch.testing.assert_close(
        reference.selected_token_ranks, actual.selected_token_ranks, rtol=0, atol=0
    )
    assert reference.cu_num_generated_tokens == actual.cu_num_generated_tokens
    assert (
        reference.cu_num_generated_tokens_tensor
        is actual.cu_num_generated_tokens_tensor
    )
