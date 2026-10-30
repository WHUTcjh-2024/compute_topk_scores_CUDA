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


def compare_outputs(logits, reference, actual, *, exact_ids=False, custom=False):
    """Validate IDs as well as scores; only permit IDs differing at equal logits.

    Reference scores remain authoritative. Comparing raw gathered logits also
    prevents all-NaN logprobs from concealing incorrect Top-K membership.
    Custom lists and sampled IDs always require exact equality.
    """
    a, b = reference.logprob_token_ids, actual.logprob_token_ids
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    assert actual.logprobs.dtype == torch.float32
    assert actual.selected_token_ranks.dtype == torch.int64
    torch.testing.assert_close(a[:, 0], b[:, 0], rtol=0, atol=0)
    if exact_ids or custom:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        torch.testing.assert_close(
            logits.gather(1, a.long()),
            logits.gather(1, b.long()),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        if b.shape[1] > 1:
            sorted_ids = b[:, 1:].sort(dim=1).values
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
