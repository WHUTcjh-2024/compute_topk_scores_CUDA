"""CPU tests of the comparator, not evidence of CUDA correctness."""

from types import SimpleNamespace

import pytest
import torch

from topk_cuda.validation import compare_outputs


def output(ids, scores=None, rank=2):
    return SimpleNamespace(
        logprob_token_ids=torch.tensor([ids], dtype=torch.int64),
        logprobs=torch.tensor([scores or [0.0] * len(ids)], dtype=torch.float32),
        selected_token_ranks=torch.tensor([rank], dtype=torch.int64),
        cu_num_generated_tokens=None,
        cu_num_generated_tokens_tensor=None,
    )


def test_equal_logits_allow_different_ids_but_exact_mode_rejects():
    x = torch.tensor([[5.0, 5.0, 1.0]])
    ref, got = output([2, 0]), output([2, 1])
    compare_outputs(x, ref, got)
    with pytest.raises(AssertionError):
        compare_outputs(x, ref, got, exact_ids=True)


def test_nan_scores_cannot_hide_wrong_membership():
    x = torch.tensor([[float("nan"), 5.0, 1.0]])
    ref = output([2, 0, 1], [float("nan")] * 3)
    got = output([2, 0, 2], [float("nan")] * 3)
    with pytest.raises(AssertionError):
        compare_outputs(x, ref, got)


def test_duplicate_topk_ids_rejected_even_with_equal_values():
    x = torch.ones(1, 3)
    with pytest.raises(AssertionError):
        compare_outputs(x, output([2, 0, 1]), output([2, 0, 0]))


def test_wrong_rank_rejected():
    x = torch.ones(1, 3)
    with pytest.raises(AssertionError):
        compare_outputs(x, output([2, 0]), output([2, 0], rank=1))


def test_custom_order_and_duplicates_are_preserved():
    x = torch.ones(1, 3)
    compare_outputs(x, output([2, 1, 1]), output([2, 1, 1]), custom=True)
    with pytest.raises(AssertionError):
        compare_outputs(x, output([2, 1, 1]), output([2, 0, 1]), custom=True)
