from types import SimpleNamespace

import pytest
import torch

from topk_cuda import compute_topk_scores_cuda, load_extension
from topk_cuda.validation import compare_outputs, original_function

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def original():
    fn = original_function()
    load_extension()
    return fn


def reference_output(original, x, k, sampled, **kwargs):
    """Only skip int32 gather when the installed original actually rejects it."""
    try:
        return original(x, k, sampled, **kwargs)
    except RuntimeError as error:
        message = str(error).lower()
        int32_gather = (
            kwargs.get("logits_mode", False)
            and sampled.dtype == torch.int32
            and (k == 0 or kwargs.get("max_per_req_token_ids", 0) > 0)
        )
        if (
            int32_gather
            and "gather" in message
            and "dtype" in message
            and "int64" in message
        ):
            pytest.skip(f"installed original rejects int32 gather: {error}")
        raise


@pytest.mark.parametrize("batch", [1, 2, 8, 32])
@pytest.mark.parametrize("vocab", [128, 1024, 32000, 64000, 128000])
@pytest.mark.parametrize("k", [1, 5, 10, 20, 50])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matrix(original, batch, vocab, k, dtype):
    torch.manual_seed(401)
    x = torch.randn(batch, vocab, device="cuda", dtype=dtype)
    sampled = torch.randint(vocab, (batch,), device="cuda")
    compare_outputs(x, original(x, k, sampled), compute_topk_scores_cuda(x, k, sampled))


@pytest.mark.parametrize(
    "case",
    [
        "ties",
        "large",
        "negative",
        "masked",
        "all_inf",
        "nan",
        "positive_inf",
        "sample_nan",
        "signed_zero",
    ],
)
@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_special_values(original, case, logits_mode, dtype):
    x = torch.randn(2, 1031, device="cuda", dtype=dtype)
    sampled = torch.tensor([0, 1030], device="cuda")
    if case == "ties":
        x.fill_(1)
    elif case == "large":
        x = x * 10 + 10000
    elif case == "negative":
        x = x * 10 - 10000
    elif case == "masked":
        x[:, 100:] = -torch.inf
    elif case == "all_inf":
        x.fill_(-torch.inf)
    elif case == "nan":
        x[:, 1:4] = torch.nan
    elif case == "positive_inf":
        x[:, 2] = torch.inf
    elif case == "sample_nan":
        x[:, 0] = torch.nan
    elif case == "signed_zero":
        x.zero_()
        x[:, ::2] = -0.0
    ref = original(x, 50, sampled, logits_mode=logits_mode)
    got = compute_topk_scores_cuda(x, 50, sampled, logits_mode=logits_mode)
    compare_outputs(x, ref, got)
    if case in ("ties", "signed_zero"):
        torch.testing.assert_close(
            got.logprob_token_ids[:, 1:], torch.arange(50, device="cuda").expand(2, -1)
        )
    if case == "all_inf":
        assert got.selected_token_ranks.tolist() == [8192, 8192]


@pytest.mark.parametrize("k", [0, 1, 50, 128])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("logits_mode", [False, True])
def test_strided_rows_extreme_samples_and_metadata(original, k, id_dtype, logits_mode):
    x = torch.arange(128, device="cuda", dtype=torch.float32).repeat(4, 1)[::2]
    sampled = torch.tensor([127, 0], device="cuda", dtype=id_dtype)
    for boundaries in ([0, 1, 2], torch.tensor([0, 1, 2], device="cuda"), None):
        kwargs = {"cu_num_logits": boundaries, "logits_mode": logits_mode}
        compare_outputs(
            x,
            reference_output(original, x, k, sampled, **kwargs),
            compute_topk_scores_cuda(x, k, sampled, **kwargs),
            exact_ids=True,
        )


@pytest.mark.parametrize("k", [0, 5, 20])
@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("custom_width", [3, 128])
def test_mixed_custom_lists(original, k, logits_mode, id_dtype, dtype, custom_width):
    # Ordinary rows have ties; custom rows retain ordering, duplicates and padding.
    x = (torch.arange(1031, device="cuda") % 13).to(dtype).repeat(4, 1)
    sampled = torch.tensor([4, 6, 9, 10], device="cuda", dtype=id_dtype)
    custom_ids = torch.zeros(3, custom_width, device="cuda", dtype=torch.int32)
    custom_ids[1] = torch.arange(custom_width, device="cuda", dtype=torch.int32) % 7
    custom_ids[2, :2] = torch.tensor([8, 5], device="cuda", dtype=torch.int32)
    state = SimpleNamespace(
        num_token_ids=SimpleNamespace(
            gpu=torch.tensor([0, custom_width, 2], device="cuda", dtype=torch.int32)
        ),
        token_ids=SimpleNamespace(gpu=custom_ids),
    )
    kwargs = {
        "logprob_token_ids_state": state,
        "expanded_idx_mapping": torch.tensor(
            [1, 0, 2, 1], device="cuda", dtype=torch.int32
        ),
        "max_per_req_token_ids": custom_width,
        "logits_mode": logits_mode,
    }
    compare_outputs(
        x,
        reference_output(original, x, k, sampled, **kwargs),
        compute_topk_scores_cuda(x, k, sampled, **kwargs),
        custom_counts=[custom_width, 0, 2, custom_width],
        num_logprobs=k,
    )


def test_nondefault_stream_and_graph(original):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(8, 32000, device="cuda")
        sampled = torch.zeros(8, dtype=torch.int64, device="cuda")
        for _ in range(3):
            compute_topk_scores_cuda(x, 10, sampled)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = compute_topk_scores_cuda(x, 10, sampled)
        x.normal_()
        graph.replay()
        ref = original(x, 10, sampled)
    stream.synchronize()
    compare_outputs(x, ref, actual)


def test_real_vllm_custom_state(original):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState

    state = LogprobTokenIdsState(2, torch.device("cuda"))
    state.add_request(0, SamplingParams(logprob_token_ids=[7, 5, 7]))
    state.apply_staged_writes()
    x = torch.arange(128, device="cuda", dtype=torch.float32).repeat(2, 1)
    sampled = torch.tensor([8, 9], device="cuda")
    kwargs = {
        "logprob_token_ids_state": state,
        "expanded_idx_mapping": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        "max_per_req_token_ids": 3,
    }
    compare_outputs(
        x,
        original(x, 5, sampled, **kwargs),
        compute_topk_scores_cuda(x, 5, sampled, **kwargs),
        custom_counts=[3, 0],
        num_logprobs=5,
    )


@pytest.mark.parametrize("vocab", [8191, 8192, 8193])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rank_padding_boundary(original, vocab, dtype):
    x = torch.zeros(3, vocab, device="cuda", dtype=dtype)
    x[0].fill_(-torch.inf)
    x[1, 0] = -torch.inf
    # Row 2's sampled zero ties with all vocab entries, without padding matches.
    sampled = torch.zeros(3, device="cuda", dtype=torch.int64)
    ref = original(x, 5, sampled)
    got = compute_topk_scores_cuda(x, 5, sampled)
    compare_outputs(x, ref, got)
    padded = ((vocab + 8191) // 8192) * 8192
    assert got.selected_token_ranks.tolist() == [padded, padded, vocab]


def test_k_exceeding_tile_width(original):
    x = torch.arange(1031, device="cuda", dtype=torch.float32)[None, :]
    sampled = torch.zeros(1, device="cuda", dtype=torch.int64)
    compare_outputs(
        x,
        original(x, 1025, sampled),
        compute_topk_scores_cuda(x, 1025, sampled),
        exact_ids=True,
    )


def test_empty_and_rejected_shapes(original):
    x = torch.empty(0, 128, device="cuda")
    sampled = torch.empty(0, device="cuda", dtype=torch.int64)
    got = compute_topk_scores_cuda(x, 5, sampled)
    assert got.logprobs.shape == (0, 6)
    x = torch.randn(2, 128, device="cuda")
    sampled = torch.zeros(2, device="cuda", dtype=torch.int64)
    for k in (-1, 129):
        with pytest.raises((AssertionError, RuntimeError)):
            original(x, k, sampled)
        with pytest.raises(RuntimeError):
            compute_topk_scores_cuda(x, k, sampled)
    with pytest.raises(RuntimeError, match="unit inner stride"):
        compute_topk_scores_cuda(x[:, ::2], 5, sampled)
