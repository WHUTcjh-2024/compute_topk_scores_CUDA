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
    # Upstream gather in logits_mode requires int64 when K=0.
    if logits_mode and k == 0 and id_dtype == torch.int32:
        pytest.skip("upstream gather rejects int32 indices")
    for boundaries in ([0, 1, 2], torch.tensor([0, 1, 2], device="cuda"), None):
        kwargs = {"cu_num_logits": boundaries, "logits_mode": logits_mode}
        compare_outputs(
            x,
            original(x, k, sampled, **kwargs),
            compute_topk_scores_cuda(x, k, sampled, **kwargs),
            exact_ids=True,
        )


@pytest.mark.parametrize("k", [0, 5, 20])
@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_mixed_custom_lists(original, k, logits_mode, id_dtype):
    if logits_mode and id_dtype == torch.int32:
        pytest.skip("upstream gather rejects int32 custom output indices")
    x = torch.arange(1031, device="cuda", dtype=torch.float32).repeat(3, 1)
    sampled = torch.tensor([4, 6, 9], device="cuda", dtype=id_dtype)
    state = SimpleNamespace(
        num_token_ids=SimpleNamespace(
            gpu=torch.tensor([0, 3], device="cuda", dtype=torch.int32)
        ),
        token_ids=SimpleNamespace(
            gpu=torch.tensor([[0, 0, 0], [5, 5, 7]], device="cuda", dtype=torch.int32)
        ),
    )
    kwargs = {
        "logprob_token_ids_state": state,
        "expanded_idx_mapping": torch.tensor(
            [1, 0, 1], device="cuda", dtype=torch.int32
        ),
        "max_per_req_token_ids": 3,
        "logits_mode": logits_mode,
    }
    compare_outputs(
        x,
        original(x, k, sampled, **kwargs),
        compute_topk_scores_cuda(x, k, sampled, **kwargs),
        custom=True,
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
        custom=True,
    )


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
