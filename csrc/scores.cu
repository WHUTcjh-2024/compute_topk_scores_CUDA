#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <climits>
#include <optional>
#include <tuple>

namespace {
constexpr int Threads = 256;
constexpr int Tile = 1024;

struct Candidate { float value; int64_t id; };
struct Better {
  __device__ Candidate operator()(Candidate a, Candidate b) const {
    if (a.id < 0) return b;
    if (b.id < 0) return a;
    const bool an = isnan(a.value), bn = isnan(b.value);
    if (an != bn) return an ? a : b;
    if (!an && a.value != b.value) return a.value > b.value ? a : b;
    return a.id < b.id ? a : b;
  }
};
struct PropagateMax {
  __device__ float operator()(float a, float b) const {
    return isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
  }
};
struct Indices {
  const void* data;
  bool wide;
  __device__ int64_t at(int64_t i) const {
    return wide ? static_cast<const int64_t*>(data)[i]
                : static_cast<const int32_t*>(data)[i];
  }
};

// Cache a tile once, then select K maxima. No vocabulary-wide full sort.
template <typename T>
__global__ void tile_topk(const T* x, int64_t stride, int64_t vocab,
                          int64_t k, int64_t tiles, float* values, int64_t* ids) {
  using Reduce = cub::BlockReduce<Candidate, Threads>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ int64_t winner;
  Candidate local[Tile / Threads];
  const int64_t row = blockIdx.x, tile = blockIdx.y;
  #pragma unroll
  for (int j = 0; j < Tile / Threads; ++j) {
    const int64_t col = tile * Tile + threadIdx.x + j * Threads;
    local[j] = col < vocab ? Candidate{float(x[row * stride + col]), col}
                           : Candidate{-INFINITY, -1};
  }
  for (int64_t n = 0; n < k; ++n) {
    Candidate best{-INFINITY, -1};
    #pragma unroll
    for (int j = 0; j < Tile / Threads; ++j) best = Better{}(best, local[j]);
    best = Reduce(temp).Reduce(best, Better{});
    if (threadIdx.x == 0) {
      const int64_t offset = (row * tiles + tile) * k + n;
      values[offset] = best.value;
      ids[offset] = best.id;
      winner = best.id;
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < Tile / Threads; ++j)
      if (local[j].id == winner) local[j].id = -1;
    __syncthreads();
  }
}

template <typename I>
__global__ void assemble(float* values, int64_t* candidates, int64_t count,
                         I* out, bool* valid, int64_t cols, int64_t k,
                         Indices sampled, Indices mapping, Indices counts,
                         Indices custom_ids, int64_t custom_stride,
                         int64_t requests, int64_t custom_width, bool custom,
                         int64_t vocab) {
  using Reduce = cub::BlockReduce<Candidate, Threads>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ int64_t winner;
  const int64_t row = blockIdx.x;
  const int64_t sample = sampled.at(row);
  CUDA_KERNEL_ASSERT(sample >= 0 && sample < vocab);
  for (int64_t col = threadIdx.x; col < cols; col += Threads) {
    out[row * cols + col] = col == 0 ? sample : 0;
    valid[row * cols + col] = col == 0;
  }
  __syncthreads();
  if (custom) {
    const int64_t req = mapping.at(row);
    CUDA_KERNEL_ASSERT(req >= 0 && req < requests);
    const int64_t n = counts.at(req);
    CUDA_KERNEL_ASSERT(n >= 0 && n <= custom_width && n < cols);
    if (n > 0) {
      for (int64_t j = threadIdx.x; j < n; j += Threads) {
        const int64_t id = custom_ids.at(req * custom_stride + j);
        CUDA_KERNEL_ASSERT(id >= 0 && id < vocab);
        out[row * cols + 1 + j] = id;
        valid[row * cols + 1 + j] = true;
      }
      return;
    }
  }
  for (int64_t n = 0; n < k; ++n) {
    Candidate best{-INFINITY, -1};
    for (int64_t j = threadIdx.x; j < count; j += Threads)
      best = Better{}(best, {values[row * count + j], candidates[row * count + j]});
    best = Reduce(temp).Reduce(best, Better{});
    if (threadIdx.x == 0) {
      winner = best.id;
      out[row * cols + n + 1] = best.id;
      valid[row * cols + n + 1] = true;
    }
    __syncthreads();
    for (int64_t j = threadIdx.x; j < count; j += Threads)
      if (candidates[row * count + j] == winner) candidates[row * count + j] = -1;
    __syncthreads();
  }
}

template <typename T, typename I>
__global__ void score(const T* x, int64_t stride, int64_t vocab,
                      const I* ids, const bool* valid, int64_t cols,
                      float* out, bool logits_mode) {
  using Reduce = cub::BlockReduce<float, Threads>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ float row_max, log_sum;
  const int64_t row = blockIdx.x;
  const T* data = x + row * stride;
  if (!logits_mode) {
    float m = -INFINITY;
    for (int64_t j = threadIdx.x; j < vocab; j += Threads)
      m = PropagateMax{}(m, float(data[j]));
    m = Reduce(temp).Reduce(m, PropagateMax{});
    if (threadIdx.x == 0) row_max = m;
    __syncthreads();
    float sum = 0;
    for (int64_t j = threadIdx.x; j < vocab; j += Threads)
      sum += expf(float(data[j]) - row_max);
    sum = Reduce(temp).Sum(sum);
    if (threadIdx.x == 0) log_sum = logf(sum);
    __syncthreads();
  }
  for (int64_t j = threadIdx.x; j < cols; j += Threads) {
    const int64_t offset = row * cols + j;
    float value = -INFINITY;
    if (valid[offset]) {
      value = float(data[ids[offset]]);
      if (!logits_mode) value = value - row_max - log_sum;
    }
    out[offset] = value;
  }
}

template <typename T>
__global__ void ranks(const T* x, int64_t stride, int64_t vocab,
                      Indices sampled, int64_t* out) {
  using Reduce = cub::BlockReduce<int64_t, Threads>;
  __shared__ typename Reduce::TempStorage temp;
  const int64_t row = blockIdx.x, id = sampled.at(row);
  CUDA_KERNEL_ASSERT(id >= 0 && id < vocab);
  const float target = float(x[row * stride + id]);
  int64_t n = 0;
  for (int64_t j = threadIdx.x; j < vocab; j += Threads)
    n += float(x[row * stride + j]) >= target;
  n = Reduce(temp).Sum(n);
  if (threadIdx.x == 0) {
    // Upstream's 8192-wide Triton loop compares masked -inf lanes too.
    if (target == -INFINITY) n += ((vocab + 8191) / 8192) * 8192 - vocab;
    out[row] = n;
  }
}

void check_indices(const at::Tensor& t, const at::Tensor& x, int dim) {
  TORCH_CHECK(t.device() == x.device(), "indices must be on logits device");
  TORCH_CHECK(t.scalar_type() == at::kInt || t.scalar_type() == at::kLong,
              "indices must be int32 or int64");
  TORCH_CHECK(t.dim() == dim && t.stride(dim - 1) == 1,
              "indices require unit inner stride and expected rank");
}
Indices view(const at::Tensor& t) { return {t.const_data_ptr(), t.scalar_type() == at::kLong}; }

std::tuple<at::Tensor, at::Tensor, at::Tensor> compute(
    const at::Tensor& x, const at::Tensor& sampled, int64_t k,
    int64_t max_custom, bool logits_mode,
    const std::optional<at::Tensor>& mapping,
    const std::optional<at::Tensor>& counts,
    const std::optional<at::Tensor>& custom_ids) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1,
              "logits must be CUDA [B,V] with unit inner stride");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
              x.scalar_type() == at::kBFloat16, "unsupported logits dtype");
  const int64_t batch = x.size(0), vocab = x.size(1);
  TORCH_CHECK(vocab > 0 && k >= 0 && k <= vocab && max_custom >= 0,
              "require V > 0, 0 <= K <= V, max_custom >= 0");
  TORCH_CHECK(batch <= INT_MAX && (vocab + Tile - 1) / Tile <= 65535,
              "shape exceeds CUDA launch grid limits");
  check_indices(sampled, x, 1);
  TORCH_CHECK(sampled.numel() == batch, "sampled IDs must have shape [B]");
  const bool custom = max_custom > 0;
  Indices map{nullptr, false}, cnt{nullptr, false}, tok{nullptr, false};
  int64_t custom_stride = 0, requests = 0, custom_width = 0;
  if (custom) {
    TORCH_CHECK(mapping && counts && custom_ids, "missing custom state");
    check_indices(*mapping, x, 1);
    check_indices(*counts, x, 1);
    check_indices(*custom_ids, x, 2);
    TORCH_CHECK(mapping->numel() == batch && counts->numel() == custom_ids->size(0),
                "invalid custom state shapes");
    map = view(*mapping); cnt = view(*counts); tok = view(*custom_ids);
    custom_stride = custom_ids->stride(0);
    requests = counts->numel(); custom_width = custom_ids->size(1);
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const int64_t cols = 1 + std::max(k, max_custom);
  const auto id_type = custom || k == 0 ? sampled.scalar_type() : at::kLong;
  auto ids = at::empty({batch, cols}, x.options().dtype(id_type));
  auto scores = at::empty({batch, cols}, x.options().dtype(at::kFloat));
  auto rank = at::empty({batch}, x.options().dtype(at::kLong));
  if (batch == 0) return {ids, scores, rank};
  auto valid = at::empty({batch, cols}, x.options().dtype(at::kBool));
  const int64_t tiles = (vocab + Tile - 1) / Tile;
  const int64_t tile_k = std::min<int64_t>(k, Tile);
  auto values = at::empty({batch, tiles * tile_k}, x.options().dtype(at::kFloat));
  auto candidates = at::empty({batch, tiles * tile_k}, x.options().dtype(at::kLong));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "topk_scores", [&] {
    if (k > 0) {
      tile_topk<scalar_t><<<dim3(batch, tiles), Threads, 0, stream>>>(
          x.const_data_ptr<scalar_t>(), x.stride(0), vocab, tile_k, tiles,
          values.mutable_data_ptr<float>(), candidates.mutable_data_ptr<int64_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    // Keep logits and output-ID dispatch independent.
    auto launch = [&](auto id_tag) {
      using I = decltype(id_tag);
      assemble<I><<<batch, Threads, 0, stream>>>(
          values.mutable_data_ptr<float>(), candidates.mutable_data_ptr<int64_t>(),
          tiles * tile_k, ids.mutable_data_ptr<I>(), valid.mutable_data_ptr<bool>(),
          cols, k, view(sampled), map, cnt, tok, custom_stride, requests,
          custom_width, custom, vocab);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      score<scalar_t, I><<<batch, Threads, 0, stream>>>(
          x.const_data_ptr<scalar_t>(), x.stride(0), vocab, ids.const_data_ptr<I>(),
          valid.const_data_ptr<bool>(), cols, scores.mutable_data_ptr<float>(), logits_mode);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (id_type == at::kLong) launch(int64_t{});
    else launch(int32_t{});
    ranks<scalar_t><<<batch, Threads, 0, stream>>>(
        x.const_data_ptr<scalar_t>(), x.stride(0), vocab, view(sampled),
        rank.mutable_data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {ids, scores, rank};
}
}  // namespace

TORCH_LIBRARY(topk_scores_cuda, m) {
  m.def("compute(Tensor logits, Tensor sampled, int k, int max_custom, bool logits_mode, "
        "Tensor? mapping=None, Tensor? counts=None, Tensor? custom_ids=None) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(topk_scores_cuda, CUDA, m) { m.impl("compute", &compute); }
