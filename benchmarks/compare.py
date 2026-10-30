"""Run: python -m benchmarks.compare --output results/h100.json

Install the locked vLLM checkout in the same CUDA-enabled Python environment.
The extension builds before any timing. Default matrix includes FP32 because
the actual sampler may supply FP32 logits. --vocab accepts model vocabulary sizes.
"""

import argparse
import hashlib
import itertools
import json
import platform
import statistics
import subprocess
from pathlib import Path

import torch

from topk_cuda import compute_topk_scores_cuda, load_extension
from topk_cuda.validation import compare_outputs, original_function


def measure(fn, args, kwargs, iterations):
    start, stop = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iterations):
        result = fn(*args, **kwargs)
    stop.record()
    stop.synchronize()
    # Keep the final output alive until the timing boundary.
    del result
    return start.elapsed_time(stop) * 1000 / iterations


def command_output(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument(
        "--vocab", type=int, nargs="+", default=[32000, 64000, 128000, 256000]
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--logits-mode", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations and repeats must be positive")
    if min(args.batch + args.vocab) < 1 or min(args.k) < 0:
        parser.error("invalid workload dimensions")
    if not torch.cuda.is_available():
        parser.error("CUDA device required; no performance results produced")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if args.require_h100 and "H100" not in props.name:
        parser.error(f"H100 required, found {props.name}")
    original = original_function()
    load_extension()
    root = Path(__file__).resolve().parents[1]
    report = {
        "baseline": json.loads((root / "baseline.lock.json").read_text()),
        "environment": {
            "gpu": props.name,
            "capability": [props.major, props.minor],
            "memory_bytes": props.total_memory,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "nvcc": command_output(["nvcc", "--version"]),
            "nvidia_smi": command_output(["nvidia-smi"]),
        },
        "implementation_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                root / "csrc/scores.cu",
                root / "topk_cuda/__init__.py",
                Path(__file__).resolve(),
            ]
        },
        "settings": {
            **vars(args),
            "output": str(args.output),
            "timing": "eager full entry point, CUDA events, alternating order",
            "stream": "same dedicated stream for both paths",
            "seed": 401,
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = torch.cuda.Stream()
    for batch, vocab, k, dtype in itertools.product(
        args.batch, args.vocab, args.k, args.dtype
    ):
        if k > vocab:
            continue
        torch.manual_seed(401)
        with torch.cuda.stream(stream), torch.inference_mode():
            x = torch.randn(batch, vocab, device="cuda", dtype=getattr(torch, dtype))
            sampled = torch.randint(vocab, (batch,), device="cuda")
            inputs = (x, k, sampled)
            kwargs = {"logits_mode": args.logits_mode}
            compare_outputs(
                x,
                original(*inputs, **kwargs),
                compute_topk_scores_cuda(*inputs, **kwargs),
            )
            for _ in range(args.warmup):
                original(*inputs, **kwargs)
                compute_topk_scores_cuda(*inputs, **kwargs)
            stream.synchronize()
            samples = [[], []]
            functions = [original, compute_topk_scores_cuda]
            for repeat in range(args.repeats):
                for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                    samples[index].append(
                        measure(functions[index], inputs, kwargs, args.iterations)
                    )
        old, new = [statistics.median(sample) for sample in samples]
        row = {
            "B": batch,
            "V": vocab,
            "K": k,
            "dtype": dtype,
            "original_us": old,
            "cuda_us": new,
            "speedup": old / new,
            "original_samples_us": samples[0],
            "cuda_samples_us": samples[1],
            "correctness": "passed (tie-aware)",
        }
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"B={batch} V={vocab} K={k} {dtype}: original={old:.3f} us "
            f"CUDA={new:.3f} us speedup={old / new:.3f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
