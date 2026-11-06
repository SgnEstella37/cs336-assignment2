"""Benchmark torch.compile on Assignment 1 naive attention (CS336 A2, 4.2a).

Same configurations as naive benchmark: batch=8, head dims=[16,32,64,128],
sequence lengths=[256,1024,4096,8192,16384], float32,
100 synchronized forward passes and 100 synchronized backward passes.

Run from the repository root:
    uv run python -m cs336_systems.benchmark_compiled_attention

Each configuration is measured in a separate subprocess to isolate CUDA memory
and to survive out-of-memory errors. Results are saved incrementally to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


HEAD_DIMS = (16, 32, 64, 128)
SEQ_LENS = (256, 1024, 4096, 8192, 16384)
RESULT_PREFIX = "COMPILED_ATTENTION_RESULT_JSON="
COLUMNS = (
    "batch_size",
    "head_dim",
    "seq_len",
    "dtype",
    "iterations",
    "forward_mean_ms",
    "forward_std_ms",
    "backward_mean_ms",
    "backward_std_ms",
    "memory_before_backward_allocated_mib",
    "memory_before_backward_reserved_mib",
    "peak_allocated_mib",
    "one_score_matrix_gib",
    "status",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark torch.compile CS336 Attention")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-dims", type=int, nargs="+", default=list(HEAD_DIMS))
    parser.add_argument("--seq-lens", type=int, nargs="+", default=list(SEQ_LENS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cs336_systems/compiled_attention_benchmark_results.csv"),
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=900,
        help="Maximum time per configuration; timed-out configurations are recorded.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("Batch size and iterations must be positive; warmup >= 0")
    if any(x <= 0 for x in (*args.head_dims, *args.seq_lens)):
        parser.error("All head dimensions and sequence lengths must be positive")
    if args.timeout_seconds <= 0:
        parser.error("Timeout must be positive")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device: torch.device):
    """Return (result, elapsed_ms); CUDA events measure GPU elapsed time."""
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        synchronize(device)  # Required after every forward/backward pass.
        return result, start.elapsed_time(end)

    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def mean_and_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def base_row(args: argparse.Namespace, head_dim: int, seq_len: int) -> dict:
    # One explicit (batch_size, seq_len, seq_len) FP32 attention score tensor.
    score_gib = args.batch_size * seq_len * seq_len * 4 / (1024**3)
    return {
        "batch_size": args.batch_size,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "dtype": "float32",
        "iterations": args.iterations,
        "forward_mean_ms": "",
        "forward_std_ms": "",
        "backward_mean_ms": "",
        "backward_std_ms": "",
        "memory_before_backward_allocated_mib": "",
        "memory_before_backward_reserved_mib": "",
        "peak_allocated_mib": "",
        "one_score_matrix_gib": round(score_gib, 4),
        "status": "ERROR",
        "error": "",
    }


def run_one(args: argparse.Namespace) -> dict:
    """One configuration in one process; compilation is done during warmup."""
    from cs336_basics.model import scaled_dot_product_attention

    d, n = args.head_dims[0], args.seq_lens[0]
    row = base_row(args, d, n)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine")

    # Compilation is lazy; the first warmup pass triggers forward/backward compilation.
    compiled_attention = torch.compile(scaled_dot_product_attention)

    torch.manual_seed(0)
    q, k, v = (
        torch.randn(args.batch_size, n, d, device=device, dtype=torch.float32,
                    requires_grad=True)
        for _ in range(3)
    )
    grad_output = torch.ones_like(q)

    def attention():
        # Compile OUR naive implementation, NOT torch.nn.functional SDPA.
        return compiled_attention(q, k, v)

    def clear_grads():
        for tensor in (q, k, v):
            tensor.grad = None

    # Warm up both passes; this also compiles forward AND backward outside timing.
    for _ in range(args.warmup):
        clear_grads()
        out = attention()
        synchronize(device)
        out.backward(grad_output)
        synchronize(device)
        del out
    clear_grads()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Time compiled forward passes with autograd enabled.
    forward_times = []
    for _ in range(args.iterations):
        out, elapsed_ms = timed_call(attention, device)
        forward_times.append(elapsed_ms)
        del out  # Don't accumulate computation graphs across iterations.
    synchronize(device)

    # Each measured backward has a fresh, untimed forward graph.
    # Q/K/V grads are cleared to prevent accumulating gradients across runs.
    backward_times = []
    for step in range(args.iterations):
        clear_grads()
        out = attention()
        synchronize(device)

        if step == 0 and device.type == "cuda":
            row["memory_before_backward_allocated_mib"] = round(
                torch.cuda.memory_allocated(device) / (1024**2), 2
            )
            row["memory_before_backward_reserved_mib"] = round(
                torch.cuda.memory_reserved(device) / (1024**2), 2
            )

        _, elapsed_ms = timed_call(lambda: out.backward(grad_output), device)
        backward_times.append(elapsed_ms)
        del out

    synchronize(device)
    if device.type == "cuda":
        row["peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 2
        )

    f_mean, f_std = mean_and_std(forward_times)
    b_mean, b_std = mean_and_std(backward_times)
    row.update(
        forward_mean_ms=round(f_mean, 4),
        forward_std_ms=round(f_std, 4),
        backward_mean_ms=round(b_mean, 4),
        backward_std_ms=round(b_std, 4),
        status="OK",
    )
    return row


def worker_main(args: argparse.Namespace) -> None:
    try:
        row = run_one(args)
    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        row = base_row(args, args.head_dims[0], args.seq_lens[0])
        row["status"] = "OOM"
        row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"[:500]
    except Exception as exc:
        row = base_row(args, args.head_dims[0], args.seq_lens[0])
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"[:500]
    print(RESULT_PREFIX + json.dumps(row, allow_nan=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args)
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No NVIDIA CUDA GPU detected. Run this benchmark on the GPU server.")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    combinations = [(d, n) for d in args.head_dims for n in args.seq_lens]
    print(f"Compiled Attention: {len(combinations)} configurations, "
          f"batch={args.batch_size}, float32, {args.iterations} passes per direction")
    print(f"CSV output: {output.resolve()}", flush=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        file.flush()

        for index, (d, n) in enumerate(combinations, start=1):
            print(f"[{index}/{len(combinations)}] head_dim={d}, seq_len={n}", flush=True)
            cmd = [
                sys.executable, "-m", "cs336_systems.benchmark_compiled_attention",
                "--worker", "--device", args.device,
                "--batch-size", str(args.batch_size),
                "--head-dims", str(d), "--seq-lens", str(n),
                "--warmup", str(args.warmup),
                "--iterations", str(args.iterations),
            ]
            row = base_row(args, d, n)
            try:
                completed = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=args.timeout_seconds, check=False,
                )
                result_line = next(
                    (line for line in completed.stdout.splitlines()
                     if line.startswith(RESULT_PREFIX)), None
                )
                if result_line is not None:
                    row = json.loads(result_line[len(RESULT_PREFIX):])
                else:
                    row["status"] = "ERROR"
                    stderr = completed.stderr.strip().splitlines()
                    row["error"] = (stderr[-1] if stderr else
                                    f"Worker exited with code {completed.returncode}")[:500]
            except subprocess.TimeoutExpired:
                row["status"] = "TIMEOUT"
                row["error"] = f"Exceeded {args.timeout_seconds}s per configuration"
            except Exception as exc:
                row["status"] = "ERROR"
                row["error"] = f"{type(exc).__name__}: {exc}"[:500]

            writer.writerow(row)
            file.flush()  # Keep previous results if a later run OOMs or is interrupted.
            print(
                f"  {row['status']} | fwd={row['forward_mean_ms']} ms | "
                f"bwd={row['backward_mean_ms']} ms | "
                f"pre-bwd allocated={row['memory_before_backward_allocated_mib']} MiB"
                + (f" | {row['error']}" if row['error'] else ""),
                flush=True,
            )

    print(f"Done. Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
