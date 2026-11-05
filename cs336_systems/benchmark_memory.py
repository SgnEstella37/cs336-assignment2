"""End-to-end benchmarking for the CS336 basics Transformer.

Supports FP32, explicitly BF16 parameters, and BF16 autocast with FP32 parameters.
"""

import argparse
import statistics
from contextlib import nullcontext
from timeit import default_timer
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.annotated_attention import annotated_scaled_dot_product_attention


MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CS336 Transformer")

    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "full_step"],
        default="forward",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)

    # Optional overrides for the predefined model sizes.
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use BF16 autocast while keeping model parameters in FP32.",
    )

    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Record one CUDA memory snapshot after warmup (instead of timing steps).",
    )
    parser.add_argument(
        "--memory-output",
        type=str,
        default=None,
        help="Snapshot output path; defaults to profiles/memory_<config>.pickle.",
    )
    parser.add_argument(
        "--memory-max-entries",
        type=int,
        default=100000,
        help="Maximum number of CUDA allocation events to record.",
    )

    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be >= 0 and --steps must be >= 1")
    if min(args.batch_size, args.context_length, args.vocab_size) < 1:
        parser.error("Batch size, context length, and vocab size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use an NVIDIA GPU.")
    if args.profile_memory and args.device != "cuda":
        parser.error("--profile-memory requires --device cuda")
    if args.memory_max_entries < 1:
        parser.error("--memory-max-entries must be positive")
    if args.mixed_precision and args.device != "cuda":
        parser.error("--mixed-precision requires --device cuda")
    if args.mixed_precision and args.dtype != "float32":
        parser.error("Use --dtype float32 together with --mixed-precision")
    return args


def synchronize(device):
    """Wait until previously submitted CUDA work finishes."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_step(model, optimizer, x, targets, mode, mixed_precision=False, stage_callback=None):
    """Run one step, optionally enabling BF16 autocast for forward and loss."""
    if mode != "forward":
        optimizer.zero_grad(set_to_none=True)

    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if mixed_precision
        else nullcontext()
    )

    # Autocast is used for forward/loss, not for backward or optimizer.
    with precision_context:
        with nvtx.range("forward"):
            logits = model(x)
        if stage_callback is not None:
            stage_callback("forward")

        if mode != "forward":
            with nvtx.range("loss"):
                loss = F.cross_entropy(logits.transpose(1, 2), targets)
            if stage_callback is not None:
                stage_callback("loss")

    if mode != "forward":
        with nvtx.range("backward"):
            loss.backward()
        if stage_callback is not None:
            stage_callback("backward")

    if mode == "full_step":
        with nvtx.range("optimizer"):
            optimizer.step()
        if stage_callback is not None:
            stage_callback("optimizer")

    # A dtype is only metadata; printing it outside the timed loop avoids
    # print overhead in the measurements.
    return logits.dtype



def profile_memory(model, optimizer, x, targets, args, device):
    """Record one fully synchronized step and write its CUDA memory timeline.

    The pre-step warmup happens before recording. Stage statistics are printed
    in order; the memory_viz x-axis does not automatically show phase labels.
    """
    precision_label = "bf16_autocast" if args.mixed_precision else args.dtype
    output = Path(args.memory_output) if args.memory_output else Path(
        "profiles"
    ) / f"memory_{args.model_size}_ctx{args.context_length}_{args.mode}_{precision_label}.pickle"
    output.parent.mkdir(parents=True, exist_ok=True)

    stage_stats = []

    def mark_stage(name):
        synchronize(device)
        mib = 1024 ** 2
        stage_stats.append(
            (
                name,
                torch.cuda.memory_allocated(device) / mib,
                torch.cuda.max_memory_allocated(device) / mib,
            )
        )
        # The next stage's peak is measured starting at this boundary.
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Private PyTorch API, as used in the assignment; may vary by version.
    torch.cuda.memory._record_memory_history(max_entries=args.memory_max_entries)
    try:
        with nvtx.range("measurement"):
            logits_dtype = run_step(
                model, optimizer, x, targets, args.mode,
                mixed_precision=args.mixed_precision,
                stage_callback=mark_stage,
            )
            synchronize(device)
        torch.cuda.memory._dump_snapshot(str(output))
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Parameter dtype: {next(model.parameters()).dtype}")
    print(f"Logits dtype: {logits_dtype}")
    print("CUDA allocator memory at the end of each phase:")
    for name, allocated_mib, peak_mib in stage_stats:
        print(
            f"  {name:>9}: active={allocated_mib:.1f} MiB, "
            f"peak since previous stage={peak_mib:.1f} MiB"
        )
    print(f"Memory snapshot: {output.resolve()}")
    print("Open the .pickle file at https://pytorch.org/memory_viz")

def main():
    args = parse_args()
    torch.manual_seed(0)

    config = MODEL_CONFIGS[args.model_size].copy()
    for key in ("d_model", "d_ff", "num_layers", "num_heads"):
        override = getattr(args, key)
        if override is not None:
            config[key] = override

    if any(value < 1 for value in config.values()):
        raise ValueError("All model hyperparameters must be positive")
    if config["d_model"] % config["num_heads"] != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if (config["d_model"] // config["num_heads"]) % 2 != 0:
        raise ValueError("The attention head dimension must be even for RoPE")

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    # With --mixed-precision, args.dtype must be float32, so the stored
    # model parameters stay FP32 while eligible operations use BF16.
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device=device, dtype=dtype)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Reuse the same random batch; data generation is not timed.
    x = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    precision_name = (
        "BF16 autocast (FP32 parameters)"
        if args.mixed_precision
        else f"No autocast ({args.dtype} parameters)"
    )
    print(f"Model: {args.model_size} {config}")
    print(f"Mode: {args.mode} | Device: {device} | Precision: {precision_name}")
    print(f"Warm-up: {args.warmup} | Measured steps: {args.steps}")

    # Preserve NVTX annotations used for the earlier Nsight experiments.
    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    with nvtx.range("warmup"):
        for _ in range(args.warmup):
            run_step(
                model, optimizer, x, targets, args.mode,
                mixed_precision=args.mixed_precision,
            )
            synchronize(device)

    if args.profile_memory:
        profile_memory(model, optimizer, x, targets, args, device)
        return

    times_ms = []
    last_logits_dtype = None

    with nvtx.range("measurement"):
        for _ in range(args.steps):
            synchronize(device)
            start = default_timer()
            last_logits_dtype = run_step(
                model, optimizer, x, targets, args.mode,
                mixed_precision=args.mixed_precision,
            )
            synchronize(device)
            times_ms.append((default_timer() - start) * 1000)

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    print(f"Parameter dtype: {next(model.parameters()).dtype}")
    print(f"Logits dtype: {last_logits_dtype}")
    print(f"Step times (ms): {[round(t, 2) for t in times_ms]}")
    print(f"Mean: {mean_ms:.2f} ms | Std: {std_ms:.2f} ms")


if __name__ == "__main__":
    main()
