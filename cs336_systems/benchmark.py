"""End-to-end benchmarking for the CS336 basics Transformer."""

import torch.cuda.nvtx as nvtx

import cs336_basics.model as basics_model

from cs336_systems.annotated_attention import (
    annotated_scaled_dot_product_attention,
)

import argparse
import statistics
from timeit import default_timer
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


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
    help="Enable BF16 autocast while keeping model parameters in FP32",
)

    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be >= 0 and --steps must be >= 1")
    if min(args.batch_size, args.context_length, args.vocab_size) < 1:
        parser.error("Batch size, context length, and vocab size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use an NVIDIA GPU, or --device cpu for a smoke test")
    return args


def synchronize(device):
    """Wait until previously submitted CUDA work finishes."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def run_step(model, optimizer, x, targets, mode, mixed_precision=False):
    """Run one benchmark step with optional BF16 autocast."""

    if mode != "forward":
        optimizer.zero_grad(set_to_none=True)

    if mixed_precision:
        precision_context = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    else:
        precision_context = nullcontext()

    with precision_context:
        # Forward
        with nvtx.range("forward"):
            logits = model(x)

        if mode != "forward":
            # Loss
            with nvtx.range("loss"):
                loss = F.cross_entropy(
                    logits.transpose(1, 2), targets
                )

    # Backward
    if mode != "forward":
        with nvtx.range("backward"):
            loss.backward()

    # Optimizer
    if mode == "full_step":
        with nvtx.range("optimizer"):
            optimizer.step()



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
    if args.mixed_precision:
        if args.device != "cuda":
            raise ValueError("BF16 mixed precision requires CUDA")
        if args.dtype != "float32":
            raise ValueError("Use --dtype float32 with --mixed-precision")

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device=device, dtype=dtype)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Same random batch is reused, so data generation is not timed.
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

    print(f"Model: {args.model_size} {config}")
    print(f"Mode: {args.mode} | Device: {device} | Dtype: {args.dtype}")
    print(f"Warm-up: {args.warmup} | Measured steps: {args.steps}")

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # Warm-up: execute but do not record times.
    with nvtx.range("warmup"):
        for _ in range(args.warmup):
            run_step(model, optimizer, x, targets, args.mode)
            synchronize(device)

    # Measure each complete step, waiting for the GPU to finish.
    times_ms = []
    with nvtx.range("measurement"):
        for _ in range(args.steps):
            synchronize(device)
            start = default_timer()
            run_step(model, optimizer, x, targets, args.mode)
            synchronize(device)
            times_ms.append((default_timer() - start) * 1000)

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    print(f"Step times (ms): {[round(t, 2) for t in times_ms]}")
    print(f"Mean: {mean_ms:.2f} ms | Std: {std_ms:.2f} ms")


if __name__ == "__main__":
    main()
