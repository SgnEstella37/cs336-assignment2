# Benchmarking Results (Problem 2.1.3)

Machine: NVIDIA GeForce RTX 4090  
Defaults: `batch_size=4`, `context_length=512`, `vocab_size=10000`, `dtype=float32`  
Timer: `timeit.default_timer` + `torch.cuda.synchronize()` after each step

## (b) Warmup = 5, Steps = 10

### small

| Mode | Mean (ms) | Std (ms) |
|------|-----------|----------|
| forward | 24.46 | 0.10 |
| forward_backward | 90.10 | 0.14 |
| full_step | 101.71 | 0.41 |

Estimated breakdown (by subtraction; % of `full_step`):

| Stage | Estimate (ms) | Share of full_step | How |
|-------|---------------|--------------------|-----|
| forward | 24.46 | 24.0% | measured |
| backward | 65.64 | 64.5% | forward_backward − forward |
| optimizer | 11.61 | 11.4% | full_step − forward_backward |
| **full_step** | **101.71** | **100%** | measured |

Raw step times:

- forward: `[24.68, 24.47, 24.47, 24.47, 24.48, 24.47, 24.39, 24.33, 24.34, 24.52]`
- forward_backward: `[89.94, 89.88, 90.15, 90.24, 90.1, 90.18, 90.01, 90.3, 90.01, 90.2]`
- full_step: `[101.45, 101.26, 101.9, 101.58, 101.39, 101.54, 101.94, 102.64, 101.42, 102.0]`

### medium

| Mode | Mean (ms) | Std (ms) |
|------|-----------|----------|
| forward | 74.79 | 0.29 |
| forward_backward | 243.02 | 0.99 |
| full_step | 278.57 | 0.46 |

Estimated breakdown (by subtraction; % of `full_step`):

| Stage | Estimate (ms) | Share of full_step | How |
|-------|---------------|--------------------|-----|
| forward | 74.79 | 26.8% | measured |
| backward | 168.23 | 60.4% | forward_backward − forward |
| optimizer | 35.55 | 12.8% | full_step − forward_backward |
| **full_step** | **278.57** | **100%** | measured |

Raw step times:

- forward: `[74.59, 74.55, 74.59, 74.57, 74.56, 75.08, 74.96, 74.86, 75.4, 74.76]`
- forward_backward: `[243.14, 241.73, 242.39, 242.06, 241.69, 243.54, 244.04, 244.44, 243.68, 243.5]`
- full_step: `[278.42, 279.02, 278.72, 278.1, 279.48, 278.29, 278.2, 277.97, 278.73, 278.8]`

### large

| Mode | Mean (ms) | Std (ms) |
|------|-----------|----------|
| forward | TBD | TBD |
| forward_backward | TBD | TBD |
| full_step | TBD | TBD |

### xl

| Mode | Mean (ms) | Std (ms) |
|------|-----------|----------|
| forward | TBD | TBD |
| forward_backward | TBD | TBD |
| full_step | TBD | TBD |

### 10B

| Mode | Mean (ms) | Std (ms) |
|------|-----------|----------|
| forward | TBD | TBD |
| forward_backward | TBD | TBD |
| full_step | TBD | TBD |

Notes / OOM:

- 

## (c) Warmup ablation

Setting: `--mode forward`, `steps=10`, RTX 4090.

### small

| Warmup | Mean (ms) | Std (ms) | Notes |
|--------|-----------|----------|-------|
| 0 | 45.88 | 67.52 | first measured step is a huge outlier |
| 1 (run 1) | 24.51 | 0.15 | |
| 1 (run 2) | 24.44 | 0.07 | repeat |
| 2 | 24.44 | 0.07 | |
| 3 | 24.46 | 0.05 | |
| 5 (from b) | 24.46 | 0.10 | baseline |

Raw step times:

- warmup=0: `[238.05, 24.46, 24.8, 24.53, 24.45, 24.48, 24.51, 24.65, 24.51, 24.33]`
- warmup=1 (run 1): `[24.42, 24.45, 24.53, 24.62, 24.44, 24.43, 24.47, 24.88, 24.55, 24.34]`
- warmup=1 (run 2): `[24.48, 24.46, 24.47, 24.47, 24.45, 24.51, 24.31, 24.33, 24.37, 24.51]`
- warmup=2: `[24.46, 24.42, 24.51, 24.45, 24.48, 24.51, 24.49, 24.33, 24.38, 24.34]`
- warmup=3: `[24.48, 24.48, 24.48, 24.48, 24.54, 24.47, 24.5, 24.4, 24.43, 24.37]`

### medium

| Warmup | Mean (ms) | Std (ms) | Notes |
|--------|-----------|----------|-------|
| 0 | 94.29 | 62.31 | first measured step is a huge outlier |
| 1 | 74.53 | 0.39 | |
| 2 | 74.57 | 0.20 | |
| 5 (from b) | 74.79 | 0.29 | baseline |

Raw step times:

- warmup=0: `[271.62, 74.37, 75.15, 74.36, 74.36, 74.34, 74.54, 74.32, 75.03, 74.86]`
- warmup=1: `[74.33, 74.36, 74.43, 74.59, 74.36, 74.5, 74.37, 74.41, 74.36, 75.61]`
- warmup=2: `[74.39, 74.7, 75.02, 74.44, 74.73, 74.53, 74.49, 74.64, 74.39, 74.41]`

## Writeup scratch

### (b) 1–2 sentences

On RTX 4090, the small model takes 24.46±0.10 ms for forward, about 65.6 ms for backward, and about 11.6 ms for the optimizer step (~24% / 65% / 11% of a full step). Medium takes 74.79 / 168.23 / 35.55 ms (~27% / 60% / 13%); stds remain small relative to the means.

### (c) 2–3 sentences

With zero warm-up, the mean/std blow up because the first measured step is much slower (small: ~238 ms vs ~24.5 ms steady-state; medium: ~272 ms vs ~74.5 ms), consistent with one-time CUDA/cuBLAS/allocator initialization. With 1–2 warm-up steps, both models already match the warmup=5 baseline within noise, so a few warm-up steps are enough to remove the cold-start bias.
