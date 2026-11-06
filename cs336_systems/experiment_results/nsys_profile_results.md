# Problem: nsys_profile

Machine: NVIDIA GeForce RTX 4090  
Defaults (unless noted): `batch_size=4`, `dtype=float32`, `warmup=5`, `steps=10`  
Profiler: Nsight Systems (`nsys`), NVTX capture on `measurement`

---

## (a) Forward Pass Runtime

| Source | Forward time |
|--------|----------------|
| Nsight Systems (NVTX `forward` avg) | **31.90 ms** |
| Earlier Python benchmark (`timeit` + `cuda.synchronize`) | **24.46 ms** |

The average forward-pass NVTX duration measured with Nsight Systems was **31.90 ms**, compared to **24.46 ms** from the earlier Python benchmark. The difference is likely due to profiling overhead and different measurement boundaries, since the NVTX range does not explicitly synchronize the GPU.

> Note for yourself: the handout also asks for **two model sizes** and **three power-of-two context lengths > 128**. If you ran those configs, add a table of nsys vs Python times here.

---

## (b) Most Time-Consuming CUDA Kernel

| Setting | Top kernel | Share of GPU kernel time | Invocations |
|---------|------------|--------------------------|-------------|
| Forward only | `ampere_sgemm_128x64_tn` | **54.7%** | **84 per forward** (840 over 10 measured steps) |
| Forward + backward | `ampere_sgemm_128x64_tn` | **14.6%** | (same family remained #1; share dropped) |

The `ampere_sgemm_128x64_tn` kernel consumed the most cumulative GPU time during the forward pass (**54.7%**) and was invoked **84 times per forward pass**. It remained the most time-consuming individual kernel during forward and backward combined, although its share decreased to **14.6%**.

---

## (c) Non-Matrix-Multiplication Kernels

Besides matrix multiplications, elementwise operations, exponential computations, and reduction kernels consumed non-trivial GPU time. For example:

| Example (forward) | Approx. share of GPU kernel time |
|-------------------|----------------------------------|
| Vectorized elementwise kernel | **5.5%** |
| Vectorized elementwise kernel | **4.5%** |
| Exponential kernel (`exp_kernel_cuda`) | **3.9%** |

---

## (d) Full Training Step vs. Forward Only

| Mode | GEMM share of cumulative GPU kernel time | Non-GEMM share |
|------|------------------------------------------|----------------|
| Forward only (inference) | **~64.58%** | **~35.42%** |
| Full step (forward → loss → backward → AdamW) | **~42.83%** | **~57.17%** |

The fraction of cumulative GPU kernel time spent on GEMM operations decreased from approximately **64.58%** during forward-only inference to **42.83%** during a full training step with AdamW. Meanwhile, non-GEMM kernels increased from **35.42% to 57.17%**, reflecting additional gradient computations and optimizer updates.

---

## (e) Softmax vs. Matrix Multiplication in Attention

Within self-attention, across **10 forward passes**:

| Op class | Approx. total GPU time |
|----------|------------------------|
| Softmax-related kernels | **~27.23 ms** |
| Two matrix multiplications (combined) | **~15.63 ms** |

Despite requiring roughly **50× fewer FLOPs**, softmax took longer than the attention matmuls due to less efficient data reuse, reductions, exponentials, and intermediate memory operations.
