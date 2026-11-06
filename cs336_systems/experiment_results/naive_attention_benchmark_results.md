# Problem (pytorch_attention): Naive Attention Benchmarking

## Machine

| Item | Spec |
|------|------|
| GPU | **1 × NVIDIA RTX 40-series 48GB**（实例标注：48G RTX40系 × 1） |
| Usable VRAM (from OOM / `nvidia`-style report) | **~47.40 GiB** |
| CPU | 20 cores |
| System RAM | 90 GB |
| Region | 上海二C |
| Image | `cuda128_torch291_py312`（CUDA 12.8, PyTorch 2.9.1, Python 3.12） |

## Settings

- `batch_size=8`, `dtype=float32`, no multi-head
- warmup=5, **100** forward + **100** backward
- Implementation: `cs336_basics.model.scaled_dot_product_attention` (naive; not FlashAttention)
- Timer: CUDA events + `torch.cuda.synchronize`
- Memory: `torch.cuda.memory_allocated` immediately before the first measured backward

## Results (20 configurations)

| head_dim | seq_len | Forward mean (ms) | Backward mean (ms) | Mem before bwd (MiB) | Status |
|---------:|--------:|------------------:|-------------------:|---------------------:|:------:|
| 16 | 256 | 0.2226 | 0.4881 | 20.90 | OK |
| 16 | 1024 | 0.3427 | 0.9520 | 82.84 | OK |
| 16 | 4096 | 7.0206 | 17.6335 | 1050.62 | OK |
| 16 | 8192 | 27.7468 | 69.6521 | 4133.00 | OK |
| 16 | 16384 | — | — | — | **OOM** |
| 32 | 256 | 0.2185 | 0.4910 | 21.52 | OK |
| 32 | 1024 | 0.3485 | 1.0538 | 85.34 | OK |
| 32 | 4096 | 7.0191 | 17.7289 | 1060.62 | OK |
| 32 | 8192 | 27.7665 | 69.9058 | 4153.00 | OK |
| 32 | 16384 | — | — | — | **OOM** |
| 64 | 256 | 0.2204 | 0.4780 | 22.77 | OK |
| 64 | 1024 | 0.3506 | 1.0600 | 90.34 | OK |
| 64 | 4096 | 7.0849 | 17.7382 | 1080.62 | OK |
| 64 | 8192 | 27.9183 | 69.8352 | 4193.00 | OK |
| 64 | 16384 | — | — | — | **OOM** |
| 128 | 256 | 0.2241 | 0.4832 | 25.27 | OK |
| 128 | 1024 | 0.3772 | 1.0749 | 100.34 | OK |
| 128 | 4096 | 7.3977 | 18.2816 | 1120.62 | OK |
| 128 | 8192 | 29.2463 | 71.4356 | 4273.00 | OK |
| 128 | 16384 | — | — | — | **OOM** |

OOM detail (all four `seq_len=16384` runs): CUDA tried to allocate an additional **~8 GiB** while the process already held ~40+ GiB; free memory was only ~6.7–6.9 GiB.

## Brief conclusions

1. **OOM 边界**：在这张约 48GB（可用 ~47.4 GiB）的 RTX 40 系卡上，所有 `seq_len=16384` 都 OOM；`8192` 仍可跑完。瓶颈几乎不随 `head_dim` 改变——四个 head dim 都在同一 seq 处爆掉。

2. **显存随序列长度近似按 \(N^2\) 涨**：例如 `head_dim=16` 时，backward 前显存从 4096→8192（\(N\) 翻倍）大约 1051→4133 MiB（约 4×），符合 naive attention 物化 \(B\times N\times N\) 分数/概率矩阵的行为。`head_dim` 只带来很小的额外占用（Q/K/V/O 的 \(O(BND)\)）。

3. **时间同样被长序列主导**：同一 `head_dim` 下，4096→8192 时 forward/backward 都大幅变慢；同一 `seq_len` 下换 `head_dim`（16→128）时间变化相对很小。Backward 大约是 forward 的 **2–2.5×**。

4. **含义**：naive attention 的训练显存大头是为 backward 保存的 \(N\times N\) 激活；要消掉这笔成本，需要避免物化完整 attention 矩阵（例如 FlashAttention 的 tiling / online softmax），或配合 checkpointing 等手段用算力换显存。

## Note for writeup

Handout 还要求对「最小 OOM 配置」做更完整的 memory accounting（可用 Assignment 1 的公式）。上面表格已给出实测；正式段落里建议补上 \(B N^2\) 量级的估算与实测对比。
