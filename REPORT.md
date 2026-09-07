# Why GPU mode is slower

Re-measured 2026-09-04 on a desktop workstation — a current multi-core x86 CPU,
a current NVIDIA GPU on CUDA, NVMe storage. 1 661 frames of 2048 × 2448 RGB BMP,
24.4 GB in total, every mode back to back in one session. Below, *N* is the
machine's physical core count.

| Mode | img/s | vs CPU |
|---|---|---|
| **CPU**, best width | **377.0** | 1.00× |
| CPU, *N* processes (the default) | 376.6 | 1.00× |
| GPU, CUDA, batch 1 | 158.7 | 0.42× |
| GPU, CUDA, batch 32 | 112.7 | 0.30× |

Run-to-run spread is 1.00–1.10×, so both GPU rows are real losses, not ties.
**GPU mode never wins.**


## Batching makes it worse, not better

Full width × batch cross, same 1 661 frames:

| img/s | batch 1 | 4 | 16 | 32 | 64 | 1 → 64 |
|---|---|---|---|---|---|---|
| width 1 (serial) | **99.6** | 76.9 | 70.9 | 62.5 | 47.4 | 2.10× worse |
| width 8 (default) | **158.7** | 140.2 | 135.3 | 112.7 | 112.5 | 1.41× worse |
| width 16 | **148.1** | 141.5 | 128.2 | 120.1 | 102.8 | 1.44× worse |

Batch 1 is the fastest column at **every** width — the opposite of the usual
"batch it and the GPU gets fast" expectation, and the reason the default batch
size is **1**. The degradation is steepest with no threads to hide it behind,
which is what a staging-traffic explanation predicts.

## Why

Two measured facts, together:

GPU mode underperforms because the workload is bound by host memory bandwidth 
rather than compute. Since the GPU-offloaded luma step accounts for only ~17% 
of total processing (capping theoretical speedup at 1.21× under Amdahl’s law), 
offloading it adds substantial memory overhead—staging buffers, pinned memory 
copies, and PCIe transfers—that directly worsens host bus saturation. Compounding 
this, the CPU path benefits from true multi-process concurrency without GIL 
contention, while the GPU path relies on threaded coordination within a single 
process, making the round-trip data movement costs easily outweigh any 
device-side compute gains.

## When a GPU would win

| Condition | Why it flips |
|---|---|
| Heavier per-pixel work — resize, normalise, tile, augment, several ops fused | The 17 % becomes most of the cost, so Amdahl stops binding |
| Compressed input (JPEG/PNG) | Decode becomes the dominant cost and can move to the device |
| Data already resident in VRAM (part of a training pipeline) | The round trip disappears entirely |
| Much smaller frames | Per-frame host traffic stops dominating the memory path |

None of those describe a three-coefficient dot product on uncompressed 15 MB
frames. GPU mode is kept in the app for exactly this comparison.


