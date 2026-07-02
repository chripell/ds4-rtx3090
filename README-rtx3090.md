# DwarfStar (ds4) - RTX 3090 RAM-Streaming Optimization Guide

This repository contains local optimizations specifically tailored for running the **DeepSeek-V4-Flash** model in high-speed RAM-streaming mode using a PC with **128GB CPU RAM** and a dedicated **24GB RTX 3090 GPU**.

---

## What Has Been Done

1. **Direct PCIe DMA Streaming (`DS4_CUDA_STREAM_FROM_RAM=1`)**:
   We modified the CUDA backend (`ds4_cuda.cu`) to support a custom `DS4_CUDA_STREAM_FROM_RAM` environment variable. When set, it bypasses the file descriptor `pread` staging loop (which uses slow system calls and user-to-kernel double copying) and executes direct `cudaMemcpy(..., HostToDevice)` copies from page-locked host memory.
2. **MAP_SHARED Model Mapping on Linux**:
   We modified the GGUF model loader (`ds4.c`) to use `MAP_SHARED` instead of `MAP_PRIVATE` when running on Linux. This ensures that the memory mapped region has a stable, directly indexable physical backing that supports CUDA host page pinning (`cudaHostRegister`).
3. **Weight Cache Chunk Reduction (`DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256`)**:
   We changed the CUDA weight arena chunk size to allocate in `256MB` increments instead of the default `1.75GB`. This resolves critical VRAM fragmentation OOMs on 24GB GPUs when compiling and executing dense graph operations with large context sizes.

---

## 3 Optimal Invocation Profiles (RTX 3090)

Ensure the following environment variables are set in your shell before launching any invocation:
```bash
export CUDA_VISIBLE_DEVICES=0
export DS4_CUDA_STREAM_FROM_RAM=1
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256
```

### Profile A: Maximum Generation Speed (Low Latency / Chat)
* **Goal**: Maximize token generation throughput.
* **Context Size**: `32768` (32K tokens)
* **Expert Cache Size**: `8GB` *(keeps $\approx 1,213$ active experts resident on GPU to minimize PCIe cache misses)*
* **Prefill Chunk Size**: `1024`
* **VRAM Usage**: $\approx 19.2\text{ GB}$
* **Performance Metrics (Measured)**:
  * **Prefill Speed**: **`6.42 t/s`**
  * **Generation Speed**: **`9.04 t/s`**
* **Command**:
  ```bash
  export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=4

  ./ds4 \
    -m ./ds4flash.gguf \
    --ssd-streaming \
    --ssd-streaming-cache-experts 8GB \
    --prefill-chunk 1024 \
    --ctx 32768
  ```

---

### Profile B: Large Context Capacity (Heavy Coding / Long Documents)
* **Goal**: Balance context length with active generation speed.
* **Context Size**: `150000` (150K tokens)
* **Expert Cache Size**: `4GB` *(holds $\approx 606$ active experts, freeing 4GB VRAM for the KV cache)*
* **Prefill Chunk Size**: `4096` *(maximizes prompt ingestion rate)*
* **VRAM Usage**: $\approx 22.0\text{ GB}$ *(close to 100% RTX 3090 capacity)*
* **Performance Metrics (Measured)**:
  * **Prefill Speed**: **`7.02 t/s`**
  * **Generation Speed**: **`7.46 t/s`**
* **Command**:
  ```bash
  export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=4

  ./ds4 \
    -m ./ds4flash.gguf \
    --ssd-streaming \
    --ssd-streaming-cache-experts 4GB \
    --prefill-chunk 4096 \
    --ctx 150000
  ```

---

### Profile C: Extreme Context Capacity (Retrieval / Massive Log Parsing)
* **Goal**: Push context length to its absolute limit while keeping a minimal expert cache.
* **Context Size**: `535000` (535K tokens)
* **Expert Cache Size**: `4GB` *(holds $\approx 606$ active experts)*
* **Prefill Chunk Size**: `1024` *(kept at 1024; setting to 2048 at this context size adds 1.02GB of VRAM and causes OOM)*
* **VRAM Usage**: $\approx 22.8\text{ GB}$
* **Performance Metrics (Measured)**:
  * **Prefill Speed**: **`6.96 t/s`**
  * **Generation Speed**: **`7.23 t/s`**
* **Command**:
  ```bash
  export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=4

  ./ds4 \
    -m ./ds4flash.gguf \
    --ssd-streaming \
    --ssd-streaming-cache-experts 4GB \
    --prefill-chunk 1024 \
    --ctx 535000
  ```

---

## Caching Tradeoffs & Analysis

1. **Prefill Processing (Prompt Ingestion)**:
   * Setting `--prefill-chunk 4096` speeds up prompt ingestion compared to `1024` (improving prefill from **`6.42 t/s`** to **`7.02 t/s`**). This is because executing larger matrix-multiplications in parallel allows the GPU to achieve higher computational density.
2. **Generation Speed (Cache Size Impact)**:
   * Having a larger **8GB** cache yields **`9.04 t/s`** during decode generation.
   * Shrinking the cache to **4GB** drops generation speed to **`7.46 t/s`** (a **17.5% slowdown**).
   * *Conclusion*: This directly demonstrates that a smaller GPU expert cache leads to more cache misses, forcing the engine to wait for missing expert weights to stream from CPU RAM over PCIe during each token generation phase.
3. **Generation Speed (Context Scaling Impact)**:
   * Increasing the context size from `150K` to `535K` under the same 4GB cache size results in a minor generation speed decrease (from **`7.46 t/s`** to **`7.23 t/s`**). This is due to the natural scaling attention overhead of tracking a larger KV cache history.
