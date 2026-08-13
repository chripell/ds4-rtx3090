# DwarfStar (ds4) - RTX 3090 RAM-Streaming Guide

Optimized configuration for running **DeepSeek-V4-Flash** on a single **RTX 3090 (24GB)** with RAM streaming over PCIe.

---

## 1. System Pre-checks

Before running the engine, ensure your system allows pinning the memory required for direct PCIe streaming:

* **Locked Memory Limit (`ulimit -l`)**: Must be set to `unlimited`. This is required so the CUDA driver can pin the 80GB mapped model in system RAM. 
  * Check current limit: `ulimit -l`
  * Set temporarily: `ulimit -l unlimited` (or edit `/etc/security/limits.conf` for persistence).
* *(Note: Disabling the IOMMU via kernel parameters is generally not necessary as long as the memory is successfully pinned via anonymous mappings.)*

---

## 2. Environment Setup

Set these environment variables in your shell before running `ds4` or `ds4-bench`:

```bash
export CUDA_VISIBLE_DEVICES=0
export DS4_ANON_MMAP=1
export DS4_MTP_SPEC_DISABLE=1
export DS4_CUDA_STREAM_FROM_RAM=1
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256
export DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512
export DS4_CUDA_MMQ=0
export DS4_CUDA_NO_WINDOW_ATTENTION=1
```

### What these flags do:
* **`DS4_ANON_MMAP=1`**: Loads the 80GB model into anonymous memory at startup to bypass Linux kernel `cudaHostRegister` pinning restrictions on file-backed mappings.
* **`DS4_MTP_SPEC_DISABLE=1`**: Disables Speculative Decoding/MTP batching. This restores full single-token throughput on single-GPU systems.
* **`DS4_CUDA_STREAM_FROM_RAM=1`**: Bypasses system call file reads and streams expert weights directly from the pinned RAM over PCIe.
* **`DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256`**: Allocates weight memory in 256MB chunks to prevent VRAM fragmentation OOMs on 24GB cards.
* **`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512`**: Lowers the cache safety floor so intermediate weight allocations do not hit premature out-of-memory guards.
* **`DS4_CUDA_MMQ=0`**: Disables the heavy multi-hundred-megabyte prefill MMQ weights per layer to conserve VRAM for KV cache.
* **`DS4_CUDA_NO_WINDOW_ATTENTION=1`**: Uses the stable attention prefill kernels across all sequence lengths.

---

## 2. Running Inference (`ds4`)

```bash
./ds4 \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 21GB \
  --prefill-chunk 1024 \
  --ctx 32768 \
  -p "Write a Python function to reverse a string."
```

*(For interactive chat mode, simply omit `-p "..."`)*

---

## 3. Running Benchmarks (`ds4-bench`)

```bash
./ds4-bench \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 21GB \
  --prefill-chunk 1024 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 4096 \
  --step-incr 2048 \
  --gen-tokens 128
```
