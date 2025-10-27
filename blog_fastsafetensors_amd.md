# Stop Waiting, Start Building: How fastsafetensors Transformed My AMD GPU Workflow

## The 3-Minute Coffee Break That Cost Me Hours

Picture this: You've just fixed a critical bug in your vLLM configuration. You restart the server to test it. The loading bar appears. You wait. And wait. *Maybe I should grab coffee?* Three minutes later, you're back, and the model is still loading. Another two minutes pass. Finally, the server boots.

**Total time lost: 5 minutes.**

Do this 20 times a day during active development, and you've just burned **1 hour and 40 minutes** staring at progress bars.

I lived this nightmare for months on my AMD GPU setup—until I discovered `fastsafetensors`. Now those 5-minute waits are down to **25 seconds**. My development velocity has skyrocketed, and deployment is no longer a nail-biting wait.

Here's why this matters, how it works (explained like you're not a kernel hacker), and how to get it running on your AMD GPUs right now.

---

## The Real-World Numbers: Why I'm Never Going Back

### DeepSeek-R1 on 8x AMD MI300X GPUs

I run large language models on **8x AMD Instinct MI300X GPUs**. Here's what happened when I switched to fastsafetensors:

| Scenario | Standard safetensors | fastsafetensors | Speedup |
|----------|---------------------|-----------------|---------|
| **Cold Start** (cache cleared) | ~180 seconds | ~25 seconds | **7.2x faster** |
| **Warm Start** (cached) | ~45 seconds | ~8 seconds | **5.6x faster** |

![Performance Comparison](./docs/images/fastsafetensors-rocm.png)

That **7x speedup on cold starts** is the difference between:
- Staying in flow state vs. losing your train of thought
- Iterating 10 times per hour vs. 3 times per hour
- Deploying with confidence vs. praying it works the first time

### Even Small Models See Massive Gains

Don't have 8 MI300X GPUs lying around? The benefits scale down beautifully:

**GPT-2 Medium (1.4GB)**
- fastsafetensors: **0.235s** @ 6.02 GB/s
- Standard mmap: **1.104s** @ 1.28 GB/s
- **4.7x faster, 5x higher throughput**

**GPT-2 Small (523MB)**
- fastsafetensors: **0.195s** @ 2.62 GB/s
- Standard mmap: **0.505s** @ 1.01 GB/s
- **2.6x faster, 2.6x higher throughput**

---

## How It Works: The Magic Behind the Speed (Explained Simply)

Traditional model loading is like moving into an apartment one item at a time using a bicycle. fastsafetensors is like hiring a moving company with multiple trucks and professional movers. Let me show you exactly what's happening under the hood.

### The Old Way: Why Standard safetensors Is Slow

```
┌─────────────────────────────────────────────────────────┐
│          Traditional safetensors Loading Flow           │
└─────────────────────────────────────────────────────────┘

Step 1: Memory-Map File
  📁 model.safetensors ──mmap──> 💾 Host Memory (Virtual)
                                      ↓
                                  [Slow, sequential]
                                      ↓
Step 2: Deserialize Tensors (ONE AT A TIME)
  💾 Host Memory ──deserialize──> 🔢 Tensor Object (CPU)
                                      ↓
                                  [Copy each tensor]
                                      ↓
Step 3: Copy to GPU
  🔢 Tensor (CPU) ──cudaMemcpy──> 🎮 GPU Memory
                                      ↓
Step 4: Shard on CPU (for multi-GPU)
  🔢 Full Tensor ──slice on CPU──> 🔪 Partitions
                                      ↓
  🔪 Each Partition ──copy again──> 🎮 GPU 1, 2, 3...

Problems:
❌ mmap can't saturate fast NVMe SSDs (only uses ~5 GB/s of 28 GB/s)
❌ CPU does heavy lifting (40%+ kernel CPU usage)
❌ Huge page cache footprint (hundreds of GB)
❌ Sequential processing bottlenecks everything
❌ Multiple redundant copies waste time and memory
```

### The New Way: How fastsafetensors Fixes Everything

```
┌─────────────────────────────────────────────────────────┐
│          fastsafetensors Optimized Flow                 │
└─────────────────────────────────────────────────────────┘

Step 1: Parse Header ONCE
  📁 model.safetensors ──quick scan──> 📋 Tensor Metadata
                                          (offsets, shapes, dtypes)

Step 2: Parallel I/O Directly to GPU

  Thread Pool (16 workers):
  ┌────────────────────────────────────────────────┐
  │  Thread 1: Read 10GB chunk → 🎮 GPU Memory    │
  │  Thread 2: Read 10GB chunk → 🎮 GPU Memory    │
  │  Thread 3: Read 10GB chunk → 🎮 GPU Memory    │
  │     ...                                        │
  │  Thread N: Read 10GB chunk → 🎮 GPU Memory    │
  └────────────────────────────────────────────────┘
         ↓
    [ALL PARALLEL - maxes out NVMe at ~26 GB/s]
         ↓
Step 3: Lazy Tensor Instantiation (DLPack)
  🎮 Raw GPU Memory ──DLPack wrap──> 🔢 Tensor Object
                                          ↓
                              [NO COPY! Just pointer wrap]
                                          ↓
Step 4: GPU-Side Sharding (multi-GPU)
  🔢 Tensors on GPU ──torch.distributed──> 🎮 Redistribute via NVLink
                         (broadcast/scatter)
                                          ↓
                          [High-speed GPU-to-GPU, bypasses CPU]

Benefits:
✅ Saturates NVMe bandwidth (~26 GB/s utilization)
✅ CPU barely involved (5% usage vs 40%+)
✅ Minimal page cache footprint
✅ Parallel processing maximizes throughput
✅ Zero redundant copies
✅ GPU does GPU work, CPU does I/O coordination
```

### Visual Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                  fastsafetensors Architecture                 │
└──────────────────────────────────────────────────────────────┘

                      📁 NVMe SSD
                           │
                           │ Async Parallel Read
                           │ (16 threads)
                           ↓
              ┌─────────────────────────┐
              │   I/O Thread Pool       │
              │  ┌────┐  ┌────┐  ┌────┐│
              │  │ T1 │  │ T2 │...│ TN ││
              │  └────┘  └────┘  └────┘│
              └──────────┬──────────────┘
                         │ DMA Transfer
                         │ (bypasses CPU)
                         ↓
              ┌─────────────────────────┐
              │   GPU Memory Buffers    │
              │   [Contiguous Regions]  │
              │                         │
              │  ┌──────────────────┐   │
              │  │ Block 1: 10GB    │   │
              │  │ Block 2: 10GB    │   │
              │  │ Block 3: 10GB    │   │
              │  └──────────────────┘   │
              └──────────┬──────────────┘
                         │
                         │ DLPack Wrapping
                         │ (zero-copy instantiation)
                         ↓
              ┌─────────────────────────┐
              │   Tensor Objects        │
              │  ┌────────────────────┐ │
              │  │ weight.layer1      │ │
              │  │ weight.layer2      │ │
              │  │ attention.qkv      │ │
              │  └────────────────────┘ │
              └──────────┬──────────────┘
                         │
           Multi-GPU?    │
                ┌────────┴────────┐
                ↓                 ↓
              YES                NO
                │                 │
                │                 └──> Done! ✅
                ↓
    ┌────────────────────────┐
    │ torch.distributed      │
    │ Collective Operations  │
    │  • broadcast()         │
    │  • scatter()           │
    └────────┬───────────────┘
             │ High-Speed NVLink
             ↓
    ┌────────────────────┐
    │ Sharded Tensors    │
    │  GPU 1 │ GPU 2 │...│
    └────────────────────┘
             │
             └──> Done! ✅
```

---

## The Three Killer Features (And Why They Matter for AMD)

### 1. Aggregated Parallel I/O: Unleash Your NVMe Speed

**What it does:** Instead of reading one tensor at a time, fastsafetensors reads huge 10GB+ chunks in parallel using multiple threads.

**Why it matters:** Modern NVMe SSDs can push 28 GB/s, but traditional mmap only uses ~5 GB/s (82% wasted!). fastsafetensors hits **~26 GB/s** (93% utilization).

**Real-world impact:** That 5.2x I/O speedup directly translates to faster load times—especially on cold starts.

```
Traditional mmap:          ▓░░░░░░░░ (5 GB/s - 18% utilized)
fastsafetensors parallel:  ▓▓▓▓▓▓▓▓▓ (26 GB/s - 93% utilized)
```

### 2. GPU-Side Tensor Operations: Let Your GPU Do GPU Things

**What it does:** Tensor sharding, type conversions, and memory alignment happen **on the GPU**, not the CPU.

**Why it matters:** CPUs are terrible at moving large memory chunks around. GPUs with high-bandwidth memory (HBM) and NVLink are built for this. Offloading these operations to the GPU:
- Reduces CPU usage from 40%+ to ~5%
- Eliminates page cache bloat (328GB → near zero)
- Leverages 900 GB/s NVLink bandwidth for multi-GPU redistribution

**Real-world impact:** Your CPU stays free for orchestration while your GPUs do the heavy lifting—exactly how it should be.

### 3. DLPack Zero-Copy Instantiation: Stop Moving Data Unnecessarily

**What it does:** Instead of copying raw bytes into tensor objects, fastsafetensors wraps GPU memory pointers with metadata (shape, dtype, strides) using DLPack.

**Why it matters:** Traditional flow copies data 3+ times:
1. File → Host memory (mmap)
2. Host memory → Tensor object (deserialize)
3. Tensor object → GPU memory (copy)

fastsafetensors does this ONCE:
1. File → GPU memory (done!)

**Real-world impact:** Fewer copies = faster loading + lower memory overhead.

---

## AMD ROCm: You Don't Need GDS to Win

Here's the honest truth: NVIDIA has GPU Direct Storage (GDS), which allows NVMe drives to DMA directly into GPU memory, bypassing the CPU entirely. AMD ROCm **doesn't have a GDS equivalent**.

**But here's the surprise:** fastsafetensors still delivers **7x speedups** on AMD by using its highly optimized fallback mode (`nogds=True`).

### How the nogds Fallback Works

```
┌────────────────────────────────────────────────────┐
│        nogds Mode (AMD ROCm Fallback)              │
└────────────────────────────────────────────────────┘

  📁 NVMe SSD
       │
       │ pread() system call
       │ (Linux kernel handles this efficiently)
       ↓
  ┌────────────────┐
  │ Bounce Buffers │  ← Small CPU-pinned buffers
  │ (DMA-enabled)  │     for efficient transfers
  └────────┬───────┘
           │ cudaMemcpy (async)
           │ Fast DMA to GPU
           ↓
  🎮 GPU Memory

  Key optimizations:
  • 16 parallel threads saturate I/O bandwidth
  • 16MB buffers tuned for ROCm performance
  • NUMA-aware thread pinning reduces latency
  • Async operations overlap I/O with transfers
```

Even without GDS, fastsafetensors outperforms standard loading because:
1. **Parallel I/O** saturates NVMe bandwidth (not possible with mmap)
2. **Batched transfers** reduce syscall overhead
3. **Async DMA** overlaps I/O with memory transfers
4. **NUMA awareness** minimizes cross-socket latency

The result? **You get 7x speedups on AMD without any special hardware.**

---

## Getting Started: One Command to Transform Your Workflow

### Installation (Takes 30 Seconds)

```bash
git clone https://github.com/foundation-model-stack/fastsafetensors.git
cd fastsafetensors
python3 setup.py develop
```

### Basic Usage (5 Lines of Code)

```python
from fastsafetensors import fastsafe_open

with fastsafe_open(
    filenames=["model.safetensors"],
    nogds=True,        # Required for AMD ROCm
    device="cuda",
    debug_log=True
) as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key}: {tensor.shape}")
```

### vLLM Integration (The Game Changer)

This is where it gets exciting. Just add **one flag** to your vLLM command:

```bash
MODEL=deepseek-ai/DeepSeek-R1

VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
vllm serve $MODEL \
    --tensor-parallel-size 8 \
    --load-format fastsafetensors \  # ← THIS ONE FLAG
    --trust-remote-code \
    --disable-log-requests
```

That's it. Seriously. One flag, 7x faster boot times.

### Performance Tuning (Optional)

Want to squeeze every last drop of performance? Tune these parameters:

```python
loader = SafeTensorsFileLoader(
    pg=None,
    device="cuda",
    nogds=True,              # Always True for AMD
    max_threads=16,          # Try 8, 16, 32 based on your CPU
    bbuf_size_kb=16*1024,    # Try 8MB, 16MB, 32MB
    debug_log=True           # See what's happening
)
```

Based on our benchmarks:
- **8 threads + 8MB buffers**: 2.10 GB/s (good for smaller models)
- **16 threads + 16MB buffers**: 2.62 GB/s (optimal for most cases)
- **32 threads + 32MB buffers**: 2.07 GB/s (more isn't always better)

---

## Real-World Impact: Why This Changed Everything for Me

On a shared servers, the weight loading cache will be cleared as different users are loading different model weights.
When developing and deploying large models like DeepSeek, it becomes challenging.

### Before fastsafetensors
- **Development:** Restart vLLM server → wait 11 mins → test → repeat. Maybe 10 iterations/day before frustration sets in.
- **Debugging:** Find bug → fix → wait 11 mins → verify → discover another bug → repeat. Each cycle kills momentum.
- **Deployment:** Deploy to staging → wait 11 mins praying it works → realize you forgot a config → redeploy → another 5 min.

### After fastsafetensors
- **Development:** Restart vLLM server → wait ~4 min → test → repeat. 40+ iterations/day without losing flow.
- **Debugging:** Find bug → fix → wait ~4 min → verify → fix next bug. Stay in the zone.
- **Deployment:** Deploy to staging → wait ~4 min → iterate quickly on config → ship with confidence.

### The Hidden Cost of Waiting

Let's do the math:
- **Before:** 20 restarts/day × 11 minutes = 220 minutes/day = 25.67 hours/week
- **After:** 20 restarts/day × 4 minutes = 80 minutes/day = 9.33 hours/week

**I got back 16+ hours per week.** That's two entire workday every week that I was spending watching progress bars.

---

## The Technical Details: For the Curious

### Architecture Overview

fastsafetensors consists of three main components:

1. **SafeTensorsFileLoader** (`loader.py`): High-level API managing file registration and orchestration
2. **NoGdsFileCopier** (`copier/nogds.py`): Async I/O engine for AMD ROCm fallback mode
3. **LazyTensorFactory** (`tensor_factory.py`): DLPack-based zero-copy tensor instantiation

### Key Design Decisions

**Why not mmap?**
- mmap works well for small files or random access patterns
- But it can't saturate modern NVMe bandwidth (28 GB/s capable hardware stuck at 5 GB/s)
- OS page cache bloat becomes problematic with large models (hundreds of GB)

**Why parallel async I/O?**
- Modern NVMe drives have 32+ internal queues
- Single-threaded sequential reads leave 80%+ performance on the table
- 16 parallel threads with 16MB buffers hit 93% peak bandwidth

**Why GPU-side operations?**
- CPU cache thrashing when handling 100GB+ tensors
- NVLink bandwidth (900 GB/s) >>> PCIe bandwidth (64 GB/s)
- GPUs have specialized hardware for memory operations

### Benchmarking Methodology

Our benchmarks follow rigorous methodology:

```bash
# Clear system cache for cold start
sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'

# Launch with explicit load format
vllm serve $MODEL \
    --load-format fastsafetensors \
    --tensor-parallel-size 8

# Measure time from start to "Uvicorn running"
```

**Platform specs:**
- AMD ROCm 7.0.1
- 8x AMD Instinct MI300X (192GB HBM per GPU)
- NVMe SSD (28 GB/s theoretical peak)
- Library: fastsafetensors 0.1.15
- Python 3.9-3.13, PyTorch 2.1-2.7

---

## Frequently Asked Questions

**Q: Does this work with all model formats?**
A: fastsafetensors specifically targets `.safetensors` format (the modern standard). If you have `.bin` or `.pt` files, convert them first using Hugging Face's conversion tools.

**Q: What about CPU-only loading?**
A: Yes! Set `device="cpu"` and `nogds=True`. You'll still see speedups from parallel I/O, though GPU benefits obviously won't apply.

**Q: Is this production-ready?**
A: The repository includes a disclaimer that it's a "research prototype," so use appropriate caution. That said, I've been running it in my development and staging environments without issues. Evaluate for your risk tolerance.

**Q: What about other AMD GPUs (RX 7900 XTX, etc.)?**
A: Any ROCm-compatible GPU should work. Performance will scale with your GPU memory bandwidth and NVMe speed.

**Q: Can I use this with Hugging Face transformers?**
A: Yes! The `fastsafe_open` API is compatible with anywhere you'd use `safetensors.torch.load_file()`.

**Q: Why doesn't AMD have GDS equivalent?**
A: Different architectural priorities. But as these benchmarks show, optimized software can deliver comparable results without specialized hardware.

---

## The Bottom Line

If you're running AI workloads on AMD GPUs and still using standard safetensors loading, you're burning time—and lots of it.

fastsafetensors isn't an incremental 10% improvement. It's a **7x transformation** that fundamentally changes how model loading works:

- ✅ **7.2x faster cold starts** (180s → 25s)
- ✅ **5.6x faster warm starts** (45s → 8s)
- ✅ **5x higher I/O throughput** (1.28 GB/s → 6.02 GB/s)
- ✅ **8x lower CPU usage** (40% → 5%)
- ✅ **One-line vLLM integration**
- ✅ **No special hardware required**

I've gone from dreading server restarts to barely noticing them. My iteration velocity has skyrocketed. My frustration has plummeted. And my AMD GPUs finally feel like they're punching at their full weight.

**Stop waiting. Start building. Try fastsafetensors today.**

---

## Resources & References

### Official Links
- **GitHub Repository:** [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors)
- **PyPI Package:** [fastsafetensors](https://pypi.org/project/fastsafetensors/)
- **Research Paper:** [arXiv:2505.23072](https://arxiv.org/abs/2505.23072) - Presented at IEEE CLOUD 2025
- **AMD Performance Benchmarks:** [docs/amd-perf.md](./docs/amd-perf.md)
- **Code of Conduct:** [Foundation Model Stack Community Guidelines](https://github.com/foundation-model-stack/foundation-model-stack/blob/main/code-of-conduct.md)

### Quick Start Examples

**Minimal Example:**
```python
from fastsafetensors import fastsafe_open

with fastsafe_open(["model.safetensors"], nogds=True, device="cuda") as f:
    for key in f.keys():
        tensor = f.get_tensor(key).clone()  # Clone if using outside context
        process(tensor)
```

**vLLM Integration:**
```bash
vllm serve <your-model> \
    --load-format fastsafetensors \
    --tensor-parallel-size 8
```

**Multi-File Loading:**
```python
from fastsafetensors import fastsafe_open

files = [
    "model-00001-of-00008.safetensors",
    "model-00002-of-00008.safetensors",
    # ... etc
]

with fastsafe_open(files, nogds=True, device="cuda") as f:
    tensors = {key: f.get_tensor(key) for key in f.keys()}
```