# AMD Alternatives to NVIDIA GPU Direct Storage - Research Summary

## Executive Summary

**Bottom Line**: AMD does **NOT** have a ROCm/Instinct equivalent to NVIDIA's GPUDirect Storage for datacenter GPUs like MI300X as of 2025.

AMD has **Smart Access Storage** for gaming/consumer GPUs, but it's **not available** for Instinct (datacenter) GPUs or integrated into ROCm.

## NVIDIA GPUDirect Storage (GDS)

### What It Is
- Technology enabling direct data path between NVMe storage and GPU memory
- Bypasses CPU and system memory entirely
- Uses NVIDIA's cuFile API
- Part of NVIDIA Magnum IO suite
- Supported on Linux x86-64 (not Windows)
- Available with CUDA

### Performance Benefits
- Eliminates CPU bounce buffer
- Reduces latency
- Increases throughput for large model loading
- Critical for AI/ML workloads with massive datasets

### Ecosystem Support
- DDN, Dell Technologies, Excelero, HPE, IBM Storage
- Micron, NetApp, Pavilion, ScaleFlux, VAST, WekaIO
- Hardware support from HighPoint (Rocket 7638D - PCIe Gen5 switch adapter)

## AMD's Options (and Their Limitations)

### 1. Smart Access Storage ❌ (Gaming Only)

**What It Is**:
- AMD's answer to Microsoft DirectStorage API
- Provides direct SSD-to-GPU connection bypassing system RAM
- GPU handles asset compression/decompression

**Limitations for Your Use Case**:
- ❌ **Gaming/Consumer focused** - Targets Ryzen CPUs + Radeon GPUs
- ❌ **Not for Instinct GPUs** - No MI300X, MI250X, or datacenter support
- ❌ **Not in ROCm** - Not part of ROCm software stack
- ❌ **No Linux server support** - Designed for Windows gaming workloads
- ❌ **Requires certified NVMe drives** - Limited vendor support list

**Status**: Not applicable for MI300X servers running ROCm.

### 2. ROCm RDMA ⚠️ (Network Only)

**What It Is**:
- AMD kernel driver exposes RDMA through PeerDirect interfaces
- Allows NICs to directly read/write GPU memory
- Part of GPU-aware MPI implementation

**Limitations**:
- ✅ Works for **network** transfers (GPU-to-GPU over fabric)
- ❌ Does **NOT** work for **storage** transfers (NVMe-to-GPU)
- ⚠️ Different use case than GPUDirect Storage

**Status**: Useful for distributed training, not for model loading.

### 3. Standard ROCm I/O ❌ (No Direct Path)

**What Happens**:
- Traditional path: NVMe → CPU/System Memory → GPU Memory
- Uses standard Linux I/O with `pread()` or `mmap()`
- CPU copies data to GPU with `hipMemcpy()`

**Performance**:
- Slower than GPUDirect Storage
- CPU overhead
- System memory bandwidth bottleneck
- But can be parallelized (what fastsafetensors does!)

## Industry Context

### Community Requests

From GitHub discussion [ROCm#2442](https://github.com/ROCm/ROCm/discussions/2442):
> "Can ROCm or any AMD GPU get a smart access storage like feature"

**Status**: Requested feature, not yet implemented as of 2025.

### Missing AI/ML Features

Per SemiAnalysis benchmarks:
> "AMD is currently lacking support for many inference features such as:
> - Disaggregated prefill
> - Smart Routing
> - **NVMe KV Cache Tiering** ❌"

NVMe KV Cache Tiering would require direct GPU-to-storage access (like GDS).

### ROCm 7.0 (2025) - No Direct Storage

ROCm 7.0 changelog focuses on:
- ✅ FP4/FP6 datatype support
- ✅ Linux distribution integration (RHEL, Ubuntu, SUSE, Fedora)
- ✅ Radeon GPU support expansion
- ✅ PyTorch/ONNX on Windows
- ❌ **No mention of direct storage capabilities**

## Hardware Alternatives

### HighPoint Rocket 7638D (2025)

**What It Is**:
- PCIe Gen5 switch adapter for GPU-Direct NVMe storage
- **Hardware** solution to provide direct data path
- Requires third-party **software stack** to complete GDS implementation

**Compatibility**:
- ✅ Designed for **NVIDIA GPUs**
- ❓ **Unknown if works with AMD GPUs** - likely NVIDIA-specific
- Would require cuFile or equivalent software layer

**Status**: Probably NVIDIA-only; no AMD equivalent announced.

## Impact on Your Use Case (MI300X + 190 Safetensors Files)

### What You DON'T Have
❌ GPU Direct Storage (cuFile)
❌ Smart Access Storage (consumer tech)
❌ Direct NVMe-to-GPU transfers

### What You DO Have
✅ **ROCm HIP runtime** - Standard GPU operations
✅ **Parallel CPU reading** - fastsafetensors multi-threading
✅ **Pinned memory** - `hipHostAlloc()` for faster transfers
✅ **Tensor parallelism** - Distribute load across multiple MI300X

### Performance Expectations

| Technology | Path | Your MI300X Status |
|------------|------|-------------------|
| Standard loading | NVMe → RAM → GPU | ✅ Default (16 min) |
| fastsafetensors (nogds) | NVMe → RAM → GPU (parallel) | ✅ Works (5-8 min) |
| fastsafetensors + TP | Split across 4 GPUs | ✅ Works (4 min) |
| NVIDIA GDS | NVMe → GPU direct | ❌ N/A on AMD |
| AMD equivalent | - | ❌ Doesn't exist |

## Recommendations

### Short Term (Now)

1. ✅ **Use fastsafetensors with nogds=True**
   - Parallel reading still gives 2-3x speedup
   - CPU bounce buffer is optimized

2. ✅ **Leverage tensor parallelism**
   - Each MI300X loads 1/4 of the files
   - Best performance available on AMD

3. ✅ **Optimize system**
   - Use fast NVMe (PCIe 4.0/5.0)
   - Enable NUMA affinity
   - Large system RAM for buffers

### Medium Term (Monitor)

1. ⏳ **Watch for AMD Smart Access Storage** expansion
   - Currently gaming-only
   - May come to datacenter GPUs in future
   - Would require ROCm integration

2. ⏳ **Check ROCm roadmap**
   - Future versions may add direct storage
   - Community has requested this feature

3. ⏳ **Hardware solutions**
   - HighPoint or similar may support AMD
   - Requires AMD software stack development

### Long Term (Alternatives)

1. **Software optimizations**
   - Aggressive prefetching
   - Better parallelization
   - Memory-mapped I/O tuning

2. **Architecture changes**
   - Disaggregated storage nodes
   - Pre-load to system RAM
   - Cached model layers

## Conclusion

**For MI300X in 2025:**

| Feature | NVIDIA | AMD |
|---------|--------|-----|
| Direct GPU-Storage | ✅ GPUDirect Storage | ❌ Not available |
| cuFile API | ✅ Yes | ❌ No equivalent |
| Consumer direct storage | ✅ RTX IO | ⚠️ Smart Access (gaming only) |
| ROCm integration | N/A | ❌ None |
| Datacenter support | ✅ Instinct GPUs | ❌ Not yet |

**Reality**: You're stuck with CPU bounce buffers, but fastsafetensors' parallel implementation still provides significant speedup (2-4x) over naive loading.

**Best available solution**: fastsafetensors in nogds mode + tensor parallelism across your MI300X GPUs.

## References

1. [NVIDIA GPUDirect Storage Overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
2. [AMD Smart Access Storage](https://www.digitaltrends.com/computing/what-is-amd-smart-access-storage/)
3. [ROCm GitHub Discussion #2442](https://github.com/ROCm/ROCm/discussions/2442)
4. [AMD ROCm 7.0 Release](https://www.amd.com/en/blogs/2025/rocm7-supercharging-ai-and-hpc-infrastructure.html)
5. [SemiAnalysis AMD vs NVIDIA Benchmark](https://semianalysis.com/2025/05/23/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens/)
6. [HighPoint Rocket 7638D](https://www.techpowerup.com/341324/highpoint-technologies-introduces-rocket-7638d-industrys-first-hardware-architecture-for-gpu-direct-nvme-storage)

---

*Research conducted: October 2025*
*ROCm version: 7.0.1*
*AMD GPU: MI300X*
