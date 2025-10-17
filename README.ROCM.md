# fastsafetensors - ROCm/HIP Port for AMD GPUs

This is a HIP-ified port of fastsafetensors for AMD GPUs with ROCm support.

## What Was Changed

This port converts NVIDIA CUDA-specific code to work with AMD's HIP/ROCm platform:

### Core Changes

1. **CUDA Runtime → HIP Runtime**
   - `cudaMemcpy` → `hipMemcpy`
   - `cudaHostAlloc` → `hipHostAlloc`
   - `cudaMalloc` → `hipMalloc`
   - `libcudart.so` → `libamdhip64.so`

2. **New Files Created**
   - `fastsafetensors/cpp/ext.hip.cpp` - Hipified C++ implementation
   - `fastsafetensors/cpp/ext.hip.hpp` - Hipified header file
   - `setup.rocm.py` - ROCm-aware setup script

3. **GPU Direct Storage (GDS)**
   - cuFile API is NVIDIA-specific and **not available on AMD**
   - The code gracefully falls back to `nogds` mode (CPU bounce buffer)
   - This is expected behavior on AMD GPUs

## Requirements

- ROCm 7.0+ (tested on ROCm 7.0.1)
- AMD MI300X or compatible GPU
- Python 3.9-3.13
- PyTorch 2.1+ with ROCm support
- pybind11

## Installation

```bash
cd /app/deepep/fastsafetensors
python3 setup.rocm.py install
```

Or for development:

```bash
python3 setup.rocm.py develop
```

## Usage

The API is identical to the original fastsafetensors:

```python
import fastsafetensors
from fastsafetensors import fastsafe_open

# Load tensors (will use nogds mode on AMD)
with fastsafe_open(filenames=["model.safetensors"], nogds=True, device="cuda", debug_log=True) as f:
    for key in f.get_keys():
        tensor = f.get_tensor(key)
```

## Performance Notes

### Expected Behavior

- **ROCm GPU detected**: ✅ Yes (reported as "cuda_found")
- **cuFile support**: ❌ No (AMD doesn't have GPU Direct Storage)
- **Fallback mode**: Uses CPU bounce buffer with parallel threads

### Performance Comparison

Without GPU Direct Storage, fastsafetensors still provides significant speedups through:

1. **Parallel file loading** - Multiple threads read files concurrently
2. **Batched tensor instantiation** - Reduces overhead
3. **GPU-offloaded operations** - Type conversions and sharding on GPU
4. **Pinned memory transfers** - `hipHostAlloc` for faster CPU<->GPU copies

For your use case (190 x 3GB files, 5s each = ~16 minutes):
- **Standard loading**: ~16 minutes
- **fastsafetensors (nogds)**: Estimated 5-8 minutes (2-3x faster)
- **Tensor parallelism**: With 4 MI300X GPUs, each loads 48 files (~4 minutes)

## Testing

Verify the installation:

```bash
python3 -c "
import fastsafetensors.cpp as cpp
cpp.load_nvidia_functions()
print('ROCm GPU found:', cpp.is_cuda_found())
print('cuFile found:', cpp.is_cufile_found())
print('Alignment:', cpp.get_alignment_size())
"
```

Expected output:
```
ROCm GPU found: True
cuFile found: False
Alignment: 4096
```

## Integration with vLLM

To use with vLLM on ROCm:

1. Install this hipified fastsafetensors
2. Set environment variable:
   ```bash
   export USE_FASTSAFETENSOR=true
   ```
3. Launch vLLM:
   ```bash
   vllm serve <model> --load-format fastsafetensors --tensor-parallel-size <num_gpus>
   ```

## Limitations

1. **No GPU Direct Storage**: AMD GPUs don't support NVIDIA's cuFile/GDS
   - Falls back to CPU bounce buffer automatically
   - Still faster than standard loading due to parallelization

2. **Function Names**: Internal variable names still use "cuda" prefix (e.g., `cuda_fns`, `is_cuda_found()`)
   - This is intentional to minimize code changes
   - "CUDA" here means "GPU" in general

## Troubleshooting

### Module won't load
```bash
# Check ROCm installation
rocm-smi --version

# Verify HIP library
ls -l /opt/rocm/lib/libamdhip64.so*

# Check Python can find ROCm
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Compilation errors
```bash
# Set ROCM_PATH if non-standard location
export ROCM_PATH=/opt/rocm-7.0.1

# Clean and rebuild
python3 setup.rocm.py clean --all
python3 setup.rocm.py build_ext --inplace
```

## Technical Details

### Hipification Process

1. Used `hipify-perl` from ROCm 7.0.1 to automatically convert CUDA→HIP
2. Manually fixed function pointers (`cudaDeviceMalloc` → `hipDeviceMalloc`)
3. Changed dynamic library loading from `libcudart.so` → `libamdhip64.so`
4. Removed hipify warning comments from generated files
5. Updated build system to link against ROCm libraries

### Files Modified

- `fastsafetensors/cpp/ext.hip.cpp` - Main implementation
- `fastsafetensors/cpp/ext.hip.hpp` - Header definitions
- `setup.rocm.py` - Build configuration

Original files remain untouched for NVIDIA GPU compatibility.

## Contributing

This is an unofficial port. For issues specific to the ROCm version, please note that this is a research prototype adapted for AMD GPUs. The original NVIDIA version is maintained by the Foundation Model Stack team.

## License

Apache-2.0 (same as original fastsafetensors)

## Credits

- Original fastsafetensors: IBM Research / Foundation Model Stack
- ROCm port: Community contribution
- Paper: Yoshimura et al., "Speeding up Model Loading with fastsafetensors" (IEEE CLOUD 2025)
