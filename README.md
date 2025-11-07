# fastsafetensors Custom Package Index

This repository hosts custom Python package indices for fastsafetensors.

## ROCm Index

Install fastsafetensors with ROCm support:

```bash
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/
```

## CUDA Index

Install fastsafetensors with CUDA support:

```bash
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/
```

## What's Included

- fastsafetensors ROCm/CUDA builds for Python 3.9-3.13
- All dependencies mirrored from PyPI

This ensures `pip install` works completely offline from PyPI with just this index.

## For Maintainers

See the main branch's [PUBLISHING.md](https://github.com/EmbeddedLLM/fastsafetensors-rocm/blob/main/PUBLISHING.md) for publishing instructions.
