"""
CUDA Wrapper - A simple wrapper for cuda-python with PyTorch compatibility.

Wiki: wiki/07-cuda-wrapper.md (full reference).
Related: wiki/06-worker-instances.md (where contexts are entered),
         wiki/01-architecture.md (process model).

Provides:
- CUDAManager: Main process coordinator (device detection, config generation)
- CUDAContext: Per-process CUDA context management with kernel compilation/caching
- GPUArray: Device memory wrapper with easy host<->device transfers
- LaunchConfig classes: Block/grid configuration with shared memory management
- ParallelReducer: Efficient parallel reduction for GPU arrays
"""

from .manager import CUDAManager
from .context import CUDAContext, CUDAKernel, CUDAStream, GPUArray
from .launch_config import (
    DeviceProperties,
    _LaunchConfig,
    LaunchConfig1D,
    LaunchConfig2D,
    LaunchConfigGrid2D,
)
from .reduction import ParallelReducer

__all__ = [
    "CUDAManager",
    "CUDAContext", 
    "CUDAKernel", 
    "CUDAStream",
    "GPUArray",
    "DeviceProperties",
    "LaunchConfig1D",
    "LaunchConfig2D",
    "LaunchConfigGrid2D",
    "ParallelReducer",
]
