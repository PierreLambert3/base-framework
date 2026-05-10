"""
CUDAManager - Main process coordinator for CUDA wrapper.

Wiki: wiki/07-cuda-wrapper.md (section 7.2).

Responsible for:
- Detecting GPU device and architecture
- Generating picklable configuration for worker processes
"""

from pathlib import Path
from cuda.bindings import driver as cuda


class CUDAManager:
    """
    Main process CUDA coordinator.
    
    Detects device capabilities and provides configuration for worker processes.
    Does NOT compile kernels - that's done lazily by each CUDAContext.
    
    Usage:
        manager = CUDAManager(device_id=0, kernel_dir="kernels")
        config = manager.get_config(uses_pytorch=True)
        # Pass config to worker processes
    """
    
    def __init__(self, device_id: int = 0, kernel_dir: str | Path = "kernels"):
        """
        Initialize CUDA manager.
        
        Args:
            device_id: CUDA device index to use
            kernel_dir: Directory containing .cu kernel source files
        """
        self.device_id = device_id
        self.kernel_dir = Path(kernel_dir).resolve()
        
        # Initialize CUDA and get device info
        cuda.cuInit(0)
        
        _, self._device = cuda.cuDeviceGet(device_id)
        
        # Get device name
        _, name = cuda.cuDeviceGetName(256, self._device)
        self.device_name = name.decode().strip()
        
        # Get compute capability
        _, major = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 
            self._device
        )
        _, minor = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 
            self._device
        )
        
        self.compute_capability = (major, minor)
        self.arch = f"sm_{major}{minor}"
        
    def get_config(self, uses_pytorch: bool = True) -> dict:
        """
        Get picklable configuration for a worker process.
        
        Args:
            uses_pytorch: If True, worker will use primary context (shared with PyTorch).
                          If False, worker will create its own isolated context.
        
        Returns:
            Dictionary with configuration that can be pickled and sent to workers.
        """
        return {
            "device_id": self.device_id,
            "arch": self.arch,
            "kernel_dir": str(self.kernel_dir),
            "uses_pytorch": uses_pytorch,
        }
    
    def create_context(self, uses_pytorch: bool = True):
        """
        Create a CUDAContext for use in the current process.
        
        Args:
            uses_pytorch: If True, uses primary context (shared with PyTorch).
                          If False, creates an isolated context.
        
        Returns:
            CUDAContext that can be used as a context manager.
            
        Example:
            with manager.create_context(uses_pytorch=False) as ctx:
                gpu_arr = ctx.m(numpy_array)
                kernel = ctx.get_kernel("my_module", "my_kernel", config)
                kernel.launch(None, gpu_arr, np.int32(n))
        """
        from .context import CUDAContext
        config = self.get_config(uses_pytorch=uses_pytorch)
        return CUDAContext(config)
    
    def __repr__(self):
        return (
            f"CUDAManager(device={self.device_id}, "
            f"name='{self.device_name}', "
            f"arch={self.arch})"
        )
