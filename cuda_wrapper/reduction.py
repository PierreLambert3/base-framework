"""
ParallelReducer - Efficient parallel reduction for GPU arrays.

Provides sum and mean reductions for arbitrary-length arrays using
optimized CUDA kernels with pre-computed launch configurations.

Strategy (following Mark Harris's principles):
- n <= max_block_size: Single-block reduction
- n > max_block_size: Two-pass reduction (partial sums + final reduction)

Usage:
    # Create reducer once with fixed size
    reducer = ParallelReducer(cuda_ctx, n=4096)
    
    # Call repeatedly without recomputing shapes
    reducer.reduce_sum(stream, input_gpu, output_gpu, output_index=0)
    reducer.reduce_mean(stream, input_gpu, output_gpu, output_index=0)
"""

import numpy as np
from .launch_config import DeviceProperties, LaunchConfig1D


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


class ParallelReducer:
    """
    Efficient parallel reduction with pre-computed launch configurations.
    
    Instantiate once with fixed array length, then call reduce_sum/reduce_mean
    repeatedly without overhead of recomputing kernel shapes.
    
    Args:
        cuda_ctx: CUDAContext instance
        n: Number of elements to reduce (fixed at instantiation)
        props: Optional DeviceProperties (created if not provided)
    
    Example:
        reducer = ParallelReducer(ctx, n=4096)
        reducer.reduce_sum(stream, rewards_gpu, rewards_gpu, output_index=0)
        # Result is now in rewards_gpu[0]
    """
    
    # Block size for large reductions (power of 2, <= 1024)
    _BLOCK_SIZE = 256
    
    # Maximum number of blocks for first pass (limits partials buffer size)
    _MAX_BLOCKS = 1024
    
    def __init__(self, cuda_ctx: 'CUDAContext', n: int, props: DeviceProperties = None):
        self._ctx = cuda_ctx
        self._n = int(n)
        self._props = props if props is not None else DeviceProperties()
        
        if self._n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        
        # Pre-compute strategy and shapes
        self._is_single_block = (self._n <= self._props.max_threads_per_block)
        
        # Pre-allocate numpy scalars for kernel arguments (avoid allocation on each call)
        self._n_uint32 = np.uint32(self._n)
        
        if self._is_single_block:
            self._setup_single_block()
        else:
            self._setup_two_pass()
        
        # Kernels (lazy-loaded on first use)
        self._kernel_sum = None
        self._kernel_mean = None
        self._kernel_partial = None
        self._kernel_finalize_sum = None
        self._kernel_finalize_mean = None
    
    def _setup_single_block(self):
        """Configure for single-block reduction."""
        # Block size = next power of 2 >= n, capped at max_threads_per_block
        block_size = min(_next_power_of_2(self._n), self._props.max_threads_per_block)
        
        # Ensure block size is at least warp size for proper reduction
        block_size = max(block_size, self._props.warp_size)
        
        # Shared memory: one float per warp
        n_warps = block_size // self._props.warp_size
        smem_n32bits = n_warps
        
        self._single_block_size = block_size
        self._single_smem_bytes = smem_n32bits * 4
        
        # No partials buffer needed
        self._partials_buffer = None
        self._num_blocks = 1
    
    def _setup_two_pass(self):
        """Configure for two-pass reduction (large arrays)."""
        block_size = self._BLOCK_SIZE
        
        # Number of blocks for first pass
        # Use enough blocks to cover the array, but cap at _MAX_BLOCKS
        num_blocks = min((self._n + block_size - 1) // block_size, self._MAX_BLOCKS)
        
        # Shared memory for both passes: one float per warp
        n_warps = block_size // self._props.warp_size
        smem_n32bits = n_warps
        
        self._first_pass_block_size = block_size
        self._first_pass_num_blocks = num_blocks
        self._first_pass_smem_bytes = smem_n32bits * 4
        
        # Second pass: reduce num_blocks partial sums
        second_block_size = min(_next_power_of_2(num_blocks), self._props.max_threads_per_block)
        second_block_size = max(second_block_size, self._props.warp_size)
        second_n_warps = second_block_size // self._props.warp_size
        
        self._second_pass_block_size = second_block_size
        self._second_pass_smem_bytes = second_n_warps * 4
        
        # Pre-compute uint32 for num_blocks
        self._num_partials_uint32 = np.uint32(num_blocks)
        
        # Allocate partials buffer
        self._partials_buffer = self._ctx.zeros(num_blocks, dtype=np.float32)
        self._num_blocks = num_blocks
    
    def _ensure_kernels_loaded(self):
        """Lazily load kernels on first use."""
        if self._kernel_sum is not None:
            return
        
        from .context import CUDAKernel
        from cuda.bindings import driver as cuda
        
        # Load the reduction module
        module_name = "basics/reduction"
        if module_name not in self._ctx._modules:
            self._ctx._load_module(module_name)
        module = self._ctx._modules[module_name]
        
        if self._is_single_block:
            # Single-block kernels
            _, func_sum = cuda.cuModuleGetFunction(module, b"reduce_sum_single_block")
            _, func_mean = cuda.cuModuleGetFunction(module, b"reduce_mean_single_block")
            
            # Create kernel wrappers with pre-computed config
            self._kernel_sum = _FixedKernel(
                func_sum, "reduce_sum_single_block",
                grid=(1, 1, 1),
                block=(self._single_block_size, 1, 1),
                smem_bytes=self._single_smem_bytes
            )
            self._kernel_mean = _FixedKernel(
                func_mean, "reduce_mean_single_block",
                grid=(1, 1, 1),
                block=(self._single_block_size, 1, 1),
                smem_bytes=self._single_smem_bytes
            )
        else:
            # Two-pass kernels
            _, func_partial = cuda.cuModuleGetFunction(module, b"reduce_sum_partial_blocks")
            _, func_finalize_sum = cuda.cuModuleGetFunction(module, b"reduce_partials_finalize")
            _, func_finalize_mean = cuda.cuModuleGetFunction(module, b"reduce_partials_finalize_mean")
            
            self._kernel_partial = _FixedKernel(
                func_partial, "reduce_sum_partial_blocks",
                grid=(self._first_pass_num_blocks, 1, 1),
                block=(self._first_pass_block_size, 1, 1),
                smem_bytes=self._first_pass_smem_bytes
            )
            self._kernel_finalize_sum = _FixedKernel(
                func_finalize_sum, "reduce_partials_finalize",
                grid=(1, 1, 1),
                block=(self._second_pass_block_size, 1, 1),
                smem_bytes=self._second_pass_smem_bytes
            )
            self._kernel_finalize_mean = _FixedKernel(
                func_finalize_mean, "reduce_partials_finalize_mean",
                grid=(1, 1, 1),
                block=(self._second_pass_block_size, 1, 1),
                smem_bytes=self._second_pass_smem_bytes
            )
    
    def reduce_sum(self, stream: 'CUDAStream', input_arr: 'GPUArray', 
                   output_arr: 'GPUArray', output_index: int = 0):
        """
        Compute sum of input array, write to output[output_index].
        
        Args:
            stream: CUDA stream (or None for blocking)
            input_arr: Input GPUArray to reduce
            output_arr: Output GPUArray (can be same as input for in-place)
            output_index: Index in output array to write result (default: 0)
        """
        self._ensure_kernels_loaded()
        output_idx = np.uint32(output_index)
        
        if self._is_single_block:
            self._kernel_sum.launch(stream, input_arr, output_arr, self._n_uint32, output_idx)
        else:
            # First pass: compute partial sums
            self._kernel_partial.launch(stream, input_arr, self._partials_buffer, self._n_uint32)
            # Second pass: reduce partials to final sum
            self._kernel_finalize_sum.launch(stream, self._partials_buffer, output_arr, 
                                             self._num_partials_uint32, output_idx)
    
    def reduce_mean(self, stream: 'CUDAStream', input_arr: 'GPUArray', 
                    output_arr: 'GPUArray', output_index: int = 0):
        """
        Compute mean of input array, write to output[output_index].
        
        Args:
            stream: CUDA stream (or None for blocking)
            input_arr: Input GPUArray to reduce
            output_arr: Output GPUArray (can be same as input for in-place)
            output_index: Index in output array to write result (default: 0)
        """
        self._ensure_kernels_loaded()
        output_idx = np.uint32(output_index)
        
        if self._is_single_block:
            self._kernel_mean.launch(stream, input_arr, output_arr, self._n_uint32, output_idx)
        else:
            # First pass: compute partial sums
            self._kernel_partial.launch(stream, input_arr, self._partials_buffer, self._n_uint32)
            # Second pass: reduce partials and divide by n
            self._kernel_finalize_mean.launch(stream, self._partials_buffer, output_arr,
                                              self._num_partials_uint32, self._n_uint32, output_idx)
    
    @property
    def n(self) -> int:
        """Number of elements this reducer is configured for."""
        return self._n
    
    @property
    def is_single_block(self) -> bool:
        """True if using single-block reduction (n <= max_block_size)."""
        return self._is_single_block
    
    def __repr__(self):
        mode = "single-block" if self._is_single_block else f"two-pass ({self._num_blocks} blocks)"
        return f"ParallelReducer(n={self._n}, mode={mode})"


class _FixedKernel:
    """
    Lightweight kernel wrapper with fixed launch configuration.
    
    Avoids the overhead of CUDAKernel's validation for hot paths
    where we know the configuration is correct.
    """
    
    def __init__(self, function, name: str, grid: tuple, block: tuple, smem_bytes: int):
        self._function = function
        self.name = name
        self._grid = grid
        self._block = block
        self._smem_bytes = smem_bytes
        self._internal_stream = None
    
    def _get_internal_stream(self):
        """Lazily create internal stream for blocking launches."""
        if self._internal_stream is None:
            from cuda.bindings import driver as cuda
            _, handle = cuda.cuStreamCreate(0)
            self._internal_stream = handle
        return self._internal_stream
    
    def launch(self, stream, *args):
        """Launch kernel with given arguments."""
        import ctypes
        from cuda.bindings import driver as cuda
        from .context import GPUArray
        
        # Determine stream handle
        if stream is None:
            stream_handle = self._get_internal_stream()
            blocking = True
        else:
            stream_handle = stream.handle
            blocking = False
        
        # Build kernel params
        if args:
            values = []
            types = []
            for arg in args:
                if isinstance(arg, GPUArray):
                    values.append(int(arg._ptr))
                    types.append(ctypes.c_void_p)
                elif isinstance(arg, (np.floating, np.integer)):
                    values.append(arg.item())
                    dtype = np.dtype(type(arg))
                    if dtype == np.dtype('float32'):
                        types.append(ctypes.c_float)
                    elif dtype == np.dtype('uint32'):
                        types.append(ctypes.c_uint32)
                    elif dtype == np.dtype('int32'):
                        types.append(ctypes.c_int32)
                    else:
                        raise TypeError(f"Unsupported scalar dtype: {dtype}")
                else:
                    raise TypeError(f"Unsupported argument type: {type(arg)}")
            kernel_params = (tuple(values), tuple(types))
        else:
            kernel_params = 0
        
        cuda.cuLaunchKernel(
            self._function,
            self._grid[0], self._grid[1], self._grid[2],
            self._block[0], self._block[1], self._block[2],
            self._smem_bytes,
            stream_handle,
            kernel_params,
            0
        )
        
        if blocking:
            cuda.cuStreamSynchronize(stream_handle)
