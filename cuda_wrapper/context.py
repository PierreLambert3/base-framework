"""
CUDAContext - Per-process CUDA context management.

Wiki: wiki/07-cuda-wrapper.md (sections 7.3 -- 7.4).
Related: wiki/06-worker-instances.md (where workers enter the context).

Provides:
- Context creation (primary or isolated based on uses_pytorch)
- Kernel compilation and caching
- Stream management
- GPU memory allocation and management
"""

from pathlib import Path
import ctypes
import numpy as np
from cuda.bindings import driver as cuda
from .compiler import compile_kernel
from .launch_config import _LaunchConfig


# Supported numpy dtypes and their sizes
_SUPPORTED_DTYPES = {
    np.dtype('float32'): 4,
    np.dtype('float64'): 8,
    np.dtype('int32'): 4,
    np.dtype('int64'): 8,
    np.dtype('uint8'): 1,
    np.dtype('uint16'): 2,
    np.dtype('uint32'): 4,
    np.dtype('uint64'): 8,
    np.dtype('int8'): 1,
    np.dtype('int16'): 2,
}


class GPUArray:
    """
    Wrapper around a GPU memory allocation.
    
    Stores metadata (shape, dtype, size) and provides convenient methods
    for data transfer and access.
    
    Do not instantiate directly - use CUDAContext.m() instead.
    """
    
    def __init__(self, ptr: int, shape: tuple, dtype: np.dtype, ctx: 'CUDAContext'):
        self._ptr = ptr
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._ctx = ctx
        self._freed = False
        
        # Compute derived properties
        self._size = int(np.prod(shape))  # Total number of elements
        self._itemsize = self._dtype.itemsize
        self._nbytes = self._size * self._itemsize
    
    @property
    def ptr(self) -> int:
        """Raw device pointer (for kernel arguments)."""
        if self._freed:
            raise RuntimeError("GPUArray has been freed")
        return self._ptr
    
    @property
    def shape(self) -> tuple:
        """Shape of the array."""
        return self._shape
    
    @property
    def dtype(self) -> np.dtype:
        """Data type of elements."""
        return self._dtype
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._size
    
    @property
    def nbytes(self) -> int:
        """Total size in bytes."""
        return self._nbytes
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)
    
    def to_host(self, out: np.ndarray = None, stream: 'CUDAStream' = None) -> np.ndarray:
        """
        Copy data from GPU to host.
        
        Args:
            out: Optional pre-allocated numpy array to copy into.
                 Must have matching shape and dtype.
            stream: If provided, copy is async. If None (default), copy is synchronous.
                    For true async, `out` should be page-locked (pinned) memory.
        
        Returns:
            Numpy array with the data (either `out` or a new array).
        """
        if self._freed:
            raise RuntimeError("GPUArray has been freed")
        
        if out is None:
            out = np.empty(self._shape, dtype=self._dtype)
        else:
            if out.shape != self._shape:
                raise ValueError(f"Output shape {out.shape} doesn't match GPUArray shape {self._shape}")
            if out.dtype != self._dtype:
                raise ValueError(f"Output dtype {out.dtype} doesn't match GPUArray dtype {self._dtype}")
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError("Output array must be C-contiguous")
        
        if stream is None:
            cuda.cuMemcpyDtoH(out.ctypes.data, self._ptr, self._nbytes)
        else:
            cuda.cuMemcpyDtoHAsync(out.ctypes.data, self._ptr, self._nbytes, stream.handle)
        return out
    get = to_host
    
    def copy_from(self, arr: np.ndarray, stream: 'CUDAStream' = None):
        """
        Copy data from host numpy array to this GPU array.
        
        Args:
            arr: Numpy array with matching shape and dtype.
            stream: If provided, copy is async. If None (default), copy is synchronous.
                    For true async, `arr` should be page-locked (pinned) memory
                    and must remain valid until stream.sync().
        """
        if self._freed:
            raise RuntimeError("GPUArray has been freed")
        
        if arr.shape != self._shape:
            raise ValueError(f"Input shape {arr.shape} doesn't match GPUArray shape {self._shape}")
        if arr.dtype != self._dtype:
            raise ValueError(f"Input dtype {arr.dtype} doesn't match GPUArray dtype {self._dtype}")
        if not arr.flags['C_CONTIGUOUS']:
            raise ValueError("Input array must be C-contiguous")
        
        if stream is None:
            cuda.cuMemcpyHtoD(self._ptr, arr.ctypes.data, self._nbytes)
        else:
            cuda.cuMemcpyHtoDAsync(self._ptr, arr.ctypes.data, self._nbytes, stream.handle)
    set = copy_from

    def zero(self, stream: 'CUDAStream' = None):
        """
        Fill the GPU array with zeros.
        
        Args:
            stream: If provided, operation is async. If None (default), synchronous.
        """
        if self._freed:
            raise RuntimeError("GPUArray has been freed")
        if stream is None:
            cuda.cuMemsetD8(self._ptr, 0, self._nbytes)
        else:
            cuda.cuMemsetD8Async(self._ptr, 0, self._nbytes, stream.handle)
    clear = zero
    
    def _mark_freed(self):
        """Mark as freed (called by CUDAContext)."""
        self._freed = True
        self._ptr = 0
    
    def __repr__(self):
        status = "freed" if self._freed else f"ptr=0x{int(self._ptr):x}"
        return f"GPUArray(shape={self._shape}, dtype={self._dtype}, {status})"


class CUDAStream:
    def __init__(self, handle):
        self._handle = handle
        
    @property
    def handle(self):
        return self._handle
    
    def synchronize(self):
        """Block until all operations in this stream complete."""
        cuda.cuStreamSynchronize(self._handle)
    sync = synchronize
    synchronise = synchronize
        
    def _destroy(self):
        """Destroy the stream. Called by CUDAContext on cleanup."""
        if self._handle is not None:
            cuda.cuStreamDestroy(self._handle)
            self._handle = None


class CUDAKernel:
    """
    Wrapper around a CUDA kernel function with fixed launch configuration.
    
    Automatically packs kernel arguments using cuda-python's efficient (values, types)
    tuple approach, which delegates packing to C code.
    
    Validation of argument count and types is performed during the first N launches
    (controlled by _CHECK_THRESHOLD) to catch bugs early without permanent overhead.
    
    IMPORTANT: Only numpy typed scalars are accepted for scalar arguments.
    Python int/float are rejected to prevent silent overflow/precision loss.
    
    Supported argument types:
        - GPUArray: passed as device pointer (c_void_p)
        - np.float32: 32-bit float (c_float)
        - np.float64: 64-bit double (c_double)
        - np.int8, np.int16, np.int32, np.int64: signed integers
        - np.uint8, np.uint16, np.uint32, np.uint64: unsigned integers
    
    Example:
        kernel = ctx.get_kernel("my_module", "my_kernel", config)
        
        # Launch (blocking) - use numpy scalars for type safety
        kernel.launch(None, gpu_input, gpu_output, np.int32(n))
        
        # Launch (async)
        kernel.launch(stream, gpu_input, gpu_output, np.int32(n))
        stream.sync()
    """
    
    # Number of launches during which argument validation is performed
    _CHECK_THRESHOLD = 100
    
    # Map numpy dtype to ctypes type for scalars
    _NUMPY_TO_CTYPE = {
        np.dtype('float32'): ctypes.c_float,
        np.dtype('float64'): ctypes.c_double,
        np.dtype('int8'): ctypes.c_int8,
        np.dtype('int16'): ctypes.c_int16,
        np.dtype('int32'): ctypes.c_int32,
        np.dtype('int64'): ctypes.c_int64,
        np.dtype('uint8'): ctypes.c_uint8,
        np.dtype('uint16'): ctypes.c_uint16,
        np.dtype('uint32'): ctypes.c_uint32,
        np.dtype('uint64'): ctypes.c_uint64,
    }
    
    # Expected sizes for each numpy dtype
    _NUMPY_DTYPE_SIZE = {
        np.dtype('float32'): 4,
        np.dtype('float64'): 8,
        np.dtype('int8'): 1,
        np.dtype('int16'): 2,
        np.dtype('int32'): 4,
        np.dtype('int64'): 8,
        np.dtype('uint8'): 1,
        np.dtype('uint16'): 2,
        np.dtype('uint32'): 4,
        np.dtype('uint64'): 8,
    }
    
    def __init__(self, function, name: str, config: _LaunchConfig):
        self._function = function
        self.name = name
        self._config = config
        self._internal_stream = None  # Lazy-created for blocking launches
        
        # Introspect parameter info (sizes) from the kernel
        self._param_sizes = self._get_param_sizes()
        self._num_params = len(self._param_sizes)
        
        # Built lazily on first launch based on actual arg types
        self._param_types: tuple = None
        self._arg_is_gpuarray: tuple = None
        
        # Validation counter
        self._times_checked = 0

    def new_launch_config(self, config: _LaunchConfig):
        self._config = config
        
    def _get_param_sizes(self) -> list:
        """Get parameter sizes from kernel introspection."""
        sizes = []
        param_idx = 0
        while True:
            err, offset, size = cuda.cuFuncGetParamInfo(self._function, param_idx)
            if err != cuda.CUresult.CUDA_SUCCESS:
                break
            sizes.append(size)
            param_idx += 1
        return sizes
    
    def _build_param_types(self, args: tuple) -> tuple:
        """
        Build the ctypes types tuple from actual arguments.
        Called once on first launch to determine the type signature.
        Assumes args have been validated.
        
        Also builds _arg_is_gpuarray pattern for fast extraction.
        """
        types = []
        is_gpuarray = []
        for arg in args:
            if isinstance(arg, GPUArray):
                types.append(ctypes.c_void_p)
                is_gpuarray.append(True)
            elif isinstance(arg, (np.floating, np.integer)):
                dtype = np.dtype(type(arg))
                ctype = self._NUMPY_TO_CTYPE.get(dtype)
                if ctype is None:
                    raise TypeError(f"Unsupported numpy dtype: {dtype}")
                types.append(ctype)
                is_gpuarray.append(False)
            else:
                # Should not reach here if validation passed
                raise TypeError(f"Unsupported argument type: {type(arg)}")
        
        self._arg_is_gpuarray = tuple(is_gpuarray)
        self._values_buffer = [None] * len(args)  # Pre-allocate reusable buffer
        return tuple(types)
    
    def _validate_args(self, args: tuple):
        """
        Validate argument count and types.
        Called during the first _CHECK_THRESHOLD launches to catch bugs early.
        
        Only accepts:
        - GPUArray for pointer parameters
        - numpy scalar types (np.int32, np.float32, etc.) for scalar parameters
        
        Rejects Python int/float to prevent silent overflow/precision loss.
        
        Also verifies that args match the cached pattern (if already built).
        """
        # Check count
        if len(args) != self._num_params:
            raise ValueError(
                f"Kernel '{self.name}' expects {self._num_params} arguments, "
                f"got {len(args)}"
            )
        
        # Check each argument
        for i, (arg, expected_size) in enumerate(zip(args, self._param_sizes)):
            
            arg_is_gpuarray = isinstance(arg, GPUArray)
            if self._arg_is_gpuarray is not None:
                expected_gpuarray = self._arg_is_gpuarray[i]
                if arg_is_gpuarray != expected_gpuarray:
                    expected_type = "GPUArray" if expected_gpuarray else "numpy scalar"
                    got_type = "GPUArray" if arg_is_gpuarray else type(arg).__name__
                    print(f"Argument {i}: expected {expected_type} (from first launch), got {got_type}")
                    raise TypeError(
                        f"Argument {i}: expected {expected_type} (from first launch), "
                        f"got {got_type}. Argument types must be consistent across launches."
                    )

            if isinstance(arg, GPUArray):
                if arg._freed:
                    raise RuntimeError(f"Argument {i}: GPUArray has been freed")
                if expected_size != 8:
                    raise TypeError(
                        f"Argument {i}: GPUArray (pointer) requires 8-byte param, "
                        f"but kernel expects {expected_size}-byte param"
                    )
            elif isinstance(arg, (np.floating, np.integer)):
                # Get the numpy dtype and verify size matches kernel expectation
                dtype = np.dtype(type(arg))
                actual_size = self._NUMPY_DTYPE_SIZE.get(dtype)
                
                if actual_size is None:
                    raise TypeError(
                        f"Argument {i}: unsupported numpy dtype {dtype}. "
                        f"Use one of: {list(self._NUMPY_DTYPE_SIZE.keys())}"
                    )
                
                if actual_size != expected_size:
                    raise TypeError(
                        f"Argument {i}: {dtype} is {actual_size} bytes, "
                        f"but kernel expects {expected_size}-byte param. "
                        f"Use the correct numpy type for this parameter."
                    )
            elif isinstance(arg, (int, float)):
                # Reject Python int/float - require explicit numpy types
                raise TypeError(
                    f"Argument {i}: Python {type(arg).__name__} not allowed. "
                    f"Use numpy scalar (e.g., np.int32({arg}) or np.float32({arg})) "
                    f"to ensure correct size and prevent overflow/precision loss."
                )
            else:
                raise TypeError(
                    f"Argument {i}: unsupported type {type(arg).__name__}. "
                    f"Use GPUArray or numpy scalars (np.int32, np.float32, etc.)."
                )

    
    def _extract_values(self, args: tuple) -> tuple:
        """
        Extract raw values from arguments for kernel launch.
        Uses cached pattern (_arg_is_gpuarray) to avoid isinstance checks.
        Uses pre-allocated buffer to avoid list allocation.
        
        Returns tuple of Python primitives that ctypes can convert:
        - GPUArray -> int (pointer address)
        - numpy scalar -> native Python type via .item()
        """
        buf         = self._values_buffer
        is_gpuarray = self._arg_is_gpuarray
        for i, arg in enumerate(args):
            buf[i] = int(arg._ptr) if is_gpuarray[i] else arg.item()
        return tuple(buf)
        
    @property
    def handle(self):
        return self._function
    
    @property
    def grid(self) -> tuple:
        return self._config.grid
    
    @property
    def block(self) -> tuple:
        return self._config.block
    
    @property
    def shared_mem_bytes(self) -> int:
        return self._config.shared_mem_bytes
    
    @property
    def num_params(self) -> int:
        """Number of kernel parameters."""
        return self._num_params
    
    def _get_internal_stream(self):
        """Lazily create internal stream for blocking launches."""
        if self._internal_stream is None:
            _, handle = cuda.cuStreamCreate(0)
            self._internal_stream = handle
        return self._internal_stream
    
    def launch(self, stream: 'CUDAStream', *args):
        """
        Launch the kernel with its fixed configuration.
        
        Args:
            stream: CUDAStream for async launch, or None for blocking launch.
                    - If None: launches on internal stream and blocks until complete.
                    - If stream: launches async, caller must sync.
            *args: Kernel arguments. Supported types:
                   - GPUArray: passed as device pointer
                   - np.float32(x), np.float64(x): float scalars
                   - np.int32(x), np.uint32(x), etc.: integer scalars
        
        Example:
            kernel.launch(None, gpu_arr, np.int32(n))  # blocking
            kernel.launch(stream, gpu_arr, np.int32(n))  # async
        """
        # Validation phase: check args during first N launches
        if self._times_checked < self._CHECK_THRESHOLD:
            self._validate_args(args)
            if self._param_types is None:
                self._param_types = self._build_param_types(args)
            self._times_checked += 1
        
        # Stream handling
        if stream is None:
            stream_handle = self._get_internal_stream()
            blocking = True
        else:
            stream_handle = stream.handle
            blocking = False
        
        # Build kernel params using efficient (values, types) tuple approach
        # cuda-python handles the packing in C code
        if args:
            values = self._extract_values(args)
            kernel_params = (values, self._param_types)
        else:
            kernel_params = 0
        
        cuda.cuLaunchKernel(
            self._function,
            self._config.grid[0], self._config.grid[1], self._config.grid[2],
            self._config.block[0], self._config.block[1], self._config.block[2],
            self._config.shared_mem_bytes,
            stream_handle,
            kernel_params,
            0
        )
        
        if blocking:
            cuda.cuStreamSynchronize(stream_handle)
    run = launch
    
    def _destroy(self):
        """Clean up internal stream if created."""
        if self._internal_stream is not None:
            cuda.cuStreamDestroy(self._internal_stream)
            self._internal_stream = None


class CUDAContext:
    """
    Per-process CUDA context manager.
    
    Handles context creation (primary or isolated), kernel loading, 
    memory management, and resource cleanup.

    NEED TO CALL __enter__() TO INITIALIZE BEFORE USE. and __exit__() TO CLEANUP.
    
    Usage:
        with CUDAContext(config) as ctx:
            # Memory allocation
            gpu_arr = ctx.m(numpy_array)  # Allocate and copy
            gpu_zeros = ctx.zeros((1000,), np.float32)  # Allocate zeros
            
            # Kernel execution (module_name = .cu file, kernel_name = __global__ function)
            kernel = ctx.get_kernel("my_module", "my_kernel", launch_config)
            stream = ctx.create_stream()
            kernel.launch(stream, arg1, arg2, ...)
            stream.synchronize()
            
            # Copy back
            result = gpu_arr.to_host()
            
            # Explicit free (optional - free_all() called on exit)
            ctx.free(gpu_arr)
    """
    
    def __init__(self, config: dict):
        """
        Initialize CUDA context.
        
        Args:
            config: Configuration dict from CUDAManager.get_config()
        """
        self._config = config
        self._device_id = config["device_id"]
        self._arch = config["arch"]
        self._kernel_dir = Path(config["kernel_dir"])
        self._uses_pytorch = config["uses_pytorch"]
        
        self._context = None
        self._device = None
        self._owns_context = False  # True if we created our own context (not primary)
        
        self._modules = {}  # module_name -> CUmodule (keyed by .cu filename without extension)
        self._streams = []  # Track streams for cleanup
        self._allocations = {}  # ptr -> GPUArray (track for free_all)
        
    def __enter__(self):
        self._initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False
    
    def _initialize(self):
        """Initialize CUDA context."""
        cuda.cuInit(0)
        
        _, self._device = cuda.cuDeviceGet(self._device_id)
        
        if self._uses_pytorch:
            # Use primary context (shared with PyTorch)
            _, self._context = cuda.cuDevicePrimaryCtxRetain(self._device)
            cuda.cuCtxSetCurrent(self._context)
            self._owns_context = False
        else:
            # Create our own isolated context
            _, self._context = cuda.cuCtxCreate(None, 0, self._device)
            self._owns_context = True
    enter = _initialize
    
    def _cleanup(self):
        """Clean up CUDA resources."""
        # Free all GPU allocations first
        self.free_all()
        
        # Destroy streams
        for stream in self._streams:
            stream._destroy()
        self._streams.clear()
        
        # Unload modules
        for module in self._modules.values():
            cuda.cuModuleUnload(module)
        self._modules.clear()
        
        # Release/destroy context
        if self._context is not None:
            if self._owns_context:
                cuda.cuCtxDestroy(self._context)
            else:
                # Release primary context
                cuda.cuDevicePrimaryCtxRelease(self._device)
            self._context = None
    exit  = _cleanup
    leave = _cleanup
    
    # =========================================================================
    # Memory Management
    # =========================================================================
    
    def m(self, arr: np.ndarray, copy: bool = True) -> GPUArray:
        """
        Allocate GPU memory matching a numpy array.
        
        Args:
            arr: Numpy array to match (shape and dtype).
            copy: If True (default), copy data to GPU. If False, only allocate.
        
        Returns:
            GPUArray with allocated device memory.
        
        Raises:
            TypeError: If dtype is bool (use uint8 instead).
            TypeError: If dtype is not supported.
        """
        arr = np.asarray(arr)
        
        # Check for bool - explicit error, no silent conversion
        if arr.dtype == np.bool_:
            raise TypeError(
                "Boolean arrays are not supported. Use uint8 instead:\n"
                "  arr = arr.astype(np.uint8)"
            )
        
        # Check dtype is supported
        if arr.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"Unsupported dtype: {arr.dtype}. Supported: {list(_SUPPORTED_DTYPES.keys())}"
            )
        
        # Ensure contiguous (C-order for speed)
        arr = np.ascontiguousarray(arr)
        
        # Allocate device memory
        nbytes = arr.nbytes
        _, ptr = cuda.cuMemAlloc(nbytes)
        
        # Create wrapper
        gpu_arr = GPUArray(ptr, arr.shape, arr.dtype, self)
        
        # Track allocation
        self._allocations[ptr] = gpu_arr
        
        # Copy data if requested
        if copy:
            cuda.cuMemcpyHtoD(ptr, arr.ctypes.data, nbytes)
        
        return gpu_arr
    
    def zeros(self, shape, dtype=np.float32) -> GPUArray:
        """
        Allocate GPU memory initialized to zeros.
        
        Args:
            shape: Shape tuple (e.g., (1000,) or (100, 64)).
            dtype: Numpy dtype (default: float32).
        
        Returns:
            GPUArray with zero-initialized device memory.
        """
        dtype = np.dtype(dtype)
        
        if dtype == np.bool_:
            raise TypeError("Boolean arrays are not supported. Use uint8 instead.")
        
        if dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"Unsupported dtype: {dtype}")
        
        if isinstance(shape, int):
            shape = (shape,)
        
        size = int(np.prod(shape))
        nbytes = size * dtype.itemsize
        
        # Allocate
        _, ptr = cuda.cuMemAlloc(nbytes)
        
        # Zero-fill
        cuda.cuMemsetD8(ptr, 0, nbytes)
        
        # Create wrapper
        gpu_arr = GPUArray(ptr, tuple(shape), dtype, self)
        self._allocations[ptr] = gpu_arr
        
        return gpu_arr
    
    def empty(self, shape, dtype=np.float32) -> GPUArray:
        """
        Allocate GPU memory without initialization (faster than zeros).
        
        Args:
            shape: Shape tuple.
            dtype: Numpy dtype (default: float32).
        
        Returns:
            GPUArray with uninitialized device memory.
        """
        # Create a dummy array just for shape/dtype, don't copy
        dummy = np.empty(shape if isinstance(shape, tuple) else (shape,), dtype=dtype)
        return self.m(dummy, copy=False)
    
    def free(self, gpu_arr: GPUArray):
        """
        Free a GPU array's device memory.
        
        Args:
            gpu_arr: GPUArray to free.
        """
        if gpu_arr._freed:
            return  # Already freed, no-op
        
        ptr = gpu_arr._ptr
        
        # Free device memory
        cuda.cuMemFree(ptr)
        
        # Remove from tracking
        if ptr in self._allocations:
            del self._allocations[ptr]
        
        # Mark as freed
        gpu_arr._mark_freed()
    
    def free_all(self):
        """
        Free all GPU allocations tracked by this context.
        
        Called automatically on context exit.
        """
        for ptr, gpu_arr in list(self._allocations.items()):
            cuda.cuMemFree(ptr)
            gpu_arr._mark_freed()
        self._allocations.clear()
    
    def copy_d2d(self, src: GPUArray, dst: GPUArray, stream: 'CUDAStream' = None):
        """
        Device-to-device copy.
        
        Args:
            src: Source GPUArray.
            dst: Destination GPUArray (must have same nbytes).
            stream: If provided, copy is async. If None (default), copy is synchronous.
        """
        if src._freed or dst._freed:
            raise RuntimeError("Cannot copy freed GPUArray")
        if src.nbytes != dst.nbytes:
            raise ValueError(f"Size mismatch: src={src.nbytes} bytes, dst={dst.nbytes} bytes")
        
        if stream is None:
            cuda.cuMemcpyDtoD(dst._ptr, src._ptr, src.nbytes)
        else:
            cuda.cuMemcpyDtoDAsync(dst._ptr, src._ptr, src.nbytes, stream.handle)
    
    # =========================================================================
    # Memory Stats
    # =========================================================================
    
    def get_memory_stats(self) -> dict:
        """
        Get comprehensive GPU memory statistics.
        
        Returns a dictionary with:
        - 'device_free_bytes': Free memory on the device (from OS perspective)
        - 'device_total_bytes': Total memory on the device
        - 'device_used_bytes': Used memory on the device (device_total - device_free)
        - 'context_allocated_bytes': Total bytes allocated by THIS context (tracked internally)
        - 'context_num_allocations': Number of active allocations in this context
        - 'context_allocations': List of dicts with details about each allocation
        
        Note: 'device_free_bytes' reflects device-wide usage, including allocations
        from other contexts/processes. 'context_allocated_bytes' is specific to
        this CUDAContext instance.
        
        Returns:
            Dictionary with memory statistics.
        """
        # Query device-wide memory info via cuMemGetInfo
        _, free_bytes, total_bytes = cuda.cuMemGetInfo()
        
        # Calculate context-specific stats from our tracked allocations
        context_allocated = 0
        allocation_details = []
        
        for ptr, gpu_arr in self._allocations.items():
            if not gpu_arr._freed:
                context_allocated += gpu_arr.nbytes
                allocation_details.append({
                    'ptr': hex(int(ptr)),
                    'shape': gpu_arr.shape,
                    'dtype': str(gpu_arr.dtype),
                    'nbytes': gpu_arr.nbytes,
                })
        
        return {
            'device_free_bytes': int(free_bytes),
            'device_total_bytes': int(total_bytes),
            'device_used_bytes': int(total_bytes - free_bytes),
            'context_allocated_bytes': context_allocated,
            'context_num_allocations': len(allocation_details),
            'context_allocations': allocation_details,
        }
    
    def get_memory_stats_formatted(self) -> str:
        """
        Get a human-readable formatted string of GPU memory statistics.
        
        Returns:
            Formatted string with memory stats.
        """
        stats = self.get_memory_stats()
        
        def _fmt_bytes(n: int) -> str:
            """Format bytes as human-readable string."""
            if n >= 1024**3:
                return f"{n / 1024**3:.2f} GB"
            elif n >= 1024**2:
                return f"{n / 1024**2:.2f} MB"
            elif n >= 1024:
                return f"{n / 1024:.2f} KB"
            else:
                return f"{n} B"
        
        lines = [
            "=== GPU Memory Stats ===",
            f"Device Total:    {_fmt_bytes(stats['device_total_bytes'])}",
            f"Device Used:     {_fmt_bytes(stats['device_used_bytes'])}",
            f"Device Free:     {_fmt_bytes(stats['device_free_bytes'])}",
            f"---",
            f"Context Allocated: {_fmt_bytes(stats['context_allocated_bytes'])} ({stats['context_num_allocations']} allocations)",
        ]
        
        if stats['context_allocations']:
            lines.append("Allocations:")
            for i, alloc in enumerate(stats['context_allocations']):
                lines.append(
                    f"  [{i}] {alloc['shape']} {alloc['dtype']} = {_fmt_bytes(alloc['nbytes'])}"
                )
        
        return "\n".join(lines)
    
    def print_memory_stats(self):
        """Print GPU memory statistics to stdout."""
        print(self.get_memory_stats_formatted())
    
    # =========================================================================
    # Stream Management
    # =========================================================================
    
    def create_stream(self) -> CUDAStream:
        """Create a new CUDA stream."""
        _, handle = cuda.cuStreamCreate(0)
        stream = CUDAStream(handle)
        self._streams.append(stream)
        return stream
    stream = create_stream # Alias
    
    def synchronize(self):
        """Synchronize the current context (wait for all work to complete)."""
        cuda.cuCtxSynchronize()
    
    def get_kernel(self, module_name: str, kernel_name: str, config: _LaunchConfig) -> CUDAKernel:
        """
        Get a kernel by name with fixed launch configuration, compiling if necessary.
        
        Args:
            module_name: Name of the .cu file (without extension), e.g., "test_reduction"
            kernel_name: Name of the extern "C" __global__ function, e.g., "test_reduce_sum_1d"
            config: LaunchConfig (1D, 2D, or Grid2D) defining grid, block, and shared memory
            
        Returns:
            CUDAKernel wrapper with the specified configuration
            
        Example:
            # Load kernel "test_reduce_sum_1d" from "test_reduction.cu"
            kernel = ctx.get_kernel("test_reduction", "test_reduce_sum_1d", config)
        """
        # Load module if not already loaded
        if module_name not in self._modules:
            self._load_module(module_name)
        
        module = self._modules[module_name]
        
        # Get function from module
        _, function = cuda.cuModuleGetFunction(module, kernel_name.encode())
        
        return CUDAKernel(function, kernel_name, config)
    
    def _load_module(self, module_name: str):
        """Load a module (compiled .cu file), compiling if necessary.
        
        Args:
            module_name: Name of the .cu file without extension (e.g., "test_reduction")
        """
        cu_file = self._kernel_dir / f"{module_name}.cu"
        cubin_file = self._kernel_dir / "compiled" / f"{module_name}.cubin"
        
        if not cu_file.exists():
            raise FileNotFoundError(f"Kernel source file not found: {cu_file}")
        
        # Compile if needed (with file locking for multi-process safety)
        _, has_compiled = compile_kernel(cu_file, cubin_file, self._arch)
        if has_compiled:
            print(f"\033[33mCompiled module: {cu_file} -> {cubin_file}\033[0m")
        
        # Load the module
        _, module = cuda.cuModuleLoad(str(cubin_file).encode())
        
        self._modules[module_name] = module

    def check_yoself(self):
        import numpy as np
        n = 1000
        arr_initially = np.random.uniform(size=(n,)).astype(np.float32)
        gpu_arr = self.m(arr_initially)
        cpu_arr = gpu_arr.get()
        if not np.allclose(arr_initially, cpu_arr):
            print("\033[31mArrays not equal after transfer to GPU and back!\033[0m")
            return False
        if np.isnan(cpu_arr).any():
            print("\033[31mNANs in array!\033[0m")
            return False
        output_gpu = self.m(np.zeros((1,), dtype=np.float32))
        from .launch_config import DeviceProperties, LaunchConfig1D
        props = DeviceProperties()
        launch_config = LaunchConfig1D(props, n_workers=n)
        kernel = self.get_kernel("test_reduction", "test_reduce_sum_1d", launch_config)
        expected = np.sum(cpu_arr)
        kernel.launch(None, gpu_arr, output_gpu, np.uint32(n))
        result = output_gpu.get()[0]
        if not np.isclose(expected, result):
            print("\033[31mReduction result incorrect!\033[0m")
            print("array before : ", cpu_arr[:10], "...")
            print("array after  : ", gpu_arr.get()[:10], "...")
            print("expected sum: ", expected)
            print("result sum: ", result)
            return False
        return True