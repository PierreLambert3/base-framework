"""
Launch configuration classes for CUDA kernels.

Wiki: wiki/07-cuda-wrapper.md (section 7.5).

Provides LaunchConfig classes that compute optimal launch configurations
based on device properties and shared memory requirements.
"""

from cuda.bindings import driver as cuda


class DeviceProperties:
    """
    Cache of device properties needed for launch configuration.
    Query once and reuse for all LaunchConfig instances.
    """
    
    def __init__(self, device_id: int = 0):
        cuda.cuInit(0)
        _, device = cuda.cuDeviceGet(device_id)
        
        _, self.max_threads_per_block = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device
        )
        _, self.max_shared_memory_per_block = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device
        )
        _, self.max_grid_dim_x = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device
        )
        _, self.max_grid_dim_y = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device
        )
        _, self.max_grid_dim_z = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device
        )
        _, self.warp_size = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, device
        )
        _, self.max_block_dim_x = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device
        )
        _, self.max_block_dim_y = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device
        )
        _, self.max_block_dim_z = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device
        )
        
        # Derived
        self.max_shared_memory_n32bits = self.max_shared_memory_per_block // 4


class _LaunchConfig:
    """
    Base class for kernel launch configurations (internal, do not use directly).
    
    Computes block and grid dimensions along with shared memory requirements.
    Fixed at instantiation time - shapes don't change at launch.
    
    Shared memory formula:
        total_32bits = (threads_per_block * smem_n32bits_per_thread) + smem_const_n32bits
        total_bytes = total_32bits * 4
    """
    
    def __init__(
        self,
        props: DeviceProperties,
        n_workers: int,
        smem_n32bits_per_thread: int = 0,
        smem_const_n32bits: int = 0,
    ):
        self._props = props
        self.n_workers = n_workers
        self._smem_per_thread = smem_n32bits_per_thread
        self._smem_const = smem_const_n32bits
        
        # To be set by subclasses
        self._block = (1, 1, 1)
        self._grid = (1, 1, 1)
        self._shared_mem_n32bits = 0
    
    @property
    def block(self) -> tuple:
        """Block dimensions (bx, by, bz)."""
        return self._block
    
    @property
    def grid(self) -> tuple:
        """Grid dimensions (gx, gy, gz)."""
        return self._grid
    
    @property
    def shared_mem_bytes(self) -> int:
        """Shared memory size in bytes."""
        return self._shared_mem_n32bits * 4
    
    def _compute_shared_mem(self, threads_per_block: int) -> int:
        """Compute shared memory in 32-bit words."""
        return (threads_per_block * self._smem_per_thread) + self._smem_const


class LaunchConfig1D(_LaunchConfig):
    """
    1D launch configuration: 1D block, 1D grid.
    
    Finds optimal block size (multiple of warp size) that respects shared memory limits.
    Maximizes threads per block while staying within constraints.
    
    Example:
        props = DeviceProperties()
        cfg = LaunchConfig1D(props, n_workers=10000, smem_n32bits_per_thread=2)
        kernel.launch(cfg.grid, cfg.block, shared_mem=cfg.shared_mem_bytes)
    """
    
    def __init__(
        self,
        props: DeviceProperties,
        n_workers: int,
        smem_n32bits_per_thread: int = 0,
        smem_const_n32bits: int = 0,
    ):
        super().__init__(props, int(n_workers), int(smem_n32bits_per_thread), int(smem_const_n32bits))
        
        self.block_x: int = 0
        self.block_y: int = 1
        self.n_blocks: int = 0
        
        self._compute_config()
    
    def _compute_config(self):
        """Compute optimal 1D block and grid configuration."""
        warp = self._props.warp_size
        max_threads = self._props.max_threads_per_block
        max_smem = self._props.max_shared_memory_n32bits
        
        n_threads = self.n_workers

        grid_x  = (n_threads + max_threads - 1) // max_threads
        block_x = (n_threads + grid_x - 1) // grid_x
        if block_x % warp != 0:
            block_x += warp - (block_x % warp)
        smem_used = self._compute_shared_mem(block_x)
        while smem_used > max_smem:
            if block_x == 1:
                raise ValueError(
                    f"Cannot fit shared memory even with 1 thread per block. "
                    f"Required: {smem_used * 4} bytes, Max: {max_smem * 4} bytes"
                )
            block_x -= warp
            if block_x <= 0:
                block_x = 1
            smem_used = self._compute_shared_mem(block_x)
        
        # Compute grid size
        n_blocks = (self.n_workers + block_x - 1) // block_x
        
        # Validate grid dimension
        if n_blocks > self._props.max_grid_dim_x:
            raise ValueError(
                f"Grid dimension ({n_blocks}) exceeds device limit ({self._props.max_grid_dim_x}). "
                f"Reduce n_workers or increase block size."
            )
        
        self.block_x = block_x
        self.block_y = 1
        self.n_blocks = n_blocks
        
        self._block = (block_x, 1, 1)
        self._grid = (n_blocks, 1, 1)
        self._shared_mem_n32bits = smem_used
        
        self._sanity_check()
    
    def _sanity_check(self):
        """Verify configuration is valid."""
        bx, by, bz = self._block
        gx, gy, gz = self._grid
        threads_per_block = bx * by * bz
        
        assert self.n_workers <= threads_per_block * gx * gy * gz, "Not enough threads for workers"
        assert threads_per_block == 1 or threads_per_block % self._props.warp_size == 0, "Block size must be warp-aligned"
        assert bx > 0 and gx > 0, "Block and grid dimensions must be positive"


class LaunchConfig2D(_LaunchConfig):
    """
    2D launch configuration: 2D block (block_x x block_y), 1D grid.
    
    User provides block_x (1st dimension / X dimension).
    block_x is rounded UP to the nearest multiple of 32 (buffer threads added if needed).
    block_y (2nd dimension / Y dimension) is maximized within constraints.
    
    Use case: When block_x needs to be a specific size for warp-level operations
    (e.g., reductions along X using warp shuffles).
    
    The kernel should check bounds: threadIdx.x < useful_block_x to skip buffer threads.
    
    Example:
        props = DeviceProperties()
        cfg = LaunchConfig2D(props, n_workers=10000, block_x_requested=50)
        # cfg.useful_block_x = 50 (original), cfg.block[0] = 64 (padded to warp)
    """
    
    def __init__(
        self,
        props: DeviceProperties,
        n_workers: int,
        block_x_requested: int,
        smem_n32bits_per_thread: int = 0,
        smem_const_n32bits: int = 0,
    ):
        super().__init__(props, int(n_workers), int(smem_n32bits_per_thread), int(smem_const_n32bits))
        
        self.block_x: int = int(block_x_requested)
        self.block_y: int = 1
        self.n_blocks: int = 0
        self.buffer: int = 0
        self.useful_block_x: int = int(block_x_requested)  # Original requested value
        
        self._requested_block_x = int(block_x_requested)
        self._compute_config()
    
    def _compute_config(self):
        """Compute optimal 2D block and 1D grid configuration."""
        warp = self._props.warp_size
        max_threads = self._props.max_threads_per_block
        max_smem = self._props.max_shared_memory_n32bits
        
        # Round block_x UP to multiple of warp size
        if (self.block_x % warp) != 0:
            import warnings
            self.buffer = warp - (self.block_x % warp)
            self.block_x += self.buffer
            warnings.warn(
                f"LaunchConfig2D: block_x={self._requested_block_x} not divisible by {warp}, "
                f"added {self.buffer} buffer threads -> block_x={self.block_x}",
                UserWarning
            )
            assert (self.block_x % warp) == 0
        self.usefull_x = self.block_x - self.buffer
        
        self.block_y = 1
        smem_used = self._compute_shared_mem(self.block_x * self.block_y)
        assert smem_used <= max_smem, ( "Initial block_x exceeds shared memory limit" )
        while self.block_x * (self.block_y + 1) <= max_threads and self._compute_shared_mem(self.block_x * (self.block_y + 1)) <= max_smem:
            # Check if increasing block_y still fits in shared memory
            smem_used = self._compute_shared_mem(self.block_x * (self.block_y + 1))
            if smem_used > max_smem:
                break
            self.block_y += 1
        
        useful_per_block = self.usefull_x * self.block_y
        self.n_blocks = (self.n_workers + useful_per_block - 1) // useful_per_block

        self._block = (self.block_x, self.block_y, 1)
        self._grid = (self.n_blocks, 1, 1)
        self._shared_mem_n32bits = self._compute_shared_mem(self.block_x * self.block_y)
        
        self._sanity_check()
    
    def _sanity_check(self):
        """Verify configuration is valid."""
        threads_per_block_all = self.block_x * self.block_y
        threads_per_block_useful = self.useful_block_x * self.block_y
        
        assert threads_per_block_all <= self._props.max_threads_per_block
        assert self.block_x >= self._requested_block_x
        assert (self.block_x % self._props.warp_size) == 0
        assert self.n_workers <= threads_per_block_useful * self.n_blocks
        assert self.block_x > 0 and self.block_y > 0 and self.n_blocks > 0 and threads_per_block_useful > 0


class LaunchConfigGrid2D(_LaunchConfig):
    """
    2D grid configuration: 1D block, 2D grid.
    
    Use case: Processing 2D data where grid_dim_x is a fixed row width and
    grid_dim_y varies based on number of rows.
    
    Example:
        props = DeviceProperties()
        cfg = LaunchConfigGrid2D(props, n_workers=10240, n_workers_per_grid_row=1024)
    """
    
    def __init__(
        self,
        props: DeviceProperties,
        n_workers: int,
        n_workers_per_grid_row: int,
        smem_n32bits_per_thread: int = 0,
        smem_const_n32bits: int = 0,
        block_size_multiple_of: int = 32,
    ):
        super().__init__(props, n_workers, smem_n32bits_per_thread, smem_const_n32bits)
        
        warp = props.warp_size
        if n_workers % n_workers_per_grid_row != 0:
            raise ValueError(f"n_workers ({n_workers}) must be divisible by n_workers_per_grid_row ({n_workers_per_grid_row})")
        if n_workers_per_grid_row % warp != 0:
            raise ValueError(f"n_workers_per_grid_row ({n_workers_per_grid_row}) must be divisible by warp size ({warp})")
        
        self._n_workers_per_row = n_workers_per_grid_row
        self._block_multiple = block_size_multiple_of
        
        self.block_x: int = 0
        self.grid_x: int = 0
        self.grid_y: int = 0
        
        self._compute_config()
    
    def _compute_config(self):
        """Compute 1D block and 2D grid configuration."""
        warp = self._props.warp_size
        max_threads = self._props.max_threads_per_block
        max_smem = self._props.max_shared_memory_n32bits
        
        # Start with block size just above row width, reduce to fit
        block_x = max_threads
        while block_x > self._n_workers_per_row + warp:
            block_x -= warp
        
        # Reduce for shared memory
        smem_used = self._compute_shared_mem(block_x)
        while smem_used > max_smem:
            if block_x <= warp:
                block_x = 1
                smem_used = self._compute_shared_mem(1)
                if smem_used > max_smem:
                    raise ValueError("Cannot fit shared memory even with 1 thread")
                break
            block_x -= warp
            smem_used = self._compute_shared_mem(block_x)
        
        # Validate block_multiple constraint
        if block_x % self._block_multiple != 0:
            raise ValueError(
                f"Block size {block_x} is not divisible by block_size_multiple_of={self._block_multiple}"
            )
        
        self.block_x = block_x
        
        # Compute grid dimensions
        self.grid_x = (self._n_workers_per_row + block_x - 1) // block_x
        self.grid_y = self.n_workers // self._n_workers_per_row
        
        # Validate grid dimensions
        if self.grid_x > self._props.max_grid_dim_x:
            raise ValueError(f"Grid X ({self.grid_x}) exceeds device limit ({self._props.max_grid_dim_x})")
        if self.grid_y > self._props.max_grid_dim_y:
            raise ValueError(f"Grid Y ({self.grid_y}) exceeds device limit ({self._props.max_grid_dim_y})")
        
        self._block = (block_x, 1, 1)
        self._grid = (self.grid_x, self.grid_y, 1)
        self._shared_mem_n32bits = smem_used
        
        # Buffer threads per row (in the last block of each row)
        total_threads_per_row = block_x * self.grid_x
        self._buffer_threads_per_row = total_threads_per_row - self._n_workers_per_row
        
        self._sanity_check()
    
    def _sanity_check(self):
        """Verify configuration is valid."""
        assert self.grid_x <= self._props.max_grid_dim_x
        assert self.grid_y <= self._props.max_grid_dim_y
        
        total_useful = (self.block_x * self.grid_x - self._buffer_threads_per_row) * self.grid_y
        assert self.n_workers == total_useful, f"Worker count mismatch: {self.n_workers} vs {total_useful}"
        assert self.block_x == 1 or self.block_x % self._props.warp_size == 0
