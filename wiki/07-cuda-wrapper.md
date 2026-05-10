# 7. CUDA wrapper

> Source: [cuda_wrapper/](../cuda_wrapper/) — `manager.py`, `context.py`,
> `compiler.py`, `launch_config.py`, `reduction.py`.

A thin layer over the `cuda-python` driver bindings, designed for
multi-process scientific apps. It provides:

* `CUDAManager` — main-process device detection and config factory.
* `CUDAContext` — per-process context (primary or isolated), kernel
  loading + caching, memory + stream management.
* `GPUArray` — device allocation wrapper with safe host↔device transfers.
* `CUDAKernel` — kernel handle with frozen launch configuration and
  type-checked arguments.
* `LaunchConfig1D / LaunchConfig2D / LaunchConfigGrid2D` — block/grid
  helpers that respect the device's shared memory and warp constraints.
* `ParallelReducer` — drop-in parallel reduction over a `GPUArray`.

## 7.1 The two-process pattern

```
main process                       worker process
─────────────                      ──────────────
CUDAManager(device_id, kernel_dir)
   │
   ├─ create_context()  →  CUDAContext (NOT entered)  ──pickled to child──►
                                                                          ctx.enter()        # in routine()
                                                                          ctx.m(np_arr)
                                                                          ctx.get_kernel(...)
                                                                          kernel.launch(stream, …)
                                                                          ctx.exit()
```

CUDA contexts are **process-local**. Always create them in the main
process, send them down to the child *un-entered*, and call `enter()` /
`exit()` (or use `with` form) inside the child.

## 7.2 `CUDAManager`

```python
mgr = CUDAManager(device_id=0, kernel_dir="kernels")
ctx = mgr.create_context(uses_pytorch=False)   # un-entered
```

* Detects `device_name`, `compute_capability`, `arch` (e.g. `sm_89`).
* `kernel_dir` is the directory containing your `.cu` source files.
  Compiled `.cubin` files are cached under `kernel_dir/compiled/`.
* `uses_pytorch=True` → uses CUDA's *primary* context (shared with
  PyTorch). `False` → creates an isolated context.

## 7.3 `CUDAContext`

Use as a context manager **or** with explicit `enter()` / `exit()`:

```python
ctx = mgr.create_context(uses_pytorch=False)
ctx.enter()
gpu_a = ctx.m(np_arr)              # alloc + copy host→device
gpu_z = ctx.zeros((1024,), np.float32)
stream = ctx.stream()              # create a CUDA stream
kernel = ctx.get_kernel("update_positions", "update_positions", launch_cfg)
kernel.launch(stream, gpu_a, np.uint32(1024))
stream.sync()
host = gpu_a.to_host()
ctx.exit()                         # frees streams, modules, allocations, context
```

Memory management:

* `ctx.m(arr)` / `ctx.zeros(shape, dtype)` return a `GPUArray`.
* `gpu.copy_from(host)`, `gpu.to_host(out=…, stream=…)`, `gpu.zero()`.
* All allocations are tracked; `free_all()` is called automatically on
  `exit()`. You can call `ctx.free(gpu_arr)` manually if memory is tight.
* `ctx.print_memory_stats()` prints device + per-context usage.

## 7.4 Kernels

A `__global__` function in a `.cu` file becomes a `CUDAKernel`:

```python
ctx.get_kernel(module_name, kernel_name, launch_config)
```

* `module_name` = file name without extension (e.g. `"update_positions"`).
  The wrapper compiles `<kernel_dir>/<module>.cu` into
  `<kernel_dir>/compiled/<module>.cubin` via `nvcc`, with file-locking so
  multiple worker processes don't race. Recompiles when any included
  `.cuh` is newer than the cubin.
* `kernel_name` = the `extern "C" __global__` function name.
* `launch_config` = a `LaunchConfig1D/2D/Grid2D` instance pre-computed
  using `DeviceProperties()`.

Argument typing is **strict**:

* `GPUArray` → device pointer.
* `np.uint32(x)`, `np.float32(x)`, etc. → matching ctypes scalar.
* **Python `int` / `float` are rejected** (silent overflow / precision
  hazard).

The first 100 launches validate types and check kernel parameter sizes;
after that the cached pattern is used directly for performance.

```python
kernel.launch(None,    a, b, np.int32(n))   # blocking (uses internal stream)
kernel.launch(stream,  a, b, np.int32(n))   # async; you must stream.sync()
```

## 7.5 Launch configurations

```python
from cuda_wrapper import DeviceProperties, LaunchConfig1D
props = DeviceProperties()
cfg   = LaunchConfig1D(props, n_workers=10_000, smem_n32bits_per_thread=2)
```

* `LaunchConfig1D(n_workers, smem_n32bits_per_thread, smem_const_n32bits)`
  — picks block size to maximise threads/block while honouring shared
  memory limits and warp alignment.
* `LaunchConfig2D(n_workers, block_x_requested, …)` — for kernels that
  need a fixed `block.x` (e.g. warp-shuffle reductions on rows).
  Pads `block_x` up to a warp; you check
  `threadIdx.x < useful_block_x` in the kernel.
* `LaunchConfigGrid2D(n_workers, n_workers_per_grid_row, …)` — 1D block
  with a 2D grid.

## 7.6 `ParallelReducer`

[`cuda_wrapper/reduction.py`](../cuda_wrapper/reduction.py) provides a
re-usable parallel reducer for sums / max / etc. over a `GPUArray`. Used
by reduction kernels under [kernels/basics/](../kernels/basics/).

## 7.7 Where to put your kernels

Put `.cu` files under `kernels/` (or any directory you point `kernel_dir`
at). Helpers can live in subdirs and be `#include`d with quotes:

```cpp
#include "basics/reduction.cuh"
```

The compiler driver follows quoted includes recursively to detect
out-of-date binaries. Compiled cubins go to `kernels/compiled/` (gitignored
or not, your choice).

## 7.8 Sanity check

`CUDAContext.check_yoself()` allocates a small array, copies it back, and
checks the round-trip. The framework calls this once at the start of every
worker `routine()` to fail fast on broken environments.
