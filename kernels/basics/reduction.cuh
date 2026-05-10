/**
 * Optimized parallel reduction primitives.
 * 
 * Based on Mark Harris's "Optimizing Parallel Reduction in CUDA" with
 * modern warp shuffle instructions for best performance.
 * 
 * Key optimizations:
 * 1. Sequential addressing (no shared memory bank conflicts)
 * 2. First add during global load (halves thread blocks needed)
 * 3. Warp shuffle for final warp (no shared memory, no __syncthreads)
 * 4. Loop unrolling for known block sizes
 */

#pragma once

// =============================================================================
// Warp-level reductions (using shuffle, no shared memory needed)
// =============================================================================

/**
 * Warp reduction sum using shuffle down.
 * All threads in the warp must participate.
 * Result is valid only in lane 0.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Warp reduction sum with specified width (must be power of 2, <= 32).
 * Useful when reducing fewer than 32 elements.
 */
__device__ __forceinline__ float warp_reduce_sum_n(float val, int width) {
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// =============================================================================
// Block-level reductions (1D block)
// =============================================================================

/**
 * Block reduction sum for 1D blocks.
 * 
 * @param val     Thread's value to reduce
 * @param shared  Shared memory array of size (blockDim.x / 32) floats
 * @return        Sum of all values (valid only in thread 0)
 * 
 * Requires: blockDim.x must be a multiple of 32 (warp size)
 */
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & 31;        // threadIdx.x % 32
    int wid = threadIdx.x >> 5;         // threadIdx.x / 32
    int nwarps = blockDim.x >> 5;       // blockDim.x / 32
    
    // Each warp reduces to its lane 0
    val = warp_reduce_sum(val);
    
    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // First warp reduces the warp sums
    // Only first nwarps threads participate
    val = (threadIdx.x < nwarps) ? shared[threadIdx.x] : 0.0f;
    
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// =============================================================================
// 2D Block reductions (reduce along X dimension for each Y row)
// =============================================================================

/**
 * Row-wise reduction for 2D blocks.
 * Each row (threadIdx.y) independently reduces its X dimension.
 * 
 * @param val     Thread's value to reduce
 * @param shared  Shared memory: float[blockDim.y][blockDim.x / 32]
 *                Layout: shared[threadIdx.y * (blockDim.x/32) + warp_in_row]
 * @param row_width  Useful width of each row (threads beyond this contribute 0)
 * @return        Row sum (valid only in threadIdx.x == 0 of each row)
 * 
 * Requires: blockDim.x must be a multiple of 32
 * 
 * Use case: Multiple independent reductions in parallel, e.g., reducing
 * columns of a matrix where each row of threads handles one column.
 */
__device__ __forceinline__ float block_reduce_sum_2d(float val, float* shared, int row_width) {
    int lane = threadIdx.x & 31;
    int wid_in_row = threadIdx.x >> 5;          // Which warp within this row
    int warps_per_row = blockDim.x >> 5;
    
    // Zero out buffer threads (beyond useful row_width)
    if (threadIdx.x >= row_width) {
        val = 0.0f;
    }
    
    // Warp reduction
    val = warp_reduce_sum(val);
    
    // Shared memory offset for this row
    int row_shared_offset = threadIdx.y * warps_per_row;
    
    // Lane 0 of each warp writes to shared
    if (lane == 0) {
        shared[row_shared_offset + wid_in_row] = val;
    }
    __syncthreads();
    
    // First warp of each row reduces warp sums
    val = (threadIdx.x < warps_per_row) ? shared[row_shared_offset + threadIdx.x] : 0.0f;
    
    if (wid_in_row == 0) {
        val = warp_reduce_sum_n(val, warps_per_row);
    }
    
    return val;
}

/**
 * Simpler 2D reduction when blockDim.x <= 32 (single warp per row).
 * No shared memory needed.
 * 
 * @param val       Thread's value
 * @param row_width Useful width (threads >= row_width contribute 0)
 * @return          Row sum (valid in threadIdx.x == 0)
 */
__device__ __forceinline__ float warp_reduce_sum_2d(float val, int row_width) {
    // Zero buffer threads
    if (threadIdx.x >= row_width) {
        val = 0.0f;
    }
    
    // Reduce within warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    return val;
}

// =============================================================================
// Full array reduction kernel helpers
// =============================================================================

/**
 * Grid-stride loop reduction with first-add-during-load optimization.
 * Reduces a large array to one value per block.
 * 
 * @param input   Input array
 * @param output  Output array (one element per block)
 * @param n       Number of elements
 * @param shared  Shared memory for block reduction
 * 
 * Launch with: grid = (num_blocks,), block = (block_size,)
 * where block_size is multiple of 32
 */
__device__ __forceinline__ void grid_reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    float* shared
) {
    float sum = 0.0f;
    
    // Grid-stride loop: each thread sums multiple elements
    // This is the "first add during load" optimization
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (idx < n) {
        sum += input[idx];
        idx += stride;
    }

    
    
    // Block reduction
    sum = block_reduce_sum(sum, shared);
    // sum += 200.0f;   // purposeful failure

    // Thread 0 writes block result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}
