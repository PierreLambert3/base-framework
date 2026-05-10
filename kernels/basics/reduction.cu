/**
 * Parallel reduction kernels for arbitrary-length arrays.
 * 
 * Based on Mark Harris's "Optimizing Parallel Reduction in CUDA" with
 * modern warp shuffle instructions for best performance.
 * 
 * Strategy:
 * - n <= max_block_size: Single-block reduction
 * - n > max_block_size: Two-pass reduction (partial sums + final reduction)
 * 
 * All kernels write result to output[output_index], allowing in-place
 * reduction where result is stored in input[0].
 */

#include "reduction.cuh"


// =============================================================================
// Single-block reduction (n <= block_size, typically 1024)
// =============================================================================

/**
 * Reduce sum for arrays that fit in a single block.
 * 
 * @param input         Input array to reduce
 * @param output        Output array (result written to output[output_index])
 * @param n             Number of elements to reduce
 * @param output_index  Index in output array to write result
 * 
 * Launch: grid=(1,), block=(block_size,) where block_size >= n and is power of 2
 * Shared memory: (block_size / 32) * sizeof(float) bytes
 */
extern "C" __global__ void reduce_sum_single_block(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t n,
    uint32_t output_index
) {
    extern __shared__ float sdata[];
    
    uint32_t tid = threadIdx.x;
    
    // Load with bounds check (threads beyond n contribute 0)
    float val = (tid < n) ? input[tid] : 0.0f;
    
    // Block reduction
    float sum = block_reduce_sum(val, sdata);
    
    // Thread 0 writes result
    if (tid == 0) {
        output[output_index] = sum;
    }
}

/**
 * Reduce mean for arrays that fit in a single block.
 * Same as reduce_sum_single_block but divides by n at the end.
 */
extern "C" __global__ void reduce_mean_single_block(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t n,
    uint32_t output_index
) {
    extern __shared__ float sdata[];
    
    uint32_t tid = threadIdx.x;
    
    // Load with bounds check
    float val = (tid < n) ? input[tid] : 0.0f;
    
    // Block reduction
    float sum = block_reduce_sum(val, sdata);
    
    // Thread 0 writes mean
    if (tid == 0) {
        output[output_index] = sum / (float)n;
    }
}


// =============================================================================
// Two-pass reduction for large arrays (n > max_block_size)
// =============================================================================

/**
 * First pass: Each block reduces its portion using grid-stride loops.
 * Writes partial sums to partials array (one per block).
 * 
 * @param input     Input array to reduce
 * @param partials  Output array for partial sums (size = num_blocks)
 * @param n         Total number of elements
 * 
 * Launch: grid=(num_blocks,), block=(BLOCK_SIZE,)
 * Shared memory: (BLOCK_SIZE / 32) * sizeof(float) bytes
 */
extern "C" __global__ void reduce_sum_partial_blocks(
    const float* __restrict__ input,
    float* __restrict__ partials,
    uint32_t n
) {
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    
    // Grid-stride loop: each thread accumulates multiple elements
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    while (idx < n) {
        sum += input[idx];
        idx += stride;
    }
    
    // Block reduction
    sum = block_reduce_sum(sum, sdata);
    
    // Thread 0 of each block writes partial sum
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = sum;
    }
}

/**
 * Second pass: Reduce partial sums from first pass.
 * Single block reduces num_partials values and writes final result.
 * 
 * @param partials      Array of partial sums from first pass
 * @param output        Output array (result written to output[output_index])
 * @param num_partials  Number of partial sums to reduce
 * @param output_index  Index in output array to write result
 * 
 * Launch: grid=(1,), block=(block_size,) where block_size >= num_partials
 * Shared memory: (block_size / 32) * sizeof(float) bytes
 */
extern "C" __global__ void reduce_partials_finalize(
    const float* __restrict__ partials,
    float* __restrict__ output,
    uint32_t num_partials,
    uint32_t output_index
) {
    extern __shared__ float sdata[];
    
    uint32_t tid = threadIdx.x;
    
    // Load partial sum (or 0 if beyond num_partials)
    float val = (tid < num_partials) ? partials[tid] : 0.0f;
    
    // Block reduction
    float sum = block_reduce_sum(val, sdata);
    
    // Thread 0 writes result
    if (tid == 0) {
        output[output_index] = sum;
    }
}

/**
 * Second pass for mean: Reduce partials and divide by original n.
 * 
 * @param partials      Array of partial sums
 * @param output        Output array
 * @param num_partials  Number of partial sums
 * @param original_n    Original array length (for division)
 * @param output_index  Index to write result
 */
extern "C" __global__ void reduce_partials_finalize_mean(
    const float* __restrict__ partials,
    float* __restrict__ output,
    uint32_t num_partials,
    uint32_t original_n,
    uint32_t output_index
) {
    extern __shared__ float sdata[];
    
    uint32_t tid = threadIdx.x;
    
    // Load partial sum
    float val = (tid < num_partials) ? partials[tid] : 0.0f;
    
    // Block reduction
    float sum = block_reduce_sum(val, sdata);
    
    // Thread 0 writes mean
    if (tid == 0) {
        output[output_index] = sum / (float)original_n;
    }
}
