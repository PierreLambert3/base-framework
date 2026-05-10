/**
 * Test kernels for reduction primitives.
 * These are extern "C" kernels that can be loaded via the CUDA wrapper.
 */

#include "basics/reduction.cuh"

/**
 * Simple 1D block reduction test.
 * Each block reduces its portion of the input array.
 * 
 * @param input   Input array of floats
 * @param output  Output array (one float per block)
 * @param n       Total number of elements
 * 
 * Shared memory: (blockDim.x / 32) * sizeof(float) bytes
 */
extern "C" __global__ void test_reduce_sum_1d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load with bounds check
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Block reduction
    float sum = block_reduce_sum(val, sdata);
    
    // Thread 0 writes result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

/**
 * Grid-stride reduction for large arrays.
 * Each thread accumulates multiple elements before block reduction.
 * 
 * @param input   Input array
 * @param output  Output array (one float per block)
 * @param n       Total number of elements
 * 
 * Shared memory: (blockDim.x / 32) * sizeof(float) bytes
 */
extern "C" __global__ void test_reduce_sum_grid(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float sdata[];
    grid_reduce_sum(input, output, n, sdata);
}

/**
 * 2D block reduction test.
 * Each row of threads (along X) independently reduces its elements.
 * 
 * @param input       Input array, row-major: input[row * row_width + col]
 * @param output      Output array: one value per row (size = n_rows)
 * @param row_width   Number of useful elements per row
 * @param n_rows      Number of rows
 * 
 * Grid: (1, n_rows / blockDim.y)
 * Block: (block_x, block_y) where block_x >= row_width and is multiple of 32
 * 
 * Shared memory: blockDim.y * (blockDim.x / 32) * sizeof(float) bytes
 */
extern "C" __global__ void test_reduce_sum_2d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_width,
    int n_rows
) {
    extern __shared__ float sdata[];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= n_rows) return;
    
    // Load element (buffer threads load 0)
    float val = 0.0f;
    if (threadIdx.x < row_width) {
        val = input[row * row_width + threadIdx.x];
    }
    
    // Row-wise reduction
    float sum = block_reduce_sum_2d(val, sdata, row_width);
    
    // Thread 0 of each row writes result
    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

/**
 * Single-warp 2D reduction (when row_width <= 32).
 * More efficient than block_reduce_sum_2d for small rows.
 * 
 * @param input       Input array, row-major
 * @param output      Output array: one value per row
 * @param row_width   Elements per row (<= 32)
 * @param n_rows      Number of rows
 * 
 * Grid: (1, ceil(n_rows / blockDim.y))
 * Block: (32, block_y)
 * No shared memory needed.
 */
extern "C" __global__ void test_reduce_sum_2d_warp(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_width,
    int n_rows
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= n_rows) return;
    
    // Load element
    float val = 0.0f;
    if (threadIdx.x < row_width) {
        val = input[row * row_width + threadIdx.x];
    }
    
    // Warp reduction (no shared memory)
    float sum = warp_reduce_sum_2d(val, row_width);
    
    // Lane 0 writes
    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}
