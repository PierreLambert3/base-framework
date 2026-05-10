/**
 * update_positions.cu
 *
 * Wiki: wiki/07-cuda-wrapper.md (kernel compilation pipeline),
 *       wiki/06-worker-instances.md (called from CustomWorker.run_simulation_chunk),
 *       wiki/09-example-walkthrough.md.
 *
 * Advances 2D points by `n_steps` timesteps with constant velocities, bouncing
 * off the unit square [0, 1]^2 walls. Each thread handles one point.
 *
 * Layout:
 *   positions[i*2 + 0] = x_i,   positions[i*2 + 1] = y_i
 *   velocities[i*2 + 0] = vx_i, velocities[i*2 + 1] = vy_i
 */

#include <stdint.h>


extern "C" __global__ void update_positions(
    float* __restrict__ positions,
    float* __restrict__ velocities,
    uint32_t n_points,
    float    dt,
    uint32_t n_steps
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    float x  = positions[i * 2 + 0];
    float y  = positions[i * 2 + 1];
    float vx = velocities[i * 2 + 0];
    float vy = velocities[i * 2 + 1];

    for (uint32_t s = 0; s < n_steps; ++s) {
        x += vx * dt;
        y += vy * dt;

        // Mirror-bounce on x
        if (x < 0.0f) { x = -x;        vx = -vx; }
        if (x > 1.0f) { x = 2.0f - x;  vx = -vx; }
        // Mirror-bounce on y
        if (y < 0.0f) { y = -y;        vy = -vy; }
        if (y > 1.0f) { y = 2.0f - y;  vy = -vy; }
    }

    positions[i * 2 + 0]  = x;
    positions[i * 2 + 1]  = y;
    velocities[i * 2 + 0] = vx;
    velocities[i * 2 + 1] = vy;
}
