/**
 * Advances a swarm of 3D points whose motion is biased by an attraction
 * impulse toward the orthogonal projection of each point onto its assigned
 * line segment. One thread per point.
 *
 * Per-point modulation: `point_mods` is a packed (N, 5) float32 array with
 * column layout [spring, jitter, dt, damping, upwards]. Each thread multiplies
 * the global k_attract / noise_sigma / dt / k_damping by its own row.
 *
 * Endpoint bias: the attraction target is the orthogonal projection q plus
 * `alpha * (nearest_endpoint - p_next)`. The "nearest endpoint" is chosen
 * from the predicted next position `p_next = p + v * dt_i` rather than p.
 *
 * Next-position formulation: both the projection target and the nearest-end
 * choice use `p_next`, not `p`. The damped Euler integration step still
 * updates the original (px, py, pz).
 *
 * Line data: `lines` is a packed (N_lines, 9) float32 array with column layout
 * [ax, ay, az, bx, by, bz, nx, ny, nz] where (nx, ny, nz) is the "up" normal
 * vector for the line. Stride is 9 floats per line.
 *
 * Upwards interaction (col 4 of point_mods): when upwards_mul != 0, a
 * conditional attraction toward the orthogonal projection of the CURRENT
 * (non-Nesterov) position is applied, with intensity |upwards_mul| / r^2,
 * only when sign(dot(point - proj_current, normal)) matches sign(upwards_mul).
 */

#include <stdint.h>
#include "basics/randoms.cuh"


#define MOMENTUM 0.9f

// NESTEROV_COEFF should be the time constant of MOMENTUM, in iterations (time until 1 becomes 1/e)
#define NESTEROV_COEFF 0.67f


#define ATTRACTION_MULTIPLIER 0.04f
#define NEIGHATTRACT_SCALE 0.92f
#define NOISE_INTENSITY 0.5f
#define UPAWARDS_INTENSITY 0.005f


__device__ void perturbation_from_normal_vector(
    float px, float py, float pz,
    float* vx, float* vy, float* vz,
    uint32_t line_idx,
    const float* lines,
    float upwards_mul,
    uint32_t* seed_thd
) {
    if(upwards_mul < 0.001f) {return;}
    float scale = upwards_mul * UPAWARDS_INTENSITY;

    const float nx = lines[line_idx * 9u + 6];
    const float ny = lines[line_idx * 9u + 7];
    const float nz = lines[line_idx * 9u + 8];

    float r = rand(seed_thd) * 2.0f - 1.0f;
    vx[0] += scale * fabsf(r * r * r * r) * 2.0f * nx * 2.0f;
    r = rand(seed_thd) * 2.0f - 1.0f;
    vy[0] += scale * fabsf(r * r * r * r) * 2.0f * ny * 2.0f;
    r = rand(seed_thd) * 2.0f - 1.0f;
    vz[0] += scale * fabsf(r * r * r * r) * 2.0f * nz * 2.0f;
}

__device__ __forceinline__ void random_impulse(
    uint32_t* seed_thd,
    float* vx, float* vy, float* vz,
    float scale
) {
    float r = rand(seed_thd) * 2.0f - 1.0f;
    vx[0] += scale * r * fabsf(r) * 2.0;
    r = rand(seed_thd) * 2.0f - 1.0f;
    vy[0] += scale * r * fabsf(r) * 2.0;
    r = rand(seed_thd) * 2.0f - 1.0f;
    vz[0] += scale * r * fabsf(r) * 2.0;
}


__device__ void attract_to_assigned_line(
    float px, float py, float pz,
    float* vx, float* vy, float* vz,
    uint32_t line_idx,
    const float* lines,
    float scale,
    uint32_t* seed_thd
) {
    const uint32_t l9 = line_idx * 9u;
    // point a and b are the endpoints of the segment
    const float ax = lines[l9 + 0];
    const float ay = lines[l9 + 1];
    const float az = lines[l9 + 2];
    const float bx = lines[l9 + 3];
    const float by = lines[l9 + 4];
    const float bz = lines[l9 + 5];

    // line direction vector
    const float d_x = bx - ax;
    const float d_y = by - ay;
    const float d_z = bz - az;
    const float len2 = d_x * d_x + d_y * d_y + d_z * d_z + 1e-12f;

    // projection
    float t = ((px - ax) * d_x + (py - ay) * d_y + (pz - az) * d_z) / len2;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    const float qx = ax + t * d_x;
    const float qy = ay + t * d_y;
    const float qz = az + t * d_z + 5.0f; // line is a bit above the page

    // attraction toward the projection
    vx[0] += scale * (qx - px);
    vy[0] += scale * (qy - py);
    vz[0] += scale * (qz - pz);
}

__device__ void chainlike_attraction(
    const float* positions,
    uint32_t n_points,
    float px, float py, float pz,
    float* vx, float* vy, float* vz,
    const uint32_t* lines_idx,
    uint32_t line_idx,
    const float* lines,
    uint32_t* seed_thd
) {
    float scale = NEIGHATTRACT_SCALE;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {return;}
    
    // attract to two first neighbours
    uint32_t neighbour1 = i+1u;
    uint32_t neighbour2 = i-1u;
    if (neighbour1 >= n_points || neighbour2 >= n_points || neighbour1 < 0 || neighbour2 < 0) {
        /* neighbour1 = (neighbour1 + n_points) % n_points;
        neighbour2 = (neighbour2 + n_points) % n_points; */
        return; // no wrap-around
    }
    float vx_neigh1 = positions[neighbour1 * 3u + 0] - px;
    float vy_neigh1 = positions[neighbour1 * 3u + 1] - py;
    float vz_neigh1 = positions[neighbour1 * 3u + 2] - pz;
    float vx_neigh2 = positions[neighbour2 * 3u + 0] - px;
    float vy_neigh2 = positions[neighbour2 * 3u + 1] - py;
    float vz_neigh2 = positions[neighbour2 * 3u + 2] - pz;
    vx[0] += scale * (vx_neigh1 + vx_neigh2);
    vy[0] += scale * (vy_neigh1 + vy_neigh2);
    vz[0] += scale * (vz_neigh1 + vz_neigh2);
    
    ///*

    // repulsion from the -2 and +2 neighbours
    float repulsion = 0.2f;
    float beta = 1.0f;
    uint32_t neighbour1_ = i+2u;
    uint32_t neighbour2_ = i-2u;
    if (neighbour1_ >= n_points || neighbour2_ >= n_points || neighbour1_ < 0 || neighbour2_ < 0) {
        /* neighbour1_ = (neighbour1_ + n_points) % n_points;
        neighbour2_ = (neighbour2_ + n_points) % n_points; */
        return; // no wrap-around
    }
    vx_neigh1 = -positions[neighbour1_ * 3u + 0] + px;
    vy_neigh1 = -positions[neighbour1_ * 3u + 1] + py;
    vz_neigh1 = -positions[neighbour1_ * 3u + 2] + pz;
    vx_neigh2 = -positions[neighbour2_ * 3u + 0] + px;
    vy_neigh2 = -positions[neighbour2_ * 3u + 1] + py;
    vz_neigh2 = -positions[neighbour2_ * 3u + 2] + pz;
    float vtot_sq_rep = (vx_neigh1 + vx_neigh2) * (vx_neigh1 + vx_neigh2) +  
                (vy_neigh1 + vy_neigh2) * (vy_neigh1 + vy_neigh2) +  
                (vz_neigh1 + vz_neigh2) * (vz_neigh1 + vz_neigh2);
    float norm = 1.0f / ((__frsqrt_rn(beta*vtot_sq_rep+1e-4f)));
    float lr = repulsion * scale * (1.0f / (1.0f + norm));
    vx[0] += lr * (vx_neigh1 + vx_neigh2);
    vy[0] += lr * (vy_neigh1 + vy_neigh2);
    vz[0] += lr * (vz_neigh1 + vz_neigh2);

    // repulsion from the -3 and +3 neighbours
    repulsion *= 0.25f;
    neighbour1_ = i+3u;
    neighbour2_ = i-3u;
    if (neighbour1_ >= n_points || neighbour2_ >= n_points || neighbour1_ < 0 || neighbour2_ < 0) {
        /* neighbour1_ = (neighbour1_ + n_points) % n_points;
        neighbour2_ = (neighbour2_ + n_points) % n_points; */
        return; // no wrap-around
    }
    vx_neigh1 = -positions[neighbour1_ * 3u + 0] + px;
    vy_neigh1 = -positions[neighbour1_ * 3u + 1] + py;
    vz_neigh1 = -positions[neighbour1_ * 3u + 2] + pz;
    vx_neigh2 = -positions[neighbour2_ * 3u + 0] + px;
    vy_neigh2 = -positions[neighbour2_ * 3u + 1] + py;
    vz_neigh2 = -positions[neighbour2_ * 3u + 2] + pz;
    vtot_sq_rep = (vx_neigh1 + vx_neigh2) * (vx_neigh1 + vx_neigh2) +  
                (vy_neigh1 + vy_neigh2) * (vy_neigh1 + vy_neigh2) +  
                (vz_neigh1 + vz_neigh2) * (vz_neigh1 + vz_neigh2);
    norm = 1.0f / __frsqrt_rn(beta*vtot_sq_rep+1e-6f+1e-4f);
    lr = repulsion * scale * (1.0f / (1.0f + norm));
    vx[0] += lr * (vx_neigh1 + vx_neigh2);
    vy[0] += lr * (vy_neigh1 + vy_neigh2);
    vz[0] += lr * (vz_neigh1 + vz_neigh2);

    //*/

}


extern "C" __global__ void points_mode_motion(
    float*         __restrict__ positions,
    float*         __restrict__ velocities,
    const uint32_t* __restrict__ lines_idx,
    const float*   __restrict__ lines,
    const float*   __restrict__ point_mods,   // (N, 5): [spring, jitter, dt, damping, upwards]
    uint32_t n_points,
    float    dt,
    uint32_t seed_global
) {
    /*
    const float spring_mul  = point_mods[m5 + 0];
    const float jitter_mul  = point_mods[m5 + 1];
    const float dt_mul      = point_mods[m5 + 2];
    const float damping_mul = point_mods[m5 + 3];
    const float upwards_mul = point_mods[m5 + 4];
    */
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {return;}

    // point position and velocity
    const uint32_t p3 = i * 3u;
    float px = positions[p3 + 0]; float py = positions[p3 + 1]; float pz = positions[p3 + 2];
    float vx = velocities[p3 + 0]; float vy = velocities[p3 + 1]; float vz = velocities[p3 + 2];    
    uint32_t line_idx = lines_idx[i];

    // decay velocity
    vx *= MOMENTUM; vy *= MOMENTUM; vz *= MOMENTUM;

    // Nesterov location and dt adjustment
    dt *= point_mods[i * 5u + 2]; 
    float px_nest = px + vx * dt * NESTEROV_COEFF;
    float py_nest = py + vy * dt * NESTEROV_COEFF;
    float pz_nest = pz + vz * dt * NESTEROV_COEFF;

    // random perturbation
    uint32_t seed_thd = generate_unique_seed_fast(seed_global, i);
    random_impulse(&seed_thd, &vx, &vy, &vz, 
        NOISE_INTENSITY * point_mods[i * 5u + 1]);

    // attraction to assigned line
    attract_to_assigned_line(px_nest, py_nest, pz_nest, &vx, &vy, &vz,
        line_idx, lines, 
        ATTRACTION_MULTIPLIER * point_mods[i * 5u + 0], &seed_thd);

    // normal vector can perturbate in that direction
    perturbation_from_normal_vector(px_nest, py_nest, pz_nest, &vx, &vy, &vz,
        line_idx, lines, point_mods[i * 5u + 4], &seed_thd);


    // every point in a warp attract to the two +1/-1 points
    chainlike_attraction(positions, n_points, px_nest, py_nest, pz_nest, &vx, &vy, &vz, lines_idx,
        line_idx, lines, &seed_thd);
    
    float threshold = 160.0f;
    float sq_norm = vx * vx + vy * vy + vz * vz;
    if(sq_norm > threshold * threshold) {
        float inv_norm = __frsqrt_rn(sq_norm + 1e-6f);
        vx *= threshold * inv_norm;
        vy *= threshold * inv_norm;
        vz *= threshold * inv_norm;
    }
    
    positions[p3 + 0]  = px + vx * dt;
    positions[p3 + 1]  = py + vy * dt;
    positions[p3 + 2]  = pz + vz * dt;
    velocities[p3 + 0] = vx;
    velocities[p3 + 1] = vy;
    velocities[p3 + 2] = vz;
}
