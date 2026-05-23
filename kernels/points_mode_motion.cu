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


#define NESTEROV_COEFF 1.6f
#define MOMENTUM 0.97f

#define ATTRACTION_MULTIPLIER 0.5f
#define ENDPOINTEDNESS 0.5f
#define NOISE_INTENSITY 0.201f


__device__ __forceinline__ void random_impulse(
    uint32_t* seed_thd,
    float* vx, float* vy, float* vz,
    float scale, float dt
) {
    float r = rand(seed_thd) * 2.0f - 1.0f;
    scale *= dt;
    vx[0] += scale * r * r * r * 30.0;
    r = rand(seed_thd) * 2.0f - 1.0f;
    vy[0] += scale * r * r * r * 22.0;
    r = rand(seed_thd) * 2.0f - 1.0f;
    vz[0] += scale * r * r * r * 22.0;
}


__device__ void attract_to_assigned_line(
    float px, float py, float pz,
    float* vx, float* vy, float* vz,
    int32_t line_idx,
    const float* lines,
    float scale,
    float dt, uint32_t* seed_thd
) {
    const uint32_t l9 = ((uint32_t)line_idx) * 9u;
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
    const float qz = az + t * d_z + 14.0f; // line is a bit above the page

    // attraction toward the projection
    scale *= dt;
    vx[0] += scale * (qx - px);
    vy[0] += scale * (qy - py);
    vz[0] += scale * (qz - pz);
}


__device__ void perturbation_from_normal_vector(
    float px, float py, float pz,
    float* vx, float* vy, float* vz,
    int32_t line_idx,
    const float* lines,
    float upwards_mul, float dt,
    uint32_t* seed_thd
) {
    const float nx = lines[line_idx * 9u + 6];
    const float ny = lines[line_idx * 9u + 7];
    const float nz = lines[line_idx * 9u + 8];

    float intensity = upwards_mul * dt;
    if(intensity < 0.01f) {return;}

    float r = rand(seed_thd);
    r *= r * 1.2f;
    r *= rand(seed_thd);
    r *= r * 1.2f;
    r *= rand(seed_thd);
    r *= r * 1.2f;
    r *= 0.02f;


    vx[0] += intensity * r * nx;
    vy[0] += intensity * r * ny;
    vz[0] += intensity * r * nz;

}

extern "C" __global__ void points_mode_motion(
    float*         __restrict__ positions,
    float*         __restrict__ velocities,
    const int32_t* __restrict__ lines_idx,
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
        NOISE_INTENSITY * point_mods[i * 5u + 1], dt);

    // attraction to assigned line
    attract_to_assigned_line(px_nest, py_nest, pz_nest, &vx, &vy, &vz,
        lines_idx[i], lines, 
        ATTRACTION_MULTIPLIER * point_mods[i * 5u + 0], dt, &seed_thd);

    // normal vector can perturbate in that direction
    perturbation_from_normal_vector(px_nest, py_nest, pz_nest, &vx, &vy, &vz,
        lines_idx[i], lines, point_mods[i * 5u + 4], dt, &seed_thd);

    // ...

    
    

    /*
    // Per-point modulation: [spring, jitter, dt, damping, upwards]
    

    const float k_a     = attraction_multiplier * spring_mul;
    const float k_d     = momentum             * damping_mul;
    const float dt_i    = dt          * dt_mul;
    const float noise_i = noise_sigma * jitter_mul;

    // Load assigned line endpoints and normal vector (stride 9: [ax,ay,az, bx,by,bz, nx,ny,nz])
    const int32_t li = line_idx[i];
    const uint32_t l9 = ((uint32_t)li) * 9u;
    const float ax = lines[l9 + 0];
    const float ay = lines[l9 + 1];
    const float az = lines[l9 + 2];
    const float bx = lines[l9 + 3];
    const float by = lines[l9 + 4];
    const float bz = lines[l9 + 5];
    const float nx = lines[l9 + 6];
    const float ny = lines[l9 + 7];
    const float nz = lines[l9 + 8];

    // Predicted next position (used for projection target + endpoint pick)
    const float nestrov_mul = 0.2f; 
    const float px_next = px + vx * dt_i * nestrov_mul;
    const float py_next = py + vy * dt_i * nestrov_mul;
    const float pz_next = pz + vz * dt_i * nestrov_mul;

    // Direction of the segment
    const float dx   = bx - ax;
    const float dy   = by - ay;
    const float dz   = bz - az;
    const float len2 = dx * dx + dy * dy + dz * dz + 1e-12f;

    // Orthogonal projection onto the segment, clamped to [0, 1]
    float t = ((px_next - ax) * dx + (py_next - ay) * dy + (pz_next - az) * dz) / len2;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    const float qx = ax + t * dx;
    const float qy = ay + t * dy;
    const float qz = az + t * dz;

    // Pick the endpoint closer to p_next (squared distance)
    const float dax = ax - px_next;
    const float day = ay - py_next;
    const float daz = az - pz_next;
    const float dbx = bx - px_next;
    const float dby = by - py_next;
    const float dbz = bz - pz_next;
    const float dA2 = dax * dax + day * day + daz * daz;
    const float dB2 = dbx * dbx + dby * dby + dbz * dbz;
    const float qEndx = (dA2 <= dB2) ? ax : bx;
    const float qEndy = (dA2 <= dB2) ? ay : by;
    const float qEndz = (dA2 <= dB2) ? az : bz;


    // Attraction acceleration toward the projection + endpoint bias
    float acc_x = k_a * ((qx - px_next) + endpointedness * (qEndx - px_next));
    float acc_y = k_a * ((qy - py_next) + endpointedness * (qEndy - py_next));
    float acc_z = k_a * ((qz - pz_next) + endpointedness * (qEndz - pz_next));

    // Upwards interaction: attraction toward projection of CURRENT position,
    // intensity |upwards_mul| / r^2, conditional on sign(dot(p-q, normal)).
    if (upwards_mul != 0.0f) {
        // Projection of current (non-Nesterov) position onto the segment
        float t_cur = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / len2;
        t_cur = fminf(fmaxf(t_cur, 0.0f), 1.0f);
        const float qx_cur = ax + t_cur * dx;
        const float qy_cur = ay + t_cur * dy;
        const float qz_cur = az + t_cur * dz;

        const float dvx = px - qx_cur;
        const float dvy = py - qy_cur;
        const float dvz = pz - qz_cur;
        const float d2_cur = dvx * dvx + dvy * dvy + dvz * dvz + 1e-12f;
        const float dot_n  = dvx * nx + dvy * ny + dvz * nz;

        if ((upwards_mul > 0.0f && dot_n > 0.0f) || (upwards_mul < 0.0f && dot_n < 0.0f)) {
            // scale = |upwards_mul| / r^2 * (1/r) = |upwards_mul| / r^3
            // direction toward projection = -d_vec / r, so acc = scale * (-d_vec)
            const float abs_mul = fabsf(upwards_mul) ;//    * (8.0f * dot_n * dot_n);
            float scale   = abs_mul * rsqrtf(d2_cur) / d2_cur;

            scale /= (d2_cur + 1e-8f);

            acc_x += scale * (-dvx);
            acc_y += scale * (-dvy);
            acc_z += scale * (-dvz);
        }
    }

    // Per-step jitter (per-thread RNG seeded by tid XOR seed)
    uint32_t seed = generate_unique_seed_fast(seed_global, i);
    acc_x += (rand(&seed) * 2.0f - 1.0f) * noise_i;
    acc_y += (rand(&seed) * 2.0f - 1.0f) * noise_i;
    acc_z += (rand(&seed) * 2.0f - 1.0f) * noise_i;

    // Damped semi-implicit Euler integration
    float damp = 1.0f - k_d * dt_i;
    if (damp < 0.0f) damp = 0.0f;
    vx = vx * damp + acc_x * dt_i;
    vy = vy * damp + acc_y * dt_i;
    vz = vz * damp + acc_z * dt_i;
    */

    /* float norm2_v = vx * vx + vy * vy + vz * vz; 
    float threshold = 10.0f;
    if(norm2_v > threshold){
        float norm_v = sqrtf(norm2_v);
        vx *= 0.6f;
        vy *= 0.6f;
        vz *= 0.6f;
    } */


    positions[p3 + 0]  = px + vx * dt;
    positions[p3 + 1]  = py + vy * dt;
    positions[p3 + 2]  = pz + vz * dt;
    velocities[p3 + 0] = vx;
    velocities[p3 + 1] = vy;
    velocities[p3 + 2] = vz;
}
