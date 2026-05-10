
#pragma once

__device__ __forceinline__ void fill_random_uint32_t(uint32_t* rand_state){
    *rand_state ^= *rand_state << 13u;
    *rand_state ^= *rand_state >> 17u;
    *rand_state ^= *rand_state << 5u;
}

__device__ __forceinline__ float rand_between(uint32_t* rand_state, float min, float max){
    fill_random_uint32_t(rand_state);
    float scale = ((float)rand_state[0]) / 4294967296.0f; // [0.0f, 1.0f[
    return min + scale * (max - min);
}

__device__ __forceinline__ float rand(uint32_t* rand_state){
    fill_random_uint32_t(rand_state);
    return ((float)rand_state[0]) / 4294967296.0f; // [0.0f, 1.0f[
}

__device__ __forceinline__ uint32_t murmurhash3(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

__device__ __forceinline__ void chaoticise_seed(uint32_t* seed){
    seed[0] = murmurhash3(seed[0]);
    fill_random_uint32_t(seed);
    seed[0] = murmurhash3(seed[0]);
}

__device__ __forceinline__ uint32_t generate_unique_seed(uint32_t global_seed, uint32_t tid){
    uint32_t seed = global_seed + (tid + 1) * 0x9e3779b9;
    chaoticise_seed(&seed);
    return seed;
}

__device__ __forceinline__ uint32_t generate_unique_seed_fast(uint32_t global_seed, uint32_t tid){
    uint32_t seed = global_seed + (tid + 1) * 0x9e3779b9;
    seed = murmurhash3(seed);
    return seed;
}
