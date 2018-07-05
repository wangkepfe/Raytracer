#pragma once

#include "material.cuh"
#include "ray.cuh"

// generate a cosine weighted hemisphere sampler : generate uniform points on a disk, and then project them up to the hemisphere
__device__ static float3 cosineSampleHemisphere(float u1, float u2){
    float r = sqrtf(u1);
    float theta = 2 * M_PI * u2;
 
    float x = r * cos(theta);
    float y = r * sin(theta);
 
    return make_float3(x, y, sqrtf(max(0.0f, 1 - u1)));
}

__device__ static void diffuseShader(
    Ray& r,
    float3& mask,
    const float3& x,
    const float3& n,
    const Material& mat,
    curandState_t* randstates,
    int idx
) {
    float r1 = curand_uniform(&randstates[idx]);
    float r2 = curand_uniform(&randstates[idx]);
    float3 rdSamp = cosineSampleHemisphere(r1, r2);

    float3 w = n; 
    float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
    float3 v = cross(w, u);

    float3 d = normalize(u * rdSamp.x + v * rdSamp.y + w * rdSamp.z);

    r.orig = x + n * 0.05f;     // offset ray origin slightly to prevent self intersection
    r.dir = d;

    mask *= mat.col;       // multiply with colour of object       
    mask *= dot(d, n);      // weigh light contribution using cosine of angle between incident light and normal
    mask *= 2;              // fudge factor
}