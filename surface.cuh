#pragma once

#include "cutil_math.h"

#include <curand.h>
#include <curand_kernel.h>

// generate a cosine weighted hemisphere sampler : generate uniform points on a disk, and then project them up to the hemisphere
__device__ static float3 cosineSampleHemisphere(float u1, float u2){
    float r = sqrtf(u1);
    float theta = 2 * M_PI * u2;
 
    float x = r * cos(theta);
    float y = r * sin(theta);
 
    return make_float3(x, y, sqrtf(max(0.0f, 1 - u1)));
}

__device__ static void diffuseSurface(
    float3& rayOrig,
    float3& rayDir,
    float3& color,
    const float3& hitPoint,
    const float3& normalAtHitPoint,
    const float3& materialColor,
    curandState_t* randstates,
    int idx
) {
    float r1 = curand_uniform(&randstates[idx]);
    float r2 = curand_uniform(&randstates[idx]);
    float3 rdSamp = cosineSampleHemisphere(r1, r2);

    float3 w = normalAtHitPoint; 
    float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
    float3 v = cross(w, u);

    float3 d = normalize(u * rdSamp.x + v * rdSamp.y + w * rdSamp.z);

    rayOrig = hitPoint + normalAtHitPoint * 0.05f;     // offset ray origin slightly to prevent self intersection
    rayDir = d;

    color *= materialColor;       // multiply with colour of object       
    color *= dot(d, normalAtHitPoint);      // weigh light contribution using cosine of angle between incident light and normal
    color *= 2;              // fudge factor
}