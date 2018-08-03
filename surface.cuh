#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include "cutil_math.h"
#include "implicitGeometry.cuh"

__device__ inline static float3 uniformHemisphereSample(float r1, float r2){
    // x^2 + y^2 = 1
    float x = sqrtf(r1);
    float y = sqrtf(max(0.0f, 1.0f - r1));

    float theta = 2.0f * M_PI * r2;

    return make_float3(
        x * cos(theta),
        x * sin(theta),
        y
    );
}

__device__ static void diffuseSurface(
    Ray& ray,
    float3& colorMask,
    const float3& hitPoint,
    const float3& normalAtHitPoint,
    const float3& materialColor,
    curandState_t* randstates,
    int idx
) {
    // defined parameters
    float fudgeFactor = 1.6f;

    // random sample
    float r1 = curand_uniform(&randstates[idx]);
    float r2 = curand_uniform(&randstates[idx]);
    float3 rdSamp = uniformHemisphereSample(r1, r2);

    // uvw coords
    float3 w = normalAtHitPoint; 
    float3 u = normalize(cross((fabs(w.x) > 0.1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
    float3 v = cross(w, u);

    // random out direction
    float3 d = normalize(u * rdSamp.x + v * rdSamp.y + w * rdSamp.z);

    ray.orig = hitPoint + normalAtHitPoint * M_EPSILON;
    ray.dir = d;

    // material + incident factor + fudge factor
    colorMask *= materialColor; 
    colorMask *= dot(d, normalAtHitPoint);
    colorMask *= fudgeFactor;
}

__device__ static void specularSurface(
    Ray& ray,
    float3& colorMask,
    const float3& hitPoint,
    const float3& normalAtHitPoint,
    const float3& materialColor,
    curandState_t* randstates,
    int idx
) {
    // defined parameters
    float diffusePercent = 0.4f;
    float glossyFactor = 16.0f;
    float fudgeFactor = 2.5f;

    // mixture of diffuse and specular, decided by probability
    float r3 = curand_uniform(&randstates[idx]);

    // diffuse
    if (r3 < diffusePercent) {
        diffuseSurface(
            ray,
            colorMask,
            hitPoint,
            normalAtHitPoint,
            materialColor,
            randstates,
            idx
        );
        return;
    }

    // fuzz specular

    float3 d;
    float3 refl = reflect(ray.dir, normalAtHitPoint); 
    do {
        // random sample
        float r1 = curand_uniform(&randstates[idx]);
        float r2 = curand_uniform(&randstates[idx]);
        float3 rdSamp = uniformHemisphereSample(r1, r2);

        // uvw coords based on reflection
        float3 w = refl; 
        float3 u = normalize(cross((fabs(w.x) > 0.1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
        float3 v = cross(w, u);

        // out direction based on reflection + random
        d = normalize(u * rdSamp.x + v * rdSamp.y + w * rdSamp.z);
    } 
    while (dot(d, normalAtHitPoint) <= 0); // please dont sneak through the wall
    
    ray.orig = hitPoint + normalAtHitPoint * M_EPSILON;
    ray.dir = d;

    // glossy level incident factor + fudge factor
    colorMask *= powf(dot(d, refl), glossyFactor);
    colorMask *= fudgeFactor;
}

__device__ static void mirrorSurface(
    Ray& ray,
    float3& colorMask,
    const float3& hitPoint,
    const float3& normalAtHitPoint,
    const float3& materialColor
) {
    ray.orig = hitPoint + normalAtHitPoint * M_EPSILON;     
    ray.dir = reflect(ray.dir, normalAtHitPoint);

    colorMask *= materialColor;
}

__device__ static void transparentSurface(
    Ray& ray,
    float3& colorMask,
    const float3& hitPoint,
    const float3& normalAtHitPoint,
    const float3& materialColor,
    curandState_t* randstates,
    int idx
) {
    
}