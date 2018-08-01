#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include "cutil_math.h"
#include "implicitGeometry.cuh"

// generate a cosine weighted hemisphere sampler : generate uniform points on a disk, and then project them up to the hemisphere
__device__ static float3 cosineSampleHemisphere(float u1, float u2){
    float r = sqrtf(u1);
    float theta = 2 * M_PI * u2;
 
    float x = r * cos(theta);
    float y = r * sin(theta);
 
    return make_float3(x, y, sqrtf(max(0.0f, 1 - u1)));
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
    float r1 = curand_uniform(&randstates[idx]);
    float r2 = curand_uniform(&randstates[idx]);
    float3 rdSamp = cosineSampleHemisphere(r1, r2);

    float3 w = normalAtHitPoint; 
    float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
    float3 v = cross(w, u);

    float3 d = normalize(u * rdSamp.x + v * rdSamp.y + w * rdSamp.z);

    ray.orig = hitPoint + normalAtHitPoint * 0.05f;     // offset ray origin slightly to prevent self intersection
    ray.dir = d;

    colorMask *= materialColor;       // multiply with colour of object       
    colorMask *= dot(d, normalAtHitPoint);      // weigh light contribution using cosine of angle between incident light and normal
    colorMask *= 2;              // fudge factor
}

__device__ static void specularSurface(
    Ray& ray,
    float3& colorMask,
    const float3& hitPoint,
    const float3& normalAtHitPoint,
    const float3& materialColor
) {
    ray.orig = hitPoint + normalAtHitPoint * 0.05f;     
    ray.dir = reflect(ray.dir, normalAtHitPoint);

    colorMask *= materialColor;
}

__device__ static void refractionSurface(
    Ray& ray,
    float3& colorMask,
    bool isIntoSurface,
    const float3& hitPoint,
    const float3& normalAtHitPoint,
    const float3& materialColor,
    curandState_t* randstates,
    int idx
) {
    float3 x, d;
    float3 n = isIntoSurface ? normalAtHitPoint : normalAtHitPoint * -1;

    float nc = 1.0f;  // Index of Refraction air
    float nt = 1.5f;  // Index of Refraction glass/water
    float nnt = isIntoSurface ? nc / nt : nt / nc;  // IOR ratio of refractive materials

    float ddn = dot(ray.dir, normalAtHitPoint);
    float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

    if (cos2t < 0.0f) // total internal reflection 
    {
        d = reflect(ray.dir, n); //d = r.dir - 2.0f * n * dot(n, r.dir);
        x += normalAtHitPoint * 0.01f;
    }
    else // cos2t > 0
    {
        // compute direction of transmission ray
        float3 tdir = normalize(ray.dir * nnt - n * ((isIntoSurface ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

        float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
        float c = 1.0f - (isIntoSurface ? -ddn : dot(tdir, n));
        float Re = R0 + (1.f - R0) * c * c * c * c * c;
        float Tr = 1 - Re; // Transmission
        float P = 0.25f + 0.5f * Re;
        float RP = Re / P;
        float TP = Tr / (1.0f - P);

        // randomly choose reflection or transmission ray
        if (curand_uniform(&randstates[idx]) < 0.25f) // reflection ray
        {
            colorMask *= RP;
            d = reflect(ray.dir, n);
            x += normalAtHitPoint * 0.02f;
        }
        else // transmission ray
        {
            colorMask *= TP;
            d = tdir; //r = Ray(x, tdir); 
            x += normalAtHitPoint * 0.0005f; // epsilon must be small to avoid artefacts
        }
    }
    ray.orig = x;
	ray.dir = d;
}