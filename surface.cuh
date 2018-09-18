#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include "cutil_math.h"
#include "implicitGeometry.cuh"
#include "geometry.cuh"
#include "sceneAttributes.cuh"

__device__ inline static float3 uniformHemisphereSample(float r1, float r2) {
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

__device__ inline static float3 uniformHemisphereSampleFrom (
    float3 inDir,
    curandState_t* randstates,
    int idx 
) {
    // random sample
    float r1 = curand_uniform(&randstates[idx]);
    float r2 = curand_uniform(&randstates[idx]);
    float3 rdSamp = uniformHemisphereSample(r1, r2);

    // uvw coords
    float3 w = inDir; 
    float3 u = normalize(cross((fabs(w.x) > 0.1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
    float3 v = cross(w, u);

    // random out direction
    return normalize(u * rdSamp.x + v * rdSamp.y + w * rdSamp.z);
}

__device__ inline static float3 importanceSampling (
    const Attr& attr,
    const float3& hitPoint, 
    curandState_t* randstates,
    int idx 
) {
    // randomly choose one of the light source
    float r0 = curand_uniform(&randstates[idx]);
    r0 = clamp(r0, M_EPSILON, 1.0f - M_EPSILON);
    uint lightSecondIndex = attr.numberOfLights * r0;
    Geometry geometry = attr.geometries[attr.lightIndices[lightSecondIndex]];

    // only handle sphere light for now!
    // to do: make it generic for all kinds of lights
    if (geometry.geometryType == SPHERE) {
        Sphere lightSphere = attr.spheres[geometry.geometryIdx];

        float3 sphereCenter = lightSphere.orig;
        float sphereRadius = lightSphere.rad;

        // random point on light bulb
        float3 pointOnLight = sphereCenter 
            + sphereRadius 
            * uniformHemisphereSampleFrom(hitPoint - lightSphere.orig, randstates, idx);

        // random direction from hit point to light (important direction for the hit point)
        float3 importantDir = normalize(pointOnLight - hitPoint);

        return importantDir;
    }

    printf("not sphere light\n");
    return make_float3(1.0f, 0.0f, 0.0f);
}

__device__ static void diffuseSurface(
    Ray& ray,
    float3& colorMask,

    const float3& hitPoint,
    const float3& normalAtHitPoint,
    const float3& materialColor,

    const Attr& attr,

    curandState_t* randstates,
    int idx
) {
    // defined parameters
    float fudgeFactor = 1.6f;

    float3 d;

    // importance sampling
    float r0 = curand_uniform(&randstates[idx]);

    // the probabily of distribution is a pure guess:
    // - higher: good for surface exposed to the light
    // - lower: the opposite
    if (r0 < 0.1f) {
        // go to light directly
        d = importanceSampling(attr, hitPoint, randstates, idx);
    } else {
        // random out direction: from normal create a hemisphere sample
        d = uniformHemisphereSampleFrom(normalAtHitPoint, randstates, idx);
    }

    ray.orig = hitPoint + normalAtHitPoint * M_EPSILON;
    ray.dir = d;

    // material + incident factor + fudge factor
    colorMask *= materialColor; 
    colorMask *= dot(d, normalAtHitPoint);
    colorMask *= fudgeFactor;
}

// __device__ static void specularSurface(
//     Ray& ray,
//     float3& colorMask,
//     const float3& hitPoint,
//     const float3& normalAtHitPoint,
//     const float3& materialColor,
//     curandState_t* randstates,
//     int idx
// ) {
//     // defined parameters
//     float diffusePercent = 0.4f;
//     float glossyFactor = 16.0f;
//     float fudgeFactor = 2.5f;

//     // mixture of diffuse and specular, decided by probability
//     float r3 = curand_uniform(&randstates[idx]);

//     // diffuse
//     if (r3 < diffusePercent) {
//         diffuseSurface(
//             ray,
//             colorMask,
//             hitPoint,
//             normalAtHitPoint,
//             materialColor,
//             randstates,
//             idx
//         );
//         return;
//     }

//     // fuzz specular

//     float3 d;
//     float3 refl = reflect(ray.dir, normalAtHitPoint); 
//     do {
//         // random sample
//         float r1 = curand_uniform(&randstates[idx]);
//         float r2 = curand_uniform(&randstates[idx]);
//         float3 rdSamp = uniformHemisphereSample(r1, r2);

//         // uvw coords based on reflection
//         float3 w = refl; 
//         float3 u = normalize(cross((fabs(w.x) > 0.1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
//         float3 v = cross(w, u);

//         // out direction based on reflection + random
//         d = normalize(u * rdSamp.x + v * rdSamp.y + w * rdSamp.z);
//     } 
//     while (dot(d, normalAtHitPoint) <= 0); // please dont sneak through the wall
    
//     ray.orig = hitPoint + normalAtHitPoint * M_EPSILON;
//     ray.dir = d;

//     // glossy level incident factor + fudge factor
//     colorMask *= powf(dot(d, refl), glossyFactor);
//     colorMask *= fudgeFactor;
// }

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

    bool isIntoSurface,
    const float3& hitPoint,
    const float3& normalAtHitPoint,
    curandState_t* randstates,
    int idx
) {
    float3 d;
    float3 x = hitPoint;

    float3 n = isIntoSurface ? normalAtHitPoint : (normalAtHitPoint * -1.0f);
    float3 nl = normalAtHitPoint;

    float nc = 1.0f;  // Index of Refraction air
    float nt = 1.5f;  // Index of Refraction glass/water

    float nnt = isIntoSurface ? nc / nt : nt / nc;  // IOR ratio of refractive materials

    float ddn = dot(ray.dir, nl);
    float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

    if (cos2t < 0.0f) // total internal reflection 
    {
        d = reflect(ray.dir, n);
        x += nl * 0.01f;
    }
    else // cos2t > 0
    {
        // compute direction of transmission ray
        float3 tdir = normalize(ray.dir * nnt - n * ((isIntoSurface ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

        float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
        float c = 1.f - (isIntoSurface ? -ddn : dot(tdir, n));
        float Re = R0 + (1.f - R0) * c * c * c * c * c;
        float Tr = 1 - Re; // Transmission
        float P = .25f + .5f * Re;
        float RP = Re / P;
        float TP = Tr / (1.f - P);

        // randomly choose reflection or transmission ray
        if (curand_uniform(&randstates[idx]) < 0.25f) // reflection ray
        {
            colorMask *= RP;
            d = reflect(ray.dir, n);
            x += nl * 0.02f;
        }
        else // transmission ray
        {
            colorMask *= TP;
            d = tdir; //r = Ray(x, tdir); 
            x += nl * 0.0005f; // epsilon must be small to avoid artefacts
        }
    }

    ray.dir = d;
    ray.orig = x;
}