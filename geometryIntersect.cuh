#pragma once

#include "cutil_math.h"

inline __device__ float intersectSphereRay(
    float sphereRadius,
    const float3& sphereCenter,
    const float3& rayOrig,
    const float3& rayDir)
{
    float3 op = sphereCenter - rayOrig;
    float t, epsilon = 0.0001f;
    float b = dot(op, rayDir);
    float disc = b*b - dot(op, op) + sphereRadius*sphereRadius;
    if (disc < 0) return 0;
    else disc = sqrtf(disc);
    return ((t = b - disc) > epsilon) ? t : (((t = b + disc) > epsilon) ? t : 0); 
}

inline __device__ float3 getSphereNormal(
    const float3& pointOnSphere,
    const float3& sphereCenter,
    float3 rayDir)
{
    float3 n = normalize(pointOnSphere - sphereCenter);    // normal
    float3 nl = dot(n, rayDir) < 0 ? n : n * -1; // front facing normal
    return nl;
}