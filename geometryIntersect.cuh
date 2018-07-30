#pragma once

#include "cutil_math.h"
#include "implicitGeometry.cuh"

inline __device__ float intersectSphereRay(
    const Sphere& sphere,
    const Ray& ray)
{
    float3 op = sphere.orig - ray.orig;
    float t, epsilon = 0.0001f;
    float b = dot(op, ray.dir);
    float disc = b * b - dot(op, op) + sphere.rad * sphere.rad;
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