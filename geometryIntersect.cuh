#pragma once

#include "cutil_math.h"
#include "implicitGeometry.cuh"

inline __device__ float3 minf3(float3 a, float3 b){ return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ float3 maxf3(float3 a, float3 b){ return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float minf1(float a, float b){ return a < b ? a : b; }
inline __device__ float maxf1(float a, float b){ return a > b ? a : b; }

// sphere
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
    float3 rayDir,
    bool& isIntoSurface)
{
    float3 n = normalize(pointOnSphere - sphereCenter);    // normal
    isIntoSurface = dot(n, rayDir) < 0;
    float3 nl = isIntoSurface ? n : n * -1; // front facing normal
    return nl;
}

/* 
// This version is for understanding the algorithm

inline __device__ float intersectAABBRay(
    const AABB& aabb,
    const Ray& ray)
{
    enum {LEFT, RIGHT, MIDDLE};
    int3 quadrant;
    float3 maxT, candidatePlane;
    bool inside = true;
    float EPSILON = 0.001f;
    float3 hitPoint;

    // find candidate plane 
    if      (ray.orig.x < aabb.min.x) { quadrant.x = LEFT;  candidatePlane.x = aabb.min.x; inside = false; }
    else if (ray.orig.x > aabb.max.x) { quadrant.x = RIGHT; candidatePlane.x = aabb.max.x; inside = false; }
    else                              { quadrant.x = MIDDLE;                                               }

    if      (ray.orig.y < aabb.min.y) { quadrant.y = LEFT;  candidatePlane.y = aabb.min.y; inside = false; }
    else if (ray.orig.y > aabb.max.y) { quadrant.y = RIGHT; candidatePlane.y = aabb.max.y; inside = false; }
    else                              { quadrant.y = MIDDLE;                                               }

    if      (ray.orig.z < aabb.min.z) { quadrant.z = LEFT;  candidatePlane.z = aabb.min.z; inside = false; }
    else if (ray.orig.z > aabb.max.z) { quadrant.z = RIGHT; candidatePlane.z = aabb.max.z; inside = false; }
    else                              { quadrant.z = MIDDLE;                                               }

    // ray origin inside the box
    if (inside) {
        return 0;
    }

    // Calculate T distances to candidate planes
    if (quadrant.x != MIDDLE && fabs(ray.dir.x) > EPSILON) { maxT.x = (candidatePlane.x - ray.orig.x) / ray.dir.x; } else { maxT.x = -1.0f; }
    if (quadrant.y != MIDDLE && fabs(ray.dir.y) > EPSILON) { maxT.y = (candidatePlane.y - ray.orig.y) / ray.dir.y; } else { maxT.y = -1.0f; }
    if (quadrant.z != MIDDLE && fabs(ray.dir.z) > EPSILON) { maxT.z = (candidatePlane.z - ray.orig.z) / ray.dir.z; } else { maxT.z = -1.0f; }

    // Get largest of the maxT's for final choice of intersection
    int whichPlane = 0;
    float maxt = maxT.x;
    if (maxT.y > maxt) { whichPlane = 1; maxt = maxT.y; }
    if ((whichPlane == 0 && maxT.z > maxt) || (whichPlane == 1 && maxT.z > maxt)) { whichPlane = 2; maxt = maxT.z; }

    // Check final candidate actually inside box
    if (maxt < 0.0f) return 0;

    if (whichPlane == 0) {
        hitPoint.x = candidatePlane.x;
        hitPoint.y = ray.orig.y + maxt * ray.dir.y;
        hitPoint.z = ray.orig.z + maxt * ray.dir.z;
    } else if (whichPlane == 1) {
        hitPoint.x = ray.orig.x + maxt * ray.dir.x;
        hitPoint.y = candidatePlane.y;
        hitPoint.z = ray.orig.z + maxt * ray.dir.z;
    } else if (whichPlane == 2) {
        hitPoint.x = ray.orig.x + maxt * ray.dir.x;
        hitPoint.y = ray.orig.y + maxt * ray.dir.y;
        hitPoint.z = candidatePlane.z;
    }

    if (hitPoint.x < aabb.min.x || hitPoint.x > aabb.max.x || 
        hitPoint.y < aabb.min.y || hitPoint.y > aabb.max.y || 
        hitPoint.z < aabb.min.z || hitPoint.z > aabb.max.z)
        return 0;

    return maxt;
}
*/

// AABB
inline __device__ float intersectAABBRay(
    const AABB& aabb,
    const Ray& ray)
{
    float epsilon = 0.001f; // required to prevent self intersection

    float3 tmin = (aabb.min - ray.orig) / ray.dir;
    float3 tmax = (aabb.max - ray.orig) / ray.dir;

    float3 real_min = minf3(tmin, tmax);
    float3 real_max = maxf3(tmin, tmax);

    float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
    float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);

    if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
    else return 0;
}

inline __device__ float3 getAABBNormal(
    const float3& pointOnAABB,
    const AABB& aabb,
    float3 rayDir)
{
    float3 normal;
    float epsilon = 0.001f;

    if      (fabs(aabb.min.x - pointOnAABB.x) < epsilon) normal = make_float3(-1, 0, 0);
    else if (fabs(aabb.max.x - pointOnAABB.x) < epsilon) normal = make_float3(1, 0, 0);
    else if (fabs(aabb.min.y - pointOnAABB.y) < epsilon) normal = make_float3(0, -1, 0);
    else if (fabs(aabb.max.y - pointOnAABB.y) < epsilon) normal = make_float3(0, 1, 0);
    else if (fabs(aabb.min.z - pointOnAABB.z) < epsilon) normal = make_float3(0, 0, -1);
    else normal = make_float3(0, 0, 1);
    
    float3 nl = dot(normal, rayDir) < 0 ? normal : (-normal); // front facing normal
    return nl;
}