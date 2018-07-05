#pragma once

#include "ray.cuh"
#include "material.cuh"

inline __device__ float3 minf3(float3 a, float3 b){ return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ float3 maxf3(float3 a, float3 b){ return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float minf1(float a, float b){ return a < b ? a : b; }
inline __device__ float maxf1(float a, float b){ return a > b ? a : b; }

struct AABB{
    float3 min;
    float3 max;
    Material mat;

    __device__ float intersect(const Ray& r){
        float epsilon = 0.001f; // required to prevent self intersection

		float3 tmin = (min - r.orig) / r.dir;
		float3 tmax = (max - r.orig) / r.dir;

		float3 real_min = minf3(tmin, tmax);
		float3 real_max = maxf3(tmin, tmax);

		float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
		float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);

		if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
		else return 0;
    }

    __device__ float3 getNormalAt(const float3 &point, const Ray& r){
        float3 normal = make_float3(0.f, 0.f, 0.f);
		float epsilon = 0.001f;

		if (fabs(min.x - point.x) < epsilon) normal = make_float3(-1, 0, 0);
		else if (fabs(max.x - point.x) < epsilon) normal = make_float3(1, 0, 0);
		else if (fabs(min.y - point.y) < epsilon) normal = make_float3(0, -1, 0);
		else if (fabs(max.y - point.y) < epsilon) normal = make_float3(0, 1, 0);
		else if (fabs(min.z - point.z) < epsilon) normal = make_float3(0, 0, -1);
        else normal = make_float3(0, 0, 1);
        
        float3 nl = dot(normal, r.dir) < 0 ? normal : normal * -1; // front facing normal

		return nl;
    }
};