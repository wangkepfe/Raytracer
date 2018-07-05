#pragma once

#include "ray.cuh"
#include "material.cuh"

struct Sphere{
    float rad; 
    float3 pos;
    Material mat;

    __device__ float intersect(const Ray& r){
        float3 op = pos - r.orig;
        float t, epsilon = 0.0001f;
        float b = dot(op, r.dir);
        float disc = b*b - dot(op, op) + rad*rad;
        if (disc < 0) return 0;
        else disc = sqrtf(disc);
        return ((t = b - disc) > epsilon) ? t : (((t = b + disc) > epsilon) ? t : 0); 
    }

    __device__ float3 getNormalAt(const float3 &point, const Ray& r){
        float3 n = normalize(point - pos);    // normal
        float3 nl = dot(n, r.dir) < 0 ? n : n * -1; // front facing normal
        return nl;
    }
};