#pragma once

#include "cutil_math.h"

struct Ray{
    float3 orig;
    float3 dir;

    __device__ Ray(const float3& orig, const float3& dir) : orig{orig}, dir{dir} {}
};