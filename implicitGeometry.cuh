#pragma once

#include "cutil_math.h"

struct Ray{
    float3 orig;
    float3 dir;
};

struct Sphere{
    float3 orig;
    float rad;
};

struct AABB{
    float3 min;
    float3 max;
};