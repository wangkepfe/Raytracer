#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include "cutil_math.h"

enum ReflectionType { DIFF, SPEC, REFR };

struct Material{
    float3 emi;
    float3 col;
    ReflectionType refl;
};