#pragma once

#include "cutil_math.h"

enum SurfaceTypeEnum{
    DIFFUSE,
    SPECULAR,
    MIRROR,
    TRANSPARENT,
};

struct Material{
    SurfaceTypeEnum surfaceType;
    float3 colorEmission;
    float3 surfaceColor;
};