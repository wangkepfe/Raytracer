#pragma once

#include "cutil_math.h"

enum{
    SPHERE,
    AABB,
    MESH,
};

struct Geometry{
    int geometryType;
    int geometryIdx;
    int materialIdx;
};