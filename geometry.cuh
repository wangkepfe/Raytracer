#pragma once

#include "cutil_math.h"

enum{
    IMPLICIT_SPHERE,
    IMPLICIT_AABB,
    TRIANGLE_MESH,
};

struct Geometry{
    int geometryType;
    int geometryIdx;
    int materialIdx;
};