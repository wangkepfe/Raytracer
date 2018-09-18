#pragma once

#include "cutil_math.h"

// miscellaneous
__constant__ static float sunSize = 0.97f;
__constant__ static float3 sunDir{0.0f, 1.0f, 0.0f};
__constant__ static float3 sunColor{1.0f, 0.875f, 0.75f};
__constant__ static float3 skyColor{0.5f, 0.8f, 0.9f};
__constant__ static float3 mistColor{0.02f, 0.02f, 0.02f};

// camera
__constant__ static float3 camOrig{0.0f, 0.0f, 0.0f};
__constant__ static float3 camDir{0.0f, 0.0f, 1.0f};
__constant__ static float camFov = 0.5135f;


__constant__ static float pRussianRoulette = 0.7f;