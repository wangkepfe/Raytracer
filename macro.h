#pragma once

#define KERNAL_LOOP 4
#define SAMPLES 512
#define RAY_BOUNCE 6

#define __SETTING__RESOLUTION 2

#if __SETTING__RESOLUTION == 1
    #define WIDTH 1024 
    #define HEIGHT 768
#elif __SETTING__RESOLUTION == 2
    #define WIDTH 512 
    #define HEIGHT 384
#else
    #define WIDTH 512 
    #define HEIGHT 384
#endif

#define PATCH_WIDTH 256
#define PATCH_HEIGHT 192
#define PATCH_NUM_X (WIDTH / PATCH_WIDTH)
#define PATCH_NUM_Y (HEIGHT / PATCH_HEIGHT)

#define BLOCK_SIZE 8
#define NUM_BLOCKS (PATCH_WIDTH*PATCH_HEIGHT/BLOCK_SIZE/BLOCK_SIZE)

#define M_PI 3.14159265359f
#define M_INF 1e20f
#define M_EPSILON 0.0001f