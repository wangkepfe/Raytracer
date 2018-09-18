#pragma once

// quality control
#define KERNAL_LOOP 6
#define SAMPLES 64

#define MAX_RAY_BOUNCE 50

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

// cuda dimension
#define PATCH_WIDTH 128
#define PATCH_HEIGHT 128
#define PATCH_NUM_X (WIDTH / PATCH_WIDTH)
#define PATCH_NUM_Y (HEIGHT / PATCH_HEIGHT)

#define BLOCK_SIZE 8
#define NUM_BLOCKS (PATCH_WIDTH*PATCH_HEIGHT/BLOCK_SIZE/BLOCK_SIZE)

// M_MATH
#define M_PI 3.14159265359f
#define M_INF 1e20f
#define M_EPSILON 0.0001f

// functions
#define CUDA_MALLOC_MEMCPY_HOST_TO_DEVICE(__TYPE__,__DEVICE_ARRAY__,__HOST_ARRAY__,__SIZE__) \
__TYPE__* __DEVICE_ARRAY__; \
cudaMalloc(&__DEVICE_ARRAY__, __SIZE__); \
cudaMemcpy(__DEVICE_ARRAY__, __HOST_ARRAY__, __SIZE__, cudaMemcpyHostToDevice);