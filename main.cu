#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include "io_utils.h"
#include "macro.h"

#include "geometryIntersect.cuh"
#include "surface.cuh"

#define NUM_SPHERES 4

struct Attr{
    float* spheres;
    int* materials;
};

__global__ void renderKernal (float3 *output, Attr attr, curandState_t* randstates) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = (HEIGHT - y - 1) * WIDTH + x; 

    float* spheres = attr.spheres;
    int* materials = attr.materials;

    float3 camOrig = make_float3(0.0f, 0.0f, 0.0f);
    float3 camDir = normalize(make_float3(0.0f, 0.0f, 1.0f));

    float fov = 0.5135f;
    float3 deltaX = make_float3(WIDTH * fov / HEIGHT, 0.0f, 0.0f);
    float3 deltaY = make_float3(0.0f, fov, 0.0f);

    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < SAMPLES; s++) {
        float3 rayDirection = normalize(camDir + deltaX * (x * 2.0f / WIDTH - 1.0f) + deltaY * (y * 2.0f / HEIGHT - 1.0f));

        float3 currentRayOrig = camOrig;
        float3 currentRayDir = rayDirection; 

        float3 accumulativeColor = make_float3(0.0f, 0.0f, 0.0f);
        float3 colorMask = make_float3(1.0f, 1.0f, 1.0f);

        for (int bounces = 0; bounces < RAY_BOUNCE; bounces++) {

            for (int i = 0; i < NUM_SPHERES; ++i) {
                float sphereRadius = spheres[i * 4];
                float3 sphereCenter = make_float3(spheres[i * 4 + 1], spheres[i * 4 + 2], spheres[i * 4 + 3]);
                int material = materials[i];
    
                float distanceCameraToObject = intersectSphereRay(sphereRadius, sphereCenter, currentRayOrig, currentRayDir);
    
                if (distanceCameraToObject == 0)
                    continue;

                // emission
                accumulativeColor += colorMask * make_float3(1.0f, 1.0f, 1.0f);

                //
                float3 hitPoint = currentRayOrig + currentRayDir * distanceCameraToObject;
                float3 normalAtHitPoint = getSphereNormal(hitPoint, sphereCenter, currentRayDir);
                float3 materialColor = make_float3(0.9f, 0.3f, 0.2f);
        
                // diffuse
                if (material == 1) {
                    diffuseSurface(
                        currentRayOrig,
                        currentRayDir,
                        accumulativeColor,
                        hitPoint,
                        normalAtHitPoint,
                        materialColor,
                        randstates,
                        idx
                    );
                }

                // final color
                finalColor += accumulativeColor / SAMPLES;               
            }
        }
    }

    output[idx] = make_float3(clamp(finalColor.x, 0.0f, 1.0f), clamp(finalColor.y, 0.0f, 1.0f), clamp(finalColor.z, 0.0f, 1.0f));
}

__global__ void initRandStates(unsigned int seed, curandState_t* randstates) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = (HEIGHT - y - 1) * WIDTH + x;

    curand_init(seed, x, y, &randstates[idx]);
}

int main(){
    // define dim
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);   
    dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1);

    // rand states
    curandState_t* randstates;
    cudaMalloc((void**) &randstates, NUM_BLOCKS * sizeof(curandState_t));

    // run
    initRandStates<<<grid, block>>>(time(NULL), randstates);

    // output
    float3* output_h = new float3[WIDTH * HEIGHT];
    float3* output_d;
    cudaMalloc(&output_d, WIDTH * HEIGHT * sizeof(float3));

    // scene
    float spheres[4 * NUM_SPHERES] {
        10.0f, 0.0f, 0.0f, 100.0f,
        10.0f, 30.0f, 0.0f, 100.0f,
        10.0f, 30.0f, 30.0f, 100.0f,
        10.0f, 0.0f, 30.0f, 100.0f};
    int materials[NUM_SPHERES] {0, 1, 1, 1};

    float* spheres_h = new float[4 * NUM_SPHERES];
    int* materials_h = new int[NUM_SPHERES];
    
    for (int i = 0; i < 4 * NUM_SPHERES; ++i) spheres_h[i] = spheres[i];
    for (int i = 0; i < NUM_SPHERES; ++i) materials_h[i] = materials[i];
        
    float* spheres_d;
    int* materials_d;

    cudaMalloc(&spheres_d, 4 * NUM_SPHERES * sizeof(float));
    cudaMalloc(&materials_d, NUM_SPHERES * sizeof(int));

    // copy host to device
    cudaMemcpy(spheres_d, spheres_h, 4 * NUM_SPHERES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(materials_d, materials_h, NUM_SPHERES * sizeof(int), cudaMemcpyHostToDevice);

    Attr attr{spheres_d, materials_d};
    
    // run
    renderKernal <<< grid, block >>> (output_d, attr, randstates);

    // copy device to host
    cudaMemcpy(output_h, output_d, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

    // output
    writeToPPM("result.ppm", WIDTH, HEIGHT, output_h);

    // clean
    cudaFree(spheres_d);  
    cudaFree(output_d);  
    cudaFree(randstates);
    delete[] spheres_h;
    delete[] output_h;
}