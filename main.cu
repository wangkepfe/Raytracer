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
#include "implicitGeometry.cuh"

enum{
    EMISSION_ONLY,
    DIFFUSE,
    SPECULAR,
    REFRACTION,
};

enum{
    IMPLICIT_SPHERE,
    IMPLICIT_AABB,
};

struct Material{
    int surfaceType;
    float3 colorEmission;
    float3 surfaceColor;
};

struct Geometry{
    int geometryType;
    int geometryIdx;
    int materialIdx;
};

struct Attr{
    int numberOfObject;
    Sphere* spheres;
    Material* materials;
    Geometry* geometries;
};

__global__ void renderKernal (float3 *output, Attr attr, curandState_t* randstates) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = (HEIGHT - y - 1) * WIDTH + x; 

    float3 camOrig = make_float3(0.0f, 0.0f, 0.0f);
    float3 camDir = normalize(make_float3(0.0f, 0.0f, 1.0f));

    float fov = 0.5135f;
    float3 deltaX = make_float3(WIDTH * fov / HEIGHT, 0.0f, 0.0f);
    float3 deltaY = make_float3(0.0f, fov, 0.0f);

    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < SAMPLES; s++) {
    //for (int s = 0; s < 1; s++) {
        float3 rayDirection = normalize(camDir + deltaX * (x * 2.0f / WIDTH - 1.0f) + deltaY * (y * 2.0f / HEIGHT - 1.0f));

        Ray currentRay {camOrig, rayDirection};

        float3 accumulativeColor = make_float3(0.0f, 0.0f, 0.0f);
        float3 colorMask = make_float3(1.0f, 1.0f, 1.0f);

        for (int bounces = 0; bounces < RAY_BOUNCE; bounces++) {
        //for (int bounces = 0; bounces < 1; bounces++) {

            for (int i = 0; i < attr.numberOfObject; ++i) {
            //for (int i = 0; i < 1; ++i) {
                Geometry geometry = attr.geometries[i];

                if (geometry.geometryType == IMPLICIT_SPHERE) {
                    Sphere sphere = attr.spheres[geometry.geometryIdx];
                    Material material = attr.materials[geometry.materialIdx];

                    float distanceCameraToObject = intersectSphereRay(sphere, currentRay);
    
                    if (distanceCameraToObject < 0.04f)
                        continue;
                    
                    // emission
                    accumulativeColor += colorMask * material.colorEmission;

                    float3 hitPoint = currentRay.orig + currentRay.dir * distanceCameraToObject;
                    float3 normalAtHitPoint = getSphereNormal(hitPoint, sphere.orig, currentRay.dir);
            
                    // diffuse
                    if (material.surfaceType == DIFFUSE) {
                        diffuseSurface(
                            currentRay,
                            colorMask,
                            hitPoint,
                            normalAtHitPoint,
                            material.surfaceColor,
                            randstates,
                            idx
                        );
                    }

                    finalColor += accumulativeColor / SAMPLES;    
                    //finalColor += accumulativeColor;         
                }        
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
    Sphere spheres[] {
        {float3{0.0f, 0.0f, 100.0f} ,10.0f},
        {float3{30.0f, 0.0f, 100.0f}, 10.0f},
        {float3{30.0f, 30.0f, 100.0f} ,10.0f},
        {float3{0.0f, 30.0f, 100.0f}, 10.0f}
    };

    Material materials[] {
        {EMISSION_ONLY, float3{1.0f, 1.0f, 1.0f}, float3{1.0f, 1.0f, 1.0f}},
        {DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{1.0f, 0.2f, 0.1f}}
    };

    Geometry geometries[] {
        {IMPLICIT_SPHERE, 0, 0},
        {IMPLICIT_SPHERE, 1, 1},
        {IMPLICIT_SPHERE, 2, 1},
        {IMPLICIT_SPHERE, 3, 1}
    };

    Sphere* spheres_d;
    Material* materials_d;
    Geometry* geometries_d;

    cudaMalloc(&spheres_d, sizeof(spheres));
    cudaMalloc(&materials_d, sizeof(materials));
    cudaMalloc(&geometries_d, sizeof(geometries));

    cudaMemcpy(spheres_d, spheres, sizeof(spheres), cudaMemcpyHostToDevice);
    cudaMemcpy(materials_d, materials, sizeof(materials), cudaMemcpyHostToDevice);
    cudaMemcpy(geometries_d, geometries, sizeof(geometries), cudaMemcpyHostToDevice);

    Attr attr {
        sizeof(geometries) / sizeof(Geometry), 
        spheres_d, 
        materials_d, 
        geometries_d
    };
    
    // run
    renderKernal <<< grid, block >>> (output_d, attr, randstates);

    // copy device to host
    cudaMemcpy(output_h, output_d, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

    // output
    writeToPPM("result.ppm", WIDTH, HEIGHT, output_h);

    // clean
    cudaFree(spheres_d);  
    cudaFree(materials_d);
    cudaFree(geometries_d);

    cudaFree(output_d);  
    cudaFree(randstates);

    delete[] output_h;
}