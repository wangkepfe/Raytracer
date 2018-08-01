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
    AABB* aabbs;
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

    for (int s = 0; s < SAMPLES; s++) {//sample
        float3 rayDirection = normalize(camDir + deltaX * (x * 2.0f / WIDTH - 1.0f) + deltaY * (y * 2.0f / HEIGHT - 1.0f));

        Ray currentRay {camOrig, rayDirection};

        float3 accumulativeColor = make_float3(0.0f, 0.0f, 0.0f);
        float3 colorMask = make_float3(1.0f, 1.0f, 1.0f);

        for (int bounces = 0; bounces < RAY_BOUNCE; bounces++) {//bounce
            float3 hitPoint;
            float3 normalAtHitPoint;

            float nearestIntersectionDistance = 10000.0f;
            bool hitEmptyVoidSpace = true;
            int hitObjectMaterialIdx = 0;

            for (int objectIdx = 0; objectIdx < attr.numberOfObject; ++objectIdx) {// scene intersection
                Geometry geometry = attr.geometries[objectIdx];

                if (geometry.geometryType == IMPLICIT_SPHERE) {
                    Sphere sphere = attr.spheres[geometry.geometryIdx];
                    float distanceCameraToObject = intersectSphereRay(sphere, currentRay);
    
                    if (distanceCameraToObject > 0.001f && distanceCameraToObject < nearestIntersectionDistance) {
                        hitEmptyVoidSpace = false;
                        nearestIntersectionDistance = distanceCameraToObject;

                        hitPoint = currentRay.orig + currentRay.dir * distanceCameraToObject;
                        normalAtHitPoint = getSphereNormal(hitPoint, sphere.orig, currentRay.dir);
                        hitObjectMaterialIdx = geometry.materialIdx;
                    }
                } 
                else if (geometry.geometryType == IMPLICIT_AABB) {
                    AABB aabb = attr.aabbs[geometry.geometryIdx];
                    float distanceCameraToObject = intersectAABBRay(aabb, currentRay);
    
                    if (distanceCameraToObject > 0.001f && distanceCameraToObject < nearestIntersectionDistance) {
                        hitEmptyVoidSpace = false;
                        nearestIntersectionDistance = distanceCameraToObject;

                        hitPoint = currentRay.orig + currentRay.dir * distanceCameraToObject;
                        normalAtHitPoint = getAABBNormal(hitPoint, aabb, currentRay.dir);
                        hitObjectMaterialIdx = geometry.materialIdx;
                    }                    
                }
            }// end of scene intersection

            if (hitEmptyVoidSpace){
                break;// break out of bounce
            }

            // surface
            Material material = attr.materials[hitObjectMaterialIdx];

            accumulativeColor += colorMask * material.colorEmission;

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
            else if (material.surfaceType == SPECULAR) {
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
            else if (material.surfaceType == REFRACTION) {
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
        }//end of bounce
        finalColor += accumulativeColor / SAMPLES;
    }//end of sample

    output[idx] = make_float3(clamp(finalColor.x, 0.0f, 1.0f), clamp(finalColor.y, 0.0f, 1.0f), clamp(finalColor.z, 0.0f, 1.0f));
}

__global__ void initRandStates(unsigned int seed, curandState_t* randstates) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = (HEIGHT - y - 1) * WIDTH + x;

    curand_init(seed, idx, 0, &randstates[idx]);
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
        {float3{0.0f, 40.0f, 120.0f} ,10.0f},
        {float3{30.0f, -30.0f, 100.0f}, 10.0f},
    };

    AABB aabbs[] {
        {float3{-5000.0f, -50.0f, -10.0f},float3{5000.0f, -40.0f, 5000.0f}},
        {float3{-30.0f, -40.0f, 130.0f},float3{-10.0f, -20.0f, 120.0f}}
    };

    Material materials[] {
        {EMISSION_ONLY, float3{2.0f, 1.75f, 1.5f}, float3{1.0f, 1.0f, 1.0f}},
        {DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{1.0f, 1.0f, 1.0f}},
        {DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{0.9f, 0.1f, 0.1f}},
        {DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{0.1f, 0.1f, 0.9f}}
    };

    Geometry geometries[] {
        {IMPLICIT_AABB, 0, 1},
        {IMPLICIT_AABB, 1, 2},
        {IMPLICIT_SPHERE, 0, 0},
        {IMPLICIT_SPHERE, 1, 3}
    };

    Sphere* spheres_d;
    AABB* aabbs_d;
    Material* materials_d;
    Geometry* geometries_d;

    cudaMalloc(&spheres_d, sizeof(spheres));
    cudaMalloc(&aabbs_d, sizeof(aabbs));
    cudaMalloc(&materials_d, sizeof(materials));
    cudaMalloc(&geometries_d, sizeof(geometries));

    cudaMemcpy(spheres_d, spheres, sizeof(spheres), cudaMemcpyHostToDevice);
    cudaMemcpy(aabbs_d, aabbs, sizeof(aabbs), cudaMemcpyHostToDevice);
    cudaMemcpy(materials_d, materials, sizeof(materials), cudaMemcpyHostToDevice);
    cudaMemcpy(geometries_d, geometries, sizeof(geometries), cudaMemcpyHostToDevice);

    Attr attr {
        sizeof(geometries) / sizeof(Geometry), 
        spheres_d, 
        aabbs_d,
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