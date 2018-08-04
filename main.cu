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
    DIFFUSE,
    SPECULAR,
    MIRROR,
    TRANSPARENT,
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

__constant__ float sunSize = 0.97f;
__constant__ float3 sunDir{0.0f, 1.0f, 0.0f};
__constant__ float3 sunColor{1.0f, 0.875f, 0.75f};
__constant__ float3 skyColor{0.5f, 0.8f, 0.9f};
__constant__ float3 mistColor{0.02f, 0.02f, 0.02f};

__constant__ float3 camOrig{0.0f, 0.0f, 0.0f};
__constant__ float3 camDir{0.0f, 0.0f, 1.0f};
__constant__ float camFov = 0.5135f;

__global__ void renderKernal (
    float3 *output,
    unsigned int patch_width_offset,
    unsigned int patch_height_offset,
    Attr attr, 
    curandState_t* randstates)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = (PATCH_HEIGHT - y - 1) * PATCH_WIDTH + x;

    unsigned int realX = patch_width_offset + x;   
    unsigned int realY = patch_height_offset + y;

    float3 deltaX = make_float3(WIDTH * camFov / HEIGHT, 0.0f, 0.0f);
    float3 deltaY = make_float3(0.0f, camFov, 0.0f);

    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f);

    float3 rayDirection = normalize(camDir + deltaX * (realX * 2.0f / WIDTH - 1.0f) + deltaY * (realY * 2.0f / HEIGHT - 1.0f));

    for (int s = 0; s < SAMPLES; s++) {//sample
        Ray currentRay {camOrig, rayDirection};

        float3 accumulativeColor = make_float3(0.0f, 0.0f, 0.0f);
        float3 colorMask = make_float3(1.0f, 1.0f, 1.0f);

        for (int bounces = 0; bounces < RAY_BOUNCE; bounces++) {//bounce
            float3 hitPoint;
            float3 normalAtHitPoint;
            bool isIntoSurface = true;

            float nearestIntersectionDistance = M_INF;
            bool hitEmptyVoidSpace = true;
            int hitObjectMaterialIdx = 0;

            for (int objectIdx = 0; objectIdx < attr.numberOfObject; ++objectIdx) {// scene intersection
                Geometry geometry = attr.geometries[objectIdx];

                if (geometry.geometryType == IMPLICIT_SPHERE) {
                    Sphere sphere = attr.spheres[geometry.geometryIdx];
                    float distanceToObject = intersectSphereRay(sphere, currentRay);
    
                    if (distanceToObject > 0.001f && distanceToObject < nearestIntersectionDistance) {
                        hitEmptyVoidSpace = false;
                        nearestIntersectionDistance = distanceToObject;

                        hitPoint = currentRay.orig + currentRay.dir * distanceToObject;
                        normalAtHitPoint = getSphereNormal(hitPoint, sphere.orig, currentRay.dir, isIntoSurface);
                        hitObjectMaterialIdx = geometry.materialIdx;
                    }
                } 
                else if (geometry.geometryType == IMPLICIT_AABB) {
                    AABB aabb = attr.aabbs[geometry.geometryIdx];
                    float distanceToObject = intersectAABBRayBothSide(aabb, currentRay);
    
                    if (distanceToObject > 0.001f && distanceToObject < nearestIntersectionDistance) {
                        hitEmptyVoidSpace = false;
                        nearestIntersectionDistance = distanceToObject;

                        hitPoint = currentRay.orig + currentRay.dir * distanceToObject;
                        normalAtHitPoint = getAABBNormal(hitPoint, aabb, currentRay.dir, isIntoSurface);
                        hitObjectMaterialIdx = geometry.materialIdx;
                    }                    
                }
            }// end of scene intersection

            if (hitEmptyVoidSpace) {
                if (dot(currentRay.dir, sunDir) > sunSize) // sun
                    accumulativeColor += colorMask * sunColor;
                else if (bounces == 0) // sky
                    accumulativeColor += colorMask * skyColor;
                else // mist
                    accumulativeColor += colorMask * mistColor;

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
                specularSurface(
                    currentRay,
                    colorMask,

                    hitPoint,
                    normalAtHitPoint,
                    material.surfaceColor,
                    randstates,
                    idx
                );
            }
            else if (material.surfaceType == MIRROR) {
                mirrorSurface(
                    currentRay,
                    colorMask,

                    hitPoint,
                    normalAtHitPoint,
                    material.surfaceColor
                );
            }
            else if (material.surfaceType == TRANSPARENT) {
                transparentSurface(
                    currentRay,
                    colorMask,

                    isIntoSurface,
                    hitPoint,
                    normalAtHitPoint,
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
    unsigned int idx = (PATCH_HEIGHT - y - 1) * PATCH_WIDTH + x;

    curand_init(seed, idx, 0, &randstates[idx]);
}

int main(){
    // define dim
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);   
    dim3 grid(PATCH_WIDTH / block.x, PATCH_HEIGHT / block.y, 1);

    // rand states
    curandState_t* randstates;
    cudaMalloc((void**) &randstates, NUM_BLOCKS * sizeof(curandState_t));
    initRandStates<<<grid, block>>>(time(NULL), randstates);

    // scene
    Sphere spheres[] {
        {float3{0.0f, -25.0f, 100.0f} ,15.0f},
        {float3{50.0f, -30.0f, 100.0f}, 10.0f},
        {float3{30.0f, -20.0f, 160.0f}, 20.0f},
        {float3{-40.0f, -25.0f, 90.0f}, 15.0f},
        {float3{40.0f, -5.0f, 130.0f}, 15.0f}
    };

    AABB aabbs[] {
        {float3{-M_INF, -50.0f, -M_INF},float3{M_INF, -40.0f, M_INF}},
        {float3{-80.0f, -40.0f, -10.0f},float3{80.0f, 80.0f, 200.0f}},
        {float3{-60.0f, -40.0f, 130.0f},float3{-30.0f, -10.0f, 160.0f}},
        {float3{30.0f, -40.0f, 120.0f},float3{50.0f, -20.0f, 140.0f}}
    };

    Material materials[] {
        {DIFFUSE, float3{1.0f, 1.0f, 1.0f}, float3{0.75f, 0.75f, 0.75f}},
        {DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{0.75f, 0.75f, 0.75f}},
        {DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{0.9f, 0.2f, 0.1f}},
        {DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{0.1f, 0.2f, 0.9f}},
        {MIRROR,  float3{0.0f, 0.0f, 0.0f}, float3{0.1f, 0.9f, 0.1f}},
        {TRANSPARENT, float3{0.0f, 0.0f, 0.0f}, float3{1.0f, 1.0f, 1.0f}},
        {SPECULAR, float3{0.0f, 0.0f, 0.0f}, float3{1.0f, 1.0f, 0.0f}}
    };

    Geometry geometries[] {
        //{IMPLICIT_AABB, 0, 1},
        {IMPLICIT_AABB, 1, 1},
        {IMPLICIT_AABB, 2, 2},
        {IMPLICIT_SPHERE, 0, 0},
        {IMPLICIT_SPHERE, 1, 3},
        {IMPLICIT_SPHERE, 2, 4},
        {IMPLICIT_SPHERE, 3, 5},
        {IMPLICIT_SPHERE, 4, 6},
        {IMPLICIT_AABB, 3, 5}
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
    
    float3* output = new float3[WIDTH * HEIGHT];
    for (int i = 0; i < WIDTH * HEIGHT; ++i) output[i] = make_float3(0.0f, 0.0f, 0.0f);
    float3* output_h = new float3[PATCH_WIDTH * PATCH_HEIGHT];
    float3* output_d;
    cudaMalloc(&output_d, PATCH_WIDTH * PATCH_HEIGHT * sizeof(float3));

    unsigned int progressRecord = 0;
    printf("Rendering...0%%\n");

    for (unsigned int patch_i = 0; patch_i < PATCH_NUM_X; ++patch_i) {
        for (unsigned int patch_j = 0; patch_j < PATCH_NUM_Y; ++patch_j) {

            for (unsigned int kernalLoop_i = 0; kernalLoop_i < KERNAL_LOOP; ++kernalLoop_i) {   
                renderKernal <<< grid, block >>> (
                    output_d,
                    patch_i*PATCH_WIDTH,
                    (PATCH_NUM_Y - patch_j - 1)*PATCH_HEIGHT,
                    attr,
                    randstates);
                cudaMemcpy(output_h, output_d, PATCH_WIDTH * PATCH_HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);
                
                for (unsigned int i = 0; i < PATCH_WIDTH; ++i) {
                    for (unsigned int j = 0; j < PATCH_HEIGHT; ++j) {
                        output[(patch_j*PATCH_HEIGHT + j) * WIDTH + patch_i*PATCH_WIDTH + i] += output_h[j * PATCH_WIDTH + i];
                    }
                }  

                unsigned int progressPercent = ((patch_i * PATCH_NUM_Y + patch_j) * KERNAL_LOOP + kernalLoop_i) * 10 / PATCH_NUM_X / PATCH_NUM_Y / KERNAL_LOOP;
                if (progressRecord != progressPercent) {
                    progressRecord = progressPercent;
                    printf("Rendering...%d0%%\n", progressRecord);
                }
            }
            
            // temp file
            // char name[50];
            // sprintf(name, "patch_%d_%d.ppm", patch_i, patch_j);
            // writeToPPM(name, WIDTH, HEIGHT, output);
        }
    }

    for (int i = 0; i < WIDTH * HEIGHT; ++i) output[i] /= KERNAL_LOOP;

    printf("Rendering...100%%\n");
    printf("Done!\n");
    
    // output
    writeToPPM("result.ppm", WIDTH, HEIGHT, output);

    // clean
    cudaFree(spheres_d); 
    cudaFree(aabbs_d);  
    cudaFree(materials_d);
    cudaFree(geometries_d);

    cudaFree(output_d);  
    cudaFree(randstates);

    delete[] output;
    delete[] output_h;
}