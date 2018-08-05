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
#include "meshGeometry.cuh"

enum{
    DIFFUSE,
    SPECULAR,
    MIRROR,
    TRANSPARENT,
};

enum{
    IMPLICIT_SPHERE,
    IMPLICIT_AABB,
    TRIANGLE_MESH,
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
    TriangleMesh* triMeshes;
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
    uint patch_width_offset,
    uint patch_height_offset,
    Attr attr, 
    curandState_t* randstates)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;   
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint idx = (PATCH_HEIGHT - y - 1) * PATCH_WIDTH + x;

    uint realX = patch_width_offset + x;   
    uint realY = patch_height_offset + y;

    float3 deltaX = make_float3(WIDTH * camFov / HEIGHT, 0.0f, 0.0f);
    float3 deltaY = make_float3(0.0f, camFov, 0.0f);

    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f);

    float3 rayDirection = normalize(camDir + deltaX * (realX * 2.0f / WIDTH - 1.0f) + deltaY * (realY * 2.0f / HEIGHT - 1.0f));

    for (uint s = 0; s < SAMPLES; s++) {//sample
        Ray currentRay {camOrig, rayDirection};

        float3 accumulativeColor = make_float3(0.0f, 0.0f, 0.0f);
        float3 colorMask = make_float3(1.0f, 1.0f, 1.0f);

        for (uint bounces = 0; bounces < RAY_BOUNCE; bounces++) {//bounce
            float3 hitPoint;
            float3 normalAtHitPoint;
            bool isIntoSurface = true;

            float nearestIntersectionDistance = M_INF;
            bool hitEmptyVoidSpace = true;
            int hitObjectMaterialIdx = 0;

            for (uint objectIdx = 0; objectIdx < attr.numberOfObject; ++objectIdx) {// scene intersection
                Geometry geometry = attr.geometries[objectIdx];

                if (geometry.geometryType == IMPLICIT_SPHERE) {
                    Sphere sphere = attr.spheres[geometry.geometryIdx];
                    float distanceToObject = intersectSphereRay(sphere, currentRay);
    
                    if (distanceToObject > M_EPSILON && distanceToObject < nearestIntersectionDistance) {
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
    
                    if (distanceToObject > M_EPSILON && distanceToObject < nearestIntersectionDistance) {
                        hitEmptyVoidSpace = false;
                        nearestIntersectionDistance = distanceToObject;

                        hitPoint = currentRay.orig + currentRay.dir * distanceToObject;
                        normalAtHitPoint = getAABBNormal(hitPoint, aabb, currentRay.dir, isIntoSurface);
                        hitObjectMaterialIdx = geometry.materialIdx;
                    }                    
                }
                else if (geometry.geometryType == TRIANGLE_MESH) {
                    TriangleMesh triMesh = attr.triMeshes[geometry.geometryIdx];
                    float distanceToObject = RayTriangleMeshIntersection(normalAtHitPoint, isIntoSurface, triMesh, currentRay);

                    if (distanceToObject > M_EPSILON && distanceToObject < nearestIntersectionDistance) {
                        hitEmptyVoidSpace = false;
                        nearestIntersectionDistance = distanceToObject;
                        hitPoint = currentRay.orig + currentRay.dir * distanceToObject;
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

__global__ void initRandStates(uint seed, curandState_t* randstates) {
    uint x = blockIdx.x*blockDim.x + threadIdx.x;   
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint idx = (PATCH_HEIGHT - y - 1) * PATCH_WIDTH + x;

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
    AABB myHouseAABB = AABB{float3{-80.0f, -40.0f, -10.0f},float3{80.0f, 80.0f, 200.0f}};
    Sphere sphereOnTheCeiling = Sphere{float3{0.0f, 80.0f, 100.0f} ,20.0f};

    TriangleMesh bunnyMesh = loadObj("stanford_bunny.obj");
    scaleTriangleMesh(bunnyMesh, float3{5.0f, 5.0f, 5.0f});
    translateTriangleMesh(bunnyMesh, float3{0.0f, -10.0f, 120.0f});

    Material whiteDiffuse{DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{0.75f, 0.75f, 0.75f}};
    Material whiteLight{DIFFUSE, float3{1.0f, 1.0f, 1.0f}, float3{0.75f, 0.75f, 0.75f}};

    Sphere spheres[] {
        sphereOnTheCeiling
    };
    AABB aabbs[] {
        myHouseAABB
    };
    TriangleMesh triMeshes[] {
        bunnyMesh
    };
    Material materials[] {
        whiteDiffuse,
        whiteLight
    };
    Geometry myHouse{IMPLICIT_AABB, 0, 0};
    Geometry myCeilingLight{IMPLICIT_SPHERE, 0, 1};

    Geometry geometries[] {
        myHouse,
        myCeilingLight,
        {TRIANGLE_MESH, 0, 0}
    };

    Sphere* spheres_d;
    AABB* aabbs_d;
    TriangleMesh* triMeshes_d;
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

    // tri mesh cuda malloc and memcopy
    uint numOfTriMesh = sizeof(triMeshes) / sizeof(TriangleMesh);
    cudaMalloc(&triMeshes_d, sizeof(triMeshes));
    for (uint i = 0; i < numOfTriMesh; ++i) {
        cudaMalloc(&(triMeshes_d[i].vertexPositions), sizeof(triMeshes[i].vertexPositions));
        cudaMalloc(&(triMeshes_d[i].faces), sizeof(triMeshes[i].faces));

        cudaMemcpy(&(triMeshes_d[i].vertexNum), &(triMeshes[i].vertexNum), sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(&(triMeshes_d[i].faceNum), &(triMeshes[i].faceNum), sizeof(uint), cudaMemcpyHostToDevice);

        cudaMemcpy(triMeshes_d[i].vertexPositions, triMeshes[i].vertexPositions, sizeof(triMeshes[i].vertexPositions), cudaMemcpyHostToDevice);
        cudaMemcpy(triMeshes_d[i].faces, triMeshes[i].faces, sizeof(triMeshes[i].faces), cudaMemcpyHostToDevice);  
    }


    Attr attr {
        sizeof(geometries) / sizeof(Geometry), 
        spheres_d, 
        aabbs_d,
        triMeshes_d,
        materials_d, 
        geometries_d
    };
    
    float3* output = new float3[WIDTH * HEIGHT];
    for (uint i = 0; i < WIDTH * HEIGHT; ++i) output[i] = make_float3(0.0f, 0.0f, 0.0f);
    float3* output_h = new float3[PATCH_WIDTH * PATCH_HEIGHT];
    float3* output_d;
    cudaMalloc(&output_d, PATCH_WIDTH * PATCH_HEIGHT * sizeof(float3));

    uint progressRecord = 0;
    printf("Rendering...0%%\n");

    for (uint patch_i = 0; patch_i < PATCH_NUM_X; ++patch_i) {
        for (uint patch_j = 0; patch_j < PATCH_NUM_Y; ++patch_j) {

            for (uint kernalLoop_i = 0; kernalLoop_i < KERNAL_LOOP; ++kernalLoop_i) {   
                renderKernal <<< grid, block >>> (
                    output_d,
                    patch_i*PATCH_WIDTH,
                    (PATCH_NUM_Y - patch_j - 1)*PATCH_HEIGHT,
                    attr,
                    randstates);
                cudaMemcpy(output_h, output_d, PATCH_WIDTH * PATCH_HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);
                
                for (uint i = 0; i < PATCH_WIDTH; ++i) {
                    for (uint j = 0; j < PATCH_HEIGHT; ++j) {
                        output[(patch_j*PATCH_HEIGHT + j) * WIDTH + patch_i*PATCH_WIDTH + i] += output_h[j * PATCH_WIDTH + i];
                    }
                }  

                uint progressPercent = ((patch_i * PATCH_NUM_Y + patch_j) * KERNAL_LOOP + kernalLoop_i) * 10 / PATCH_NUM_X / PATCH_NUM_Y / KERNAL_LOOP;
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

    for (uint i = 0; i < WIDTH * HEIGHT; ++i) output[i] /= KERNAL_LOOP;

    printf("Rendering...100%%\n");
    printf("Done!\n");
    
    // output
    writeToPPM("result.ppm", WIDTH, HEIGHT, output);

    // clean
    cudaFree(spheres_d); 
    cudaFree(aabbs_d);  
    cudaFree(materials_d);
    cudaFree(geometries_d);

    for (uint i = 0; i < numOfTriMesh; ++i) {
        cudaFree(triMeshes_d[i].vertexPositions);
        cudaFree(triMeshes_d[i].faces);
    }
    cudaFree(triMeshes_d);  

    cudaFree(output_d);  
    cudaFree(randstates);

    deleteTriangleMeshArray(triMeshes, numOfTriMesh);
    delete[] output;
    delete[] output_h;
}