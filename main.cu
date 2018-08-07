/***********************************
 *                                 *
 *      A very n1ce raytracer      *
 *                                 *
 * with 100% authentic home-made   *
 * CUDA codes and imported codes   *
 * from USA and all over the world *
 * NO bug-free warranty            *
 *                                 *
 ***********************************/
 
// Author: Ke Wang
// Summer 2018

// C/C++
#include <iostream>
#include <time.h>

// CUDA
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

// Macro and constants
#include "macro.h"
#include "constants.cuh"

// C++ header libs
#include "io_utils.h"

// CUDA header libs
#include "geometry.cuh"
#include "geometryIntersect.cuh"
#include "implicitGeometry.cuh"
#include "material.cuh"
#include "surface.cuh"
#include "meshGeometry.cuh"

struct Attr{
    uint numberOfObject;

    // implicit primitives
    Sphere* spheres;
    AABB* aabbs;

    // tri mesh
    Vertex* vertices;
    Face* faces;
    TriMesh* triMeshes;

    // mat
    Material* materials;

    // geo
    Geometry* geometries;
};

__global__ void renderKernal (
    float3 *output,
    uint2 patch_offset,
    Attr attr, 
    curandState_t* randstates)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;   
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint idx = (PATCH_HEIGHT - y - 1) * PATCH_WIDTH + x;

    uint realX = patch_offset.x + x;   
    uint realY = patch_offset.y + y;

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
                    TriMesh triMesh_h = attr.triMeshes[geometry.geometryIdx];
                    TriMesh_d triMesh(triMesh_h.vertexNum, triMesh_h.faceNum, attr.vertices + triMesh_h.vertexStartIndex, attr.faces + triMesh_h.faceStartIndex);

                    float distanceToObject = RayTriMeshIntersection(normalAtHitPoint, isIntoSurface, triMesh, currentRay);

                    if (distanceToObject > M_EPSILON && distanceToObject < nearestIntersectionDistance) {
                        hitEmptyVoidSpace = false;
                        nearestIntersectionDistance = distanceToObject;
                        hitPoint = currentRay.orig + currentRay.dir * distanceToObject;
                        hitObjectMaterialIdx = geometry.materialIdx;
                    }
                }
            }// end of scene intersection

            if (hitEmptyVoidSpace) {
                if (dot(currentRay.dir, sunDir) > sunSize)
                    accumulativeColor += colorMask * sunColor;
                else if (bounces == 0)
                    accumulativeColor += colorMask * skyColor;
                else
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
    curandState_t* randstates_d;
    cudaMalloc((void**) &randstates_d, NUM_BLOCKS * sizeof(curandState_t));
    initRandStates<<<grid, block>>>(time(NULL), randstates_d);

    // build the scene
    AABB myHouseAABB = AABB {float3{-80.0f, -40.0f, -10.0f},float3{80.0f, 80.0f, 200.0f}};
    Sphere sphereOnTheCeiling = Sphere {float3{0.0f, 80.0f, 100.0f} ,20.0f};

    Vertex* vertices = new Vertex[VERTEX_POOL_SIZE];
    Face* faces = new Face[FACE_POOL_SIZE];

    uint verticeRoof = 0;
    uint faceRoof = 0;

    TriMesh bunnyMesh;

    loadObj("stanford_bunny.obj", vertices, faces, bunnyMesh, verticeRoof, faceRoof);

    scaleTriMesh(vertices, faces, bunnyMesh, float3{5.0f, 5.0f, 5.0f});
    translateTriMesh(vertices, faces, bunnyMesh, float3{0.0f, -10.0f, 120.0f});

    Material whiteDiffuse {DIFFUSE, float3{0.0f, 0.0f, 0.0f}, float3{0.75f, 0.75f, 0.75f}};
    Material whiteLight {DIFFUSE, float3{1.0f, 1.0f, 1.0f}, float3{0.75f, 0.75f, 0.75f}};

    Sphere spheres[] {
        sphereOnTheCeiling
    };
    AABB aabbs[] {
        myHouseAABB
    };

    TriMesh triMeshes[] {
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

    // copy data to cuda
    CUDA_MALLOC_MEMCPY_HOST_TO_DEVICE(Sphere, spheres_d, spheres);
    CUDA_MALLOC_MEMCPY_HOST_TO_DEVICE(AABB, aabbs_d, aabbs);

    CUDA_MALLOC_MEMCPY_HOST_TO_DEVICE_SIZE(Vertex, vertices_d, vertices, verticeRoof);
    CUDA_MALLOC_MEMCPY_HOST_TO_DEVICE_SIZE(Face, faces_d, faces, faceRoof);
    CUDA_MALLOC_MEMCPY_HOST_TO_DEVICE(TriMesh, triMeshes_d, triMeshes);

    CUDA_MALLOC_MEMCPY_HOST_TO_DEVICE(Material, materials_d, materials);
    CUDA_MALLOC_MEMCPY_HOST_TO_DEVICE(Geometry, geometries_d, geometries);

    Attr attr {
        sizeof(geometries) / sizeof(Geometry), 

        spheres_d, 
        aabbs_d,

        vertices_d,
        faces_d,
        triMeshes_d,

        materials_d, 
        geometries_d
    };
    
    // start rendering
    float3* output = new float3[WIDTH * HEIGHT];
    for (uint i = 0; i < WIDTH * HEIGHT; ++i) output[i] = make_float3(0.0f, 0.0f, 0.0f);
    float3* output_patch = new float3[PATCH_WIDTH * PATCH_HEIGHT];
    float3* output_d;
    cudaMalloc(&output_d, PATCH_WIDTH * PATCH_HEIGHT * sizeof(float3));

    uint progressRecord = 0;
    printf("Rendering...0%%\n");

    for (uint patch_i = 0; patch_i < PATCH_NUM_X; ++patch_i) {
        for (uint patch_j = 0; patch_j < PATCH_NUM_Y; ++patch_j) {
            for (uint kernalLoop_i = 0; kernalLoop_i < KERNAL_LOOP; ++kernalLoop_i) {   
                renderKernal <<< grid, block >>> (
                    output_d,
                    uint2{patch_i*PATCH_WIDTH, (PATCH_NUM_Y - patch_j - 1)*PATCH_HEIGHT},
                    attr,
                    randstates_d);
                cudaMemcpy(output_patch, output_d, PATCH_WIDTH * PATCH_HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);
                
                for (uint i = 0; i < PATCH_WIDTH; ++i) {
                    for (uint j = 0; j < PATCH_HEIGHT; ++j) {
                        output[(patch_j*PATCH_HEIGHT + j) * WIDTH + patch_i*PATCH_WIDTH + i] += output_patch[j * PATCH_WIDTH + i];
                    }
                }  

                uint progressPercent = ((patch_i * PATCH_NUM_Y + patch_j) * KERNAL_LOOP + kernalLoop_i) * 10 / PATCH_NUM_X / PATCH_NUM_Y / KERNAL_LOOP;
                if (progressRecord != progressPercent) {
                    progressRecord = progressPercent;
                    printf("Rendering...%d0%%\n", progressRecord);
                }
            }
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
    
    cudaFree(vertices_d);
    cudaFree(faces_d);
    cudaFree(triMeshes_d);

    cudaFree(materials_d);
    cudaFree(geometries_d);

    cudaFree(output_d);  

    cudaFree(randstates_d);

    delete[] output;
    delete[] output_patch;
}