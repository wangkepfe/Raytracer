#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <time.h>

#include "io_utils.h"
#include "scene.cuh"
#include "macro.h"
#include "diffuse.cuh"
#include "lock.cuh"

__global__ void renderKernal (float3 *output, Scene* scene, curandState_t* randstates) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = (HEIGHT - y - 1) * WIDTH + x; 

    float3 camOrig = make_float3(50.0f, 52.0f, 295.6f);
    float3 camDir = normalize(make_float3(0.0f, -0.042612f, -1.0f));

    float3 deltaX = make_float3(WIDTH * 0.5135f / HEIGHT, 0.0f, 0.0f);
    float3 deltaY = normalize(cross(deltaX, camDir)) * 0.5135f; //(.5135 is field of view angle)

    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f); // res is final pixel color       
   
    for (int s = 0; s < SAMPLES; s++) {  // samples per pixel
        // compute primary ray
        float3 rd = camDir + deltaX*((.25 + x) / WIDTH - .5) + deltaY*((.25 + y) / HEIGHT - .5);
        Ray r = Ray(camOrig + rd * 40.0f, normalize(rd));

        // add incoming radiance to pixelcolor
        float3 sampleAccuColor = make_float3(0.0f, 0.0f, 0.0f);
        float3 colorMask = make_float3(1.0f, 1.0f, 1.0f); 

        // ray bounce loop (no Russian Roulette used) 
        for (int bounces = 0; bounces < RAY_BOUNCE; bounces++) {
            float t;
            float3 x, n;
            Material mat;      
        
            if (!scene->intersect(r, t, x, n, mat))
                break; 

            sampleAccuColor += colorMask * mat.emi;

            if (mat.refl == DIFF) {
                diffuseShader(r, colorMask, x, n, mat, randstates, idx);
            }
        }

        //printf("sampleAccuColor.x = %f\n", sampleAccuColor.x);
        finalColor += sampleAccuColor * (1.0f / (float)SAMPLES); 
    }
    output[idx] = make_float3(clamp(finalColor.x, 0.0f, 1.0f),
                              clamp(finalColor.y, 0.0f, 1.0f),
                              clamp(finalColor.z, 0.0f, 1.0f));
    // if(idx % 10000 == 0)
    //     printf("idx = %d, (%.3f, %.3f, %.3f)\n", idx, output[idx].x, output[idx].y, output[idx].z); 
    
    //float r = curand_uniform(&randstates[idx]);
    //output[idx] = make_float3(r, r, r);
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

    // output
    float3* output_h = new float3[WIDTH * HEIGHT];
    float3* output_d;
    cudaMalloc(&output_d, WIDTH * HEIGHT * sizeof(float3));

    // rand states
    curandState_t* randstates;
    cudaMalloc((void**) &randstates, NUM_BLOCKS * sizeof(curandState_t));
    
    // scene
    Scene* scene_h = new Scene;
    Scene* scene_d;
    cudaMalloc(&scene_d, sizeof(Scene));

    // copy host to device
    cudaMemcpy(scene_d, scene_h, sizeof(Scene), cudaMemcpyHostToDevice);

    // run
    initRandStates<<<grid, block>>>(time(NULL), randstates);
    renderKernal <<< grid, block >>> (output_d, scene_d, randstates);

    // copy device to host
    cudaMemcpy(output_h, output_d, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

    // output
    writeToPPM("result.ppm", WIDTH, HEIGHT, output_h);

    // clean
    cudaFree(scene_d);  
    cudaFree(output_d);  
    cudaFree(randstates);
    delete[] scene_h;
    delete[] output_h;
}