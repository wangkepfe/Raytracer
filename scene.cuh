#pragma once

#include <array>
#include "sphere.cuh"

using namespace std;

#define SCENE_SPHERE_NUM 9

struct Scene{
    Sphere spheres[SCENE_SPHERE_NUM];

    Scene() : spheres {
        // light
        Sphere {600.0f,
                float3{ 50.0f        , 681.6f - .77f, 81.6f          }, 
                Material{float3{ 2.0f, 1.8f, 1.6f }, float3{ 0.0f, 0.0f, 0.0f },  DIFF } },

        // left wall
        Sphere {1e5f ,
                float3{ 1e5f + 1.0f  , 40.8f        , 81.6f         },
                Material{float3{ 0.0f, 0.0f, 0.0f }, float3{ 0.75f, 0.25f, 0.25f}, DIFF } },

        // right wall
        Sphere {1e5f ,
            float3{ -1e5f + 99.0f, 40.8f, 81.6f },
            Material{float3{ 0.0f, 0.0f, 0.0f }, float3{ .25f, .25f, .75f }, DIFF } },

        // top wall
        Sphere {1e5f ,
            float3{ 50.0f, -1e5f + 81.6f, 81.6f },
            Material{float3{ 0.0f, 0.0f, 0.0f }, float3{ .75f, .75f, .75f }, DIFF } },

        // bottom wall
        Sphere {1e5f ,
            float3{ 50.0f, 1e5f, 81.6f },
            Material{float3{ 0.0f, 0.0f, 0.0f }, float3{ .75f, .75f, .75f }, DIFF } },

        // front wall
        Sphere {1e5f ,
            float3{ 50.0f, 40.8f, -1e5f + 600.0f },
            Material{float3{ 0.0f, 0.0f, 0.0f }, float3{ .75f, .75f, .75f }, DIFF } },

        // back wall
        Sphere {1e5f ,
            float3{ 50.0f, 40.8f, 1e5f },
            Material{float3{ 0.0f, 0.0f, 0.0f }, float3{ .75f, .75f, .75f }, DIFF } },

        // small sphere 1
        Sphere {1e5f ,
            float3{ 27.0f, 16.5f, 47.0f },
            Material{float3{ 0.0f, 0.0f, 0.0f }, float3{ 1.0f, 1.0f, 1.0f }, DIFF } },

        // small sphere 2
        Sphere {1e5f ,
            float3{ 73.0f, 16.5f, 78.0f },
            Material{float3{ 0.0f, 0.0f, 0.0f }, float3{ 1.0f, 1.0f, 1.0f }, DIFF } }
    }
    {}

    __device__ bool intersect(
        const Ray &r,   // ray
        float &t,       // distance
        float3 &x,      // hit point
        float3 &n,      // normal
        Material& mat)  // mat
    {
        float d, inf = 1e20;
        t = inf;
        Sphere s;
        for (int i = 0; i < SCENE_SPHERE_NUM; ++i) {  
            d = spheres[i].intersect(r);
            if (d != 0 && d < t){
                t = d;  
                s = spheres[i];
            }
        }
        x = r.orig + r.dir*t;
        n = s.getNormalAt(x, r);
        mat = s.mat;
        return t < inf;
    }
};