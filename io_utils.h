#ifndef __IO_UTILS_H__
#define __IO_UTILS_H__

#include <iostream>
#include "cuda_runtime.h"
#include "cutil_math.h"

#include "OBJ_Loader.h"
#include "meshGeometry.cuh"

// gamma correction
inline int toInt(float x){ return int(pow(clamp(x, 0.0f, 1.0f), 1 / 2.2) * 255 + .5); }  

inline void writeToPPM(const char* fname, int width, int height, float3* buffer){
    FILE *f = fopen(fname, "w");          
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width * height; i++)  // loop over pixels, write RGB values
        fprintf(f, "%d %d %d ", 
            toInt(buffer[i].x),
            toInt(buffer[i].y),
            toInt(buffer[i].z));
    fclose(f);
    printf("Successfully wrote result image to %s\n", fname);
}

inline TriangleMesh loadObj(const char* fname) {
    TriangleMesh meshResult;

    objl::Loader Loader;
    bool loadout = Loader.LoadFile(fname);

    if (!loadout) {
        printf("load failed!!\n");
        return {};
    }

    for (int i = 0; i < Loader.LoadedMeshes.size(); i++) {
        objl::Mesh curMesh = Loader.LoadedMeshes[i];

        meshResult.vertexNum = curMesh.Vertices.size();
        meshResult.vertexPositions = new float3[meshResult.vertexNum];

        for (int j = 0; j < curMesh.Vertices.size(); j++) {
            meshResult.vertexPositions[j].x = curMesh.Vertices[j].Position.X;
        }

        meshResult.faceNum = curMesh.Indices.size() / 3;
        meshResult.faces = new uint3[meshResult.faceNum];

        for (int j = 0; j < curMesh.Indices.size(); j += 3) {
            meshResult.faces[j / 3] = uint3{curMesh.Indices[j], curMesh.Indices[j + 1], curMesh.Indices[j + 2]};
        }

        break;
    }

    printf("Successfully loaded model %s\n", fname);

    return meshResult;
}

#endif