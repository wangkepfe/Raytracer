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
    for (int i = 0; i < width * height; i++)
        fprintf(f, "%d %d %d ", 
            toInt(buffer[i].x),
            toInt(buffer[i].y),
            toInt(buffer[i].z));
    fclose(f);
    printf("Successfully wrote result image to %s\n", fname);
}

inline void loadObj(const char* fname, MESH_PARAM_REF_ROOF) {
    objl::Loader Loader;
    bool loadout = Loader.LoadFile(fname);

    if (!loadout) {
        printf("load failed!!\n");
        return;
    }

    objl::Mesh curMesh = Loader.LoadedMeshes[0];

    // position
    meshRef.vertexNum = curMesh.Vertices.size();
    meshRef.vertexStartIndex = verticeRoof;

    for (int i = 0; i < curMesh.Vertices.size(); ++i) {
        VERTEX_POSITION(i) = float3{curMesh.Vertices[i].Position.X, curMesh.Vertices[i].Position.Y, curMesh.Vertices[i].Position.Z};
    }

    verticeRoof += meshRef.vertexNum;

    // face
    meshRef.faceNum = curMesh.Indices.size() / 3;
    meshRef.faceStartIndex = faceRoof;

    for (int i = 0; i < curMesh.Indices.size(); i += 3) {
        FACE(i / 3) = uint3{curMesh.Indices[i], curMesh.Indices[i + 1], curMesh.Indices[i + 2]};
    }

    printf("Successfully loaded model %s\n", fname);
}

#endif