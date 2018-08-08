#pragma once

#include "cutil_math.h"

// vertex, face and mesh

using Vertex = float3;
using Face = uint3;

struct Mesh{
    uint vertexNum;
    uint faceNum;

    Vertex* vertices;
    Face* faces;

    //AABB boundingBox;
};

// for gpu communication, we need SOA only data structure

struct Mesh_IndexOnly { // index only
    uint vertexNum;
    uint faceNum;
    
    uint vertexStartIndex;
    uint faceStartIndex;
    
    //AABB boundingBox;
};

struct Meshes_SOA { // struct of array only
    Vertex* vertices;
    Face* faces;
    Mesh_IndexOnly* meshes;
};

void convertMeshAOSToSOA(Mesh* meshPtr, uint meshNum, Meshes_SOA& meshSOA) {
    uint totalVertexNum = 0;
    uint totalFaceNum = 0;

    for (uint meshIndex = 0; meshIndex < meshNum; ++meshIndex) {
        totalVertexNum += meshPtr[meshIndex].vertexNum;
        totalFaceNum += meshPtr[meshIndex].faceNum;
    }

    meshSOA.vertices = new Vertex[totalVertexNum];
    meshSOA.faces = new Face[totalFaceNum];
    meshSOA.meshes = new Mesh_IndexOnly[meshNum];

    uint vertexIndex = 0;
    uint faceIndex = 0;

    for (uint meshIndex = 0; meshIndex < meshNum; ++meshIndex) {

        meshSOA.meshes[meshIndex].vertexNum = meshPtr[meshIndex].vertexNum;
        meshSOA.meshes[meshIndex].faceNum = meshPtr[meshIndex].faceNum;
        meshSOA.meshes[meshIndex].vertexStartIndex = vertexIndex;
        meshSOA.meshes[meshIndex].faceStartIndex = faceIndex;

        for (uint i = 0; i < meshPtr[meshIndex].vertexNum; ++i) {
            meshSOA.vertices[vertexIndex++] = meshPtr[meshIndex].vertices[i];
        }

        for (uint i = 0; i < meshPtr[meshIndex].faceNum; ++i) {
            meshSOA.faces[faceIndex++] = meshPtr[meshIndex].faces[i];
        }

        delete[] meshPtr[meshIndex].vertices;
        delete[] meshPtr[meshIndex].faces;
    }
}

void deleteMeshSOA(Meshes_SOA& meshSOA) {
    delete[] meshSOA.vertices;
    delete[] meshSOA.faces;
    delete[] meshSOA.meshes;
}

Mesh getMeshFromSOA(Meshes_SOA& meshSOA, uint meshIndex) {
    Mesh mesh;
    Mesh_IndexOnly meshIndexOnly = meshSOA.meshes[meshIndex];
    mesh.vertexNum = meshIndexOnly.vertexNum;
    mesh.faceNum = meshIndexOnly.faceNum;
    mesh.vertices = meshSOA.vertices[meshIndexOnly.vertexStartIndex];
    mesh.faces = meshSOA.faces[meshIndexOnly.faceStartIndex];
    return mesh;
}

// basic mesh manipulation

#define VERT_P(i) mesh.vertices[i]

static void translateMesh(Mesh& mesh, const float3& transVec) {
    for (uint i = 0; i < mesh.vertexNum; ++i) {
        VERT_P(i) += transVec;
    }
}

static void scaleMesh(Mesh& mesh, const float3& scaleVec) {
    for (uint i = 0; i < meshRef.vertexNum; ++i) {
        VERT_P(i).x *= scaleVec.x;
        VERT_P(i).y *= scaleVec.y;
        VERT_P(i).z *= scaleVec.z;
    }
}

#define ROTATE_MESH_HELPER(_X_,_Y_,_Z_)                                      \
if (rotateVec._X_ != 0) {                                                    \
    float cosTheta = cos(rotateVec._X_);                                     \
    float sinTheta = sin(rotateVec._X_);                                     \
    for (uint i = 0; i < meshRef.vertexNum; ++i) {                           \
        VERT_P(i)._Y_ = VERT_P(i)._Y_ * cosTheta - VERT_P(i)._Z_ * sinTheta; \
        VERT_P(i)._Z_ = VERT_P(i)._Y_ * sinTheta + VERT_P(i)._Z_ * cosTheta; \
    }                                                                        \
}

static void rotateMesh(Mesh& mesh, const float3& rotateVec) {
    ROTATE_MESH_HELPER(x,y,z)
    ROTATE_MESH_HELPER(y,x,z)
    ROTATE_MESH_HELPER(z,x,y)
}