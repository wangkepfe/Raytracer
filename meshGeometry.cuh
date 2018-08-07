#pragma once

#include "cutil_math.h"

struct Vertex{
    float3 position;
    //float3 normal;
    //float2 uv;
};

using Face = uint3;

struct TriMesh{
    uint vertexNum;
    uint faceNum;
    
    uint vertexStartIndex;
    uint faceStartIndex;
    
    //AABB boundingBox;
};

void printTriMesh(const TriMesh& m)
{
    std::cout << "vertexNum is " << m.vertexNum << std::endl
              << "faceNum is " << m.faceNum << std::endl
              << "vertexStartIndex is " << m.vertexStartIndex << std::endl
              << "faceStartIndex is " << m.faceStartIndex << std::endl
              << std::endl;
}

struct TriMesh_d{
    uint vertexNum;
    uint faceNum;

    Vertex* vertices;
    Face* faces;

    //AABB boundingBox;

    __device__ TriMesh_d(uint vertexNum, uint faceNum, Vertex* vertices, Face* faces)
    :vertexNum {vertexNum},
    faceNum {faceNum},
    vertices {vertices},
    faces {faces}
    {}
};

#define MESH_PARAM_REF Vertex* vertices, Face* faces, TriMesh& meshRef
#define MESH_PARAM_REF_ROOF Vertex* vertices, Face* faces, TriMesh& meshRef, uint& verticeRoof, uint& faceRoof

#define VERTEX_POSITION(__I__) vertices[meshRef.vertexStartIndex + __I__].position
#define FACE(__I__) faces[meshRef.faceStartIndex + __I__]

static void translateTriMesh(MESH_PARAM_REF, const float3& transVec) {
    for (uint i = 0; i < meshRef.vertexNum; ++i) {
        VERTEX_POSITION(i) += transVec;
    }
}

static void scaleTriMesh(MESH_PARAM_REF, const float3& scaleVec) {
    for (uint i = 0; i < meshRef.vertexNum; ++i) {
        VERTEX_POSITION(i).x *= scaleVec.x;
        VERTEX_POSITION(i).y *= scaleVec.y;
        VERTEX_POSITION(i).z *= scaleVec.z;
    }
}

static void rotateTriMesh(MESH_PARAM_REF, const float3& rotateVec) {
    if (rotateVec.x != 0) {
        float cosTheta = cos(rotateVec.x);
        float sinTheta = sin(rotateVec.x);
        for (uint i = 0; i < meshRef.vertexNum; ++i) {
            VERTEX_POSITION(i).y = VERTEX_POSITION(i).y * cosTheta - VERTEX_POSITION(i).z * sinTheta;
            VERTEX_POSITION(i).z = VERTEX_POSITION(i).y * sinTheta + VERTEX_POSITION(i).z * cosTheta;
        }
    }

    if (rotateVec.y != 0) {
        float cosTheta = cos(rotateVec.y);
        float sinTheta = sin(rotateVec.y);
        for (uint i = 0; i < meshRef.vertexNum; ++i) {
            VERTEX_POSITION(i).x = VERTEX_POSITION(i).x * cosTheta - VERTEX_POSITION(i).z * sinTheta;
            VERTEX_POSITION(i).z = VERTEX_POSITION(i).x * sinTheta + VERTEX_POSITION(i).z * cosTheta;
        }
    }

    if (rotateVec.z != 0) {
        float cosTheta = cos(rotateVec.z);
        float sinTheta = sin(rotateVec.z);
        for (uint i = 0; i < meshRef.vertexNum; ++i) {
            VERTEX_POSITION(i).x = VERTEX_POSITION(i).x * cosTheta - VERTEX_POSITION(i).y * sinTheta;
            VERTEX_POSITION(i).y = VERTEX_POSITION(i).x * sinTheta + VERTEX_POSITION(i).y * cosTheta;
        }
    }
}