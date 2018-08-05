#pragma once

#include "cutil_math.h"

struct TriangleMesh{
    uint vertexNum;
    uint faceNum;

    float3* vertexPositions;
    uint3* faces;

    // future features
    //AABB boundingBox;
    //float3* faceVertexNormals;
    //float3* faceVertexUVs;
};

static void translateTriangleMesh(TriangleMesh& meshRef, const float3& transVec) {
    for (uint i = 0; i < meshRef.vertexNum; ++i) {
        meshRef.vertexPositions[i] += transVec;
    }
}

static void scaleTriangleMesh(TriangleMesh& meshRef, const float3& scaleVec) {
    for (uint i = 0; i < meshRef.vertexNum; ++i) {
        meshRef.vertexPositions[i].x *= scaleVec.x;
        meshRef.vertexPositions[i].y *= scaleVec.y;
        meshRef.vertexPositions[i].z *= scaleVec.z;
    }
}

static void rotateTriangleMesh(TriangleMesh& meshRef, const float3& rotateVec) {
    if (rotateVec.x != 0) {
        float cosTheta = cos(rotateVec.x);
        float sinTheta = sin(rotateVec.x);
        for (uint i = 0; i < meshRef.vertexNum; ++i) {
            meshRef.vertexPositions[i].y = meshRef.vertexPositions[i].y * cosTheta - meshRef.vertexPositions[i].z * sinTheta;
            meshRef.vertexPositions[i].z = meshRef.vertexPositions[i].y * sinTheta + meshRef.vertexPositions[i].z * cosTheta;
        }
    }

    if (rotateVec.y != 0) {
        float cosTheta = cos(rotateVec.y);
        float sinTheta = sin(rotateVec.y);
        for (uint i = 0; i < meshRef.vertexNum; ++i) {
            meshRef.vertexPositions[i].x = meshRef.vertexPositions[i].x * cosTheta - meshRef.vertexPositions[i].z * sinTheta;
            meshRef.vertexPositions[i].z = meshRef.vertexPositions[i].x * sinTheta + meshRef.vertexPositions[i].z * cosTheta;
        }
    }

    if (rotateVec.z != 0) {
        float cosTheta = cos(rotateVec.z);
        float sinTheta = sin(rotateVec.z);
        for (uint i = 0; i < meshRef.vertexNum; ++i) {
            meshRef.vertexPositions[i].x = meshRef.vertexPositions[i].x * cosTheta - meshRef.vertexPositions[i].y * sinTheta;
            meshRef.vertexPositions[i].y = meshRef.vertexPositions[i].x * sinTheta + meshRef.vertexPositions[i].y * cosTheta;
        }
    }
}

static void deleteTriangleMeshArray(TriangleMesh* ptr, uint size) {
    for (uint i = 0; i < size; ++i) {
        delete [] ptr[i].vertexPositions;
        delete [] ptr[i].faces;
    }
    delete [] ptr;
}