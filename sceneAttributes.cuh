#pragma once

#include "geometry.cuh"
#include "implicitGeometry.cuh"
#include "material.cuh"
#include "meshGeometry.cuh"

struct Attr {
    uint numberOfObject;
    uint numberOfLights;

    // implicit primitives
    Sphere* spheres;
    AxisAlignedBoundingBox* aabbs;

    // array of vertices, faces and meshIndexOnlys
    Meshes_SOA meshSOA;

    // mat
    Material* materials;

    // geo
    Geometry* geometries;

    // light idx
    uint* lightIndices;
};