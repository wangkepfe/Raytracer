#pragma once 

#include "meshGeometry.cuh"

#include <iostream>
#include <vector>
#include <fstream>

Mesh loadObj(const std::string filename) {

	std::ifstream in(filename.c_str());

	if (!in.good()) {
		//std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good\n";
		system("PAUSE");
		exit(0);
	}

	std::vector<float3> vertices;
	std::vector<uint3> faces;

	char buffer[256], str[255];
	float f1, f2, f3;
	uint ui1, ui2, ui3;

	while (!in.getline(buffer, 255).eof()) {
		
		buffer[255] = '\0';
		sscanf_s(buffer, "%s", str, 255);

		if (buffer[0] == 'v') {
			if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) != 3) {
				//std::cout << "ERROR: vertex not in wanted format\n";
				exit(-1);
			}
			vertices.push_back(make_float3(f1, f2, f3));
		} else if (buffer[0] == 'f') {
			if (sscanf(buffer, "f %u %u %u", &ui1, &ui2, &ui3) != 3) {
				//std::cout << "ERROR: I don't know the format of that face\n";
				exit(-1);
			}
			faces.push_back(make_uint3(ui1, ui2, ui3));
		}
	}

	Mesh mesh;
	mesh.vertexNum = vertices.size();
	mesh.faceNum = faces.size();
	mesh.vertices = vertices.data();
	mesh.faces = faces.data();
	return mesh;

	// std::cout << "obj file loaded: number of faces:" << mesh.faces.size() << " number of vertices:" << mesh.verts.size() << std::endl;
	// std::cout << "obj bounding box: min:(" << mesh.bounding_box[0].x << "," << mesh.bounding_box[0].y << "," << mesh.bounding_box[0].z << ") max:"
	// 	<< mesh.bounding_box[1].x << "," << mesh.bounding_box[1].y << "," << mesh.bounding_box[1].z << ")" << std::endl;
}