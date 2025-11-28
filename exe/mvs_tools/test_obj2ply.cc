//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <ctime>

#include "util/obj.h"
#include "util/ply.h"
#include "util/misc.h"
#include "util/tinyxml.h"
#include "util/roi_box.h"
#include "base/common.h"
#include "util/exception_handler.h"

#include "base/version.h"
#include "../Configurator_yaml.h"


using namespace std;
using namespace sensemap;

#define EPSILON std::numeric_limits<float>::epsilon()
#define MAX_INT std::numeric_limits<int>::max()

std::string configuration_file_path;

void TransObj2Ply(TriangleMesh &obj_mesh, PlyMesh &ply_mesh){
	bool has_rgb = (obj_mesh.vertex_colors_.size() != 0);
	bool has_normal = (obj_mesh.vertex_normals_.size() != 0);
	ply_mesh.vertices.resize(obj_mesh.vertices_.size());
	ply_mesh.faces.resize(obj_mesh.faces_.size());

	for (int i=0; i<obj_mesh.vertices_.size(); i++){
		ply_mesh.vertices.at(i).x = obj_mesh.vertices_.at(i)(0);
		ply_mesh.vertices.at(i).y = obj_mesh.vertices_.at(i)(1);
		ply_mesh.vertices.at(i).z = obj_mesh.vertices_.at(i)(2);
		if (has_rgb) {
			ply_mesh.vertices.at(i).r = obj_mesh.vertex_colors_.at(i)(0) * 255;
			ply_mesh.vertices.at(i).g = obj_mesh.vertex_colors_.at(i)(1) * 255;
			ply_mesh.vertices.at(i).b = obj_mesh.vertex_colors_.at(i)(2) * 255;
		}
		if (has_normal) {
			ply_mesh.vertices.at(i).nx = obj_mesh.vertex_normals_.at(i)(0);
			ply_mesh.vertices.at(i).ny = obj_mesh.vertex_normals_.at(i)(1);
			ply_mesh.vertices.at(i).nz = obj_mesh.vertex_normals_.at(i)(2);
		}
	}

	for (int j=0; j<obj_mesh.faces_.size(); j++){
		ply_mesh.faces.at(j).vertex_idx1 = obj_mesh.faces_[j](0);
		ply_mesh.faces.at(j).vertex_idx2 = obj_mesh.faces_[j](1);
		ply_mesh.faces.at(j).vertex_idx3 = obj_mesh.faces_[j](2);
	}
}

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
	
	if (argc != 3 && argc !=4){
		std::cout << "Error! Input: in-model-path out-model-path (write_brinary)\n"
				  << "eg: /test_obj2ply ./workspace/0/dense/model.obj ./workspace/0/dense/model.ply (1)" << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
	} 

	bool write_brinary = true;
	const std::string in_model_path = std::string(argv[1]);
	std::string out_model_path = std::string(argv[2]);
	if (argc == 4){
		write_brinary = std::atoi(argv[3]);
	}
	std::cout << "write_brinary: " << write_brinary << std::endl;

	if (out_model_path.substr(out_model_path.length() - 4) == ".ply"){
		out_model_path = out_model_path.substr(0, out_model_path.length() - 4);
	}

	std::cout << "in_model_path: " << in_model_path << std::endl;

	if (out_model_path.empty()) {
		std::cerr << "Error! Output model path is empty!" << std::endl;
		return StateCode::INVALID_INPUT_PARAM;
	}

    // read obj file
    TriangleMesh obj_model;
    if (!ReadTriangleMeshObj(in_model_path, obj_model, true)) {
		return StateCode::INVALID_INPUT_PARAM;
	}

	// transform and save dae file
	out_model_path = out_model_path+".ply";
	PlyMesh ply_model;
	TransObj2Ply(obj_model, ply_model);
	bool save_normal = obj_model.vertex_normals_.size() != 0;
	bool save_rgb = obj_model.vertex_colors_.size() != 0;
	std::cout << save_rgb << std::endl;
	if (write_brinary){
		WriteBinaryPlyMesh(out_model_path, ply_model, save_normal, save_rgb);
	} else {
		WriteTextPlyMesh(out_model_path, ply_model, save_normal, save_rgb);
	}
	std::cout << "out_dae_model_path: " << out_model_path << std::endl;

	// // xmlFile.ReadDae();

	return 0;
}