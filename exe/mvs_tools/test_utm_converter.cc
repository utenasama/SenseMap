//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <gflags/gflags.h>

#include "util/obj.h"
#include "util/ply.h"
#include "util/misc.h"
#include "util/gps_reader.h"
#include "base/common.h"
#include "util/exception_handler.h"

#include "base/version.h"

DEFINE_string(model_type, "model", "model type: model or pointcloud");
DEFINE_string(trans_path, "", "the path of transformation matrix");
DEFINE_string(in_model_path, "", "the path of input model");
DEFINE_string(out_model_path, "", "the path of transformed input model");

using namespace sensemap;

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

    std::string help_info = StringPrintf("Usage: \n" \
        "./test_utm_converter --model_type=model\n" \
        "                      --trans_path=trans_file\n" \
        "                      --in_model_path=input_model\n" \
        "                      --out_model_path=output_model\n");
    google::SetUsageMessage(help_info.c_str());

    google::ParseCommandLineFlags(&argc, &argv, false);

    if (argc != 5) {
        std::cout << google::ProgramUsage() << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    std::cout << "model_type: " << FLAGS_model_type << std::endl;
    std::cout << "trans_path: " << FLAGS_trans_path << std::endl;
    std::cout << "in_model_path: " << FLAGS_in_model_path << std::endl;
    std::cout << "out_model_path: " << FLAGS_out_model_path << std::endl;

    bool has_trans = ExistsFile(FLAGS_trans_path.c_str());

    Eigen::RowMatrix3x4d trans = Eigen::RowMatrix3x4d::Identity();
    std::ifstream file(FLAGS_trans_path.c_str(), std::ifstream::in);
    if (file.is_open()) {
        file >> trans(0, 0) >> trans(0, 1) >> trans(0, 2) >> trans(0, 3);
        file >> trans(1, 0) >> trans(1, 1) >> trans(1, 2) >> trans(1, 3);
        file >> trans(2, 0) >> trans(2, 1) >> trans(2, 2) >> trans(2, 3);
    } else {
        has_trans = false;
        std::cout << StringPrintf("File %s open failed!\n", FLAGS_trans_path.c_str());
    }
    file.close();

    std::cout << "trans: " << std::endl << trans << std::endl;

    if (FLAGS_model_type.compare("model") == 0 || 
        FLAGS_model_type.compare("model-sem") == 0) {
        std::cout << "in_model_path: " << FLAGS_in_model_path << std::endl;
        if (!ExistsFile(FLAGS_in_model_path)) {
            std::cout << StringPrintf("Error! Input Model file does not "
                                    "exists!\n");
		    return StateCode::INVALID_INPUT_PARAM;
        }

        TriangleMesh obj_model;
        if (FLAGS_model_type.compare("model-sem") == 0) {
            ReadTriangleMeshObj(FLAGS_in_model_path, obj_model, true, true);
        } else {
            ReadTriangleMeshObj(FLAGS_in_model_path, obj_model, true);
        }
        if (obj_model.faces_.empty()) {
            std::cout << StringPrintf("Warning! Input Model file is empty!\n");
		    return StateCode::INVALID_INPUT_PARAM;
        }
        for (size_t i = 0; i < obj_model.vertices_.size(); ++i) {
            if (!has_trans) {
                continue;
            }
            Eigen::Vector3d vtx = trans * obj_model.vertices_[i].homogeneous();

            double latitude, longitude, altitude;
            GPSReader::LocationToGps(vtx, &latitude, &longitude, &altitude);
            Eigen::Vector3d utm = GPSReader::gpsToUTM(latitude, longitude);
            vtx[0] = utm[0]; vtx[1] = utm[1]; vtx[2] = altitude;
            obj_model.vertices_[i] = vtx;

            if (!obj_model.vertex_normals_.empty()) {
                Eigen::Vector3d nvtx = obj_model.vertex_normals_[i];
                obj_model.vertex_normals_[i] = (trans.block<3, 3>(0, 0) * nvtx).normalized();
            }
        }
        std::cout << "out_model_path: " << FLAGS_out_model_path << std::endl;
        if (FLAGS_model_type.compare("model-sem") == 0) {
            WriteTriangleMeshObj(FLAGS_out_model_path, obj_model, true, true);
        } else {
            WriteTriangleMeshObj(FLAGS_out_model_path, obj_model);
        }
    } else if (FLAGS_model_type.compare("pointcloud") == 0) {
        
        std::cout << "in_points_path: " << FLAGS_in_model_path << std::endl;
        if (!ExistsFile(FLAGS_in_model_path)) {
            std::cout << StringPrintf("Error! Input Fused file does not "
                                    "exists!\n");
		    return StateCode::INVALID_INPUT_PARAM;
        }
        std::vector<PlyPoint> ply_points = ReadPly(FLAGS_in_model_path);
        size_t num_point = ply_points.size();
        std::vector<double> Xs(num_point);
        std::vector<double> Ys(num_point);
        std::vector<double> Zs(num_point);
        std::vector<uint8_t> rs(num_point);
        std::vector<uint8_t> gs(num_point);
        std::vector<uint8_t> bs(num_point);
        std::vector<double> nXs, nYs, nZs;
        for (size_t i = 0; i < num_point; ++i) {
            auto& point = ply_points.at(i);
            Eigen::Vector3d X(point.x, point.y, point.z);
            if (has_trans) {
                X = trans * X.homogeneous();
                double latitude, longitude, altitude;
                GPSReader::LocationToGps(X, &latitude, &longitude, &altitude);
                Eigen::Vector3d utm = GPSReader::gpsToUTM(latitude, longitude);
                Xs[i] = utm[0]; Ys[i] = utm[1]; Zs[i] = altitude;
            } else {
                Xs[i] = X[0]; Ys[i] = X[1]; Zs[i] = X[2];
            }
            rs[i] = point.r;
            gs[i] = point.g;
            bs[i] = point.b;
        }

        std::cout << "out_points_path: " << FLAGS_out_model_path << std::endl;
        // WriteBinaryPlyPoints(FLAGS_out_model_path, ply_points);
        WriteBinaryPlyPoints(FLAGS_out_model_path, Xs, Ys, Zs, nXs, nYs, nZs, rs, gs, bs, false, true);
    }

    return 0;
}
