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
#include "nlohmann/json.hpp"

DEFINE_string(ecef_trans_path, "", "the path of ecef transformation matrix");
DEFINE_string(model_trans_path, "", "the path of model transformation matrix");
DEFINE_string(in_model_path, "", "the path of input model");
DEFINE_string(out_geojson_path, "", "the path of output model with geojson format");

using namespace sensemap;
using json = nlohmann::json;

typedef struct JsonPolygon {
    // point: longitue, attitude, altitude
    typedef std::vector<double> point;
    typedef std::vector<point> points;

    std::vector<points> plgs;

} JsonPolygon;

typedef struct JsonFtrNode {
    int id;
    JsonPolygon polygon;
} JsonFtrNode;

void to_json(json &j, const JsonPolygon &node) {
    j["type"] = "Polygon";
    j["coordinates"] = node.plgs;
}

void to_json(json &j, const JsonFtrNode &node) {
    j["type"] = "Feature";
    j["properties"]["id"] = node.id;
    j["geometry"] = node.polygon;
}

typedef struct PolygonMesh {
    typedef std::vector<int> VIDS;
    const int MAX_LINE_LENGTH = 10000;
    std::vector<Eigen::Vector3d> vs;
    std::vector<Eigen::Vector3d> coords_wgs84;
    std::vector<VIDS> fs;

    bool Load(const std::string &filename) {
        vs.clear();
        fs.clear();
        coords_wgs84.clear();

        FILE *file = fopen(filename.c_str(), "r");
        if (!file) {
            std::cerr << "File does not exist! Please check the file path " << filename << std::endl;
            return false;
        }
        char line_buf[MAX_LINE_LENGTH];
        while (fgets(line_buf, MAX_LINE_LENGTH, file))
        {
            if (!strncmp(line_buf, "v ", 2))
            {
                Eigen::Vector3d vertex;
                sscanf(line_buf + 2, "%lf %lf %lf", &vertex[0], &vertex[1], &vertex[2]);
                vs.push_back(vertex);
            }
            if (!strncmp(line_buf, "f ", 2))
            {
                std::string vids_str(line_buf + 1);
                VIDS vids;
                int num = 0, t = 0;
                for (int i = vids_str.length() - 1; i >= 0; -- i) {
                    if (t == 0) {
                        num = 0, t = 1;
                    }
                    if (vids_str[i] >= '0' && vids_str[i] <= '9') {
                        num += t * (vids_str[i] - '0');
                        t *= 10;
                    } else {
                        if (num > 0) vids.emplace_back(num - 1);
                        t = 0;
                    }
                }
                std::reverse(vids.begin(), vids.end());
                fs.emplace_back(vids);
            }
        }
        fclose(file);
        std::cout << "vtxs.size(): " << vs.size() << std::endl;
        std::cout << "faces.size(): " << fs.size() << std::endl;
        return true;
    }

    bool WriteGeoJson(const std::string &filename) {
        
        std::vector<JsonFtrNode> ftr_nodes;
        
        for (int i = 0; i < fs.size(); ++ i) {
            const auto &vids = fs[i];
            JsonPolygon plg;
            plg.plgs.emplace_back(JsonPolygon::points());
            auto &points = plg.plgs[0];

            for (auto vid : vids) {
                const auto &coord = coords_wgs84[vid];

                JsonPolygon::point pt;
                pt.emplace_back(coord[0]);
                pt.emplace_back(coord[1]);
                pt.emplace_back(coord[2]);

                points.emplace_back(pt);
            }
            points.emplace_back(points[0]);

            JsonFtrNode ftr_node;
            ftr_node.id = i;
            ftr_node.polygon = plg;

            ftr_nodes.emplace_back(ftr_node);
        }

        json j;
        j["type"] = "FeatureCollection";
        j["features"] = ftr_nodes;
        j["crs"]["properties"]["name"] = "urn:ogc:def:crs:OGC:1.3:CRS84";
        j["crs"]["type"] = "CRS84";
        std::ofstream ofile(filename);
        ofile << j << std::endl;
        return false;
    }
} PolygonMesh;

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

    std::string help_info = StringPrintf("Usage: \n" \
        "./test_geojson_converter \n" \
        "                      --ecef_trans_path=ecef_trans_file\n" \
        "                      --model_trans_path=model_trans_file(for *trans* obj)\n" \
        "                      --in_model_path=input_model\n" \
        "                      --out_geojson_path=output_model\n");
    google::SetUsageMessage(help_info.c_str());

    if (argc < 4) {
        std::cout << google::ProgramUsage() << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    google::ParseCommandLineFlags(&argc, &argv, false);

    std::cout << "ecef_trans_path: " << FLAGS_ecef_trans_path << std::endl;
    std::cout << "model_trans_path: " << FLAGS_model_trans_path << std::endl;
    std::cout << "in_model_path: " << FLAGS_in_model_path << std::endl;
    std::cout << "out_geojson_path: " << FLAGS_out_geojson_path << std::endl;

    if (FLAGS_in_model_path.empty()) {
        std::cout << StringPrintf("Err! in_model_path empty\n");
		return StateCode::INVALID_INPUT_PARAM;
    }

    if (FLAGS_out_geojson_path.empty()) {
        std::cout << StringPrintf("Err! out_geojson_path empty\n");
		return StateCode::INVALID_INPUT_PARAM;
    }

    if (!ExistsFile(FLAGS_in_model_path.c_str())) {
        std::cout << StringPrintf("Err! File %s open failed!\n", FLAGS_in_model_path.c_str());
		return StateCode::INVALID_INPUT_PARAM;
    }
    const bool has_ecef_trans = ExistsFile(FLAGS_ecef_trans_path.c_str());
    Eigen::RowMatrix3x4d ecef_trans = Eigen::RowMatrix3x4d::Identity();
    if (has_ecef_trans) {
        std::ifstream file(FLAGS_ecef_trans_path.c_str(), std::ifstream::in);
        if (file.is_open()) {
            file >> ecef_trans(0, 0) >> ecef_trans(0, 1) >> ecef_trans(0, 2) >> ecef_trans(0, 3);
            file >> ecef_trans(1, 0) >> ecef_trans(1, 1) >> ecef_trans(1, 2) >> ecef_trans(1, 3);
            file >> ecef_trans(2, 0) >> ecef_trans(2, 1) >> ecef_trans(2, 2) >> ecef_trans(2, 3);
            file.close();
        } else {
            std::cout << StringPrintf("File %s open failed!\n", FLAGS_ecef_trans_path.c_str());
		    return StateCode::INVALID_INPUT_PARAM;
        }
    } else {
        std::cout << StringPrintf("Err! No ecef trans file.\n");
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }
    std::cout << "ecef trans: " << std::endl << ecef_trans << std::endl;

    const bool has_model_trans = ExistsFile(FLAGS_model_trans_path.c_str());
    Eigen::RowMatrix3x4d model_trans = Eigen::RowMatrix3x4d::Identity();
    if (has_model_trans) {
        std::ifstream file(FLAGS_model_trans_path.c_str(), std::ifstream::in);
        if (file.is_open()) {
            file >> model_trans(0, 0) >> model_trans(0, 1) >> model_trans(0, 2) >> model_trans(0, 3);
            file >> model_trans(1, 0) >> model_trans(1, 1) >> model_trans(1, 2) >> model_trans(1, 3);
            file >> model_trans(2, 0) >> model_trans(2, 1) >> model_trans(2, 2) >> model_trans(2, 3);
            file.close();
        } else {
            std::cout << StringPrintf("File %s open failed!\n", FLAGS_model_trans_path.c_str());
		    return StateCode::INVALID_INPUT_PARAM;
        }
    } else {
        std::cout << StringPrintf("Warning! No model trans file.\n");
    }
    std::cout << "model trans: " << std::endl << model_trans << std::endl;

    // load polygon mesh
    {
        PolygonMesh plg_mesh;
        bool res = plg_mesh.Load(FLAGS_in_model_path);
        if (!res) {
            std::cout << StringPrintf("Err! Fail to load model.\n");
		    return StateCode::INVALID_INPUT_PARAM;
        }

        if (plg_mesh.fs.empty()) {
            std::cout << StringPrintf("Warning! Empty faces.\n");
            return 0;
        }
        
        for (int i = 0; i < plg_mesh.vs.size(); ++ i) {
            auto vtx = plg_mesh.vs[i];

            // model_trans.obj -> model.obj
            if (has_model_trans) {
                puts("Inverse trans");
                Eigen::RowMatrix3d rot;
                rot << -1, 0, 0, 0, 0, 1, 0, 1, 0;
                auto rot_model_trans = rot * model_trans;

                Eigen::RowMatrix4d mat_trans44 = Eigen::RowMatrix4d::Identity();
                mat_trans44.block<3, 4>(0, 0) = rot_model_trans;
                vtx = (mat_trans44.inverse() * vtx.homogeneous()).head<3>();
            }

            // model.obj -> ecef -> WGS84
            double latitude, longitude, altitude;
            vtx = ecef_trans * vtx.homogeneous();
            GPSReader::LocationToGps(vtx, &latitude, &longitude, &altitude);

            plg_mesh.coords_wgs84.emplace_back(Eigen::Vector3d(longitude, latitude, altitude));
            // plg_mesh.coords_wgs84.emplace_back(plg_mesh.vs[i]);
        }

        std::cout << "Write model: " << FLAGS_out_geojson_path << std::endl;
        plg_mesh.WriteGeoJson(FLAGS_out_geojson_path);
    } 

    return 0;
}
