//Copyright (c) 2021, SenseTime Group.
//All rights reserved.
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "util/ply.h"
#include "util/misc.h"
#include "util/exception_handler.h"
#include "base/version.h"

using namespace sensemap;

int main(int argc, char** argv) {

	PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
    
    if (argc < 3) {
        std::cout << "Please enter ./test_ply2pnts input.ply output.pnts" << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    FILE* fp = fopen(argv[2], "wb");
    if (fp == nullptr) {
        std::cout << "Open " << argv[2] << " failed." << std::endl;
		return StateCode::INVALID_INPUT_PARAM;
    }

    //read ply
    std::string ply_path(argv[1]);
    bool dreal = ReadPointRealType(ply_path);
    
    std::vector<sensemap::PlyPoint> ply_points;
    std::vector<double> Xs;
    std::vector<double> Ys;
    std::vector<double> Zs;
    std::vector<double> nXs;
    std::vector<double> nYs;
    std::vector<double> nZs;
    std::vector<uint8_t> rs;
    std::vector<uint8_t> gs;
    std::vector<uint8_t> bs;
    if (!dreal) {
        ply_points = sensemap::ReadPly(ply_path);
        std::cout << "ply points size: " << ply_points.size() << std::endl;
        if (ply_points.empty()) {
            std::cout << "no point in ply file!" << std::endl;
            return StateCode::INVALID_INPUT_PARAM;
        }
    } else {
        sensemap::ReadPly(ply_path, Xs, Ys, Zs, nXs, nYs, nZs, rs, gs, bs);
        std::cout << "ply points size: " << Xs.size() << std::endl;
        if (Xs.empty()) {
            std::cout << "no point in ply file!" << std::endl;
            return StateCode::INVALID_INPUT_PARAM;
        }
    }

    //calculate feature table binary byte length
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<uint8_t> RGBs;

    if (!dreal) {
        for (const auto& point : ply_points) {
            positions.emplace_back(point.x);
            positions.emplace_back(point.y);
            positions.emplace_back(point.z);
            normals.emplace_back(point.nx);
            normals.emplace_back(point.ny);
            normals.emplace_back(point.nz);
            RGBs.emplace_back(point.r);
            RGBs.emplace_back(point.g);
            RGBs.emplace_back(point.b);
        }
    } else {
        for (int i=0; i<Xs.size(); i++) {
            positions.emplace_back((float)Xs.at(i));
            positions.emplace_back((float)Ys.at(i));
            positions.emplace_back((float)Zs.at(i));
            normals.emplace_back((float)nXs.at(i));
            normals.emplace_back((float)nYs.at(i));
            normals.emplace_back((float)nZs.at(i));
            RGBs.emplace_back(rs.at(i));
            RGBs.emplace_back(gs.at(i));
            RGBs.emplace_back(bs.at(i));
        }
    }

    //form featureTableJson and featureTableBinary
    int positions_length = positions.size();
    int normals_length = normals.size();
    int RGB_length = RGBs.size();

    //align 8, add zeros to the end of binary
    unsigned int binary_byte_length = positions_length * 4 + RGB_length;
    // unsigned int binary_byte_length = positions_length * 4 + normals_length * 4 + RGB_length;
    while (binary_byte_length % 8 != 0) {
        binary_byte_length++;
        RGBs.emplace_back((uint8_t)0);
        RGB_length++;
    }

    std::string feature_table_Json;
    feature_table_Json += "{\"POINTS_LENGTH\":";
    feature_table_Json += std::to_string(positions_length / 3);
    feature_table_Json += ",\"POSITION\":{\"byteOffset\":0";
    // feature_table_Json += ",\"POSITION\":{\"byteOffset\":0},\"NORMAL\":{\"byteOffset\":";
    // feature_table_Json += std::to_string(positions_length * 4);
    feature_table_Json += "},\"RGB\":{\"byteOffset\":";
    feature_table_Json += std::to_string(positions_length * 4);
    // feature_table_Json += std::to_string(positions_length * 4 + normals_length * 4);
    feature_table_Json += "}}";

    //align 8, add spaces to the end of json
    while (feature_table_Json.length() % 8 != 0) {
        feature_table_Json += " ";
    }

    std::cout << feature_table_Json << std::endl;
    std::cout << "json size: " << feature_table_Json.length() << " binary byte length: " << binary_byte_length << std::endl;

    unsigned int whole_file_length = 4 * sizeof(char) + 6 * sizeof(int) + feature_table_Json.length() + binary_byte_length;

    char pnts[] = "pnts";
    unsigned int header[] = { 1, whole_file_length, (unsigned int)feature_table_Json.length(), binary_byte_length, 0, 0 };

    fwrite(pnts, sizeof(char), 4, fp);
    fwrite(header, sizeof(unsigned int), 6, fp);
    fwrite(feature_table_Json.c_str(), sizeof(unsigned char), feature_table_Json.length(), fp);
    fwrite(positions.data(), sizeof(float), positions_length, fp);
    // fwrite(normals.data(), sizeof(float), normals_length, fp);
    fwrite(RGBs.data(), sizeof(char), RGB_length, fp);
    fclose(fp);

    return 0;
}