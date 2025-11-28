//Copyright (c) 2021, SenseTime Group.
//All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "las_include/laswriter.hpp"
#include "util/ply.h"
#include "util/misc.h"
#include "util/exception_handler.h"
#include "base/version.h"

using namespace sensemap;

int main(int argc, char** argv) {

	PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

    LASwriteOpener laswriteopener;

    if (argc < 3) {
        std::cout << "Please enter ./test_ply2las input.ply output.las" << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    laswriteopener.set_file_name(argv[2]);

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

    // check output
    if (!laswriteopener.active())
    {
        fprintf(stderr,"ERROR: no output specified\n");
		return StateCode::INVALID_INPUT_PARAM;
    }

    // init header
    LASheader lasheader;
    lasheader.point_data_format = 2;
    lasheader.point_data_record_length = 26;

    // init point
    LASpoint laspoint;
    laspoint.init(&lasheader, lasheader.point_data_format, lasheader.point_data_record_length, 0);

    // open laswriter
    LASwriter* laswriter = laswriteopener.open(&lasheader);
    if (laswriter == 0)
    {
        fprintf(stderr, "ERROR: could not open laswriter\n");
		return StateCode::INVALID_INPUT_PARAM;
    }

    // write points
    if (!dreal) {
        for (const auto& point : ply_points) {
            laspoint.set_x((F64)point.x);
            laspoint.set_y((F64)point.y);
            laspoint.set_z((F64)point.z);
            laspoint.set_R((U16)point.r);
            laspoint.set_G((U16)point.g);
            laspoint.set_B((U16)point.b);

            laswriter->write_point(&laspoint);
            laswriter->update_inventory(&laspoint);
        }
    } else {
        for (int i=0; i<Xs.size(); i++) {
            laspoint.set_x((F64)Xs.at(i));
            laspoint.set_y((F64)Ys.at(i));
            laspoint.set_z((F64)Zs.at(i));
            laspoint.set_R((U16)rs.at(i));
            laspoint.set_G((U16)gs.at(i));
            laspoint.set_B((U16)bs.at(i));

            laswriter->write_point(&laspoint);
            laswriter->update_inventory(&laspoint);
        }
    }
    // update the header
    laswriter->update_header(&lasheader, TRUE);

    // close the writer
    I64 total_bytes = laswriter->close();
    delete laswriter;
    std::cout << "Convert Done!" << std::endl;

    return 0;
}