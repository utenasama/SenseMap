// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "util/panorama.h"
#include <fstream>
#include "string.h"
#include <iostream>
#include "util/misc.h"
#include "util/logging.h"

using namespace sensemap;

int main(int argc, char* argv[]){

    std::string input_coord = argv[1];
    std::string output_coord = argv[2];
    
    if(argc > 3){
        CHECK(argc == 9);
    }

    int perspective_image_width = (argc>3) ? atoi(argv[3]):600;
    int perspective_image_height = (argc>3) ? atoi(argv[4]):600;
    int perspective_image_count = (argc>3) ? atoi(argv[5]): 6; 
    double fov_w = (argc>3) ? std::stod(argv[6]) : 60;
    int panorama_image_width = (argc>3) ? atoi(argv[7]):5760;
    int panorama_image_height = (argc>3) ? atoi(argv[8]):2880;


    std::shared_ptr<Panorama> panorama = std::make_shared<Panorama>();

    panorama->PerspectiveParamsProcess(perspective_image_width, perspective_image_height, perspective_image_count,
                                       fov_w, panorama_image_width, panorama_image_height);
    

    std::ifstream input_coord_file(input_coord);
    if(!input_coord_file.is_open()){
        std::cout<<"unable to open input coord file "<<input_coord<<std::endl;
        return -1;
    }

    std::ofstream output_coord_file(output_coord);
    if(!output_coord_file.is_open()){
        std::cout<<"unable to open output coord file "<<output_coord<<std::endl;
        return -1;
    }

    std::string line;
    std::string item;
    while(std::getline(input_coord_file, line)){
        
        std::stringstream line_stream(line);

        // sub_image_id
        std::getline(line_stream, item, ' ');
        int sub_image_id = std::stod(item);

        // sub_x
        std::getline(line_stream, item, ' ');
        double sub_x = std::stod(item);

        // sub_y
        std::getline(line_stream, item, ' ');
        double sub_y = std::stod(item);

        double x, y;

        panorama->ConvertPerspectiveCoordToPanorama(sub_image_id,sub_x,sub_y,x,y);

        output_coord_file<<x<<" "<<y<<std::endl;
    }

    input_coord_file.close();
    output_coord_file.close();

    return 0;
}