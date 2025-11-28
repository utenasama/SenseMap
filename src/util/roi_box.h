//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_UTIL_ROI_BOX_H_
#define SENSEMAP_UTIL_ROI_BOX_H_

#include <fstream>
#include <iostream>
#include <float.h>
#include <sys/stat.h>
#include <boost/filesystem/path.hpp>

#include <Eigen/Core>

#include "util/obj.h"

#define EPSILON std::numeric_limits<float>::epsilon()
#define MAX_INT std::numeric_limits<int>::max()

namespace sensemap {

struct Box{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    float x_min = -1;
    float y_min = -1;
    float z_min = -1;
    float x_max = -1;
    float y_max = -1;
    float z_max = -1;

    Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();

    float x_box_min = -1;
    float y_box_min = -1;
    float z_box_min = -1;
    float x_box_max = -1;
    float y_box_max = -1;
    float z_box_max = -1;

    float border_width = -1;
    float border_factor = -1;

    Box operator+=(const Box &other){
        if (std::abs(this->x_min - this->x_max) <  EPSILON || 
            std::abs(this->y_min - this->y_max) <  EPSILON ){
            this->x_min = other.x_min;
            this->y_min = other.y_min;
            this->z_min = other.z_min;
            this->x_max = other.x_max;
            this->y_max = other.y_max;
            this->z_max = other.z_max;

            this->x_box_min = other.x_box_min;
            this->y_box_min = other.y_box_min;
            this->z_box_min = other.z_box_min;
            this->x_box_max = other.x_box_max;
            this->y_box_max = other.y_box_max;
            this->z_box_max = other.z_box_max;
        } else{
            this->x_min = std::min(this->x_min, other.x_min);
            this->y_min = std::min(this->y_min, other.y_min);
            this->z_min = std::min(this->z_min, other.z_min);
            this->x_max = std::max(this->x_max, other.x_max);
            this->y_max = std::max(this->y_max, other.y_max);
            this->z_max = std::max(this->z_max, other.z_max);

            this->x_box_min = std::min(this->x_box_min, other.x_box_min);
            this->y_box_min = std::min(this->y_box_min, other.y_box_min);
            this->z_box_min = std::min(this->z_box_min, other.z_box_min);
            this->x_box_max = std::max(this->x_box_max, other.x_box_max);
            this->y_box_max = std::max(this->y_box_max, other.y_box_max);
            this->z_box_max = std::max(this->z_box_max, other.z_box_max);
        }
        return *this;
    }

    void SetBoundary(){
        if (std::abs(x_min - x_max) < std::numeric_limits<float>::epsilon() || 
            std::abs(y_min - y_max) < std::numeric_limits<float>::epsilon()){
            std::cout << "Error: invalid box boundary" << std::endl;
            return;
        }
        
        if (border_width < 0 && border_factor < 0){
            border_width = std::min((x_max -x_min), (y_max -y_min)) / 100.0f;
        } else if (border_factor > 0){
            float border_width_1 = std::min((x_max -x_min), (y_max -y_min)) * border_factor;
            border_width = std::max(border_width, border_width_1);
        }

        x_box_min = x_min - border_width;
        y_box_min = y_min - border_width;
        x_box_max = x_max + border_width;
        y_box_max = y_max + border_width;
    }

    void SetBoundary(float border_width_c, float border_factor_c){
        border_width = border_width_c;
        border_factor = border_factor_c;
        SetBoundary();
    }


    void ResetBoundary(float scale_factor){
        if (border_width < 0 && border_factor < 0){
            border_factor = 0.01;
        }
        SetBoundary(border_width * scale_factor, border_factor * scale_factor);
    }


    struct MeshBox ToMeshBox(){
        struct MeshBox meshbox;
        meshbox.x_min = x_min;
        meshbox.x_max = x_max;
        meshbox.y_min = y_min;
        meshbox.y_max = y_max;
        meshbox.rot = rot;
        meshbox.border_factor = border_factor;
        meshbox.border_width = border_width;

        meshbox.SetBoundary();
        // meshbox.Print();
        return meshbox;
    };
};

static void WriteBoundBoxText(const std::string& path, struct Box& box){
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# x_min y_min x_max y_max" << std::endl;

    std::ostringstream line;
    line << box.x_min << " ";
    line << box.y_min << " ";
    line << box.x_max << " ";
    line << box.y_max << " ";
    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);
    file << line_string << std::endl << std::endl;

    file << "# transformation 3x3" << std::endl;

    std::ostringstream transformation;
    transformation 
        << box.rot(0,0) << " " << box.rot(0,1) << " " << box.rot(0,2) << " "
        << box.rot(1,0) << " " << box.rot(1,1) << " " << box.rot(1,2) << " "
        << box.rot(2,0) << " " << box.rot(2,1) << " " << box.rot(2,2) << " ";
    std::string transformation_string = transformation.str();
    transformation_string = transformation_string.substr(0, transformation_string.size() - 1);
    file << transformation_string << std::endl;

    file.close();
}

static void ReadBoundBoxText(const std::string& path, struct Box& box){
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    // x_min, y_min, x_max, y_max
    std::getline(file, line);
    StringTrim(&line);
    while (line.empty() || line[0] == '#') {
        std::getline(file, line);
        StringTrim(&line);
    }
    std::stringstream line_stream(line);
    std::getline(line_stream, item, ' ');
    box.x_min = std::stof(item);
    std::getline(line_stream, item, ' ');
    box.y_min = std::stof(item);
    std::getline(line_stream, item, ' ');
    box.x_max = std::stof(item);
    std::getline(line_stream, item, ' ');
    box.y_max = std::stof(item);

    box.z_min =  -FLT_MAX;
    box.z_max = FLT_MAX;

    // rotation matrix
    std::getline(file, line);
    StringTrim(&line);
    while (line.empty() || line[0] == '#') {
        std::getline(file, line);
        StringTrim(&line);
    }
    std::stringstream rot_line_stream(line);
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            std::getline(rot_line_stream, item, ' ');
            box.rot(i,j) = std::stof(item);
        }
    }
}

} // namespace

#endif