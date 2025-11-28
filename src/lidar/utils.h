//Copyright (c) 2023, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_LIDAR_UTILS_H_
#define SENSEMAP_LIDAR_UTILS_H_

#include <fstream>

#include "util/types.h"
#include "util/misc.h"
#include "util/ply.h"

#include "base/pose.h"

namespace sensemap {

namespace {
const static double kNaN = std::numeric_limits<double>::quiet_NaN();
}

struct RigNames{
    std::string name;
    uint64_t time;
    Eigen::Vector3d t = Eigen::Vector3d(kNaN, kNaN, kNaN);
    Eigen::Vector4d q = Eigen::Vector4d(kNaN, kNaN, kNaN, kNaN);
    std::string pcd;
    std::string img;
    std::string img2;
    std::string img3;

    void Init(std::string p, std::string i, std::string i2 = "", std::string i3 = ""){
        pcd = p; 
        img = i;
        img2 = i2;
        img3 = i3;
    }

    void Init(std::string name1, uint64_t time1, Eigen::Vector3d t1,  Eigen::Vector4d q1, 
              std::string p, std::string i, std::string i2 = "", std::string i3 = ""){
        name = name1;
        time = time1;
        t = t1;
        q = q1;
        pcd = p; 
        img = i;
        img2 = i2;
        img3 = i3;
    }
};

void ReadRigList(const std::string file_path, std::vector<RigNames>& list);

}

#endif
