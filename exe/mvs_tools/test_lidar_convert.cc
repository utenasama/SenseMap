//Copyright (c) 2021, SenseTime Group.
//All rights reserved.
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "../system_io.h"

#include "base/common.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/point2d.h"
#include "base/pose.h"
#include "base/camera_models.h"
#include "base/undistortion.h"
#include "base/reconstruction_manager.h"
#include "base/projection.h"

#include "lidar/voxel_map.h"

#include "graph/correspondence_graph.h"
#include "controllers/incremental_mapper_controller.h"

#include "lidar/pcd.h"
#include "lidar/lidar_sweep.h"

#include "util/ply.h"
#include "util/proc.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/exception_handler.h"

#include "base/version.h"

using namespace sensemap;

std::string workspace_path;
static std::unordered_map<std::string, image_t> image_name_map;

struct RigNamesNew{
    std::string name;
    double time;
    // std::vector<float> t;
    // std::vector<float> q;
    Eigen::Vector3d t;
    Eigen::Vector4d q;
    std::string pcd;
    std::string img;
    std::string img2;
    std::string img3;
    void Init(std::string name1, double time1, Eigen::Vector3d t1,  Eigen::Vector4d q1, 
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

void ReadRigList(std::vector<RigNamesNew>& list, const std::string file_path){
    std::vector<sensemap::PlyPoint > points;
    if (!ExistsFile(file_path)){
        std::cout << "file is empty, " << file_path << std::endl;
        return;
    }
    std::ifstream ifs;
    //打开文件
    ifs.open(file_path.c_str(), std::ios::in);
    //定义一个字符串
    std::string str;
    //从文件中读取数据
    while(getline(ifs, str))
    {
        // std::cout << str << std::endl;
        std::string info_str = str;
        std::vector<std::string> strparams = StringSplit(info_str, ",");

        double time1 = std::atof(strparams.at(0).c_str());
        Eigen::Vector3d t1(std::atof(strparams.at(1).c_str()),
                           std::atof(strparams.at(2).c_str()),
                           std::atof(strparams.at(3).c_str()));

        Eigen::Vector4d q1(std::atof(strparams.at(7).c_str()),
                           std::atof(strparams.at(4).c_str()),
                           std::atof(strparams.at(5).c_str()),
                           std::atof(strparams.at(6).c_str()));

        std::string pcd;
        getline(ifs, pcd);
        std::string img1,img2,img3;
        getline(ifs, img1);
        getline(ifs, img2);
        getline(ifs, img3);
        
        std::string name = GetPathBaseName(pcd);
        name.substr(0, name.length() - 4);
        RigNamesNew rig_name;
        // rig_name.Init("points/" + GetPathBaseName(pcd), "cam0/" + GetPathBaseName(img1));
        rig_name.Init(name, time1, t1, q1,"points/" + GetPathBaseName(pcd), 
                      "camera/front/" + GetPathBaseName(img1), 
                      "camera/left/" + GetPathBaseName(img2), 
                      "camera/right/" + GetPathBaseName(img3));
        // std::cout << root_path << ", " << pcd << std::endl;

        list.push_back(rig_name);

        const auto R = QuaternionToRotationMatrix(q1);
        sensemap::PlyPoint pnt;
        pnt.x = t1.x();
        pnt.y = t1.y();
        pnt.z = t1.z();
        // pnt.nx = R(0,2);
        // pnt.ny = R(1,2);
        // pnt.nz = R(2,2);
        pnt.nx = R(0,0);
        pnt.ny = R(1,0);
        pnt.nz = R(2,0);
        points.push_back(pnt);
    }
    std::sort(list.begin(),list.end(),[](const RigNamesNew& a ,const RigNamesNew& b){
		return a.time < b.time;
	});
    std::cout << "read in " << list.size() << " frame." << std::endl;
    WriteTextPlyPoints(file_path + "_t.ply", points, true, false);
    return;
}

int main(int argc, char** argv) {

    PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");
    PrintHeading(std::string("Version: ") + __VERSION__);

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());


    workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string lidar_prior_pose_file = param.GetArgument("lidar_prior_pose_file", "");

    std::string sparse_path = JoinPaths(workspace_path, "0", DENSE_DIR, SPARSE_DIR);
    std::cout << "sparse_path: " << sparse_path << std::endl;
    std::shared_ptr<Reconstruction> rect;
    rect.reset(new Reconstruction());
    rect->ReadReconstruction(sparse_path);
    std::cout << "ReadReconstruction " << std::endl;
    std::vector<RigNamesNew> rig_list;
    ReadRigList(rig_list, lidar_prior_pose_file);
    std::unordered_map<std::string, size_t> rig_map;
    for (size_t i = 0; i < rig_list.size(); i++){
        rig_map[rig_list.at(i).name] = i; 
    }
    std::cout << "rig: " << rig_list.at(0).name << std::endl;
    std::cout << "lidar: " << rect->LidarSweeps().at(0).Name() << std::endl;

    int num = 0;
    for (const auto& lidar : rect->LidarSweeps()){
        const auto base_name = GetPathBaseName(lidar.second.Name());
        if (rig_map.find(base_name) != rig_map.end()){
            const auto rig_id = rig_map[base_name];
            // Eigen::Matrix3x4d trans = ComposeProjectionMatrix(rig_list[rig_id].q, rig_list[rig_id].t);
            // Eigen::Matrix3x4d inv_trans = InvertProjectionMatrix(trans);
            Eigen::Vector4d qvec;
            Eigen::Vector3d tvec;
            InvertPose(rig_list[rig_id].q, rig_list[rig_id].t, &qvec, &tvec);

            auto& lidar_t = rect->LidarSweep(lidar.first);
            lidar_t.SetQvec(qvec);
            lidar_t.SetTvec(tvec);
            num++;
        }
    }
    std::cout << "convert lidar " << num << std::endl;
    CreateDirIfNotExists(sparse_path + "_prior");
    rect->WriteReconstruction(sparse_path + "_prior");

    return 0;


}