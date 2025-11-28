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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

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

static std::unordered_map<std::string, image_t> image_name_map;
std::string root_path = "";
int img_width=4032, img_height= 3040;

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

void ColorMap(const float intensity, uint8_t &r, uint8_t &g, uint8_t &b){
    if(intensity <= 0.25) {
        r = 0;
        g = 0;
        b = (intensity) * 4 * 255;
        return;
    }

    if(intensity <= 0.5) {
        r = 0;
        g = (intensity - 0.25) * 4 * 255;
        b = 255;
        return;
    }

    if(intensity <= 0.75) {
        r = 0;
        g = 255;
        b = (0.75 - intensity) * 4 * 255;
        return;
    }

    if(intensity <= 1.0) {
        r = (intensity - 0.75) * 4 * 255;
        g = (1.0 - intensity) * 4 * 255;
        b = 0;
        return;
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
        rig_name.Init(name, time1, t1, q1,
                      JoinPaths(root_path, "points/" + GetPathBaseName(pcd)), 
                      JoinPaths(root_path, "camera/front/" + GetPathBaseName(img1)), 
                      JoinPaths(root_path, "camera/left/" + GetPathBaseName(img2)), 
                      JoinPaths(root_path, "camera/right/" + GetPathBaseName(img3)));
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

void GetCameraParam(std::vector<Eigen::Matrix3d>& k_vec,
    std::vector<Eigen::Matrix<double, 1, 4>>& dist_vec){
    
    Eigen::Matrix3d k_0,k_1,k_2;
    Eigen::Matrix<double, 1, 4> dist_0, dist_1, dist_2;
    k_0<< 1181.43, 0., 1993.77,
                0.,1182.05, 1484.64,
                0.,0.,1.;
    k_1<< 1177.28, 0., 2027.71,
                0.,1180.46, 1498.09,
                0.,0.,1.;
    k_2<< 1184.45, 0., 1981.81,
                0., 1190.91, 1547.79,
                0.,0.,1.;
    k_vec.push_back(k_0);
    k_vec.push_back(k_1);
    k_vec.push_back(k_2);

    dist_0<<-0.009953, -0.000346, -0.003221, 0.000838;
    dist_1<<-0.015083, 0.0019, -0.002356, 0.000106;
    dist_2<<-0.015354, 0.005416, -0.00497, 0.000623;
    dist_vec.push_back(dist_0);
    dist_vec.push_back(dist_1);
    dist_vec.push_back(dist_2);
}


void Distortion(std::vector<double> &extra_params, const double u, const double v, double* du, double* dv) {
    const double k1 = extra_params[0];
    const double k2 = extra_params[1];
    const double k3 = extra_params[2];
    const double k4 = extra_params[3];
    // std::cout<<"k1: "<<k1<<" k2: "<<k2<<" k3: "<<k3<<" k4: "<<k4<<std::endl;
    const double r = ceres::sqrt(u * u + v * v);

    // std::cout<<"r: "<<r<<std::endl;
    if (r > double(std::numeric_limits<double>::epsilon())) {
        const double theta = ceres::atan(r);
        // std::cout<<"theta: "<<theta<<std::endl;
        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta8 = theta4 * theta4;
        const double thetad =
            theta * (double(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        *du = u * thetad / r - u;
        *dv = v * thetad / r - v;
    } else {
        *du = double(0);
        *dv = double(0);
    }
}

// i = 0 front camera, i = 1 left camera, i = 2 right camera;
Eigen::Matrix3x4d GetLidarTrans(int i){
    std::vector<Eigen::Matrix4d> cam_to_front_vec;
    Eigen::Matrix4d cam0_to_front = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d cam_left_to_front = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d cam_right_to_front = Eigen::Matrix4d::Identity();
    cam_left_to_front<<   0.00260646, -0.000730117,    -0.999996 ,  -0.0518803,
                        -0.00404651 ,    0.999992,  -0.00074066,  0.000829048,
                        0.999988  , 0.00404842   ,0.00260348 ,  -0.0432069,
                        0,0,0,1;
    cam_right_to_front<<0.0168137, -0.00722524,    0.999833,   0.0513387,
                        0.00178167,    0.999973,  0.00719629, 0.000443776,
                        -0.999857,  0.00166038,   0.0168261,  -0.0402498,
                        0,0,0,1;
    cam_to_front_vec.push_back(cam0_to_front);
    cam_to_front_vec.push_back(cam_left_to_front);
    cam_to_front_vec.push_back(cam_right_to_front);  

    //lidar to cam_front
    double lidar_rx  = 2.596775198735603012e-03;
    double lidar_ry = -8.659574932656155175e-03;
    double lidar_rz = -5.742026523323420090e-03;
    double lidar_px = -6.738956845405014162e-02;
    double lidar_py = 2.789874885480689553e-03;
    double lidar_pz = 2.303947065661732588e-02;

    Eigen::AngleAxisf rollAngle = 
                Eigen::AngleAxisf(lidar_rx, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle = 
                Eigen::AngleAxisf(lidar_ry , Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle = 
                Eigen::AngleAxisf(lidar_rz, Eigen::Vector3f::UnitZ());
    Eigen::Matrix3f R = yawAngle.matrix() * pitchAngle.matrix() * rollAngle.matrix();
    Eigen::Vector3d tvec_lidar(lidar_px,lidar_py,lidar_pz);

    //lidar to img
    Eigen::Matrix4d T_lidar_to_ref = Eigen::Matrix4d::Identity();
    T_lidar_to_ref.block<3, 3>(0, 0) = R.cast<double>();
    T_lidar_to_ref.block<3, 1>(0, 3) = tvec_lidar;

    Eigen::Matrix4d World2CV;
    World2CV<<0, -1, 0, 0,0, 0, -1, 0,1, 0, 0, 0,0, 0, 0, 1;

    Eigen::Matrix4d cam_to_front = cam_to_front_vec[i];

    Eigen::Matrix4d T_lidar_to_cam =cam_to_front.inverse() * World2CV * T_lidar_to_ref;

    return T_lidar_to_cam.topRows(3);
}


void GetColor(std::vector<sensemap::PlyPoint>& pc, 
    const RigNamesNew& rig_list){
    
    std::vector<sensemap::PlyPoint> color_pc;
    color_pc.reserve(pc.size());

    std::vector<Eigen::Matrix3d> k_vec;
    std::vector<Eigen::Matrix<double, 1, 4>> dist_vec;

    GetCameraParam(k_vec, dist_vec);

    std::vector<cv::Mat> images;
    images.push_back(cv::imread(rig_list.img));
    images.push_back(cv::imread(rig_list.img2));
    images.push_back(cv::imread(rig_list.img3));

    for (int i = 0; i < pc.size(); i++){
        Eigen::Vector3d point3d_l;
        point3d_l << pc[i].x, pc[i].y, pc[i].z;

        for (int k = 0; k < 3; k++){
            const auto T = GetLidarTrans(k);
            Eigen::Vector3d point3d_w = T.block<3,3>(0,0)*point3d_l+T.block<3,1>(0,3);
            if(point3d_w(2)<0.1){
                continue;
            }

            Eigen::Vector3d point3d, point_cam, point_image;
            point3d << point3d_w(0) / point3d_w(2), 
                    point3d_w(1) / point3d_w(2), 1;
            
            double du,dv;
            std::vector<double> dist_coeff = 
                {dist_vec[k](0,0),dist_vec[k](0,1),dist_vec[k](0,2),dist_vec[k](0,3)};
            Distortion(dist_coeff, point3d(0), point3d(1), &du, &dv);
            point_cam << point3d(0) + du, point3d(1) + dv, 1.0;

            point_image = k_vec[k] * point_cam;

            if (point_image(0)<0 || point_image(0)>img_width ||
                point_image(1)<0 || point_image(1)>img_height){
                continue;
            }
            auto pnt = pc.at(i);
            // pnt.b = images[k].at<cv::Vec3b>(point_image(0),point_image(1))[0];
            // pnt.g = images[k].at<cv::Vec3b>(point_image(0),point_image(1))[1];
            // pnt.r = images[k].at<cv::Vec3b>(point_image(0),point_image(1))[2];
            pnt.b = images[k].at<cv::Vec3b>(point_image(1),point_image(0))[0];
            pnt.g = images[k].at<cv::Vec3b>(point_image(1),point_image(0))[1];
            pnt.r = images[k].at<cv::Vec3b>(point_image(1),point_image(0))[2];
            color_pc.push_back(pnt);
            break;
        }
    }
    color_pc.shrink_to_fit();
    pc.swap(color_pc);
}

void TransPoints(std::vector<sensemap::PlyPoint>& pc, 
                 const RigNamesNew& rig_list){
    // Eigen::Matrix3x4d trans = InvertProjectionMatrix(ComposeProjectionMatrix(rig_list.q, rig_list.t));
    Eigen::Matrix3x4d trans = ComposeProjectionMatrix(rig_list.q, rig_list.t);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,4>(0,0) = trans;

    std::vector<sensemap::PlyPoint> t_pc;
    t_pc.reserve(pc.size());
    for (int i = 0; i < pc.size(); i++){
        Eigen::Vector4d pnt_v(pc.at(i).x, pc.at(i).y, pc.at(i).z, 1.0);
        Eigen::Vector4d pnt_t = T * pnt_v;

        auto pnt = pc.at(i);
        pnt.x = pnt_t.x();
        pnt.y = pnt_t.y();
        pnt.z = pnt_t.z();
        t_pc.push_back(pnt);
    }
    pc.swap(t_pc);
}

int main(int argc, char** argv) {

	PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
    
    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());


    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    root_path = GetParentDir(workspace_path);
    std::vector<RigNamesNew> rig_list;
    ReadRigList(rig_list, root_path + "/file_list_withlidarpose.txt");
    // exit(-1);
    std::string save_path = root_path + "/output_pc";
    boost::filesystem::remove_all(save_path);
    CreateDirIfNotExists(save_path);

    std::vector<sensemap::PlyPoint> points;
    for (int i = 0; i < rig_list.size(); i = i+ 2){
        std::cout << rig_list.at(i).pcd << "\n" << rig_list.at(i).img << "\n"
            << rig_list.at(i).time << "\t" << rig_list.at(i).q.transpose() << std::endl;

        std::string lidar_path = rig_list.at(i).pcd;
        auto pc = ReadPCD(lidar_path);
        std::vector<sensemap::PlyPoint> ply_points = Convert2Ply(pc);
        GetColor(ply_points, rig_list.at(i));
        TransPoints(ply_points, rig_list.at(i));
        WriteTextPlyPoints(save_path + "/" + GetPathBaseName(rig_list.at(i).pcd) + "_t.ply", ply_points, false, true);

        points.insert(points.end(), ply_points.begin(), ply_points.end());
    }
    WriteTextPlyPoints(save_path + "/" + "A_word.ply", points, false, true);
    
    return 0;
}