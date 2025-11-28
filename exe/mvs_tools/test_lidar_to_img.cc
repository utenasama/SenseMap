//Copyright (c) 2021, SenseTime Group.
//All rights reserved.
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include<ceres/ceres.h>

#include "lidar/pcd.h"
#include "util/ply.h"
#include "util/misc.h"
#include "util/exception_handler.h"
#include "base/version.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "base/pose.h"

#include <fstream>
// #include <nlohmann/json.hpp>
// using json = nlohmann::json;

using namespace sensemap;
using namespace std;

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

std::vector<float> RGBDParamsToVector(const std::string& rgbd_params) {
    std::vector<std::string> strparams = StringSplit(rgbd_params, ",");
    std::vector<float> params;
    for (auto strparam : strparams) {
        params.push_back(std::atof(strparam.c_str()));
    }
    return params;
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

void ProjectPoints(Eigen::Matrix3d& K, Eigen::Matrix<double, 1, 4>& dist, Eigen::Matrix4d& T,
                   PCDPointCloud ptr_cloud, 
                   std::vector<cv::Point2d>& cv_points,
                   std::vector<cv::Point3d>& cv_points_color){
    // std::cout << "&" << ptr_cloud->width * ptr_cloud->height;
    std::cout<<"ptr_cloud.info.num_points: "<<ptr_cloud.info.num_points<<std::endl;
    std::cout<<"ptr_cloud.info.width: "<<ptr_cloud.info.width<<" ptr_cloud.info.height: "<<ptr_cloud.info.height<<std::endl;
    int img_width=4032, img_height= 3040;
    for(int i = 0; i < ptr_cloud.info.num_points; i++){
        long int pnt_height = i % ptr_cloud.info.height;
        long int pnt_width = i / ptr_cloud.info.height;
        // if(ptr_cloud.point_cloud[pnt_height][pnt_width].z <= 0.01){
        //     continue;
        // }
        Eigen::Vector3d point3d_w;
        point3d_w<<ptr_cloud.point_cloud[pnt_height][pnt_width].x, ptr_cloud.point_cloud[pnt_height][pnt_width].y,ptr_cloud.point_cloud[pnt_height][pnt_width].z;
        // std::cout<<"point3d_w:   "<<point3d_w<<std::endl;
        point3d_w=T.block<3,3>(0,0)*point3d_w+T.block<3,1>(0,3);
        // std::cout<<"point3d_w trans: "<<point3d_w<<std::endl;
        if(point3d_w(2)<0.1)
            continue;
        Eigen::Vector3d point3d, point_cam, point_image;
        point3d << point3d_w(0) / point3d_w(2), 
                   point3d_w(1) / point3d_w(2), 1;
        
        double du,dv;
        // std::cout<<"dist af: "<<dist<<std::endl;
        std::vector<double> dist_coeff = {dist(0,0),dist(0,1),dist(0,2),dist(0,3)};
        Distortion(dist_coeff, point3d(0), point3d(1), &du, &dv);
        // std::cout<<"du: "<<du<<" dv: "<<dv<<std::endl;
        point_cam << point3d(0) + du, point3d(1) + dv, 1.0;
        // std::cout<<"point_cam: "<<point_cam<<std::endl;

        point_image = K * point_cam;
        // std::cout<<"point_image: "<<point_image<<std::endl;
        // std::cout<<"\n\n\n"<<std::endl;
        if (point_image(0)>=0 && point_image(0)<img_width && point_image(1)>=0 && point_image(1)<img_height){
            cv_points.push_back(cv::Point2d(point_image(0), point_image(1)));
            static int point_valid_cnt=0;
            // std::cout<<"point valid: "<<point_valid_cnt++<<std::endl;
            // cout << point_image(0) << " " << point_image(1) << "&";

            double grayValue = min(max((double)((ptr_cloud.point_cloud[pnt_height][pnt_width].z-1) * 255 / 2.5), 0.0), 255.0);
            unsigned char pixel[3];
            if (grayValue <= 51)
            {
                pixel[0] = 255;
                pixel[1] = grayValue * 5;
                pixel[2] = 0;
            }
            else if (grayValue <= 102)
            {
                grayValue -= 51;
                pixel[0] = 255 - grayValue * 5;
                pixel[1] = 255;
                pixel[2] = 0;
            }
            else if (grayValue <= 153)
            {
                grayValue -= 102;
                pixel[0] = 0;
                pixel[1] = 255;
                pixel[2] = grayValue * 5;
            }
            else if (grayValue <= 204)
            {
                grayValue -= 153;
                pixel[0] = 0;
                pixel[1] = 255 - static_cast<unsigned char>(grayValue * 128.0 / 51 + 0.5);
                pixel[2] = 255;
            }
            else if (grayValue <= 255)
            {
                grayValue -= 204;
                pixel[0] = 0;
                pixel[1] = 127 - static_cast<unsigned char>(grayValue * 127.0 / 51 + 0.5);
                pixel[2] = 255;
            }

            cv_points_color.push_back(cv::Point3d(pixel[0], pixel[1], pixel[2]));
        }else{
            // std::cout<<"invalid"<<std::endl;
        }
    }
    std::cout << "\t" << cv_points.size() << "&" << cv_points_color.size() ; 
}

void drawPoints(cv::Mat& image, std::vector<cv::Point2d>& cv_points, std::vector<cv::Point3d>& cv_points_color){
    std::cout<<"cv_points.size(): "<<cv_points.size()<<std::endl;
    for(int i = 0; i < cv_points.size(); i++){
        // std::cout<<"cv_points: "<<cv_points.at(i)<<std::endl;
        cv::circle(image, cv_points.at(i), 5, cv::Scalar(cv_points_color.at(i).x, 
                cv_points_color.at(i).y, cv_points_color.at(i).z), -1);
        // circle(image, cv_points.at(i), 10, Scalar(0,225,0), -1);
    }
}

int main(int argc, char** argv) {

	PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
    
    // if (argc < 3) {
    //     std::cout << "Please enter ./test_lidar_to_img input.pcd input1.jpg input2.jpg input3.jpg" << std::endl;
	// 	return StateCode::NO_MATCHING_INPUT_PARAM;
    // }

    std::ifstream ifs;
    //打开文件
    ifs.open("./file_list.txt", std::ios::in);
    //定义一个字符串
    std::string str;
    //从文件中读取数据
    while(getline(ifs, str))
    {
        std::cout << str << std::endl;
    
    std::string img1,img2,img3;
    getline(ifs, img1);
    getline(ifs, img2);
    getline(ifs, img3);


    const std::string in_pcd_path = str;
    // const std::string in_img_path=std::string(argv[2]);
    std::vector<std::string> in_img_paths;
    in_img_paths.push_back(img1);
    in_img_paths.push_back(img2);
    in_img_paths.push_back(img3);
    auto pc = ReadPCD(in_pcd_path);
    // std::ifstream f("./camera_calibration.json");
    // json data = json::parse(f);
    std::vector<cv::Mat> out_images;

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

    Eigen::Matrix3d k_0,k_1,k_2;
    Eigen::Matrix<double, 1, 4> dist_0, dist_1, dist_2;
    std::vector<Eigen::Matrix3d> k_vec;
    std::vector<Eigen::Matrix<double, 1, 4>> dist_vec;
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


    for(int i=0;i<3;i++){
        std::cout<<"i: "<<i<<std::endl;
        // data["value0"][intrinsics]
        Eigen::Matrix3d k_c0;
        Eigen::Matrix<double, 1, 4> dist_c0;
        k_c0=k_vec[i];
        dist_c0=dist_vec[i];

        // k_c0<< data["value0"]["intrinsics"][i]["intrinsics"]["fx"], 0., data["value0"]["intrinsics"][i]["intrinsics"]["cx"],
        //         0.,data["value0"]["intrinsics"][i]["intrinsics"]["fy"], data["value0"]["intrinsics"][i]["intrinsics"]["cy"],
        //         0.,0.,1.;
        // dist_c0 << data["value0"]["intrinsics"][i]["intrinsics"]["k1"],data["value0"]["intrinsics"][i]["intrinsics"]["k2"],
        //         data["value0"]["intrinsics"][i]["intrinsics"]["k3"],data["value0"]["intrinsics"][i]["intrinsics"]["k4"];
        std::cout<<"k_c0: "<<k_c0<<std::endl;
        std::cout<<"dist_c0: "<<dist_c0<<std::endl;

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

        // Eigen::Matrix3f rot;
		// rot << 0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0;
		// Eigen::Matrix3f rot=Eigen::Matrix3f::Identity();
		Eigen::Matrix3f R = yawAngle.matrix() * pitchAngle.matrix() * rollAngle.matrix();

        Eigen::Vector3d tvec_lidar(lidar_px,lidar_py,lidar_pz);

        //lidar to img
        Eigen::Matrix4d T_lidar_to_cam = Eigen::Matrix4d::Identity();
        T_lidar_to_cam.block<3, 3>(0, 0) = R.cast<double>();
        T_lidar_to_cam.block<3, 1>(0, 3) = tvec_lidar;

        Eigen::Matrix4d World2CV;
        World2CV<<0, -1, 0, 0,0, 0, -1, 0,1, 0, 0, 0,0, 0, 0, 1;
        std::cout<<"T_lidar_to_cam bf: "<<T_lidar_to_cam<<std::endl;
    
        Eigen::Matrix4d cam_to_front = Eigen::Matrix4d::Identity();
        // Eigen::Vector4d cam_to_front_r_q={data["value0"]["T_imu_cam"][i]["qw"],data["value0"]["T_imu_cam"][i]["qx"],
        //                                 data["value0"]["T_imu_cam"][i]["qy"],data["value0"]["T_imu_cam"][i]["qz"]};
        // Eigen::Vector3d cam_to_front_t={data["value0"]["T_imu_cam"][i]["px"],data["value0"]["T_imu_cam"][i]["py"],data["value0"]["T_imu_cam"][i]["pz"]};
        // std::cout<<"cam_to_front_r_q: "<<cam_to_front_r_q<<std::endl;
        // std::cout<<"cam_to_front_t: "<<cam_to_front_t<<std::endl;
        
        // Eigen::Matrix3d cam_to_front_r = QuaternionToRotationMatrix(cam_to_front_r_q);
        // cam_to_front.block<3,3>(0,0)=cam_to_front_r;
        // cam_to_front.block<3, 1>(0, 3) = cam_to_front_t;
        cam_to_front = cam_to_front_vec[i];
        std::cout<<"cam_to_front: "<<cam_to_front<<std::endl;
        std::cout<<"cam_to_front inverse: "<<cam_to_front.inverse()<<std::endl;
        

        T_lidar_to_cam=cam_to_front.inverse() * World2CV * T_lidar_to_cam;
        std::cout<<"T_lidar_to_cam af aix: "<<T_lidar_to_cam<<std::endl;

        // Eigen::Matrix4d T_cam_to_front = Eigen::Matrix4d::Identity();
        // Eigen::Matrix4d T_lidar_to_front = Eigen::Matrix4d::Identity();
        // T_lidar_to_front= World2CV * T_lidar_to_cam;
        // T_cam_to_front = cam_to_front;
        // std::cout<<"T_lidar_to_front: "<<T_lidar_to_front<<std::endl;
        // std::cout<<"T_cam_to_front"<<T_cam_to_front<<std::endl;

        std::vector<cv::Point2d> cv_points;
        std::vector<cv::Point3d> cv_points_color;
        std::cout<<"kc0: "<<k_c0<<std::endl;
        ProjectPoints(k_c0, dist_c0, T_lidar_to_cam, pc, cv_points, cv_points_color);
        
        cv::Mat image_f = cv::imread(in_img_paths[i]);
        drawPoints(image_f, cv_points, cv_points_color);
        // std::string out_name_part = in_pcd_path.substr(8,10);
        // std::string out_name="./out_result/"+ out_name_part+ "_"+std::to_string(i)+".jpg";
        // cv::imwrite(out_name,image_f);
        out_images.push_back(image_f);
    }
    if(out_images.size()!=3){
        std::cout<<"name1: "<<in_pcd_path<<" img1: "<<in_img_paths[0]<<std::endl;
        std::cout<<" img2: "<<in_img_paths[1]<<" img3: "<<in_img_paths[2]<<std::endl;
        continue;
    }
    bool continue_flag=false;
    for(int a=0;a<3;a++){
        if(out_images[a].empty()){
            continue_flag=true;
        }
    }
    if(continue_flag){
        std::cout<<"name1: "<<in_pcd_path<<" img1: "<<in_img_paths[0]<<std::endl;
        std::cout<<" img2: "<<in_img_paths[1]<<" img3: "<<in_img_paths[2]<<std::endl;
        continue;
    }
    cv::Mat out_img_result;
    cv::hconcat(out_images, out_img_result);
    std::string out_name_part = in_pcd_path.substr(8,14);
    std::string out_image_name="./out_result/"+ out_name_part+ ".jpg";
    cv::imwrite(out_image_name, out_img_result);

    }
    // std::vector<sensemap::PlyPoint> ply_points;
    // ply_points.reserve(pc.info.num_points);
    // for (int i = 0; i < pc.info.num_points; i++){
    //     PlyPoint pnt;
    //     long int pnt_height = i % pc.info.height;
    //     long int pnt_width = i / pc.info.height;
    //     pnt.x = pc.point_cloud[pnt_height][pnt_width].x;
    //     pnt.y = pc.point_cloud[pnt_height][pnt_width].y;
    //     pnt.z = pc.point_cloud[pnt_height][pnt_width].z;
    //     // ColorMap(pc.point_cloud[pnt_height][pnt_width].intensity,
    //     //          pnt.r, pnt.g, pnt.b);
    //     ColorMap(float(pnt_height) / 32,
    //              pnt.r, pnt.g, pnt.b);
    //     ply_points.push_back(pnt);
    // }
    // WriteTextPlyPoints("./lidar_pcd.ply", ply_points, false, true);
    return 0;
}