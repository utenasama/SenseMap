#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <boost/filesystem/path.hpp>

#include "base/common.h"
#include "base/version.h"
#include "util/obj.h"
#include "util/gps_reader.h"
#include "util/exception_handler.h"
#include "../Configurator_yaml.h"

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

std::string configuration_file_path;

using namespace sensemap;

// bool LoadGpsOrigin(const std::string& gps_origin_path, std::vector<double>& vec_gps){
//     vec_gps.clear();
//     std::ifstream file(gps_origin_path);
//     CHECK(file.is_open()) << gps_origin_path;

//     std::string line;
//     std::string item;

//     while (std::getline(file, line)) {
//         StringTrim(&line);

//         if (line.empty() || line[0] == '#') {
//             continue;
//         }

//         std::stringstream line_stream(line);
//         while (!line_stream.eof()) {
//             std::getline(line_stream, item, ' ');
//             vec_gps.push_back(std::stod(item));
//         }
//     }

//     file.close();
//     return (vec_gps.size() == 3);
// };

int LongLat2XY(double longitude,double latitude,double &X,double &Y)
{   
    longitude -= 114.0; //EPSG:4547 CGCS2000 / 3-degree Gauss-Kruger CM 114E  

    int ProjNo=0; int ZoneWide; //带宽
    double longitude1,latitude1, longitude0,latitude0, X0,Y0, xval,yval;
    double a, f, e2, ee, NN, T, C, A, M, iPI;
    iPI = 0.0174532925199433;  //3.1415926535898/180.0;
    ZoneWide = 3;  //3度带宽
    //ZoneWide = 6; 6度带宽
    //a=6378245.0; f=1.0/298.3; //54年北京坐标系参数
    //a=6378140.0; f=1/298.257; //80年西安坐标系参数
    //a = 6378137.0; f = 1.0/298.257223563;//WGS84坐标系参数
    a = 6378137.0; f = 1/298.257222101;//cgcs2000坐标系参数
    //ProjNo = (int)(longitude / ZoneWide) ;      //6度带
    //longitude0 = ProjNo * ZoneWide + ZoneWide / 2; //6度带
    ProjNo = (int)(longitude / ZoneWide+0.5) ;
    // ProjNo = (int)(longitude / ZoneWide) ; //--带号
    longitude0 = ProjNo * ZoneWide ; //--中央子午线
    longitude0 = longitude0 * iPI ;//--中央子午线转化为弧度
    latitude0=0;
    longitude1 = longitude * iPI ; //经度转换为弧度
    latitude1 = latitude * iPI ; //纬度转换为弧度
    e2=2*f-f*f;

    ee=e2*(1.0-e2);
    NN=a/sqrt(1.0-e2*sin(latitude1)*sin(latitude1));
    T=tan(latitude1)*tan(latitude1);
    C=ee*cos(latitude1)*cos(latitude1);
    A=(longitude1-longitude0)*cos(latitude1);

    M=a*((1-e2/4-3*e2*e2/64-5*e2*e2*e2/256)*latitude1-(3*e2/8+3*e2*e2/32+45*e2*e2*e2/1024)*sin(2*latitude1)
         +(15*e2*e2/256+45*e2*e2*e2/1024)*sin(4*latitude1)-(35*e2*e2*e2/3072)*sin(6*latitude1));
    xval = NN*(A+(1-T+C)*A*A*A/6+(5-18*T+T*T+72*C-58*ee)*A*A*A*A*A/120);
    yval = M+NN*tan(latitude1)*(A*A/2+(5-T+9*C+4*C*C)*A*A*A*A/24
                                +(61-58*T+T*T+600*C-330*ee)*A*A*A*A*A*A/720);
    //X0 = 1000000L*(ProjNo+1)+500000L; //6度带
    X0 = 1000000L*ProjNo+500000L;  //3度带
    Y0 = 0;
    xval = xval+X0; yval = yval+Y0;

    X= xval;
    Y= yval;
    //printf("%lf   %lf\r\n",xval,yval);
    return 1;
}

Eigen::RowMatrix3x4d ComputeTransformationToEpsg4547(const TriangleMesh& mesh_, 
    Eigen::RowMatrix3x4d& trans, double epsg_geoid_height = 0.0) {
    std::vector<Eigen::Vector3d> src_points;
    std::vector<Eigen::Vector3d> tgt_points;
    src_points.reserve(mesh_.vertices_.size());
    tgt_points.reserve(mesh_.vertices_.size());

    int zone, sourth_or_north = 0;
    double latitude, longitude, altitude;
    for (auto vtx : mesh_.vertices_) {
        src_points.push_back(vtx);
        Eigen::Vector3d Xw = trans * vtx.homogeneous();
        GPSReader::LocationToGps(Xw, &latitude, &longitude, &altitude);
        double X, Y;
        LongLat2XY(longitude, latitude, X, Y);
        tgt_points.emplace_back(X, Y, altitude + epsg_geoid_height);
        // zone = (int)utm.z();
        sourth_or_north = (latitude >= 0 ? 1 : -1);
    }

    // sourth_or_north_ = sourth_or_north;
    // zone_no_ = zone;
    // utm_zone_ = MapToUTMZone(zone, sourth_or_north);
    // std::cout << "UTM Zone: " << zone_no_ << std::endl;

    Eigen::Matrix<double, 3, Eigen::Dynamic> src_mat(3, src_points.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_mat(3, tgt_points.size());
    for (size_t i = 0; i < src_points.size(); ++i) {
        src_mat.col(i) = src_points[i];
        dst_mat.col(i) = tgt_points[i];
    }

    Eigen::RowMatrix3x4d model;
    model = Eigen::umeyama(src_mat, dst_mat, false).topLeftCorner(3, 4);
    // std::cout << "model: " << std::endl << model << std::endl;
    
    return model;
}

int main(int argc, char* argv[]){
    PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
    PrintHeading1("EPSG 4547 Convert");

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string trans_path = param.GetArgument("trans_path", "");
    if (workspace_path.empty() || trans_path.empty()){
        std::cout << "workspace_path.empty() || trans_path.empty() " << std::endl;
        return -1;
    }
    std::string epsg_reference = param.GetArgument("epsg_reference", "origin");
    double epsg_geoid_height = static_cast<double>(param.GetArgument("epsg_geoid_height", 0.0f));

    auto ned_ecef_path = JoinPaths(workspace_path, "ned_to_ecef.txt");
    auto gps_origin_path = JoinPaths(workspace_path, "gps_origin.txt");
    auto reconstruction_path = JoinPaths(workspace_path, std::to_string(0));
    auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
    auto model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);

    Eigen::RowMatrix3x4d trans_ned_epsg;
    if (epsg_reference.compare("model") == 0){
        PrintHeading2("model");
        Eigen::RowMatrix3x4d trans;
        std::ifstream fin(ned_ecef_path, std::ifstream::in);
        if (fin.is_open()) {
            fin >> trans(0, 0) >> trans(0, 1) >> trans(0, 2) >> trans(0, 3);
            fin >> trans(1, 0) >> trans(1, 1) >> trans(1, 2) >> trans(1, 3);
            fin >> trans(2, 0) >> trans(2, 1) >> trans(2, 2) >> trans(2, 3);
        }
        fin.close();
        std::cout << "ned to ecef: " << std::endl;
        std::cout << trans << std::endl;

        TriangleMesh mesh;
        if (!ExistsFile(model_path)){
            std::cout << "Error: no exist model in " << model_path << std::endl;
            return -1;
        }
        ReadTriangleMeshObj(model_path, mesh, true);
        mesh.vertex_colors_.clear();
        mesh.vertex_normals_.clear();
        mesh.vertex_labels_.clear();
        mesh.vertex_status_.clear();
        mesh.vertex_visibilities_.clear();
        mesh.face_normals_.clear();

        trans_ned_epsg = ComputeTransformationToEpsg4547(mesh, trans, epsg_geoid_height);
        std::cout << "trans_ned_epsg: \n" << MAX_PRECISION << trans_ned_epsg << std::endl;
    }else if (epsg_reference.compare("origin") == 0){
        std::vector<double> vec_gps_origin;
        if (!boost::filesystem::exists(gps_origin_path)){
            std::cout << "Error: no exist file in " << gps_origin_path << std::endl;
            return -1;
        }
        bool flag = LoadGpsOrigin(gps_origin_path, vec_gps_origin);
        std::cout << "ReaderOptions set ori_gps_origin (" << flag << "):" 
                  << vec_gps_origin[0] << ", " << vec_gps_origin[1] << ", " 
                  << vec_gps_origin[2] << std::endl;
        // vec_gps_origin = {22.53454928119445, 113.8951514006111,  292.503}; // 22.53454928119445 113.8951514006111  292.503
    
        double atitude = vec_gps_origin[0], longitude= vec_gps_origin[1], X = 0, Y = 0;
        LongLat2XY(longitude, atitude, X, Y);

        // Eigen::RowMatrix3x4d trans_ned_epsg;
        trans_ned_epsg << 0.0, 1.0, 0.0, X,
                        1.0, 0.0, 0.0, Y,
                        0.0, 0.0, -1.0, vec_gps_origin[2] + epsg_geoid_height;
        std::cout << "trans_ned_epsg:\n" << MAX_PRECISION << trans_ned_epsg << std::endl;
    } else {
        std::cout << "Error: epsg_reference is " << epsg_reference << std::endl;
        return -1;
    }

    Eigen::RowMatrix3x4d trans_ned_nerf;
    {
        std::cout << "trans_path: " << trans_path << std::endl;
        cv::FileStorage fs;
        fs.open(trans_path.c_str(), cv::FileStorage::READ);
        cv::Mat trans_mat;
        // std::cout << "Type = " << fs["transMatrix"].type() << std::endl;
        if(fs["transMatrix"].type() != cv::FileNode::MAP){
            std::cout << "ERROR: Input yaml error !!" << std::endl;
            exit(-1);
        }
        fs["transMatrix"] >> trans_mat;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                trans_ned_nerf(i, j) = trans_mat.at<double>(i, j);
            }
        }

        Eigen::RowMatrix3d rot;
        rot << -1, 0, 0, 0, 0, 1, 0, 1, 0;
        trans_ned_nerf = rot * trans_ned_nerf;
    }
    std::cout << "trans_ned_nerf:\n" << trans_ned_nerf << std::endl;

    Eigen::RowMatrix3x4d trans_nerf_epsg;
    Eigen::RowMatrix4d trans_ned_nerf_4 = Eigen::RowMatrix4d::Identity();
    trans_ned_nerf_4.topRows(3) = trans_ned_nerf;
    trans_nerf_epsg = trans_ned_epsg * trans_ned_nerf_4.transpose();
    std::cout << "trans_nerf_epsg:\n" << MAX_PRECISION  << trans_nerf_epsg << std::endl;

    std::ofstream file(JoinPaths(workspace_path, "trans_to_epsg4547.txt"), std::ofstream::out);
    file << MAX_PRECISION << trans_nerf_epsg << std::endl;
    file.close();

    return 0;
}
