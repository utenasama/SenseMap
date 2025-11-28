#include <fstream>
#include "base/reconstruction_manager.h"
#include <boost/filesystem/path.hpp>
#include <dirent.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <util/gps_reader.h>

using namespace sensemap;

int main(int argc, char* argv[]){
    std::string trans_path = argv[1];
    std::string gps_coord_file = argv[2];

    Eigen::Matrix3x4d target_trans;
    cv::Mat target_trans_mat;

    cv::FileStorage fs(trans_path, cv::FileStorage::READ);
    fs["transMatrix"] >> target_trans_mat;
    fs.release();
    cv::cv2eigen(target_trans_mat, target_trans);

    std::ifstream file(gps_coord_file);
    if(!file.is_open()){
        return -1;
    }

    double lon,lat,alt;
    file>>lon>>lat>>alt;

    GPSReader gps_reader;
    Eigen::Vector3d loc = gps_reader.gpsToLocation(lon,lat,alt);

    Eigen::Vector3d trans_loc = target_trans.block<3,3>(0,0) * loc + target_trans.block<3,1>(0,3);

    


    std::cout<<"transformed loc: "<<trans_loc<<std::endl;
}