#include <fstream>
#include <boost/filesystem/path.hpp>
#include <dirent.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "util/logging.h"


template <typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> &src, cv::Mat &dst) {
    if (!(src.Flags & Eigen::RowMajorBit)) {
        cv::Mat _src(src.cols(), src.rows(), cv::DataType<_Tp>::type, (void *)src.data(), src.stride() * sizeof(_Tp));
        cv::transpose(_src, dst);
    } else {
        cv::Mat _src(src.rows(), src.cols(), cv::DataType<_Tp>::type, (void *)src.data(), src.stride() * sizeof(_Tp));
        _src.copyTo(dst);
    }
}

int main(int argc, char *argv[]) {
    
    std::string gps_trans_txt_path = argv[1];
    std::string gps_trans_yaml_path = argv[2];

    Eigen::Matrix<double,3,3> r_current;
    Eigen::Vector3d t_current;
    
    Eigen::Matrix<double,3,4> transform;

    std::ifstream file_gps_trans(gps_trans_txt_path.c_str());
    CHECK(file_gps_trans.is_open());
    std::cout<<"gps_trans_txt_path: "<<gps_trans_txt_path<<std::endl;
    for(size_t i = 0;i < 3; i++){
        double term;
        file_gps_trans >> term;
        t_current(i) = -term;
    }
    for(size_t i = 0; i<3; i++){
        for(size_t j = 0; j<3; j++){
            file_gps_trans >> r_current(i,j);
        }
    }
    file_gps_trans.close();

    transform.block<3,3>(0,0) = r_current;
    transform.block<3,1>(0,3) = r_current * t_current;
    std::cout<<transform<<std::endl;

    cv::Mat result_trans;
    eigen2cv(transform, result_trans);
    cv::FileStorage fs(gps_trans_yaml_path, cv::FileStorage::WRITE);
    fs << "transMatrix" << result_trans;
    fs.release();
    return 0;
}