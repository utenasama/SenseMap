#include <fstream>
#include "base/reconstruction_manager.h"
#include <boost/filesystem/path.hpp>
#include <dirent.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace sensemap;

int main(int argc, char *argv[]) {
    std::string workspace_path = argv[1];
    
    std::string gps_trans_current_path = argv[2];
    std::string gps_trans_reference_path = argv[3];
    std::string target_trans_path;

    if(argc >=5){
        target_trans_path = argv[4];
    }

    Eigen::Matrix<double,3,3> r_current;
    Eigen::Vector3d t_current;
    Eigen::Matrix<double,3,3> r_reference;
    Eigen::Vector3d t_reference;

    Eigen::Matrix<double,3,4> transform;

    std::ifstream file_gps_trans_current(gps_trans_current_path.c_str());
    CHECK(file_gps_trans_current.is_open());

    for(size_t i = 0;i < 3; i++){
        double term;
        file_gps_trans_current >> term;
        t_current(i) = -term;
    }
    for(size_t i = 0; i<3; i++){
        for(size_t j = 0; j<3; j++){
            file_gps_trans_current >> r_current(i,j);
        }
    }
    file_gps_trans_current.close();


    std::ifstream file_gps_trans_reference(gps_trans_reference_path.c_str());
    CHECK(file_gps_trans_reference.is_open());

    for(size_t i = 0;i < 3; i++){
        double term;
        file_gps_trans_reference >> term;
        t_reference(i) = -term;
    }
    for(size_t i = 0; i<3; i++){
        for(size_t j = 0; j<3; j++){
            file_gps_trans_reference >> r_reference(i,j);
        }
    }
    file_gps_trans_reference.close();


    Eigen::Matrix3d r_final = r_reference * (r_current.transpose());
    Eigen::Vector3d t_final = r_reference * t_reference - r_reference * t_current;

    transform.block<3,1>(0,3) = t_final;
    transform.block<3,3>(0,0) = r_final;

    if (argc >= 5) {
        Eigen::Matrix3x4d target_trans;
        cv::Mat target_trans_mat;

        cv::FileStorage fs(target_trans_path, cv::FileStorage::READ);
        fs["transMatrix"] >> target_trans_mat;
        fs.release();
        cv::cv2eigen(target_trans_mat, target_trans);
        transform.block<3,1>(0,3) = target_trans.block<3,1>(0,3) + target_trans.block<3,3>(0,0) * t_final;
        transform.block<3,3>(0,0) = target_trans.block<3,3>(0,0) * r_final;

    }

    cv::Mat result_trans;
    cv::eigen2cv(transform, result_trans);
    std::string current_to_target_trans_file = workspace_path +"/current_to_taget.yaml";

    cv::FileStorage result_fs(current_to_target_trans_file, cv::FileStorage::WRITE);
    result_fs << "transMatrix" << result_trans;
    result_fs.release();

    auto org_reconstruction = std::make_shared<Reconstruction>();
    CHECK(boost::filesystem::exists(workspace_path+"/0"));
    org_reconstruction->ReadReconstruction(workspace_path + "/0", false);
    org_reconstruction->TransformReconstruction(transform, false);

    if (!boost::filesystem::exists(workspace_path+"/0-trans")) {
        boost::filesystem::create_directories(workspace_path+"/0-trans");
    }
    org_reconstruction->WriteBinary(workspace_path + "/0-trans");

    //write result
    std::ofstream file_poses(workspace_path + "/0-trans/poses.txt");
    CHECK(file_poses.is_open());
    
    auto registered_image_ids = org_reconstruction->RegisterImageIds();
    for(const auto& image_id: registered_image_ids){
        class Image& image = org_reconstruction->Image(image_id);
        file_poses << image.Name() << " " << image.Tvec()[0] << " " << image.Tvec()[1] << " " << image.Tvec()[2] << " "
                   << image.Qvec()[0] << " " << image.Qvec()[1] << " " << image.Qvec()[2] << " " << image.Qvec()[3]
                   << std::endl;
    }
    file_poses.close();

    org_reconstruction.reset();

    auto keyframe_reconstruction = std::make_shared<Reconstruction>();
    CHECK(boost::filesystem::exists(workspace_path + "/0/KeyFrames"));
    keyframe_reconstruction->ReadReconstruction(workspace_path + "/0/KeyFrames", false);
    keyframe_reconstruction->TransformReconstruction(transform, false);
    
    if (!boost::filesystem::exists(workspace_path+"/0-trans/KeyFrames")) {
        boost::filesystem::create_directories(workspace_path+"/0-trans/KeyFrames");
    }
    keyframe_reconstruction->WriteBinary(workspace_path + "/0-trans/KeyFrames");
    keyframe_reconstruction.reset();

    
    return 0;
}
