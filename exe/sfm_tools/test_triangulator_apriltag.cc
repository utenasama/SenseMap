// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <time.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>

#include <boost/filesystem/path.hpp>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "../Configurator.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/reconstruction.h"
#include "base/similarity_transform.h"
#include "container/feature_data_container.h"
#include "container/scene_graph_container.h"
#include "util/tag_scale_recover.h"
#include "util/timer.h"
#include "util/alignment.h"
#include "util/types.h"

#include "util/misc.h"
#include "base/version.h"

using namespace sensemap;

std::string image_path;
std::string workspace_path;
// Set the AprilTag Edge
double Real_AprilTag_Edge = 0.113;  // m -- A4

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

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

bool OutputAlignmentFile(std::vector<std::pair<std::string, Eigen::Vector3d>> &alignment_points,
                         const std::string path) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    // std::sort(alignment_points.begin(), alignment_points.end(),
    //           [](const std::pair<std::string, Eigen::Vector3d> &v1, const std::pair<std::string, Eigen::Vector3d> &v2) {
    //               return v1.first < v2.first;
    //           });
    file << "# Apriltag Detection list with one line of data per detection:" << std::endl;
    file << "#   TAG_ID, X, Y, Z" << std::endl;
    file << "# Number of Apriltags: " << alignment_points.size() << std::endl;

    for (const auto &alignment_point : alignment_points) {
        std::ostringstream line;
        std::string line_string;

        line << alignment_point.first << " ";

        // (x, y ,z)
        line << alignment_point.second[0] << " ";
        line << alignment_point.second[1] << " ";
        line << alignment_point.second[2];
        file << line.str() << std::endl;
    }
    file.close();
    return 0;
}

bool OutputTagPointFile(std::vector<std::pair<std::string, Eigen::Vector3d>> &alignment_points,
                        std::vector<Eigen::Vector3d> &alignment_tag_points,
                        const std::string path) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    // std::sort(alignment_points.begin(), alignment_points.end(),
    //           [](const std::pair<std::string, Eigen::Vector3d> &v1, const std::pair<std::string, Eigen::Vector3d> &v2) {
    //               return v1.first < v2.first;
    //           });
    file << "# Apriltag Detection list with one line of data per detection:" << std::endl;
    file << "#   TAG-ID_POINT-ID, X, Y, Z" << std::endl;
    file << "# Number of Apriltags: " << alignment_points.size() << std::endl;

    for (int i = 0; i < alignment_points.size(); i++) {
        std::string tag_name = alignment_points[i].first;
        auto tag_name_split = StringSplit(tag_name, "_");
        std::string tag_id_str = tag_name_split[0];

        for (int j = 0; j < 4; j++) {
            std::ostringstream line;
            std::string line_string;
            line << tag_id_str+"_"+std::to_string(j+1) << " ";

            std::cout << i*4 + j << std::endl;
            // (x, y ,z)
            line << alignment_tag_points[i*4 + j][0] << " ";
            line << alignment_tag_points[i*4 + j][1] << " ";
            line << alignment_tag_points[i*4 + j][2] << " 0 0 0 1";
            file << line.str() << std::endl;
        }
    }
    file.close();
    return 0;
}

int main(int argc, char *argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: Apriltag-triangulator-")+__VERSION__);

    if (argc < 3) {
        std::cout << "Usage test_triangulator_apriltag 1.workspace_path 2.Apriltag Square Size(m, eg. for A4 paper "
                     "size = 0.113) (Options)3.Tag Measurement txt path\n";
        exit(-1);
    }

    workspace_path = std::string(argv[1]);
    Real_AprilTag_Edge = std::atof(argv[2]);

    std::string tag_measurement_path = " ";
    std::map<std::pair<std::string, std::string>, double> tag_measurements;
    bool camera_rig = true;
    FILE *fs;
    fs = fopen((workspace_path + "/tag_statistics.txt").c_str(), "w");
    if (argc >= 4) {
        // FIXME: TMP baned tag measurement recover scale method
        fprintf(
            fs, "%s\n",
            StringPrintf("ERROR: Tag Measurement Scale Recover is invalid")
                .c_str());
        fflush(fs); 
        exit(0);

        // tag_measurement_path = std::string(argv[3]);
        // ReadTagMeasurement(tag_measurement_path, tag_measurements);
    }

    std::cout << "Given Apriltag Square Size : " << Real_AprilTag_Edge << " m" << std::endl;

    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    std::cout << "Camera Rig = " << camera_rig << std::endl;

    // Read feature file
    PrintHeading1("Loading feature data ");
    Timer timer;
    timer.Start();
    if (dirExists(workspace_path + "/features.bin")) {
        feature_data_container->ReadImagesBinaryDataWithoutDescriptor(workspace_path + "/features.bin");
        if (dirExists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container->ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        feature_data_container->ReadAprilTagBinaryData(workspace_path + "/apriltags.bin");
        feature_data_container->ReadSubPanoramaBinaryData(workspace_path + "/sub_panorama.bin");
    } else {
        std::cerr << "ERROR: Current workspace do not contain features.bin " << workspace_path << std::endl;
        fprintf(
            fs, "%s\n",
            StringPrintf("ERROR: Current workspace do not contain features.bin")
                .c_str());
        fflush(fs); 
        exit(0);
    }
    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    // Read scene graph .txt or .bin
    PrintHeading1("Loading scene graph matching data");
    timer.Start();
    if (dirExists(workspace_path + "/scene_graph.bin")) {
        scene_graph_container->ReadSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
    } else if (dirExists(workspace_path + "/scene_graph.txt")) {
        scene_graph_container->ReadSceneGraphData(workspace_path + "/scene_graph.txt");
    } else {
        std::cerr << "ERROR: Current workspace do not contain scene_graph.bin or scene_graph.txt " << workspace_path
                  << std::endl;
        fprintf(
            fs, "%s\n",
            StringPrintf("ERROR: Current workspace do not contain scene_graph.bin or scene_graph.txt")
                .c_str());
        fflush(fs); 
        exit(0);
    }
    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    auto correspondence_graph = scene_graph_container->CorrespondenceGraph();

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph_container->Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph_container->Cameras();

    // FeatureDataContainer data_container;
    std::vector<image_t> image_ids = feature_data_container->GetImageIds();

    std::cout << "image_ids.size() = " << image_ids.size() << std::endl;

    for (const auto image_id : image_ids) {
        const Image &image = feature_data_container->GetImage(image_id);
        if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        images[image_id] = image;
        const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
        // const std::vector<Eigen::Vector2d> points = FeatureKeypointsToPointsVector(keypoints);
        // images[image_id].SetPoints2D(points);
        images[image_id].SetPoints2D(keypoints);

        // std::cout << image.CameraId() << std::endl;
        const Camera &camera = feature_data_container->GetCamera(image.CameraId());
        // std::cout << image.CameraId() << std::endl;
        if (!scene_graph_container->ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
        }

        const PanoramaIndexs &panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        images[image_id].SetLocalImageIndices(local_image_indices);

        if (correspondence_graph->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
        } else {
            // std::cout << "Do not contain this image" << std::endl;
        }
    }
    std::cout << "Load correspondece graph finished" << std::endl;
    correspondence_graph->Finalize();

    // Check the old reconstruction exist or not
    if (!ExistsDir(workspace_path + "/0/")) {
        std::cerr << "ERROR: Input reconstruction path is not a directory  " << workspace_path + "/0/" << std::endl;
        fprintf(
            fs, "%s\n",
            StringPrintf("ERROR: Input reconstruction path is not a directory")
                .c_str());
        fflush(fs); 
        exit(0);
    }

    // Create output path
    // if (!boost::filesystem::exists(workspace_path)) {
    //     boost::filesystem::create_directories(workspace_path);
    // }

    PrintHeading1("Loading model");

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    timer.Start();
    std::cout << "model path : " << workspace_path + "/0/" << std::endl;
    reconstruction->ReadReconstruction(workspace_path + "/0/", camera_rig);

    // Add local camera index for reconstruction
    for (const auto &image_id : reconstruction->RegisterImageIds()) {
        const PanoramaIndexs &panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(feature_data_container->GetKeypoints(image_id).size());
        for (size_t i = 0; i < feature_data_container->GetKeypoints(image_id).size(); ++i) {
            if (panorama_indices.size() == 0 &&
                feature_data_container->GetCamera(feature_data_container->GetImage(image_id).CameraId())
                        .NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        reconstruction->Image(image_id).SetLocalImageIndices(local_image_indices);
    }

#ifdef DEBUG_APRILTAG_TRI
    std::shared_ptr<Reconstruction> reconstruction_old = std::make_shared<Reconstruction>();
    reconstruction_old->ReadReconstruction(workspace_path + "/0/");
#endif

    TagScaleRecover::TagScaleRecoverOptions scale_recover_options;
    scale_recover_options.workspace_path = workspace_path;
    scale_recover_options.tag_size = Real_AprilTag_Edge;
    TagScaleRecover scale_recover(scale_recover_options);
    double scale = scale_recover.ComputeScale(feature_data_container, scene_graph_container, reconstruction);

    std::cout << "scale = " << scale << std::endl;
    
    if (scale != 0 && !std::isnan(scale)) {
        // Create Apriltag result folder
        std::string apriltag_result_path = workspace_path + "/apriltag";
        if (!boost::filesystem::exists(apriltag_result_path)) {
            boost::filesystem::create_directories(apriltag_result_path);
        }

        // Construct yaml file
        Eigen::Matrix3x4d scale_trans = Eigen::Matrix3x4d::Identity();
        // std::cout << "Initial scale_trans = " << scale_trans << std::endl;

        scale_trans = scale_trans * scale;
        // std::cout << "Final Scale trans = " << scale_trans << std::endl;
        cv::Mat result_trans;

        eigen2cv(scale_trans, result_trans);
        cv::FileStorage fsm(apriltag_result_path + "/scale_trans.yaml", cv::FileStorage::WRITE);
        fsm << "transMatrix" << result_trans;
        fsm.release();

        OutputAlignmentFile(scale_recover.alignment_points_, apriltag_result_path + "/alignment.txt");

        OutputTagPointFile(scale_recover.alignment_points_, scale_recover.alignment_tag_points_, apriltag_result_path + "/tag_points.txt");
    }

    return 0;
}