// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/obj.h"
#include "util/ply.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

using namespace sensemap;

void LoadFeatureContainer(const std::string& workspace_path, 
                          FeatureDataContainer& feature_data_container) {
    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
        }
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.txt"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.txt"))) {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), false);
            feature_data_container.ReadLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
        } else {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), false);
        }
        feature_data_container.ReadImagesData(JoinPaths(workspace_path, "/features.txt"));
    } else {
        exist_feature_file = false;
    }

    // If the camera model is Spherical
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.txt"))) {
            feature_data_container.ReadSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
        }
    }

    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
            feature_data_container.ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.txt"))) {
            feature_data_container.ReadPieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
        }
    }
    
    // Check the GPS file exist or not.
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/gps.bin"))) {
            feature_data_container.ReadGPSBinaryData(JoinPaths(workspace_path, "/gps.bin"));
        } else {
            std::cout << " Warning! Existing feature data do not contain gps data" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {

    const std::string workspace_path(argv[1]);

    FeatureDataContainer feature_data_container;
    LoadFeatureContainer(workspace_path, feature_data_container);

    const std::string rec_path = workspace_path + "/0";
    Reconstruction reconstruction;
    reconstruction.ReadBinary(rec_path);

    int num_prior = 0;
    const std::vector<image_t> registered_image_ids = reconstruction.RegisterImageIds();
    for (const auto & image_id : registered_image_ids) {
        const auto image_name = reconstruction.Image(image_id).Name();
        if (feature_data_container.ExistImage(image_name)){
            const auto feature_image = feature_data_container.GetImage(image_name);
            if (feature_image.HasTvecPrior()) {
                reconstruction.Image(image_id).SetTvecPrior(feature_image.TvecPrior());
                if (feature_image.HasQvecPrior()) {
                    reconstruction.Image(image_id).SetQvecPrior(feature_image.QvecPrior());
                } else {
                    reconstruction.Image(image_id).SetQvecPrior(reconstruction.Image(image_id).Qvec());
                }
                reconstruction.Image(image_id).RtkFlag() = feature_image.RtkFlag();
                std::cout << "add prior: " << image_name << ", " << reconstruction.Image(image_id).RtkFlag() << ", " << feature_image.RtkFlag() << std::endl;
                num_prior++;
            }
        }
    }
    std::cout << "RegisterImageIds, PriorSize: " << registered_image_ids.size() << "," << num_prior << std::endl;

    Eigen::Matrix3x4d matrix_to_align = reconstruction.AlignWithPriorLocations();
    // Eigen::Matrix3x4d matrix_to_geo = reconstruction.NormalizeWoScale();
    reconstruction.OutputPriorResidualsTxt(workspace_path);

    Eigen::Matrix4d h_matrix_to_align = Eigen::Matrix4d::Identity();
    // h_matrix_to_align.block<3, 4>(0, 0) = matrix_to_align;
    // Eigen::Matrix3x4d M = matrix_to_geo * h_matrix_to_align;
    Eigen::Matrix3x4d M = h_matrix_to_align.block<3, 4>(0, 0);

    const std::string gps_rec_path = workspace_path + "/0-gps";
    boost::filesystem::create_directories(gps_rec_path);
    reconstruction.AddPriorToResult();
    reconstruction.WriteBinary(gps_rec_path);

    {
        std::ofstream file((rec_path + "/matrix_to_gps.txt"), std::ofstream::out);
        file << MAX_PRECISION << M(0, 0) << " " << M(0, 1) << " " 
                << M(0, 2) << " " << M(0, 3) << std::endl;
        file << MAX_PRECISION << M(1, 0) << " " << M(1, 1) << " " 
                << M(1, 2) << " " << M(1, 3) << std::endl;
        file << MAX_PRECISION << M(2, 0) << " " << M(2, 1) << " " 
                << M(2, 2) << " " << M(2, 3) << std::endl;
        file.close();

        // const std::string model_path = workspace_path + "/0/dense/model.obj";
        // if (ExistsFile(model_path)) {
        //     TriangleMesh model;
        //     ReadTriangleMeshObj(model_path, model, true, false);
        //     for (size_t i = 0; i < model.vertices_.size(); ++i) {
        //         auto & X = model.vertices_.at(i);
        //         X = M * X.homogeneous();
        //         auto & Xn = model.vertex_normals_.at(i);
        //         Xn = M.block<3, 3>(0, 0) * Xn;
        //         Xn.normalize();
        //     }
        //     WriteTriangleMeshObj(workspace_path + "/0/dense/model-gps.obj", model);
        // }
        // const std::string fused_path = workspace_path + "/0/dense/random-sampled-fused.ply";
        // if (ExistsFile(fused_path)) {
        //     std::vector<PlyPoint> points = ReadPly(fused_path);
        //     for (size_t i = 0; i < points.size(); ++i) {
        //         PlyPoint & point = points.at(i);
        //         Eigen::Vector3d X(point.x, point.y, point.z);
        //         Eigen::Vector3d Xn(point.nx, point.ny, point.nz);
        //         X = M * X.homogeneous();
        //         Xn = M.block<3, 3>(0, 0) * Xn;
        //         Xn.normalize();
        //         point.x = X[0];
        //         point.y = X[1];
        //         point.z = X[2];
        //         point.nx = Xn[0];
        //         point.ny = Xn[1];
        //         point.nz = Xn[2];
        //     }
        //     const std::string trans_fused_path = workspace_path + "/0/dense/random-sampled-fused-gps.ply";
        //     WriteBinaryPlyPoints(trans_fused_path, points);
        // }
    }

    return 0;
}