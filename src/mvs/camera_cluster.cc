//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"
#include "utils.h"
//#include "util/obj.h"
#include "base/common.h"
#include "base/pose.h"

#include "camera_cluster.h"
#include <stdlib.h>


namespace sensemap {
namespace mvs {

void GetCameraFrustum(Eigen::Vector3f &C, Eigen::Vector3f &TL, Eigen::Vector3f &TR, Eigen::Vector3f &BR, Eigen::Vector3f &BL,
                      const Image &image, const Camera &camera, float depth) {
    Eigen::RowMatrix3f K = camera.CalibrationMatrix().cast<float>();
    Eigen::RowMatrix3f R = QuaternionToRotationMatrix(image.Qvec()).cast<float>();
    Eigen::Vector3f T = image.Tvec().cast<float>();
    C = -R.transpose() * T;
    Eigen::RowMatrix3x4f inv_P;
    utility::ComposeInverseProjectionMatrix(K.data(), R.data(), T.data(), inv_P.data());

    std::size_t width = camera.Width();
    std::size_t height = camera.Height();
//    float focal_length_x = K(0, 0);
//    float focal_length_y = K(1, 1);

    TL = inv_P * Eigen::Vector4f(0.0f, 0.0f, depth, 1.0f);
    TR = inv_P * Eigen::Vector4f(width * depth, 0.0f, depth, 1.0f);
    BR = inv_P * Eigen::Vector4f(width * depth, height * depth, depth, 1.0f);
    BL = inv_P * Eigen::Vector4f(0.0f, height * depth, depth, 1.0f);
}

void CameraCluster::Options::Print() const {
  PrintHeading2("CameraCluster::Options");
  PrintOption(min_pts_per_cluster);
  PrintOption(max_pts_per_cluster);
  PrintOption(min_mappts_per_cluster);
  PrintOption(max_mappts_per_cluster);
  PrintOption(min_num_pixels);
}

bool CameraCluster::Options::Check() const {
  CHECK_OPTION_GT(min_pts_per_cluster, 0);
//  CHECK_OPTION_GT(max_pts_per_cluster, 0);
  CHECK_OPTION_LE(min_pts_per_cluster, max_pts_per_cluster);
  CHECK_OPTION_GT(min_mappts_per_cluster, 0);
//  CHECK_OPTION_GT(max_mappts_per_cluster, 0);
  CHECK_OPTION_LE(min_mappts_per_cluster, max_mappts_per_cluster);
  CHECK_OPTION_GE(min_num_pixels, 0);
  return true;
}

CameraCluster::CameraCluster(const Options& options,
                                     const std::string& workspace_path)
    : num_reconstruction_(0),
      options_(options),
      workspace_path_(workspace_path) {
  CHECK(options_.Check());
}

void CameraCluster::ReadWorkspace() {
  num_reconstruction_ = 0;
  std::cout << "Reading workspace..." << std::endl;
    for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
        const auto& reconstruction_path =
            JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
//        const auto& dense_reconstruction_path =
//            JoinPaths(reconstruction_path, DENSE_DIR);
//        if (!ExistsDir(dense_reconstruction_path)) {
//            break;
//        }

        num_reconstruction_++;
    }
}

void CameraCluster::Run() {
  options_.Print();
  std::cout << std::endl;

  ReadWorkspace();

  for (size_t reconstruction_idx = 0; reconstruction_idx < num_reconstruction_;
       reconstruction_idx++) {

    PrintHeading1(StringPrintf("Clustering# %d", reconstruction_idx));

    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    auto reconstruction_path =
      JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
    auto cluster_reconstruction_path = JoinPaths(reconstruction_path, CLUSTER_DIR);
    if (ExistsDir(cluster_reconstruction_path)) {
        CHECK(boost::filesystem::remove_all(cluster_reconstruction_path));
    }
//    CreateDirIfNotExists(cluster_reconstruction_path);
    CHECK(boost::filesystem::create_directory(cluster_reconstruction_path));

//    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
//    reconstruction->ReadReconstruction(reconstruction_path);
////    {
////        std::string file_name = "./ori_sfm_" + std::to_string(reconstruction_idx) + ".obj";
////        FILE *file = fopen(file_name.c_str(), "w+");
////
////        Eigen::Vector3f color;
////        color[0] = rand() / (float)RAND_MAX;
////        color[1] = rand() / (float)RAND_MAX;
////        color[2] = rand() / (float)RAND_MAX;
////
////        std::size_t face_idx = 1;
////        for (const auto &mappoint : reconstruction->MapPoints()) {
////            const auto &XYZ = mappoint.second.XYZ();
////            fprintf(file, "v %lf %lf %lf %f %f %f\n", XYZ[0], XYZ[1], XYZ[2], color[0], color[1], color[2]);
////            face_idx++;
////        }
////
////        Eigen::Vector3f C, TL, TR, BR, BL;
////        for (auto image_id : reconstruction->RegisterImageIds()) {
////            const Image &image = reconstruction->Image(image_id);
////            const Camera &camera = reconstruction->Camera(image.CameraId());
////            GetCameraFrustum(C, TL, TR, BR, BL, image, camera, 1.0f);
////            fprintf(file, "v %f %f %f %f %f %f\n", C[0], C[1], C[2], color[0], color[1], color[2]);
////            fprintf(file, "v %f %f %f %f %f %f\n", TL[0], TL[1], TL[2], color[0], color[1], color[2]);
////            fprintf(file, "v %f %f %f %f %f %f\n", TR[0], TR[1], TR[2], color[0], color[1], color[2]);
////            fprintf(file, "v %f %f %f %f %f %f\n", BR[0], BR[1], BR[2], color[0], color[1], color[2]);
////            fprintf(file, "v %f %f %f %f %f %f\n", BL[0], BL[1], BL[2], color[0], color[1], color[2]);
////            fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 2, face_idx + 1);
////            fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 3, face_idx + 2);
////            fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 4, face_idx + 3);
////            fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 1, face_idx + 4);
////            fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 2, face_idx + 3);
////            fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 3, face_idx + 4);
////            face_idx += 5;
////        }
////
////        fclose(file);
////    }
//
//    std::shared_ptr<ReconstructionManager> manager = Cluster(reconstruction);
//    if (!manager) {
//        continue;
//    }
//
//    manager->Write(cluster_reconstruction_path);
////    {
////        Eigen::Vector3f C, TL, TR, BR, BL;
////        Eigen::Vector3f color;
////        std::size_t face_idx;
////        for (std::size_t cluster_idx = 0; cluster_idx < manager->Size(); cluster_idx++) {
////            const auto cluster_reconstruction = manager->Get(cluster_idx);
////            std::string file_name = "./cluster_sfm_" + std::to_string(reconstruction_idx) + '_' + std::to_string(cluster_idx) + ".obj";
////            FILE *file = fopen(file_name.c_str(), "w+");
////
////            color[0] = rand() / (float)RAND_MAX;
////            color[1] = rand() / (float)RAND_MAX;
////            color[2] = rand() / (float)RAND_MAX;
////
////            face_idx = 1;
////            for (const auto &mappoint : cluster_reconstruction->MapPoints()) {
////                const auto &XYZ = mappoint.second.XYZ();
////                fprintf(file, "v %lf %lf %lf %f %f %f\n", XYZ[0], XYZ[1], XYZ[2], color[0], color[1], color[2]);
////                face_idx++;
////            }
////
////            for (auto image_id : cluster_reconstruction->RegisterImageIds()) {
////                const Image &image = cluster_reconstruction->Image(image_id);
////                const Camera &camera = cluster_reconstruction->Camera(image.CameraId());
////                GetCameraFrustum(C, TL, TR, BR, BL, image, camera, 1.0f);
////                fprintf(file, "v %f %f %f %f %f %f\n", C[0], C[1], C[2], color[0], color[1], color[2]);
////                fprintf(file, "v %f %f %f %f %f %f\n", TL[0], TL[1], TL[2], color[0], color[1], color[2]);
////                fprintf(file, "v %f %f %f %f %f %f\n", TR[0], TR[1], TR[2], color[0], color[1], color[2]);
////                fprintf(file, "v %f %f %f %f %f %f\n", BR[0], BR[1], BR[2], color[0], color[1], color[2]);
////                fprintf(file, "v %f %f %f %f %f %f\n", BL[0], BL[1], BL[2], color[0], color[1], color[2]);
////                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 2, face_idx + 1);
////                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 3, face_idx + 2);
////                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 4, face_idx + 3);
////                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 1, face_idx + 4);
////                fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 2, face_idx + 3);
////                fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 3, face_idx + 4);
////                face_idx += 5;
////            }
////
////            fclose(file);
////        }
////    }
    auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
    if (!ExistsDir(dense_reconstruction_path)) {
        continue;
    }

    auto sparse_reconstruction_path = JoinPaths(dense_reconstruction_path, SPARSE_DIR);
    if (!ExistsDir(sparse_reconstruction_path)) {
        continue;
    }

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    reconstruction->ReadReconstruction(sparse_reconstruction_path);

    auto fused_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
    if (!ExistsFile(fused_path)) {
        continue;
    }
    std::vector<PlyPoint> fused_points = ReadPly(fused_path);

    auto fused_vis_path = fused_path + ".vis";
    if (!ExistsFile(fused_vis_path)) {
        continue;
    }
    std::vector<std::vector<uint32_t> > fused_points_visibility;
    ReadPointsVisibility(fused_vis_path, fused_points_visibility);
    /*{
        std::string file_name = "./ori_mvs_" + std::to_string(reconstruction_idx) + ".obj";
        FILE *file = fopen(file_name.c_str(), "w+");

        Eigen::Vector3f color;
        color[0] = rand() / (float)RAND_MAX;
        color[1] = rand() / (float)RAND_MAX;
        color[2] = rand() / (float)RAND_MAX;

        std::size_t face_idx = 1;
        for (const auto &fused_point : fused_points) {
            fprintf(file, "v %f %f %f %f %f %f\n", fused_point.x, fused_point.y, fused_point.z, color[0], color[1], color[2]);
            face_idx++;
        }

        Eigen::Vector3f C, TL, TR, BR, BL;
        for (auto image_id : reconstruction->RegisterImageIds()) {
            const Image &image = reconstruction->Image(image_id);
            const Camera &camera = reconstruction->Camera(image.CameraId());
            GetCameraFrustum(C, TL, TR, BR, BL, image, camera, 1.0f);
            fprintf(file, "v %f %f %f %f %f %f\n", C[0], C[1], C[2], color[0], color[1], color[2]);
            fprintf(file, "v %f %f %f %f %f %f\n", TL[0], TL[1], TL[2], color[0], color[1], color[2]);
            fprintf(file, "v %f %f %f %f %f %f\n", TR[0], TR[1], TR[2], color[0], color[1], color[2]);
            fprintf(file, "v %f %f %f %f %f %f\n", BR[0], BR[1], BR[2], color[0], color[1], color[2]);
            fprintf(file, "v %f %f %f %f %f %f\n", BL[0], BL[1], BL[2], color[0], color[1], color[2]);
            fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 2, face_idx + 1);
            fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 3, face_idx + 2);
            fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 4, face_idx + 3);
            fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 1, face_idx + 4);
            fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 2, face_idx + 3);
            fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 3, face_idx + 4);
            face_idx += 5;
        }

        fclose(file);
    }*/

    std::vector<std::vector<std::size_t> > fused_point_idxs_clusters;
    std::vector<std::vector<std::vector<uint32_t> > > fused_points_visibility_clusters;
    std::shared_ptr<ReconstructionManager> manager
        = Cluster(fused_point_idxs_clusters, fused_points_visibility_clusters,
                  reconstruction, fused_points, fused_points_visibility);
    if (!manager) {
        continue;
    }

    for (std::size_t cluster_idx = 0; cluster_idx < manager->Size(); cluster_idx++) {
        auto cluster_path = JoinPaths(cluster_reconstruction_path, std::to_string(cluster_idx));
        CreateDirIfNotExists(cluster_path);
        auto dense_cluster_path = JoinPaths(cluster_path, DENSE_DIR);
        CreateDirIfNotExists(dense_cluster_path);

        const auto cluster_reconstruction = manager->Get(cluster_idx);
        cluster_reconstruction->WriteReconstruction(dense_cluster_path);

        const auto &cluster_fused_point_idxs = fused_point_idxs_clusters[cluster_idx];
        std::size_t num_cluster_fused_points = cluster_fused_point_idxs.size();
        std::vector<PlyPoint> cluster_fused_points(num_cluster_fused_points);
        for (std::size_t i = 0; i < num_cluster_fused_points; i++) {
            cluster_fused_points[i] = fused_points[cluster_fused_point_idxs[i]];
        }
        auto cluster_fused_path = JoinPaths(dense_cluster_path, FUSION_NAME);
        WriteBinaryPlyPoints(cluster_fused_path, cluster_fused_points, false, true);

        auto cluster_fused_vis_path = cluster_fused_path + ".vis";
        WritePointsVisibility(cluster_fused_vis_path, fused_points_visibility_clusters[cluster_idx]);
        /*{
            std::string file_name = "./cluster_mvs_" + std::to_string(reconstruction_idx) + '_' + std::to_string(cluster_idx) + ".obj";
            FILE *file = fopen(file_name.c_str(), "w+");

            Eigen::Vector3f color;
            color[0] = rand() / (float)RAND_MAX;
            color[1] = rand() / (float)RAND_MAX;
            color[2] = rand() / (float)RAND_MAX;

            std::size_t face_idx = 1;
            for (const auto &fused_point : cluster_fused_points) {
                fprintf(file, "v %f %f %f %f %f %f\n", fused_point.x, fused_point.y, fused_point.z, color[0], color[1], color[2]);
                face_idx++;
            }

            Eigen::Vector3f C, TL, TR, BR, BL;
            for (auto image_id : cluster_reconstruction->RegisterImageIds()) {
                const Image &image = cluster_reconstruction->Image(image_id);
                const Camera &camera = cluster_reconstruction->Camera(image.CameraId());
                GetCameraFrustum(C, TL, TR, BR, BL, image, camera, 1.0f);
                fprintf(file, "v %f %f %f %f %f %f\n", C[0], C[1], C[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", TL[0], TL[1], TL[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", TR[0], TR[1], TR[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", BR[0], BR[1], BR[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", BL[0], BL[1], BL[2], color[0], color[1], color[2]);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 2, face_idx + 1);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 3, face_idx + 2);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 4, face_idx + 3);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 1, face_idx + 4);
                fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 2, face_idx + 3);
                fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 3, face_idx + 4);
                face_idx += 5;
            }

            fclose(file);
        }*/
    }
  }
}

std::shared_ptr<ReconstructionManager>
  CameraCluster::Cluster(std::shared_ptr<Reconstruction> reconstruction) {
    if (!reconstruction) {
        return NULL;
    }

    std::cout << reconstruction->NumRegisterImages() << " registered images" << std::endl;
    std::cout << reconstruction->NumCameras() << " cameras" << std::endl;
    std::cout << reconstruction->NumMapPoints() << " map points" << std::endl;

    const auto &register_image_ids = reconstruction->RegisterImageIds();
    std::vector<image_t> unclustered_image_ids = register_image_ids;

    std::unordered_map<mappoint_t, int> mappoint_visited_map;
    for (const auto &mappoint : reconstruction->MapPoints()) {
        mappoint_visited_map[mappoint.first] = -1;
    }
    mappoint_t num_clustered_mappoints = 0;

    std::shared_ptr<ReconstructionManager> manager = std::make_shared<ReconstructionManager>();
    std::size_t cluster_idx = 0;
    while (!unclustered_image_ids.empty() && num_clustered_mappoints < reconstruction->NumMapPoints()) {
        std::size_t num_unclustered_images = unclustered_image_ids.size();
        std::vector<std::pair<point2D_t, image_t> > unclustered_images;
        unclustered_images.reserve(num_unclustered_images);
        for (auto image_id: unclustered_image_ids) {
            const auto &image = reconstruction->Image(image_id);
            std::pair<point2D_t, image_t> unclustered_image(0, image_id);
            for (const auto &point_2d : image.Points2D()) {
                if (point_2d.HasMapPoint() && mappoint_visited_map[point_2d.MapPointId()] == -1) {
                    unclustered_image.first++;
                }
            }
            unclustered_images.push_back(unclustered_image);
        }
        std::sort(unclustered_images.begin(), unclustered_images.end(), std::greater<std::pair<point2D_t, image_t> >());

        std::unordered_map<image_t, std::size_t> image_idx_map;
        for (std::size_t i = 0; i < num_unclustered_images; i++) {
            image_idx_map[unclustered_images[i].second] = i;
        }

        std::vector<std::unordered_map<std::size_t, mappoint_t> > common_mappoints(num_unclustered_images);
        for (const auto &mappoint : reconstruction->MapPoints()) {
            if (mappoint_visited_map[mappoint.first] != -1) {
                continue;
            }

            const auto &track = mappoint.second.Track();
            const auto &elements = track.Elements();
            std::size_t num_elements = elements.size();
            for (std::size_t i = 0; i < num_elements; i++) {
                auto it = image_idx_map.find(elements[i].image_id);
                if (it == image_idx_map.end()) {
                    continue;
                }

                auto image_idx1 = it->second;
                for (std::size_t j = i + 1; j < num_elements; j++) {
                    auto it = image_idx_map.find(elements[j].image_id);
                    if (it == image_idx_map.end()) {
                        continue;
                    }

                    auto image_idx2 = it->second;
                    common_mappoints[image_idx1][image_idx2]++;
                    common_mappoints[image_idx2][image_idx1]++;
                }
            }
        }

        std::vector<unsigned char> image_visited_map(num_unclustered_images, 0);
        mappoint_t num_visited_mappoints = 0;

        EIGEN_STL_UMAP(image_t, Image) cluster_images;
        EIGEN_STL_UMAP(camera_t, Camera) cluster_cameras;
        EIGEN_STL_UMAP(mappoint_t, MapPoint) cluster_mappoints;
        for (std::size_t i = 0; i < num_unclustered_images && num_visited_mappoints < options_.min_mappts_per_cluster; i++) {
            if (image_visited_map[i]) {
                continue;
            }

            std::queue<std::size_t> Q;
            Q.push(i);
            image_visited_map[i] = 255;

            while (!Q.empty() && num_visited_mappoints < options_.max_mappts_per_cluster) {
                std::size_t image_idx = Q.front();
                Q.pop();

                image_t image_id = unclustered_images[image_idx].second;
                const auto &image = reconstruction->Image(image_id);
                cluster_images[image_id] = image;

                const auto &camera = reconstruction->Camera(image.CameraId());
                if (cluster_cameras.find(image.CameraId()) == cluster_cameras.end()) {
                    cluster_cameras[image.CameraId()] = camera;
                }

                for (const auto &point_2d : image.Points2D()) {
                    if (!point_2d.HasMapPoint()) {
                        continue;
                    }

                    const auto &mappoint = reconstruction->MapPoint(point_2d.MapPointId());
                    cluster_mappoints[point_2d.MapPointId()] = mappoint;

                    auto it = mappoint_visited_map.find(point_2d.MapPointId());
                    if (it->second == -1) {
                        num_visited_mappoints++;
                        it->second = cluster_idx;
                    }
                }

                std::priority_queue<std::pair<mappoint_t, std::size_t> > H;
                for (const auto &common_image : common_mappoints[image_idx]) {
                    H.emplace(common_image.second, common_image.first);
                }
                while (!H.empty()) {
                    std::size_t image_idx2 = H.top().second;
                    H.pop();
                    if (image_visited_map[image_idx2]) {
                        continue;
                    }

                    Q.push(image_idx2);
                    image_visited_map[image_idx2] = 255;
                }
            }
        }

        manager->Add();
        std::shared_ptr<Reconstruction> cluster_reconstruction = manager->Get(cluster_idx);
        std::cout << "Cluster: " << cluster_idx << std::endl;

        for (auto &image : cluster_images) {
            Image cluster_image;

            cluster_image.SetImageId(image.second.ImageId());
            cluster_image.SetName(image.second.Name());
            cluster_image.SetCameraId(image.second.CameraId());
            cluster_image.SetLabelId(image.second.LabelId());
            cluster_image.SetRegistered(false);

            cluster_image.SetPoseFlag(image.second.HasPose());
            cluster_image.SetQvec(image.second.Qvec());
            cluster_image.SetTvec(image.second.Tvec());
            cluster_image.SetQvecPrior(image.second.QvecPrior());
            cluster_image.SetTvecPrior(image.second.TvecPrior());

            auto points_2d = image.second.Points2D();
            for (auto &point_2d : points_2d) {
                point_2d.SetMask(false);
                point_2d.SetMapPointId(kInvalidMapPointId);
            }
            cluster_image.SetPoints2D(points_2d);

            cluster_reconstruction->AddImage(cluster_image);
            cluster_reconstruction->RegisterImage(cluster_image.ImageId());
        }
        std::cout << cluster_reconstruction->NumRegisterImages() << " registered images clustered" << std::endl;

        for (const auto &camera : cluster_cameras) {
            cluster_reconstruction->AddCamera(camera.second);
        }
        std::cout << cluster_reconstruction->NumCameras() << " cameras clustered" << std::endl;

        for (auto &mappoint : cluster_mappoints) {
            const auto &elements = mappoint.second.Track().Elements();
            Track track;
            for (const auto &element : elements) {
                if (cluster_images.find(element.image_id) != cluster_images.end()) {
                    track.AddElement(element);
                }
            }

            cluster_reconstruction->AddMapPoint(mappoint.second.XYZ(), std::move(track), mappoint.second.Color());
        }
        std::cout << cluster_reconstruction->NumMapPoints() << " map points clustered" << std::endl;
        std::cout << num_visited_mappoints << " visited map points" << std::endl;

        for (auto it = unclustered_image_ids.begin(); it != unclustered_image_ids.end(); ) {
            if (cluster_images.find(*it) != cluster_images.end()) {
                it = unclustered_image_ids.erase(it);
            } else {
                it++;
            }
        }
        num_clustered_mappoints += num_visited_mappoints;

        cluster_idx++;
    }
    std::size_t cluster_num = cluster_idx;

    /*{
        std::string file_name = "./clustered_sfm.obj";
        FILE *file = fopen(file_name.c_str(), "w+");

        std::vector<Eigen::Vector3f> colors(cluster_num);
        for (std::size_t cluster_idx = 0; cluster_idx < cluster_num; cluster_idx++) {
            auto &color = colors[cluster_idx];
            color[0] = rand() / (float)RAND_MAX;
            color[1] = rand() / (float)RAND_MAX;
            color[2] = rand() / (float)RAND_MAX;
        }

        std::size_t face_idx = 1;
        for (const auto &visited_mappoint : mappoint_visited_map) {
            if (visited_mappoint.second == -1) {
                continue;
            }

            const auto &mappoint = reconstruction->MapPoint(visited_mappoint.first);
            const auto &XYZ = mappoint.XYZ();
            const auto &color = colors[visited_mappoint.second];
            fprintf(file, "v %lf %lf %lf %f %f %f\n", XYZ[0], XYZ[1], XYZ[2], color[0], color[1], color[2]);
            face_idx++;
        }

        Eigen::Vector3f C, TL, TR, BR, BL;
        for (std::size_t cluster_idx = 0; cluster_idx < manager->Size(); cluster_idx++) {
            const auto cluster_reconstruction = manager->Get(cluster_idx);
            const auto &color = colors[cluster_idx];
            for (auto image_id : cluster_reconstruction->RegisterImageIds()) {
                const Image &image = cluster_reconstruction->Image(image_id);
                const Camera &camera = cluster_reconstruction->Camera(image.CameraId());
                GetCameraFrustum(C, TL, TR, BR, BL, image, camera, 1.0f);
                fprintf(file, "v %f %f %f %f %f %f\n", C[0], C[1], C[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", TL[0], TL[1], TL[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", TR[0], TR[1], TR[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", BR[0], BR[1], BR[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", BL[0], BL[1], BL[2], color[0], color[1], color[2]);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 2, face_idx + 1);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 3, face_idx + 2);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 4, face_idx + 3);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 1, face_idx + 4);
                fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 2, face_idx + 3);
                fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 3, face_idx + 4);
                face_idx += 5;
            }
        }
        fclose(file);
    }*/
    return manager;
}

std::shared_ptr<ReconstructionManager>
  CameraCluster::Cluster(std::vector<std::vector<std::size_t> > &fused_point_idxs_clusters,
                         std::vector<std::vector<std::vector<uint32_t> > > &fused_points_visibility_clusters,
                         std::shared_ptr<Reconstruction> reconstruction,
                         const std::vector<PlyPoint> &fused_points,
                         const std::vector<std::vector<uint32_t> > &fused_points_visibility) {
    if (!reconstruction) {
        return NULL;
    }

    if (fused_points.size() != fused_points_visibility.size()) {
        return NULL;
    }

    std::cout << reconstruction->NumRegisterImages() << " registered images" << std::endl;
    std::cout << reconstruction->NumCameras() << " cameras" << std::endl;
    std::cout << reconstruction->NumMapPoints() << " map points" << std::endl;

    const auto &register_image_ids = reconstruction->RegisterImageIds();
    std::vector<image_t> unclustered_image_ids = register_image_ids;

    std::size_t num_fused_points = fused_points.size();
    std::cout << num_fused_points << " fused points" << std::endl;
    std::vector<int> fused_point_visited_map(num_fused_points, -1);
    std::size_t num_clustered_fused_points = 0;
    std::unordered_map<image_t, std::vector<std::size_t> > image_fused_points_map;
    for (std::size_t i = 0; i < num_fused_points; i++) {
        for (uint32_t image_idx : fused_points_visibility[i]) {
            auto image_id = register_image_ids[image_idx];
            image_fused_points_map[image_id].push_back(i);
        }
    }

    std::shared_ptr<ReconstructionManager> manager = std::make_shared<ReconstructionManager>();
    std::vector<std::vector<std::size_t> >().swap(fused_point_idxs_clusters);
    std::vector<std::vector<std::vector<uint32_t> > >().swap(fused_points_visibility_clusters);
    std::vector<int> fused_point_clustered_map(num_fused_points);
    std::size_t cluster_idx = 0;
    while (!unclustered_image_ids.empty() && num_clustered_fused_points < num_fused_points) {
        std::size_t num_unclustered_images = unclustered_image_ids.size();
        std::vector<std::pair<std::size_t, image_t> > unclustered_images;
        unclustered_images.reserve(num_unclustered_images);
        for (auto image_id: unclustered_image_ids) {
            const auto &image = reconstruction->Image(image_id);
            std::pair<std::size_t, image_t> unclustered_image(0, image_id);
            for (auto fused_point_idx : image_fused_points_map[image_id]) {
                if (fused_point_visited_map[fused_point_idx] == -1) {
                    unclustered_image.first++;
                }
            }
            unclustered_images.push_back(unclustered_image);
        }
        std::sort(unclustered_images.begin(), unclustered_images.end(), std::greater<std::pair<std::size_t, image_t> >());

        std::unordered_map<image_t, std::size_t> image_idx_map;
        for (std::size_t i = 0; i < num_unclustered_images; i++) {
            image_idx_map[unclustered_images[i].second] = i;
        }

        std::vector<std::unordered_map<std::size_t, std::size_t> > common_fused_points(num_unclustered_images);
        for (std::size_t i = 0; i < num_fused_points; i++) {
            if (fused_point_visited_map[i] != -1) {
                continue;
            }

            const auto &fused_point_visibility = fused_points_visibility[i];
            std::size_t num_visibilities = fused_point_visibility.size();
            for (std::size_t i = 0; i < num_visibilities; i++) {
                auto image_id = register_image_ids[fused_point_visibility[i]];
                auto it = image_idx_map.find(image_id);
                if (it == image_idx_map.end()) {
                    continue;
                }

                auto image_idx1 = it->second;
                for (std::size_t j = i + 1; j < num_visibilities; j++) {
                    auto image_id = register_image_ids[fused_point_visibility[j]];
                    auto it = image_idx_map.find(image_id);
                    if (it == image_idx_map.end()) {
                        continue;
                    }

                    auto image_idx2 = it->second;
                    common_fused_points[image_idx1][image_idx2]++;
                    common_fused_points[image_idx2][image_idx1]++;
                }
            }
        }

        std::vector<unsigned char> image_visited_map(num_unclustered_images, 0);
        mappoint_t num_visited_fused_points = 0;

        EIGEN_STL_UMAP(image_t, Image) cluster_images;
        EIGEN_STL_UMAP(camera_t, Camera) cluster_cameras;
        EIGEN_STL_UMAP(mappoint_t, MapPoint) cluster_mappoints;
        memset(fused_point_clustered_map.data(), 0, num_fused_points * sizeof(int));

        for (std::size_t i = 0; i < num_unclustered_images && num_visited_fused_points < options_.min_pts_per_cluster; i++) {
            if (image_visited_map[i]) {
                continue;
            }

            std::queue<std::size_t> Q;
            Q.push(i);
            image_visited_map[i] = 255;

            while (!Q.empty() && num_visited_fused_points < options_.max_pts_per_cluster) {
                std::size_t image_idx = Q.front();
                Q.pop();

                image_t image_id = unclustered_images[image_idx].second;
                const auto &image = reconstruction->Image(image_id);
                cluster_images[image_id] = image;

                const auto &camera = reconstruction->Camera(image.CameraId());
                if (cluster_cameras.find(image.CameraId()) == cluster_cameras.end()) {
                    cluster_cameras[image.CameraId()] = camera;
                }

                for (const auto &point_2d : image.Points2D()) {
                    if (!point_2d.HasMapPoint()) {
                        continue;
                    }

                    const auto &mappoint = reconstruction->MapPoint(point_2d.MapPointId());
                    cluster_mappoints[point_2d.MapPointId()] = mappoint;
                }

                for (auto fused_point_idx : image_fused_points_map[image_id]) {
                    auto &fused_point_clustered = fused_point_clustered_map[fused_point_idx];
                    fused_point_clustered++;
                    if (fused_point_clustered >= options_.min_num_pixels) {
                        auto &fused_point_visited = fused_point_visited_map[fused_point_idx];
                        if (fused_point_visited == -1) {
                            num_visited_fused_points++;
                            fused_point_visited = cluster_idx;
                        }
                    }
                }

                std::priority_queue<std::pair<std::size_t, std::size_t> > H;
                for (const auto &common_image : common_fused_points[image_idx]) {
                    H.emplace(common_image.second, common_image.first);
                }
                while (!H.empty()) {
                    std::size_t image_idx2 = H.top().second;
                    H.pop();
                    if (image_visited_map[image_idx2]) {
                        continue;
                    }

                    Q.push(image_idx2);
                    image_visited_map[image_idx2] = 255;
                }
            }
        }

        manager->Add();
        std::shared_ptr<Reconstruction> cluster_reconstruction = manager->Get(cluster_idx);
        std::cout << "Cluster: " << cluster_idx << std::endl;

        image_idx_map.clear();
        for (auto &image : cluster_images) {
            Image cluster_image;

            cluster_image.SetImageId(image.second.ImageId());
            cluster_image.SetName(image.second.Name());
            cluster_image.SetCameraId(image.second.CameraId());
            cluster_image.SetLabelId(image.second.LabelId());
            cluster_image.SetRegistered(false);

            cluster_image.SetPoseFlag(image.second.HasPose());
            cluster_image.SetQvec(image.second.Qvec());
            cluster_image.SetTvec(image.second.Tvec());
            cluster_image.SetQvecPrior(image.second.QvecPrior());
            cluster_image.SetTvecPrior(image.second.TvecPrior());

            auto points_2d = image.second.Points2D();
            for (auto &point_2d : points_2d) {
                point_2d.SetMask(false);
                point_2d.SetMapPointId(kInvalidMapPointId);
            }
            cluster_image.SetPoints2D(points_2d);

            cluster_reconstruction->AddImage(cluster_image);
            cluster_reconstruction->RegisterImage(cluster_image.ImageId());

            image_idx_map[image.first] = cluster_reconstruction->NumRegisterImages();
        }
        std::cout << cluster_reconstruction->NumRegisterImages() << " registered images clustered" << std::endl;

        for (const auto &camera : cluster_cameras) {
            cluster_reconstruction->AddCamera(camera.second);
        }
        std::cout << cluster_reconstruction->NumCameras() << " cameras clustered" << std::endl;

        for (auto &mappoint : cluster_mappoints) {
            const auto &elements = mappoint.second.Track().Elements();
            Track track;
            for (const auto &element : elements) {
                if (cluster_images.find(element.image_id) != cluster_images.end()) {
                    track.AddElement(element);
                }
            }

            cluster_reconstruction->AddMapPoint(mappoint.second.XYZ(), std::move(track), mappoint.second.Color());
        }
        std::cout << cluster_reconstruction->NumMapPoints() << " map points clustered" << std::endl;

        std::cout << num_visited_fused_points << " visited fused points" << std::endl;

        fused_point_idxs_clusters.push_back(std::vector<std::size_t>());
        auto &cluster_fused_point_idxs = fused_point_idxs_clusters[cluster_idx];
        for (std::size_t i = 0; i < num_fused_points; i++) {
            auto &fused_point_clustered = fused_point_clustered_map[i];
            if (fused_point_clustered >= options_.min_num_pixels) {
                fused_point_clustered = cluster_fused_point_idxs.size();
                cluster_fused_point_idxs.push_back(i);
            } else {
                fused_point_clustered = -1;
            }
        }
        std::size_t num_cluster_fused_points = cluster_fused_point_idxs.size();
        std::cout << num_cluster_fused_points << " fused points clustered" << std::endl;

        fused_points_visibility_clusters.push_back(std::vector<std::vector<uint32_t> >(num_cluster_fused_points));
        auto &cluster_fused_points_visibility = fused_points_visibility_clusters[cluster_idx];
        for (const auto &image : cluster_images) {
            auto image_id = image.first;
            auto image_idx = image_idx_map[image_id];
            for (auto fused_point_idx : image_fused_points_map[image_id]) {
                const auto &fused_point_clustered = fused_point_clustered_map[fused_point_idx];
                if (fused_point_clustered == -1) {
                    continue;
                }

                cluster_fused_points_visibility[fused_point_clustered].push_back(image_idx);
            }
        }

        for (auto it = unclustered_image_ids.begin(); it != unclustered_image_ids.end(); ) {
            if (cluster_images.find(*it) != cluster_images.end()) {
                it = unclustered_image_ids.erase(it);
            } else {
                it++;
            }
        }
        num_clustered_fused_points += num_visited_fused_points;

        cluster_idx++;
    }
    std::size_t cluster_num = cluster_idx;

    /*{
        std::string file_name = "./clustered_mvs.obj";
        FILE *file = fopen(file_name.c_str(), "w+");

        std::vector<Eigen::Vector3f> colors(cluster_num);
        for (std::size_t cluster_idx = 0; cluster_idx < cluster_num; cluster_idx++) {
            auto &color = colors[cluster_idx];
            color[0] = rand() / (float)RAND_MAX;
            color[1] = rand() / (float)RAND_MAX;
            color[2] = rand() / (float)RAND_MAX;
        }

        std::size_t face_idx = 1;
        for (std::size_t i = 0; i < num_fused_points; i++) {
            if (fused_point_visited_map[i] == -1) {
                continue;
            }

            const auto &fused_point = fused_points[i];
            const auto &color = colors[fused_point_visited_map[i]];
            fprintf(file, "v %f %f %f %f %f %f\n", fused_point.x, fused_point.y, fused_point.z, color[0], color[1], color[2]);
            face_idx++;
        }

        Eigen::Vector3f C, TL, TR, BR, BL;
        for (std::size_t cluster_idx = 0; cluster_idx < manager->Size(); cluster_idx++) {
            const auto cluster_reconstruction = manager->Get(cluster_idx);
            const auto &color = colors[cluster_idx];
            for (auto image_id : cluster_reconstruction->RegisterImageIds()) {
                const Image &image = cluster_reconstruction->Image(image_id);
                const Camera &camera = cluster_reconstruction->Camera(image.CameraId());
                GetCameraFrustum(C, TL, TR, BR, BL, image, camera, 1.0f);
                fprintf(file, "v %f %f %f %f %f %f\n", C[0], C[1], C[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", TL[0], TL[1], TL[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", TR[0], TR[1], TR[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", BR[0], BR[1], BR[2], color[0], color[1], color[2]);
                fprintf(file, "v %f %f %f %f %f %f\n", BL[0], BL[1], BL[2], color[0], color[1], color[2]);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 2, face_idx + 1);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 3, face_idx + 2);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 4, face_idx + 3);
                fprintf(file, "f %ld %ld %ld\n", face_idx, face_idx + 1, face_idx + 4);
                fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 2, face_idx + 3);
                fprintf(file, "f %ld %ld %ld\n", face_idx + 1, face_idx + 3, face_idx + 4);
                face_idx += 5;
            }
        }
        fclose(file);
    }*/
    return manager;
}

}  // namespace mvs
}  // namespace sensemap
