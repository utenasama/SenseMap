// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "optim/cluster_merge/cluster_merge_optimizer.h"

#include "base/cost_functions.h"
#include "base/matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/similarity_transform.h"

#include "optim/global_motions/utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

#include <estimators/absolute_pose.h>
#include <fstream>
#include "estimators/generalized_absolute_pose.h"
#include "estimators/triangulation.h"
#include "optim/ransac/loransac.h"
#include "util/proc.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define Test_enable

namespace sensemap {

ClusterMergeOptimizer::ClusterMergeOptimizer(std::shared_ptr<ClusterMergeOptions> options,
                                             const CorrespondenceGraph* correspondence_graph)
    : options_(std::move(options)), full_correspondence_graph_(correspondence_graph) {}

void ClusterMergeOptimizer::WriteMergeResult(const std::string& filename) {
    std::cout << "Write Merge Result" << std::endl;
    reconstruction_ = std::make_shared<Reconstruction>();

    *(reconstruction_.get()) = *(reconstruction_manager_->Get(0).get());
    Eigen::Matrix3x4d transform;
    Eigen::Vector4d qvec(1, 0, 0, 0);
    Eigen::Vector3d tvec(0, 0, 0);
    transform.leftCols<3>() = QuaternionToRotationMatrix(qvec);
    transform.rightCols<1>() = tvec;
    for (cluster_t index = 1; index < reconstruction_manager_->Size(); index++) {
        const auto& cur_reconstruction = reconstruction_manager_->Get(index);
        reconstruction_->Merge(*cur_reconstruction.get(), transform, 8.0);
    }

    std::string rec_path = filename;

    if (boost::filesystem::exists(rec_path)) {
        boost::filesystem::remove_all(rec_path);
    }
    boost::filesystem::create_directories(rec_path);
    reconstruction_->WriteReconstruction(rec_path, true);
}

void ClusterMergeOptimizer::WriteReconstructionResult(const std::string& filename) {
    std::cout << "Write Reconstruction Result" << std::endl;

    std::string rec_path = filename;

    if (boost::filesystem::exists(rec_path)) {
        boost::filesystem::remove_all(rec_path);
    }
    boost::filesystem::create_directories(rec_path);
    reconstruction_->WriteReconstruction(rec_path, true);
}

void ClusterMergeOptimizer::FindImageNeighbor() {
    // Find all the image neighbor for all the reconstruction
    std::cout << "\n[ClusterMergeOptimizer] Find Image Neighbor" << std::endl;
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    for (int i = 0; i < reconsturction_ids.size(); i++) {
        cluster_t index = reconsturction_ids[i];

        std::cout << "get reconstruction " << index << std::endl;
        const auto& cur_reconstruction = reconstruction_manager_->Get(index);
        std::cout << "got reconstruction " << index << std::endl;
        const auto reconstruct_ids = cur_reconstruction->RegisterImageIds();
        std::cout << "reconstruction image_ids.size() " << reconstruct_ids.size() << std::endl;

        const auto& image_pairs = cur_reconstruction->ImagePairs();

        std::cout << "reconstruction image_pairs.size() " << image_pairs.size() << std::endl;
        cur_reconstruction->ComputeBaselineDistance();
        double baseline_distance = cur_reconstruction->baseline_distance;
        std::cout << "Average neighbor baseline distance :" << baseline_distance << std::endl;

        std::vector<std::pair<image_pair_t, size_t>> neighbors;

        for (const auto& image_pair : image_pairs) {
            image_t image_id1, image_id2;
            sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
            if (image_id1 < image_id2) {
                std::swap(image_id1, image_id2);
            }

            if (!cur_reconstruction->ExistsImage(image_id1)) {
                std::cout << image_id1 << " not exist " << std::endl;
                continue;
            }
            if (!cur_reconstruction->ExistsImage(image_id2)) {
                std::cout << image_id2 << " not exist " << std::endl;
                continue;
            }

            class Image& image1 = cur_reconstruction->Image(image_id1);
            class Image& image2 = cur_reconstruction->Image(image_id2);

            class Camera& camera1 = cur_reconstruction->Camera(image1.CameraId());
            class Camera& camera2 = cur_reconstruction->Camera(image2.CameraId());

            if (!image1.IsRegistered() || !image2.IsRegistered()) {
                continue;
            }

            if (cur_reconstruction->IsolatedImage(3, image_id1, baseline_distance) ||
                cur_reconstruction->IsolatedImage(3, image_id2, baseline_distance)) {
                std::cout << "Isolated image: " << image1.Name() << " or " << image2.Name() << std::endl;
                continue;
            }

            const auto& corrs = full_correspondence_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);

            if (corrs.size() < options_->NeighborCorrespondece) {
                continue;
            }

            size_t consistent_corr_count = 0;
            for (const auto& corr : corrs) {
                const class Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
                const class Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);

                if (point2D1.HasMapPoint() && point2D2.HasMapPoint() &&
                    (point2D1.MapPointId() == point2D2.MapPointId())) {
                    consistent_corr_count++;
                }
            }

            bool consistent_neighbor = false;
            if (image1.LabelId() == image2.LabelId()) {
                Eigen::Vector3d current_baseline = image1.ProjectionCenter() - image2.ProjectionCenter();
                image_t id_difference = image_id1 > image_id2 ? (image_id1 - image_id2) : (image_id2 - image_id1);

                if (id_difference < options_->NeighborNumberThreshold / 2 &&
                    current_baseline.norm() < options_->neighbor_distance_factor_wrt_averge_baseline *
                                                  static_cast<double>(id_difference) * baseline_distance &&
                    consistent_corr_count > 0) {
                    consistent_neighbor = true;
                }
            }

            if (consistent_corr_count < options_->normal_edge_min_common_points && !consistent_neighbor) {
                continue;
            }

            neighbors.emplace_back(std::make_pair(image_pair.first, consistent_corr_count));
        }
        std::cout << "normal_edge of reconstruction " << index << " is " << neighbors.size() << std::endl;
        normal_edge_candidate_image_neighbor_[index] = neighbors;

        std::sort(neighbors.begin(), neighbors.end(),
                  [&](const std::pair<image_pair_t, size_t>& e1, const std::pair<image_pair_t, size_t>& e2) {
                      return e1.second > e2.second;
                  });

        for (const auto& neighbor : neighbors) {
            image_t image_id1, image_id2;
            sensemap::utility::PairIdToImagePair(neighbor.first, &image_id1, &image_id2);
            if (image_id1 < image_id2) {
                std::swap(image_id1, image_id2);
            }
            if (!normal_edge_image_neighbor_[index][image_id1].count(image_id2) &&
                normal_edge_image_neighbor_[index][image_id1].size() < options_->NeighborNumberThreshold) {
                normal_edge_image_neighbor_[index][image_id1].insert(image_id2);
                class Image& image1 = cur_reconstruction->Image(image_id1);
                class Image& image2 = cur_reconstruction->Image(image_id2);
                point2D_t num_correspondence =
                    full_correspondence_graph_->NumCorrespondencesBetweenImages(image_id1, image_id2);
            }
        }
    }
}

size_t ClusterMergeOptimizer::MergeTracks(const mappoint_t mappoint_id) {
    if (!reconstruction_->ExistsMapPoint(mappoint_id)) {
        return 0;
    }

    const double max_squared_reproj_error = options_->merge_max_reproj_error * options_->merge_max_reproj_error;

    const auto& mappoint = reconstruction_->MapPoint(mappoint_id);

    for (const auto& track_el : mappoint.Track().Elements()) {
        const std::vector<CorrespondenceGraph::Correspondence>& corrs =
            full_correspondence_graph_->FindCorrespondences(track_el.image_id, track_el.point2D_idx);

        for (const auto corr : corrs) {
            if (!reconstruction_->ExistsImage(corr.image_id)) {
                continue;
            }
            const auto& image = reconstruction_->Image(corr.image_id);
            if (!image.IsRegistered()) {
                continue;
            }
            if (corr.point2D_idx >= image.Points2D().size()) {
                continue;
            }
            const Point2D& corr_point2D = image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint() || corr_point2D.MapPointId() == mappoint_id ||
                merge_trials_[mappoint_id].count(corr_point2D.MapPointId()) > 0) {
                continue;
            }
            // Try to merge the two Map Points.
            if (!reconstruction_->ExistsMapPoint(corr_point2D.MapPointId())) {
                std::cout << " Reconstruction do not have this map point " << std::endl;
                continue;
            }

            const MapPoint& corr_mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());

            merge_trials_[mappoint_id].insert(corr_point2D.MapPointId());
            merge_trials_[corr_point2D.MapPointId()].insert(mappoint_id);

            // Weighted average of point locations, depending on track length.
            const Eigen::Vector3d merged_xyz =
                (mappoint.Track().Length() * mappoint.XYZ() + corr_mappoint.Track().Length() * corr_mappoint.XYZ()) /
                (mappoint.Track().Length() + corr_mappoint.Track().Length());

            // Count number of inlier track elements of the merged track.
            bool merge_success = true;
            for (const Track* track : {&mappoint.Track(), &corr_mappoint.Track()}) {
                for (const auto test_track_el : track->Elements()) {
                    const Image& test_image = reconstruction_->Image(test_track_el.image_id);
                    const Camera& test_camera = reconstruction_->Camera(test_image.CameraId());
                    const Point2D& test_point2D = test_image.Point2D(test_track_el.point2D_idx);

                    if (test_camera.NumLocalCameras() > 1) {
                        uint32_t local_camera_id = test_image.LocalImageIndices()[test_track_el.point2D_idx];

                        Eigen::Vector4d local_qvec;
                        Eigen::Vector3d local_tvec;

                        test_camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

                        Eigen::Matrix3d global_R =
                            QuaternionToRotationMatrix(local_qvec) * QuaternionToRotationMatrix(test_image.Qvec());

                        Eigen::Vector3d global_T =
                            local_tvec + QuaternionToRotationMatrix(local_qvec) * test_image.Tvec();

                        if (CalculateSquaredReprojectionErrorRig(
                                test_point2D.XY(), merged_xyz, RotationMatrixToQuaternion(global_R), global_T,
                                local_camera_id, test_camera) > max_squared_reproj_error) {
                            merge_success = false;
                            break;
                        }
                    } else {
                        if (CalculateSquaredReprojectionError(test_point2D.XY(), merged_xyz, test_image.Qvec(),
                                                              test_image.Tvec(),
                                                              test_camera) > max_squared_reproj_error) {
                            merge_success = false;
                            break;
                        }
                    }
                }
                if (!merge_success) {
                    break;
                }
            }

            // Only accept merge if all track elements are inliers.
            if (merge_success) {
                const size_t num_merged = mappoint.Track().Length() + corr_mappoint.Track().Length();
                if (!reconstruction_->ExistsMapPoint(mappoint_id)) {
                    std::cout << "Map point not exisit ??? " << mappoint_id << std::endl;
                    break;
                }
                if (!reconstruction_->ExistsMapPoint(corr_point2D.MapPointId())) {
                    std::cout << "Map point not exisit ??? " << corr_point2D.MapPointId() << std::endl;
                    break;
                }

                const mappoint_t merged_mappoint_id =
                    reconstruction_->MergeMapPoints(mappoint_id, corr_point2D.MapPointId());

                modified_mappoint_ids_.erase(mappoint_id);
                modified_mappoint_ids_.erase(corr_point2D.MapPointId());
                modified_mappoint_ids_.insert(merged_mappoint_id);

                // Merge merged Map Point and return,
                //   as the original points are deleted.
                const size_t num_merged_recursive = MergeTracks(merged_mappoint_id);

                if (num_merged_recursive > 0) {
                    return num_merged_recursive;
                } else {
                    return num_merged;
                }
            }
        }
    }

    return 0;
}

void ClusterMergeOptimizer::MergeReconstructionTracks() {
    std::cout << "Start Merge Tracks" << std::endl;
    merge_trials_.clear();

    size_t num_merged = 0;

    for (auto mappoint_id : reconstruction_->MapPointIds()) {
        num_merged += MergeTracks(mappoint_id);
    }

    std::cout << "Merge " << num_merged << " Map points" << std::endl;
}

size_t ClusterMergeOptimizer::Triangulate(const mappoint_t mappoint_id) {
    size_t num_tri = 0;
    if (!reconstruction_->ExistsMapPoint(mappoint_id)) {
        std::cout << " Do not exist map point ..." << std::endl;
        return num_tri;
    }

    class MapPoint& mappoint = reconstruction_->MapPoint(mappoint_id);
    class Track& track = mappoint.Track();
    if (track.Length() < 2) {
        // Need at least two observations for triangulation.
        return num_tri;
    } else if (track.Length() == 2) {
        const TrackElement& track_el = track.Element(0);
        if (full_correspondence_graph_->IsTwoViewObservation(track_el.image_id, track_el.point2D_idx)) {
            return num_tri;
        }
    }

    // Setup data for triangulation estimation.
    std::vector<TriangulationEstimator::PointData> point_data;
    point_data.resize(track.Length());
    std::vector<TriangulationEstimator::PoseData> pose_data;
    pose_data.resize(track.Length());
    for (size_t i = 0; i < track.Length(); ++i) {
        const TrackElement& track_el = track.Elements()[i];
        const Image& image = reconstruction_->Image(track_el.image_id);
        const Camera& camera = reconstruction_->Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);

        point_data[i].point = point2D.XY();
        point_data[i].point_normalized = camera.ImageToWorld(point_data[i].point);
        pose_data[i].proj_matrix = image.ProjectionMatrix();
        pose_data[i].proj_center = image.ProjectionCenter();
        pose_data[i].camera = &camera;

        if (camera.ModelName().compare("SPHERICAL") == 0) {
            point_data[i].point_bearing = camera.ImageToBearing(point_data[i].point);
        }

        // For camera rig the data are re-prepared
        if (camera.NumLocalCameras() > 1) {
            uint32_t local_camera_id = image.LocalImageIndices()[track_el.point2D_idx];

            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;

            camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

            Eigen::Matrix3d global_R =
                QuaternionToRotationMatrix(local_qvec) * QuaternionToRotationMatrix(image.Qvec());

            Eigen::Vector3d global_T = local_tvec + QuaternionToRotationMatrix(local_qvec) * image.Tvec();

            pose_data[i].proj_matrix = ComposeProjectionMatrix(global_R, global_T);
            pose_data[i].proj_center = -global_R.transpose() * global_T;

            point_data[i].point_normalized = camera.LocalImageToWorld(local_camera_id, point_data[i].point);
        }
    }

    // Setup estimation options.
    EstimateTriangulationOptions tri_options;
    tri_options.min_tri_angle = DegToRad(1.5);
    tri_options.residual_type = TriangulationEstimator::ResidualType::ANGULAR_ERROR;
    tri_options.ransac_options.max_error = DegToRad(2.0);  // DegToRad(5.0);
    tri_options.ransac_options.confidence = 0.9999;
    tri_options.ransac_options.min_inlier_ratio = 0.02;
    tri_options.ransac_options.max_num_trials = 10000;

    // Enforce exhaustive sampling for small track lengths.
    const size_t kExhaustiveSamplingThreshold = 15;
    if (point_data.size() <= kExhaustiveSamplingThreshold) {
        tri_options.ransac_options.min_num_trials = NChooseK(point_data.size(), 2);
    }

    // Estimate triangulation.
    Eigen::Vector3d xyz;
    std::vector<char> inlier_mask;
    if (!EstimateTriangulation(tri_options, point_data, pose_data, &inlier_mask, &xyz)) {
        return num_tri;
    }

    mappoint.SetXYZ(xyz);
    num_tri++;

    return num_tri;
}

void ClusterMergeOptimizer::TriangulateReconstruction() {
    size_t num_tri = 0;

    for (auto mappoint_id : reconstruction_->MapPointIds()) {
        num_tri += Triangulate(mappoint_id);
    }

    std::cout << "Triangulate " << num_tri << " Map points" << std::endl;
}

size_t ClusterMergeOptimizer::FindCorrespondences(const image_t image_id,
                                     const point2D_t point2D_idx,
                                     const size_t transitivity,
                                     std::vector<CorrData>* corrs_data) {

  std::vector<CorrespondenceGraph::Correspondence> corrs;
    //   full_correspondence_graph_->FindTransitiveCorrespondences(
    //       image_id, point2D_idx, transitivity);
  full_correspondence_graph_->FindTransitiveCorrespondences(image_id, point2D_idx, transitivity, &corrs);
    

  if (corrs.size() < 1){
    return 0;
  }
//   std::cout << "corrs size: " << corrs.size() << std::endl;
//   exit(-1);

  corrs_data->clear();
  corrs_data->reserve(corrs.size());

  size_t num_triangulated = 0;

  for (const CorrespondenceGraph::Correspondence corr : corrs) {
    const Image& corr_image = reconstruction_->Image(corr.image_id);
    if (!corr_image.IsRegistered()) {
      continue;
    }

    const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());
    // if (HasCameraBogusParams(options, corr_camera)) {
    //   continue;
    // }

    if (corr_image.Point2D(corr.point2D_idx).HasMapPoint()) {
      num_triangulated += 1;
      continue;
    }

    CorrData corr_data;
    corr_data.image_id = corr.image_id;
    corr_data.point2D_idx = corr.point2D_idx;
    corr_data.image = &corr_image;
    corr_data.camera = &corr_camera;
    corr_data.point2D = &corr_image.Point2D(corr.point2D_idx);

    corrs_data->push_back(std::move(corr_data));

  }

  return num_triangulated;
}


size_t ClusterMergeOptimizer::CompleteLoopPoint(const image_t image_id, const point2D_t point2D_idx) {
    size_t num_completed = 0;

    const size_t mappoint_id = reconstruction_->Image(image_id).Point2D(point2D_idx).MapPointId();
    if (!reconstruction_->ExistsMapPoint(mappoint_id)) {
        return num_completed;
    }

    const double max_squared_reproj_error = 4.0 * 4.0;

    const MapPoint& mappoint = reconstruction_->MapPoint(mappoint_id);

    std::vector<TrackElement> queue = mappoint.Track().Elements();

    const std::vector<CorrespondenceGraph::Correspondence>& corrs =
        full_correspondence_graph_->FindCorrespondences(image_id, point2D_idx);

    for (const auto corr : corrs) {
        if (image_neighbor_between_cluster_[image_id].find(corr.image_id) 
            == image_neighbor_between_cluster_[image_id].end()){
            continue;
        }
        const Image& image = reconstruction_->Image(corr.image_id);
        if (!image.IsRegistered()) {
            continue;
        }
        if (corr.image_id == image_id){
            continue;
        }

        const Point2D& point2D = image.Point2D(corr.point2D_idx);
        if (point2D.HasMapPoint()) {
            continue;
        }

        const Camera& camera = reconstruction_->Camera(image.CameraId());
        // if (camera.NumLocalCameras()==1 && HasCameraBogusParams(options, camera)) {
        //     continue;
        // }

        if(camera.NumLocalCameras()>1){
            if (corr.image_id != image_id) {
                uint32_t local_camera_id = 
                    image.LocalImageIndices()[corr.point2D_idx]; 
        
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;

                camera.GetLocalCameraExtrinsic(local_camera_id,
                                                    local_qvec,
                                                    local_tvec);

                Eigen::Matrix3d global_R = QuaternionToRotationMatrix(local_qvec)*
                        QuaternionToRotationMatrix(image.Qvec()); 

                Eigen::Vector3d global_T = local_tvec + 
                        QuaternionToRotationMatrix(local_qvec) *
                        image.Tvec();

                if (CalculateSquaredReprojectionErrorRig(
                    point2D.XY(), mappoint.XYZ(), RotationMatrixToQuaternion(global_R),
                    global_T,local_camera_id,camera)> max_squared_reproj_error){           
                    continue;
                }
            }
        }
        else{
            if (CalculateSquaredReprojectionError(
                point2D.XY(), mappoint.XYZ(), image.Qvec(), image.Tvec(),
                camera) > max_squared_reproj_error) {
                continue;
            }        
        }

        // Success, add observation to point track.
        const TrackElement track_el(corr.image_id, corr.point2D_idx);
        reconstruction_->AddObservation(mappoint_id, track_el);
        modified_mappoint_ids_.insert(mappoint_id);

        num_completed += 1;
    }
    
    return num_completed;
}

void ClusterMergeOptimizer::CompleteLoopImages() {
    std::cout << "[Pose Graph Optimizer] Start Complete Loop Images" << std::endl;

    std::set<image_t> loop_images_ids;
    size_t before_modified_size = modified_mappoint_ids_.size();
    for (const auto cluster_loop_image_ids : all_loop_image_ids_){
        std::set<image_t> loop_cluster_image_ids;
        for (const auto image_pair_id : cluster_loop_image_ids.second){
            image_t image_id1, image_id2;
            sensemap::utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);
            loop_cluster_image_ids.insert(image_id1);
            loop_cluster_image_ids.insert(image_id2);
        }
        size_t num_cluster_images = loop_cluster_image_ids.size();
        std::cout << "=> Cluster - " << cluster_loop_image_ids.first 
                  << ", pair size: " << cluster_loop_image_ids.second.size() 
                  << ", loop image size: " << num_cluster_images << std::endl;

        size_t interval = std::min( num_cluster_images / 10 + 1, (size_t)100);
        size_t num_insert = 0;
        auto CompleteImage = [&](const image_t image_id) {
            size_t num_tris = 0;
            const Image& image = reconstruction_->Image(image_id);
            if (!image.IsRegistered()) {
                return num_tris;
            }

            const Camera& camera = reconstruction_->Camera(image.CameraId());
            // Setup estimation options.
            EstimateTriangulationOptions tri_options;
            tri_options.min_tri_angle = DegToRad(3.0);
            tri_options.residual_type = TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
            tri_options.ransac_options.max_error = 12.0;
            tri_options.ransac_options.confidence = 0.9999;
            tri_options.ransac_options.min_inlier_ratio = 0.02;
            tri_options.ransac_options.max_num_trials = 10000;

            if (camera.NumLocalCameras() > 2) {
                tri_options.min_tri_angle = DegToRad(0.004);
            }

            if(camera.ModelName().compare("SPHERICAL")==0){
                tri_options.residual_type =
                        TriangulationEstimator::ResidualType::ANGULAR_ERROR;
                tri_options.ransac_options.max_error =
                        DegToRad(2.0);
            }
        
            // Correspondence data for reference observation in given image. We iterate
            // over all observations of the image and each observation once becomes
            // the reference correspondence.

            CorrData ref_corr_data;
            ref_corr_data.image_id = image_id;
            ref_corr_data.image = &image;
            ref_corr_data.camera = &camera;

            // Container for correspondences from reference observation to other images.
            std::vector<CorrData> corrs_data;
            // std::cout << "image.NumPoints2D(): " << image.NumPoints2D() << std::endl;
            std::size_t num_haspoint = 0, num_inmask = 0;
            for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
                const Point2D& point2D = image.Point2D(point2D_idx);
                // if (!point2D.InMask()) {
                //     num_inmask++;
                //     continue;
                // }
                if (point2D.HasMapPoint()) {
                    // Complete existing track.
                    // num_tris += Complete(options, point2D.MapPointId());
                    num_tris += CompleteLoopPoint(image_id, point2D_idx);
                    num_haspoint++;
                    continue;
                }

                const size_t num_triangulated = 
                    FindCorrespondences(image_id, point2D_idx, 1, &corrs_data);
                // if (num_triangulated || corrs_data.empty()) {
                if (corrs_data.empty()) {
                    continue;
                }
                // std::cout << "corrs_data.size(): " << corrs_data.size() << std::endl;
                ref_corr_data.point2D = &point2D;
                ref_corr_data.point2D_idx = point2D_idx;
                corrs_data.push_back(ref_corr_data);

                // Setup data for triangulation estimation.
                std::vector<TriangulationEstimator::PointData> point_data;
                point_data.resize(corrs_data.size());
                std::vector<TriangulationEstimator::PoseData> pose_data;
                pose_data.resize(corrs_data.size());
                for (size_t i = 0; i < corrs_data.size(); ++i) {
                    const CorrData& corr_data = corrs_data[i];
                    point_data[i].point = corr_data.point2D->XY();
                    point_data[i].point_normalized = corr_data.camera->ImageToWorld(point_data[i].point);
                    pose_data[i].proj_matrix = corr_data.image->ProjectionMatrix();
                    pose_data[i].proj_center = corr_data.image->ProjectionCenter();
                    pose_data[i].camera = corr_data.camera;

                    if(corr_data.camera->ModelName().compare("SPHERICAL")==0){
                        point_data[i].point_bearing = 
                            corr_data.camera->ImageToBearing(point_data[i].point);
                    }

                    // For camera rig the data are re-prepared
                    if (corr_data.camera->NumLocalCameras() > 1){

                        uint32_t local_camera_id =
                            corr_data.image->LocalImageIndices()[corr_data.point2D_idx];

                        Eigen::Vector4d local_qvec;
                        Eigen::Vector3d local_tvec;

                        corr_data.camera->GetLocalCameraExtrinsic(local_camera_id,
                                                                local_qvec, local_tvec);

                        Eigen::Matrix3d global_R = QuaternionToRotationMatrix(local_qvec) *
                                                QuaternionToRotationMatrix(corr_data.image->Qvec());

                        Eigen::Vector3d global_T = local_tvec +
                                                QuaternionToRotationMatrix(local_qvec) *
                                                    corr_data.image->Tvec();

                        pose_data[i].proj_matrix = ComposeProjectionMatrix(global_R, global_T);
                        pose_data[i].proj_center = -global_R.transpose() * global_T;

                        point_data[i].point_normalized =
                            corr_data.camera->LocalImageToWorld(local_camera_id,
                                                                point_data[i].point);
                    }
                }

                // // Enforce exhaustive sampling for small track lengths.
                // const size_t kExhaustiveSamplingThreshold = 15;
                // if (point_data.size() <= kExhaustiveSamplingThreshold) {
                //     tri_options.ransac_options.min_num_trials = NChooseK(point_data.size(), 2);
                // }

                // Estimate triangulation.
                Eigen::Vector3d xyz;
                std::vector<char> inlier_mask;
                if (!EstimateTriangulation(tri_options, point_data, pose_data, &inlier_mask, &xyz)) {
                    continue;
                }

                // Add inliers to estimated track.
                Track track;
                track.Reserve(corrs_data.size());
                for (size_t i = 0; i < inlier_mask.size(); ++i) {
                    if (inlier_mask[i]) {
                        const CorrData& corr_data = corrs_data[i];
                        if (reconstruction_->Image(corr_data.image_id).Point2D(corr_data.point2D_idx).HasMapPoint()){
                            continue;
                        }
                        track.AddElement(corr_data.image_id, corr_data.point2D_idx);
                        num_tris += 1;
                    }
                }
                {
                    const mappoint_t mappoint_id = reconstruction_->AddMapPoint(xyz, std::move(track));
                    modified_mappoint_ids_.insert(mappoint_id);
                    std::unique_lock<std::mutex> lock(add_mappoint_mutex_);
                }
            }
            if (num_insert % interval == 0){
                std::cout << "\r complete image " << num_insert << "/"
                            << num_cluster_images << std::flush;
            }
            num_insert++;
            // std::cout << "num_haspoint: " << num_haspoint << ", " << num_inmask << std::endl;

            // exit(-1);
            return num_tris;
        };

        const size_t num_eff_threads = std::min((size_t)GetEffectiveNumThreads(-1), loop_cluster_image_ids.size());
        thread_pool.reset(new ThreadPool(num_eff_threads));
        for(const auto image_id : loop_cluster_image_ids){
            thread_pool->AddTask(CompleteImage, image_id);
        }
        thread_pool->Wait();
    }
   
    std::cout << "\nComplete " << loop_images_ids.size() << " images , modified_mappoint_ids_ size: " << before_modified_size 
              << " -> "<< modified_mappoint_ids_.size() << std::endl; 
    return;
}

void ClusterMergeOptimizer::WritePoseGraphResult(const std::string& folder_name) {
    std::string rec_path = folder_name;

    if (boost::filesystem::exists(rec_path)) {
        boost::filesystem::remove_all(rec_path);
    }
    boost::filesystem::create_directories(rec_path);

    // Output the image poses
    std::string pose_file_path = rec_path + "/pose.txt";
    std::string tmp_pose_file_path = rec_path + "/pose_cluster.txt";

    std::ofstream pose_file(pose_file_path, std::ios::trunc);
    std::ofstream tmp_pose_file(tmp_pose_file_path, std::ios::trunc);
    CHECK(pose_file.is_open()) << pose_file_path;

    for (const auto& poses_within_cluster : poses_) {
        for (const auto& pose : poses_within_cluster.second) {
            // Inverse the pose
            auto tvec = pose.second.tvec;
            auto rot = pose.second.qvec.normalized().toRotationMatrix();

            Eigen::Vector4d qvec;

            // Inverse
            tvec = -rot.transpose() * tvec;
            qvec = RotationMatrixToQuaternion(rot.transpose());

            // Write the image id
            auto pose_id = (poses_within_cluster.first + 1) * 1e6 + pose.first;
            pose_file << std::setprecision(8) << pose_id << " ";

            // Write the position
            pose_file << tvec(0) << " " << tvec(1) << " " << tvec(2) << " ";

            tmp_pose_file << std::setprecision(8) << pose_id << " " << tvec(0) << " " << tvec(1) << " " << tvec(2)
                          << " " << poses_within_cluster.first << std::endl;

            // Write the rotation
            pose_file << qvec(1) << " " << qvec(2) << " " << qvec(3) << " " << qvec(0) << std::endl;
        }
    }
    pose_file.close();

    // Output the normal edges
    std::string normal_edge_file_path = rec_path + "/nolmal_edges.txt";
    std::string loop_edge_file_path = rec_path + "/loop_edges.txt";

    std::ofstream normal_edge_file(normal_edge_file_path, std::ios::trunc);
    std::ofstream loop_edge_file(loop_edge_file_path, std::ios::trunc);

    CHECK(normal_edge_file.is_open()) << pose_file_path;
    CHECK(loop_edge_file.is_open()) << pose_file_path;

    for (const auto& edge : edges_) {
        // Write the edge
        auto id_begin = (edge.id_begin.first + 1) * 1e6 + edge.id_begin.second;
        auto id_end = (edge.id_end.first + 1) * 1e6 + edge.id_end.second;
        if (edge.label == 1) {
            normal_edge_file << std::setprecision(8) << id_begin << " " << id_end << std::endl;
        } else {
            // Write the edge
            loop_edge_file << std::setprecision(8) << id_begin << " " << id_end << std::endl;
        }
    }

    normal_edge_file.close();
    loop_edge_file.close();
}

// Move all the cluster in the same cordinate
//  using the calculated global transforms
void ClusterMergeOptimizer::InitialPoseGraph(const std::vector<std::shared_ptr<Reconstruction>>& reconstructions,
                                             const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) & global_transforms,
                                             const std::vector<cluster_t>& cluster_ordered) {
    std::cout << "[Pose Graph Optimizer] Start the Pose Graph Initialization" << std::endl;
    std::cout << "reconstructions.size() " << reconstructions.size() << std::endl;
    std::cout << "cluster_ordered.size() " << cluster_ordered.size() << std::endl;

    // Create the empty initial reconstruction manager
    reconstruction_manager_ = std::make_shared<ReconstructionManager>();

    reconstruction_manager_->Add(cluster_ordered[0]);
    *(reconstruction_manager_->Get(cluster_ordered[0]).get()) = *(reconstructions[cluster_ordered[0]].get());

    for (size_t index = 1; index < cluster_ordered.size(); ++index) {
        // Check the cluster ordered id is in the reconstruction
        CHECK(index >= 0 && index < reconstructions.size());

        // Copy the reconstruction
        reconstruction_manager_->Add(cluster_ordered[index]);
        *(reconstruction_manager_->Get(cluster_ordered[index]).get()) =
            *(reconstructions[cluster_ordered[index]].get());

        // Transform the current reconstruction
        //  using calculated global transform
        reconstruction_manager_->Get(cluster_ordered[index])
            ->TransformReconstruction(global_transforms.at(cluster_ordered[index]));
    }
}

// Calculate the relative similiarty transform
//  between the transformed cluster using calculated common points
void ClusterMergeOptimizer::RefineRelativeTransform(const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) &
                                                        relative_transforms,
                                                    const std::vector<cluster_t>& cluster_ordered) {
    std::cout << "[Pose Graph Optimizer] Start Refine relative Transform" << std::endl;

    for (auto relative_transform : relative_transforms) {
        cluster_t cluster_1, cluster_2;

        // Convert cluster pair id to cluster id
        utility::PairIdToImagePair(relative_transform.first, &cluster_1, &cluster_2);

        // Check the cluster 1 and cluster 2
        //   are in the cluster ordered vector
        auto cluster_1_it = find(cluster_ordered.begin(), cluster_ordered.end(), cluster_1);
        auto cluster_2_it = find(cluster_ordered.begin(), cluster_ordered.end(), cluster_2);

        if (cluster_1_it != cluster_ordered.end() && cluster_2_it != cluster_ordered.end()) {
            if (cluster_1 == cluster_2) {
                continue;
            }
            std::cout << "[Pose Graph Optimizer] Get relative transform from  " << cluster_1 << " to " << cluster_2
                      << std::endl;

            // Refine the cluster_id to reconstruction id
            // const auto refine_cluster_1 = manager_id_map_[cluster_1];
            // const auto refine_cluster_2 = manager_id_map_[cluster_2];
            // std::cout << "[Pose Graph Optimizer] Refine relative transform as  " << refine_cluster_1 << " to "
            //           << refine_cluster_2 << std::endl;

            auto refine_pair_id = utility::ImagePairToPairId(cluster_1, cluster_2);
            relative_transforms_.emplace(refine_pair_id, relative_transform.second);
        }
    }
}

// Find Loop Image Candidate
void ClusterMergeOptimizer::FindLoopImageCandidate() {
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    std::cout << "reconsturction_ids.size() " << reconsturction_ids.size() << std::endl;

    auto strong_loops = full_correspondence_graph_->GetStrongLoopPairs();
    auto normal_loops = full_correspondence_graph_->GetLoopPairsInfo();
    auto normal_pairs = full_correspondence_graph_->GetNormalPairs();

    std::vector<image_pair_t> candidate_pairs;
    for (auto pair : strong_loops) {
        candidate_pairs.push_back(pair);
    }
    for (auto pair_info : normal_loops) {
        auto pair = pair_info.first;
        candidate_pairs.push_back(pair);
    }
    // if(candidate_pairs.size()==0){
        for (auto pair : normal_pairs) {
            candidate_pairs.push_back(pair);
        }
    // }
    std::cout << "[ClusterMergeOptimizer] Loop Images Candidate Size: " << candidate_pairs.size() << std::endl;

    for (int i = 0; i < reconsturction_ids.size(); i++) {
        for (int j = i + 1; j < reconsturction_ids.size(); j++) {
            cluster_t cluster_1 = reconsturction_ids[i];
            cluster_t cluster_2 = reconsturction_ids[j];
            auto cluster_pair_id = utility::ImagePairToPairId(cluster_1, cluster_2);
            std::cout << "[ClusterMergeOptimizer] Look for the loop image between  " << cluster_1 << " and "
                      << cluster_2 << std::endl;

            auto reconstruction_src = reconstruction_manager_->Get(cluster_1);
            auto reconstruction_dst = reconstruction_manager_->Get(cluster_2);

            const std::vector<image_t> ids_src = reconstruction_src->RegisterImageIds();
            const std::vector<image_t> ids_dst = reconstruction_dst->RegisterImageIds();

            // set of common images
            std::unordered_set<image_t> image_id_set;
            size_t common_image_counter = 0;

            std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> image_pairs;

            std::cout << "options_->clusters_close_images_distance "
                      << (double)(options_->clusters_close_images_distance) *
                             static_cast<double>(options_->current_iteration)
                      << std::endl;

            double min_cluster_distance = std::numeric_limits<double>::max();
            double max_cluster_distance = 0;
            int min_image_id1, min_image_id2;
            int max_image_id1, max_image_id2;
            for (const auto& image_id1 : ids_src) {
                for (const auto& image_id2 : ids_dst) {
                    const class Image& image1 = reconstruction_src->Image(image_id1);
                    const class Image& image2 = reconstruction_dst->Image(image_id2);
                    const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id1, image_id2);
                    if (std::find(candidate_pairs.begin(),candidate_pairs.end(),pair_id) != candidate_pairs.end() &&
                        (image1.LabelId() != image2.LabelId() ||
                         (image1.LabelId() == image2.LabelId() &&
                          std::abs(static_cast<double>(image_id1) - static_cast<double>(image_id2)) <= 50))) {
                        if ((image1.ProjectionCenter() - image2.ProjectionCenter()).norm() < min_cluster_distance) {
                            min_cluster_distance = (image1.ProjectionCenter() - image2.ProjectionCenter()).norm();
                            min_image_id1 = image_id1;
                            min_image_id2 = image_id2;
                        }
                        // if ((image1.ProjectionCenter() - image2.ProjectionCenter()).norm() > max_cluster_distance) {
                        //     max_cluster_distance = (image1.ProjectionCenter() - image2.ProjectionCenter()).norm();
                        //     max_image_id1 = image_id1;
                        //     max_image_id2 = image_id2;
                        // }
                    }
                }
            }
            std::cout << "min_cluster_distance " << min_cluster_distance << " " << min_image_id1 << " " << min_image_id2
                      << std::endl;
            // std::cout << "max_cluster_distance " << max_cluster_distance << " " << max_image_id1 << " " << max_image_id2
            //           << std::endl;
            std::cout << "distance threshold "
                      << max_cluster_distance / (static_cast<double>(options_->max_iter_time) -
                                                 (double)(options_->current_iteration) + 1) +
                             min_cluster_distance
                      << std::endl;
            double threshold = std::min(min_cluster_distance,options_->clusters_close_images_distance)*(static_cast<double>(options_->max_iter_time) +
                                                       (double)(options_->current_iteration) + 1);
            std::cout << "distance threshold "
                      << threshold
                      << std::endl;

            // Get candidate pairs between the two clusters
            for (const auto& image_id1 : ids_src) {
                for (const auto& image_id2 : ids_dst) {
                    if (image_id1 == image_id2 && !image_id_set.count(image_id1)) {
                        image_pairs.emplace_back(std::make_pair(std::make_pair(image_id1, image_id2), 100000));
                        image_id_set.insert(image_id1);
                        common_image_counter++;
                        continue;
                    }

                    const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id1, image_id2);
                    bool strong_loop = false;

                    const class Image& image1 = reconstruction_src->Image(image_id1);
                    const class Image& image2 = reconstruction_dst->Image(image_id2);

                    if (std::find(candidate_pairs.begin(),candidate_pairs.end(),pair_id) != candidate_pairs.end() ||
                        (image1.LabelId() == image2.LabelId() &&
                         abs(static_cast<double>(image_id1) - static_cast<double>(image_id2)) <= 20)||(image1.LabelId() != image2.LabelId())) {
                        strong_loop = true;
                    }
                    if ((strong_loop || (image1.ProjectionCenter() - image2.ProjectionCenter()).norm() < threshold)) {
                        point2D_t num_correspondence =
                            full_correspondence_graph_->NumCorrespondencesBetweenImages(image_id1, image_id2);
                        if (options_->detect_strong_loop && !strong_loop) {
                            continue;
                        }
                        if (num_correspondence > options_->loop_edge_min_pose_inlier_num &&
                            !options_->OnlyAddSameImageID) {
                            // std::cout << "add image_pair " << image_id1 << " " << image_id2 << " " << num_correspondence
                            //           << " " << (image1.ProjectionCenter() - image2.ProjectionCenter()).norm() << " "
                            //           << strong_loop << std::endl;
                            image_pairs.emplace_back(
                                std::make_pair(std::make_pair(image_id1, image_id2), num_correspondence));
                            if (all_candidate_loop_image_ids_.count(cluster_pair_id)) {
                                if (num_correspondence > 0) {
                                    auto candidate_loop_image_pair = utility::ImagePairToPairId(image_id1, image_id2);
                                    all_candidate_loop_image_ids_[cluster_pair_id].push_back(
                                        std::make_pair(candidate_loop_image_pair, num_correspondence));
                                }
                            } else {
                                if (num_correspondence > 0) {
                                    auto candidate_loop_image_pair = utility::ImagePairToPairId(image_id1, image_id2);
                                    all_candidate_loop_image_ids_[cluster_pair_id] = {
                                        std::make_pair(candidate_loop_image_pair, num_correspondence)};
                                }
                            }
                        }
                    }
                }
            }

            if (image_pairs.empty()) {
                std::cout << "[ClusterMergeOptimizer] cannot find loop candidates " << std::endl;
                continue;
            }

            std::cout << "[ClusterMergeOptimizer] Loop candidate Image pair size() = "
                      << image_pairs.size() - common_image_counter
                      << ", Common Image pair size() = " << common_image_counter << std::endl;

            std::sort(image_pairs.begin(), image_pairs.end(),
                      [&](std::pair<std::pair<image_t, image_t>, point2D_t>& e1,
                          std::pair<std::pair<image_t, image_t>, point2D_t>& e2) { return e1.second > e2.second; });

            // even the spread of loop edge
            int divided_part_size = std::ceil((double)ids_src.size() / (double)options_->max_loop_image_between_clusters);
            std::cout << "[ClusterMergeOptimizer] each division size: " << divided_part_size << " "
                      << (double)ids_src.size() / (double)options_->max_loop_image_between_clusters << std::endl;
            std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> remain_image_pairs;
            std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> selected_image_pairs;
            for (auto part_idx = 0; part_idx < options_->max_loop_image_between_clusters; part_idx++) {
                // std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> part_image_pairs;
                // image_t max_image_id = (part_idx + 1) * divided_part_size;
                // image_t min_image_id = part_idx * divided_part_size;
                for (auto image_pair : image_pairs) {
                //     // std::cout<<"image_pair.first.first "<<image_pair.first.first<<std::endl;
                //     if (image_pair.first.first >= min_image_id && image_pair.first.first < max_image_id) {
                //         part_image_pairs.push_back(image_pair);
                //     }
                    selected_image_pairs.push_back(image_pair);
                }
                // // std::cout<<min_image_id<<" "<<max_image_id<<" part_image_pairs.size()
                // // "<<part_image_pairs.size()<<std::endl;
                // std::sort(part_image_pairs.begin(), part_image_pairs.end(),
                //           [&](std::pair<std::pair<image_t, image_t>, point2D_t>& e1,
                //               std::pair<std::pair<image_t, image_t>, point2D_t>& e2) { return e1.second > e2.second; });
                // if (part_image_pairs.size() >= 1) {
                //     selected_image_pairs.push_back(part_image_pairs[0]);

                //     // std::cout<<"add loop about "<<part_image_pairs[0].first.first<<"
                //     // "<<part_image_pairs[0].first.second<<" "<<part_image_pairs[0].second<<std::endl;
                // }
                // if (part_image_pairs.size() >= 2) {
                //     remain_image_pairs.insert(remain_image_pairs.end(), part_image_pairs.begin() + 1,
                //                               part_image_pairs.end());
                // }
            }
            // std::cout << "[ClusterMergeOptimizer] selected image_pairs size: " << selected_image_pairs.size()
            //           << std::endl;
            // std::cout << "[ClusterMergeOptimizer] remain image_pairs size: " << remain_image_pairs.size() << std::endl;

            // if (selected_image_pairs.size() < options_->max_loop_image_between_clusters) {
            //     std::sort(remain_image_pairs.begin(), remain_image_pairs.end(),
            //               [&](std::pair<std::pair<image_t, image_t>, point2D_t>& e1,
            //                   std::pair<std::pair<image_t, image_t>, point2D_t>& e2) { return e1.second > e2.second; });
            //     selected_image_pairs.insert(selected_image_pairs.end(), remain_image_pairs.begin(),
            //                                 remain_image_pairs.end());
            // }
            std::cout << "[ClusterMergeOptimizer] final selected image_pairs size: " << selected_image_pairs.size()
                      << std::endl;

            const size_t max_num_image_pair = selected_image_pairs.size() > options_->max_loop_image_between_clusters
                                                  ? options_->max_loop_image_between_clusters
                                                  : selected_image_pairs.size();

            // const size_t max_num_image_pair = image_pairs.size() > options_->max_loop_image_between_clusters
            //                                       ? options_->max_loop_image_between_clusters
            //                                       : image_pairs.size();

            // Verify candidate pairs with pose estimation
            for (size_t pair_id = 0; pair_id < max_num_image_pair; pair_id++) {
                auto image_pair = selected_image_pairs[pair_id];

                image_t image_id_src = image_pair.first.first;
                image_t image_id_dst = image_pair.first.second;

                if (image_id_src == image_id_dst) {
                    auto common_image_pair = utility::ImagePairToPairId(image_id_src, image_id_dst);
                    all_loop_image_ids_[cluster_pair_id].insert(common_image_pair);
                } else if (!options_->only_process_same_image) {
                    Edge edge;
                    if (options_->EnableLoopVerified &&
                        !EstimatePnP(cluster_1, cluster_2, image_id_src, image_id_dst, edge, false)) {
                        continue;
                    }
                    auto candidate_loop_image_pair = utility::ImagePairToPairId(image_id_src, image_id_dst);
                    all_loop_image_ids_[cluster_pair_id].insert(candidate_loop_image_pair);
                }
            }
            std::cout << "[ClusterMergeOptimizer] candidate loop image between cluster " << cluster_1 << " and "
                      << cluster_2 << ": " << all_loop_image_ids_[cluster_pair_id].size() << std::endl;
        }
    }
}

// Find Loop Image Candidate
void ClusterMergeOptimizer::FindCommonViewImageCandidate() {
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    std::cout << "reconsturction_ids.size() " << reconsturction_ids.size() << std::endl;

    auto strong_loops = full_correspondence_graph_->GetStrongLoopPairs();
    auto normal_loops = full_correspondence_graph_->GetLoopPairsInfo();
    auto normal_pairs = full_correspondence_graph_->GetNormalPairs();

    std::cout << "strong_loops, normal_loops, normal_pairs size: " 
        << strong_loops.size() << ", " << normal_loops.size() << ", " 
        << normal_pairs.size() << std::endl;

    std::vector<image_pair_t> candidate_pairs;
    for (auto pair : strong_loops) {
        candidate_pairs.push_back(pair);
    }
    for (auto pair_info : normal_loops) {
        auto pair = pair_info.first;
        candidate_pairs.push_back(pair);
    }
    for (auto pair : normal_pairs) {
        candidate_pairs.push_back(pair);
    }
    std::cout << "[ClusterMergeOptimizer] Loop Images Candidate Size: " << candidate_pairs.size() << std::endl;

    for (int i = 0; i < reconsturction_ids.size(); i++) {
        for (int j = i + 1; j < reconsturction_ids.size(); j++) {
            cluster_t cluster_1 = reconsturction_ids[i];
            cluster_t cluster_2 = reconsturction_ids[j];
            auto cluster_pair_id = utility::ImagePairToPairId(cluster_1, cluster_2);
            std::cout << "[ClusterMergeOptimizer] Look for the loop image between  " << cluster_1 << " and "
                      << cluster_2 << std::endl;

            auto reconstruction_src = reconstruction_manager_->Get(cluster_1);
            auto reconstruction_dst = reconstruction_manager_->Get(cluster_2);

            const std::vector<image_t> ids_src = reconstruction_src->RegisterImageIds();
            const std::vector<image_t> ids_dst = reconstruction_dst->RegisterImageIds();

            // set of common images
            std::unordered_set<image_t> image_id_set;
            size_t common_image_counter = 0;

            std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> image_pairs;

            // std::cout << "options_->clusters_close_images_distance "
            //           << (double)(options_->clusters_close_images_distance) *
            //                  static_cast<double>(options_->current_iteration)
            //           << std::endl;

            // Get candidate pairs between the two clusters
            std::vector<point2D_t> num_corrs_pair;
            uint64_t num_total_corrs = 0;
            for (const auto& image_id1 : ids_src) {
                for (const auto& image_id2 : ids_dst) {
                    if (image_id1 == image_id2 && !image_id_set.count(image_id1)) {
                        image_pairs.emplace_back(std::make_pair(std::make_pair(image_id1, image_id2), 100000));
                        image_id_set.insert(image_id1);
                        common_image_counter++;
                        continue;
                    }

                    const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id1, image_id2);
                    bool strong_loop = false;

                    const class Image& image1 = reconstruction_src->Image(image_id1);
                    const class Image& image2 = reconstruction_dst->Image(image_id2);

                    if (std::find(candidate_pairs.begin(),candidate_pairs.end(),pair_id) != candidate_pairs.end()) {
                        strong_loop = true;
                    }

                    point2D_t num_correspondence =
                        full_correspondence_graph_->NumCorrespondencesBetweenImages(image_id1, image_id2);

                    if (num_correspondence > options_->loop_edge_min_pose_inlier_num) {
                        image_pairs.emplace_back(
                            std::make_pair(std::make_pair(image_id1, image_id2), num_correspondence));

                        num_corrs_pair.push_back(num_correspondence);
                        num_total_corrs += num_correspondence;

                        if (all_candidate_loop_image_ids_.count(cluster_pair_id)) {
                            if (num_correspondence > 0) {
                                auto candidate_loop_image_pair = utility::ImagePairToPairId(image_id1, image_id2);
                                all_candidate_loop_image_ids_[cluster_pair_id].push_back(
                                    std::make_pair(candidate_loop_image_pair, num_correspondence));
                            }
                        } else {
                            if (num_correspondence > 0) {
                                auto candidate_loop_image_pair = utility::ImagePairToPairId(image_id1, image_id2);
                                all_candidate_loop_image_ids_[cluster_pair_id] = {
                                    std::make_pair(candidate_loop_image_pair, num_correspondence)};
                            }
                        }
                    }
                }
            }

            if (image_pairs.empty()) {
                std::cout << "[ClusterMergeOptimizer] cannot find loop candidates " << std::endl;
                continue;
            }

            std::cout << "[ClusterMergeOptimizer] Loop candidate Image pair size() = "
                      << image_pairs.size() - common_image_counter
                      << ", Common Image pair size() = " << common_image_counter << std::endl;

            std::sort(image_pairs.begin(), image_pairs.end(),
                      [&](std::pair<std::pair<image_t, image_t>, point2D_t>& e1,
                          std::pair<std::pair<image_t, image_t>, point2D_t>& e2) { return e1.second > e2.second; });

#if 0
            // // even the spread of loop edge
            // int divided_part_size = std::ceil((double)ids_src.size() / (double)options_->max_loop_image_between_clusters);
            // std::cout << "[ClusterMergeOptimizer] each division size: " << divided_part_size << " "
            //           << (double)ids_src.size() / (double)options_->max_loop_image_between_clusters << std::endl;
            // std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> remain_image_pairs;
            // std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> selected_image_pairs;
            // for (auto part_idx = 0; part_idx < options_->max_loop_image_between_clusters; part_idx++) {
            //     for (auto image_pair : image_pairs) {
            //         selected_image_pairs.push_back(image_pair);
            //     }
            // }
            // std::cout << "[ClusterMergeOptimizer] final selected image_pairs size: " << selected_image_pairs.size()
            //           << std::endl;

            // const size_t max_num_image_pair = selected_image_pairs.size() > options_->max_loop_image_between_clusters
            //                                       ? options_->max_loop_image_between_clusters
            //                                       : selected_image_pairs.size();
#else 

            double num_mean_correspondence = (double)num_total_corrs / (double)num_corrs_pair.size();
            double accum = 0.0;
            for (int i = 0; i < num_corrs_pair.size(); i++){
                accum += ((num_mean_correspondence - num_corrs_pair.at(i))
                       * (num_mean_correspondence - num_corrs_pair.at(i)));
            }
            double tdev = sqrt(accum / (num_corrs_pair.size() - 1));

            std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> selected_image_pairs;
            for (size_t pair_id = 0; pair_id < image_pairs.size(); pair_id++){
                if(image_pairs.at(pair_id).second < num_mean_correspondence - 2 * tdev){
                    break;
                }
                selected_image_pairs.push_back(image_pairs.at(pair_id));
            }
            const size_t max_num_image_pair = selected_image_pairs.size();
            // std::cout << "max_num_image_pair: " << max_num_image_pair << std::endl;
#endif  

            // Verify candidate pairs with pose estimation
            for (size_t pair_id = 0; pair_id < max_num_image_pair; pair_id++) {
                auto image_pair = selected_image_pairs[pair_id];

                image_t image_id_src = image_pair.first.first;
                image_t image_id_dst = image_pair.first.second;

                if (image_id_src == image_id_dst) {
                    auto common_image_pair = utility::ImagePairToPairId(image_id_src, image_id_dst);
                    all_loop_image_ids_[cluster_pair_id].insert(common_image_pair);
                } else if (!options_->only_process_same_image) {
                    Edge edge;
                    if (options_->EnableLoopVerified &&
                        !EstimatePnP(cluster_1, cluster_2, image_id_src, image_id_dst, edge, false)) {
                        continue;
                    }
                    auto candidate_loop_image_pair = utility::ImagePairToPairId(image_id_src, image_id_dst);
                    all_loop_image_ids_[cluster_pair_id].insert(candidate_loop_image_pair);
                }

                if (image_neighbor_between_cluster_.find(image_id_src) == image_neighbor_between_cluster_.end()){
                    std::set<image_t> nerghbors;
                    nerghbors.insert(image_id_dst);
                    image_neighbor_between_cluster_[image_id_src] = nerghbors;
                } else {
                    image_neighbor_between_cluster_[image_id_src].insert(image_id_dst);
                }

                if (image_neighbor_between_cluster_.find(image_id_dst) == image_neighbor_between_cluster_.end()){
                    std::set<image_t> nerghbors;
                    nerghbors.insert(image_id_src);
                    image_neighbor_between_cluster_[image_id_dst] = nerghbors;
                } else {
                    image_neighbor_between_cluster_[image_id_dst].insert(image_id_src);
                }
            }
            std::cout << "[ClusterMergeOptimizer] candidate loop image between cluster " << cluster_1 << " and "
                      << cluster_2 << ": " << all_loop_image_ids_[cluster_pair_id].size() << std::endl;
        }
    }
    std::cout << "[ClusterMergeOptimizer] image neighbor size: " << image_neighbor_between_cluster_.size() << std::endl;
}

void ClusterMergeOptimizer::ConstructVertex() {
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    for (cluster_t i = 0; i < reconsturction_ids.size(); i++) {
        cluster_t cluster_index = reconsturction_ids[i];
        const auto& cur_reconstruction = reconstruction_manager_->Get(cluster_index);

        // Get all the registed image id
        std::vector<image_t> reconstruction_image_ids = cur_reconstruction->RegisterImageIds();

        for (auto reconstruction_image_id : reconstruction_image_ids) {
            auto& image = cur_reconstruction->Image(reconstruction_image_id);

            auto qvec = image.Qvec();  // -- w, x, y, z
            auto tvec = image.Tvec();

            poses_[cluster_index][reconstruction_image_id].tvec = tvec;
            Eigen::Quaterniond quat(qvec[0], qvec[1], qvec[2], qvec[3]);
            poses_[cluster_index][reconstruction_image_id].qvec = quat;
            // -- Set the image pose scale as initial scale 1
            poses_[cluster_index][reconstruction_image_id].scale = 1;
            // Update image sim3 pose
            ConvertSIM3tosim3(image.Sim3pose(), quat, tvec, 1);
        }
    }
}

void ClusterMergeOptimizer::ConstructNormalEdge() {
    // Get and Add all the connected normal edge
    //  which just inside each reconstruction
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    for (cluster_t i = 0; i < reconsturction_ids.size(); i++) {
        cluster_t cluster_index = reconsturction_ids[i];
        const auto& cur_reconstruction = reconstruction_manager_->Get(cluster_index);
        // Get all the image id from current reconstruction
        std::vector<image_t> reconstruction_image_ids = cur_reconstruction->RegisterImageIds();

        std::unordered_set<image_t> image_ids_set(reconstruction_image_ids.begin(), reconstruction_image_ids.end());

        // Check the reconstruction_image_ids is larger than 2
        CHECK_GT(reconstruction_image_ids.size(), 2);
        size_t inlier_counter = 0;
        std::unordered_set<image_pair_t> image_pair_ids;
        for (auto image_id : reconstruction_image_ids) {
            bool is_common = image_neighbor_between_cluster_.find(image_id) == image_neighbor_between_cluster_.end();
            // Get all the image neighbor
            auto image_neighbors = normal_edge_image_neighbor_[cluster_index][image_id];
            for (auto image_neighbor : image_neighbors) {
                // Avoid self-matches.
                if (image_id == image_neighbor) {
                    continue;
                }

                // Avoid duplicate image pairs.
                auto pair_id = utility::ImagePairToPairId(image_id, image_neighbor);
                if (image_pair_ids.count(pair_id) > 0) {
                    continue;
                }

                image_pair_ids.insert(pair_id);

                // Get image pose
                auto pose_src = poses_[cluster_index][image_id];
                auto pose_dst = poses_[cluster_index][image_neighbor];

                // Calculate the Normal edge Sab by Ta and Tb
                auto t_src = pose_src.tvec;
                auto r_src = pose_src.qvec.normalized().toRotationMatrix();
                auto t_dst = pose_dst.tvec;
                auto r_dst = pose_dst.qvec.normalized().toRotationMatrix();

                auto r_normal_rot = r_src * r_dst.inverse();
                auto t_normal = t_src - r_normal_rot * t_dst;
                Eigen::Quaterniond q_normal(r_normal_rot);

                Edge edge;
                edge.label = 1;
                edge.id_begin.first = cluster_index;
                edge.id_begin.second = image_id;
                edge.id_end.first = cluster_index;
                edge.id_end.second = image_neighbor;
                edge.relative_pose.tvec = t_normal;
                edge.relative_pose.qvec = q_normal;
                edge.relative_pose.scale = 1;
                if ( is_common && image_neighbor_between_cluster_.find(image_neighbor) == image_neighbor_between_cluster_.end()){
                    edge.weight = 100;
                } else {
                    edge.weight = 20;
                }

                // edge.t_be.spose = new double[7];
                edges_.emplace_back(edge);
                inlier_counter++;
            }
        }
        std::cout << "[ClusterMergeOptimizer::ConstructNormalEdge] Cluster-" << cluster_index
            << " Normal Edge Inliner Number : " << inlier_counter << std::endl;
    }
}

bool ClusterMergeOptimizer::EstimatePnP(cluster_t cluster_1, cluster_t cluster_2, image_t image_id_src,
                                        image_t image_id_dst, Edge& edge, bool optimize_pose) {
    const auto& reconstruction_src = reconstruction_manager_->Get(cluster_1);
    const auto& reconstruction_dst = reconstruction_manager_->Get(cluster_2);

    auto image_src = reconstruction_src->Image(image_id_src);
    auto image_dst = reconstruction_dst->Image(image_id_dst);
    auto camera_src = reconstruction_src->Camera(image_src.CameraId());
    auto camera_dst = reconstruction_dst->Camera(image_dst.CameraId());

    Eigen::Vector3d tvec1 = image_src.Tvec();
    Eigen::Vector4d qvec1 = image_src.Qvec();
    Eigen::Vector3d tvec2 = image_dst.Tvec();
    Eigen::Vector4d qvec2 = image_dst.Qvec();

    Eigen::Quaterniond q1(qvec1[0], qvec1[1], qvec1[2], qvec1[3]);
    Eigen::Quaterniond q2(qvec2[0], qvec2[1], qvec2[2], qvec2[3]);

    // Compute all the match 3d point
    auto correspondences = full_correspondence_graph_->FindCorrespondencesBetweenImages(image_id_src, image_id_dst);

    std::vector<Eigen::Vector2d> points2D;
    std::vector<Eigen::Vector3d> points3D;
    std::vector<GP3PEstimator::X_t> points2D_normalized;
    Camera camera;
    std::vector<int> tri_camera_indices;
    std::vector<uint32_t> local_camera_indices;
    std::vector<Eigen::Matrix3x4d> local_transforms;

    // Obtain 2D-2D correspondences
    size_t looped_point_count_in_src = 0;
    size_t looped_point_count_in_dst = 0;
    for (auto correspondece : correspondences) {
        if (image_src.Point2D(correspondece.point2D_idx1).HasMapPoint()) {
            looped_point_count_in_src++;
        }
        if (image_dst.Point2D(correspondece.point2D_idx2).HasMapPoint()) {
            looped_point_count_in_dst++;
        }
    }

    if (looped_point_count_in_src > looped_point_count_in_dst) {
        camera = camera_dst;
    } else {
        camera = camera_src;
    }

    if (camera.NumLocalCameras() > 1) {
        local_transforms.resize(camera.NumLocalCameras());
        for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;
            camera.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
            Eigen::Matrix3x4d local_transform = ComposeProjectionMatrix(local_qvec, local_tvec);
            local_transforms[i] = local_transform;
        }
    }

    if (looped_point_count_in_src > looped_point_count_in_dst) {
        local_camera_indices = image_dst.LocalImageIndices();

        for (auto correspondence : correspondences) {
            if (image_src.Point2D(correspondence.point2D_idx1).HasMapPoint()) {
                auto point2D_1 = image_src.Point2D(correspondence.point2D_idx1);
                auto point2D_2 = image_dst.Point2D(correspondence.point2D_idx2);
                points2D.push_back(point2D_2.XY());
                auto mappoint_1 = reconstruction_src->MapPoint(point2D_1.MapPointId());
                points3D.push_back(mappoint_1.XYZ());

                if (camera.NumLocalCameras() > 1) {
                    uint32_t local_camera_id = local_camera_indices[correspondence.point2D_idx2];
                    tri_camera_indices.push_back(local_camera_id);
                    Eigen::Vector2d point2D_normalized = camera.LocalImageToWorld(local_camera_id, point2D_2.XY());

                    points2D_normalized.emplace_back();
                    points2D_normalized.back().rel_tform = local_transforms[local_camera_id];
                    points2D_normalized.back().xy = point2D_normalized;
                }
            }
        }

    } else {
        local_camera_indices = image_src.LocalImageIndices();

        for (auto correspondence : correspondences) {
            if (image_dst.Point2D(correspondence.point2D_idx2).HasMapPoint()) {
                auto point2D_1 = image_src.Point2D(correspondence.point2D_idx1);
                auto point2D_2 = image_dst.Point2D(correspondence.point2D_idx2);
                points2D.push_back(point2D_1.XY());
                auto mappoint_2 = reconstruction_dst->MapPoint(point2D_2.MapPointId());
                points3D.push_back(mappoint_2.XYZ());

                if (camera.NumLocalCameras() > 1) {
                    uint32_t local_camera_id = local_camera_indices[correspondence.point2D_idx1];
                    tri_camera_indices.push_back(local_camera_id);
                    Eigen::Vector2d point2D_normalized = camera.LocalImageToWorld(local_camera_id, point2D_1.XY());

                    points2D_normalized.emplace_back();
                    points2D_normalized.back().rel_tform = local_transforms[local_camera_id];
                    points2D_normalized.back().xy = point2D_normalized;
                }
            }
        }
    }
    std::cout << "[ClusterMergeOptimizer::EstimatePnP] 2D-3D correspondenece count: " << points2D.size() << std::endl;
    if (points2D.size() < options_->loop_edge_min_pose_inlier_num) {
        return false;
    }

    Eigen::Vector4d loop_qvec;
    Eigen::Vector3d loop_tvec;
    if (camera.NumLocalCameras() == 1) {
        AbsolutePoseEstimationOptions abs_pose_options;
        abs_pose_options.num_threads = -1;
        abs_pose_options.ransac_options.max_error = 12.0;
        abs_pose_options.ransac_options.min_inlier_ratio = 0.25;
        abs_pose_options.ransac_options.min_num_trials = 30;
        abs_pose_options.ransac_options.confidence = 0.9999;

        size_t num_inliers = 0;
        std::vector<char> inlier_mask;

        if (!EstimateAbsolutePose(abs_pose_options, points2D, points3D, &loop_qvec, &loop_tvec, &camera, &num_inliers,
                                  &inlier_mask)) {
            std::cout << "EstimateAbsolutePose failed!, not a valid loop" << std::endl;
            return false;
        }

        std::cout << "[ClusterMergeOptimizer::EstimatePnP] num_inlier: " << num_inliers << std::endl;
        if (num_inliers < static_cast<size_t>(options_->loop_edge_min_pose_inlier_num)) {
            std::cout << "num_inliers " << num_inliers << ", lower than " << options_->loop_edge_min_pose_inlier_num
                      << ", not a valid loop" << std::endl;
            return false;
        }

        if (optimize_pose) {
            AbsolutePoseRefinementOptions abs_pose_refinement_options;
            abs_pose_refinement_options.refine_focal_length = false;
            abs_pose_refinement_options.refine_extra_params = false;

            if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask, points2D, points3D, 
                                    1, std::vector<uint64_t>(),
                                    std::vector<double>(), &loop_qvec, &loop_tvec, &camera)) {
                std::cout << "RefineAbsolutePose failed!, not a valid loop" << std::endl;
                return false;
            }
        }

    } else {
        RANSACOptions ransac_options;
        ransac_options.max_error = 1 - cos(0.0299977504);
        ransac_options.min_inlier_ratio = 0.25;
        ransac_options.min_num_trials = 30;
        ransac_options.confidence = 0.9999;

        RANSAC<GP3PEstimator> ransac(ransac_options);
        const auto report = ransac.Estimate(points2D_normalized, points3D);
        size_t num_inliers = report.support.num_inliers;
        std::vector<char> inlier_mask = report.inlier_mask;

        std::cout << "[ClusterMergeOptimizer::EstimatePnP] num_inlier: " << num_inliers << std::endl;
        if (num_inliers < static_cast<size_t>(options_->loop_edge_min_pose_inlier_num)) {
            std::cout << "num_inliers " << num_inliers << ", lower than " << options_->loop_edge_min_pose_inlier_num
                      << ", not a valid loop" << std::endl;
            return false;
        }

        Eigen::Matrix3d r = report.model.block<3, 3>(0, 0);
        Eigen::Vector3d t = report.model.col(3);

        loop_qvec = RotationMatrixToQuaternion(r);
        loop_tvec = t;

        if (optimize_pose) {
            AbsolutePoseRefinementOptions abs_pose_refinement_options;
            abs_pose_refinement_options.refine_focal_length = false;
            abs_pose_refinement_options.refine_extra_params = false;

            if (!RefineAbsolutePoseRig(abs_pose_refinement_options, inlier_mask, points2D, points3D, 
                                       1, std::vector<uint64_t>(), std::vector<double>(), 
                                       tri_camera_indices, &loop_qvec, &loop_tvec, &camera)) {
                std::cout << "RefineAbsolutePose failed!" << std::endl;
                return false;
            }
        }
    }

    Eigen::Quaterniond loop_q(loop_qvec[0], loop_qvec[1], loop_qvec[2], loop_qvec[3]);
    Eigen::Vector3d loop_t = loop_tvec;

    Eigen::Quaterniond relative_q;
    Eigen::Vector3d relative_t;

    if (looped_point_count_in_src > looped_point_count_in_dst) {
        relative_q = q1 * loop_q.conjugate();
        relative_t = tvec1 - relative_q * loop_t;
    } else {
        relative_q = loop_q * q2.conjugate();
        relative_t = loop_t - relative_q * tvec2;
    }

    if (!IsNaN(loop_qvec) && !IsNaN(loop_t)) {
        edge.label = 2;
        edge.id_begin.first = cluster_1;
        edge.id_begin.second = image_id_src;
        edge.id_end.first = cluster_2;
        edge.id_end.second = image_id_dst;
        edge.relative_pose.tvec = relative_t;
        edge.relative_pose.qvec = relative_q;
        edge.relative_pose.scale = 1;
        return true;
    }
    return false;
}

void ClusterMergeOptimizer::ConstructLoopEdge() {
    std::cout << "[ClusterMergeOptimizer::ConstructLoopEdge] Add Loop Edge " << std::endl;

    // Calculate the all the loop edge which need to be calculate
    size_t num_edge = 0;
    for (const auto& loop_image_ids : all_loop_image_ids_) {
        num_edge = num_edge + loop_image_ids.second.size();
    }

    // Add Loop edges
    size_t calculate_edge_counter = 0;
    size_t success_estimate = 0;
    size_t fail_estimate = 0;
    std::unordered_set<image_pair_t> image_pair_id_set;
    for (const auto& loop_image_ids : all_loop_image_ids_) {
        cluster_t cluster_1, cluster_2;
        // Convert cluster pair id to cluster id
        utility::PairIdToImagePair(loop_image_ids.first, &cluster_1, &cluster_2);
        const auto& reconstruction_src = reconstruction_manager_->Get(cluster_1);
        const auto& reconstruction_dst = reconstruction_manager_->Get(cluster_2);

        for (const auto& image_pair_id : loop_image_ids.second) {
            calculate_edge_counter++;

            // Print the progress
            // std::cout << "\r";
            std::cout << "Calculate Loop constrain [" << calculate_edge_counter << " / " << num_edge << "]"
                      << std::endl;
            //   << std::flush;

            // Get image_ids from image pair id
            image_t image_id_1, image_id_2, image_id_src, image_id_dst;
            utility::PairIdToImagePair(image_pair_id, &image_id_1, &image_id_2);

            // Avoid dupilcation
            if (image_pair_id_set.count(image_pair_id)) {
                continue;
            }
            image_pair_id_set.insert(image_pair_id);

            // Correct the image id by check which reconstruction exist it
            if (reconstruction_src->ExistsImage(image_id_1) && reconstruction_dst->ExistsImage(image_id_2)) {
                image_id_src = image_id_1;
                image_id_dst = image_id_2;

            } else {
                image_id_src = image_id_2;
                image_id_dst = image_id_1;
            }

            // If find same image id
            if (image_id_src == image_id_dst) {
                Edge edge;
                edge.label = 2;
                edge.id_begin.first = cluster_1;
                edge.id_begin.second = image_id_src;
                edge.id_end.first = cluster_2;
                edge.id_end.second = image_id_dst;
                edge.relative_pose.tvec = Eigen::Vector3d(0, 0, 0);
                edge.relative_pose.qvec = Eigen::Quaterniond(1, 0, 0, 0);
                edge.relative_pose.scale = 1;
                edges_.emplace_back(edge);
                success_estimate++;
                continue;
            }

            Edge edge;
            if (EstimatePnP(cluster_1, cluster_2, image_id_src, image_id_dst, edge, true)) {
                edges_.emplace_back(edge);
                success_estimate++;
            } else {
                fail_estimate++;
            }
        }
    }
    std::cout << "\n[ClusterMergeOptimizer::ConstructLoopEdge] Success estimate loop edge = " << success_estimate << std::endl;
    std::cout << "[ClusterMergeOptimizer::ConstructLoopEdge] Estimate fail size = " << fail_estimate << std::endl;
}

void ClusterMergeOptimizer::OptimizationSim3() {
    // Construct ceres problem
    std::cout << "[Pose Graph Optimizer] Construct Sim3 Ceres Problem" << std::endl;

    std::unordered_map<cluster_pair_t, std::unordered_set<image_pair_t>> image_pair_map;
    size_t loop_edge_counter = 0;

    std::unordered_map<cluster_t, std::unordered_set<image_t>> images_in_problem;
    for (const auto& edge : edges_) {
        auto pose_begin_iter = poses_[edge.id_begin.first].find(edge.id_begin.second);
        CHECK(pose_begin_iter != poses_[edge.id_begin.first].end())
            << "Pose with ID: " << edge.id_begin.second << " not found.";
        auto pose_end_iter = poses_[edge.id_end.first].find(edge.id_end.second);
        CHECK(pose_end_iter != poses_[edge.id_end.first].end())
            << "Pose with ID: " << edge.id_end.second << " not found.";

        auto pair_id = utility::ImagePairToPairId(edge.id_begin.second, edge.id_end.second);

        auto cluster_pair_id = utility::ImagePairToPairId(edge.id_begin.first, edge.id_end.first);

        if (image_pair_map[cluster_pair_id].count(pair_id)) {
            std::cout << "Found dulicated edge " << edge.id_begin.second << " , " << edge.id_end.second << std::endl;
            continue;
        }

        image_pair_map[cluster_pair_id].insert(pair_id);

        // Convert SIM3 to sim3
        Eigen::Vector7d spose;
        ConvertSIM3tosim3(spose, edge.relative_pose.qvec, edge.relative_pose.tvec, edge.relative_pose.scale);

        if (edge.id_begin.first != edge.id_end.first) {
            // if (edge.id_begin.second != edge.id_end.second) {
            //     std::cout << " Image id 1 = " << edge.id_begin.second << " , with cluster = " << edge.id_begin.first
            //               << " . Image id 2 = " << edge.id_end.second << " , with cluster = " << edge.id_end.first
            //               << " , with correspondece number = " << edge.correspondence_num << std::endl;
            // }
            loop_edge_counter++;
        }
        double weight = edge.weight;
        optimizer_->AddConstraint(edge.id_begin, edge.id_end, spose, weight);
        std::pair<cluster_t, image_t> tmp_id = edge.id_begin;
        images_in_problem[tmp_id.first].insert(tmp_id.second);
        tmp_id = edge.id_end;
        images_in_problem[tmp_id.first].insert(tmp_id.second);
    }

    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    CHECK_GT(reconsturction_ids.size(), 1);

    std::cout<<"pose graph use_prior_relative_pose: "<<options_->use_prior_relative_pose<<std::endl;
    // fix slam relative pose
    if (options_->use_prior_relative_pose) {
        std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
        cluster_t slam_cluster_index = reconsturction_ids[options_->use_prior_relative_pose_id];
        const auto& slam_reconstruction = reconstruction_manager_->Get(slam_cluster_index);
        std::vector<image_t> slam_image_ids = slam_reconstruction->RegisterImageIds();
        std::sort(slam_image_ids.begin(), slam_image_ids.end());
        for (auto image_idx1 = 0; image_idx1 < slam_image_ids.size(); image_idx1++) {
            // std::cout<<"slam edge "<<slam_image_ids[image_idx-1]<<" "<<slam_image_ids[image_idx]<<std::endl;
            for (auto image_idx2 = image_idx1+1; image_idx2 < image_idx1 +10 && image_idx2 < slam_image_ids.size(); image_idx2++) {
                const auto image1 = slam_reconstruction->Image(slam_image_ids[image_idx1]);
                const auto image2 = slam_reconstruction->Image(slam_image_ids[image_idx2]);
                // Calculate the Normal edge Sab by Ta and Tb
                auto t_src = image1.Tvec();
                auto r_src = image1.RotationMatrix();
                auto t_dst = image2.Tvec();
                auto r_dst = image2.RotationMatrix();

                auto r_normal_rot = r_src * r_dst.inverse();
                auto t_normal = t_src - r_normal_rot * t_dst;
                Eigen::Quaterniond q_normal(r_normal_rot);
                Eigen::Vector7d spose;
                ConvertSIM3tosim3(spose, q_normal, t_normal, 1);
                optimizer_->AddConstraint({slam_cluster_index, slam_image_ids[image_idx1]},
                                        {slam_cluster_index, slam_image_ids[image_idx2]}, spose);
            }
        }
    }

    // Fix the loop images
    std::cout << "Fix first frame" << std::endl;
    std::cout << " Loop Edge num = " << loop_edge_counter << std::endl;
    // set first frame as constant
    image_t id_begin, id_end;
    cluster_t cluster_id_begin, cluster_id_end;
    auto itorator = image_pair_map.begin()->second.begin();
    utility::PairIdToImagePairOrdered(*itorator, &id_begin, &id_end);
    utility::PairIdToImagePairOrdered(image_pair_map.begin()->first, &cluster_id_begin, &cluster_id_end);

    // if (poses_[cluster_id_begin].find(id_begin) == poses_[cluster_id_begin].end()) {
    //     optimizer_->SetSIM3ScaleParameterConstant(std::make_pair(cluster_id_begin, id_end));
    // } else {
    //     optimizer_->SetSIM3ScaleParameterConstant(std::make_pair(cluster_id_begin, id_begin));
    // }

    // fix the pose of cluster1 (old reconstruction)
    if (options_->fixed_original_reconstruction) {
        std::cout << "Fix original reconstruction" << std::endl;
        reconsturction_ids = reconstruction_manager_->getReconstructionIds();
        cluster_t cluster_index = reconsturction_ids[options_->original_reconstruction_id];
        const auto& cur_reconstruction = reconstruction_manager_->Get(cluster_index);
        // Get all the registed image id
        std::vector<image_t> reconstruction_image_ids = cur_reconstruction->RegisterImageIds();
        std::cout << "Fix original reconstruction of " << cluster_index << std::endl;
        std::cout << "Fix original reconstruction has images " << reconstruction_image_ids.size() << std::endl;

        for (auto reconstruction_image_id : reconstruction_image_ids) {
            if (images_in_problem[cluster_index].count(reconstruction_image_id) > 0) {
                // optimizer_->SetSIM3PoseParameterConstant(std::make_pair(cluster_index, reconstruction_image_id));
                optimizer_->SetParameterConstant(std::make_pair(cluster_index, reconstruction_image_id));
            }
        }

        image_const_pose_ids_.insert(reconstruction_image_ids.begin(), reconstruction_image_ids.end());
    }

    if (options_->fixed_reconstruction_pose) {
        std::cout << "Fix reconstruction pose" << std::endl;
        CHECK(reconsturction_ids.size() > 1);
        reconsturction_ids = reconstruction_manager_->getReconstructionIds();
        cluster_t cluster_index = reconsturction_ids[options_->fixed_reconstruction_pose_id];
        const auto& cur_reconstruction = reconstruction_manager_->Get(cluster_index);
        // Get all the registed image id
        std::vector<image_t> reconstruction_image_ids = cur_reconstruction->RegisterImageIds();
        std::cout << "Fix reconstruction pose of " << cluster_index << std::endl;
        std::cout << "Fix reconstruction has images " << reconstruction_image_ids.size() << std::endl;

        for (auto reconstruction_image_id : reconstruction_image_ids) {
            if (images_in_problem[cluster_index].count(reconstruction_image_id) > 0) {
                optimizer_->SetSIM3PoseParameterConstant(std::make_pair(cluster_index, reconstruction_image_id));
            }
        }
        image_const_pose_ids_.insert(reconstruction_image_ids.begin(), reconstruction_image_ids.end());
    }

    if (options_->fixed_recon_scale) {
        std::cout << "Fix reconstruction scale" << std::endl;
        reconsturction_ids = reconstruction_manager_->getReconstructionIds();
        cluster_t scale_cluster_index = reconsturction_ids[options_->fixed_reconstruction_scale_id];
        std::cout << "Fix reconstruction scale of " << scale_cluster_index << std::endl;
        const auto& scale_reconstruction = reconstruction_manager_->Get(scale_cluster_index);

        std::vector<image_t> scale_reconstruction_image_ids = scale_reconstruction->RegisterImageIds();

        std::cout << "Fix reconstruction has images " << scale_reconstruction_image_ids.size() << std::endl;

        for (auto reconstruction_image_id : scale_reconstruction_image_ids) {
            if (images_in_problem[scale_cluster_index].count(reconstruction_image_id) > 0) {
                optimizer_->SetSIM3ScaleParameterConstant(std::make_pair(scale_cluster_index, reconstruction_image_id));
            }
            // optimizer_->SetParameterConstant(std::make_pair(scale_cluster_index, reconstruction_image_id));
        }
    }

    optimizer_->Solve();
}

void ClusterMergeOptimizer::UpdateSim3ImagePose() {
    std::cout << "[Pose Graph Optimizer] Update reconstruction image "
              << "pose after the sim3 Optimization" << std::endl;
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    for (cluster_t i = 0; i < reconsturction_ids.size(); i++) {
        cluster_t index = reconsturction_ids[i];
        const auto& cur_reconstruction = reconstruction_manager_->Get(index);
        // For all the reconstruction
        auto image_ids = cur_reconstruction->RegisterImageIds();
        for (auto image_id : image_ids) {
            auto& cur_image = cur_reconstruction->Image(image_id);

            // Sim3 -> SE3
            //  [sR, t; 0, 1] -> [R, t/s; 0, 1]
            //  Convert spose to Sim3
            double scale;
            Eigen::Quaterniond qvec;
            Eigen::Vector3d tvec;
            Convertsim3toSIM3(cur_image.Sim3pose(), qvec, tvec, scale);

            // Conver the t to t/s
            tvec = 1. / scale * tvec;

            // Update Image pose using sim3 pose
            cur_reconstruction->Image(image_id).SetTvec(tvec);
            cur_reconstruction->Image(image_id).SetQvec(
                RotationMatrixToQuaternion(qvec.normalized().toRotationMatrix()));

            // Get the original image pose T_i
            auto ori_tvec = poses_[index][image_id].tvec;
            auto ori_rot = poses_[index][image_id].qvec.normalized().toRotationMatrix();

            // Get correct sim3 S_i
            auto correct_tvec = scale * tvec;
            auto correct_rot = qvec.normalized().toRotationMatrix();
            auto correct_scale = scale;

            // Calculate S_i^-1
            auto correct_rot_inv = correct_rot.inverse();
            auto correct_tvec_inv = correct_rot_inv * ((-1. / correct_scale) * correct_tvec);
            auto correct_scale_inv = 1. / correct_scale;

            // S_i^-1 * T_i
            auto transform_rot = correct_rot_inv * ori_rot;
            auto transform_tvec = correct_scale_inv * (correct_rot_inv * ori_tvec) + correct_tvec_inv;
            auto transform_scale = correct_scale_inv;

            Vertex pose;
            pose.tvec = transform_tvec;
            auto transform_qvec = RotationMatrixToQuaternion(transform_rot);
            pose.qvec = Eigen::Quaterniond(transform_qvec(0), transform_qvec(1), transform_qvec(2), transform_qvec(3));
            pose.scale = transform_scale;
            image_transform_[index][image_id] = pose;

            // Update poses_
            poses_[index][image_id].tvec = cur_reconstruction->Image(image_id).Tvec();
            auto correct_qvec = cur_reconstruction->Image(image_id).Qvec();
            poses_[index][image_id].qvec =
                Eigen::Quaterniond(correct_qvec(0), correct_qvec(1), correct_qvec(2), correct_qvec(3));
            poses_[index][image_id].scale = 1;

            // reconstruction_->Image(image_id).SetTvec(poses[image_id].p);
            // reconstruction_->Image(image_id).SetQvec(RotationMatrixToQuaternion(poses[image_id].q.normalized().toRotationMatrix()));
        }
    }
}

void ClusterMergeOptimizer::UpdateSim3MapPoint() {
    std::cout << "[Pose Graph Optimizer] Update reconstruction "
              << "after the sim3 Optimization" << std::endl;

    // For all the reconstruction
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    for (cluster_t i = 0; i < reconsturction_ids.size(); i++) {
        cluster_t index = reconsturction_ids[i];
        const auto& cur_reconstruction = reconstruction_manager_->Get(index);
        // Get all the mappoint ids
        auto mappoint_ids = cur_reconstruction->MapPointIds();
        for (auto mappoint_id : mappoint_ids) {
            // Get current mappoint
            auto cur_mappoint = cur_reconstruction->MapPoint(mappoint_id);

            // FIXME: Get the first track element image id
            auto image_id = cur_mappoint.Track().Elements().begin()->image_id;

            // Get the pose from this image id
            auto transform = image_transform_[index][image_id];

            // Get the translation and rotation
            auto tvec = transform.tvec;
            auto quat = transform.qvec.normalized();
            auto rot = transform.qvec.normalized().toRotationMatrix();
            auto qvec = Eigen::Vector4d(quat.w(), quat.x(), quat.y(), quat.z());

            // Create sim3 transform to update mappoint
            SimilarityTransform3 mappoint_transform(transform.scale, qvec, tvec);

            // Transform the mappoint
            auto xyz = cur_mappoint.XYZ();
            mappoint_transform.TransformPoint(&xyz);

            // Update mappoint pose
            cur_reconstruction->MapPoint(mappoint_id).SetXYZ(xyz);
        }
    }
}

void ClusterMergeOptimizer::OptimizationSE3() {
    // Construct ceres problem
    std::cout << "[Pose Graph Optimizer] Construct "
              << "SE3 Ceres Problem" << std::endl;

    for (const auto& edge : edges_) {
        auto pose_begin_iter = poses_[edge.id_begin.first].find(edge.id_begin.second);
        CHECK(pose_begin_iter != poses_[edge.id_begin.first].end())
            << "Pose with ID: " << edge.id_begin.second << " not found.";

        auto pose_end_iter = poses_[edge.id_end.first].find(edge.id_end.second);
        CHECK(pose_end_iter != poses_[edge.id_end.first].end())
            << "Pose with ID: " << edge.id_end.second << " not found.";

        // Add to ceres problem
        optimizer_->AddConstraint(edge.id_begin, edge.id_end, edge.relative_pose.qvec, edge.relative_pose.tvec);
    }

    std::cout << "[Pose Graph Optimizer] Add Edge finished" << std::endl;

    // set first frame as constant
    auto pose_start_iter = poses_[edges_.begin()->id_begin.first].find(edges_.begin()->id_begin.second);

    CHECK(pose_start_iter != poses_[edges_.begin()->id_begin.first].end())
        << "Pose with ID: " << edges_.begin()->id_begin.second << " not found.";
    optimizer_->SetParameterConstant(edges_.begin()->id_begin);

    optimizer_->Solve();
}

void ClusterMergeOptimizer::UpdateSE3ImagePose() {
    std::cout << "[Pose Graph Optimizer] Update "
              << "reconstruction image pose after the Optimization" << std::endl;
    // For all the reconstruction
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    for (cluster_t i = 0; i < reconsturction_ids.size(); i++) {
        cluster_t cluster_index = reconsturction_ids[i];
        const auto& cur_reconstruction = reconstruction_manager_->Get(cluster_index);
        auto image_ids = cur_reconstruction->RegisterImageIds();
        for (auto image_id : image_ids) {
            auto cur_image = cur_reconstruction->Image(image_id);

            // For each image get the update pose
            auto opt_tvec = cur_image.Tvec();
            auto opt_rot = cur_image.RotationMatrix();

            // Get the original image pose
            auto ori_tvec = poses_[cluster_index][image_id].tvec;
            auto ori_rot = poses_[cluster_index][image_id].qvec.normalized().toRotationMatrix();

            // Calculate the transform for map point
            auto rot_transform = opt_rot.inverse() * ori_rot;
            auto t_transform = opt_rot.inverse() * (ori_tvec - opt_tvec);

            Vertex pose;
            pose.tvec = t_transform;
            auto qvec = RotationMatrixToQuaternion(rot_transform);
            pose.qvec.w() = qvec[0];
            pose.qvec.x() = qvec[1];
            pose.qvec.y() = qvec[2];
            pose.qvec.z() = qvec[3];
            image_transform_[cluster_index][image_id] = pose;
        }
    }
}

void ClusterMergeOptimizer::UpdateSE3MapPoint() {
    std::cout << "[Pose Graph Optimizer] Update "
              << "reconstruction after the Optimization" << std::endl;

    // For all the reconstruction
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    for (cluster_t i = 0; i < reconsturction_ids.size(); i++) {
        cluster_t index = reconsturction_ids[i];
        const auto& cur_reconstruction = reconstruction_manager_->Get(index);
        // Get all the mappoint ids
        auto mappoint_ids = cur_reconstruction->MapPointIds();
        for (auto mappoint_id : mappoint_ids) {
            // Get current mappoint
            auto cur_mappoint = cur_reconstruction->MapPoint(mappoint_id);

            // FIXME: Get the first track element image id
            auto image_id = cur_mappoint.Track().Elements().begin()->image_id;

            // Get the pose from this image id
            auto transform = image_transform_[index][image_id];

            // Get the translation and rotation
            auto tvec = transform.tvec;
            auto rot = transform.qvec.normalized().toRotationMatrix();

            auto xyz = cur_mappoint.XYZ();
            xyz = rot * xyz + tvec;

            // Update mappoint pose
            cur_reconstruction->MapPoint(mappoint_id).SetXYZ(xyz);
        }
    }
}

// Output all the candidate loop edge which has coorepondence with their correpsondence
void ClusterMergeOptimizer::OutputCorrespondence() {
    // std::unordered_map<Edge, size_t> candidate_edges;  // -- Store the candidate edge with their coorespondence
    std::vector<ClusterMergeOptimizer::Edge> candidate_edges;
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    for (cluster_t i = 0; i < reconsturction_ids.size(); i++) {
        cluster_t cluster_1 = reconsturction_ids[i];
        for (cluster_t j = 1; j < reconsturction_ids.size(); j++) {
            cluster_t cluster_2 = reconsturction_ids[j];
            if (cluster_1 != cluster_2) {
                // Convert cluster pair id to cluster id
                std::cout << "[Pose Graph Optimizer] Compute the common image between  " << cluster_1 << " and "
                          << cluster_2 << std::endl;

                auto reconstruction_src = reconstruction_manager_->Get(cluster_1);
                auto reconstruction_dst = reconstruction_manager_->Get(cluster_2);

                // Get two candidate image sets from different reconstructions
                const std::vector<image_t> ids_src = reconstruction_src->RegisterImageIds();
                const std::vector<image_t> ids_dst = reconstruction_dst->RegisterImageIds();

                // Find Image pair with correspondence or have the same image id
                std::unordered_set<image_t> image_id_set;
                size_t common_image_counter = 0;
                std::vector<std::pair<image_t, image_t>> image_pairs;

                for (const auto& image_id1 : ids_src) {
                    for (const auto& image_id2 : ids_dst) {
                        // if (image_id1 == image_id2 && !image_id_set.count(image_id1)) {
                        //     image_pairs.emplace_back(std::make_pair(image_id1, image_id2));
                        //     image_id_set.insert(image_id1);
                        //     continue;
                        // }

                        point2D_t num_correspondence =
                            full_correspondence_graph_->NumCorrespondencesBetweenImages(image_id1, image_id2);

                        if (num_correspondence) {
                            image_pairs.emplace_back(std::make_pair(image_id1, image_id2));
                        }

                        // // Calculate the different between two image id
                        // image_t image_id_diff = image_id1 > image_id2 ? image_id1 - image_id2 : image_id2 -
                        // image_id1;

                        // //
                        // bool cor_check_condition = options_->OnlyAddCloseImageID ? image_id_diff < 20 : true;

                        // bool cor_check_codition_2 = !options_->OnlyAddSameImageID;
                        // if (num_correspondence > options_->Corespondence2DThreshold && cor_check_condition &&
                        //     cor_check_codition_2) {
                        //     image_pairs.emplace_back(std::make_pair(image_id1, image_id2));
                        // }
                    }
                }

                if (image_pairs.empty()) {
                    continue;
                }

                for (const auto& image_pair : image_pairs) {
                    ClusterMergeOptimizer::Edge e;
                    e.id_begin.first = cluster_1;
                    e.id_begin.second = image_pair.first;
                    e.id_end.first = cluster_2;
                    e.id_end.second = image_pair.second;
                    candidate_edges.emplace_back(e);
                }
            }
        }
    }
    // Output all the loop candidate edge
    std::string all_loop_edge_file_path = options_->optimized_reconsturctions_path + "/all_loop_edges.txt";
    std::ofstream all_loop_edge_file(all_loop_edge_file_path, std::ios::trunc);

    CHECK(all_loop_edge_file.is_open()) << all_loop_edge_file_path;

    for (const auto& edge : candidate_edges) {
        // Write the edge
        auto id_begin = (edge.id_begin.first + 1) * 1e6 + edge.id_begin.second;
        auto id_end = (edge.id_end.first + 1) * 1e6 + edge.id_end.second;

        // Write the edge
        all_loop_edge_file << std::setprecision(8) << id_begin << " " << id_end << std::endl;
    }

    all_loop_edge_file.close();
}

void WriteLoopImagePairs(const std::string path,
                         const std::unordered_map<cluster_pair_t, std::vector<std::pair<image_pair_t, point2D_t>>>&
                             all_candidate_loop_image_ids,
                         std::unordered_map<cluster_pair_t, std::set<image_pair_t>>& all_loop_image_ids) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    size_t num_candidate_loop_image_ids = all_candidate_loop_image_ids.size();
    std::cout << "write num_candidate_loop_image_ids: " << num_candidate_loop_image_ids << std::endl;
    file.write((char*)&num_candidate_loop_image_ids, sizeof(size_t));
    for (auto cluster_pair : all_candidate_loop_image_ids) {
        file.write((char*)&cluster_pair.first, sizeof(cluster_pair_t));
        std::cout << "write cluster_pair.first: " << cluster_pair.first << std::endl;
        size_t num_image_pair = cluster_pair.second.size();
        std::cout << "write num_image_pair: " << num_image_pair << std::endl;
        file.write((char*)&num_image_pair, sizeof(size_t));
        for (auto image_pair : cluster_pair.second) {
            file.write((char*)&image_pair.first, sizeof(image_pair_t));
            file.write((char*)&image_pair.second, sizeof(point2D_t));
            bool choosed = false;
            if (all_loop_image_ids[cluster_pair.first].count(image_pair.first)) {
                choosed = true;
            }
            file.write((char*)&choosed, sizeof(bool));
            // std::cout << "write image_pair.first: " << image_pair.first << std::endl;
            // std::cout << "write image_pair.second: " << image_pair.second << std::endl;
            // std::cout << "write choosed: " << choosed << std::endl;
        }
    }
}

void ReadLoopImagePairs(
    const std::string path,
    std::unordered_map<cluster_pair_t, std::vector<std::pair<image_pair_t, point2D_t>>>& all_candidate_loop_image_ids,
    std::unordered_map<cluster_pair_t, std::set<image_pair_t>>& all_loop_image_ids) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    // read clusters_ordered
    size_t num_candidate_loop_image_ids;
    file.read(reinterpret_cast<char*>(&num_candidate_loop_image_ids), sizeof(size_t));
    // std::cout << "read num_candidate_loop_image_ids: " << num_candidate_loop_image_ids << std::endl;
    std::vector<std::pair<cluster_pair_t, std::set<image_pair_t>>> tmp_all_candidate_loop_image_ids;
    for (size_t i = 0; i < num_candidate_loop_image_ids; ++i) {
        cluster_pair_t cluster_pair_id;
        file.read(reinterpret_cast<char*>(&cluster_pair_id), sizeof(cluster_pair_t));
        // std::cout << "read cluster_pair.first: " << cluster_pair_id << std::endl;

        size_t num_image_pair;
        file.read(reinterpret_cast<char*>(&num_image_pair), sizeof(size_t));
        // std::cout << "read num_image_pair: " << num_image_pair << std::endl;
        all_candidate_loop_image_ids[cluster_pair_id].resize(num_image_pair);
        for (size_t j = 0; j < num_image_pair; ++j) {
            std::pair<image_pair_t, point2D_t> image_pair;
            file.read(reinterpret_cast<char*>(&image_pair.first), sizeof(image_pair_t));
            file.read(reinterpret_cast<char*>(&image_pair.second), sizeof(point2D_t));
            bool choosed;
            file.read(reinterpret_cast<char*>(&choosed), sizeof(bool));
            std::cout << "read image_pair.first: " << image_pair.first << std::endl;
            std::cout << "read image_pair.second: " << image_pair.second << std::endl;
            std::cout << "read choosed: " << choosed << std::endl;
            all_candidate_loop_image_ids[cluster_pair_id][j] = image_pair;
            if (choosed) {
                all_loop_image_ids[cluster_pair_id].insert(image_pair.first);
            }
        }
    }
}

void WriteNormalImagePairs(const std::string path,
                           const std::unordered_map<cluster_t, std::vector<std::pair<image_pair_t, size_t>>>&
                               normal_edge_candidate_image_neighbor) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    size_t num_normal_edge_image_neighbor = normal_edge_candidate_image_neighbor.size();
    // std::cout << "write num_normal_edge_image_neighbor: " << num_normal_edge_image_neighbor << std::endl;
    file.write((char*)&num_normal_edge_image_neighbor, sizeof(size_t));
    for (auto cluster_pair : normal_edge_candidate_image_neighbor) {
        file.write((char*)&cluster_pair.first, sizeof(cluster_t));
        // std::cout << "write cluster_pair.first: " << cluster_pair.first << std::endl;
        size_t num_image_pair = cluster_pair.second.size();
        // std::cout << "write num_image_pair: " << num_image_pair << std::endl;
        file.write((char*)&num_image_pair, sizeof(size_t));
        for (auto image_pair : cluster_pair.second) {
            file.write((char*)&image_pair.first, sizeof(image_pair_t));
            // std::cout << "write image_pair.first: " << image_pair.first << std::endl;
            file.write((char*)&image_pair.second, sizeof(size_t));
            // std::cout << "write image_pair.second: " << image_pair.second << std::endl;
        }
    }
}

void ReadNormalImagePairs(
    const std::string path,
    std::unordered_map<cluster_t, std::unordered_map<image_t, std::set<image_t>>>& normal_edge_image_neighbor,
    std::unordered_map<cluster_t, std::vector<std::pair<image_pair_t, size_t>>>& normal_edge_candidate_image_neighbor) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    // read clusters_ordered
    size_t num_normal_edge_image_neighbor;
    file.read(reinterpret_cast<char*>(&num_normal_edge_image_neighbor), sizeof(size_t));
    std::cout << "read num_normal_edge_image_neighbor: " << num_normal_edge_image_neighbor << std::endl;
    for (size_t i = 0; i < num_normal_edge_image_neighbor; ++i) {
        cluster_t cluster_id;
        file.read(reinterpret_cast<char*>(&cluster_id), sizeof(cluster_t));
        // std::cout << "read cluster_pair.first: " << cluster_id << std::endl;

        size_t num_image_pair;
        file.read(reinterpret_cast<char*>(&num_image_pair), sizeof(size_t));
        // std::cout << "read num_image_pair: " << num_image_pair << std::endl;
        std::vector<std::pair<image_pair_t, size_t>> neighbors;
        for (size_t j = 0; j < num_image_pair; ++j) {
            image_pair_t image_pair_id;
            file.read(reinterpret_cast<char*>(&image_pair_id), sizeof(image_pair_t));
            // std::cout << "read image_pair.first: " << image_pair_id << std::endl;
            size_t corres_num;
            file.read(reinterpret_cast<char*>(&corres_num), sizeof(size_t));
            // std::cout << "read image_pair.second: " << corres_num << std::endl;
            neighbors.emplace_back(std::make_pair(image_pair_id, corres_num));
        }
        normal_edge_candidate_image_neighbor[cluster_id] = neighbors;

        std::sort(neighbors.begin(), neighbors.end(),
                  [&](const std::pair<image_pair_t, size_t>& e1, const std::pair<image_pair_t, size_t>& e2) {
                      return e1.second > e2.second;
                  });

        for (const auto& neighbor : neighbors) {
            image_t image_id1, image_id2;
            sensemap::utility::PairIdToImagePair(neighbor.first, &image_id1, &image_id2);
            if (image_id1 < image_id2) {
                std::swap(image_id1, image_id2);
            }

            if (!normal_edge_image_neighbor[cluster_id][image_id1].count(image_id2)) {
                normal_edge_image_neighbor[cluster_id][image_id1].insert(image_id2);
            }
        }
    }
}

void WriteOptimizedReconstructions(const std::string path, const int index,
                                   const std::shared_ptr<Reconstruction>& reconstruction) {
    std::string pose_save_path = JoinPaths(path, "/cluster" + std::to_string(index + 1) + "/0");
    std::cout << "write pose path: " << pose_save_path << std::endl;
    if (!boost::filesystem::exists(pose_save_path)) {
        boost::filesystem::create_directories(pose_save_path);
    }
    reconstruction->WriteBinary(pose_save_path);
}

void ClusterMergeOptimizer::MergeByPoseGraph(const std::vector<std::shared_ptr<Reconstruction>>& reconstructions,
                                             const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) & global_transforms,
                                             const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) &
                                                 relative_transforms,
                                             const std::vector<cluster_t>& cluster_ordered,
                                             std::shared_ptr<Reconstruction>& reconstruction) {
    std::cout << "[Pose Graph Optimizer] Start "
              << "the Pose Graph Optimization" << std::endl;

    std::cout << "options_->clusters_close_images_distance " << options_->clusters_close_images_distance << std::endl;
    std::cout << "options_->max_iter_time " << options_->max_iter_time << std::endl;

    Timer timer;

    // Preprocessing
    // Move all the cluster in the same cordinate
    //   using the calculated global transforms
    std::cout << "Cluster order: " << cluster_ordered.size() << std::endl;
    InitialPoseGraph(reconstructions, global_transforms, cluster_ordered);

    // Get related relative transform form cluster_ordered
    RefineRelativeTransform(relative_transforms, cluster_ordered);

    timer.Start();

    for (int iter = 0; iter < options_->max_iter_time; iter++) {
        options_->current_iteration = iter + 1;
        // poses_.clear();
        edges_.clear();
        all_loop_image_ids_.clear();
        all_candidate_loop_image_ids_.clear();
        normal_edge_image_neighbor_.clear();
        normal_edge_candidate_image_neighbor_.clear();
        // Calculate all the image neighbor
        std::cout << "options_->load_normal_image_pairs " << options_->load_normal_image_pairs << std::endl;
        if (options_->load_normal_image_pairs) {
            std::cout << "load normal image pairs" << std::endl;
            ReadNormalImagePairs(options_->normal_image_pairs_path, normal_edge_image_neighbor_,
                                 normal_edge_candidate_image_neighbor_);

        } else {
            // std::cout << "FindImageNeighbor begin: " << full_correspondence_graph_->NumImagePairs() << std::endl;
            FindImageNeighbor();
            // std::cout << "FindImageNeighbor after: " << full_correspondence_graph_->NumImagePairs() << std::endl;
            // exit(-1);
        }

        std::cout << "Find Image neighbor cost " << timer.ElapsedMinutes() << " [min]" << std::endl;
        if (options_->save_normal_image_pairs) {
            WriteNormalImagePairs(options_->normal_image_pairs_path, normal_edge_candidate_image_neighbor_);
        }

        std::cout << "Construct Vertex " << std::endl;
        // Construct Vertex
        ConstructVertex();

        // Find the common image between each cluster which map point
        // similiraty transform consist
        //  with the similiarty transform calulcated above
        timer.Reset();
        timer.Start();
        if (options_->load_loop_image_pairs) {
            std::cout << "load loop image pairs" << std::endl;
            ReadLoopImagePairs(options_->loop_image_pairs_path, all_candidate_loop_image_ids_, all_loop_image_ids_);
        } else {
            FindCommonViewImageCandidate();
        }
        // std::cout << "all_loop_image_ids_, all_candidate_loop_image_ids_: " 
        //     << all_loop_image_ids_.begin()->second.size() << ", " 
        //     << all_candidate_loop_image_ids_.begin()->second.size() << std::endl;
        std::cout << "Find loop Image cost " << timer.ElapsedMinutes() << " [min]" << std::endl;
        if (options_->save_loop_image_pairs) {
            WriteLoopImagePairs(options_->loop_image_pairs_path, all_candidate_loop_image_ids_, all_loop_image_ids_);
        }

        // Output the pose graph before the optimization
        if (options_->debug_info) WriteMergeResult(options_->optimized_reconsturctions_path + "/Reconstruction_Before_optimization");

        // Construct Edge
        ConstructNormalEdge();
        timer.Reset();
        timer.Start();
        ConstructLoopEdge();
        std::cout << "Construct loop edge cost " << timer.ElapsedMinutes() << " [min]" << std::endl;

        if (options_->debug_info) WritePoseGraphResult(options_->optimized_reconsturctions_path + "/Pose_Graph_Before");

        if (options_->debug_info) OutputCorrespondence();  // -- Output all the candidate loop edge

        // Set the reconstruction and create pose graph optimizer
        PoseGraphOptimizer::Options options;
        options.optimization_method = options_->optimization_method;
        options.lossfunction_enable = options_->lossfunction_enable;
        options.max_num_iterations = options_->max_optimization_iteration_num;

        optimizer_ = new PoseGraphOptimizer(options, reconstruction_manager_);

        if (options_->optimization_method == PoseGraphOptimizer::OPTIMIZATION_METHOD::SE3) {
            // SE3 Pose Graph Optimization
            OptimizationSE3();
            // Update the image pose using the calculate delta pose
            UpdateSE3ImagePose();
            // Update the cooresponding mapping points
            // using calculate the delta pose
            // UpdateSE3MapPoint();
        } else if (options_->optimization_method == PoseGraphOptimizer::OPTIMIZATION_METHOD::SIM3) {
            // SIM3 Pose Graph Optimization
            OptimizationSim3();
            // Update the image pose using the calculate delta pose
            UpdateSim3ImagePose();
            // Update the cooresponding mapping points
            //  using calculate the delta pose
            UpdateSim3MapPoint();
        }

        if (options_->save_optimized_reconsturctions) {
            std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
            reconstruction_ = std::make_shared<Reconstruction>();
            *(reconstruction_.get()) = *(reconstruction_manager_->Get(reconsturction_ids[0]).get());
            // if (options_->save_optimized_reconsturctions) {
            //     WriteOptimizedReconstructions(options_->optimized_reconsturctions_path, reconsturction_ids[0],
            //     reconstruction_);
            // }
            Eigen::Matrix3x4d transform;
            Eigen::Vector4d qvec(1, 0, 0, 0);
            Eigen::Vector3d tvec(0, 0, 0);
            transform.leftCols<3>() = QuaternionToRotationMatrix(qvec);
            transform.rightCols<1>() = tvec;
            for (cluster_t index = 1; index < reconsturction_ids.size(); index++) {
                const auto& cur_reconstruction = reconstruction_manager_->Get(reconsturction_ids[index]);
                reconstruction_->Merge(*cur_reconstruction.get(), transform, 8.0);

                if (options_->save_optimized_reconsturctions) {
                    std::string rec_path =
                        options_->optimized_reconsturctions_path + "/pose_graph" + std::to_string(iter + 1);
                    std::cout << "rec_path " << rec_path << std::endl;
                    if (!boost::filesystem::exists(rec_path)) {
                        boost::filesystem::create_directories(rec_path);
                    }
                    boost::filesystem::create_directories(rec_path);
                    reconstruction_->WriteReconstruction(rec_path, true);
                }
            }
        }
    }

    if (options_->debug_info) WritePoseGraphResult(options_->optimized_reconsturctions_path + "/Pose_Graph_After");

    // Output the pose graph after the optimization
    if (options_->debug_info) WriteMergeResult(options_->optimized_reconsturctions_path + "/Reconstruction_After_optimization");

    // Get the reconstruction
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    reconstruction_ = std::make_shared<Reconstruction>();
    *(reconstruction_.get()) = *(reconstruction_manager_->Get(reconsturction_ids[0]).get());
    // if (options_->save_optimized_reconsturctions) {
    //     WriteOptimizedReconstructions(options_->optimized_reconsturctions_path, reconsturction_ids[0],
    //     reconstruction_);
    // }
    Eigen::Matrix3x4d transform;
    Eigen::Vector4d qvec(1, 0, 0, 0);
    Eigen::Vector3d tvec(0, 0, 0);
    transform.leftCols<3>() = QuaternionToRotationMatrix(qvec);
    transform.rightCols<1>() = tvec;
    for (cluster_t index = 0; index < reconsturction_ids.size(); index++) {
        const auto& cur_reconstruction = reconstruction_manager_->Get(reconsturction_ids[index]);
        reconstruction_->Merge(*cur_reconstruction.get(), transform, 8.0);

        if (options_->save_optimized_reconsturctions) {
            std::cout << "save optimized pose " << std::endl;
            WriteOptimizedReconstructions(options_->optimized_reconsturctions_path + "/pose_graph",
                                          reconsturction_ids[index], cur_reconstruction);
        }
    }

    // Retriangulation
    TriangulateReconstruction();

    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }

    if (options_->debug_info) WriteReconstructionResult(options_->optimized_reconsturctions_path + "/Triangulation_Result");
    reconstruction_->FilterAllMapPoints(2, 48, 3.0);  // -- 40

    // Merge the close mappoint
    MergeReconstructionTracks();

    reconstruction_->FilterAllMapPoints(3, 24, 3.0);  // -- 40

    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }

    if (options_->debug_info) WriteReconstructionResult(options_->optimized_reconsturctions_path + "/Merge_Result");
    if (options_->save_optimized_reconsturctions) {
        std::string rec_path = options_->optimized_reconsturctions_path + "/pose_graph_tri";
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }
        boost::filesystem::create_directories(rec_path);
        reconstruction_->WriteReconstruction(rec_path, true);
    }
    reconstruction = std::make_shared<Reconstruction>();

    *(reconstruction.get()) = *reconstruction_.get();
}

const std::unordered_map<image_t, std::set<image_t>>&  
ClusterMergeOptimizer::GetImageNeighborBetweenCluster(){
    return image_neighbor_between_cluster_;
}

const std::unordered_set<image_t>& ClusterMergeOptimizer::GetConstImageIds(){
    return image_const_pose_ids_;
}

std::vector<std::vector<image_t>> 
ClusterMergeOptimizer::GetClusterImageIds(bool base_dir = false){
    std::vector<int> reconsturction_ids = reconstruction_manager_->getReconstructionIds();
    
    std::vector<std::vector<image_t>> cluster_image_ids;
    if (!base_dir){
        for (int id = 0; id < reconsturction_ids.size(); id++){
            std::vector<image_t> temp_images_ids 
                = reconstruction_manager_->Get(id)->RegisterImageIds();
            cluster_image_ids.emplace_back(temp_images_ids);
        }
    } else {
        for (int id = 0; id < reconsturction_ids.size(); id++){
            const auto& temp_reconstruction = reconstruction_manager_->Get(id);
            const auto& tmp_image_names = temp_reconstruction->GetImageNames();

            std::map<label_t, std::vector<image_t>> labelled_image_clusters;

            std::map<std::string, label_t> coarse_label_list;
            label_t coarse_label_index = 1;
            for (const auto image_name : tmp_image_names){
                std::string name = image_name.first;
                std::string coarse_dir = name.substr(0, name.find("/"));
                if (coarse_label_list.find(coarse_dir) == coarse_label_list.end()) {
                    coarse_label_list.emplace(coarse_dir, coarse_label_index);
                    coarse_label_index++;
                }
                label_t current_label = coarse_label_list.at(coarse_dir);
                labelled_image_clusters[current_label].push_back(image_name.second);
            }

            for (label_t label_id = 1; label_id < coarse_label_index; label_id++){
                cluster_image_ids.emplace_back(labelled_image_clusters[label_id]);
            }
        }
    }
    return cluster_image_ids;
}
}  // namespace sensemap
