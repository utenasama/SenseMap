//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/logging.h"
#include "util/math.h"
#include "base/projection.h"
#include "base/essential_matrix.h"
#include "estimators/utils.h"
#include "estimators/triangulation.h"
#include "incremental_triangulator.h"
#include "base/pose.h"

namespace sensemap {

bool IncrementalTriangulator::Options::Check() const {
    CHECK_OPTION_GE(max_transitivity, 0);
    CHECK_OPTION_GT(create_max_angle_error, 0);
    CHECK_OPTION_GT(continue_max_angle_error, 0);
    CHECK_OPTION_GT(merge_max_reproj_error, 0);
    CHECK_OPTION_GT(complete_max_reproj_error, 0);
    CHECK_OPTION_GE(complete_max_transitivity, 0);
    CHECK_OPTION_GT(re_max_angle_error, 0);
    CHECK_OPTION_GE(re_min_ratio, 0);
    CHECK_OPTION_LE(re_min_ratio, 1);
    CHECK_OPTION_GE(re_max_trials, 0);
    CHECK_OPTION_GT(min_angle, 0);
    return true;
}

IncrementalTriangulator::IncrementalTriangulator(
    const std::shared_ptr<CorrespondenceGraph> correspondence_graph,
    std::shared_ptr<Reconstruction> reconstruction)
    : correspondence_graph_(correspondence_graph),
      reconstruction_(reconstruction) {}

size_t IncrementalTriangulator::TriangulateImage(const Options& options, const image_t image_id) {
    CHECK(options.Check());

    size_t num_tris = 0;

    ClearCaches();

    Image& image = reconstruction_->Image(image_id);
    if (!image.IsRegistered()) {
        return num_tris;
    }

    const Camera& camera = reconstruction_->Camera(image.CameraId());
    if (HasCameraBogusParams(options, camera)) {
        return num_tris;
    }

    // Correspondence data for reference observation in given image. We iterate
    // over all observations of the image and each observation once becomes
    // the reference correspondence.
    CorrData ref_corr_data;
    ref_corr_data.image_id = image_id;
    ref_corr_data.image = &image;
    ref_corr_data.camera = &camera;

    std::vector<CorrData> corrs_data;

    size_t create_point_count = 0;
    size_t attempt_create_point_count = 0;
    size_t continue_point_count = 0;
    // Try to triangulate all image observations.
    size_t corrs_count = 0;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
        ++point2D_idx) {
        image.Point2D(point2D_idx).SetMask(true);
        
        const size_t num_triangulated =
            Find(options, image_id, point2D_idx,
                static_cast<size_t>(options.max_transitivity), &corrs_data);
        if (corrs_data.empty()) {
            continue;
        }
        corrs_count++;

        ref_corr_data.point2D_idx = point2D_idx;
        ref_corr_data.point2D = &image.Point2D(point2D_idx);

        if (num_triangulated == 0) {
            corrs_data.push_back(ref_corr_data);
            size_t create_points = Create(options, corrs_data);
            num_tris += create_points;
            create_point_count += create_points;
            attempt_create_point_count++;
        } else {
            // Continue correspondences to existing Map Points.
            size_t continue_points = Continue(options, ref_corr_data, corrs_data);
            num_tris += continue_points;
            continue_point_count += continue_points;
            // Create points from correspondences that are not continued.
            corrs_data.push_back(ref_corr_data);
            size_t create_points = Create(options, corrs_data);
            num_tris += create_points;
            create_point_count += create_points;
        }
    }

    std::cout << "Triangulator corrs count: " << corrs_count << std::endl;
    std::cout << "Triangulator attempt to create points: " << attempt_create_point_count << std::endl;
    std::cout << "Triangulator created points: " << create_point_count << std::endl;
    std::cout << "Triangulator continued points: " << continue_point_count << std::endl;
    return num_tris;
}

size_t IncrementalTriangulator::CompleteImage(const Options& options,
                                              const image_t image_id) {
    CHECK(options.Check());

    size_t num_tris = 0;

    ClearCaches();

    const Image& image = reconstruction_->Image(image_id);
    if (!image.IsRegistered()) {
        return num_tris;
    }

    const Camera& camera = reconstruction_->Camera(image.CameraId());
    if (camera.NumLocalCameras()==1 && HasCameraBogusParams(options, camera)) {
        return num_tris;
    }

    // Setup estimation options.
    EstimateTriangulationOptions tri_options;
    tri_options.min_tri_angle = DegToRad(options.min_angle);
    tri_options.residual_type = TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
    tri_options.ransac_options.max_error = options.complete_max_reproj_error;
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
                DegToRad(options.create_max_angle_error);
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

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        if (!point2D.InMask()) {
            continue;
        }
        if (point2D.HasMapPoint()) {
            // Complete existing track.
            num_tris += Complete(options, point2D.MapPointId());
            continue;
        }

        if (options.ignore_two_view_tracks &&
            correspondence_graph_->IsTwoViewObservation(image_id, point2D_idx)) {
            continue;
        }

        const size_t num_triangulated = Find(options, image_id, point2D_idx, 
            static_cast<size_t>(options.max_transitivity), &corrs_data);
        if (num_triangulated || corrs_data.empty()) {
            continue;
        }

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

        // Enforce exhaustive sampling for small track lengths.
        const size_t kExhaustiveSamplingThreshold = 15;
        if (point_data.size() <= kExhaustiveSamplingThreshold) {
            tri_options.ransac_options.min_num_trials = NChooseK(point_data.size(), 2);
        }

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
                track.AddElement(corr_data.image_id, corr_data.point2D_idx);
                num_tris += 1;
            }
        }

        const mappoint_t mappoint_id = reconstruction_->AddMapPoint(xyz, std::move(track));
        modified_mappoint_ids_.insert(mappoint_id);
    }

    return num_tris;
}

size_t IncrementalTriangulator::CompleteTracks(
    const Options& options, const std::unordered_set<mappoint_t>& mappoint_ids) {
    CHECK(options.Check());

    size_t num_completed = 0;

    ClearCaches();

    for (const mappoint_t mappoint_id : mappoint_ids) {
        num_completed += Complete(options, mappoint_id);
    }

    return num_completed;
}

size_t IncrementalTriangulator::CompleteAllTracks(const Options& options) {
    CHECK(options.Check());

    size_t num_completed = 0;

    ClearCaches();

    for (const mappoint_t mappoint_id : reconstruction_->MapPointIds()) {
        num_completed += Complete(options, mappoint_id);
    }

    return num_completed;
}

size_t IncrementalTriangulator::MergeTracks(
    const Options& options, const std::unordered_set<mappoint_t>& mappoint_ids) {
    CHECK(options.Check());

    size_t num_merged = 0;

    ClearCaches();

    for (const mappoint_t mappoint_id : mappoint_ids) {
        num_merged += Merge(options, mappoint_id);
    }

    return num_merged;
}

size_t IncrementalTriangulator::MergeAllTracks(const Options& options) {
    CHECK(options.Check());

    size_t num_merged = 0;

    ClearCaches();

    for (const mappoint_t mappoint_id : reconstruction_->MapPointIds()) {
        num_merged += Merge(options, mappoint_id);
    }

    return num_merged;
}

size_t IncrementalTriangulator::Retriangulate(const Options& options) {
    CHECK(options.Check());

    size_t num_tris = 0;

    ClearCaches();

    Options re_options = options;
    re_options.continue_max_angle_error = options.re_max_angle_error;
    re_options.create_max_angle_error = options.re_max_angle_error;
    for (const auto& image_pair : reconstruction_->ImagePairs()) {
        // Only perform retriangulation for under-reconstructed image pairs.
        const double tri_ratio = static_cast<double>(image_pair.second.first) / image_pair.second.second;
        if (tri_ratio >= options.re_min_ratio) {
            continue;
        }

        // Check if images are registered yet.

        image_t image_id1;
        image_t image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);

        const Image& image1 = reconstruction_->Image(image_id1);
        if (!image1.IsRegistered()) {
            continue;
        }

        const Image& image2 = reconstruction_->Image(image_id2);
        if (!image2.IsRegistered()) {
            continue;
        }

        // Only perform retriangulation for a maximum number of trials.

        int& num_re_trials = re_num_trials_[image_pair.first];
        if (num_re_trials >= options.re_max_trials) {
            continue;
        }
        num_re_trials += 1;

        const Camera& camera1 = reconstruction_->Camera(image1.CameraId());
        const Camera& camera2 = reconstruction_->Camera(image2.CameraId());
        if ((camera1.NumLocalCameras()==1)&&HasCameraBogusParams(options, camera1) ||
            (camera2.NumLocalCameras()==1)&&HasCameraBogusParams(options, camera2)) {
        continue;
        }

        // Find correspondences and perform retriangulation.

        const FeatureMatches& corrs = correspondence_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);

        for (const auto& corr : corrs) {
            const Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
            const Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);

            // Two cases are possible here: both points belong to the same Map Point
            // or to different Map Points. In the former case, there is nothing
            // to do. In the latter case, we do not attempt retriangulation,
            // as retriangulated correspondences are very likely bogus and
            // would therefore destroy both Map Points if merged.
            if (point2D1.HasMapPoint() && point2D2.HasMapPoint()/* ||
                !point2D1.InMask() || !point2D2.InMask()*/) {
                continue;
            }

            CorrData corr_data1;
            corr_data1.image_id = image_id1;
            corr_data1.point2D_idx = corr.point2D_idx1;
            corr_data1.image = &image1;
            corr_data1.camera = &camera1;
            corr_data1.point2D = &point2D1;

            CorrData corr_data2;
            corr_data2.image_id = image_id2;
            corr_data2.point2D_idx = corr.point2D_idx2;
            corr_data2.image = &image2;
            corr_data2.camera = &camera2;
            corr_data2.point2D = &point2D2;

            if (point2D1.HasMapPoint() && !point2D2.HasMapPoint()) {
                const std::vector<CorrData> corrs_data1 = {corr_data1};
                num_tris += Continue(re_options, corr_data2, corrs_data1);
            } else if (!point2D1.HasMapPoint() && point2D2.HasMapPoint()) {
                const std::vector<CorrData> corrs_data2 = {corr_data2};
                num_tris += Continue(re_options, corr_data1, corrs_data2);
            } else if (!point2D1.HasMapPoint() && !point2D2.HasMapPoint()) {
                const std::vector<CorrData> corrs_data = {corr_data1, corr_data2};
                // Do not use larger triangulation threshold as this causes
                // significant drift when creating points (options vs. re_options).
                num_tris += Create(options, corrs_data);
            }
            // Else both points have a Map Point, but we do not want to
            // merge points in retriangulation.
        }
    }

    return num_tris;
}

size_t IncrementalTriangulator::Retriangulate(const Options& options,std::unordered_set<image_t>* image_set) {
    CHECK(options.Check());

    size_t num_tris = 0;

    ClearCaches();

    Options re_options = options;
    re_options.continue_max_angle_error = options.re_max_angle_error;
    re_options.create_max_angle_error = options.re_max_angle_error;

    for (const auto& image_pair : reconstruction_->ImagePairs()) {
        // Only perform retriangulation for under-reconstructed image pairs.
        const double tri_ratio = static_cast<double>(image_pair.second.first) / image_pair.second.second;
        if (tri_ratio >= options.re_min_ratio) {
            continue;
        }

        // Check if images are registered yet.

        image_t image_id1;
        image_t image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);

        if(image_set!=NULL){
            if(image_set->count(image_id1)==0&&image_set->count(image_id2)==0){
                continue;            
            }
        }


        const Image& image1 = reconstruction_->Image(image_id1);
        if (!image1.IsRegistered()) {
            continue;
        }

        const Image& image2 = reconstruction_->Image(image_id2);
        if (!image2.IsRegistered()) {
            continue;
        }

        // Only perform retriangulation for a maximum number of trials.

        int& num_re_trials = re_num_trials_[image_pair.first];
        if (num_re_trials >= options.re_max_trials) {
            continue;
        }
        num_re_trials += 1;

        const Camera& camera1 = reconstruction_->Camera(image1.CameraId());
        const Camera& camera2 = reconstruction_->Camera(image2.CameraId());
        if (HasCameraBogusParams(options, camera1) ||
            HasCameraBogusParams(options, camera2)) {
        continue;
        }

        // Find correspondences and perform retriangulation.

        const FeatureMatches& corrs = correspondence_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);

        for (const auto& corr : corrs) {
            const Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
            const Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);

            // Two cases are possible here: both points belong to the same Map Point
            // or to different Map Points. In the former case, there is nothing
            // to do. In the latter case, we do not attempt retriangulation,
            // as retriangulated correspondences are very likely bogus and
            // would therefore destroy both Map Points if merged.
            if (point2D1.HasMapPoint() && point2D2.HasMapPoint()/* ||
                !point2D1.InMask() || !point2D2.InMask()*/) {
                continue;
            }

            CorrData corr_data1;
            corr_data1.image_id = image_id1;
            corr_data1.point2D_idx = corr.point2D_idx1;
            corr_data1.image = &image1;
            corr_data1.camera = &camera1;
            corr_data1.point2D = &point2D1;

            CorrData corr_data2;
            corr_data2.image_id = image_id2;
            corr_data2.point2D_idx = corr.point2D_idx2;
            corr_data2.image = &image2;
            corr_data2.camera = &camera2;
            corr_data2.point2D = &point2D2;

            if (point2D1.HasMapPoint() && !point2D2.HasMapPoint()) {
                const std::vector<CorrData> corrs_data1 = {corr_data1};
                num_tris += Continue(re_options, corr_data2, corrs_data1);
            } else if (!point2D1.HasMapPoint() && point2D2.HasMapPoint()) {
                const std::vector<CorrData> corrs_data2 = {corr_data2};
                num_tris += Continue(re_options, corr_data1, corrs_data2);
            } else if (!point2D1.HasMapPoint() && !point2D2.HasMapPoint()) {
                const std::vector<CorrData> corrs_data = {corr_data1, corr_data2};
                // Do not use larger triangulation threshold as this causes
                // significant drift when creating points (options vs. re_options).
                num_tris += Create(options, corrs_data);
            }
            // Else both points have a Map Point, but we do not want to
            // merge points in retriangulation.
        }
    }

    return num_tris;
}

size_t IncrementalTriangulator::RetriangulateAllTracks(const Options& options) {
    size_t num_tri = 0;

    for (const mappoint_t mappoint_id : reconstruction_->MapPointIds()) {
        num_tri += Recreate(options, mappoint_id);
    }

    return num_tri;
}

const std::unordered_set<mappoint_t>& IncrementalTriangulator::GetModifiedMapPoints() {
    // First remove any missing Map Points from the set.
    for (auto it = modified_mappoint_ids_.begin(); it != modified_mappoint_ids_.end();) {
        if (reconstruction_->ExistsMapPoint(*it)) {
            ++it;
        } else {
            modified_mappoint_ids_.erase(it++);
        }
    }
    return modified_mappoint_ids_;
}
void IncrementalTriangulator::AddModifiedMapPoint(mappoint_t mappoint_id){
    modified_mappoint_ids_.insert(mappoint_id);
}

void IncrementalTriangulator::ClearModifiedMapPoints() {
    modified_mappoint_ids_.clear();
}

void IncrementalTriangulator::ClearCaches() {
    camera_has_bogus_params_.clear();
    merge_trials_.clear();
}

size_t IncrementalTriangulator::Find(const Options& options,
                                     const image_t image_id,
                                     const point2D_t point2D_idx,
                                     const size_t transitivity,
                                     std::vector<CorrData>* corrs_data) {

  std::vector<CorrespondenceGraph::Correspondence> corrs;
    //   correspondence_graph_->FindTransitiveCorrespondences(
    //       image_id, point2D_idx, transitivity);
    correspondence_graph_->FindTransitiveCorrespondences(image_id, point2D_idx, 1, &corrs);

  corrs_data->clear();
  corrs_data->reserve(corrs.size());

  size_t num_triangulated = 0;

  for (const CorrespondenceGraph::Correspondence corr : corrs) {
    const Image& corr_image = reconstruction_->Image(corr.image_id);
    if (!corr_image.IsRegistered()) {
      continue;
    }

    const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());
    if (HasCameraBogusParams(options, corr_camera)) {
      continue;
    }

    CorrData corr_data;
    corr_data.image_id = corr.image_id;
    corr_data.point2D_idx = corr.point2D_idx;
    corr_data.image = &corr_image;
    corr_data.camera = &corr_camera;
    corr_data.point2D = &corr_image.Point2D(corr.point2D_idx);

    corrs_data->push_back(corr_data);

    if (corr_data.point2D->HasMapPoint()) {
      num_triangulated += 1;
    }
  }

  return num_triangulated;
}

size_t IncrementalTriangulator::Create(const Options& options, const std::vector<CorrData>& corrs_data) {
    // Extract correspondences without an existing triangulated observation.
    std::vector<CorrData> create_corrs_data;
    create_corrs_data.reserve(corrs_data.size());
    for (const CorrData& corr_data : corrs_data) {
        if (!corr_data.point2D->HasMapPoint()) {
            create_corrs_data.push_back(corr_data);
        }
    }

    if (create_corrs_data.size() < 2) {
        // Need at least two observations for triangulation.
        return 0;
    } else if (options.ignore_two_view_tracks && create_corrs_data.size() == 2) {
        const CorrData& corr_data1 = create_corrs_data[0];
        if (correspondence_graph_->IsTwoViewObservation(corr_data1.image_id,
                                                        corr_data1.point2D_idx)) {
            return 0;
        }
    }

    // Setup data for triangulation estimation.
    std::vector<TriangulationEstimator::PointData> point_data;
    point_data.resize(create_corrs_data.size());
    std::vector<TriangulationEstimator::PoseData> pose_data;
    pose_data.resize(create_corrs_data.size());
    for (size_t i = 0; i < create_corrs_data.size(); ++i) {
        const CorrData& corr_data = create_corrs_data[i];
        point_data[i].point = corr_data.point2D->XY();
        point_data[i].point_normalized =
            corr_data.camera->ImageToWorld(point_data[i].point);
        pose_data[i].proj_matrix = corr_data.image->ProjectionMatrix();
        pose_data[i].proj_center = corr_data.image->ProjectionCenter();
        pose_data[i].camera = corr_data.camera;
        if(corr_data.camera->ModelName().compare("SPHERICAL")==0){
            point_data[i].point_bearing = 
                    corr_data.camera->ImageToBearing(point_data[i].point);
        }
        // For camera rig the data are re-prepared
        if(corr_data.camera->NumLocalCameras()>1){
            
            uint32_t local_camera_id = 
                corr_data.image->LocalImageIndices()[corr_data.point2D_idx]; 
            
            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;

            corr_data.camera->GetLocalCameraExtrinsic(local_camera_id,
                                                      local_qvec,local_tvec);


            Eigen::Matrix3d global_R = QuaternionToRotationMatrix(local_qvec)*
                            QuaternionToRotationMatrix(corr_data.image->Qvec()); 

            Eigen::Vector3d global_T = local_tvec + 
                            QuaternionToRotationMatrix(local_qvec) *
                            corr_data.image->Tvec();

            pose_data[i].proj_matrix = ComposeProjectionMatrix(global_R,global_T);
            pose_data[i].proj_center = -global_R.transpose()*global_T;

            point_data[i].point_normalized = 
                corr_data.camera->LocalImageToWorld(local_camera_id,
                                                    point_data[i].point);
        }
    }

    // Setup estimation options.
    EstimateTriangulationOptions tri_options;
    tri_options.min_tri_angle = DegToRad(options.min_angle);
    tri_options.residual_type =
        TriangulationEstimator::ResidualType::ANGULAR_ERROR;
    tri_options.ransac_options.max_error =
        DegToRad(options.create_max_angle_error);
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
    if (!EstimateTriangulation(tri_options, point_data, pose_data, &inlier_mask,
                               &xyz)) {
        return 0;
    }

    // Add inliers to estimated track.
    Track track;
    track.Reserve(create_corrs_data.size());
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        const CorrData& corr_data = create_corrs_data[i];
        if (inlier_mask[i] || corr_data.point2D->InOverlap()) {
            track.AddElement(corr_data.image_id, corr_data.point2D_idx);
        }
    }

    if (track.Length() < 2) {
        return 0;
    }

    const size_t track_length = track.Length();

    // Add estimated point to reconstruction.
    const mappoint_t mappoint_id = reconstruction_->AddMapPoint(xyz, std::move(track));
    modified_mappoint_ids_.insert(mappoint_id);

    const size_t kMinRecursiveTrackLength = 3;
    if (create_corrs_data.size() - track_length >= kMinRecursiveTrackLength) {
        return track_length + Create(options, create_corrs_data);
    }
    return track_length;
}

size_t IncrementalTriangulator::Recreate(
    const Options& options,
    const mappoint_t mappoint_id) {
    size_t num_tri = 0;
    if (!reconstruction_->ExistsMapPoint(mappoint_id)) {
        std::cout << "Mappoint not exist ... " << std::endl;
        return num_tri;
    }

    class MapPoint& mappoint = reconstruction_->MapPoint(mappoint_id);
    class Track& track = mappoint.Track();
    if (track.Length() < 2) {
        std::cout << "Track length smaller than 2 ... " << std::endl;
        // Need at least two observations for triangulation.
        return num_tri;
    } else if (options.ignore_two_view_tracks && track.Length() == 2) {
        const TrackElement& track_el = track.Element(0);
        if (correspondence_graph_->IsTwoViewObservation(track_el.image_id, track_el.point2D_idx)) {
            //std::cout << "Fail at two view geometry ..." << std::endl;
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

        // For camera rig
        if(camera.NumLocalCameras()>1){
            
            uint32_t local_camera_id = 
                image.LocalImageIndices()[track_el.point2D_idx]; 
            
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

            pose_data[i].proj_matrix = ComposeProjectionMatrix(global_R,global_T);
            pose_data[i].proj_center = -global_R.transpose()*global_T;

            point_data[i].point_normalized = 
                        camera.LocalImageToWorld(local_camera_id,
                                                 point_data[i].point);
        }
        if(camera.ModelName().compare("SPHERICAL")==0){
            point_data[i].point_bearing = camera.ImageToBearing(point_data[i].point);
        }
    }

    // Setup estimation options.
    EstimateTriangulationOptions tri_options;
    tri_options.min_tri_angle = DegToRad(options.min_angle);
    tri_options.residual_type = TriangulationEstimator::ResidualType::ANGULAR_ERROR;
    tri_options.ransac_options.max_error = DegToRad(options.create_max_angle_error);
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
        //std::cout << "Fail at triangulation ... " << std::endl;
        return num_tri;
    }

    mappoint.SetXYZ(xyz);
    num_tri++;

    return num_tri;
}

size_t IncrementalTriangulator::Continue(
    const Options& options, const CorrData& ref_corr_data,
    const std::vector<CorrData>& corrs_data) {
    // No need to continue, if the reference observation is triangulated.
    if (ref_corr_data.point2D->HasMapPoint()) {
        return 0;
    }

    double best_angle_error = std::numeric_limits<double>::max();
    size_t best_idx = std::numeric_limits<size_t>::max();

    for (size_t idx = 0; idx < corrs_data.size(); ++idx) {
        const CorrData& corr_data = corrs_data[idx];
        if (!corr_data.point2D->HasMapPoint()) {
            continue;
        }

        const MapPoint& mappoint =
            reconstruction_->MapPoint(corr_data.point2D->MapPointId());

      
        double angle_error; 

        if(ref_corr_data.camera->NumLocalCameras()>1){
            uint32_t local_camera_id = 
                ref_corr_data.image->LocalImageIndices()[ref_corr_data.point2D_idx];

            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;

            ref_corr_data.camera->GetLocalCameraExtrinsic(local_camera_id,
                                                      local_qvec,local_tvec);


            Eigen::Matrix3d global_R = QuaternionToRotationMatrix(local_qvec)*
                            QuaternionToRotationMatrix(ref_corr_data.image->Qvec()); 

            Eigen::Vector3d global_T = local_tvec + 
                            QuaternionToRotationMatrix(local_qvec) *
                            ref_corr_data.image->Tvec();
            Eigen::Vector4d global_qvec = RotationMatrixToQuaternion(global_R);

            angle_error = CalculateAngularErrorRig(ref_corr_data.point2D->XY(),
                                                   mappoint.XYZ(),global_qvec,
                                                   global_T,local_camera_id,
                                                   *ref_corr_data.camera);
        }
        else{
            angle_error = CalculateAngularError(ref_corr_data.point2D->XY(), 
                                                mappoint.XYZ(), 
                                                ref_corr_data.image->Qvec(),
                                                ref_corr_data.image->Tvec(), 
                                                *ref_corr_data.camera);
        }
        
        if (angle_error < best_angle_error) {
            best_angle_error = angle_error;
            best_idx = idx;
        }
    }

    const double max_angle_error = DegToRad(options.continue_max_angle_error);
    if (best_idx != std::numeric_limits<size_t>::max() &&
        (best_angle_error <= max_angle_error || corrs_data[best_idx].point2D->InOverlap())) {
        const CorrData& corr_data = corrs_data[best_idx];
        const TrackElement track_el(ref_corr_data.image_id,
                                    ref_corr_data.point2D_idx);
        reconstruction_->AddObservation(corr_data.point2D->MapPointId(), track_el);
        modified_mappoint_ids_.insert(corr_data.point2D->MapPointId());
        return 1;
    }

    return 0;
}

size_t IncrementalTriangulator::Merge(const Options& options,
                                      const mappoint_t mappoint_id) {
    if (!reconstruction_->ExistsMapPoint(mappoint_id)) {
        return 0;
    }
    const double max_squared_reproj_error =
        options.merge_max_reproj_error * options.merge_max_reproj_error;

    const auto& mappoint = reconstruction_->MapPoint(mappoint_id);

    for (const auto& track_el : mappoint.Track().Elements()) {
        const std::vector<CorrespondenceGraph::Correspondence>& corrs =
            correspondence_graph_->FindCorrespondences(track_el.image_id,
                                                    track_el.point2D_idx);
        class Image& ref_image = reconstruction_->Image(track_el.image_id);
        class Camera& ref_camera = reconstruction_->Camera(ref_image.CameraId());

        for (const auto corr : corrs) {
            const auto& image = reconstruction_->Image(corr.image_id);
            if (!image.IsRegistered()) {
                continue;
            }

            if (corr.point2D_idx >= image.Points2D().size()){
                continue;
            }

            const Point2D& corr_point2D = image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint() ||
                corr_point2D.MapPointId() == mappoint_id ||
                merge_trials_[mappoint_id].count(corr_point2D.MapPointId()) > 0) {
                continue;
            }

            // Try to merge the two Map Points.

            const MapPoint& corr_mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());

            merge_trials_[mappoint_id].insert(corr_point2D.MapPointId());
            merge_trials_[corr_point2D.MapPointId()].insert(mappoint_id);

            // Weighted average of point locations, depending on track length.
            const Eigen::Vector3d merged_xyz =
                (mappoint.Track().Length() * mappoint.XYZ() +
                corr_mappoint.Track().Length() * corr_mappoint.XYZ()) /
                (mappoint.Track().Length() + corr_mappoint.Track().Length());

            // Count number of inlier track elements of the merged track.
            bool merge_success = true;
            for (const Track* track : {&mappoint.Track(), &corr_mappoint.Track()}) {
                for (const auto test_track_el : track->Elements()) {
                    const Image& test_image = reconstruction_->Image(test_track_el.image_id);
                    const Camera& test_camera = reconstruction_->Camera(test_image.CameraId());
                    const Point2D& test_point2D = test_image.Point2D(test_track_el.point2D_idx);
                    
                    //For camera-rig

                    if(test_camera.NumLocalCameras()>1){
                        uint32_t local_camera_id = 
                        test_image.LocalImageIndices()[test_track_el.point2D_idx]; 
            
                        Eigen::Vector4d local_qvec;
                        Eigen::Vector3d local_tvec;

                        test_camera.GetLocalCameraExtrinsic(local_camera_id,
                                                        local_qvec,
                                                        local_tvec);

                        Eigen::Matrix3d global_R = QuaternionToRotationMatrix(local_qvec)*
                            QuaternionToRotationMatrix(test_image.Qvec()); 

                        Eigen::Vector3d global_T = local_tvec + 
                            QuaternionToRotationMatrix(local_qvec) *
                            test_image.Tvec();

                        if (CalculateSquaredReprojectionErrorRig(
                                test_point2D.XY(), merged_xyz, RotationMatrixToQuaternion(global_R),
                                global_T,local_camera_id,test_camera)> max_squared_reproj_error){
                            merge_success = false;
                             break;
                        }
                    }
                    else{
                        if (CalculateSquaredReprojectionError(
                                test_point2D.XY(), merged_xyz, test_image.Qvec(),
                                test_image.Tvec(), test_camera) > max_squared_reproj_error) {
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

                const mappoint_t merged_mappoint_id = reconstruction_->MergeMapPoints(mappoint_id, corr_point2D.MapPointId());

                modified_mappoint_ids_.erase(mappoint_id);
                modified_mappoint_ids_.erase(corr_point2D.MapPointId());
                modified_mappoint_ids_.insert(merged_mappoint_id);

                // Merge merged Map Point and return, as the original points are deleted.
                const size_t num_merged_recursive = Merge(options, merged_mappoint_id);
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

size_t IncrementalTriangulator::Complete(const Options& options,
                                         const mappoint_t mappoint_id) {
    size_t num_completed = 0;

    if (!reconstruction_->ExistsMapPoint(mappoint_id)) {
        return num_completed;
    }

    const double max_squared_reproj_error = options.complete_max_reproj_error * options.complete_max_reproj_error;

    const MapPoint& mappoint = reconstruction_->MapPoint(mappoint_id);

    std::vector<TrackElement> queue = mappoint.Track().Elements();

    const int max_transitivity = options.complete_max_transitivity;
    for (int transitivity = 0; transitivity < max_transitivity; ++transitivity) {
        if (queue.empty()) {
            break;
        }

        const auto prev_queue = queue;
        queue.clear();

        for (const TrackElement queue_elem : prev_queue) {
            const std::vector<CorrespondenceGraph::Correspondence>& corrs =
                correspondence_graph_->FindCorrespondences(queue_elem.image_id,
                                                            queue_elem.point2D_idx);

            for (const auto corr : corrs) {
                const Image& image = reconstruction_->Image(corr.image_id);
                if (!image.IsRegistered()) {
                    continue;
                }

                const Point2D& point2D = image.Point2D(corr.point2D_idx);
                if (point2D.HasMapPoint()) {
                    continue;
                }

                const Camera& camera = reconstruction_->Camera(image.CameraId());
                if (camera.NumLocalCameras()==1 && HasCameraBogusParams(options, camera)) {
                    continue;
                }

                if(camera.NumLocalCameras()>1){
                    if (corr.image_id != queue_elem.image_id) {
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

                // Recursively complete track for this new correspondence.
                if (transitivity < max_transitivity - 1) {
                    queue.emplace_back(corr.image_id, corr.point2D_idx);
                }

                num_completed += 1;
            }
        }
    }

    return num_completed;
}

bool IncrementalTriangulator::HasCameraBogusParams(const Options& options,
                                                   const Camera& camera){ 
    if(camera.NumLocalCameras()>1){
        return false;
    }
                                                 
    if(camera.ModelName().compare("SPHERICAL")==0||
       camera.ModelName().compare("UNIFIED")==0 ||
       camera.ModelName().compare("OPENCV_FISHEYE") == 0){
        return false;
    }
    const auto it = camera_has_bogus_params_.find(camera.CameraId());
    if (it == camera_has_bogus_params_.end()) {
        const bool has_bogus_params = camera.HasBogusParams(
            options.min_focal_length_ratio, options.max_focal_length_ratio,
            options.max_extra_param);
        camera_has_bogus_params_.emplace(camera.CameraId(), has_bogus_params);
        return has_bogus_params;
    } else {
        return it->second;
    }
}

}
