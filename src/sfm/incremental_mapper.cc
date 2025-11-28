// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "incremental_mapper.h"

#include <ceres/rotation.h>

#include <Eigen/Dense>
#include <array>
#include <fstream>

#include "base/image_reader.h"
#include "base/matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/triangulation.h"
#include "base/cost_functions.h"
#include "estimators/absolute_pose.h"
#include "estimators/generalized_absolute_pose.h"
#include "estimators/global_pose_estimation/robust_rotation.h"
#include "estimators/mappoint_alignment.h"
#include "estimators/similarity_transform.h"
#include "estimators/two_view_geometry.h"
#include "estimators/triangulation.h"
#include "estimators/utils.h"
#include "estimators/camera_alignment.h"
#include "optim/pose_graph_optimizer.h"
#include "util/logging.h"
#include "util/rgbd_helper.h"
#include "util/proc.h"
#include "util/ceres_types.h"
// #include "optim/block_bundle_adjustment.h"
#include "optim/global_motions/utils.h"

#include "RGBDAlign/RGBDAlignUtility.h"
#include "RGBDAlign/RGBDRegistration.h"
#include "RGBDAlign/FeatureAlign.h"
#include "RGBDAlign/PointCloudAlign.h"
#include "util/imageconvert.h"

#include "lidar/utils.h"
#include "lidar/absolute_pose.h"

#ifdef USE_OPENBLAS
#include "openblas/cblas.h"
#endif
namespace sensemap {

namespace {

namespace debug {
void PrintSolverSummary(const ceres::Solver::Summary &summary) {
    std::cout << std::right << std::setw(16) << "Residuals : ";
    std::cout << std::left << summary.num_residuals_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Parameters : ";
    std::cout << std::left << summary.num_effective_parameters_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Iterations : ";
    std::cout << std::left << summary.num_successful_steps + summary.num_unsuccessful_steps << std::endl;

    std::cout << std::right << std::setw(16) << "Time : ";
    std::cout << std::left << summary.total_time_in_seconds << " [s]" << std::endl;

    std::cout << std::right << std::setw(16) << "Initial cost : ";
    std::cout << std::right << std::setprecision(6) << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
              << std::endl;

    std::cout << std::right << std::setw(16) << "Final cost : ";
    std::cout << std::right << std::setprecision(6) << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
              << std::endl;

    std::cout << std::right << std::setw(16) << "Termination : ";

    std::string termination = "";

    switch (summary.termination_type) {
        case ceres::CONVERGENCE:
            termination = "Convergence";
            break;
        case ceres::NO_CONVERGENCE:
            termination = "No convergence";
            break;
        case ceres::FAILURE:
            termination = "Failure";
            break;
        case ceres::USER_SUCCESS:
            termination = "User success";
            break;
        case ceres::USER_FAILURE:
            termination = "User failure";
            break;
        default:
            termination = "Unknown";
            break;
    }

    std::cout << std::right << termination << std::endl;
    std::cout << std::endl;
}
}

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
    std::sort(image_ranks.begin(), image_ranks.end(),
              [](const std::pair<image_t, float>& image1, const std::pair<image_t, float>& image2) {
                  return image1.second > image2.second;
              });

    sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
    for (const auto& image : image_ranks) {
        sorted_images_ids->push_back(image.first);
    }

    image_ranks.clear();
}

float RankNextImageMaxVisiblePointsNum(const Image& image) { return static_cast<float>(image.NumVisibleMapPoints()); }

float RankNextImageMaxVisiblePointsRatio(const Image& image) {
    return static_cast<float>(image.NumVisibleMapPoints()) / static_cast<float>(image.NumObservations());
}

float RankNextImageMinUncertainty(const Image& image) { return static_cast<float>(image.MapPointVisibilityScore()); }

float RankNextImageTimeWeightMinUncertainty(const Image& image, const float weight) { return weight * static_cast<float>(image.MapPointVisibilityScore()); }

float RankNextImageWeightedMinUncertainty(const Image& image, const int last_keyframe_idx,
                                          const IncrementalMapper::Options& options) {
    static float tan1 = std::tan(1.0);
    int cur_image_id = image.ImageId();
    int image_diff = std::fabs(cur_image_id - last_keyframe_idx);
    if (image_diff >= options.local_region_repetitive) {
        return static_cast<float>(image.MapPointVisibilityScore());
    } else {
        float weight = 1.0 / std::atan(image_diff + tan1 - 1);
        return weight * image.MapPointVisibilityScore();
    }
}

Eigen::Vector3d PixelToUnitDepthRay(const Eigen::Vector2d& pixel, const Eigen::Matrix3d& rotation,
                                    const Camera& camera) {
    // Remove the effect of calibration.
    const Eigen::Vector2d undistorted_pixel = camera.ImageToWorld(pixel);
    const Eigen::Vector3d undistorted_point(undistorted_pixel(0), undistorted_pixel(1), 1.0);
    // Apply rotation.
    const Eigen::Vector3d direction = rotation.transpose() * undistorted_point;
    return direction;
}

float EstimateMemoryFromResiduals(size_t num_residuals) {
    float max_ram = (0.319241 * num_residuals + 47509.342775) * 1e-6;
    return max_ram;
}

void IterativeTrackSelection(const Reconstruction *reconstruction, 
                             std::vector<class MapPoint *> &mappoints,
                             const size_t max_cover_per_view) {
    for (auto &mappoint : reconstruction->MapPoints()) {
        mappoints.emplace_back((class MapPoint *)&mappoint.second);
    }

    std::sort(mappoints.begin(), mappoints.end(), [](class MapPoint *mappoint1, class MapPoint *mappoint2) {
        const class Track *track1 = &mappoint1->Track();
        const class Track *track2 = &mappoint2->Track();
        if (track1->Elements().size() == track2->Elements().size()) {
            return mappoint1->Error() < mappoint2->Error();
        } else {
            return track1->Elements().size() > track2->Elements().size();
        }
    });

    std::vector<unsigned char> inlier_masks(mappoints.size(), 0);
    std::unordered_map<image_t, int> cover_view;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (inlier_masks.at(i)) {
            continue;
        }
        auto track = mappoints.at(i)->Track();
        bool selected = false;
        for (auto track_elem : track.Elements()) {
            image_t image_id = track_elem.image_id;
            std::unordered_map<image_t, int>::iterator it = 
                cover_view.find(image_id);
            if (it == cover_view.end() || it->second < max_cover_per_view) {
                cover_view[image_id]++;
                selected = true;
                break;
            }
        }

        inlier_masks[i] = selected;

        if (selected) {
            std::unordered_set<image_t> track_images;
            for (auto track_elem : track.Elements()) {
                track_images.insert(track_elem.image_id);
            }
            for (auto image_id : track_images) {
                cover_view[image_id]++;
            }
        } else {
            mappoints.at(i) = NULL;
        }
    }
    int i, j;
    for (i = 0, j = 0; i < mappoints.size(); ++i) {
        if (mappoints[i]) {
            mappoints[j] = mappoints[i];
            j = j + 1;
        }
    }
    mappoints.resize(j);
}

void WritePly(const std::string& path,
              const std::vector<Eigen::Vector3d>& points,
              const std::vector<Eigen::Vector3i>& point_colors,
              const std::vector<Eigen::Vector2i>& edges) {
    std::ofstream file(path);
    CHECK(file.is_open()) << path;

    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "element vertex " << points.size() << std::endl;

    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;

    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;

    file << "element edge " << edges.size() << std::endl;
    file << "property int vertex1" << std::endl;
    file << "property int vertex2" << std::endl;

    file << "end_header" << std::endl;

    const size_t num_point = points.size();
    for (size_t i = 0; i < points.size(); ++i) {
        auto point = points[i];
        auto color = point_colors[i];
        file << point[0] << " " << point[1] << " " << point[2];
        file << " " << static_cast<int>(color[0]) << " "
            << static_cast<int>(color[1]) << " " << static_cast<int>(color[2]);
        file << std::endl;
    }
    for (auto edge : edges) {
        file << edge[0] << " " << edge[1] << std::endl;
    }

    file.close();
}

}  // namespace

bool IncrementalMapper::Options::Check() const {
    CHECK_OPTION_GT(init_min_num_inliers, 0);
    CHECK_OPTION_GT(init_max_error, 0.0);
    CHECK_OPTION_GE(init_max_forward_motion, 0.0);
    CHECK_OPTION_LE(init_max_forward_motion, 1.0);
    CHECK_OPTION_GE(init_min_tri_angle, 0.0);
    CHECK_OPTION_GE(init_max_reg_trials, 1);
    CHECK_OPTION_GT(abs_pose_max_error, 0.0);
    CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
    CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
    CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
    CHECK_OPTION_GE(local_ba_num_images, 2);
    CHECK_OPTION_GE(local_ba_min_tri_angle, 0.0);
    CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
    CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
    CHECK_OPTION_GE(max_extra_param, 0.0);
    CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
    CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
    CHECK_OPTION_GE(max_reg_trials, 1);
    return true;
}

IncrementalMapper::IncrementalMapper(std::shared_ptr<SceneGraphContainer> scene_graph_container)
    : scene_graph_container_(scene_graph_container),
      reconstruction_(nullptr),
      triangulator_(nullptr),
      num_total_reg_images_(0),
      num_shared_reg_images_(0),
      prev_init_image_pair_id_(kInvalidImagePairId),
      acc_min_dist_kf_(0.0f),
      seq_num_kf_(0),
      last_keyframe_idx(-1),
      global_ba_count_(0) {}

void IncrementalMapper::BeginReconstruction(std::shared_ptr<Reconstruction> reconstruction) {
    std::cout << "IncrementalMapper::BeginReconstruction" << std::endl;
    CHECK(!reconstruction_);
    reconstruction_ = reconstruction;
    reconstruction_->SetUp(scene_graph_container_);
    triangulator_.reset(new IncrementalTriangulator(scene_graph_container_->CorrespondenceGraph(), reconstruction));

    // mark the old map.
    image_t max_image_id = 0;
    const auto & all_image_ids = scene_graph_container_->GetImageIds();
    for (auto image_id : all_image_ids) {
        max_image_id = std::max(image_id, max_image_id);
    }
    fixed_images_.resize(max_image_id + 1, false);

    num_shared_reg_images_ = 0;
    for (const image_t image_id : reconstruction_->RegisterImageIds()) {
        RegisterImageEvent(image_id);
        // fixed_images_ids_.insert(image_id);
        fixed_images_[image_id] = true;
    }
    // fixed_mappoint_ids_ = reconstruction_->MapPointIds();

    prev_init_image_pair_id_ = kInvalidImagePairId;
    prev_init_two_view_geometry_ = TwoViewGeometry();

    refined_cameras_.clear();
    filtered_images_.clear();
    num_reg_trials_.clear();
    // init_image_pairs_.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
    CHECK_NOTNULL(reconstruction_.get());

    if (discard) {
        for (const image_t image_id : reconstruction_->RegisterImageIds()) {
            DeRegisterImageEvent(image_id);
        }
        image_to_lidar_map_.clear();
        keyframe_ids_.clear();
    }

    reconstruction_->TearDown();
    reconstruction_ = std::shared_ptr<Reconstruction>();
    triangulator_.reset();
}

bool IncrementalMapper::EstimateCameraOrientations(const IncrementalMapper::Options& options) {
    const auto& correspondence_graph = scene_graph_container_->CorrespondenceGraph();
    const auto& image_pairs = correspondence_graph->ImagePairs();

    std::unique_ptr<GlobalRotationEstimator> rotation_estimator;

    // Initialize the orientation estimations by walking along the maximum
    // spanning tree.
    auto nodes = OrientationsFromMaximumSpanningTree(*correspondence_graph, &orientations_);

    // Robust L1L2 Solver
    RobustRotationEstimator::Options robust_rotation_estimator_options;
    rotation_estimator.reset(new RobustRotationEstimator(robust_rotation_estimator_options));

    // Return false if the rotation estimation does not succeed.
    if (!rotation_estimator->EstimateRotations(image_pairs, &orientations_)) {
        return false;
    }

    // Set the camera orientations of all views that were successfully estimated.
    for (const auto& orientation : orientations_) {
        Eigen::Vector4d qvec;
        ceres::AngleAxisToQuaternion(orientation.second.data(), qvec.data());
        reconstruction_->Image(orientation.first).SetQvec(qvec);
    }

    // Convert to sfm coordinate system.
    const float angle_thres = std::cos(DEG2RAD(15.0f));
    auto image_ids = scene_graph_container_->GetImageIds();

    struct CorrespondenceGraph::ImagePair image_pair1, image_pair2;
    image_t ref_image_id, src_image_id1, src_image_id2;
    bool find_image_pair1 = false, find_image_pair2 = false;

    for (auto image_id : image_ids) {
//    image_id = root_id;
//{
        auto neighbor_ids = correspondence_graph->ImageNeighbor(image_id);

        ref_image_id = image_id;
        find_image_pair1 = find_image_pair2 = false;
        
        for (auto neighbor_id : neighbor_ids) {
            if (neighbor_id <= image_id) {
                continue;
            }
            image_pair1 = correspondence_graph->ImagePair(image_id, neighbor_id);
            std::cout << image_pair1.num_correspondences << std::endl;
            if (image_pair1.num_correspondences < 500) {
                continue;
            }
            auto tvec1 = image_pair1.two_view_geometry.tvec.normalized();
            image_t first_image_id = neighbor_id;
            for (auto neighbor_id : neighbor_ids) {
                if (neighbor_id <= first_image_id) {
                    continue;
                }
                image_pair2 = correspondence_graph->ImagePair(image_id, neighbor_id);
                std::cout << image_pair2.num_correspondences << " ";
                if (image_pair2.num_correspondences < 500) {
                    continue;
                }
                auto tvec2 = image_pair2.two_view_geometry.tvec.normalized();
                auto cos_thres = tvec1.dot(tvec2);
                if (cos_thres < angle_thres || cos_thres > -angle_thres) {
                    src_image_id1 = first_image_id;
                    src_image_id2 = neighbor_id;
                    find_image_pair1 = true;
                    find_image_pair2 = true;
                    break;
                }
            }
            std::cout << std::endl;
            if (find_image_pair1) {
                break;
            }
        }
        if (find_image_pair1) {
            break;
        }
    }
    if (find_image_pair1) {
//        ref_image_id = 11;
//        src_image_id1 = 18;
//        src_image_id2 = 19;
        Image ref_image = scene_graph_container_->Image(ref_image_id);
        Image src_image1 = scene_graph_container_->Image(src_image_id1);
        Image src_image2 = scene_graph_container_->Image(src_image_id2);

        std::cout << "find triple: " << ref_image_id << " " << src_image_id1 << " " << src_image_id2 << std::endl;
        std::cout << "ref: " << ref_image.Name() << std::endl;
        std::cout << "src1: " << src_image1.Name() << std::endl;
        std::cout << "src2: " << src_image2.Name() << std::endl;

        Eigen::Vector3d tvec1, tvec2;
        Eigen::Matrix3d R1, R2;
        tvec1 = (src_image1.TvecPrior() - ref_image.TvecPrior()).normalized();
        tvec2 = (src_image2.TvecPrior() - ref_image.TvecPrior()).normalized();
        Eigen::Vector3d zaxis = tvec1.cross(tvec2).normalized();
        Eigen::Vector3d yaxis = tvec1.cross(zaxis).normalized();
        Eigen::Vector3d xaxis = yaxis.cross(zaxis);
        R1.row(0) = xaxis;
        R1.row(1) = yaxis;
        R1.row(2) = zaxis;
        std::cout << R1 << std::endl;

        tvec1 = image_pair1.two_view_geometry.tvec.normalized();
        tvec2 = image_pair2.two_view_geometry.tvec.normalized();
        zaxis = tvec1.cross(tvec2).normalized();
        yaxis = tvec1.cross(zaxis).normalized();
        xaxis = yaxis.cross(zaxis);
        R2.row(0) = xaxis;
        R2.row(1) = yaxis;
        R2.row(2) = zaxis;
        std::cout << R2 << std::endl;

        Eigen::Matrix3d deltaR = R1 * R2.inverse();

        Eigen::Matrix3d root_R, ref_R;
        ceres::AngleAxisToRotationMatrix(orientations_[nodes[0]].data(), root_R.data());
        ceres::AngleAxisToRotationMatrix(orientations_[ref_image_id].data(), ref_R.data());
        auto dR = ref_R.transpose() * deltaR;

        std::cout << "Transformation to sfm coordiante system: " << std::endl;
        std::cout << dR << std::endl;

        for (const auto& orientation : orientations_) {
            auto& image = reconstruction_->Image(orientation.first);
            Eigen::Vector4d qvec = image.Qvec();
            Eigen::Vector4d trans_qvec = RotationMatrixToQuaternion((dR * QuaternionToRotationMatrix(qvec)));
            image.SetQvec(trans_qvec);
        }
    }

    return !orientations_.empty();
}

bool IncrementalMapper::FindInitialLidarPair(const image_t image_id1, const image_t image_id2, 
                                             sweep_t * sweep_id1, sweep_t * sweep_id2) {
    *sweep_id1 = FindNextSweep(image_id1);
    *sweep_id2 = FindNextSweep(image_id2);
    return (*sweep_id1 != -1) && (*sweep_id2 != -1);
}

sweep_t IncrementalMapper::FindNextSweep(const image_t image_id) {
    sweep_t sweep_id = -1;
    if (sweep_timestamps_.empty()) return sweep_id;
    
    auto image = reconstruction_->Image(image_id);
    long long image_timestamp = image.timestamp_;

    // std::cout << "image: " << image.Name() << " " << image_timestamp << ", " << std::endl;
    // std::cout << "lidar: " << sweep_timestamps_.at(0).first << ", " << sweep_timestamps_.at(0).second << std::endl;

    const std::pair<sweep_t, long long> image_timestamp_t({image_id, image_timestamp});
    
    auto next_sweep = std::lower_bound(sweep_timestamps_.begin(), sweep_timestamps_.end(), image_timestamp_t, 
        [&](const std::pair<sweep_t, long long> & a, const std::pair<sweep_t, long long> & b) {
            return a.second < b.second;
        });

    int forward = next_sweep - sweep_timestamps_.begin();
    if (forward >= sweep_timestamps_.size() || forward <= 0) {
        return -1;
    }
    std::vector<std::pair<sweep_t, long long> >::iterator prev_sweep = sweep_timestamps_.begin();
    std::advance(prev_sweep, forward - 1);
    
    long long prev_dtime = image_timestamp - prev_sweep->second;
    long long next_dtime = next_sweep->second - image_timestamp;
    if (prev_dtime > next_dtime) {
        sweep_id = next_sweep->first;
    } else {
        sweep_id = prev_sweep->first;
    }
    // if (reconstruction_->LidarSweep(sweep_id).IsRegistered()) {
    //     return -1;
    // }
    // if (sweep_id != -1) {
    //     std::cout << "lidar: " << reconstruction_->LidarSweep(sweep_id).Name() << " " << prev_sweep->second << " " << next_sweep->second << std::endl;
    // }
    return sweep_id;
}

bool IncrementalMapper::RegisterInitialLidarPair(const Options & options, 
                                                 const BundleAdjustmentOptions& ba_options, 
                                                 const image_t image_id1,
                                                 const image_t image_id2,
                                                 const sweep_t sweep_id1, 
                                                 const sweep_t sweep_id2) {
    auto & lidar_sweep1 = reconstruction_->LidarSweep(sweep_id1);
    auto & lidar_sweep2 = reconstruction_->LidarSweep(sweep_id2);

    std::cout << "lidar1: " << JoinPaths(options.lidar_path, lidar_sweep1.Name()) << std::endl;
    std::cout << "lidar2: " << JoinPaths(options.lidar_path, lidar_sweep2.Name()) << std::endl;
    
    auto pc1 = ReadPCD(JoinPaths(options.lidar_path, lidar_sweep1.Name()));
    auto pc2 = ReadPCD(JoinPaths(options.lidar_path, lidar_sweep2.Name()));
    if (pc1.info.height == 1){
        RebuildLivoxMid360(pc1, 4);
    }
    if (pc2.info.height == 1){
        RebuildLivoxMid360(pc2, 4);
    }

    // lidar_sweep1.Setup(pc1, reconstruction_->NumRegisterImages());
    // lidar_sweep2.Setup(pc2, reconstruction_->NumRegisterImages());
    lidar_sweep1.Setup(pc1, lidar_sweep1.timestamp_);
    lidar_sweep2.Setup(pc2, lidar_sweep2.timestamp_);
    // lidar_sweep1.FilterPointCloud(0.3);
    // lidar_sweep2.FilterPointCloud(0.3);

    // // TODO: estimate relative pose with IMU.
    // {
    //     if (lidar_sweep1.HasQvecPrior() && lidar_sweep1.HasTvecPrior()) {
    //         auto qvec = lidar_sweep1.QvecPrior();
    //         lidar_sweep1.SetQvec(qvec);
    //         lidar_sweep1.SetTvec(-QuaternionRotatePoint(qvec, lidar_sweep1.TvecPrior()));
    //     }
    //     if (lidar_sweep2.HasQvecPrior()) {
    //         auto qvec = lidar_sweep2.QvecPrior();
    //         lidar_sweep2.SetQvec(qvec);
    //         lidar_sweep2.SetTvec(-QuaternionRotatePoint(qvec, lidar_sweep2.TvecPrior()));
    //     }
    //     reconstruction_->RegisterLidarSweep(sweep_id1);
    //     reconstruction_->RegisterLidarSweep(sweep_id2);
    // }
    bool has_prior1 = false, has_prior2 = false;
    if (lidar_sweep1.HasQvecPrior() && lidar_sweep1.HasTvecPrior()) {
        auto qvec = lidar_sweep1.QvecPrior();
        lidar_sweep1.SetQvec(qvec);
        lidar_sweep1.SetTvec(-QuaternionRotatePoint(qvec, lidar_sweep1.TvecPrior()));
        has_prior1 = true;
    } 
    if (lidar_sweep2.HasQvecPrior() && lidar_sweep2.HasTvecPrior()) {
        auto qvec = lidar_sweep2.QvecPrior();
        lidar_sweep2.SetQvec(qvec);
        lidar_sweep2.SetTvec(-QuaternionRotatePoint(qvec, lidar_sweep2.TvecPrior()));
        has_prior2 = true;
    }

    std::vector<sweep_t> sweep_ids;
    sweep_ids.push_back(lidar_sweep1.SweepID());
    sweep_ids.push_back(lidar_sweep2.SweepID());

    std::unordered_set<sweep_t> fix_sweep_ids;
    if (has_prior1 && !has_prior2){
        lidar_sweep2.SetQvec(lidar_sweep1.Qvec());
        lidar_sweep2.SetTvec(lidar_sweep1.Tvec());

        fix_sweep_ids.insert(lidar_sweep1.SweepID());
    } else if (!has_prior1 && has_prior2){
        lidar_sweep1.SetQvec(lidar_sweep2.Qvec());
        lidar_sweep1.SetTvec(lidar_sweep2.Tvec());

        fix_sweep_ids.insert(lidar_sweep2.SweepID());
    } else if (!has_prior1 && !has_prior2){
        lidar_sweep1.SetQvec(Eigen::Vector4d(1, 0, 0, 0));
        lidar_sweep1.SetTvec(Eigen::Vector3d(0, 0, 0));
        lidar_sweep2.SetQvec(Eigen::Vector4d(1, 0, 0, 0));
        lidar_sweep2.SetTvec(Eigen::Vector3d(0, 0, 0));
    }

    if (fix_sweep_ids.empty()){
        fix_sweep_ids.insert(lidar_sweep1.SweepID());
    }
    reconstruction_->RegisterLidarSweep(sweep_id1);
    reconstruction_->RegisterLidarSweep(sweep_id2);

    if (options.debug_info){
        reconstruction_->OutputLidarPointCloud2World(
            workspace_path_ + "/init_before", true);
    }

    std::cout << "before, sweep - " << sweep_id1
        << "("<< lidar_sweep1.Qvec().transpose() 
        << ", " << lidar_sweep1.Tvec().transpose()
        << ")\nsweep - " << sweep_id2 << "(" 
        << lidar_sweep2.Qvec().transpose() 
        << ", " << lidar_sweep2.Tvec().transpose()
        << ")\n Done..." << std::endl;
    double final_cost;
    AdjustFrame2FrameBundle(options, ba_options, sweep_ids, fix_sweep_ids, final_cost);

    if (options.debug_info){
        reconstruction_->OutputLidarPointCloud2World(
            workspace_path_ + "/init_end",true);
    }

    std::cout << "after, sweep - " << sweep_id1
        << "("<< lidar_sweep1.Qvec().transpose() 
        << ", " << lidar_sweep1.Tvec().transpose()
        << ")\nsweep - " << sweep_id2 << "(" 
        << lidar_sweep2.Qvec().transpose() 
        << ", " << lidar_sweep2.Tvec().transpose()
        << ")\n RegisterInitialLidarPair Done..." << std::endl;

    image_to_lidar_map_[image_id1] = sweep_id1;
    image_to_lidar_map_[image_id2] = sweep_id2;

    return true;
}

void IncrementalMapper::ImageLidarAlignment(const Options & options, 
                                            const image_t image_id1, const image_t image_id2, 
                                            const sweep_t sweep_id1, const sweep_t sweep_id2) {
#if 0
    class Image & image1 = reconstruction_->Image(image_id1);
    class Image & image2 = reconstruction_->Image(image_id2);
    Camera &camera1 = reconstruction_->Camera(image1.CameraId());
    Camera &camera2 = reconstruction_->Camera(image2.CameraId());
    class LidarSweep & lidar_sweep1 = reconstruction_->LidarSweep(sweep_id1);
    class LidarSweep & lidar_sweep2 = reconstruction_->LidarSweep(sweep_id2);

    Eigen::Matrix4d iP1 = Eigen::Matrix4d::Identity();
    iP1.topRows(3) = image1.ProjectionMatrix();
    Eigen::Matrix4d iP2 = Eigen::Matrix4d::Identity();
    iP2.topRows(3) = image2.ProjectionMatrix();
    Eigen::Matrix4d iP12 = iP2 * iP1.inverse();
    Eigen::Vector3d itvec12 = iP12.block<3, 1>(0, 3);

    Eigen::Matrix4d h_lidar2cam_matrix = Eigen::Matrix4d::Identity();
    h_lidar2cam_matrix.topRows(3) = options.lidar_to_cam_matrix;
    
    Eigen::Matrix4d world2lidar1 = Eigen::Matrix4d::Identity();
    world2lidar1.topRows(3) = lidar_sweep1.ProjectionMatrix();
    Eigen::Matrix4d cam2world1 = world2lidar1.inverse() * h_lidar2cam_matrix.inverse();

    Eigen::Matrix4d world2lidar2 = Eigen::Matrix4d::Identity();
    world2lidar2.topRows(3) = lidar_sweep2.ProjectionMatrix();
    Eigen::Matrix4d cam2world2 = world2lidar2.inverse() * h_lidar2cam_matrix.inverse();

    Eigen::Matrix4d lP12 = cam2world2.inverse() * cam2world1;
    Eigen::Vector3d ltvec12 = lP12.block<3, 1>(0, 3);

    std::cout << ltvec12.normalized().transpose() << " " << itvec12.normalized().transpose() << std::endl;

    double scale = ltvec12.norm() / itvec12.norm();
    std::cout << "scale: " << scale << std::endl;

    Eigen::Matrix3d relative_R = cam2world1.block<3, 3>(0, 0) * iP1.block<3, 3>(0, 0);
    Eigen::Vector3d relative_tvec = cam2world1.block<3, 1>(0, 3) - image1.ProjectionCenter();

    SimilarityTransform3 tform(scale, RotationMatrixToQuaternion(relative_R), relative_tvec);

    // transform mappoints.
    const auto & mappoint_ids = reconstruction_->MapPointIds();
    for (auto mappoint_id : mappoint_ids) {
        class MapPoint & mappoint = reconstruction_->MapPoint(mappoint_id);
        Eigen::Vector3d xyz = mappoint.XYZ();
        tform.TransformPoint(&xyz);
        mappoint.SetXYZ(xyz);
    }

    tform.TransformPose(&image1.Qvec(), &image1.Tvec());  // Perfrom the calculated transform
    tform.TransformPose(&image2.Qvec(), &image2.Tvec());  // Perfrom the calculated transform

    if (camera1.NumLocalCameras() > 1){
        camera1.RescaleLocalCameraExtrinsics(scale);
    }
    if (camera2.CameraId() != camera1.CameraId() && camera2.NumLocalCameras() > 1){
        camera2.RescaleLocalCameraExtrinsics(scale);
    }

    std::cout << "image-lidar1:" << std::endl;
    auto lR1 = cam2world1.block<3, 3>(0, 0).inverse();
    auto iR1 = image1.RotationMatrix();
    
    Eigen::Matrix3d R_diff1 = (lR1.transpose()) * iR1;
    Eigen::AngleAxisd angle_axis1(R_diff1);
    double R_angle1 = angle_axis1.angle();
    std::cout << "R_diff1: " << RAD2DEG(R_angle1) << std::endl;

    std::cout << RotationMatrixToQuaternion(lR1).transpose() << " " << RotationMatrixToQuaternion(iR1).transpose() << std::endl;
    std::cout << cam2world1.block<3, 1>(0, 3).transpose() << " " << image1.ProjectionCenter().transpose() << std::endl;

    std::cout << "image-lidar2:" << std::endl;
    auto lR2 = cam2world2.block<3, 3>(0, 0).inverse();
    auto iR2 = image2.RotationMatrix();
    
    Eigen::Matrix3d R_diff2 = (lR2.transpose()) * iR2;
    Eigen::AngleAxisd angle_axis2(R_diff2);
    double R_angle2 = angle_axis2.angle();
    std::cout << "R_diff: " << RAD2DEG(R_angle2) << std::endl;

    std::cout << RotationMatrixToQuaternion(lR2).transpose() << " " << RotationMatrixToQuaternion(iR2).transpose() << std::endl;
    std::cout << cam2world2.block<3, 1>(0, 3).transpose() << " " << image2.ProjectionCenter().transpose() << std::endl;
#else
    class Image & image1 = reconstruction_->Image(image_id1);
    class Image & image2 = reconstruction_->Image(image_id2);
    Camera &camera1 = reconstruction_->Camera(image1.CameraId());
    Camera &camera2 = reconstruction_->Camera(image2.CameraId());
    class LidarSweep & lidar_sweep1 = reconstruction_->LidarSweep(sweep_id1);
    class LidarSweep & lidar_sweep2 = reconstruction_->LidarSweep(sweep_id2);

    Eigen::Matrix3x4d inv_P1 = image1.InverseProjectionMatrix();
    Eigen::Matrix3x4d inv_P2 = image2.InverseProjectionMatrix();

    Eigen::Matrix4d h_lidar2cam_matrix = Eigen::Matrix4d::Identity();
    h_lidar2cam_matrix.topRows(3) = options.lidar_to_cam_matrix;
    
    Eigen::Matrix4d world2lidar1 = Eigen::Matrix4d::Identity();
    world2lidar1.topRows(3) = lidar_sweep1.ProjectionMatrix();
    Eigen::Matrix4d world2cam1 = h_lidar2cam_matrix * world2lidar1;
    Eigen::Matrix3x4d cam2world1 = world2cam1.inverse().topRows(3);
    Eigen::Vector3d lidar_C1 = lidar_sweep1.ProjectionCenter();

    Eigen::Matrix4d world2lidar2 = Eigen::Matrix4d::Identity();
    world2lidar2.topRows(3) = lidar_sweep2.ProjectionMatrix();
    Eigen::Matrix4d world2cam2 = h_lidar2cam_matrix * world2lidar2;
    Eigen::Matrix3x4d cam2world2 = world2cam2.inverse().topRows(3);

    std::vector<Eigen::Vector3d> cam_points, lidar_points;
    cam_points.reserve(8);
    lidar_points.reserve(8);

    double scale1 = (image1.ProjectionCenter() - image2.ProjectionCenter()).norm();
    if (scale1 < 1e-6) {
        scale1 = 1.0;
    }
    double scale2 = (lidar_sweep1.ProjectionCenter() - lidar_sweep2.ProjectionCenter()).norm();
    if (scale2 < 1e-6) {
        scale2 = 1.0;
    }

    std::cout << "scale1: " << scale1 << std::endl;
    std::cout << "scale2: " << scale2 << std::endl;

    Eigen::Vector4d orig(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d x_axis(scale1 / 3, 0.0, 0.0, 1.0);
    Eigen::Vector4d y_axis(0.0, scale1 / 3, 0.0, 1.0);
    Eigen::Vector4d z_axis(0.0, 0.0, scale1 / 3, 1.0);
    // Eigen::Vector4d x_axis(1.0, 0.0, 0.0, 1.0);
    // Eigen::Vector4d y_axis(0.0, 1.0, 0.0, 1.0);
    // Eigen::Vector4d z_axis(0.0, 0.0, 1.0, 1.0);

    Eigen::Vector3d p0, p1, p2, p3;
    p0 = inv_P1 * orig;
    p1 = inv_P1 * x_axis;
    p2 = inv_P1 * y_axis;
    p3 = inv_P1 * z_axis;
    cam_points.push_back(p0);
    cam_points.push_back(p1);
    cam_points.push_back(p2);
    cam_points.push_back(p3);
    p0 = inv_P2 * orig;
    p1 = inv_P2 * x_axis;
    p2 = inv_P2 * y_axis;
    p3 = inv_P2 * z_axis;
    cam_points.push_back(p0);
    cam_points.push_back(p1);
    cam_points.push_back(p2);
    cam_points.push_back(p3);

    x_axis = Eigen::Vector4d(scale2 / 3, 0.0, 0.0, 1.0);
    y_axis = Eigen::Vector4d(0.0, scale2 / 3, 0.0, 1.0);
    z_axis = Eigen::Vector4d(0.0, 0.0, scale2 / 3, 1.0);
    p0 = cam2world1 * orig;
    p1 = cam2world1 * x_axis;
    p2 = cam2world1 * y_axis;
    p3 = cam2world1 * z_axis;
    lidar_points.push_back(p0);
    lidar_points.push_back(p1);
    lidar_points.push_back(p2);
    lidar_points.push_back(p3);
    p0 = cam2world2 * orig;
    p1 = cam2world2 * x_axis;
    p2 = cam2world2 * y_axis;
    p3 = cam2world2 * z_axis;
    lidar_points.push_back(p0);
    lidar_points.push_back(p1);
    lidar_points.push_back(p2);
    lidar_points.push_back(p3);

#if 0
    const double max_error = 0.3;
    const double max_error2 = max_error * max_error;

    RANSACOptions ransac_options;
    ransac_options.max_error = max_error;
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.9;
    ransac_options.min_num_trials = 200;
    // ransac_options.max_num_trials = 10000;
  
    LORANSAC<CameraAlignmentEstimator, CameraAlignmentEstimator> ransac(ransac_options);

    auto report = ransac.Estimate(cam_points, lidar_points);
    Eigen::Matrix3x4d trans = report.model;

    std::cout << "num_inliers: " << report.support.num_inliers << std::endl;

    size_t num_inlier = 0;
    double average_residual = 0.0, average_residual_inlier = 0.0;
    for (size_t i = 0; i < cam_points.size(); ++i) {
        Eigen::Vector3d trans_point = trans * cam_points[i].homogeneous();
        double error = (trans_point - lidar_points[i]).norm();
        average_residual += error;
        if (error < max_error) {
            average_residual_inlier += error;
            num_inlier++;
        }
    }
    std::cout<<"average residuals: "<<sqrt(average_residual /  cam_points.size())<<std::endl;
    if (num_inlier > 0) {
        std::cout<<"average residuals inliers: "<<sqrt(average_residual_inlier / num_inlier)<<std::endl;
    }
#else
    CameraAlignmentEstimator estimator;
    std::vector<Eigen::Matrix3x4d> models = estimator.Estimate(cam_points, lidar_points);
    Eigen::Matrix3x4d trans = models[0];
#endif

    SimilarityTransform3 tform(trans);

    double scale = tform.Scale();

    std::cout << "Scale: " << tform.Scale() << std::endl;
    std::cout << "Rotation: " << tform.Rotation().transpose() << std::endl;
    std::cout << "Translation: " << tform.Translation().transpose() << std::endl;

    // transform mappoints.
    const auto & mappoint_ids = reconstruction_->MapPointIds();
    for (auto mappoint_id : mappoint_ids) {
        class MapPoint & mappoint = reconstruction_->MapPoint(mappoint_id);
        Eigen::Vector3d xyz = mappoint.XYZ();
        tform.TransformPoint(&xyz);
        mappoint.SetXYZ(xyz);
    }

    // tform.TransformPose(&image1.Qvec(), &image1.Tvec());  // Perfrom the calculated transform
    // tform.TransformPose(&image2.Qvec(), &image2.Tvec());  // Perfrom the calculated transform
    const auto registered_image_ids = reconstruction_->RegisterImageIds();
    std::unordered_set<camera_t> camera_ids;
    for (auto register_image_id : registered_image_ids) {
        class Image & image = reconstruction_->Image(register_image_id);
        tform.TransformPose(&image.Qvec(), &image.Tvec());  // Perfrom the calculated transform
        if (camera_ids.find(image.CameraId()) == camera_ids.end()) {
            camera_ids.insert(image.CameraId());
        }
    }
    // for (auto camera_id : camera_ids) {
    //     class Camera & camera = reconstruction_->Camera(camera_id);
    //     if (camera.NumLocalCameras() > 1){
    //         camera.RescaleLocalCameraExtrinsics(scale);
    //     }
    // }

    std::cout << "image-lidar1:" << std::endl;
    auto lR1 = world2cam1.block<3, 3>(0, 0);
    auto iR1 = image1.RotationMatrix();
    
    Eigen::Matrix3d R_diff1 = (lR1.transpose()) * iR1;
    Eigen::AngleAxisd angle_axis1(R_diff1);
    double R_angle1 = angle_axis1.angle();
    std::cout << "R_diff1: " << RAD2DEG(R_angle1) << std::endl;

    std::cout << RotationMatrixToQuaternion(lR1).transpose() << " " << RotationMatrixToQuaternion(iR1).transpose() << std::endl;
    std::cout << -(lR1.transpose() * world2cam1.block<3, 1>(0, 3)).transpose() << " " << image1.ProjectionCenter().transpose() << std::endl;
    
    std::cout << "image-lidar2:" << std::endl;
    auto lR2 = world2cam2.block<3, 3>(0, 0);
    auto iR2 = image2.RotationMatrix();
    
    Eigen::Matrix3d R_diff2 = (lR2.transpose()) * iR2;
    Eigen::AngleAxisd angle_axis2(R_diff2);
    double R_angle2 = angle_axis2.angle();
    std::cout << "R_diff2: " << RAD2DEG(R_angle2) << std::endl;

    std::cout << RotationMatrixToQuaternion(lR2).transpose() << " " << RotationMatrixToQuaternion(iR2).transpose() << std::endl;
    std::cout << -(lR2.transpose() * world2cam2.block<3, 1>(0, 3)).transpose() << " " << image2.ProjectionCenter().transpose() << std::endl;
#endif
}

void IncrementalMapper::RefineImageLidarAlignment(const Options & options) {
    
    Eigen::Matrix4d h_lidar2cam_matrix = Eigen::Matrix4d::Identity();
    h_lidar2cam_matrix.topRows(3) = reconstruction_->lidar_to_cam_matrix;

    if (0) {
        std::vector<double> scales;
        for (const auto image_lidar_pair : image_to_lidar_map_) {
            image_t image_id = image_lidar_pair.first;
            sweep_t sweep_id = image_lidar_pair.second;
            class Image & image = reconstruction_->Image(image_id);
            class LidarSweep & lidar_sweep = reconstruction_->LidarSweep(sweep_id);
            if (!image.IsRegistered() || !lidar_sweep.IsRegistered()) {
                continue;
            }
            Eigen::Vector3d C = image.ProjectionCenter();
            Eigen::Matrix4d world2lidar = Eigen::Matrix4d::Identity();
            world2lidar.topRows(3) = lidar_sweep.ProjectionMatrix();
            Eigen::Matrix4d world2cam = h_lidar2cam_matrix * world2lidar;
            Eigen::Vector3d lidar_C = -world2cam.block<3, 3>(0, 0).transpose() * world2cam.block<3, 1>(0, 3);

            double min_dist = std::numeric_limits<double>::max();
            const auto & images_neighbor = scene_graph_container_->CorrespondenceGraph()->ImageNeighbor(image_id);
            for (auto neighbor_image_id : images_neighbor) {
                class Image & neighbor_image = reconstruction_->Image(neighbor_image_id);
                Eigen::Vector3d neighbor_C = neighbor_image.ProjectionCenter();
                double dist = (C - neighbor_C).norm();
                if (dist < 1e-3) {
                    continue;
                }
                if (image_to_lidar_map_.find(neighbor_image_id) != image_to_lidar_map_.end()) {
                    sweep_t neighbor_sweep_id = image_to_lidar_map_[neighbor_image_id];
                    class LidarSweep & neighbor_lidar_sweep = reconstruction_->LidarSweep(neighbor_sweep_id);
                    Eigen::Matrix4d world2lidar = Eigen::Matrix4d::Identity();
                    world2lidar.topRows(3) = neighbor_lidar_sweep.ProjectionMatrix();
                    Eigen::Matrix4d world2cam = h_lidar2cam_matrix * world2lidar;
                    Eigen::Vector3d neighbor_lidar_C = -world2cam.block<3, 3>(0, 0).transpose() * world2cam.block<3, 1>(0, 3);

                    double scale = (lidar_C - neighbor_lidar_C).norm() / dist;
                    scales.push_back(scale);
                }
            }
        }
        
        double s = 1.0;
        Eigen::Vector4d q(1.0, 0.0, 0.0, 0.0);
        Eigen::Vector3d t(0.0, 0.0, 0.0);

        if (scales.size() > 0) {
            const std::size_t nth = scales.size() / 2;
            std::nth_element(scales.begin(), scales.begin() + nth, scales.end());
            s = scales[nth];
        }

        std::cout << "optimized scale: " << s << std::endl;
        std::cout << "optimized qvec: " << q.transpose() << std::endl;
        std::cout << "optimized tvec: " << t.transpose() << std::endl;

        SimilarityTransform3 tform(s, q, t);

        // transform mappoints.
        const auto & mappoint_ids = reconstruction_->MapPointIds();
        for (auto mappoint_id : mappoint_ids) {
            class MapPoint & mappoint = reconstruction_->MapPoint(mappoint_id);
            Eigen::Vector3d xyz = mappoint.XYZ();
            tform.TransformPoint(&xyz);
            mappoint.SetXYZ(xyz);
        }

        std::unordered_set<camera_t> camera_ids;
        const std::vector<image_t> registered_image_ids = reconstruction_->RegisterImageIds();
        for (const auto image_id : registered_image_ids) {
            class Image & image = reconstruction_->Image(image_id);
            tform.TransformPose(&image.Qvec(), &image.Tvec());  // Perfrom the calculated transform

            camera_ids.insert(image.CameraId());
        }
        // for (auto camera_id : camera_ids) {
        //     class Camera & camera = reconstruction_->Camera(camera_id);
        //     if (camera.NumLocalCameras() > 1){
        //         camera.RescaleLocalCameraExtrinsics(s);
        //     }
        // }
    }

    reconstruction_->ComputeBaselineDistance();
    double m_baseline_distance = reconstruction_->baseline_distance;

    Eigen::Vector4d orig(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d x_axis(m_baseline_distance / 3.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d y_axis(0.0, m_baseline_distance / 3.0, 0.0, 1.0);
    Eigen::Vector4d z_axis(0.0, 0.0, m_baseline_distance / 3.0, 1.0);

    std::vector<Eigen::Vector3d> cam_points, lidar_points;
    cam_points.reserve(image_to_lidar_map_.size() * 4);
    lidar_points.reserve(image_to_lidar_map_.size() * 4);

    for (const auto image_lidar_pair : image_to_lidar_map_) {
        image_t image_id = image_lidar_pair.first;
        sweep_t sweep_id = image_lidar_pair.second;
        class Image & image = reconstruction_->Image(image_id);
        class LidarSweep & lidar_sweep = reconstruction_->LidarSweep(sweep_id);
        if (!image.IsRegistered() || !lidar_sweep.IsRegistered()) {
            continue;
        }

        Eigen::Matrix3x4d inv_P = image.InverseProjectionMatrix();

        Eigen::Matrix4d world2lidar = Eigen::Matrix4d::Identity();
        world2lidar.topRows(3) = lidar_sweep.ProjectionMatrix();
        Eigen::Matrix3x4d cam2world = (h_lidar2cam_matrix * world2lidar).inverse().topRows(3);

        Eigen::Vector3d p0, p1, p2, p3;
        p0 = inv_P * orig;
        p1 = inv_P * x_axis;
        p2 = inv_P * y_axis;
        p3 = inv_P * z_axis;
        cam_points.push_back(p0);
        cam_points.push_back(p1);
        cam_points.push_back(p2);
        cam_points.push_back(p3);

        p0 = cam2world * orig;
        p1 = cam2world * x_axis;
        p2 = cam2world * y_axis;
        p3 = cam2world * z_axis;
        lidar_points.push_back(p0);
        lidar_points.push_back(p1);
        lidar_points.push_back(p2);
        lidar_points.push_back(p3);
    }

    const double max_error = 0.3;

    RANSACOptions ransac_options;
    ransac_options.max_error = max_error;
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.25;
    ransac_options.min_num_trials = 200;
    // ransac_options.max_num_trials = 10000;
  
    LORANSAC<CameraAlignmentEstimator, CameraAlignmentEstimator> ransac(ransac_options);

    auto report = ransac.Estimate(cam_points, lidar_points);
    Eigen::Matrix3x4d trans = report.model;

    std::cout << "num_inliers: " << report.support.num_inliers << std::endl;

    size_t num_inlier = 0;
    double average_residual = 0.0, average_residual_inlier = 0.0;
    for (size_t i = 0; i < cam_points.size(); ++i) {
        Eigen::Vector3d trans_point = trans * cam_points[i].homogeneous();
        double error = (trans_point - lidar_points[i]).norm();
        average_residual += error;
        if (error < max_error) {
            average_residual_inlier += error;
            num_inlier++;
        }
    }
    std::cout<<"average residuals: "<<sqrt(average_residual /  cam_points.size())<<std::endl;
    if (num_inlier > 0) {
        std::cout<<"average residuals inliers: "<<sqrt(average_residual_inlier / num_inlier)<<std::endl;
    }

    SimilarityTransform3 tform(trans);

    double scale = tform.Scale();

    std::cout << "Optimized scale: " << tform.Scale() << std::endl;
    std::cout << "Optimized rotation: " << tform.Rotation().transpose() << std::endl;
    std::cout << "Optimized translation: " << tform.Translation().transpose() << std::endl;

    // transform mappoints.
    const auto & mappoint_ids = reconstruction_->MapPointIds();
    for (auto mappoint_id : mappoint_ids) {
        class MapPoint & mappoint = reconstruction_->MapPoint(mappoint_id);
        Eigen::Vector3d xyz = mappoint.XYZ();
        tform.TransformPoint(&xyz);
        mappoint.SetXYZ(xyz);
    }

    std::unordered_set<camera_t> camera_ids;
    const std::vector<image_t> registered_image_ids = reconstruction_->RegisterImageIds();
    for (const auto image_id : registered_image_ids) {
        class Image & image = reconstruction_->Image(image_id);
        tform.TransformPose(&image.Qvec(), &image.Tvec());  // Perfrom the calculated transform

        camera_ids.insert(image.CameraId());
    }
    // for (auto camera_id : camera_ids) {
    //     class Camera & camera = reconstruction_->Camera(camera_id);
    //     if (camera.NumLocalCameras() > 1){
    //         camera.RescaleLocalCameraExtrinsics(scale);
    //     }
    // }
}

bool IncrementalMapper::FindInitialImagePair(const Options& options, image_t* image_id1, image_t* image_id2) {
    CHECK(options.Check());

    const auto correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    std::vector<image_t> image_ids1;
    if (*image_id1 != kInvalidImageId && *image_id2 == kInvalidImageId) {
        // Only image_id1 provided.
        if (!scene_graph_container_->ExistsImage(*image_id1)) {
            return false;
        }
        image_ids1.push_back(*image_id1);
    } else if (*image_id1 == kInvalidImageId && *image_id2 != kInvalidImageId) {
        // Only image_id2 provided.
        if (!scene_graph_container_->ExistsImage(*image_id2)) {
            return false;
        }
        image_ids1.push_back(*image_id2);
    } else {
        // No initial seed image provided.
        image_ids1 = FindFirstInitialImage(options);
    }

    std::cout<<"image_ids1 size:  "<<image_ids1.size()<<std::endl;
    
    std::list<image_pair_t> init_image_pairs_list1;
    std::list<image_pair_t> init_image_pairs_list2;
    // Try to find good initial pair.
    for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
        *image_id1 = image_ids1[i1];

        const std::vector<image_t> image_ids2 = FindSecondInitialImage(options, *image_id1);

        for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
            *image_id2 = image_ids2[i2];
            if (image_id1 == image_id2) {
                continue;
            }

            const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(*image_id1, *image_id2);

            // Try every pair only once.
            if (init_image_pairs_.count(pair_id) > 0) {
                continue;
            }

            const auto& camera_1 = scene_graph_container_->Camera(scene_graph_container_->Image(*image_id1).CameraId());
            const auto& camera_2 = scene_graph_container_->Camera(scene_graph_container_->Image(*image_id2).CameraId());
            bool rig_flag = camera_1.NumLocalCameras() > 1 && camera_2.NumLocalCameras() > 1 &&
                            (camera_1.NumLocalCameras() == camera_2.NumLocalCameras());
            bool pano_flag = camera_1.ModelName().compare("SPHERICAL") == 0 && camera_2.ModelName().compare("SPHERICAL") == 0;
            bool perspect_flag = camera_1.NumLocalCameras() == 1 && camera_2.NumLocalCameras() == 1;
            if (!(rig_flag || pano_flag || perspect_flag)){
                continue;
            }

            auto image_pair = correspondence_graph->ImagePair(*image_id1, *image_id2);
            if (image_pair.two_view_geometry.config == TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC ||
                image_pair.two_view_geometry.config == TwoViewGeometry::ConfigurationType::PLANAR ||
                image_pair.two_view_geometry.config == TwoViewGeometry::ConfigurationType::PANORAMIC) {
                init_image_pairs_list2.push_back(pair_id);
            } else {
                init_image_pairs_list1.push_back(pair_id);
            }
        }
    }

    std::copy(init_image_pairs_list2.begin(), init_image_pairs_list2.end(), std::back_inserter(init_image_pairs_list1));

    for (const auto& pair_id : init_image_pairs_list1) {
        utility::PairIdToImagePair(pair_id, image_id1, image_id2);
        auto image_pair = correspondence_graph->ImagePair(*image_id1, *image_id2);

        init_image_pairs_.insert(pair_id);

        std::cout << *image_id1 << " " << *image_id2 << ", config = " << image_pair.two_view_geometry.config
                  << std::endl;

        if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
            init_image_id1 = *image_id1;
            init_image_id2 = *image_id2;
            return true;
        }
    }

    // No suitable pair found in entire dataset.
    *image_id1 = kInvalidImageId;
    *image_id2 = kInvalidImageId;
    init_image_id1 = kInvalidImageId;
    init_image_id2 = kInvalidImageId;

    return false;
}

bool IncrementalMapper::FindInitialImagePairOfflineSLAM(const Options& options, image_t* image_id1,
                                                        image_t* image_id2) {
    CHECK(options.Check());

    const auto& images = reconstruction_->Images();
    image_t min_image_id = std::numeric_limits<image_t>::max();
    image_t max_image_id = std::numeric_limits<image_t>::min();
    for (const auto& image : images) {
        if (min_image_id > image.first) {
            min_image_id = image.first;
        }
        if (max_image_id < image.first){
            max_image_id = image.first;
        }
    }
    
    const auto correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    for (image_t i = 0; i < reconstruction_->NumImages(); ++i) {

        // Only use images for initialization that are not registered in any
        // of the other reconstructions.
        if (num_registrations_.count(min_image_id + i) > 0 && num_registrations_.at(min_image_id + i) > 0) {
            continue;
        }
        *image_id1 = min_image_id + i;
        int max_interval = (i + options.initial_two_frame_interval_offline_slam > (reconstruction_->NumImages()-1))
                                ? (reconstruction_->NumImages()-1)
                                : (i + options.initial_two_frame_interval_offline_slam);

        if (!reconstruction_->ExistsImage(*image_id1)) {
            continue;
        }
        for (image_t j = i + 1; j <= max_interval; ++j) {
            
            if (num_registrations_.count(min_image_id + j) > 0 && num_registrations_.at(min_image_id + j) > 0) {
                continue;
            }
            
            *image_id2 = min_image_id + j;
            if (!reconstruction_->ExistsImage(*image_id2)) {
                continue;
            }

            if (correspondence_graph->ExistImagePair(*image_id1, *image_id2)) {
                auto image_pair = correspondence_graph->ImagePair(*image_id1, *image_id2);
                if (image_pair.num_correspondences < options.init_min_num_inliers) {
                    continue;
                }

                if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
                    init_image_id1 = *image_id1;
                    init_image_id2 = *image_id2;
                    return true;
                }
            }
        }
    }

    *image_id1 = kInvalidImageId;
    *image_id2 = kInvalidImageId;
    init_image_id1 = kInvalidImageId;
    init_image_id2 = kInvalidImageId;

    return false;
}

bool IncrementalMapper::FindInitialImagePairUncertainty(const Options& options, image_t* image_id1,
                                                        image_t* image_id2) {
    std::cout << "Building feature distribution histogram!" << std::endl;

    auto GaussianWeight = [&](double x, double y, double delta) {
        double m = (x + y) * 0.5;
        double wx = std::exp(-0.5 * (x - m) * (x - m) / (delta * delta));
        double wy = std::exp(-0.5 * (y - m) * (y - m) / (delta * delta));
        return wx * wy;
    };

    // Weighted image pair structure
    typedef struct WImagePair {
        int config;
        double dist_w;
        double disparity;
        image_pair_t pair_id;
    } WImagePair;

    const size_t bin_size = options.init_bin_size;
    const size_t max_matches_bin = options.max_matches_each_bin;

    std::vector<WImagePair> weighted_image_pairs;

    const class std::shared_ptr<class CorrespondenceGraph>& correspondence_graph =
        scene_graph_container_->CorrespondenceGraph();

    const EIGEN_STL_UMAP(image_pair_t, class CorrespondenceGraph::ImagePair)& image_pairs =
        correspondence_graph->ImagePairs();

    for (const auto& image_pair : image_pairs) {
        image_t tmp_image_id1, tmp_image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &tmp_image_id1, &tmp_image_id2);

        if ((num_registrations_.count(tmp_image_id1) > 0 && 
            num_registrations_.at(tmp_image_id1) > 0)||
            (num_registrations_.count(tmp_image_id2) > 0 && 
            num_registrations_.at(tmp_image_id2) > 0)) {
            continue;
        }

        point2D_t num_corrs = correspondence_graph->NumCorrespondencesBetweenImages(tmp_image_id1, tmp_image_id2);
        if (num_corrs <= options.init_min_num_inliers) {
            continue;
        }

        FeatureMatches feature_matches =
            correspondence_graph->FindCorrespondencesBetweenImages(tmp_image_id1, tmp_image_id2);
        const class Image& image1 = reconstruction_->Image(tmp_image_id1);
        const class Image& image2 = reconstruction_->Image(tmp_image_id2);
        const class Camera& camera1 = reconstruction_->Camera(image1.CameraId());
        const class Camera& camera2 = reconstruction_->Camera(image2.CameraId());
        size_t width1 = camera1.Width();
        size_t height1 = camera1.Height();
        size_t width2 = camera2.Width();
        size_t height2 = camera2.Height();

        const size_t hist_width1 = width1 / bin_size + 1;
        const size_t hist_height1 = height1 / bin_size + 1;
        Eigen::MatrixXi histogram1 = Eigen::MatrixXi::Zero(hist_height1, hist_width1);

        const size_t hist_width2 = width2 / bin_size + 1;
        const size_t hist_height2 = height2 / bin_size + 1;
        Eigen::MatrixXi histogram2 = Eigen::MatrixXi::Zero(hist_height2, hist_width2);

        std::vector<double> disparitys;
        double hist_val1 = 0.0, hist_val2 = 0.0;
        for (const auto& feature_match : feature_matches) {
            const class Point2D& point2D1 = image1.Point2D(feature_match.point2D_idx1);
            const class Point2D& point2D2 = image2.Point2D(feature_match.point2D_idx2);

            double disparity = (point2D1.XY() - point2D2.XY()).squaredNorm();
            disparitys.push_back(disparity);

            int bin_x1 = point2D1.X() / bin_size;
            int bin_y1 = point2D1.Y() / bin_size;
            if (bin_x1 < 0 || bin_x1 >= hist_width1 || bin_y1 < 0 || bin_y1 >= hist_height1) {
                continue;
            }
            if (histogram1(bin_y1, bin_x1) < max_matches_bin) {
                histogram1(bin_y1, bin_x1)++;
                hist_val1++;
            }

            int bin_x2 = point2D2.X() / bin_size;
            int bin_y2 = point2D2.Y() / bin_size;
            if (bin_x2 < 0 || bin_x2 >= hist_width2 || bin_y2 < 0 || bin_y2 >= hist_height2) {
                continue;
            }
            if (histogram2(bin_y2, bin_x2) < max_matches_bin) {
                histogram2(bin_y2, bin_x2)++;
                hist_val2++;
            }
        }

        // double m_disparity = Median(disparitys);
        // double init_min_disparity2 =
        //   options.init_min_disparity * options.init_min_disparity;
        // if (m_disparity < init_min_disparity2) {
        //    continue;
        //}

        double dist_val1 = 0.0, dist_val2 = 0.0;
        for (size_t i = 0; i < hist_height1; ++i) {
            for (size_t j = 0; j < hist_width1; ++j) {
                dist_val1 = histogram1(i, j) > 0 ? dist_val1 + 1 : dist_val1;
            }
        }
        for (size_t i = 0; i < hist_height2; ++i) {
            for (size_t j = 0; j < hist_width2; ++j) {
                dist_val2 = histogram2(i, j) > 0 ? dist_val2 + 1 : dist_val2;
            }
        }

        double dist_w = GaussianWeight(dist_val1, dist_val2, options.gauss_weight_bin);
        double hist_w = GaussianWeight(hist_val1, hist_val2, options.gauss_weight_hist);
        double w = dist_w * hist_w * std::min(dist_val1, dist_val2);

        WImagePair w_image_pair;
        w_image_pair.config = image_pair.second.two_view_geometry.config;
        w_image_pair.dist_w = w;
        // w_image_pair.disparity = m_disparity;
        w_image_pair.pair_id = image_pair.first;
        weighted_image_pairs.emplace_back(w_image_pair);
    }

    std::sort(weighted_image_pairs.begin(), weighted_image_pairs.end(),
              [&](WImagePair& image_pair1, WImagePair& image_pair2) {
                  if (image_pair1.config == image_pair2.config) {
                      return image_pair1.dist_w > image_pair2.dist_w;
                  } else {
                      return image_pair1.config < image_pair2.config;
                  }
              });
    for (const auto& image_pair : weighted_image_pairs) {
        const image_pair_t pair_id = image_pair.pair_id;
        sensemap::utility::PairIdToImagePair(pair_id, image_id1, image_id2);

        const class Image& image1 = reconstruction_->Image(*image_id1);
        const class Image& image2 = reconstruction_->Image(*image_id2);

        std::cout << "ImagePair#[" << image1.Name() << ", " << image2.Name() << "], w = " << image_pair.dist_w
                  << std::endl;

        // Try every pair only once.
        if (init_image_pairs_.count(pair_id) > 0) {
            continue;
        }
        init_image_pairs_.insert(pair_id);

        const auto& camera_1 = reconstruction_->Camera(image1.CameraId());
        const auto& camera_2 = reconstruction_->Camera(image2.CameraId());
        bool rig_flag = camera_1.NumLocalCameras() > 1 && camera_2.NumLocalCameras() > 1 &&
                        (camera_1.NumLocalCameras() == camera_2.NumLocalCameras());
        bool pano_flag = camera_1.ModelName().compare("SPHERICAL") == 0 && camera_2.ModelName().compare("SPHERICAL") == 0;
        bool perspect_flag = camera_1.NumLocalCameras() == 1 && camera_2.NumLocalCameras() == 1;
        if (!(rig_flag || pano_flag || perspect_flag)){
            continue;
        }

        if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
            init_image_id1 = *image_id1;
            init_image_id2 = *image_id2;
            return true;
        }
    }

    // No suitable pair found in entire dataset.
    *image_id1 = kInvalidImageId;
    *image_id2 = kInvalidImageId;
    init_image_id1 = kInvalidImageId;
    init_image_id2 = kInvalidImageId;

    return false;
}

bool IncrementalMapper::FindInitialImagePairWithKnownOrientation(const Options& options, image_t* image_id1,
                                                                 image_t* image_id2) {
    // Sort the view pairs by the number of geometrically verified matches.
    auto candidate_initial_image_id_pairs = OrderInitialImagePair(options);
    if (candidate_initial_image_id_pairs.empty()) {
        return false;
    }

    // Try to initialize the reconstruction from the candidate view pairs. An
    // initial seed is only considered valid if the baseline relative to the 3D
    // point depths is sufficient. This robustness is measured by the angle of
    // all 3D points.
    for (const auto& image_id_pair : candidate_initial_image_id_pairs) {
        sensemap::utility::PairIdToImagePair(image_id_pair, image_id1, image_id2);

        const auto& camera_1 = scene_graph_container_->Camera(scene_graph_container_->Image(*image_id1).CameraId());
        const auto& camera_2 = scene_graph_container_->Camera(scene_graph_container_->Image(*image_id2).CameraId());
        bool rig_flag = camera_1.NumLocalCameras() > 1 && camera_2.NumLocalCameras() > 1 &&
                        (camera_1.NumLocalCameras() == camera_2.NumLocalCameras());
        bool pano_flag = camera_1.ModelName().compare("SPHERICAL") == 0 && camera_2.ModelName().compare("SPHERICAL") == 0;
        bool perspect_flag = camera_1.NumLocalCameras() == 1 && camera_2.NumLocalCameras() == 1;
        if (!(rig_flag || pano_flag || perspect_flag)){
            continue;
        }

        if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
            init_image_id1 = *image_id1;
            init_image_id2 = *image_id2;
            return true;
        }
    }

    // No suitable pair found in entire dataset.
    *image_id1 = kInvalidImageId;
    *image_id2 = kInvalidImageId;
    init_image_id1 = kInvalidImageId;
    init_image_id2 = kInvalidImageId;

    return false;
}

std::vector<image_pair_t> IncrementalMapper::OrderInitialImagePair(const Options& options) {
    const auto& image_pairs = scene_graph_container_->CorrespondenceGraph()->ImagePairs();
    // Choose the initialization criterion based on the estimated triangulation
    // angle between the views and the number of features matched between the
    // view pairs.
    std::vector<std::tuple<int, point2D_t, image_pair_t>> initialization_criterion_for_view_pairs;
    initialization_criterion_for_view_pairs.reserve(image_pairs.size());
    for (const auto& image_pair : image_pairs) {
        // Estimate the triangulation angle if the orientations are known. If either
        // one of the orientations are not known then set the angle to zero. This
        // will push this view pair to the back of the list of camera
        // initialization, allowing it to still be used if all pairs with known
        // orientation fail.
        double median_triangulation_angle = 0;
        image_t image_id1;
        image_t image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        if (orientations_.find(image_id1) != orientations_.end() &&
            orientations_.find(image_id2) != orientations_.end()) {
            median_triangulation_angle = ComputeMedianTriangulationAngle(image_id1, image_id2);
        }
        // Take the scaled sqrt of the triangulation angle and round it to the
        // nearest integer. This essentially buckets the triangulation angles in a
        // geometric sequence. We additionally cap the triangulation angle at 45
        // degrees to prevent favoring view pairs with too wide of a baseline that
        // do not have sufficiently many matched features. This allows us to choose
        // the view pair with the most number of features that have a sufficiently
        // large triangulation angle.
        const int normalized_triangulation_angle = static_cast<const int>(
            std::round(2.0 * std::sqrt(std::min(median_triangulation_angle, options.max_triangulation_angle_degrees))));

        // TODO(csweeney): Prefer view pairs with known intrinsics.
        if (image_pair.second.num_correspondences > options.init_min_num_inliers) {
            // Insert negative values so that the largest triangulation angles and
            // highest number of matches appear at the front.
            initialization_criterion_for_view_pairs.emplace_back(
                -normalized_triangulation_angle, -image_pair.second.num_correspondences, image_pair.first);
        }
    }
    // Sort the views such that the view pairs with the largest triangulation
    // angles and the most matched features appear at the front.
    std::vector<image_pair_t> image_id_pairs;
    std::sort(initialization_criterion_for_view_pairs.begin(), initialization_criterion_for_view_pairs.end());
    for (int i = 0; i < initialization_criterion_for_view_pairs.size(); i++) {
        image_t image_id1;
        image_t image_id2;
        sensemap::utility::PairIdToImagePair(std::get<2>(initialization_criterion_for_view_pairs[i]), &image_id1,
                                             &image_id2);
        //		std::cout<<"ordered: "<<image_id1<<" "<<image_id2<<" "
        //		         <<-std::get<0>(initialization_criterion_for_view_pairs[i])<<" "
        //		         <<-std::get<1>(initialization_criterion_for_view_pairs[i])<<" "
        //		         <<std::endl;

        image_id_pairs.emplace_back(std::get<2>(initialization_criterion_for_view_pairs[i]));
    }
    return image_id_pairs;
}

double IncrementalMapper::ComputeMedianTriangulationAngle(const image_t image_id1, const image_t image_id2) {
    const Image& image1 = reconstruction_->Image(image_id1);
    const Image& image2 = reconstruction_->Image(image_id2);
    const Camera& camera1 = reconstruction_->Camera(image1.CameraId());
    const Camera& camera2 = reconstruction_->Camera(image2.CameraId());

    // Compute the angle between the viewing rays.
    const Eigen::Vector2d camera1_principal_point(camera1.PrincipalPointX(), camera1.PrincipalPointY());
    auto rotation1 = image1.RotationMatrix();
    const Eigen::Vector3d view1_ray = PixelToUnitDepthRay(camera1_principal_point, rotation1, camera1).normalized();
    const Eigen::Vector2d camera2_principal_point(camera2.PrincipalPointX(), camera2.PrincipalPointY());
    auto rotation2 = image2.RotationMatrix();
    const Eigen::Vector3d view2_ray = PixelToUnitDepthRay(camera2_principal_point, rotation2, camera2).normalized();
    return std::abs(RadToDeg(std::acos(view1_ray.dot(view2_ray))));
}

std::vector<std::pair<image_t, float>> IncrementalMapper::FindNextImages(const Options& options) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());
    std::cout<<"Find next images"<<std::endl;
    std::function<float(const Image&)> rank_image_func;
    switch (options.image_selection_method) {
        case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
            rank_image_func = RankNextImageMaxVisiblePointsNum;
            break;
        case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
            rank_image_func = RankNextImageMaxVisiblePointsRatio;
            break;
        case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
            rank_image_func = RankNextImageMinUncertainty;
            break;
    }

    std::vector<std::pair<image_t, float>> image_ranks;
    std::vector<std::pair<image_t, float>> other_image_ranks;

    // const class Image & last_image = reconstruction_->Image(offline_slam_last_frame_id_);

    size_t has_pose_count = 0;
    size_t not_enough_visible_mappoints_count = 0;
    size_t too_many_trials_count = 0;

    // Append images that have not failed to register before.
    for (const auto& image : reconstruction_->Images()) {
        // Skip images that have pose already estimated
        if (image.second.HasPose()) {
            has_pose_count ++;
            continue;
        }

        // Only consider images with a sufficient number of visible points.
        if (image.second.NumVisibleMapPoints() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
            not_enough_visible_mappoints_count ++;
            continue;
        }

        // Only try registration for a certain maximum number of times.
        const size_t num_reg_trials = num_reg_trials_[image.first];
        if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
            too_many_trials_count ++;
            continue;
        }

        // Get min id difference from the registered image
        double min_id_difference = std::numeric_limits<double>::max();
        image_t image_id_current = image.first;
        double weight;
        for (const auto& inner_image : reconstruction_->RegisterImageIds()) {
            double diff = abs(static_cast<double>(image_id_current) - static_cast<double>(inner_image));
            if (min_id_difference > diff) {
                min_id_difference = diff;
            }
        }

        if (min_id_difference <= options.loop_image_min_id_difference) {
            weight = 1.0;
        } else {
            weight = options.loop_image_weight;
        }

        // double time_diff = std::fabs((double)last_image.timestamp_ - (double)image.second.timestamp_) / 1e9;
        // double time_weight = time_diff > 0 ? std::exp(-time_diff * time_diff / 10000) : 1.0;
        // std::cout << "time weight: " << time_diff << " " << time_weight << std::endl;

        // If image has been filtered or failed to register, place it in the
        // second bucket and prefer images that have not been tried before.
        float rank;
        if (options.image_selection_method == Options::ImageSelectionMethod::WEIGHTED_MIN_UNCERTAINTY) {
            rank = weight * RankNextImageMinUncertainty(image.second);
        // } else if (options.image_selection_method == Options::ImageSelectionMethod::TIME_WEIGHT_MIN_UNCERTAINTY) {
        //     // rank = weight * RankNextImageTimeWeightMinUncertainty(image.second, -time_weight);
        //     rank = weight * rank_image_func(image.second);
        } else {
            rank = weight * rank_image_func(image.second);
        }

        // Enlarge image's rank that has a prior pose
        if (options.have_prior_pose && options.use_prior_aggressively &&
            options.prior_rotations.count(image.first) &&
            options.prior_translations.count(image.first)
        ) {
            rank *= 1.5;
        }

        if (last_keyframe_idx >= 0) {
            class Image& last_image = reconstruction_->Image(last_keyframe_idx);
            class Camera& last_camera = reconstruction_->Camera(last_image.CameraId());
            class Camera camera = reconstruction_->Camera(image.second.CameraId());
            if ((filtered_images_.count(image.first) == 0 && num_reg_trials == 0)&&
                last_camera.ModelId() == camera.ModelId()) {
                image_ranks.emplace_back(image.first, rank);
            } else {
                other_image_ranks.emplace_back(image.first, rank);
            }
        } else {
            if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
                image_ranks.emplace_back(image.first, rank);
            } else {
                other_image_ranks.emplace_back(image.first, rank);
            }
        }
    }

    // std::vector<image_t> ranked_images_ids;
    // SortAndAppendNextImages(image_ranks, &ranked_images_ids);
    // SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

    std::vector<std::pair<image_t, float>> ranked_images;
    std::sort(image_ranks.begin(), image_ranks.end(),
            [](const std::pair<image_t, float>& image1, const std::pair<image_t, float>& image2) {
                return image1.second > image2.second;
            });
    ranked_images.insert(ranked_images.end(), image_ranks.begin(), image_ranks.end());

    std::sort(other_image_ranks.begin(), other_image_ranks.end(),
            [](const std::pair<image_t, float>& image1, const std::pair<image_t, float>& image2) {
                return image1.second > image2.second;
            });
    ranked_images.insert(ranked_images.end(), other_image_ranks.begin(), other_image_ranks.end());

    // Append all prior frames to the end
    if (options.have_prior_pose && options.use_prior_aggressively && 
        reconstruction_->RegisterImageIds().size() > 10
    ) {
        std::unordered_set<image_t> existed_images_ids;
        // existed_images_ids.insert(ranked_images_ids.begin(), ranked_images_ids.end());
        for (auto & ranked_image : ranked_images) {
            existed_images_ids.insert(ranked_image.first);
        }

        std::vector<std::pair<image_t, float>> prior_image_ranks;
        for (const auto& image : reconstruction_->Images()) {
            if (image.second.HasPose()) {
                continue;
            }

            if (existed_images_ids.count(image.first)) {
                continue;
            }

            if (!options.have_prior_pose ||
                !options.prior_rotations.count(image.first) ||
                !options.prior_translations.count(image.first)
            ) {
                continue;
            }
            
            float min_id_difference = std::numeric_limits<double>::max();
            image_t image_id_current = image.first;
            for (const auto& inner_image : reconstruction_->RegisterImageIds()) {
                double diff = abs(static_cast<double>(image_id_current) - static_cast<double>(inner_image));
                if (min_id_difference > diff) {
                    min_id_difference = diff;
                }
            }

            prior_image_ranks.emplace_back(image.first, 1.0f / (min_id_difference + 1.0f));
        }

        // SortAndAppendNextImages(prior_image_ranks, &ranked_images_ids);
        std::sort(prior_image_ranks.begin(), prior_image_ranks.end(),
        [](const std::pair<image_t, float>& image1, const std::pair<image_t, float>& image2) {
            return image1.second > image2.second;
        });
        ranked_images.insert(ranked_images.end(), prior_image_ranks.begin(), prior_image_ranks.end());
    }
    std::cout<<"having pose image: "<< has_pose_count <<std::endl;
    std::cout<<"having not enough visible mappoint image: "<<not_enough_visible_mappoints_count<<std::endl;
    std::cout<<"having too many trials image: "<<too_many_trials_count<<std::endl;

    // return ranked_images_ids;
    return ranked_images;
}

std::vector<sweep_t> IncrementalMapper::FindNextSweeps(const Options & options, const std::vector<std::pair<image_t, float>> & next_images) {
    std::vector<sweep_t> sweep_ids;
    sweep_ids.reserve(next_images.size());
    for (auto image : next_images) {
        sweep_t sweep_id = FindNextSweep(image.first);
        sweep_ids.push_back(sweep_id);
    }
    return sweep_ids;
}

std::vector<std::pair<image_t, float>> IncrementalMapper::FindNextImagesOfflineSLAM(const Options& options, bool prev_normal_mode, bool jump_to_backward) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());

    if(prev_normal_mode){
        offline_slam_forward_ = true;
    }

    std::vector<std::pair<image_t, float>> ranked_image_ids;
    image_t start_id = offline_slam_last_frame_id_;

    const auto& images = reconstruction_->Images();
    image_t min_image_id = std::numeric_limits<image_t>::max();
    image_t max_image_id = std::numeric_limits<image_t>::min();
    for (const auto& image : images) {
        if (min_image_id > image.first) {
            min_image_id = image.first;
        }
        if (max_image_id < image.first){
            max_image_id = image.first;
        }
    }

    if (offline_slam_forward_) {
        for (uint32_t i = 1; i <= options.offline_slam_max_next_image_num; ++i) {
            image_t candidate_id = start_id + i;
            if(candidate_id <= max_image_id) {
                if (!reconstruction_->ExistsImage(candidate_id)) {
                    continue;
                }
                const auto& image = reconstruction_->Image(candidate_id);
                if (image.HasPose()) {
                    continue;
                }
                const size_t num_reg_trials = num_reg_trials_[candidate_id];
                if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
                    continue;
                }
                ranked_image_ids.emplace_back(candidate_id, 1.0f);
            }
        }
    } 
    
    if(jump_to_backward || (offline_slam_forward_ && ranked_image_ids.size() == 0)){
        offline_slam_forward_ = false;
        start_id = offline_slam_start_id_;
        offline_slam_last_keyframe_id_ = offline_slam_start_id_;
        ranked_image_ids.clear();
    }
    if(!offline_slam_forward_) {
        for (uint32_t i = 1; i <= options.offline_slam_max_next_image_num; ++i) {
            image_t candidate_id = start_id - i;
            if (candidate_id >= min_image_id) {
                if (!reconstruction_->ExistsImage(candidate_id)) {
                    continue;
                }
                const auto& image = reconstruction_->Image(candidate_id);
                if (image.HasPose()) {
                    continue;
                }
                const size_t num_reg_trials = num_reg_trials_[candidate_id];
                if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
                    continue;
                }
                ranked_image_ids.emplace_back(candidate_id, 1.0f);
            }
        }
    }

    return ranked_image_ids;
}

bool IncrementalMapper::RegisterInitialImagePair(const Options& options, const image_t image_id1,
                                                 const image_t image_id2) {
CHECK_NOTNULL(reconstruction_.get());
    CHECK_EQ(reconstruction_->NumRegisterImages(), 0);

    CHECK(options.Check());

    init_num_reg_trials_[image_id1] += 1;
    init_num_reg_trials_[image_id2] += 1;
    num_reg_trials_[image_id1] += 1;
    num_reg_trials_[image_id2] += 1;

    const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    init_image_pairs_.insert(pair_id);

    Image& image1 = reconstruction_->Image(image_id1);
    const Camera& camera1 = reconstruction_->Camera(image1.CameraId());
    const int num_local_camera1 = camera1.NumLocalCameras();

    Image& image2 = reconstruction_->Image(image_id2);
    const Camera& camera2 = reconstruction_->Camera(image2.CameraId());
    const int num_local_camera2 = camera2.NumLocalCameras();

    auto container_image1 = scene_graph_container_->Image(image_id1);
    auto container_image2 = scene_graph_container_->Image(image_id2);
    if(false && options.has_gps_prior && container_image1.RtkFlag()==50 && container_image2.RtkFlag()==50){
        std::cout << " Initialization using rtk ! "<< std::endl;
        image1.Qvec() = container_image1.QvecPrior();
        image1.Tvec() = QuaternionToRotationMatrix(image1.Qvec()) * -container_image1.TvecPrior();
        image2.Qvec() = container_image2.QvecPrior();
        image2.Tvec() = QuaternionToRotationMatrix(image2.Qvec()) * -container_image2.TvecPrior();
    } else {

        // Estimate two-view geometry

        if (!EstimateInitialTwoViewGeometry(options, image_id1, image_id2)) {
            return false;
        }

        image1.Qvec() = ComposeIdentityQuaternion();
        image1.Tvec() = Eigen::Vector3d(0, 0, 0);
        image2.Qvec() = prev_init_two_view_geometry_.qvec;
        image2.Tvec() = prev_init_two_view_geometry_.tvec;
        if (num_local_camera1 > 1 || num_local_camera2 > 1) {
            image2.Qvec() = prev_init_two_view_geometry_.qvec_rig;
            image2.Tvec() = prev_init_two_view_geometry_.tvec_rig;
        }

        bool scale_refined = false;
        if (options.with_depth) {
            Eigen::Vector4d relative_qvec;
            Eigen::Vector3d relative_tvec;
            if (EstimateRelativePoseBy3D(options, image_id1, image_id2,
                                         relative_qvec, relative_tvec)) {
                const float t_scale = relative_tvec.norm() / image2.Tvec().norm();
                std::cout << "Depth t_scale: " << t_scale << std::endl;
                image2.Tvec() *= t_scale;
                scale_refined = true;
            } else {
                std::cout << "EstimateRelativePoseBy3D failed" << std::endl;
            }
        }

        if (options.have_prior_pose &&
            options.prior_rotations.count(image_id1) && options.prior_translations.count(image_id1) &&
            options.prior_rotations.count(image_id2) && options.prior_translations.count(image_id2)
                ) {
            Eigen::Vector4d q1 = options.prior_rotations.at(image_id1);
            Eigen::Vector3d t1 = options.prior_translations.at(image_id1);
            Eigen::Matrix3d R1 = QuaternionToRotationMatrix(q1);

            Eigen::Vector4d q2 = options.prior_rotations.at(image_id2);
            Eigen::Vector3d t2 = options.prior_translations.at(image_id2);
            Eigen::Matrix3d R2 = QuaternionToRotationMatrix(q2);

            Eigen::Matrix3d p_R12 = R2 * R1.transpose();
            Eigen::Vector3d p_T12 = t2 - R2 * R1.transpose() * t1;

            std::cout << "Prior relative pose: " << std::endl;
            std::cout << p_R12 << std::endl;
            std::cout << p_T12 << std::endl;

            // image2.Qvec() = RotationMatrixToQuaternion(p_R12);
            // image2.Tvec() = p_T12;

            const float t_scale = p_T12.norm() / image2.Tvec().norm();
            std::cout << "Prior t_scale: " << t_scale << std::endl;
            image2.Tvec() *= t_scale;
            scale_refined = true;
        }

        if (options.with_depth && !scale_refined && options.rgbd_delayed_start) {
            reconstruction_->depth_enabled = false;
            std::cout << "Switch off RGBD constraints temporarily" << std::endl;
        }
    }

    const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();
    const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();
    const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
    const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();

    std::vector<std::pair<point2D_t, mappoint_t>> tri_corrs;
    std::vector<char> inlier_mask;
    // Add KeyFrame.
    AddKeyFrame(options, image_id1, tri_corrs, inlier_mask, true);
    AddKeyFrame(options, image_id2, tri_corrs, inlier_mask, true);

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();
#if 1
    const FeatureMatches& corrs = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);
    const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);
    const double max_angular_error_rad = DegToRad(2.0);
    double max_residuals = options.init_max_error * options.init_max_error;

    // Add Map Point tracks.
    Track track;
    track.Reserve(2);
    track.AddElement(TrackElement());
    track.AddElement(TrackElement());
    track.Element(0).image_id = image_id1;
    track.Element(1).image_id = image_id2;

    std::vector<uint32_t> local_image_indices1 = image1.LocalImageIndices();
    std::vector<uint32_t> local_image_indices2 = image2.LocalImageIndices();

    for (const auto& corr : corrs) {
        class Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
        class Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);
        if (point2D1.HasMapPoint() && point2D2.HasMapPoint()) {
            if (point2D1.MapPointId() != point2D2.MapPointId()) {
                reconstruction_->MergeMapPoints(point2D1.MapPointId(), point2D2.MapPointId());
            }
            continue;
        }

        if (camera1.ModelName().compare("SPHERICAL") == 0 || camera1.ModelName().compare("SPHERICAL") == 0) {
            const Eigen::Vector3d point1_N = camera1.ImageToBearing(point2D1.XY());
            const Eigen::Vector3d point2_N = camera2.ImageToBearing(point2D2.XY());
            const Eigen::Vector3d& xyz = TriangulatePointHomogeneous(proj_matrix1, proj_matrix2, point1_N, point2_N);

            const double tri_angle = CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
            double angular_error1 = CalculateAngularErrorSphericalCamera(point1_N, xyz, proj_matrix1);
            double angular_error2 = CalculateAngularErrorSphericalCamera(point2_N, xyz, proj_matrix2);

            if (tri_angle >= min_tri_angle_rad && angular_error1 < max_angular_error_rad &&
                angular_error2 < max_angular_error_rad) {
                track.Element(0).point2D_idx = corr.point2D_idx1;
                track.Element(1).point2D_idx = corr.point2D_idx2;
                reconstruction_->AddMapPoint(xyz, track);
            }
        }
        else if (num_local_camera1 > 1 || num_local_camera2 > 1)
        {
            uint32_t local_image_id1 = local_image_indices1[corr.point2D_idx1];
            uint32_t local_image_id2 = local_image_indices2[corr.point2D_idx2];

            const Eigen::Vector2d point1_N = camera1.LocalImageToWorld(local_image_id1, point2D1.XY());

            const Eigen::Vector2d point2_N = camera2.LocalImageToWorld(local_image_id2, point2D2.XY());

            Eigen::Vector4d local_qvec1, local_qvec2;
            Eigen::Vector3d local_tvec1, local_tvec2;

            camera1.GetLocalCameraExtrinsic(local_image_indices1[corr.point2D_idx1], local_qvec1, local_tvec1);
            camera2.GetLocalCameraExtrinsic(local_image_indices2[corr.point2D_idx2], local_qvec2, local_tvec2);

            const Eigen::Matrix3d local_camera_R1 =
                QuaternionToRotationMatrix(local_qvec1) * QuaternionToRotationMatrix(image1.Qvec());

            const Eigen::Vector3d local_camera_T1 =
                QuaternionToRotationMatrix(local_qvec1) * image1.Tvec() + local_tvec1;

            const Eigen::Matrix3x4d local_camera_proj_matrix1 =
                ComposeProjectionMatrix(local_camera_R1, local_camera_T1);

            const Eigen::Vector3d local_camera_proj_center1 = -local_camera_R1.transpose() * local_camera_T1;

            const Eigen::Matrix3d local_camera_R2 =
                QuaternionToRotationMatrix(local_qvec2) * QuaternionToRotationMatrix(image2.Qvec());

            const Eigen::Vector3d local_camera_T2 =
                QuaternionToRotationMatrix(local_qvec2) * image2.Tvec() + local_tvec2;

            const Eigen::Matrix3x4d local_camera_proj_matrix2 =
                ComposeProjectionMatrix(local_camera_R2, local_camera_T2);

            const Eigen::Vector3d local_camera_proj_center2 = -local_camera_R2.transpose() * local_camera_T2;

            const Eigen::Vector3d& xyz =
                TriangulatePoint(local_camera_proj_matrix1, local_camera_proj_matrix2, point1_N, point2_N);

            const double tri_angle =
                CalculateTriangulationAngle(local_camera_proj_center1, local_camera_proj_center2, xyz);

            double error1 = CalculateSquaredReprojectionErrorRig(point2D1.XY(), xyz, local_camera_proj_matrix1,
                                                                 local_image_id1, camera1);
            double error2 = CalculateSquaredReprojectionErrorRig(point2D2.XY(), xyz, local_camera_proj_matrix2,
                                                                 local_image_id2, camera2);

            if (tri_angle >= min_tri_angle_rad &&
                error1 < max_residuals && error2 < max_residuals &&
                HasPointPositiveDepth(local_camera_proj_matrix1, xyz) &&
                HasPointPositiveDepth(local_camera_proj_matrix2, xyz)) {
                bool added = point2D1.HasMapPoint() || point2D2.HasMapPoint();
                if (added) {
                    if (point2D1.HasMapPoint()) {
                        CHECK(!point2D2.HasMapPoint());
                        reconstruction_->AddObservation(point2D1.MapPointId(), TrackElement(image_id2, corr.point2D_idx2));
                    } else if (point2D2.HasMapPoint()) {
                        CHECK(!point2D1.HasMapPoint());
                        reconstruction_->AddObservation(point2D2.MapPointId(), TrackElement(image_id1, corr.point2D_idx1));
                    }
                } else {
                    track.Element(0).point2D_idx = corr.point2D_idx1;
                    track.Element(1).point2D_idx = corr.point2D_idx2;
                    reconstruction_->AddMapPoint(xyz, track);
                }
            }
        } else {
            const Eigen::Vector2d point1_N = camera1.ImageToWorld(point2D1.XY());
            const Eigen::Vector2d point2_N = camera2.ImageToWorld(point2D2.XY());
            const Eigen::Vector3d& xyz = TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);

            const double tri_angle = CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
            if (tri_angle >= min_tri_angle_rad && HasPointPositiveDepth(proj_matrix1, xyz) &&
                HasPointPositiveDepth(proj_matrix2, xyz)) {
                track.Element(0).point2D_idx = corr.point2D_idx1;
                track.Element(1).point2D_idx = corr.point2D_idx2;
                reconstruction_->AddMapPoint(xyz, track);
            }
        }
    }

    // std::cout << "num local camera: " << num_local_camera1 << " " << num_local_camera2 << std::endl;
    // std::cout << "sub_matching: " << options.sub_matching << std::endl;
    // std::cout << "self_matching: " << options.self_matching << std::endl;

    if (num_local_camera1 > 2 && num_local_camera2 > 2 && options.sub_matching && options.self_matching) {
        std::vector<Eigen::Matrix3x4d> proj_matrixs1;
        proj_matrixs1.resize(camera1.NumLocalCameras());
        for (size_t i = 0; i < camera1.NumLocalCameras(); ++i) {
            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;
            camera1.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
            Eigen::Matrix3x4d local_transform = ComposeProjectionMatrix(local_qvec, local_tvec);
            proj_matrixs1[i] = local_transform;
        }
        std::vector<Eigen::Matrix3x4d> proj_matrixs2;
        proj_matrixs2.resize(camera2.NumLocalCameras());
        for (size_t i = 0; i < camera2.NumLocalCameras(); ++i) {
            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;
            camera2.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);

            const Eigen::Matrix3d local_camera_R =
                QuaternionToRotationMatrix(local_qvec) * QuaternionToRotationMatrix(image2.Qvec());

            const Eigen::Vector3d local_camera_T =
                QuaternionToRotationMatrix(local_qvec) * image2.Tvec() + local_tvec;

            const Eigen::Matrix3x4d local_transform =
                ComposeProjectionMatrix(local_camera_R, local_camera_T);

            proj_matrixs2[i] = local_transform;
        }

        auto AddObservationIntraView = [&](const class Image& image, 
            const class Camera& camera, 
            const std::vector<uint32_t>& local_image_indices, 
            const std::vector<Eigen::Matrix3x4d>& proj_matrixs,
            const FeatureMatch& corr) {
            const class Point2D& point2D1 = image.Point2D(corr.point2D_idx1);
            const class Point2D& point2D2 = image.Point2D(corr.point2D_idx2);
            if (point2D1.HasMapPoint() && point2D2.HasMapPoint()) {
                return 0;
            }
            uint32_t local_image_id1 = local_image_indices[corr.point2D_idx1];
            uint32_t local_image_id2 = local_image_indices[corr.point2D_idx2];
            const auto& proj_matrix1 = proj_matrixs.at(local_image_id1);
            const auto& proj_matrix2 = proj_matrixs.at(local_image_id2);

            if (point2D1.HasMapPoint() && !point2D2.HasMapPoint()) {
                mappoint_t mappoint_id = point2D1.MapPointId();
                Eigen::Vector3d xyz = reconstruction_->MapPoint(mappoint_id).XYZ();
                TrackElement track_el(image.ImageId(), corr.point2D_idx2);
                if (HasPointPositiveDepth(proj_matrix2, xyz)) {
                    reconstruction_->AddObservation(mappoint_id, track_el);
                    return 1;
                }
            }
            if (point2D2.HasMapPoint() && !point2D1.HasMapPoint()) {
                mappoint_t mappoint_id = point2D2.MapPointId();
                Eigen::Vector3d xyz = reconstruction_->MapPoint(mappoint_id).XYZ();
                TrackElement track_el(image.ImageId(), corr.point2D_idx1);
                if (HasPointPositiveDepth(proj_matrix1, xyz)) {
                    reconstruction_->AddObservation(mappoint_id, track_el);
                    return 1;
                }
            }
            if (!point2D1.HasMapPoint() && !point2D2.HasMapPoint()) {
                const Eigen::Vector2d point1_N = camera.LocalImageToWorld(local_image_id1, point2D1.XY());
                const Eigen::Vector2d point2_N = camera.LocalImageToWorld(local_image_id2, point2D2.XY());
                const Eigen::Vector3d& xyz =
                TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);

                // const double tri_angle = CalculateTriangulationAngle(
                //     proj_center1, proj_center2, xyz);

                double error1 = CalculateSquaredReprojectionErrorRig(
                    point2D1.XY(), xyz, proj_matrix1, local_image_id1, camera1);
                double error2 = CalculateSquaredReprojectionErrorRig(
                    point2D2.XY(), xyz, proj_matrix2, local_image_id2, camera2);

                if (//tri_angle >= DegToRad(0.008) &&
                    error1 < max_residuals && error2 < max_residuals &&
                    HasPointPositiveDepth(proj_matrix1, xyz) &&
                    HasPointPositiveDepth(proj_matrix2, xyz)) {
                    track.Element(0).image_id = image.ImageId();
                    track.Element(1).image_id = image.ImageId();
                    track.Element(0).point2D_idx = corr.point2D_idx1;
                    track.Element(1).point2D_idx = corr.point2D_idx2;
                    reconstruction_->AddMapPoint(xyz, track);
                    return 1;
                }
            }
            return 0;
        };

        int num_triangulated1 = 0, num_triangulated2 = 0;

        const FeatureMatches& corrs1 = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id1);
        for (auto& corr : corrs1) {
            num_triangulated1 +=
            AddObservationIntraView(image1, camera1, local_image_indices1, proj_matrixs1, corr);
        }
        const FeatureMatches& corrs2 = correspondence_graph->FindCorrespondencesBetweenImages(image_id2, image_id2);
        for (auto& corr : corrs2) {
            num_triangulated2 +=
            AddObservationIntraView(image2, camera2, local_image_indices2, proj_matrixs2, corr);
        }

        std::cout << StringPrintf("Intra-View(%d) triangulation: %d\n", image_id1, num_triangulated1);
        std::cout << StringPrintf("Intra-View(%d) triangulation: %d\n", image_id2, num_triangulated2);
    }
#else
    if (num_local_camera1 <= 1 && num_local_camera2 <= 1) {
        const FeatureMatches& corrs = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);
        const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);
        const double max_angular_error_rad = DegToRad(2.0);

        // Add Map Point tracks.
        Track track;
        track.Reserve(2);
        track.AddElement(TrackElement());
        track.AddElement(TrackElement());
        track.Element(0).image_id = image_id1;
        track.Element(1).image_id = image_id2;

        std::vector<uint32_t> local_image_indices1 = image1.LocalImageIndices();
        std::vector<uint32_t> local_image_indices2 = image2.LocalImageIndices();

        for (const auto& corr : corrs) {
            class Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
            class Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);
            point2D1.SetMask(true);
            point2D2.SetMask(true);

            if (camera1.ModelName().compare("SPHERICAL") == 0 || camera2.ModelName().compare("SPHERICAL") == 0) {
                const Eigen::Vector3d point1_N = camera1.ImageToBearing(point2D1.XY());
                const Eigen::Vector3d point2_N = camera2.ImageToBearing(point2D2.XY());
                const Eigen::Vector3d& xyz = TriangulatePointHomogeneous(proj_matrix1, proj_matrix2, point1_N, point2_N);

                const double tri_angle = CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
                double angular_error1 = CalculateAngularErrorSphericalCamera(point1_N, xyz, proj_matrix1);
                double angular_error2 = CalculateAngularErrorSphericalCamera(point2_N, xyz, proj_matrix2);

                if (tri_angle >= min_tri_angle_rad && angular_error1 < max_angular_error_rad &&
                    angular_error2 < max_angular_error_rad) {
                    track.Element(0).point2D_idx = corr.point2D_idx1;
                    track.Element(1).point2D_idx = corr.point2D_idx2;
                    reconstruction_->AddMapPoint(xyz, track);
                }
            } else {
                const Eigen::Vector2d point1_N = camera1.ImageToWorld(point2D1.XY());
                const Eigen::Vector2d point2_N = camera2.ImageToWorld(point2D2.XY());
                const Eigen::Vector3d& xyz = TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);

                const double tri_angle = CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
                if (tri_angle >= min_tri_angle_rad && HasPointPositiveDepth(proj_matrix1, xyz) &&
                    HasPointPositiveDepth(proj_matrix2, xyz)) {
                    track.Element(0).point2D_idx = corr.point2D_idx1;
                    track.Element(1).point2D_idx = corr.point2D_idx2;
                    reconstruction_->AddMapPoint(xyz, track);
                }
            }
        }
    } else {
        typedef IncrementalTriangulator::CorrData CorrData;
        CorrData ref_corr_data;
        ref_corr_data.image_id = image_id1;
        ref_corr_data.image = &image1;
        ref_corr_data.camera = &camera1;

        for (point2D_t point2D_idx1 = 0; point2D_idx1 < image1.NumPoints2D(); 
            ++point2D_idx1) {
            std::vector<CorrespondenceGraph::Correspondence> corrs;
                // correspondence_graph->FindTransitiveCorrespondences(
                //     image_id1, point2D_idx1, 1);
            correspondence_graph->FindTransitiveCorrespondences(image_id1, point2D_idx1, 1, &corrs);

            if (corrs.empty()) {
                continue;
            }

            if (image1.Point2D(point2D_idx1).HasMapPoint()) {
                continue;
            }

            std::vector<CorrData> create_corrs_data;

            for (const CorrespondenceGraph::Correspondence corr : corrs) {
                if (corr.image_id != image_id1 && corr.image_id != image_id2) {
                    continue;
                }

                const Image& corr_image = reconstruction_->Image(corr.image_id);
                const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());

                if (corr_image.Point2D(corr.point2D_idx).HasMapPoint()) {
                    continue;
                }

                CorrData corr_data;
                corr_data.image_id = corr.image_id;
                corr_data.point2D_idx = corr.point2D_idx;
                corr_data.image = &corr_image;
                corr_data.camera = &corr_camera;
                corr_data.point2D = &corr_image.Point2D(corr.point2D_idx);
                create_corrs_data.push_back(corr_data);
            }

            ref_corr_data.point2D_idx = point2D_idx1;
            ref_corr_data.point2D = &image1.Point2D(point2D_idx1);
            create_corrs_data.push_back(ref_corr_data);

            if (create_corrs_data.size() < 2) {
                continue;
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
            tri_options.min_tri_angle = DegToRad(0.004);
            tri_options.residual_type =
                TriangulationEstimator::ResidualType::ANGULAR_ERROR;
            tri_options.ransac_options.max_error = DegToRad(4.0);
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
                continue;
            }

            // Add inliers to estimated track.
            Track track;
            track.Reserve(create_corrs_data.size());
            for (size_t i = 0; i < inlier_mask.size(); ++i) {
                if (inlier_mask[i]) {
                    const CorrData& corr_data = create_corrs_data[i];
                    track.AddElement(corr_data.image_id, corr_data.point2D_idx);
                }
            }

            if (track.Length() < 2) {
                continue;
            }

            // Add estimated point to reconstruction.
            reconstruction_->AddMapPoint(xyz, std::move(track));
        }
    }
#endif
    offline_slam_last_frame_id_ = image_id1 < image_id2 ? image_id2 : image_id1;
    offline_slam_last_keyframe_id_ = offline_slam_last_frame_id_;
    offline_slam_start_id_ = offline_slam_last_frame_id_;
    // keyframe_ids_.push_back(image_id1);
    // keyframe_ids_.push_back(image_id2);

    return true;
}

int IncrementalMapper::InlierWithPriorPose(const Options& options, const std::vector<Eigen::Vector2d>& tri_points2D,
                                           const std::vector<Eigen::Vector3d>& tri_points3D, const Camera& camera,
                                           const Eigen::Vector4d prior_qvec, const Eigen::Vector3d prior_tvec) {
    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = QuaternionToRotationMatrix(prior_qvec);
    proj_matrix.rightCols<1>() = prior_tvec;

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> tri_points2D_N(tri_points2D.size());
    for (size_t i = 0; i < tri_points2D.size(); ++i) {
        tri_points2D_N[i] = camera.ImageToWorld(tri_points2D[i]);
    }

    std::vector<double> residuals;
    ComputeSquaredReprojectionError(tri_points2D_N, tri_points3D, proj_matrix, &residuals);

    double abs_pose_max_error = camera.ImageToWorldThreshold(options.abs_pose_max_error);
    const double max_residual = abs_pose_max_error * abs_pose_max_error * 2.25;

    int inlier_num = 0;

    for (size_t i = 0; i < residuals.size(); ++i) {
        if (residuals[i] < max_residual) {
            inlier_num++;
        }
    }
    return inlier_num;
}

bool IncrementalMapper::EstimateCameraPose(const Options& options, const image_t image_id,
                                           std::vector<std::pair<point2D_t, mappoint_t>>& tri_corrs,
                                           std::vector<char>& inlier_mask, size_t* inlier_num) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK_GE(reconstruction_->NumRegisterImages(), 2);

    CHECK(options.Check());
    if (inlier_num != NULL) {
        *inlier_num = 0;
    }

    const size_t num_reg_images = reconstruction_->NumRegisterImages();

    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());

    CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

    num_reg_trials_[image_id] += 1;

    // Check if enough 2D-3D correspondence.
    if (image.NumVisibleMapPoints() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    // Search for 2D-3D correspondence.
    const int kCorrTransitivity = 1;

    tri_corrs.clear();
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;
    std::vector<uint64_t> mappoints_create_time;

    std::vector<mappoint_t> visible_mappoint_ids;

    bool has_bogus_params = false;
    std::unordered_map<camera_t, std::string> camera_param_map;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        std::vector<CorrespondenceGraph::Correspondence> corrs;
            // correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);
        correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<mappoint_t> mappoint_ids;

        for (const auto corr : corrs) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint()) {
                continue;
            }

            // Avoid duplicate correspondence.
            if (mappoint_ids.count(corr_point2D.MapPointId()) > 0) {
                continue;
            }

            const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());

            // Avoid correspondences to images with bogus camera parameters.
            if (camera.ModelName().compare("SPHERICAL") != 0 && 
                camera.ModelName().compare("UNIFIED") != 0 &&
                camera.ModelName().compare("OPENCV_FISHEYE") != 0 &&
                camera.HasBogusParams(options.min_focal_length_ratio, 
                                      options.max_focal_length_ratio,
                                      options.max_extra_param)) {
                has_bogus_params = true;
                camera_param_map[corr_camera.CameraId()] = corr_camera.ParamsToString();
                continue;
            }

            const MapPoint& mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());
            visible_mappoint_ids.push_back(corr_point2D.MapPointId());

            tri_corrs.emplace_back(point2D_idx, corr_point2D.MapPointId());
            mappoint_ids.insert(corr_point2D.MapPointId());
            tri_points2D.push_back(point2D.XY());
            tri_points3D.push_back(mappoint.XYZ());
            mappoints_create_time.push_back(mappoint.CreateTime());
        }
    }

    if (tri_points2D.size() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        std::cout << "has_bogus_params = " << has_bogus_params << std::endl;
        for (const auto& camera_param : camera_param_map) {
            std::cout << "ID: " << camera_param.first << ", param: " << camera_param.second << std::endl;
        }
        std::cout << "tri_points2D.size() = " << tri_points2D.size() << ", lower than "
                  << options.abs_pose_min_num_inliers << std::endl;
        return false;
    }

    // 2D-3D estimation.

    // Only refine / estimate focal length, if no focal length was specified
    // (manually or through EXIF) and if it was not already estimated previously
    // from another image (when multiple images share the same camera
    // parameters)

    AbsolutePoseEstimationOptions abs_pose_options;
    abs_pose_options.num_threads = options.num_threads;
    abs_pose_options.num_focal_length_samples = 10;
    abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
    abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
    abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
    abs_pose_options.ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    abs_pose_options.ransac_options.min_num_trials = 30;
    abs_pose_options.ransac_options.confidence = 0.9999;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    if (!options.single_camera && camera.ModelName().compare("SPHERICAL") != 0) {
        // abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = false;
    }

    size_t num_inliers = 0;
    inlier_mask.clear();

    if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D, &image.Qvec(), &image.Tvec(), &camera,
                              &num_inliers, &inlier_mask)) {
        std::cout << "EstimateAbsolutePose failed!" << std::endl;
        return false;
    }

    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers) ||
        (static_cast<double>(num_inliers) <
         static_cast<double>(tri_points2D.size()) * options.abs_pose_min_inlier_ratio)) {
        std::cout << "num_inliers " << num_inliers << ", lower than " << options.abs_pose_min_num_inliers << " or "
                  << " inlier ratio lower than " << options.abs_pose_min_inlier_ratio << std::endl;

        return false;
    }

    std::vector<double> mappoint_weights(tri_points3D.size());
    std::fill(mappoint_weights.begin(), mappoint_weights.end(), 1.0);
    if (options.map_update && options.update_with_sequential_mode) {
        for (int i = 0; i < tri_points3D.size(); ++i) {
            if (prev_mappoint_ids_.find(visible_mappoint_ids[i]) != prev_mappoint_ids_.end()) {
                mappoint_weights[i] = 20.0;
            }
        }
    }

    // Pose refinement
    if (reconstruction_->depth_enabled && options.rgbd_pose_refine_depth_weight > 0) {
        for (int i = 0; i < tri_corrs.size(); i++) {
            abs_pose_refinement_options.point_depths.emplace_back(
                image.Point2D(tri_corrs[i].first).Depth());
            abs_pose_refinement_options.point_depths_weights.emplace_back(
                options.rgbd_pose_refine_depth_weight * image.Point2D(tri_corrs[i].first).DepthWeight());
        }
    }
    if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask, tri_points2D, tri_points3D, 
                            reconstruction_->NumRegisterImages(), mappoints_create_time, 
                            mappoint_weights, &image.Qvec(), &image.Tvec(), &camera)) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }

    // Update data
    refined_cameras_.insert(image.CameraId());

    if (inlier_num != NULL) {
        *inlier_num = num_inliers;
    }

    std::cout << "inlier num: " << num_inliers << std::endl;

    image.SetPoseFlag(true);
    offline_slam_last_frame_id_ = image_id;
    offline_slam_start_id_ = image_id;

    prev_mappoint_ids_.clear();
    prev_mappoint_ids_.insert(visible_mappoint_ids.begin(), visible_mappoint_ids.end());
    
    return true;
}

bool IncrementalMapper::EstimateCameraPose(const Options & options,  Camera camera, 
                            const std::vector<Eigen::Vector2d>&  tri_points2D, 
                            const std::vector<Eigen::Vector3d>&  tri_points3D, 
                            Eigen::Vector3d& pose_tvec,
                            Eigen::Vector4d& pose_qvec,
                            std::vector<char>& inlier_mask,size_t* inlier_num) const {

    CHECK(options.Check());
    if (inlier_num != NULL) {
        *inlier_num = 0;
    }

    // 2D-3D estimation.

    // Only refine / estimate focal length, if no focal length was specified
    // (manually or through EXIF) and if it was not already estimated previously
    // from another image (when multiple images share the same camera
    // parameters)

    AbsolutePoseEstimationOptions abs_pose_options;
    abs_pose_options.num_threads = options.num_threads;
    abs_pose_options.num_focal_length_samples = 1;
    abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
    abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
    abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
    abs_pose_options.ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    abs_pose_options.ransac_options.min_num_trials = 30;
    abs_pose_options.ransac_options.confidence = 0.9999;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    if (!options.single_camera && camera.ModelName().compare("SPHERICAL") != 0) {
        abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = false;
    }

    size_t num_inliers = 0;
    inlier_mask.clear();

    if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D, &pose_qvec, &pose_tvec, &camera,
                                &num_inliers, &inlier_mask)) {
        std::cout << "EstimateAbsolutePose failed!" << std::endl;
        return false;
    }

    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers) ||
        (static_cast<double>(num_inliers) <
         static_cast<double>(tri_points2D.size()) * options.abs_pose_min_inlier_ratio)) {
        std::cout << "num_inliers " << num_inliers << ", lower than " << options.abs_pose_min_num_inliers << " or "
                  << " inlier ratio lower than " << options.abs_pose_min_inlier_ratio << std::endl;

        return false;
    }

    // Pose refinement
    if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask, tri_points2D, tri_points3D, 
                            reconstruction_->NumRegisterImages(), std::vector<uint64_t>(),
                            std::vector<double>(), &pose_qvec, &pose_tvec, &camera)) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }

    if (inlier_num != NULL) {
        *inlier_num = num_inliers;
    }

    std::cout << "inlier num: " << num_inliers << std::endl;

    return true;
}

bool IncrementalMapper::EstimateCameraPoseWithPrior(const Options& options, const image_t image_id,
                                           std::vector<std::pair<point2D_t, mappoint_t>>& tri_corrs,
                                           std::vector<char>& inlier_mask, size_t* inlier_num) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK_GE(reconstruction_->NumRegisterImages(), 2);

    size_t inlier_num_old;
    Eigen::Vector4d prior_qvec;
    Eigen::Vector3d prior_tvec;
    bool result = EstimateCameraPose(options, image_id, tri_corrs, inlier_mask, &inlier_num_old);
    if (!TryGetRelativePriorPose(options, image_id, prior_qvec, prior_tvec, true)) {
        // not available/suitable prior pose
        // use legacy pose estimate
        return result;
    }

    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());

    CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    // Search for 2D-3D correspondence.
    const int kCorrTransitivity = 1;

    tri_corrs.clear();
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;

    bool has_bogus_params = false;
    std::unordered_map<camera_t, std::string> camera_param_map;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        std::vector<CorrespondenceGraph::Correspondence> corrs;
            // correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);
        correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<mappoint_t> mappoint_ids;

        for (const auto corr : corrs) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint()) {
                continue;
            }

            // Avoid duplicate correspondence.
            if (mappoint_ids.count(corr_point2D.MapPointId()) > 0) {
                continue;
            }

            const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());

            // Avoid correspondences to images with bogus camera parameters.
            if (camera.ModelName().compare("SPHERICAL") != 0 && camera.ModelName().compare("UNIFIED") != 0 &&
                camera.ModelName().compare("OPENCV_FISHEYE") != 0 &&
                corr_camera.HasBogusParams(options.min_focal_length_ratio, options.max_focal_length_ratio,
                                           options.max_extra_param)) {
                has_bogus_params = true;
                camera_param_map[corr_camera.CameraId()] = corr_camera.ParamsToString();
                continue;
            }

            const MapPoint& mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());

            tri_corrs.emplace_back(point2D_idx, corr_point2D.MapPointId());
            mappoint_ids.insert(corr_point2D.MapPointId());
            tri_points2D.push_back(point2D.XY());
            tri_points3D.push_back(mappoint.XYZ());
        }
    }

    // Calculate inliners
    size_t num_inliers = 0;
    inlier_mask.resize(tri_corrs.size());

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> tri_points2D_N(tri_points2D.size());
    for (size_t i = 0; i < tri_points2D.size(); ++i) {
        tri_points2D_N[i] = camera.ImageToWorld(tri_points2D[i]);
    }

    Eigen::Vector4d fused_qvec = image.Qvec();
    Eigen::Vector3d fused_tvec = image.Tvec();
    std::cout << "**********************************************************" << std::endl;
    std::cout << "Image relative pose by prior: " << std::endl;
    std::cout << "Old:   " << image.Tvec().transpose() << " " << image.Qvec().transpose() << std::endl;
    std::cout << "Prior: " << prior_tvec.transpose() << " " << prior_qvec.transpose() << std::endl;
    const double weight_old = image.HasPose() ? std::sqrt(inlier_num_old * 0.005) : 0.0;
    double weight_prior = 1.0;  // prior pose equals to 200 inliers at first
    for (int level = 0; level < 15; level++, weight_prior /= 2.0) {
        // update image pose
        fused_qvec = (weight_old * image.Qvec() + weight_prior * prior_qvec) / (weight_old + weight_prior);
        fused_tvec = (weight_old * image.Tvec() + weight_prior * prior_tvec) / (weight_old + weight_prior);
        Eigen::Quaterniond normalized_quaterniond(fused_qvec(0), fused_qvec(1), fused_qvec(2), fused_qvec(3));
        normalized_quaterniond.normalize();
        fused_qvec = Eigen::Vector4d(
            normalized_quaterniond.w(), normalized_quaterniond.x(),
            normalized_quaterniond.y(), normalized_quaterniond.z()
        );
        
        Eigen::Matrix3x4d proj_matrix;
        proj_matrix.leftCols<3>() = QuaternionToRotationMatrix(fused_qvec);
        proj_matrix.rightCols<1>() = fused_tvec;

        std::vector<double> residuals;
        ComputeSquaredReprojectionError(tri_points2D_N, tri_points3D, proj_matrix, &residuals);

        // use more relaxed residual thresh: 100%, 105%, 110%, ...
        double abs_pose_max_error = (1.0 /*+ level * 0.05*/) * camera.ImageToWorldThreshold(options.abs_pose_max_error);
        const double max_residual = abs_pose_max_error * abs_pose_max_error;

        // calculate inliners
        num_inliers = 0;
        for (size_t i = 0; i < residuals.size(); ++i) {
            if (residuals[i] < max_residual) {
                num_inliers++;
                inlier_mask[i] = 1;
            }
            else {
                inlier_mask[i] = 0;
            }
        } 
        std::cout << "Weight(old): " << weight_old / weight_prior << ", inliner num: " << num_inliers << std::endl;

        // check for break
        if (!image.HasPose() /*|| num_inliers >= options.abs_pose_min_num_inliers*/ || num_inliers >= size_t(inlier_num_old * 0.33)) break;
    }
    image.Qvec() = fused_qvec;
    image.Tvec() = fused_tvec;
    std::cout << "New:   " << image.Tvec().transpose() << " " << image.Qvec().transpose() << std::endl;
    std::cout << "**********************************************************" << std::endl;
 
    if (inlier_num != NULL) {
        *inlier_num = num_inliers;
    }

    image.SetPoseFlag(true);
    offline_slam_last_frame_id_ = image_id;
    return true;
}

bool IncrementalMapper::EstimateCameraPoseWithRTK(const Options & options, 
    const image_t image_id, 
    std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
    std::vector<char>& inlier_mask, size_t* inlier_num) {

    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());

    bool success = false;
    Options local_options = options;
    do {
        if(camera.NumLocalCameras() > 1) {
            success = EstimateCameraPoseRig(local_options, image_id, tri_corrs, inlier_mask, inlier_num);
        } else {
            success = EstimateCameraPose(local_options, image_id, tri_corrs, inlier_mask, inlier_num);
        }
        if (success) {
            return true;
        }
        float abs_pose_min_inlier_ratio = local_options.abs_pose_min_inlier_ratio * 0.5;
        std::cout << StringPrintf("reset register inlier: %f -> %f\n", 
            local_options.abs_pose_min_inlier_ratio, abs_pose_min_inlier_ratio);
        local_options.abs_pose_min_inlier_ratio = abs_pose_min_inlier_ratio;
    } while(local_options.abs_pose_min_inlier_ratio > 0.1);

    std::cout << "  Register with RTK!" << std::endl;

    CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    // Search for 2D-3D correspondence.
    const int kCorrTransitivity = 1;

    tri_corrs.clear();
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        std::vector<CorrespondenceGraph::Correspondence> corrs;
            // correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);
        correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<mappoint_t> mappoint_ids;

        for (const auto corr : corrs) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint()) {
                continue;
            }

            // Avoid duplicate correspondence.
            if (mappoint_ids.count(corr_point2D.MapPointId()) > 0) {
                continue;
            }

            const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());

            // Avoid correspondences to images with bogus camera parameters.
            if (camera.ModelName().compare("SPHERICAL") != 0 && camera.ModelName().compare("UNIFIED") != 0 &&
                camera.ModelName().compare("OPENCV_FISHEYE") != 0 &&
                corr_camera.HasBogusParams(options.min_focal_length_ratio, options.max_focal_length_ratio,
                                           options.max_extra_param)) {
                continue;
            }

            const MapPoint& mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());

            tri_corrs.emplace_back(point2D_idx, corr_point2D.MapPointId());
            mappoint_ids.insert(corr_point2D.MapPointId());
            tri_points2D.push_back(point2D.XY());
            tri_points3D.push_back(mappoint.XYZ());
        }
    }

    Eigen::Matrix3d RPrior = QuaternionToRotationMatrix(image.QvecPrior());
    Eigen::Vector3d TPrior = image.TvecPrior();

    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = RPrior;
    proj_matrix.rightCols<1>() = RPrior * -TPrior;

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> tri_points2D_N(tri_points2D.size());
    for (size_t i = 0; i < tri_points2D.size(); ++i) {
        tri_points2D_N[i] = camera.ImageToWorld(tri_points2D[i]);
    }

    std::vector<double> residuals;
    ComputeSquaredReprojectionError(tri_points2D_N, tri_points3D, proj_matrix, &residuals);

    double abs_pose_max_error = camera.ImageToWorldThreshold(options.abs_pose_max_error);
    const double max_residual = abs_pose_max_error * abs_pose_max_error;
    inlier_mask.resize(residuals.size());
    std::fill(inlier_mask.begin(), inlier_mask.end(), 0);

    size_t num_inliers = 0;
    for (size_t i = 0; i < residuals.size(); ++i) {
        if (residuals[i] < max_residual) {
            num_inliers++;
            inlier_mask[i] = 1;
        }
    }
    if (inlier_num != NULL) {
        *inlier_num = num_inliers;
    }

    std::cout << "inlier num: " << num_inliers << std::endl;

    image.SetPoseFlag(true);
    image.Qvec() = image.QvecPrior();
    image.Tvec() = RPrior * -TPrior;
    return true;
}

bool IncrementalMapper::TryGetRelativePriorPose(
    const Options & options, image_t next_image_id, 
    Eigen::Vector4d & qvec, Eigen::Vector3d & tvec,
    bool is_sequential
) {
    if (!options.prior_rotations.count(next_image_id) ||
        !options.prior_translations.count(next_image_id)
    ) return false;

    constexpr double outlier_relative_frames = 0.3;
    constexpr int min_relative_frames = 4;
    constexpr int max_relative_frames = 30;

    Image & next_image = reconstruction_->Image(next_image_id);
    Eigen::Vector4d next_quaternion_prior = options.prior_rotations.at(next_image_id);
    Eigen::Matrix3d next_rotation_prior = Eigen::Quaterniond(
        next_quaternion_prior(0), next_quaternion_prior(1), 
        next_quaternion_prior(2), next_quaternion_prior(3)
    ).toRotationMatrix();
    Eigen::Vector3d next_translation_prior = options.prior_translations.at(next_image_id);
    Eigen::Vector3d next_position_prior = -next_rotation_prior.transpose() * next_translation_prior;
    Eigen::Matrix4d next_pose_prior = Eigen::Matrix4d::Identity();
    next_pose_prior.block<3, 3>(0, 0) = next_rotation_prior;
    next_pose_prior.block<3, 1>(0, 3) = next_translation_prior;

    // get nearest registered frames that also have prior poses
    auto comparer = [] (
        const std::pair<double, image_t> & a, 
        const std::pair<double, image_t> & b
    ) -> bool {
        return a.first > b.first;
    };
    std::priority_queue<
        std::pair<double, image_t>,
        std::vector<std::pair<double, image_t>>, 
        std::function<bool(const std::pair<double, image_t> &, const std::pair<double, image_t> &)>
    > priority_queue(comparer);
    for (auto base_image_id : reconstruction_->RegisterImageIds()) {
        if (!options.prior_rotations.count(base_image_id) || 
            !options.prior_translations.count(base_image_id)) continue;
        Image & base_image = reconstruction_->Image(base_image_id);

        Eigen::Vector4d base_quaternion_prior = options.prior_rotations.at(base_image_id);
        Eigen::Matrix3d base_rotation_prior = Eigen::Quaterniond(
            base_quaternion_prior(0), base_quaternion_prior(1), 
            base_quaternion_prior(2), base_quaternion_prior(3)
        ).toRotationMatrix();
        Eigen::Vector3d base_translation_prior = options.prior_translations.at(base_image_id);
        Eigen::Vector3d base_position_prior = -base_rotation_prior.transpose() * base_translation_prior;

        // sort base images by weight
        double distance = (base_position_prior - next_position_prior).norm();
        double weight = 1.0 / (distance + 1.0);
        if (is_sequential) {
            double index_diff = std::fabs((int)next_image_id - (int)base_image_id);
            weight *= std::exp(-0.02 * index_diff);    // reduce by half every ~35 frames
        }
        priority_queue.emplace(std::sqrt(weight), base_image_id);
        while (priority_queue.size() > max_relative_frames) {
            priority_queue.pop();
        }
    }

    // register by relative pose
    if (priority_queue.size() >= min_relative_frames) {
        // weighted sum quaterniond and translation
        std::vector<std::tuple<double, image_t, Eigen::Vector4d, Eigen::Vector3d>> candidate_images;
        for (int i = 0; !priority_queue.empty(); i++) {
            auto & pair = priority_queue.top();
            Image & base_image = reconstruction_->Image(pair.second);

            Eigen::Matrix3d base_rotation_current = Eigen::Quaterniond(
                base_image.Qvec(0), base_image.Qvec(1), 
                base_image.Qvec(2), base_image.Qvec(3)
            ).toRotationMatrix();
            Eigen::Vector3d base_translation_current = base_image.Tvec();
            Eigen::Matrix4d base_pose_current = Eigen::Matrix4d::Identity();
            base_pose_current.block<3, 3>(0, 0) = base_rotation_current;
            base_pose_current.block<3, 1>(0, 3) = base_translation_current;

            Eigen::Vector4d base_quaternion_prior = options.prior_rotations.at(pair.second);
            Eigen::Matrix3d base_rotation_prior = Eigen::Quaterniond(
                base_quaternion_prior(0), base_quaternion_prior(1), 
                base_quaternion_prior(2), base_quaternion_prior(3)
            ).toRotationMatrix();
            Eigen::Vector3d base_translation_prior = options.prior_translations.at(pair.second);
            Eigen::Matrix4d base_pose_prior = Eigen::Matrix4d::Identity();
            base_pose_prior.block<3, 3>(0, 0) = base_rotation_prior;
            base_pose_prior.block<3, 1>(0, 3) = base_translation_prior;

            Eigen::Matrix4d base_to_next_prior = next_pose_prior * base_pose_prior.inverse();
            Eigen::Matrix4d next_pose_current = base_to_next_prior * base_pose_current;
            Eigen::Quaterniond next_quaternion_current(next_pose_current.block<3, 3>(0, 0));

            // sum current relative pose with weight
            double weight = pair.first;                            
            Eigen::Vector4d quaterniond = Eigen::Vector4d(
                next_quaternion_current.w(), next_quaternion_current.x(),
                next_quaternion_current.y(), next_quaternion_current.z()
            );
            Eigen::Vector3d translation = next_pose_current.block<3, 1>(0, 3);
            candidate_images.emplace_back(pair.first, pair.second, quaterniond, translation);
            priority_queue.pop();
        }

        // Get averaged predicted quaterniond and translation
        double average_weight_sum = 0.0;
        Eigen::Vector4d average_quaterniond = Eigen::Vector4d::Zero();
        Eigen::Vector3d average_translation = Eigen::Vector3d::Zero();
        if (candidate_images.size() > 0) {
            size_t outlier_count = outlier_relative_frames * candidate_images.size();

            if (outlier_count > 0 && outlier_count < candidate_images.size()) {
                Eigen::Vector3d uniform_average_translation = Eigen::Vector3d::Zero();
                for (auto & item : candidate_images) {
                    uniform_average_translation += std::get<3>(item);
                }
                uniform_average_translation /= candidate_images.size();

                std::sort(candidate_images.begin(), candidate_images.end(), [&](
                    const std::tuple<double, image_t, Eigen::Vector4d, Eigen::Vector3d> & a,
                    const std::tuple<double, image_t, Eigen::Vector4d, Eigen::Vector3d> & b
                ) {
                    return (std::get<3>(a) - uniform_average_translation).squaredNorm() <
                            (std::get<3>(b) - uniform_average_translation).squaredNorm();
                });

                candidate_images.resize(candidate_images.size() - outlier_count);
            }

            std::cout << "Prior pose from: ";
            for (auto & item : candidate_images) {
                average_weight_sum += std::get<0>(item);
                average_quaterniond += std::get<0>(item) * std::get<2>(item);
                average_translation += std::get<0>(item) * std::get<3>(item);
                std::cout << std::get<1>(item) << " ";
            }
            std::cout << std::endl;

            average_quaterniond /= average_weight_sum;
            average_translation /= average_weight_sum;
        }

        if (average_weight_sum > 0.0) {
            // normalize quaterniond
            Eigen::Quaterniond normalized_quaterniond(
                average_quaterniond(0), average_quaterniond(1), 
                average_quaterniond(2), average_quaterniond(3)
            );
            normalized_quaterniond.normalize();
            average_quaterniond = Eigen::Vector4d(
                normalized_quaterniond.w(), normalized_quaterniond.x(),
                normalized_quaterniond.y(), normalized_quaterniond.z()
            );

            qvec = average_quaterniond;
            tvec = average_translation;

            return true;
        }
    }

    return false;
}

std::set<image_t> IncrementalMapper::FindCovisibleImages(const Options& options, const image_t reference_image_id) {
    std::set<image_t> covisible_neighbors;
    covisible_neighbors.insert(reference_image_id);

    for (auto& neighbor_image_id : reconstruction_->RegisterImageIds()) {
        const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(reference_image_id, neighbor_image_id);

        if (reconstruction_->ExistsImagePair(pair_id)) {
            const auto& image_pair = reconstruction_->ImagePair(pair_id);

            const double tri_ratio = static_cast<double>(image_pair.first) / image_pair.second;

            if (tri_ratio >= options.min_covisible_mappoint_ratio &&
                image_pair.first > options.min_covisible_mappoint_num) {
                covisible_neighbors.insert(neighbor_image_id);
            }
        }
    }
    return covisible_neighbors;
}


std::vector<image_t> IncrementalMapper::FindCovisibleImagesForICP(const Options& options,
                                            const image_t cur_id){

    struct link_info{
        image_t img_id;
        float dist;
        float angle_weight;
        int corres_num;
        float score;
        float tri_ratio;
        bool operator <(link_info &info){
            return score<info.score;
        }
    };

    auto &image = reconstruction_->Image(cur_id);

    Eigen::Vector4d cur_qvec = image.Qvec();
    Eigen::Vector3d cur_tvec = image.Tvec();
    Eigen::Matrix3d cur_R = QuaternionToRotationMatrix(cur_qvec);


    std::vector<link_info> infos;
    std::vector<std::pair<image_t , int>> time_neibours;

    for (auto& neighbor_image_id : reconstruction_->RegisterImageIds()) {
        if(neighbor_image_id==cur_id) continue;
        
        link_info info;
        int time_dist = neighbor_image_id >= cur_id ? neighbor_image_id - cur_id : cur_id - neighbor_image_id;
        time_neibours.push_back(std::make_pair(neighbor_image_id, time_dist));

        auto &neighbor_image = reconstruction_->Image(neighbor_image_id);
        Eigen::Vector4d neighbor_qvec = neighbor_image.Qvec();
        Eigen::Vector3d neighbor_tvec = neighbor_image.Tvec();
        Eigen::Matrix3d neighbor_R = QuaternionToRotationMatrix(neighbor_qvec);

        info.img_id = neighbor_image_id;
        info.dist = std::max(0.01, (neighbor_R.transpose()*neighbor_tvec - cur_R.transpose()*cur_tvec).norm());
        info.angle_weight = std::max((1 + neighbor_R.row(2).dot(cur_R.row(2)))/2, 0.05);
        info.corres_num = 10;
        info.tri_ratio = 0.1;

        const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(cur_id, neighbor_image_id);
        if (reconstruction_->ExistsImagePair(pair_id)) {
            const auto& image_pair = reconstruction_->ImagePair(pair_id);
            const double tri_ratio = static_cast<double>(image_pair.first) / image_pair.second;

            if (tri_ratio < options.min_covisible_mappoint_ratio &&
                image_pair.first > options.min_covisible_mappoint_num) {
                continue;
            }
            info.corres_num = std::max(10, int(image_pair.second));
            info.tri_ratio = tri_ratio;
        }
        infos.emplace_back(info);
    }
    if(!infos.empty()) {
        float min_dist = infos[0].dist;
        for (auto &info:infos) {
            if (min_dist > info.dist) min_dist = info.dist;
        }
        float max_score = 0;
        for (auto &info:infos) {
            // std::cout<<" min dist: "<<min_dist<<" "<<info.dist<<"  "<<info.angle_weight<<"  "<<info.corres_num<<std::endl;
            info.score = std::sqrt(min_dist / info.dist) * info.angle_weight * info.corres_num * info.tri_ratio;
            if (max_score < info.score) max_score = info.score;
        }
        std::sort(infos.rbegin(), infos.rend());

        int cnt = 0;
        for (auto &info:infos) {
            printf("%u ==>> %u: dist: %f | %f, angle: %f, corres: %d\n",cur_id, info.img_id, info.dist, min_dist, info.angle_weight, info.corres_num);
            cnt++;
            if(cnt==5) break;
        }

    }


    std::sort(time_neibours.begin(), time_neibours.end(),
              [](std::pair<image_t , int> &a, std::pair<image_t , int>&b){return a.second<b.second;});


    std::set<image_t> result;
    for(int i=0; i<infos.size(); i++){
        result.insert(infos[i].img_id);
        if(result.size()==5) break;
    }

    int neibour_cnt = 0;
    for(int i=0; i<time_neibours.size(); i++){
        if(result.count(time_neibours[i].first)) continue;
        if(time_neibours[i].second<100){
            result.insert(time_neibours[i].first);
            neibour_cnt++;
        }else break;
        if(neibour_cnt==3) break;
    }
    if(result.empty()) return  std::vector<image_t>();
    return std::vector<image_t>(result.begin(), result.end());

}

bool IncrementalMapper::EstimateCameraPoseWithLocalMap(const Options& options, const image_t image_id,
                                                       std::vector<std::pair<point2D_t, mappoint_t>>& tri_corrs,
                                                       std::vector<char>& inlier_mask, size_t* inlier_num) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK_GE(reconstruction_->NumRegisterImages(), 2);

    CHECK(options.Check());
    if (inlier_num != NULL) {
        *inlier_num = 0;
    }

    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());

    CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

    num_reg_trials_[image_id] += 1;

    // Check if enough 2D-3D correspondence.
    if (image.NumVisibleMapPoints() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    std::set<image_t> local_map_images = FindCovisibleImages(options, offline_slam_last_keyframe_id_);

    std::cout << "image in Local map: " << std::endl;
    for (auto local_image : local_map_images) {
        std::cout << local_image << " ";
    }
    std::cout << std::endl;

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    // Search for 2D-3D correspondence.
    const int kCorrTransitivity = 1;

    tri_corrs.clear();
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;

    bool has_bogus_params = false;
    std::unordered_map<camera_t, std::string> camera_param_map;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        std::vector<CorrespondenceGraph::Correspondence> corrs;
            // correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);
        correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<mappoint_t> mappoint_ids;

        for (const auto corr : corrs) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            if (local_map_images.find(corr_image.ImageId()) == local_map_images.end()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint()) {
                continue;
            }

            // Avoid duplicate correspondence.
            if (mappoint_ids.count(corr_point2D.MapPointId()) > 0) {
                continue;
            }

            const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());

            // Avoid correspondences to images with bogus camera parameters.
            if (camera.ModelName().compare("SPHERICAL") != 0 &&
                corr_camera.HasBogusParams(options.min_focal_length_ratio, options.max_focal_length_ratio,
                                           options.max_extra_param)) {
                has_bogus_params = true;
                camera_param_map[corr_camera.CameraId()] = corr_camera.ParamsToString();
                continue;
            }

            const MapPoint& mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());

            tri_corrs.emplace_back(point2D_idx, corr_point2D.MapPointId());
            mappoint_ids.insert(corr_point2D.MapPointId());
            tri_points2D.push_back(point2D.XY());
            tri_points3D.push_back(mappoint.XYZ());
        }
    }

    std::cout << "visible local mappoint = " << tri_points3D.size() << std::endl;

    if (tri_points2D.size() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        std::cout << "has_bogus_params = " << has_bogus_params << std::endl;
        for (const auto& camera_param : camera_param_map) {
            std::cout << "ID: " << camera_param.first << ", param: " << camera_param.second << std::endl;
        }
        std::cout << "tri_points2D.size() = " << tri_points2D.size() << ", lower than "
                  << options.abs_pose_min_num_inliers << std::endl;

        tri_corrs.clear();
        return false;
    }

    // 2D-3D estimation.

    // Only refine / estimate focal length, if no focal length was specified
    // (manually or through EXIF) and if it was not already estimated previously
    // from another image (when multiple images share the same camera
    // parameters)

    AbsolutePoseEstimationOptions abs_pose_options;
    abs_pose_options.num_threads = options.num_threads;
    abs_pose_options.num_focal_length_samples = 10;
    abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
    abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
    abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
    abs_pose_options.ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    abs_pose_options.ransac_options.min_num_trials = 30;
    abs_pose_options.ransac_options.confidence = 0.9999;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    if (!options.single_camera) {
        abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = false;
    }

    size_t num_inliers = 0;
    inlier_mask.clear();

    if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D, &image.Qvec(), &image.Tvec(), &camera,
                              &num_inliers, &inlier_mask)) {
        std::cout << "EstimateAbsolutePose failed!" << std::endl;
        return false;
    }

    std::cout << "num_inliers = " << num_inliers << std::endl;
    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)||
        (static_cast<double>(num_inliers) <
         static_cast<double>(tri_points2D.size()) * options.abs_pose_min_inlier_ratio)) {
        std::cout << "num_inliers " << num_inliers << ", lower than " << options.abs_pose_min_num_inliers << " or "
                  << " inlier ratio lower than " << options.abs_pose_min_inlier_ratio << std::endl;

        std::cout << "estimate pose with local map failed, try to use global map" << std::endl;
        tri_corrs.clear();
        return false;
    }

    // Pose refinement
    if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask, tri_points2D, tri_points3D, 
                            reconstruction_->NumRegisterImages(), std::vector<uint64_t>(),
                            std::vector<double>(), &image.Qvec(), &image.Tvec(), &camera)) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }

    // Update data
    refined_cameras_.insert(image.CameraId());

    if (inlier_num != NULL) {
        *inlier_num = num_inliers;
    }
    // std::cout<<"inlier num: "<<num<<std::endl;
    image.SetPoseFlag(true);

    offline_slam_last_frame_id_ = image_id;

    return true;
}

bool IncrementalMapper::EstimateCameraPoseRigWithLocalMap(const Options& options, const image_t image_id,
                                                          std::vector<std::pair<point2D_t, mappoint_t>>& tri_corrs,
                                                          std::vector<char>& inlier_mask, size_t* inlier_num) {

    CHECK_NOTNULL(reconstruction_.get());
    CHECK_GE(reconstruction_->NumRegisterImages(), 2);

    CHECK(options.Check());
    if (inlier_num != NULL) {
        *inlier_num = 0;
    }

    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());

    CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

    num_reg_trials_[image_id] += 1;

    // Check if enough 2D-3D correspondence.
    if (image.NumVisibleMapPoints() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    std::set<image_t> local_map_images = FindCovisibleImages(options, offline_slam_last_keyframe_id_);

    std::cout << "image in Local map: " << std::endl;
    for (auto local_image : local_map_images) {
        std::cout << local_image << " ";
    }
    std::cout << std::endl;

    std::vector<uint32_t> local_camera_indices = image.LocalImageIndices();
    std::vector<Eigen::Matrix3x4d> local_transforms;
    local_transforms.resize(camera.NumLocalCameras());

    for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        Eigen::Matrix3x4d local_transform = ComposeProjectionMatrix(local_qvec, local_tvec);
        local_transforms[i] = local_transform;
    }

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    // Search for 2D-3D correspondence.
    const int kCorrTransitivity = 1;

    tri_corrs.clear();
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;
    std::vector<GP3PEstimator::X_t> points2D_normalized;

    std::vector<int> tri_camera_indices;

    bool has_bogus_params = false;
    std::unordered_map<camera_t, std::string> camera_param_map;

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        std::vector<CorrespondenceGraph::Correspondence> corrs;
            // correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);
        correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<mappoint_t> mappoint_ids;
        uint32_t local_camera_id = local_camera_indices[point2D_idx];

        for (const auto corr : corrs) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            if (local_map_images.find(corr_image.ImageId()) == local_map_images.end()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint()) {
                continue;
            }

            // Avoid duplicate correspondence.
            if (mappoint_ids.count(corr_point2D.MapPointId()) > 0) {
                continue;
            }

            const MapPoint& mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());

            tri_corrs.emplace_back(point2D_idx, corr_point2D.MapPointId());
            mappoint_ids.insert(corr_point2D.MapPointId());

            tri_points3D.push_back(mappoint.XYZ());
            tri_points2D.push_back(point2D.XY());
            tri_camera_indices.push_back(local_camera_id);

            Eigen::Vector2d point2D_normalized = camera.LocalImageToWorld(local_camera_id, point2D.XY());

            points2D_normalized.emplace_back();
            points2D_normalized.back().rel_tform = local_transforms[local_camera_id];
            points2D_normalized.back().xy = point2D_normalized;
        }
    }

    std::cout << "tri_points3D size: " << tri_points3D.size() << std::endl;
    if (tri_points3D.size() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        std::cout << "tri_points3D.size() = " << tri_points3D.size() << ", lower than "
                  << options.abs_pose_min_num_inliers << std::endl;
        return false;
    }
    
    RANSACOptions ransac_options;
    ransac_options.max_error = 1 - cos(0.0299977504); 
    ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    ransac_options.min_num_trials = 30;
    ransac_options.confidence = 0.9999;

    RANSAC<GP3PEstimator> ransac(ransac_options);
    const auto report = ransac.Estimate(points2D_normalized, tri_points3D);

    inlier_mask = report.inlier_mask;
    size_t num_inliers = report.support.num_inliers;

    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers) ||
        (static_cast<double>(num_inliers) <
         static_cast<double>(tri_points3D.size()) * options.abs_pose_min_inlier_ratio)) {
        std::cout << "num_inliers " << num_inliers << ", lower than " << options.abs_pose_min_num_inliers << " or "
                  << " inlier ratio lower than " << options.abs_pose_min_inlier_ratio<<std::endl;
        return false; 
    }

    // re estimate with inliers

    Eigen::Matrix3d r = report.model.block<3, 3>(0, 0);
    Eigen::Vector3d t = report.model.col(3);

    image.Qvec() = RotationMatrixToQuaternion(r);
    image.Tvec() = t;

    int num_inliers_reprojection = 0;

    for (size_t i = 0; i < tri_points2D.size(); ++i) {
        int local_camera_id = tri_camera_indices[i];
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

        Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(), local_qvec);
        Eigen::Vector3d tvec = QuaternionToRotationMatrix(local_qvec) * image.Tvec() + local_tvec;

        double squared_reproj_error =
            CalculateSquaredReprojectionErrorRig(tri_points2D[i], tri_points3D[i], qvec, tvec, local_camera_id, camera);
        if (squared_reproj_error < options.abs_pose_max_error * options.abs_pose_max_error) {
            num_inliers_reprojection++;
        }
    }
    std::cout << "num inliers reprojection: " << num_inliers_reprojection << std::endl;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    // Pose refinement
    if (!RefineAbsolutePoseRig(abs_pose_refinement_options, inlier_mask, tri_points2D, tri_points3D, 
                               1, std::vector<uint64_t>(), std::vector<double>(), 
                               tri_camera_indices, &image.Qvec(), &image.Tvec(), &camera)) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }

    // Update data
    refined_cameras_.insert(image.CameraId());

    if (inlier_num != NULL) {
        *inlier_num = num_inliers;
    }
    std::cout << "inlier num: " << num_inliers << std::endl;

    image.SetPoseFlag(true);
    offline_slam_last_frame_id_ = image_id;

    return true;
}

bool IncrementalMapper::EstimateCameraPoseRig(const Options& options, const image_t image_id,
                                              std::vector<std::pair<point2D_t, mappoint_t>>& tri_corrs,
                                              std::vector<char>& inlier_mask, size_t* inlier_num) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK_GE(reconstruction_->NumRegisterImages(), 2);

    CHECK(options.Check());
    if (inlier_num != NULL) {
        *inlier_num = 0;
    }

    const size_t num_reg_images = reconstruction_->NumRegisterImages();

    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());

    CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

    num_reg_trials_[image_id] += 1;
    std::vector<uint32_t> local_camera_indices = image.LocalImageIndices();


    std::vector<Eigen::Matrix3x4d> local_transforms;
    local_transforms.resize(camera.NumLocalCameras());

    for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        Eigen::Matrix3x4d local_transform = ComposeProjectionMatrix(local_qvec, local_tvec);
        local_transforms[i] = local_transform;
    }
    std::cout<<"num visible mappoints: "<<image.NumVisibleMapPoints()<<std::endl;
    // Check if enough 2D-3D correspondence.
    if (image.NumVisibleMapPoints() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    // Search for 2D-3D correspondence.
    const int kCorrTransitivity = 1;

    tri_corrs.clear();
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;
    std::vector<GP3PEstimator::X_t> points2D_normalized;
    std::vector<uint64_t> mappoints_create_time;

    std::vector<int> tri_camera_indices;

    bool has_bogus_params = false;
    std::unordered_map<camera_t, std::string> camera_param_map;
    std::vector<mappoint_t> visible_mappoint_ids;

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        std::vector<CorrespondenceGraph::Correspondence> corrs;
            // correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);
        correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<mappoint_t> mappoint_ids;
        CHECK_LT(point2D_idx,local_camera_indices.size());
        uint32_t local_camera_id = local_camera_indices[point2D_idx];

        for (const auto corr : corrs) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint()) {
                continue;
            }

            // Avoid duplicate correspondence.
            if (mappoint_ids.count(corr_point2D.MapPointId()) > 0) {
                continue;
            }
            CHECK(reconstruction_->ExistsMapPoint(corr_point2D.MapPointId()));
            const MapPoint& mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());
            
            tri_corrs.emplace_back(point2D_idx, corr_point2D.MapPointId());
            mappoint_ids.insert(corr_point2D.MapPointId());
            visible_mappoint_ids.push_back(corr_point2D.MapPointId());

            tri_points3D.push_back(mappoint.XYZ());
            tri_points2D.push_back(point2D.XY());
            tri_camera_indices.push_back(local_camera_id);

            Eigen::Vector2d point2D_normalized = camera.LocalImageToWorld(local_camera_id, point2D.XY());

            points2D_normalized.emplace_back();
            points2D_normalized.back().rel_tform = local_transforms[local_camera_id];
            points2D_normalized.back().xy = point2D_normalized;
            mappoints_create_time.push_back(mappoint.CreateTime());
        }
    }
    
    if (tri_points3D.size() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        std::cout << "tri_points3D.size() = " << tri_points3D.size() << ", lower than "
                  << options.abs_pose_min_num_inliers << std::endl;
        return false;
    }
    std::cout << "tri_points3D size: " << tri_points3D.size() << std::endl;

    RANSACOptions ransac_options;
    ransac_options.max_error = 1 - cos(0.0299977504);  // camera.LocalImageToWorldThreshold(0,options.abs_pose_max_error);
    ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    ransac_options.min_num_trials = 30;
    ransac_options.confidence = 0.9999;
    RANSAC<GP3PEstimator> ransac(ransac_options);
    const auto report = ransac.Estimate(points2D_normalized, tri_points3D);

    inlier_mask = report.inlier_mask;
    size_t num_inliers = report.support.num_inliers;

    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers) ||
        (static_cast<double>(num_inliers) <
         static_cast<double>(tri_points3D.size()) * options.abs_pose_min_inlier_ratio)) {
        std::cout << "num_inliers " << num_inliers << ", lower than " << options.abs_pose_min_num_inliers << " or "
                  << " inlier ratio lower than " << options.abs_pose_min_inlier_ratio << std::endl;
        return false;
    }

    std::vector<double> mappoint_weights(tri_points3D.size());
    std::fill(mappoint_weights.begin(), mappoint_weights.end(), 1.0);
    if (options.map_update && options.update_with_sequential_mode) {
        for (int i = 0; i < tri_points3D.size(); ++i) {
            if (prev_mappoint_ids_.find(visible_mappoint_ids[i]) != prev_mappoint_ids_.end()) {
                mappoint_weights[i] = 20.0;
            }
        }
    }

    // re estimate with inliers

    Eigen::Matrix3d r = report.model.block<3, 3>(0, 0);
    Eigen::Vector3d t = report.model.col(3);

    image.Qvec() = RotationMatrixToQuaternion(r);
    image.Tvec() = t;

    int num_inliers_reprojection = 0;
    for (size_t i = 0; i < tri_points2D.size(); ++i) {
        int local_camera_id = tri_camera_indices[i];
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

        Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(), local_qvec);
        Eigen::Vector3d tvec = QuaternionToRotationMatrix(local_qvec) * image.Tvec() + local_tvec;

        double squared_reproj_error =
            CalculateSquaredReprojectionErrorRig(tri_points2D[i], tri_points3D[i], qvec, tvec, local_camera_id, camera);
        if (squared_reproj_error < options.abs_pose_max_error * options.abs_pose_max_error) {
            num_inliers_reprojection++;
        }
    }
    std::cout << "num inliers reprojection: " << num_inliers_reprojection << std::endl;
    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    if (!options.single_camera) {
        abs_pose_refinement_options.refine_focal_length = false;
        abs_pose_refinement_options.refine_extra_params = false;
    }

    // Pose refinement
    if (!RefineAbsolutePoseRig(abs_pose_refinement_options, inlier_mask, tri_points2D, tri_points3D, 
                               reconstruction_->NumRegisterImages(), mappoints_create_time,
                               mappoint_weights, tri_camera_indices, &image.Qvec(), &image.Tvec(), &camera)) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }

    // num_inliers_reprojection = 0;
    // for (size_t i = 0; i < tri_points2D.size(); ++i) {
    //     int local_camera_id = tri_camera_indices[i];
    //     Eigen::Vector4d local_qvec;
    //     Eigen::Vector3d local_tvec;
    //     camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

    //     Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(), local_qvec);
    //     Eigen::Vector3d tvec = QuaternionToRotationMatrix(local_qvec) * image.Tvec() + local_tvec;

    //     double squared_reproj_error =
    //         CalculateSquaredReprojectionErrorRig(tri_points2D[i], tri_points3D[i], qvec, tvec, local_camera_id, camera);
    //     if (squared_reproj_error < options.abs_pose_max_error * options.abs_pose_max_error) {
    //         num_inliers_reprojection++;
    //     }
    // }
    // std::cout << "num inliers reprojection after pose refine: " << num_inliers_reprojection << std::endl;

    // Update data
    refined_cameras_.insert(image.CameraId());

    if (inlier_num != NULL) {
        *inlier_num = num_inliers;
    }
    std::cout << "inlier num: " << num_inliers << std::endl;

    image.SetPoseFlag(true);

    offline_slam_last_frame_id_ = image_id;
    offline_slam_start_id_ = image_id;

    prev_mappoint_ids_.clear();
    prev_mappoint_ids_.insert(visible_mappoint_ids.begin(), visible_mappoint_ids.end());

    return true;
}


bool IncrementalMapper::EstimateCameraPoseRig(const Options & options, Camera camera, 
                                            const std::vector<Eigen::Vector2d>&  tri_points2D, 
                                            const std::vector<int>& tri_camera_indices,
                                            const std::vector<Eigen::Vector3d>&  tri_points3D, 
                                            Eigen::Vector3d& pose_tvec,
                                            Eigen::Vector4d& pose_qvec,
                                            std::vector<char>& inlier_mask, size_t* inlier_num) const {
    auto local_options = options;
    CHECK(local_options.Check());
    if (inlier_num != NULL) {
        *inlier_num = 0;
    }

    std::vector<Eigen::Matrix3x4d> local_transforms;
    local_transforms.resize(camera.NumLocalCameras());

    for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        Eigen::Matrix3x4d local_transform = ComposeProjectionMatrix(local_qvec, local_tvec);
        local_transforms[i] = local_transform;
    }

    std::vector<GP3PEstimator::X_t> points2D_normalized;
    for( int i = 0; i < tri_points2D.size(); i++ ) {
        Eigen::Vector2d point2D_normalized = camera.LocalImageToWorld(tri_camera_indices[i], tri_points2D[i]);
        points2D_normalized.emplace_back();
        points2D_normalized.back().rel_tform = local_transforms[tri_camera_indices[i]];
        points2D_normalized.back().xy = point2D_normalized;
    }


    if (tri_points3D.size() < static_cast<size_t>(local_options.abs_pose_min_num_inliers)) {
        std::cout << "tri_points3D.size() = " << tri_points3D.size() << ", lower than "
                  << local_options.abs_pose_min_num_inliers << std::endl;
        return false;
    }
    std::cout << "tri_points3D size: " << tri_points3D.size() << std::endl;

    RANSACOptions ransac_options;
    ransac_options.max_error =
        1 - cos(0.0299977504);  // camera.LocalImageToWorldThreshold(0,local_options.abs_pose_max_error);
    ransac_options.min_inlier_ratio = local_options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    ransac_options.min_num_trials = 30;
    ransac_options.confidence = 0.9999;

    RANSAC<GP3PEstimator>::Report report;

    bool success = false;
    size_t num_inliers;
    do {
        RANSAC<GP3PEstimator> ransac(ransac_options);
        report = ransac.Estimate(points2D_normalized, tri_points3D);
        // if (!report.success) {
        //     return false;
        // }

        inlier_mask = report.inlier_mask;
        num_inliers = report.support.num_inliers;

        if (num_inliers >= local_options.abs_pose_min_num_inliers &&
           (num_inliers >= tri_points3D.size() * local_options.abs_pose_min_inlier_ratio)) {
            success = true;
            break;
         }

        float abs_pose_min_inlier_ratio = local_options.abs_pose_min_inlier_ratio * 0.5;
        if (abs_pose_min_inlier_ratio < 0.1) {
            success = false;
            break;
        }
        std::cout << StringPrintf("reset register inlier: %f -> %f\n", local_options.abs_pose_min_inlier_ratio, 
                                                                       abs_pose_min_inlier_ratio);
        local_options.abs_pose_min_inlier_ratio = abs_pose_min_inlier_ratio;
    } while(true);

    if (!success) {
        std::cout << " inlier ratio lower than " << local_options.abs_pose_min_inlier_ratio << std::endl;
        return false;
    }

    // re estimate with inliers

    Eigen::Matrix3d r = report.model.block<3, 3>(0, 0);
    Eigen::Vector3d t = report.model.col(3);

    pose_qvec = RotationMatrixToQuaternion(r);
    pose_tvec = t;

    int num_inliers_reprojection = 0;

    for (size_t i = 0; i < tri_points2D.size(); ++i) {
        int local_camera_id = tri_camera_indices[i];
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

        Eigen::Vector4d qvec = ConcatenateQuaternions(pose_qvec, local_qvec);
        Eigen::Vector3d tvec = QuaternionToRotationMatrix(local_qvec) * pose_tvec + local_tvec;

        double squared_reproj_error =
            CalculateSquaredReprojectionErrorRig(tri_points2D[i], tri_points3D[i], qvec, tvec, local_camera_id, camera);
        if (squared_reproj_error < local_options.abs_pose_max_error * local_options.abs_pose_max_error) {
            num_inliers_reprojection++;
        }
    }
    std::cout << "num inliers reprojection: " << num_inliers_reprojection << std::endl;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    if (!local_options.single_camera) {
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = false;
    }

    // Pose refinement
    if (!RefineAbsolutePoseRig(abs_pose_refinement_options, inlier_mask, tri_points2D, tri_points3D, 
                               1, std::vector<uint64_t>(), std::vector<double>(), 
                               tri_camera_indices, &pose_qvec, &pose_tvec, &camera)) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }

    if (inlier_num != NULL) {
        *inlier_num = num_inliers;
    }
    std::cout << "inlier num: " << num_inliers << std::endl;

    return true;
}

bool IncrementalMapper::EstimateSweepPose(const Options & options,
                                          const BundleAdjustmentOptions& ba_options,
                                          const sweep_t sweep_id, 
                                          const image_t image_id) {
    auto & lidar_sweep = reconstruction_->LidarSweep(sweep_id);

    std::cout << StringPrintf("lidar#%d: %s", sweep_id, JoinPaths(options.lidar_path, lidar_sweep.Name()).c_str()) << std::endl;

    auto pc = ReadPCD(JoinPaths(options.lidar_path, lidar_sweep.Name()));
    if (pc.info.height == 1 ){
        RebuildLivoxMid360(pc, 4);
    }
    lidar_sweep.Setup(pc, lidar_sweep.timestamp_);
    // lidar_sweep.Setup(pc, reconstruction_->NumRegisterImages());
    // lidar_sweep.FilterPointCloud(0.3);

    // TODO: estimate lidar sweep pose with IMU.
    class Image & image = reconstruction_->Image(image_id);
    Eigen::Matrix4d world2cam = Eigen::Matrix4d::Identity();
    world2cam.topRows(3) = image.ProjectionMatrix();

    Eigen::Matrix4d lidar2cam = Eigen::Matrix4d::Identity();
    lidar2cam.topRows(3) = options.lidar_to_cam_matrix;

    Eigen::Matrix3x4d world2lidar = (lidar2cam.inverse() * world2cam).topRows(3);
    Eigen::Matrix3d R =  world2lidar.block<3, 3>(0, 0);
    Eigen::Vector4d qvec = RotationMatrixToQuaternion(R);
    Eigen::Vector3d tvec = world2lidar.block<3, 1>(0, 3);
    Eigen::Vector3d C = -R.transpose() * tvec;

    lidar_sweep.SetQvec(qvec);
    lidar_sweep.SetTvec(tvec);
    Eigen::Vector3d view_dirct = lidar_sweep.ViewingDirection();

    const double angle_thresh = 20.0;
    const double cos_threshold = std::cos(DEG2RAD(angle_thresh));
    // long long delta_time = std::abs(lidar_sweep.timestamp_ - image.timestamp_);
    // std::cout << "delta time: " << delta_time / 1e6 << " ms" << std::endl;
    sweep_t nearest_sweep_id = -1, near_distance_sweep_id = sweep_id;
    std::vector<std::pair<sweep_t, double> > nearest_sweeps;
    double delta_time = std::numeric_limits<double>::max(), delta_time1;
    double min_dist = std::numeric_limits<double>::max();
    // double min_dist = 1.0;
    const std::vector<sweep_t> & reg_sweep_ids = reconstruction_->RegisterSweepIds();
    for (sweep_t reg_sweep_id : reg_sweep_ids) {
        if (reg_sweep_id == sweep_id) continue;
        auto & reg_lidar_sweep = reconstruction_->LidarSweep(reg_sweep_id);
        double time_diff = std::abs((double)lidar_sweep.timestamp_ - (double)reg_lidar_sweep.timestamp_);
        if (delta_time > time_diff) {
            delta_time = time_diff;
            nearest_sweep_id = reg_sweep_id;
        }
        if (time_diff < (double)1e9) {
            nearest_sweeps.emplace_back(reg_sweep_id, time_diff);
        }
        double dist_factor = (reg_lidar_sweep.ProjectionCenter() - C).norm();
        // double cos_theta = reg_lidar_sweep.ViewingDirection().dot(view_dirct);
        // double dist = dist_factor / (cos_theta - 0.8);
        Eigen::Matrix3d R_diff = reg_lidar_sweep.RotationMatrix().transpose() * R;
        Eigen::AngleAxisd angle_axis(R_diff);
        double R_rad = angle_axis.angle();
        double R_angle = RAD2DEG(R_rad);
        if (R_angle >= 0 && R_angle < angle_thresh && dist_factor < 1.5) {
            double dist = dist_factor / (std::cos(R_rad) - cos_threshold);
            if (dist < min_dist) {
                min_dist = dist;
                near_distance_sweep_id = reg_sweep_id;
                delta_time1 = std::abs((double)lidar_sweep.timestamp_ - (double)reg_lidar_sweep.timestamp_);
            }
        }
    }

    std::cout << StringPrintf("sweep id: %d, nearest sweep id: %d, near distance sweep id: %d, delta time: %.3f ms/%.3f ms\n", 
                              sweep_id, nearest_sweep_id, near_distance_sweep_id, delta_time / 1e6, delta_time1 / 1e6);

    auto & nearest_lidar_sweep = reconstruction_->LidarSweep(nearest_sweep_id);
    auto & near_lidar_sweep = reconstruction_->LidarSweep(near_distance_sweep_id);

    std::cout << "sweep name: " << lidar_sweep.Name() << std::endl;
    std::cout << "nearest sweep name: " << nearest_lidar_sweep.Name() << std::endl;
    std::cout << "near distance sweep name: " << near_lidar_sweep.Name() << std::endl;
    std::cout << "nearest distance: " << (nearest_lidar_sweep.ProjectionCenter() - C).norm() << std::endl;
    std::cout << "near distance: " << (near_lidar_sweep.ProjectionCenter() - C).norm() << std::endl;
    Eigen::Matrix3d R_diff = near_lidar_sweep.RotationMatrix().transpose() * R;
    Eigen::AngleAxisd angle_axis(R_diff);
    double R_angle = RAD2DEG(angle_axis.angle());
    std::cout << "near angle diff: " << R_angle << std::endl;

    std::sort(nearest_sweeps.begin(), nearest_sweeps.end(), 
        [&](const std::pair<sweep_t, double> & a, const std::pair<sweep_t, double> & b) {
        return a.second < b.second;
    });
    int max_ref_lidar = std::min((int)nearest_sweeps.size(), 1);
    nearest_sweeps.resize(max_ref_lidar);
    std::cout << "nearest_sweeps: " << nearest_sweeps.size() << " " << max_ref_lidar << std::endl;
    std::cout << "nearest sweep ids: ";
    for (auto nearest : nearest_sweeps) {
        std::cout << nearest.first << " ";
    }
    std::cout << std::endl;

    std::vector<sweep_t> sweep_ids;
    std::unordered_set<sweep_t> fix_sweep_ids;

    reconstruction_->RegisterLidarSweep(sweep_id);

    bool success = false;
    if (near_distance_sweep_id != sweep_id){
        sweep_ids.push_back(sweep_id);
        sweep_ids.push_back(near_distance_sweep_id);
        fix_sweep_ids.insert(near_distance_sweep_id);

        // if (nearest_sweeps.size() > 0) {
            for (auto nearest : nearest_sweeps) {
                if (near_distance_sweep_id != nearest.first) {
                    sweep_ids.push_back(nearest.first);
                    fix_sweep_ids.insert(nearest.first);
                }
            }
        // }

        double final_cost1;
        success = AdjustFrame2FrameBundle(options, ba_options, sweep_ids, fix_sweep_ids, final_cost1);

        // if (delta_time < 1e9 && nearest_sweeps.size() > 0) {
        //     std::cout << "check with nearest LiDAR frame" << std::endl;
        //     Eigen::Vector4d old_qvec = lidar_sweep.Qvec();
        //     Eigen::Vector3d old_tvec = lidar_sweep.Tvec();
        //     Eigen::Vector3d old_C = lidar_sweep.ProjectionCenter();

        //     // lidar_sweep.SetQvec(reconstruction_->LidarSweep(nearest_sweep_id).Qvec());
        //     // lidar_sweep.SetTvec(reconstruction_->LidarSweep(nearest_sweep_id).Tvec());
        //     lidar_sweep.SetQvec(reconstruction_->LidarSweep(nearest_sweep_id).Qvec());
        //     if (lidar_velocities.find(nearest_sweep_id) != lidar_velocities.end()) {
        //         Eigen::Vector3d velocity = lidar_velocities[nearest_sweep_id];
        //         double time_diff = ((double)lidar_sweep.timestamp_ - (double)nearest_lidar_sweep.timestamp_) / (double)1e9;
        //         Eigen::Vector3d new_pos = nearest_lidar_sweep.ProjectionCenter() + velocity * time_diff;
        //         Eigen::Vector3d new_tvec = -nearest_lidar_sweep.RotationMatrix() * new_pos;
        //         std::cout << "velocity: " << velocity.transpose() << std::endl;
        //         std::cout << "time diff: " << time_diff << std::endl;
        //         std::cout << "tvec diff: " << (new_tvec - tvec).transpose() << std::endl;
        //         lidar_sweep.SetTvec(new_tvec);
        //     } else {
        //         lidar_sweep.SetTvec(nearest_lidar_sweep.Tvec());
        //     }
           
        //     sweep_ids.clear();
        //     sweep_ids.push_back(sweep_id);

        //     fix_sweep_ids.clear();
        //     for (auto nearest : nearest_sweeps) {
        //         if (sweep_id != nearest.first) {
        //             sweep_ids.push_back(nearest.first);
        //             fix_sweep_ids.insert(nearest.first);
        //         }
        //     }

        //     double final_cost2;
        //     AdjustFrame2FrameBundle(options, ba_options, sweep_ids, fix_sweep_ids, final_cost2);

        //     Eigen::Vector4d qvec_diff;
        //     qvec_diff = ConcatenateQuaternions(InvertQuaternion(old_qvec), lidar_sweep.Qvec());
        //     Eigen::AngleAxisd angle_axis(QuaternionToRotationMatrix(qvec_diff));
        //     double R_angle = angle_axis.angle();

        //     Eigen::Vector3d C = lidar_sweep.ProjectionCenter();
        //     double C_dist = (C - old_C).norm();

        //     std::cout << "pose diff between nearest LiDAR frame: " << RAD2DEG(R_angle) << "deg, " << C_dist << std::endl;
        //     std::cout << "final_cost: " << final_cost1 << ", " << final_cost2 << std::endl;

        //     // if (RAD2DEG(R_angle) < 10 && C_dist < 0.2) {
        //     if (final_cost1 < final_cost2) {
        //         std::cout << "estimate pose using near distance LiDAR frame" << std::endl;
        //         lidar_sweep.SetQvec(old_qvec);
        //         lidar_sweep.SetTvec(old_tvec);
        //     }
        // }
    } else if (nearest_sweep_id != sweep_id && nearest_sweeps.size() > 0) {
        std::cout << "estimate pose using nearest LiDAR frame" << std::endl;
        sweep_ids.push_back(sweep_id);
        sweep_ids.push_back(nearest_sweep_id);
        fix_sweep_ids.insert(nearest_sweep_id);

        for (auto nearest : nearest_sweeps) {
            if (nearest_sweep_id != nearest.first) {
                sweep_ids.push_back(nearest.first);
                fix_sweep_ids.insert(nearest.first);
            }
        }

        double final_cost1;
        success = AdjustFrame2FrameBundle(options, ba_options, sweep_ids, fix_sweep_ids, final_cost1);
    } else {
        std::cout << "Lidar Sweep " << sweep_id << "(" << lidar_sweep.Name() << ") is common view sweeps ... continue " << std::endl;
    }
    reconstruction_->DeRegisterLidarSweep(sweep_id);

    if (!success) {
        lidar_sweep.SetQvec(qvec);
        lidar_sweep.SetTvec(tvec);
    }

    bool is_debug = false;
    Eigen::Matrix4d f2f_T = Eigen::Matrix4d::Identity();
    f2f_T.topRows(3) = lidar_sweep.ProjectionMatrix();

    // if (is_debug){
    //     lidar_sweep.SetQvec(qvec);
    //     lidar_sweep.SetTvec(tvec);
    //     // return false;
    // }

    LidarAbsolutePoseRefinementOptions absolute_pose_options;
    absolute_pose_options.gradient_tolerance = 1e-6;
    absolute_pose_options.error_tolerance = 1e-5;
    absolute_pose_options.num_iteration_pose_estimation = 50;

    // Refine lidar scan pose.
    LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
    LidarPointCloud ref_corners = lidar_sweep.GetCornerPointsLessSharp();
    // LidarPointCloud ref_corners = lidar_sweep.GetCornerPointsSharp();

    std::cout << "surface points: " << ref_less_surfs.points.size() << std::endl;
    std::cout << "corner points: " << ref_corners.points.size() << std::endl;

    std::vector<lidar::OctoTree::Point> points;
    points.reserve(ref_less_surfs.points.size() + ref_less_surfs.points.size());
    for (auto & point : ref_less_surfs.points) {
        lidar::OctoTree::Point X;
        X.x = point.x;
        X.y = point.y;
        X.z = point.z;
        X.curv = point.curv;
        X.intensity = point.intensity;
        X.lifetime = point.lifetime;
        // X.lifetime = lidar_sweep.timestamp_;
        X.type = 0;
        points.push_back(X);
    }

    for (auto & point : ref_corners.points) {
        lidar::OctoTree::Point X;
        X.x = point.x;
        X.y = point.y;
        X.z = point.z;
        X.curv = point.curv;
        X.intensity = point.intensity;
        X.lifetime = point.lifetime;
        // X.lifetime = lidar_sweep.timestamp_;
        X.type = 1;
        points.push_back(X);
    }

    uint32_t m_num_visible_point = 0;
    const auto register_image_ids = reconstruction_->RegisterImageIds();
    for (auto register_image_id : register_image_ids) {
        m_num_visible_point += reconstruction_->Image(register_image_id).NumVisibleMapPoints();
    }
    m_num_visible_point /= register_image_ids.size();

    Eigen::Vector4d lidar_qvec = lidar_sweep.Qvec();
    Eigen::Vector3d lidar_tvec = lidar_sweep.Tvec();

    auto & voxel_map = reconstruction_->VoxelMap();
    bool refine_success = RefineAbsolutePose(absolute_pose_options, reconstruction_.get(), image_id, points, 
                                            voxel_map.get(), &lidar_sweep.Qvec(), &lidar_sweep.Tvec());
    if (!refine_success) {
        std::cout << "RefineAbsolutePose failed! using frame2frame pose." << std::endl << std::flush;
        lidar_sweep.Qvec() = lidar_qvec;
        lidar_sweep.Tvec() = lidar_tvec;
        // return false;
    }

    reconstruction_->RegisterLidarSweep(sweep_id);

    image_to_lidar_map_[image_id] = sweep_id;

    if (delta_time < 1e9) {
        Eigen::Vector3d vec;
        if (nearest_lidar_sweep.timestamp_ > lidar_sweep.timestamp_) {
            vec = nearest_lidar_sweep.ProjectionCenter() - lidar_sweep.ProjectionCenter();
        } else {
            vec = lidar_sweep.ProjectionCenter() - nearest_lidar_sweep.ProjectionCenter();
        }
        Eigen::Vector3d velocity = vec * (double)1e9 / (double)delta_time;
        lidar_velocities[sweep_id] = velocity;
    }

    return true;
}

bool IncrementalMapper::EstimateSweepPosesBetweenFramesSequence(const Options & options, 
                                                                const sweep_t sweep_id1, 
                                                                const sweep_t sweep_id2) {

    PrintHeading2(StringPrintf("Estimate Sweep Pose With Sequence"));

    class LidarSweep & lidar_sweep1 = reconstruction_->LidarSweep(sweep_id1);
    class LidarSweep & lidar_sweep2 = reconstruction_->LidarSweep(sweep_id2);
    
    const std::pair<sweep_t, long long> timestamp_t1({sweep_id1, lidar_sweep1.timestamp_});
    auto sweep_bound1 = std::lower_bound(sweep_timestamps_.begin(), sweep_timestamps_.end(), timestamp_t1, 
        [&](const std::pair<sweep_t, long long> & a, const std::pair<sweep_t, long long> & b) {
            return a.second < b.second;
        });

    int forward1 = sweep_bound1 - sweep_timestamps_.begin();
    if (forward1 >= sweep_timestamps_.size() || forward1 <= 0) {
        return false;
    }

    std::cout << "sweep id1: " << sweep_id1 << ", lower bound: " << sweep_bound1->first << std::endl;

    const std::pair<sweep_t, long long> timestamp_t2({sweep_id2, lidar_sweep2.timestamp_});
    auto sweep_bound2 = std::lower_bound(sweep_timestamps_.begin(), sweep_timestamps_.end(), timestamp_t2, 
        [&](const std::pair<sweep_t, long long> & a, const std::pair<sweep_t, long long> & b) {
            return a.second < b.second;
        });

    int forward2 = sweep_bound2 - sweep_timestamps_.begin();
    if (forward2 >= sweep_timestamps_.size() || forward2 <= 0) {
        return false;
    }

    std::cout << "sweep id2: " << sweep_id2 << ", lower bound: " << sweep_bound2->first << std::endl;

    auto & voxel_map = reconstruction_->VoxelMap();

    auto PropagatePose = [&](const sweep_t prev_sweep_id, const sweep_t next_sweep_id, const bool estimate_pose = false) {
        auto & prev_lidar_sweep = reconstruction_->LidarSweep(prev_sweep_id);
        auto & next_lidar_sweep = reconstruction_->LidarSweep(next_sweep_id);

        std::cout << StringPrintf("lidar#%d: %s", next_sweep_id, JoinPaths(options.lidar_path, next_lidar_sweep.Name()).c_str()) << std::endl;

        auto pc = ReadPCD(JoinPaths(options.lidar_path, next_lidar_sweep.Name()));
        if (pc.info.height == 1 ){
            RebuildLivoxMid360(pc, 4);
        }
        next_lidar_sweep.Setup(pc, next_lidar_sweep.timestamp_);
        // next_lidar_sweep.Setup(pc, reconstruction_->NumRegisterImages());
        // next_lidar_sweep.FilterPointCloud(0.3);

        double delta_time = std::abs((double)next_lidar_sweep.timestamp_ - (double)prev_lidar_sweep.timestamp_);
        std::cout << StringPrintf("prev sweep id: %d, next sweep id: %d, delta time: %.3f ms\n", prev_sweep_id, next_sweep_id, delta_time / 1e6);

        next_lidar_sweep.SetQvec(prev_lidar_sweep.Qvec());
        next_lidar_sweep.SetTvec(prev_lidar_sweep.Tvec());

        if (estimate_pose) {
            std::vector<sweep_t> sweep_ids;
            sweep_ids.push_back(prev_sweep_id);
            sweep_ids.push_back(next_sweep_id);
            
            std::unordered_set<sweep_t> fix_sweep_ids;
            fix_sweep_ids.insert(prev_sweep_id);

            reconstruction_->RegisterLidarSweep(next_sweep_id);

            BundleAdjustmentOptions ba_options;
            ba_options.max_num_iteration_frame2frame = 3;
            double final_cost;
            AdjustFrame2FrameBundle(options, ba_options, sweep_ids, fix_sweep_ids, final_cost);

            reconstruction_->DeRegisterLidarSweep(next_sweep_id);
        }

        LidarAbsolutePoseRefinementOptions absolute_pose_options;
        absolute_pose_options.gradient_tolerance = 1e-6;
        absolute_pose_options.error_tolerance = 1e-5;
        absolute_pose_options.num_iteration_pose_estimation = 10;

        // Refine lidar scan pose.
        LidarPointCloud ref_less_surfs = next_lidar_sweep.GetSurfPointsLessFlat();
        LidarPointCloud ref_corners = next_lidar_sweep.GetCornerPointsLessSharp();
        // LidarPointCloud ref_corners = next_lidar_sweep.GetCornerPointsSharp();

        std::cout << "surface points: " << ref_less_surfs.points.size() << std::endl;
        std::cout << "corner points: " << ref_corners.points.size() << std::endl;

        std::vector<lidar::OctoTree::Point> points;
        points.reserve(ref_less_surfs.points.size() + ref_corners.points.size());
        for (auto & point : ref_less_surfs.points) {
            lidar::OctoTree::Point X;
            X.x = point.x;
            X.y = point.y;
            X.z = point.z;
            X.curv = point.curv;
            X.intensity = point.intensity;
            X.lifetime = point.lifetime;
            // X.lifetime = next_lidar_sweep.timestamp_;
            X.type = 0;
            points.push_back(X);
        }

        for (auto & point : ref_corners.points) {
            lidar::OctoTree::Point X;
            X.x = point.x;
            X.y = point.y;
            X.z = point.z;
            X.curv = point.curv;
            X.intensity = point.intensity;
            X.lifetime = point.lifetime;
            // X.lifetime = next_lidar_sweep.timestamp_;
            X.type = 1;
            points.push_back(X);
        }

        if (!RefineAbsolutePose(absolute_pose_options, reconstruction_.get(), last_keyframe_idx, points, 
                                voxel_map.get(), &next_lidar_sweep.Qvec(), &next_lidar_sweep.Tvec())) {
            std::cout << "RefineAbsolutePose failed!" << std::endl << std::flush;
            return false;
        }

        const double num_inlier_threshold = 0.7;
        size_t num_inlier_corner = 0, num_valid_corner = 0;
        std::vector<double> dists;
        dists.reserve(ref_corners.points.size());
        double min_dist = std::numeric_limits<double>::max();
        double max_dist = std::numeric_limits<double>::lowest();
        Eigen::Matrix3x4d inv_proj_matrix = next_lidar_sweep.InverseProjectionMatrix();
        for (auto & point : ref_corners.points) {
            Eigen::Vector4d Xc(point.x, point.y, point.z, 1.0);
            Eigen::Vector3d Xw = inv_proj_matrix * Xc;
            lidar::OctoTree::Point X;
            X.x = Xw[0];
            X.y = Xw[1];
            X.z = Xw[2];
            X.type = 1;
            auto loc = voxel_map->LocateCornerPoint(X);
            if (loc != nullptr) {
                std::vector<Eigen::Vector3d> nearest_points(1);
                if (loc->FindKNeighbor(Xw, nearest_points, 1)) {
                    double dist = (Xw - nearest_points[0]).norm();
                    dists.push_back(dist);
                    min_dist = std::min(min_dist, dist);
                    max_dist = std::max(max_dist, dist);
                    if (dist < 0.2) {
                        num_inlier_corner++;
                    }
                }
                num_valid_corner++;
            }
        }
        
        double m_dist = 0.0;
        if (dists.size() > 0) {
            size_t nth = dists.size() / 2;
            std::nth_element(dists.begin(), dists.begin() + nth, dists.end());
            m_dist = dists[nth];
            std::cout << "min_dist: " << min_dist << std::endl;
            std::cout << "max_dist: " << max_dist << std::endl;
            std::cout << "mid_dist: " << m_dist << std::endl;
        }
        double corner_inlier_ratio = num_inlier_corner * 1.0 / (num_valid_corner + 1);
        std::cout << "corner inlier: " << num_inlier_corner << "/" << num_valid_corner << ", ratio: " << corner_inlier_ratio << std::endl;
        if (corner_inlier_ratio < num_inlier_threshold && m_dist > 0.2) {
            {
                Eigen::Matrix4d htrans = Eigen::Matrix4d::Identity();
                htrans.topRows(3) = next_lidar_sweep.ProjectionMatrix();
                Eigen::Matrix4d T = htrans.inverse();
                Eigen::Matrix4d initT = Eigen::Matrix4d::Identity();
                initT.topRows(3) = prev_lidar_sweep.InverseProjectionMatrix();

                LidarPointCloud ref_less_features, ref_less_features_t, ref_less_features_t1;
                ref_less_features = next_lidar_sweep.GetCornerPointsLessSharp();
                LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T);
                LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t1, initT);

                std::string ref_les_name = StringPrintf("%s/debug/%s/%s-opt.ply", workspace_path_.c_str(), next_lidar_sweep.Name().c_str(), next_lidar_sweep.Name().c_str());
                std::string parent_path = GetParentDir(ref_les_name);
                if (!ExistsPath(parent_path)) {
                    boost::filesystem::create_directories(parent_path);
                }
                WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t.Convert2Ply(), false, true);

                ref_les_name = StringPrintf("%s/debug/%s/%s-init.ply", workspace_path_.c_str(), next_lidar_sweep.Name().c_str(), next_lidar_sweep.Name().c_str());
                WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t1.Convert2Ply(), false, true);

                // source point cloud.
                Eigen::Matrix4d srcT = Eigen::Matrix4d::Identity();
                srcT.topRows(3) = prev_lidar_sweep.InverseProjectionMatrix();

                LidarPointCloud src_less_features, src_less_features_t;
                src_less_features = prev_lidar_sweep.GetCornerPointsLessSharp();
                LidarPointCloud::TransfromPlyPointCloud (src_less_features, src_less_features_t, srcT);

                std::string src_les_name = StringPrintf("%s/debug/%s/%s.ply", workspace_path_.c_str(), next_lidar_sweep.Name().c_str(), prev_lidar_sweep.Name().c_str());
                WriteBinaryPlyPoints(src_les_name.c_str(), src_less_features_t.Convert2Ply(), false, true);
            }
            return false;
        }

        reconstruction_->RegisterLidarSweep(next_sweep_id);
        return true;
    };

    if (forward1 < forward2) {
        bool estimate_pose = true;
        sweep_t prev_sweep_id = sweep_bound1->first;
        for (int i = forward1 + 1; i <= forward2; ++i) {
            auto iter = sweep_timestamps_.begin();
            std::advance(iter, i);
            sweep_t next_sweep_id = iter->first;

            std::cout << StringPrintf("Propagate frame %d to %d\n", prev_sweep_id, next_sweep_id);

            bool success = PropagatePose(prev_sweep_id, next_sweep_id, estimate_pose);
            if (success) {
                if (i != forward2) AppendToVoxelMap(options, next_sweep_id);
                // estimate_pose = false;
                prev_sweep_id = next_sweep_id;
            } else {
                // estimate_pose = true;
                std::cout << "Propagate failed!" << std::endl;
            }
        }
    } else {
        bool estimate_pose = true;
        sweep_t prev_sweep_id = sweep_bound1->first;
        for (int i = forward1 - 1; i >= forward2; --i) {
            auto iter = sweep_timestamps_.begin();
            std::advance(iter, i);
            sweep_t next_sweep_id = iter->first;

            std::cout << StringPrintf("Propagate frame %d to %d\n", prev_sweep_id, next_sweep_id);

            bool success = PropagatePose(prev_sweep_id, next_sweep_id, estimate_pose);
            if (success) {
                if (i != forward2) AppendToVoxelMap(options, next_sweep_id);
                // estimate_pose = false;
                prev_sweep_id = next_sweep_id;
            } else {
                // estimate_pose = true;
                std::cout << "Propagate failed!" << std::endl;
            }
        }
    }

    PrintHeading2("");

    return true;
}

bool IncrementalMapper::CheckLocalPoseConsistency(const Options& options, const image_t image_id) {
    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    const Image& image = reconstruction_->Image(image_id);

    Eigen::Vector4d qvec = image.Qvec();
    Eigen::Vector3d tvec = image.Tvec();
    Eigen::Matrix3d R = QuaternionToRotationMatrix(qvec);

    // Find registered neighbors which have strongest connectivities to the
    // current image
    std::unordered_set<image_t> neighbors = correspondence_graph->ImageNeighbor(image_id);

    std::vector<std::pair<image_t, double>> neighbor_connectivities;

    for (const auto neighbor : neighbors) {
        class Image& neighbor_image = reconstruction_->Image(neighbor);
        if (neighbor_image.IsRegistered()) {
            auto image_pair = correspondence_graph->ImagePair(image_id, neighbor);
            if (image_pair.two_view_geometry.relative_pose_valid) {
                neighbor_connectivities.emplace_back(neighbor, static_cast<double>(image_pair.num_correspondences));
            }
        }
    }

    // if no reliable neighbors exist, check is simply ignored
    if (neighbor_connectivities.size() == 0) {
        return true;
    }

    std::sort(
        neighbor_connectivities.begin(), neighbor_connectivities.end(),
        [](const std::pair<image_t, double> p1, const std::pair<image_t, double> p2) { return p1.second > p2.second; });

    if (neighbor_connectivities.size() >= 5) {
        neighbor_connectivities.resize(5);
    }
    std::cout << "Connected registered neighbor count: " << neighbor_connectivities.size() << std::endl;

    // verify the global pose via the two view geometries with the neighbors
    std::cout << "Differences of R and C: " << std::endl;
    int positive_vote = 0;
    for (const auto neighbor : neighbor_connectivities) {
        bool positive = true;

        Image& neighbor_image = reconstruction_->Image(neighbor.first);
        std::cout << neighbor_image.Name() << " ";

        Eigen::Vector4d neighbor_qvec = neighbor_image.Qvec();
        Eigen::Vector3d neighbor_tvec = neighbor_image.Tvec();
        Eigen::Matrix3d neighbor_R = QuaternionToRotationMatrix(neighbor_qvec);

        auto image_pair = correspondence_graph->ImagePair(image_id, neighbor.first);

        Eigen::Vector4d two_view_qvec = image_pair.two_view_geometry.qvec;
        Eigen::Vector3d two_view_tvec = image_pair.two_view_geometry.tvec;

        // cvec denotes the coordinate of the current camera center in the
        // coordframe of the neighbor
        Eigen::Vector3d two_view_cvec = -(QuaternionToRotationMatrix(InvertQuaternion(two_view_qvec))) * two_view_tvec;

        if (neighbor.first == image_pair.image_id2) {
            // if the two view geometry is from the current image to the neighbor,
            // the geometry should be reverted.
            two_view_qvec = InvertQuaternion(two_view_qvec);
            two_view_cvec = two_view_tvec;
        }

        Eigen::Vector4d delta_qvec =
            ConcatenateQuaternions(neighbor_qvec, ConcatenateQuaternions(two_view_qvec, InvertQuaternion(qvec)));

        Eigen::Matrix3d delta_R = QuaternionToRotationMatrix(delta_qvec);
        double delta_theta = RadToDeg(acos((delta_R.trace() - 1) * 0.5));

        if (delta_theta > 20) {
            positive = false;
        }
        std::cout << "R angle difference: " << delta_theta << " ";

        Eigen::Vector3d global_cvec = neighbor_tvec - neighbor_R * R.transpose() * tvec;

        if (global_cvec.norm() > 0 && two_view_cvec.norm() > 0) {
            double cos_t_angle = global_cvec.dot(two_view_cvec) / ((global_cvec.norm()) * two_view_cvec.norm());
            cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));
            double t_angle = RadToDeg(std::acos(cos_t_angle));
            if (t_angle > 30) {
                positive = false;
            }
            std::cout << "global cvec: " << global_cvec(0) << " " << global_cvec(1) << " " << global_cvec(2) << " ";
            std::cout << "two view cvec: " << two_view_cvec(0) << " " << two_view_cvec(1) << " " << two_view_cvec(2)
                      << " ";
            std::cout << "T angle difference: " << t_angle << std::endl;
        } else {
            std::cout << "T difference: 0" << std::endl;
        }
        if (positive) {
            ++positive_vote;
        }
    }
    std::cout << "positive vote: " << positive_vote << std::endl;
    if (static_cast<double>(positive_vote) >= static_cast<double>(neighbor_connectivities.size()) * 0.7) {
        return true;
    } else {
        return false;
    }
}

int IncrementalMapper::FindConsecutiveCameraPoseIndex(const Options& options, const image_t image_id,
                                                      const std::vector<double>& estimated_focal_length_factors,
                                                      const std::vector<Eigen::Vector4d>& qvecs,
                                                      const std::vector<Eigen::Vector3d>& tvecs) {
    struct NeighborImagePair {
        image_t image_id2;
        class Image* image2;
        point2D_t num_corrs = 0;

        Eigen::Matrix3d m_relative_R;
        Eigen::Vector3d m_relative_t;
    };

    const size_t topK = options.consecutive_camera_pose_top_k;
    const size_t consistent_neighbor_ori_thres = options.consecutive_neighbor_ori;
    const size_t consistent_neighbor_t_thres = options.consecutive_neighbor_t;

    const auto& correspondence_graph = scene_graph_container_->CorrespondenceGraph();
    const std::unordered_set<image_t>& neighbor_ids = correspondence_graph->ImageNeighbor(image_id);
    class Image& image = reconstruction_->Image(image_id);
    class Camera& camera = reconstruction_->Camera(image.CameraId());

    std::vector<NeighborImagePair> neighbor_image_pairs;

    for (const auto& neighbor_id : neighbor_ids) {
        if (std::fabs((int)image_id - (int)neighbor_id) > 10) {
            continue;
        }
        class Image& image_neighbor = reconstruction_->Image(neighbor_id);
        if (reconstruction_->ExistsImage(neighbor_id) && image_neighbor.HasPose()) {
            NeighborImagePair neighbor_image_pair;
            neighbor_image_pair.image_id2 = neighbor_id;
            neighbor_image_pair.image2 = &image_neighbor;
            neighbor_image_pair.num_corrs =
                correspondence_graph->NumCorrespondencesBetweenImages(image_id, neighbor_id);

            const auto& image_pair = correspondence_graph->ImagePair(image_id, neighbor_id);
            Eigen::Vector4d qvec = image_pair.two_view_geometry.qvec;
            Eigen::Vector3d tvec = image_pair.two_view_geometry.tvec;

            neighbor_image_pair.m_relative_R = QuaternionToRotationMatrix(qvec);
            neighbor_image_pair.m_relative_t = tvec.normalized();
            if (image_id > neighbor_id) {
                Eigen::Matrix3d m_relative_RT = neighbor_image_pair.m_relative_R.transpose();
                neighbor_image_pair.m_relative_R = m_relative_RT;
                neighbor_image_pair.m_relative_t = -m_relative_RT * neighbor_image_pair.m_relative_t;
            }
            neighbor_image_pairs.emplace_back(neighbor_image_pair);
        }
    }

    auto ConsistentCheck = [&]() {
        Eigen::Matrix3d R = QuaternionToRotationMatrix(qvecs[0]);
        Eigen::Vector3d C = -R.transpose() * tvecs[0];

        for (size_t i = 1; i < qvecs.size(); ++i) {
            Eigen::Matrix3d Ri = QuaternionToRotationMatrix(qvecs[i]);
            Eigen::Vector3d Ci = -Ri.transpose() * tvecs[i];

            Eigen::Matrix3d R_diff = R.transpose() * Ri;
            Eigen::AngleAxisd angle_axis(R_diff);
            double R_angle = angle_axis.angle();

            double cos_t_angle = Ci.dot(C) / (Ci.norm() * C.norm());
            cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));
            double t_angle = std::acos(cos_t_angle);
            if (RadToDeg(R_angle) > options.consecutive_camera_pose_orientation &&
                RadToDeg(t_angle) > options.consecutive_camera_pose_t) {
                return -1;
            }
        }
        return 0;
    };

    if (neighbor_image_pairs.size() <= 0) {
        std::cout << "No Registered Neighbors!" << std::endl;
        return ConsistentCheck();
    }

    // sort neighbors by correspondence.
    std::sort(neighbor_image_pairs.begin(), neighbor_image_pairs.end(),
              [&](const NeighborImagePair& image_pair1, const NeighborImagePair& image_pair2) {
                  return image_pair1.num_corrs > image_pair2.num_corrs;
              });

    size_t begin = 0;
    size_t end = std::min(neighbor_image_pairs.size(), begin + topK);
    for (size_t t = begin; t < end; ++t) {
        NeighborImagePair* n_image_pair = &neighbor_image_pairs[t];
        std::cout << reconstruction_->Image(image_id).Name() << ", " << n_image_pair->image2->Name() << std::endl;
    }
    for (size_t i = 0; i < qvecs.size(); ++i) {
        Eigen::Matrix3d R1 = QuaternionToRotationMatrix(qvecs[i]);
        Eigen::Vector3d t1 = tvecs[i];

        std::cout << "Solution#" << i << std::endl;
        size_t consistent_neighbor_ori = 0;
        size_t consistent_neighbor_t = 0;
        for (size_t t = begin; t < end; ++t) {
            NeighborImagePair* n_image_pair = &neighbor_image_pairs[t];
            Eigen::Vector4d qvec2 = n_image_pair->image2->Qvec();
            Eigen::Matrix3d R2 = QuaternionToRotationMatrix(qvec2);
            Eigen::Vector3d t2 = n_image_pair->image2->Tvec();

            Eigen::Matrix3d m_relative_R = R2 * R1.transpose();
            Eigen::Vector3d m_relative_t = (t2 - m_relative_R * t1).normalized();

            Eigen::Matrix3d R_diff = n_image_pair->m_relative_R.transpose() * m_relative_R;
            Eigen::AngleAxisd deltaR(R_diff);
            double d_rad = deltaR.angle();

            double cos_t_rad = n_image_pair->m_relative_t.dot(m_relative_t) /
                               (n_image_pair->m_relative_t.norm() * m_relative_t.norm());
            double t_rad = std::acos(cos_t_rad);

            double d_angle = RadToDeg(d_rad);
            double t_angle = RadToDeg(t_rad);
            d_angle = std::min(d_angle, 180.0 - d_angle);
            if (d_angle < options.consecutive_camera_pose_orientation && t_angle < options.consecutive_camera_pose_t) {
                consistent_neighbor_ori++;
                consistent_neighbor_t++;
            } else if (d_angle < options.consecutive_camera_pose_orientation) {
                consistent_neighbor_ori++;
            }
            std::cout << "d_angle = " << d_angle << ", t_angle = " << t_angle << std::endl;
        }
        if (consistent_neighbor_ori >= std::min(end - begin, consistent_neighbor_ori_thres) &&
            consistent_neighbor_t >= std::min(end - begin, consistent_neighbor_t_thres)) {
            return i;
        }
    }

    return -1;
}

bool IncrementalMapper::RegisterNextImage(const Options& options, const image_t image_id,
                                          std::vector<std::pair<point2D_t, mappoint_t>>& tri_corrs,
                                          std::vector<char>& inlier_mask, size_t* inlier_num) {
    if (!EstimateCameraPose(options, image_id, tri_corrs, inlier_mask, inlier_num)) {
        return false;
    }

    // Continue tracks
    reconstruction_->RegisterImage(image_id);
    RegisterImageEvent(image_id);

    const class Image& image = reconstruction_->Image(image_id);
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            const point2D_t point2D_idx = tri_corrs[i].first;
            const Point2D& point2D = image.Point2D(point2D_idx);
            if (!point2D.HasMapPoint()) {
                const mappoint_t mappoint_id = tri_corrs[i].second;
                const TrackElement track_el(image_id, point2D_idx);
                reconstruction_->AddObservation(mappoint_id, track_el);
            }
        }
    }

    return true;
}

bool IncrementalMapper::AddKeyFrame(const Options& options, const image_t image_id,
                                    std::vector<std::pair<point2D_t, mappoint_t>>& tri_corrs,
                                    std::vector<char>& inlier_mask, bool force, bool unordered) {
    const double max_squared_disp = options.mean_max_disparity_kf * options.mean_max_disparity_kf;

    Image& image = reconstruction_->Image(image_id);
    Eigen::Vector3d C1 = image.ProjectionCenter();

    image.SetPoseFlag(true);

    bool exceed_min_step = true;

    bool condition0 = force;

    int neighborest_id = -1;
    float min_dist_kf = std::numeric_limits<float>::max();
    uint64_t min_time_diff = std::numeric_limits<uint64_t>::max();

    const std::unordered_set<image_t>& neighbor_ids =
            scene_graph_container_->CorrespondenceGraph()->ImageNeighbor(image_id);
    if (!unordered) {
        for (const auto& neighbor_id : neighbor_ids) {
            if (reconstruction_->ExistsImage(neighbor_id) && reconstruction_->IsImageRegistered(neighbor_id)) {
                const Image& image_neighbor = reconstruction_->Image(neighbor_id);
                Eigen::Vector3d C2 = image_neighbor.ProjectionCenter();
                float dist_kf = (C1 - C2).norm();
                if (dist_kf < min_dist_kf) {
                    min_dist_kf = dist_kf;
                    neighborest_id = neighbor_id;
                }
            }
        }

        for (int step = -(options.min_keyframe_step - 1); step <= options.min_keyframe_step - 1; step++) {
            if (step < 0 && image_id <= -step) {
                continue;
            }
            image_t neighbor_id = image_id + step;

            if (reconstruction_->ExistsImage(neighbor_id) && reconstruction_->IsImageRegistered(neighbor_id)) {
                exceed_min_step = false;
                std::cout << "  => skip adding keyframe according to the min step" << std::endl;
                break;
            }
        }
    }

    if (options.lidar_sfm) {
        for (const auto& neighbor_id : neighbor_ids) {
            if (reconstruction_->ExistsImage(neighbor_id) && reconstruction_->IsImageRegistered(neighbor_id)) {
                const Image& image_neighbor = reconstruction_->Image(neighbor_id);
                uint64_t time_diff = std::abs(image.timestamp_ - image_neighbor.timestamp_);
                if (time_diff < min_time_diff) {
                    min_time_diff = time_diff;
                }
            }
        }
    }

    std::cout << "min_time_diff: "  << unordered << ", " << min_time_diff << std::endl;
    bool condition_has_large_time_diff = options.lidar_sfm ? (min_time_diff >= 2e9) : false;

    point2D_t num_visible_map_point = image.NumVisibleMapPoints();
    bool condition1 = (num_visible_map_point < options.min_visible_map_point_kf);
    std::cout << "  => num_visible_map_point = " << num_visible_map_point << std::endl;

    int inlier_count = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_count++;
        }
    }
    bool condition_inlier = inlier_count < options.min_pose_inlier_kf;
    std::cout << "  => num inlier = " << inlier_count << std::endl;

    bool condition2 = false;
    if (!unordered) {
        float avg_min_dist_kf = 0.0f;
        if (seq_num_kf_ > 1) {
            avg_min_dist_kf = acc_min_dist_kf_ / (seq_num_kf_ - 1);
        }
        condition2 = (min_dist_kf >= avg_min_dist_kf * options.avg_min_dist_kf_factor);
        std::cout << "  => min_dist_kf / avg_min_dist_kf : " << min_dist_kf << " / " << avg_min_dist_kf << std::endl;
    }

    bool condition3 = false;
    int abs_diff_kf;
    if (!unordered) {
        abs_diff_kf = (neighborest_id != -1) ? std::fabs((int)image_id - (int)neighborest_id) : 0;
        condition3 = (abs_diff_kf >= options.abs_diff_kf);
        std::cout << "  => image_id " << image_id << ", neighbor_id = " << neighborest_id << std::endl;
    }

    bool condition_has_gps = false;
    bool condition_has_prior_pose = options.prior_force_keyframe && options.have_prior_pose && options.prior_rotations.count(image_id) && options.prior_translations.count(image_id);
    if (condition_has_gps || condition_has_prior_pose || condition0 || ((condition1 || condition2 || condition3 || condition_inlier || condition_has_large_time_diff) && exceed_min_step)) {
    //if (condition_has_gps || condition0 || ((condition1 || condition2 || condition3 || condition_inlier) && exceed_min_step)) {
        if (!unordered) {
            if ((abs_diff_kf < options.abs_diff_kf && options.overlap_image_ids.count(image_id) == 0) ||
                init_image_id1 == image_id || init_image_id2 == image_id) {
                if (neighborest_id != -1) {
                    acc_min_dist_kf_ += min_dist_kf;
                }
                seq_num_kf_++;
            }
        }

        reconstruction_->RegisterImage(image_id);
        RegisterImageEvent(image_id);
        image.SetKeyFrame(true);

        last_keyframe_idx = image_id;
        if (options.offline_slam) {
            offline_slam_last_keyframe_id_ = image_id;
        }

        for (size_t i = 0; i < inlier_mask.size(); ++i) {
            if (inlier_mask[i]) {
                const point2D_t point2D_idx = tri_corrs[i].first;
                const Point2D& point2D = image.Point2D(point2D_idx);
                if (!point2D.HasMapPoint()) {
                    const mappoint_t mappoint_id = tri_corrs[i].second;
                    const TrackElement track_el(image_id, point2D_idx);
                    reconstruction_->AddObservation(mappoint_id, track_el);
                    triangulator_->AddModifiedMapPoint(mappoint_id);
                }
            }
        }
        inlier_mask.clear();
        tri_corrs.clear();
        keyframe_ids_.push_back(image_id);
        std::cout << "  => AddKeyFrame " << image_id << "(" << image.Name() << ")" << std::endl;
        return true;
    }

    return false;
}

bool IncrementalMapper::AddKeyFrameUpdate(const Options& options, const image_t image_id,
                                          std::vector<std::pair<point2D_t, mappoint_t>>& tri_corrs,
                                          std::vector<char>& inlier_mask, bool force) {
    const double max_squared_disp = options.mean_max_disparity_kf * options.mean_max_disparity_kf;

    Image& image = reconstruction_->Image(image_id);
    Eigen::Vector3d C1 = image.ProjectionCenter();

    image.SetPoseFlag(true);

    bool exceed_min_step = true;

    bool condition0 = force;

    int neighborest_id = -1;
    float min_dist_kf = std::numeric_limits<float>::max();

    const std::unordered_set<image_t>& neighbor_ids =
        scene_graph_container_->CorrespondenceGraph()->ImageNeighbor(image_id);
    for (const auto& neighbor_id : neighbor_ids) {
        if (reconstruction_->ExistsImage(neighbor_id) && reconstruction_->IsImageRegistered(neighbor_id)) {
            const Image& image_neighbor = reconstruction_->Image(neighbor_id);

            if (image_neighbor.LabelId() > 0) {
                Eigen::Vector3d C2 = image_neighbor.ProjectionCenter();
                float dist_kf = (C1 - C2).norm();
                if (dist_kf < min_dist_kf) {
                    min_dist_kf = dist_kf;
                    neighborest_id = neighbor_id;
                }
            }
        }
    }

    for (int step = -(options.min_keyframe_step - 1); step <= options.min_keyframe_step - 1; step++) {
        if (step < 0 && image_id <= -step) {
            continue;
        }
        image_t neighbor_id = image_id + step;

        if (reconstruction_->ExistsImage(neighbor_id) && reconstruction_->IsImageRegistered(neighbor_id)) {
            exceed_min_step = false;
            std::cout << "  => skip adding keyframe according to the min step" << std::endl;
            break;
        }
    }

    point2D_t num_visible_map_point = image.NumVisibleMapPoints();
    bool condition1 = (num_visible_map_point < options.min_visible_map_point_kf);
    std::cout << "  => num_visible_map_point = " << num_visible_map_point << std::endl;

    int inlier_count = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_count++;
        }
    }
    bool condition_inlier = inlier_count < options.min_pose_inlier_kf;
    std::cout << " => num inlier = " << inlier_count << std::endl;

    bool condition2 = false;

    float avg_min_dist_kf = 0.0f;
    if (seq_num_kf_ > 1) {
        avg_min_dist_kf = acc_min_dist_kf_ / (seq_num_kf_ - 1);
    }
    condition2 = (min_dist_kf >= avg_min_dist_kf * options.avg_min_dist_kf_factor);
    std::cout << "  => min_dist_kf / avg_min_dist_kf : " << min_dist_kf << " / " << avg_min_dist_kf << std::endl;

    bool condition3 = false;
    int abs_diff_kf;

    abs_diff_kf = (neighborest_id != -1) ? std::fabs((int)image_id - (int)neighborest_id) : 0;
    condition3 = (abs_diff_kf >= options.abs_diff_kf);
    std::cout << "  => image_id " << image_id << ", neighbor_id = " << neighborest_id << std::endl;

    if (condition0 || ((condition1 || condition2 || condition3 || condition_inlier) && exceed_min_step)) {
        if ((abs_diff_kf < options.abs_diff_kf && options.overlap_image_ids.count(image_id) == 0) ||
            init_image_id1 == image_id || init_image_id2 == image_id) {
            if (neighborest_id != -1) {
                acc_min_dist_kf_ += min_dist_kf;
            }
            seq_num_kf_++;
        }

        reconstruction_->RegisterImage(image_id);
        RegisterImageEvent(image_id);
        image.SetKeyFrame(true);

        last_keyframe_idx = image_id;
        if (options.offline_slam) {
            offline_slam_last_keyframe_id_ = image_id;
        }

        for (size_t i = 0; i < inlier_mask.size(); ++i) {
            if (inlier_mask[i]) {
                const point2D_t point2D_idx = tri_corrs[i].first;
                const Point2D& point2D = image.Point2D(point2D_idx);
                if (!point2D.HasMapPoint()) {
                    const mappoint_t mappoint_id = tri_corrs[i].second;
                    const TrackElement track_el(image_id, point2D_idx);
                    reconstruction_->AddObservation(mappoint_id, track_el);
                    triangulator_->AddModifiedMapPoint(mappoint_id);
                }
            }
        }
        inlier_mask.clear();
        tri_corrs.clear();
        std::cout << "  => AddKeyFrame " << image_id << "(" << image.Name() << ")" << std::endl;
        return true;
    }

    return false;
}

bool IncrementalMapper::RegisterNonKeyFrame(const Options& options, const image_t image_id) {
    CHECK_NOTNULL(reconstruction_.get());

    CHECK(options.Check());

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    class Image& image = reconstruction_->Image(image_id);
    class Camera& camera = reconstruction_->Camera(image.CameraId());

    if (!correspondence_graph->ExistsImage(image_id)) {
        std::cout << StringPrintf("  => Don't find %s in CorrespondenceGraph", image.Name().c_str()) << std::endl;
        return false;
    }

    CHECK(!image.IsRegistered());

    num_reg_trials_[image_id] += 1;

    // Search for 2D-3D correspondence.
    const int kCorrTransitivity = 1;

    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;
    std::vector<uint64_t> mappoints_create_time;
    std::vector<std::pair<point2D_t, mappoint_t>> tri_corrs;
    std::vector<char> inlier_mask;

    std::cout << StringPrintf("  => Image sees %d / %d points", image.NumVisibleMapPoints(), image.NumObservations())
              << std::endl;

    if (image.NumVisibleMapPoints() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        std::vector<CorrespondenceGraph::Correspondence> corrs;
            // correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);
        correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<mappoint_t> mappoint_ids;

        for (const auto corr : corrs) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint()) {
                continue;
            }

            // Avoid duplicate correspondence.
            if (mappoint_ids.count(corr_point2D.MapPointId()) > 0) {
                continue;
            }

            const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());

            // Avoid correspondences to images with bogus camera parameters.
            if (camera.ModelName().compare("SPHERICAL") != 0 &&
                corr_camera.HasBogusParams(options.min_focal_length_ratio, options.max_focal_length_ratio,
                                           options.max_extra_param)) {
                continue;
            }

            const MapPoint& mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());

            tri_corrs.emplace_back(point2D_idx, corr_point2D.MapPointId());
            mappoint_ids.insert(corr_point2D.MapPointId());
            tri_points2D.push_back(point2D.XY());
            tri_points3D.push_back(mappoint.XYZ());
            mappoints_create_time.push_back(mappoint.CreateTime());
        }
    }

    std::cout << "  => tri_points2D.size() = " << tri_points2D.size();
    if (tri_points2D.size() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        std::cout << ", lower than " << options.abs_pose_min_num_inliers << std::endl;
        return false;
    }
    std::cout << std::endl;

    // 2D-3D estimation.
    AbsolutePoseEstimationOptions abs_pose_options;
    abs_pose_options.num_threads = options.num_threads;
    abs_pose_options.num_focal_length_samples = 10;
    abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
    abs_pose_options.ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    abs_pose_options.ransac_options.min_num_trials = 30;
    abs_pose_options.ransac_options.confidence = 0.9999;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    if (!options.single_camera) {
        abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = false;
    }

    size_t num_inliers;
    if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D, &image.Qvec(), &image.Tvec(), &camera,
                              &num_inliers, &inlier_mask)) {
        std::cout << "EstimateAbsolutePose failed!" << std::endl;
        return false;
    }

    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        std::cout << "num_inliers " << num_inliers << ", lower than " << options.abs_pose_min_num_inliers << std::endl;
        return false;
    }

    // Pose refinement
    if (reconstruction_->depth_enabled && options.rgbd_pose_refine_depth_weight > 0) {
        for (int i = 0; i < tri_corrs.size(); i++) {
            abs_pose_refinement_options.point_depths.emplace_back(
                image.Point2D(tri_corrs[i].first).Depth());
            abs_pose_refinement_options.point_depths_weights.emplace_back(
                options.rgbd_pose_refine_depth_weight * image.Point2D(tri_corrs[i].first).DepthWeight());
        }
    }
    if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask, tri_points2D, tri_points3D, 
                            reconstruction_->NumRegisterImages(), mappoints_create_time, 
                            std::vector<double>(), &image.Qvec(), &image.Tvec(), &camera)) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }

    image.SetPoseFlag(true);

    // Update data
    refined_cameras_.insert(image.CameraId());

    reconstruction_->RegisterImage(image_id);
    RegisterImageEvent(image_id);

    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            const point2D_t point2D_idx = tri_corrs[i].first;
            const Point2D& point2D = image.Point2D(point2D_idx);
            if (!point2D.HasMapPoint()) {
                const mappoint_t mappoint_id = tri_corrs[i].second;
                const TrackElement track_el(image_id, point2D_idx);
                reconstruction_->AddObservation(mappoint_id, track_el);
                triangulator_->AddModifiedMapPoint(mappoint_id);
            }
        }
    }
    return true;
}

bool IncrementalMapper::RegisterNonKeyFrameRig(const Options& options, const image_t image_id) {
    CHECK_NOTNULL(reconstruction_.get());

    CHECK(options.Check());

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    class Image& image = reconstruction_->Image(image_id);
    class Camera& camera = reconstruction_->Camera(image.CameraId());

    if (!correspondence_graph->ExistsImage(image_id)) {
        std::cout << StringPrintf("  => Don't find %s in CorrespondenceGraph", image.Name().c_str()) << std::endl;
        return false;
    }

    CHECK(!image.IsRegistered());
    num_reg_trials_[image_id] += 1;

    std::vector<uint32_t> local_camera_indices = image.LocalImageIndices();
    std::cout << " obtain local camera params" << std::endl;

    std::vector<Eigen::Matrix3x4d> local_transforms;
    local_transforms.resize(camera.NumLocalCameras());

    for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        Eigen::Matrix3x4d local_transform = ComposeProjectionMatrix(local_qvec, local_tvec);
        local_transforms[i] = local_transform;
    }

    if (image.NumVisibleMapPoints() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    // Search for 2D-3D correspondence.
    const int kCorrTransitivity = 1;

    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;
    std::vector<uint64_t> mappoints_create_time;
    std::vector<std::pair<point2D_t, mappoint_t>> tri_corrs;
    std::vector<char> inlier_mask;

    std::vector<GP3PEstimator::X_t> points2D_normalized;

    std::vector<int> tri_camera_indices;

    bool has_bogus_params = false;
    std::unordered_map<camera_t, std::string> camera_param_map;

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        std::vector<CorrespondenceGraph::Correspondence> corrs;
            // correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);
        correspondence_graph->FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<mappoint_t> mappoint_ids;
        uint32_t local_camera_id = local_camera_indices[point2D_idx];

        for (const auto corr : corrs) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasMapPoint()) {
                continue;
            }

            // Avoid duplicate correspondence.
            if (mappoint_ids.count(corr_point2D.MapPointId()) > 0) {
                continue;
            }

            const MapPoint& mappoint = reconstruction_->MapPoint(corr_point2D.MapPointId());

            tri_corrs.emplace_back(point2D_idx, corr_point2D.MapPointId());
            mappoint_ids.insert(corr_point2D.MapPointId());

            tri_points3D.push_back(mappoint.XYZ());
            tri_points2D.push_back(point2D.XY());
            tri_camera_indices.push_back(local_camera_id);

            Eigen::Vector2d point2D_normalized = camera.LocalImageToWorld(local_camera_id, point2D.XY());

            points2D_normalized.emplace_back();
            points2D_normalized.back().rel_tform = local_transforms[local_camera_id];
            points2D_normalized.back().xy = point2D_normalized;
            mappoints_create_time.push_back(mappoint.CreateTime());
        }
    }

    if (tri_points3D.size() < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        std::cout << "tri_points3D.size() = " << tri_points3D.size() << ", lower than "
                  << options.abs_pose_min_num_inliers << std::endl;
        return false;
    }
    std::cout << "tri_points3D size: " << tri_points3D.size() << std::endl;

    RANSACOptions ransac_options;
    ransac_options.max_error = 1 - cos(0.0299977504);
    ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    ransac_options.min_num_trials = 30;
    ransac_options.confidence = 0.9999;

    RANSAC<GP3PEstimator> ransac(ransac_options);
    const auto report = ransac.Estimate(points2D_normalized, tri_points3D);

    inlier_mask = report.inlier_mask;
    size_t num_inliers = report.support.num_inliers;

    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers) ||
        (static_cast<double>(num_inliers) <
         static_cast<double>(tri_points3D.size()) * options.abs_pose_min_inlier_ratio)) {
        std::cout << "num_inliers " << num_inliers << ", lower than " << options.abs_pose_min_num_inliers
                  << " or inlier ration lower than " << options.abs_pose_min_inlier_ratio << std::endl;
        return false;
    }
    std::cout << "num inliers: " << num_inliers << std::endl;
    // re-estimate with inliers

    Eigen::Matrix3d r = report.model.block<3, 3>(0, 0);
    Eigen::Vector3d t = report.model.col(3);

    image.Qvec() = RotationMatrixToQuaternion(r);
    image.Tvec() = t;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    // Pose refinement
    if (!RefineAbsolutePoseRig(abs_pose_refinement_options, inlier_mask, tri_points2D, tri_points3D,
                               reconstruction_->NumRegisterImages(), mappoints_create_time,
                               std::vector<double>(), tri_camera_indices, &image.Qvec(), &image.Tvec(), &camera)) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }
    image.SetPoseFlag(true);


    // Update data
    refined_cameras_.insert(image.CameraId());

    reconstruction_->RegisterImage(image_id);
    RegisterImageEvent(image_id);

    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            const point2D_t point2D_idx = tri_corrs[i].first;
            const Point2D& point2D = image.Point2D(point2D_idx);
            if (!point2D.HasMapPoint()) {
                const mappoint_t mappoint_id = tri_corrs[i].second;
                const TrackElement track_el(image_id, point2D_idx);
                reconstruction_->AddObservation(mappoint_id, track_el);
                triangulator_->AddModifiedMapPoint(mappoint_id);
            }
        }
    }
    return true;
}

bool IncrementalMapper::RegisterNonKeyFrameLidar(const Options& options, const sweep_t sweep_id) {
    auto & lidar_sweep = reconstruction_->LidarSweep(sweep_id);
    long long lidar_timestamp = lidar_sweep.timestamp_;

    image_t image_id;
    const std::pair<image_t, long long> sweep_timestamp_t({sweep_id, lidar_timestamp});
    auto next_image = std::lower_bound(image_timestamps_.begin(), image_timestamps_.end(), sweep_timestamp_t, 
        [&](const std::pair<image_t, long long> & a, const std::pair<image_t, long long> & b) {
            return a.second < b.second;
        });

    int forward = next_image - image_timestamps_.begin();
    if (forward >= image_timestamps_.size() || forward <= 0) {
        return false;
    }

    std::cout << StringPrintf("lidar#%d: %s", sweep_id, JoinPaths(options.lidar_path, lidar_sweep.Name()).c_str()) << std::endl;

    auto pc = ReadPCD(JoinPaths(options.lidar_path, lidar_sweep.Name()));
    if (pc.info.height == 1 ){
        RebuildLivoxMid360(pc, 4);
    }
    lidar_sweep.Setup(pc, lidar_sweep.timestamp_);
    // lidar_sweep.Setup(pc, reconstruction_->NumRegisterImages());
    // lidar_sweep.FilterPointCloud(0.3);

    std::vector<std::pair<image_t, long long> >::iterator prev_image = image_timestamps_.begin();
    std::advance(prev_image, forward - 1);
    
    long long prev_dtime = lidar_timestamp - prev_image->second;
    long long next_dtime = next_image->second - lidar_timestamp;
    if (prev_dtime > next_dtime) {
        image_id = next_image->first;
    } else {
        image_id = prev_image->first;
    }

    class Image & image = reconstruction_->Image(image_id);

    long long delta_time = std::abs(lidar_timestamp - image.timestamp_);
    std::cout << "delta time: " << delta_time / 1e6 << " ms" << std::endl;
    if (delta_time / 1e6 > 100) { // delta time large than 100ms
        std::cout << StringPrintf("Image: %s and Lidar: %s has extreme time difference\n", 
                     image.Name().c_str(), lidar_sweep.Name().c_str());
        return false;
    }
    if (!image.IsRegistered()) {
        return false;
    }

    Eigen::Matrix4d world2cam = Eigen::Matrix4d::Identity();
    world2cam.topRows(3) = image.ProjectionMatrix();

    Eigen::Matrix4d lidar2cam = Eigen::Matrix4d::Identity();
    lidar2cam.topRows(3) = options.lidar_to_cam_matrix;

    Eigen::Matrix3x4d world2lidar = (lidar2cam.inverse() * world2cam).topRows(3);
    Eigen::Vector4d qvec = RotationMatrixToQuaternion(world2lidar.block<3, 3>(0, 0));
    Eigen::Vector3d tvec = world2lidar.block<3, 1>(0, 3);
    lidar_sweep.SetQvec(qvec);
    lidar_sweep.SetTvec(tvec);

    LidarAbsolutePoseRefinementOptions absolute_pose_options;
    absolute_pose_options.gradient_tolerance = 1e-6;
    absolute_pose_options.num_iteration_pose_estimation = 3;

    // Refine lidar scan pose.
    LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
    // LidarPointCloud ref_corners = lidar_sweep.GetCornerPointsLessSharp();
    LidarPointCloud ref_corners = lidar_sweep.GetCornerPointsSharp();

    std::vector<lidar::OctoTree::Point> points;
    points.reserve(ref_less_surfs.points.size() + ref_less_surfs.points.size());
    for (auto & point : ref_less_surfs.points) {
        lidar::OctoTree::Point X;
        X.x = point.x;
        X.y = point.y;
        X.z = point.z;
        X.intensity = point.intensity;
        X.lifetime = point.lifetime;
        // X.lifetime = lidar_sweep.timestamp_;
        X.type = 0;
        points.push_back(X);
    }

    for (auto & point : ref_corners.points) {
        lidar::OctoTree::Point X;
        X.x = point.x;
        X.y = point.y;
        X.z = point.z;
        X.intensity = point.intensity;
        X.lifetime = point.lifetime;
        // X.lifetime = lidar_sweep.timestamp_;
        X.type = 1;
        points.push_back(X);
    }

    uint32_t m_num_visible_point = 0;
    const auto register_image_ids = reconstruction_->RegisterImageIds();
    for (auto register_image_id : register_image_ids) {
        m_num_visible_point += reconstruction_->Image(register_image_id).NumVisibleMapPoints();
    }
    m_num_visible_point / register_image_ids.size();

    auto & voxel_map = reconstruction_->VoxelMap();
    if (!RefineAbsolutePose(absolute_pose_options, reconstruction_.get(), image_id, 
                            points, voxel_map.get(), &lidar_sweep.Qvec(), &lidar_sweep.Tvec())) {
        std::cout << "RefineAbsolutePose failed!" << std::endl;
        return false;
    }
    reconstruction_->RegisterLidarSweep(sweep_id);

    return true;
}

bool IncrementalMapper::ClosureDetection(const Options& options, const image_t image_id1, const image_t image_id2,
                                         double baseline_distance) {
    const auto& correspondence_graph = scene_graph_container_->CorrespondenceGraph();
    const auto& corrs = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);

    class Image& image1 = reconstruction_->Image(image_id1);
    class Image& image2 = reconstruction_->Image(image_id2);

    if(image1.LabelId()!=image2.LabelId()){
        return false;
    }

    Eigen::Vector3d current_baseline = image1.ProjectionCenter() - image2.ProjectionCenter();
    if (current_baseline.norm() < options.loop_distance_factor_wrt_averge_baseline *
                                      abs(static_cast<double>(image_id1) - static_cast<double>(image_id2)) *
                                      baseline_distance) {
        return false;
    }

    image_t id_difference = image_id1 > image_id2 ? (image_id1 - image_id2) : (image_id2 - image_id1);
    if(id_difference > options.max_id_difference_for_loop){
        return false;
    }


    size_t consistent_corr_count = 0;

    for (const auto& corr : corrs) {
        const class Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
        const class Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);

        if (point2D1.HasMapPoint() && point2D2.HasMapPoint() && (point2D1.MapPointId() == point2D2.MapPointId())) {
            consistent_corr_count++;
        }
    }

    double inconsistent_corr_ratio = 1.0 - static_cast<double>(consistent_corr_count)/ static_cast<double>(corrs.size());
    if(inconsistent_corr_ratio >= options.min_inconsistent_corr_ratio_for_loop){
        std::cout<<"consistent_corr_count: "<<consistent_corr_count<<std::endl;
        return true;
    }
    else{
        return false;
    }
}

bool IncrementalMapper::IsolatedImage(const Options& options, const image_t image_id, double baseline_distance){
    class Image& image = reconstruction_->Image(image_id);
    if (!image.IsRegistered()) {
        return false;
    }

    int neighbor_scope = options.max_id_difference_for_loop;
    bool b_isolated = true;
    for (int scope = -neighbor_scope; scope <= neighbor_scope; scope++) {
        if (scope == 0) {
            continue;
        }

        if (scope + static_cast<int>(image_id) > 0 &&
            reconstruction_->ExistsImage(scope + static_cast<int>(image_id))) {
            class Image& image_neighbor = reconstruction_->Image(scope + static_cast<int>(image_id));

            if (image_neighbor.IsRegistered()) {
                Eigen::Vector3d current_baseline = image.ProjectionCenter() - image_neighbor.ProjectionCenter();
                if (current_baseline.norm() < options.loop_distance_factor_wrt_averge_baseline *
                                                  abs(static_cast<double>(scope) * baseline_distance)) {
                    b_isolated = false;
                    break;
                }
            }
        }
    }
    return b_isolated;
}

bool IncrementalMapper::AdjustCameraByLoopClosure(const Options& options) {
    const auto& correspondence_graph = scene_graph_container_->CorrespondenceGraph();
    const auto& image_pairs = reconstruction_->ImagePairs();
    reconstruction_->ComputeBaselineDistance();
    double baseline_distance = reconstruction_->baseline_distance;
    std::cout<<"Average baseline distance :"<<baseline_distance<<std::endl;
    std::vector<Edge> m_edges, m_loop_edges;
    std::unordered_map<image_t, int> normal_edge_count_per_image;

    for (const auto& image_pair : image_pairs) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        if (image_id1 < image_id2) {
            std::swap(image_id1, image_id2);
        }

        class Image& image1 = reconstruction_->Image(image_id1);
        class Image& image2 = reconstruction_->Image(image_id2);

        class Camera& camera1 = reconstruction_->Camera(image1.CameraId());
        class Camera& camera2 = reconstruction_->Camera(image2.CameraId());
        
        if (!image1.IsRegistered() || !image2.IsRegistered()) {
            continue;
        }

        if(IsolatedImage(options,image_id1,baseline_distance) || IsolatedImage(options,image_id2,baseline_distance)){
            std::cout<<"Isolated image: "<<image1.Name()<<" or"<<image2.Name()<<std::endl;
            continue;
        }

        const auto& corrs = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);

        if (corrs.size() < options.min_loop_pose_inlier_num) {
            continue;
        }

        Eigen::Vector3d tvec1 = image1.Tvec();
        Eigen::Vector4d qvec1 = image1.Qvec();
        Eigen::Vector3d tvec2 = image2.Tvec();
        Eigen::Vector4d qvec2 = image2.Qvec();

        Eigen::Quaterniond q1(qvec1[0], qvec1[1], qvec1[2], qvec1[3]);
        Eigen::Quaterniond q2(qvec2[0], qvec2[1], qvec2[2], qvec2[3]);


        if (!ClosureDetection(options, image_id1, image_id2, baseline_distance)) {
            Eigen::Quaterniond relative_q = q1 * q2.conjugate();
            Eigen::Vector3d relative_t = tvec1 - relative_q * tvec2;

            const auto& corrs = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);
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

                if (id_difference < options.normal_edge_count_per_image / 2 &&
                    current_baseline.norm() < options.neighbor_distance_factor_wrt_averge_baseline *
                                                  static_cast<double>(id_difference) * baseline_distance &&
                    consistent_corr_count > 0) {
                    consistent_neighbor = true;
                }
            }

            if (consistent_corr_count < options.normal_edge_min_common_points && !consistent_neighbor) {
                continue;
            }

            Edge e;
            e.id_begin = image_id1;
            e.id_end = image_id2;
            e.num_corrs = consistent_corr_count;
            e.relative_pose.qvec = relative_q;
            e.relative_pose.tvec = relative_t;
            m_edges.push_back(e);

            continue;
        }
        
        std::cout<<"Found candidate loop images: "<<image1.Name()<<" "<<image2.Name()<<std::endl;


        std::vector<Eigen::Vector2d> points2D_2;
        std::vector<Eigen::Vector3d> points3D_1;
        for (const auto& corr : corrs) {
            const class Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
            const class Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);
            if (point2D1.HasMapPoint() && (!point2D2.HasMapPoint() || point2D2.MapPointId() != point2D1.MapPointId())) {
                mappoint_t mappoint_id1 = point2D1.MapPointId();
                points2D_2.push_back(point2D2.XY());
                points3D_1.push_back(reconstruction_->MapPoint(mappoint_id1).XYZ());
            }
        }
        std::cout<<"Match from 1 to 2 "<<points2D_2.size()<<std::endl;

        std::vector<Eigen::Vector2d> points2D_1;
        std::vector<Eigen::Vector3d> points3D_2;
        for (const auto& corr : corrs) {
            const class Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
            const class Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);
            if (point2D2.HasMapPoint() && (!point2D1.HasMapPoint() || point2D1.MapPointId() != point2D2.MapPointId())) {
                mappoint_t mappoint_id2 = point2D2.MapPointId();
                points2D_1.push_back(point2D1.XY());
                points3D_2.push_back(reconstruction_->MapPoint(mappoint_id2).XYZ());
            }
        }
        std::cout<<"Match from 2 to 1 "<<points2D_1.size()<<std::endl;

        bool pose_2_from_1;
        std::vector<Eigen::Vector2d> points2D;
        std::vector<Eigen::Vector3d> points3D;
        Camera* camera; 
        
        if(points2D_2.size() > points2D_1.size()){
            points2D = points2D_2;
            points3D = points3D_1;
            pose_2_from_1 = true;
            camera = &camera2;
        }
        else{
            points2D = points2D_1;
            points3D = points3D_2;
            pose_2_from_1 = false;
            camera = &camera1;
        }
        std::cout << "2D-3D correspondence size: " << points2D.size()<<std::endl;
        if (points2D.size() < static_cast<size_t>(options.min_loop_pose_inlier_num)) {
            std::cout << "Lower than "<< options.min_loop_pose_inlier_num << ", not a valid loop"<<std::endl;
            continue;
        }


        AbsolutePoseEstimationOptions abs_pose_options;
        abs_pose_options.num_threads = options.num_threads;

        abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
        abs_pose_options.ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
        abs_pose_options.ransac_options.min_num_trials = 30;
        abs_pose_options.ransac_options.confidence = 0.9999;
        

        size_t num_inliers = 0;
        std::vector<char> inlier_mask;
        Eigen::Vector4d loop_qvec;
        Eigen::Vector3d loop_tvec;

        if (!EstimateAbsolutePose(abs_pose_options, points2D, points3D, &loop_qvec, &loop_tvec, camera, &num_inliers,
                                    &inlier_mask)) {
            std::cout << "EstimateAbsolutePose failed!, not a valid loop" << std::endl;
            continue;
        }

        if (num_inliers < static_cast<size_t>(options.min_loop_pose_inlier_num)) {
            std::cout << "num_inliers " << num_inliers << ", lower than " << options.min_loop_pose_inlier_num
                        << ", not a valid loop" << std::endl;
            continue;
        }

        AbsolutePoseRefinementOptions abs_pose_refinement_options;
        abs_pose_refinement_options.refine_focal_length = false;
        abs_pose_refinement_options.refine_extra_params = false;

        if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask, points2D, points3D, 
                                reconstruction_->NumRegisterImages(), std::vector<uint64_t>(),
                                std::vector<double>(), &loop_qvec, &loop_tvec, camera)) {
            std::cout << "RefineAbsolutePose failed!, not a valid loop" << std::endl;
            continue;
        }

        Eigen::Quaterniond loop_q(loop_qvec[0], loop_qvec[1], loop_qvec[2], loop_qvec[3]);
        Eigen::Vector3d loop_t = loop_tvec;

        Eigen::Quaterniond relative_q;
        Eigen::Vector3d relative_t;
        
        if(pose_2_from_1){
            relative_q = q1 * loop_q.conjugate();
            relative_t = tvec1 - relative_q * loop_t;
        }
        else{
            relative_q = loop_q * q2.conjugate();
            relative_t = loop_t - relative_q * tvec2;
        }

        Edge e;
        e.id_begin = image_id1;
        e.id_end = image_id2;
        e.num_corrs = num_inliers;
        e.relative_pose.qvec = relative_q;
        e.relative_pose.tvec = relative_t;
        m_loop_edges.push_back(e);

        std::cout << "PoseGraph => AddConstraint[" << image1.Name() << ", " << image2.Name() << "]" << std::endl;
    }


    if(m_loop_edges.size() == 0){
        return false;
    }

    if (options.debug_info) {
        std::string recon_path =
            StringPrintf("%s/before_loop_%d/", workspace_path_.c_str(), reconstruction_->NumRegisterImages());
        boost::filesystem::create_directories(recon_path);
        reconstruction_->WriteReconstruction(recon_path, true);
    }

    if (options.optimize_sim3) {
        auto image_ids = reconstruction_->RegisterImageIds();
        for (auto image_id : image_ids) {
            auto& image = reconstruction_->Image(image_id);
            // Set SIM3 pose
            Eigen::Quaterniond quat(image.Qvec()[0], image.Qvec()[1], image.Qvec()[2], image.Qvec()[3]);
            ConvertSIM3tosim3(image.Sim3pose(), quat, image.Tvec(), 1);
        }
    }

    std::sort(m_edges.begin(), m_edges.end(),
              [&](const Edge& e1, const Edge& e2) { return e1.num_corrs > e2.num_corrs; });

    std::sort(m_loop_edges.begin(), m_loop_edges.end(),
              [&](const Edge& e1, const Edge& e2) { return e1.num_corrs > e2.num_corrs; });

    PoseGraphOptimizer::Options pose_graph_options;
    pose_graph_options.max_num_iterations = options.loop_closure_max_iter_num;
    if(options.optimize_sim3){
        pose_graph_options.optimization_method  = PoseGraphOptimizer::OPTIMIZATION_METHOD::SIM3;
    }
    else{
         pose_graph_options.optimization_method  = PoseGraphOptimizer::OPTIMIZATION_METHOD::SE3;
    }
    
    
    PoseGraphOptimizer* optimizer = new PoseGraphOptimizer(pose_graph_options, reconstruction_.get());
    std::cout<<"loop_weight is: "<<options.loop_weight<<std::endl;

    int loop_edge_count = 0;
    for (const auto& edge : m_loop_edges) {
        
        if (options.optimize_sim3) {
            Eigen::Vector7d sim3_loop;
            ConvertSIM3tosim3(sim3_loop, edge.relative_pose.qvec, edge.relative_pose.tvec, 1);
            optimizer->AddConstraint(edge.id_begin, edge.id_end, sim3_loop, options.loop_weight);
        }
        else{
            optimizer->AddConstraint(edge.id_begin, edge.id_end, edge.relative_pose.qvec, edge.relative_pose.tvec,
                                    options.loop_weight);
        }
        
        loop_edge_count++;
        if (loop_edge_count >= options.max_loop_edge_count) {
            break;
        }
    }
    std::unordered_map<image_t, int> edge_one_image;
    for (const auto& edge : m_edges) {
        if (edge_one_image.count(edge.id_begin) < options.normal_edge_count_per_image ||
            edge_one_image.count(edge.id_end) < options.normal_edge_count_per_image) {

            if(options.optimize_sim3){
                Eigen::Vector7d sim3_normal;
                ConvertSIM3tosim3(sim3_normal, edge.relative_pose.qvec, edge.relative_pose.tvec, 1);
                optimizer->AddConstraint(edge.id_begin, edge.id_end,sim3_normal);
            }
            else{
                optimizer->AddConstraint(edge.id_begin, edge.id_end, edge.relative_pose.qvec, edge.relative_pose.tvec);
            }
            edge_one_image[edge.id_begin]++;
            edge_one_image[edge.id_end]++;
        }
    }

    // set first frame as constant
    optimizer->SetParameterConstant(m_edges.begin()->id_begin);

    optimizer->Solve();
    if (options.optimize_sim3) {
        auto image_ids = reconstruction_->RegisterImageIds();
        for (auto image_id : image_ids) {
            double scale;
            Eigen::Quaterniond qvec;
            Eigen::Vector3d tvec;
            Convertsim3toSIM3(reconstruction_->Image(image_id).Sim3pose(), qvec, tvec, scale);

            // Conver the t to t/s
            tvec = 1. / scale * tvec;

            // Update Image pose using sim3 pose
            reconstruction_->Image(image_id).SetTvec(tvec);
            reconstruction_->Image(image_id).SetQvec(RotationMatrixToQuaternion(qvec.normalized().toRotationMatrix()));
        }
    }
    return true;

}

double IncrementalMapper::ComputeDisparityBetweenImages(const image_t image_id1, const image_t image_id2) {
    // Image ID starts from 0.
    if (!reconstruction_->ExistsImage(image_id1) || !reconstruction_->ExistsImage(image_id2)) {
        return 0.0;
    }

    double mean_squared_disp = 0.0;

    const class Image& image1 = reconstruction_->Image(image_id1);
    const class Image& image2 = reconstruction_->Image(image_id2);

    const auto& correspondence_graph = scene_graph_container_->CorrespondenceGraph();
    const FeatureMatches& matches = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);
    for (const auto& match : matches) {
        const class Point2D& point2D1 = image1.Point2D(match.point2D_idx1);
        const class Point2D& point2D2 = image2.Point2D(match.point2D_idx2);
        mean_squared_disp += (point2D1.XY() - point2D2.XY()).squaredNorm();
    }
    mean_squared_disp /= matches.size();

    return mean_squared_disp;
}

size_t IncrementalMapper::TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                                           const image_t image_id) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t IncrementalMapper::TriangulateMappoint(const IncrementalTriangulator::Options& tri_options,
                                              const mappoint_t mappoint_id) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->Recreate(tri_options, mappoint_id);
}

size_t IncrementalMapper::Retriangulate(const IncrementalTriangulator::Options& tri_options,
   std::unordered_set<image_t>* image_set) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->Retriangulate(tri_options,image_set);
}

size_t IncrementalMapper::Retriangulate(const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper::RetriangulateAllTracks(const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->RetriangulateAllTracks(tri_options);
}

size_t IncrementalMapper::CompleteTracks(const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper::CompleteTracks(const IncrementalTriangulator::Options& tri_options, const std::unordered_set<mappoint_t>& mappoint_ids) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->CompleteTracks(tri_options, mappoint_ids);
}

size_t IncrementalMapper::MergeTracks(const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->MergeAllTracks(tri_options);
}

size_t IncrementalMapper::MergeTracks(const IncrementalTriangulator::Options& tri_options, const std::unordered_set<mappoint_t>& mappoint_ids) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->MergeTracks(tri_options, mappoint_ids);
}

IncrementalMapper::LocalBundleAdjustmentReport IncrementalMapper::AdjustLocalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const image_t image_id,
    const std::unordered_set<mappoint_t>& mappoint_ids,
    const sweep_t next_sweep_id) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());

    const size_t num_reg_images = reconstruction_->NumRegisterImages();

    LocalBundleAdjustmentReport report;

    // Find images that have most Map Points with given image in common.
    const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);

    std::unordered_set<sweep_t> modified_sweep_ids;
    std::unordered_map<sweep_t, Eigen::Matrix3x4d> sweep_old_poses;
    // Do the bundle adjustment only if there is any connected images.
    if (local_bundle.size() > 0) {
        BundleAdjustmentConfig ba_config;
        ba_config.AddImage(image_id);
        const Image& image = reconstruction_->Image(image_id);
        const Camera& camera = reconstruction_->Camera(image.CameraId());
        if (camera.IsCameraConstant() || num_reg_images <= options.num_fix_camera_first) {
            ba_config.SetConstantCamera(image.CameraId());
        }
        
        // if (reconstruction_->ExistsLidarSweep(next_sweep_id)) {
        //     ba_config.AddSweep(next_sweep_id);
        //     ba_config.AddSweepImagePair(next_sweep_id, image_id);
        // }

        for (const image_t local_image_id : local_bundle) {
            ba_config.AddImage(local_image_id);
            const Image& local_image = reconstruction_->Image(local_image_id);
            const Camera& local_camera = reconstruction_->Camera(local_image.CameraId());
            if (local_camera.IsCameraConstant() || num_reg_images <= options.num_fix_camera_first) {
                ba_config.SetConstantCamera(local_image.CameraId());
            }

            if (image_to_lidar_map_.find(local_image_id) != image_to_lidar_map_.end()) {
                sweep_t sweep_id = image_to_lidar_map_[local_image_id];
                if (reconstruction_->ExistsLidarSweep(sweep_id) && reconstruction_->LidarSweep(sweep_id).IsRegistered()) {
                    ba_config.AddSweep(sweep_id);
                    // ba_config.SetConstantSweep(sweep_id);

                    modified_sweep_ids.insert(sweep_id);
                    sweep_old_poses[sweep_id] = reconstruction_->LidarSweep(sweep_id).InverseProjectionMatrix();

                    ba_config.AddSweepImagePair(sweep_id, local_image_id);
                }
            }
        }

        // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
        if (local_bundle.size() == 1) {
            ba_config.SetConstantPose(local_bundle[0]);

            int max_t_index = -1;
            double max_t = -1;
            const Image& second_image = reconstruction_->Image(image_id);
            for (int j = 0; j < 3; ++j) {
                if (abs(second_image.Tvec()[j]) > max_t) {
                    max_t = abs(second_image.Tvec()[j]);
                    max_t_index = j;
                }
            }
            ba_config.SetConstantTvec(image_id, {max_t_index});

        } else if (local_bundle.size() > 1) {
            ba_config.SetConstantPose(local_bundle[local_bundle.size() - 1]);

            int max_t_index = -1;
            double max_t = -1;
            const Image& second_image = reconstruction_->Image(local_bundle[local_bundle.size() - 2]);
            for (int j = 0; j < 3; ++j) {
                if (abs(second_image.Tvec()[j]) > max_t) {
                    max_t = abs(second_image.Tvec()[j]);
                    max_t_index = j;
                }
            }
            ba_config.SetConstantTvec(local_bundle[local_bundle.size() - 2], {max_t_index});

        }

        // Make sure, we refine all new and short-track Map Points, no matter if
        // they are fully contained in the local image set or not. Do not include
        // long track Map Points as they are usually already very stable and adding
        // to them to bundle adjustment and track merging/completion would slow
        // down the local bundle adjustment significantly.
        std::unordered_set<mappoint_t> variable_mappoint_ids;
        for (const mappoint_t mappoint_id : mappoint_ids) {
            const MapPoint& mappoint = reconstruction_->MapPoint(mappoint_id);
            const size_t kMaxTrackLength = 15;
            if (!mappoint.HasError() || mappoint.Track().Length() <= kMaxTrackLength) {
                ba_config.AddVariablePoint(mappoint_id);
                variable_mappoint_ids.insert(mappoint_id);
            }
        }

        // ba_config.AddSweepImagePair(image_to_lidar_map_[image_id], image_id);
        // for (auto slocal_image_id : local_bundle) {
        //     if (image_to_lidar_map_.find(local_image_id) != image_to_lidar_map_.end()) {
        //         ba_config.AddSweepImagePair(image_to_lidar_map_[local_image_id], local_image_id);
        //     }
        // }

        // if (options.map_update) {
        //     for (const mappoint_t mappoint_id : mappoint_ids) {
        //         fixed_mappoint_ids_.erase(mappoint_id);
        //     }
        // }

        // Adjust the local bundle.
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        bundle_adjuster.Solve(reconstruction_.get());

        report.num_adjusted_observations = bundle_adjuster.Summary().num_residuals / 2;

        // Merge refined tracks with other existing points.
        report.num_merged_observations = triangulator_->MergeTracks(tri_options, variable_mappoint_ids);
        // Complete tracks that may have failed to triangulate before refinement
        // of camera pose and calibration in bundle-adjustment. This may avoid
        // that some points are filtered and it helps for subsequent image
        // registrations.
        report.num_completed_observations = triangulator_->CompleteTracks(tri_options, variable_mappoint_ids);
        report.num_completed_observations += triangulator_->CompleteImage(tri_options, image_id);
    }

    // Filter both the modified images and all changed Map Points to make sure
    // there are no outlier points in the model. This results in duplicate work as
    // many of the provided Map Points may also be contained in the adjusted
    // images, but the filtering is not a bottleneck at this point.
    std::unordered_set<image_t> filter_image_ids;
    filter_image_ids.insert(image_id);
    filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
    report.num_filtered_observations = reconstruction_->FilterMapPointsInImages(
        options.filter_max_reproj_error, options.filter_min_tri_angle, filter_image_ids);
    report.num_filtered_observations +=
        reconstruction_->FilterMapPoints(options.filter_max_reproj_error, options.filter_min_tri_angle, mappoint_ids);

    auto voxelmap_begin_time = std::chrono::steady_clock::now();

    std::shared_ptr<VoxelMap> & voxel_map = reconstruction_->VoxelMap();

    uint64_t num_total_lidar_point = 0, changed_total_num_point = 0;
    std::vector<lidar::OctoTree::Point> old_lidar_points, new_lidar_points;
    for (const sweep_t sweep_id : modified_sweep_ids) {
        const class LidarSweep & lidar_sweep = reconstruction_->LidarSweep(sweep_id);
        const LidarPointCloud & ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
        const LidarPointCloud & ref_less_corners = lidar_sweep.GetCornerPointsLessSharp();
        const size_t num_surf = ref_less_surfs.points.size();
        const size_t num_corner = ref_less_corners.points.size();
        num_total_lidar_point += num_surf + num_corner;
    }
    old_lidar_points.reserve(num_total_lidar_point);
    new_lidar_points.reserve(num_total_lidar_point);

    for (const sweep_t sweep_id : modified_sweep_ids) {
        const class LidarSweep & lidar_sweep = reconstruction_->LidarSweep(sweep_id);
        Eigen::Matrix3x4d old_inv_projmat = sweep_old_poses[sweep_id];
        Eigen::Matrix3x4d new_inv_projmat = reconstruction_->LidarSweep(sweep_id).InverseProjectionMatrix();
        LidarPointCloud pc;
        LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
        LidarPointCloud ref_less_corners = lidar_sweep.GetCornerPointsLessSharp();
        const size_t num_surf = ref_less_surfs.points.size();
        pc = std::move(ref_less_surfs);
        pc += std::move(ref_less_corners);

        size_t changed_num_point = 0;
        for (size_t i = 0; i < pc.points.size(); ++i) {
            Eigen::Vector4d hpoint(pc.points[i].x, pc.points[i].y, pc.points[i].z, 1.0);
            Eigen::Vector3d newX = new_inv_projmat * hpoint;
            lidar::OctoTree::Point new_point;
            new_point.x = newX[0];
            new_point.y = newX[1];
            new_point.z = newX[2];
            new_point.lifetime = pc.points[i].lifetime;
            // new_point.lifetime = lidar_sweep.timestamp_;
            lidar::OctoTree * new_loc = nullptr;
            new_loc = voxel_map->LocateRoot(new_point);

            Eigen::Vector3d oldX = old_inv_projmat * hpoint;
            lidar::OctoTree::Point old_point;
            old_point.x = oldX[0];
            old_point.y = oldX[1];
            old_point.z = oldX[2];
            old_point.lifetime = pc.points[i].lifetime;
            // old_point.lifetime = lidar_sweep.timestamp_;
            lidar::OctoTree * old_loc = nullptr;
            old_loc = voxel_map->LocateRoot(old_point);
            if (old_loc == new_loc && old_loc != nullptr) {
                uint64_t new_locate_code = new_loc->LocateCode(new_point);
                uint64_t old_locate_code = old_loc->LocateCode(old_point);
                if (old_locate_code == new_locate_code) {
                    continue;
                }
            }
            old_lidar_points.push_back(old_point);
            new_lidar_points.push_back(new_point);
            changed_num_point++;
        }
        changed_total_num_point += changed_num_point;
    }
    if (modified_sweep_ids.size() > 0) {
        double change_ratio = (double)changed_total_num_point / (double)num_total_lidar_point * 100;
        std::cout << StringPrintf("Lidar Sweep: %.2f%% points changed\n", change_ratio) 
                  << std::flush;
    }
    auto voxelmap_end_time = std::chrono::steady_clock::now();
    auto voxelmap_change_begin_time = std::chrono::steady_clock::now();
    if (old_lidar_points.size() > 0) {
        voxel_map->UpdateVoxelMapLazy(old_lidar_points, new_lidar_points);
        voxel_map->RebuildOctree();
    }
    auto voxelmap_change_end_time = std::chrono::steady_clock::now();

    if (modified_sweep_ids.size() > 0) {
        float elapsed_time = std::chrono::duration<float>(voxelmap_end_time - voxelmap_begin_time).count();
        float elapsed_time1 = std::chrono::duration<float>(voxelmap_change_end_time - voxelmap_change_begin_time).count();
        std::cout << StringPrintf("VoxelMap update in %.3f/%.3f sec\n", elapsed_time, elapsed_time1);
    }
    return report;
}

IncrementalMapper::LocalBundleAdjustmentReport IncrementalMapper::AdjustLocalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const image_t image_id,
    const std::unordered_set<mappoint_t>& mappoint_ids,const std::unordered_set<image_t>& fixed_images) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());

    const size_t num_reg_images = reconstruction_->NumRegisterImages();

    LocalBundleAdjustmentReport report;

    // Find images that have most Map Points with given image in common.
    const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);

    // Do the bundle adjustment only if there is any connected images.
    if (local_bundle.size() > 0) {
        BundleAdjustmentConfig ba_config;
        ba_config.AddImage(image_id);
        const Image& image = reconstruction_->Image(image_id);
        const Camera& camera = reconstruction_->Camera(image.CameraId());
        if (camera.IsCameraConstant() || num_reg_images <= options.num_fix_camera_first) {
            ba_config.SetConstantCamera(image.CameraId());
        }
        for (const image_t local_image_id : local_bundle) {
            ba_config.AddImage(local_image_id);
            const Image& local_image = reconstruction_->Image(local_image_id);
            const Camera& local_camera = reconstruction_->Camera(local_image.CameraId());
            if (local_camera.IsCameraConstant() || num_reg_images <= options.num_fix_camera_first) {
                ba_config.SetConstantCamera(local_image.CameraId());
            }

            if(fixed_images.count(local_image_id)>0){
                ba_config.SetConstantPose(local_image_id);
                ba_config.SetConstantCamera(local_image.CameraId());
            }
        }

        // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
        if (local_bundle.size() == 1) {
            ba_config.SetConstantPose(local_bundle[0]);
            //ba_config.SetConstantTvec(image_id, {0});

            int max_t_index = -1;
            double max_t = -1;
            const Image& second_image = reconstruction_->Image(image_id);
            for (int j = 0; j < 3; ++j) {
                if (abs(second_image.Tvec()[j]) > max_t) {
                    max_t = abs(second_image.Tvec()[j]);
                    max_t_index = j;
                }
            }
            ba_config.SetConstantTvec(image_id, {max_t_index});


        } else if (local_bundle.size() > 1) {
            ba_config.SetConstantPose(local_bundle[local_bundle.size() - 1]);
            
            if(fixed_images.count(local_bundle[local_bundle.size() - 2])==0){
                //ba_config.SetConstantTvec(local_bundle[local_bundle.size() - 2], {0});
            
                int max_t_index = -1;
                double max_t = -1;
                const Image& second_image = reconstruction_->Image(local_bundle[local_bundle.size() - 2]);
                for (int j = 0; j < 3; ++j) {
                    if (abs(second_image.Tvec()[j]) > max_t) {
                        max_t = abs(second_image.Tvec()[j]);
                        max_t_index = j;
                    }
                }
                ba_config.SetConstantTvec(local_bundle[local_bundle.size() - 2], {max_t_index});
            }
        }

        // Make sure, we refine all new and short-track Map Points, no matter if
        // they are fully contained in the local image set or not. Do not include
        // long track Map Points as they are usually already very stable and adding
        // to them to bundle adjustment and track merging/completion would slow
        // down the local bundle adjustment significantly.
        std::unordered_set<mappoint_t> variable_mappoint_ids;
        for (const mappoint_t mappoint_id : mappoint_ids) {
            if(!reconstruction_->ExistsMapPoint(mappoint_id)){
                continue;
            }
            const MapPoint& mappoint = reconstruction_->MapPoint(mappoint_id);
            const size_t kMaxTrackLength = 15;
            if (!mappoint.HasError() || mappoint.Track().Length() <= kMaxTrackLength) {
                ba_config.AddVariablePoint(mappoint_id);
                variable_mappoint_ids.insert(mappoint_id);
            }
        }

        // if (options.map_update && !options.update_old_map) {
        //     for (const mappoint_t mappoint_id : mappoint_ids) {
        //         fixed_mappoint_ids_.erase(mappoint_id);
        //     }
        // }

        // Adjust the local bundle.
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        bundle_adjuster.Solve(reconstruction_.get());

        report.num_adjusted_observations = bundle_adjuster.Summary().num_residuals / 2;

        // Merge refined tracks with other existing points.
        report.num_merged_observations = triangulator_->MergeTracks(tri_options, variable_mappoint_ids);
        // Complete tracks that may have failed to triangulate before refinement
        // of camera pose and calibration in bundle-adjustment. This may avoid
        // that some points are filtered and it helps for subsequent image
        // registrations.
        report.num_completed_observations = triangulator_->CompleteTracks(tri_options, variable_mappoint_ids);
        report.num_completed_observations += triangulator_->CompleteImage(tri_options, image_id);
    }

    // Filter both the modified images and all changed Map Points to make sure
    // there are no outlier points in the model. This results in duplicate work as
    // many of the provided Map Points may also be contained in the adjusted
    // images, but the filtering is not a bottleneck at this point.
    std::unordered_set<image_t> filter_image_ids;
    filter_image_ids.insert(image_id);
    filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
    report.num_filtered_observations = reconstruction_->FilterMapPointsInImages(
        options.filter_max_reproj_error, options.filter_min_tri_angle, filter_image_ids);
    report.num_filtered_observations +=
        reconstruction_->FilterMapPoints(options.filter_max_reproj_error, options.filter_min_tri_angle, mappoint_ids);

    return report;
}



IncrementalMapper::LocalBundleAdjustmentReport IncrementalMapper::AdjustBatchedLocalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const std::vector<image_t>& image_ids,
    const std::unordered_set<mappoint_t>& mappoint_ids) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());

    const size_t num_reg_images = reconstruction_->NumRegisterImages();

    LocalBundleAdjustmentReport report;

    // Find images that have most Map Points with given images in common.
    std::unordered_set<image_t> local_bundle_set;
    for (const auto& image_id : image_ids) {
        const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);
        for (const auto local_image_id : local_bundle) {
            local_bundle_set.insert(local_image_id);
        }
    }
    std::vector<image_t> local_bundles;
    for (const auto local_image_id : local_bundle_set) {
        local_bundles.push_back(local_image_id);
    }

    // Do the bundle adjustment only if there is any connected images.
    if (local_bundles.size() > 0) {
        BundleAdjustmentConfig ba_config;

        for (const auto image_id : image_ids) {
            ba_config.AddImage(image_id);
            const Image& image = reconstruction_->Image(image_id);
            const Camera& camera = reconstruction_->Camera(image.CameraId());
            if (camera.IsCameraConstant() || num_reg_images <= options.num_fix_camera_first) {
                ba_config.SetConstantCamera(image.CameraId());
            }
        }
        for (const image_t local_image_id : local_bundles) {
            ba_config.AddImage(local_image_id);
            const Image& local_image = reconstruction_->Image(local_image_id);
            const Camera& local_camera = reconstruction_->Camera(local_image.CameraId());
            if (local_camera.IsCameraConstant() || num_reg_images <= options.num_fix_camera_first) {
                ba_config.SetConstantCamera(local_image.CameraId());
            }
        }

        // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
        if (local_bundles.size() == 1) {
            ba_config.SetConstantPose(local_bundles[0]);
            ba_config.SetConstantTvec(image_ids[0], {0});
        } else if (local_bundles.size() > 1) {
            ba_config.SetConstantPose(local_bundles[local_bundles.size() - 1]);
            ba_config.SetConstantTvec(local_bundles[local_bundles.size() - 2], {0});
        }

        // Make sure, we refine all new and short-track Map Points, no matter if
        // they are fully contained in the local image set or not. Do not include
        // long track Map Points as they are usually already very stable and adding
        // to them to bundle adjustment and track merging/completion would slow
        // down the local bundle adjustment significantly.
        std::unordered_set<mappoint_t> variable_mappoint_ids;
        for (const mappoint_t mappoint_id : mappoint_ids) {
            const MapPoint& mappoint = reconstruction_->MapPoint(mappoint_id);
            const size_t kMaxTrackLength = 15;
            if (!mappoint.HasError() || mappoint.Track().Length() <= kMaxTrackLength) {
                ba_config.AddVariablePoint(mappoint_id);
                variable_mappoint_ids.insert(mappoint_id);
            }
        }
        
        // if (options.map_update && !options.update_old_map) {
        //     for (const mappoint_t mappoint_id : mappoint_ids) {
        //         fixed_mappoint_ids_.erase(mappoint_id);
        //     }
        // }

        // Adjust the local bundle.
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        bundle_adjuster.Solve(reconstruction_.get());

        report.num_adjusted_observations = bundle_adjuster.Summary().num_residuals / 2;

        // Merge refined tracks with other existing points.
        report.num_merged_observations = triangulator_->MergeTracks(tri_options, variable_mappoint_ids);
        // Complete tracks that may have failed to triangulate before refinement
        // of camera pose and calibration in bundle-adjustment. This may avoid
        // that some points are filtered and it helps for subsequent image
        // registrations.
        report.num_completed_observations = triangulator_->CompleteTracks(tri_options, variable_mappoint_ids);
        for (const auto image_id : image_ids) {
            report.num_completed_observations += triangulator_->CompleteImage(tri_options, image_id);
        }
    }

    // Filter both the modified images and all changed Map Points to make sure
    // there are no outlier points in the model. This results in duplicate work as
    // many of the provided Map Points may also be contained in the adjusted
    // images, but the filtering is not a bottleneck at this point.
    std::unordered_set<image_t> filter_image_ids;
    filter_image_ids.insert(image_ids.begin(), image_ids.end());
    filter_image_ids.insert(local_bundles.begin(), local_bundles.end());
    report.num_filtered_observations = reconstruction_->FilterMapPointsInImages(
        options.filter_max_reproj_error, options.filter_min_tri_angle, filter_image_ids);
    report.num_filtered_observations +=
        reconstruction_->FilterMapPoints(options.filter_max_reproj_error, options.filter_min_tri_angle, mappoint_ids);

    return report;
}

void IncrementalMapper::AdjustUpdatedBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
    CHECK(options.Check());

    const bool constant_camera =
        !ba_options.refine_focal_length && !ba_options.refine_extra_params && !ba_options.refine_principal_point;

    double lower_bound_focal_length_factor = std::max(ba_options.lower_bound_focal_length_factor, 0.8);
    double upper_bound_focal_length_factor = std::max(ba_options.upper_bound_focal_length_factor, 1.2);

    ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);

    const std::vector<image_t> & image_ids = reconstruction_->GetNewImageIds();

    std::unordered_set<image_t> connected_image_ids;
    for (auto image_id : image_ids) {
        auto & image = reconstruction_->Image(image_id);
        const auto & points2D = image.Points2D();
        for (size_t i = 0; i < points2D.size(); ++i) {
            Point2D point2D = points2D.at(i);
            if (!point2D.HasMapPoint()) continue;
            
            const MapPoint& mappoint = reconstruction_->MapPoint(point2D.MapPointId());
            for (auto track_el : mappoint.Track().Elements()) {
                connected_image_ids.insert(track_el.image_id);
            }
        }
    }

    std::unordered_set<image_t> config_image_ids;
    std::unordered_set<mappoint_t> variable_mappoint_ids;
    std::unordered_set<mappoint_t> constant_mappoint_ids;

    std::cout << "AdjustBatchedLocalBundle: " << image_ids.size() << " images" << std::endl;

    ceres::Problem problem;

    double *local_qvec2_data;
    double *local_tvec2_data;

    for (auto image_id : image_ids) {
        auto & image = reconstruction_->Image(image_id);
        auto & camera = reconstruction_->Camera(image.CameraId());
        const auto & local_camera_indices = image.LocalImageIndices();
        const auto & points2D = image.Points2D();

        config_image_ids.insert(image_id);

        double *qvec_data = image.Qvec().data();
        double *tvec_data = image.Tvec().data();
        double *camera_params_data = camera.ParamsData();

        std::vector<double*> local_qvec_data;
        std::vector<double*> local_tvec_data;
        std::vector<double*> local_camera_params_data;

        if (camera.NumLocalCameras() > 1) {
            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();

            for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
                local_qvec_data.push_back(camera.LocalQvecsData() + 4 * i);
                local_tvec_data.push_back(camera.LocalTvecsData() + 3 * i);
                local_camera_params_data.push_back(camera.LocalIntrinsicParamsData() + i * local_param_size);
            }
        }
        
        // Time domain smoothing.
        image_t prev_image_id = -1;
        image_t next_image_id = -1;
        for (size_t i = 1; i <= 5 && prev_image_id == -1; ++i) {
            image_t neighbor_image_id = image_id - i;
            if (reconstruction_->ExistsImage(neighbor_image_id) &&
                reconstruction_->Image(neighbor_image_id).LabelId() == image.LabelId()) {
                prev_image_id = neighbor_image_id;
                break;
            }
        }

        for (size_t i = 1; i <= 5 && next_image_id == -1; ++i) {
            image_t neighbor_image_id = image_id + i;
            if (reconstruction_->ExistsImage(neighbor_image_id) &&
                reconstruction_->Image(neighbor_image_id).LabelId() == image.LabelId()) {
                next_image_id = neighbor_image_id;
                break;
            }
        }

        if (prev_image_id != -1 && next_image_id != -1) {
            Image &prev_image = reconstruction_->Image(prev_image_id);
            Image &next_image = reconstruction_->Image(next_image_id);

            const float prev_time = image_id - prev_image_id;
            const float next_time = next_image_id - image_id;

            ceres::CostFunction *cost_function = TimeDomainSmoothingCostFunction::Create(
                prev_time, next_time, 10.0);
            problem.AddResidualBlock(cost_function, loss_function, 
                qvec_data, tvec_data, 
                prev_image.Qvec().data(), prev_image.Tvec().data(),
                next_image.Qvec().data(), next_image.Tvec().data());
        }

        // Reprojection error.
        for (size_t i = 0; i < points2D.size(); ++i) {
            uint32_t local_camera_id = local_camera_indices[i];
            Point2D point2D = points2D.at(i);
            
            if (!point2D.HasMapPoint()) continue;

            MapPoint &mappoint = reconstruction_->MapPoint(point2D.MapPointId());
            variable_mappoint_ids.insert(point2D.MapPointId());

            ceres::CostFunction* cost_function = nullptr;

            if (camera.NumLocalCameras() > 1) {
                // for camera-rig
                if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_camera) {

                    double f = camera.LocalMeanFocalLength(local_camera_id);
                    Eigen::Vector3d bearing = camera.LocalImageToBearing(local_camera_id, point2D.XY());
                    cost_function =
                        LargeFovRigBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(bearing, f);
                    problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                            local_qvec_data[local_camera_id], local_tvec_data[local_camera_id],
                                            mappoint.XYZ().data());
                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                      \
    case CameraModel::kModelId:                                                             \
        cost_function = RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY(), 1.0); \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                            local_qvec_data[local_camera_id], local_tvec_data[local_camera_id],
                                            mappoint.XYZ().data(), local_camera_params_data[local_camera_id]);
                }

            } else {
                if (camera.ModelName().compare("SPHERICAL") == 0) {
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.FocalLength();
                    
                    cost_function =
                        SphericalBundleAdjustmentCostFunction<SphericalCameraModel>::Create(bearing, f);

                    local_qvec2_data = camera_params_data + 10;
                    local_tvec2_data = camera_params_data + 14;

                    problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                            mappoint.XYZ().data(), local_qvec2_data, local_tvec2_data);
                } else if(camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_camera){ 
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.MeanFocalLength();
                    cost_function =
                        LargeFovBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(bearing, f);
                    problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                            mappoint.XYZ().data());

                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                           \
    case CameraModel::kModelId:                                                                  \
        cost_function = BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                            mappoint.XYZ().data(), camera_params_data);
                }
            }

        }

        if (problem.NumResiduals() > 0) {
            // Quaternion parameterization.
            ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
            problem.SetManifold(qvec_data, quaternion_parameterization);
#else
            problem.SetParameterization(qvec_data, quaternion_parameterization);
#endif
        }
    }

    std::unordered_set<camera_t> constant_camera_ids;

    for (auto image_id : connected_image_ids) {
        if (config_image_ids.find(image_id) != config_image_ids.end()) {
            continue;
        }
        auto & image = reconstruction_->Image(image_id);
        auto & camera = reconstruction_->Camera(image.CameraId());

        const auto & local_camera_indices = image.LocalImageIndices();
        const auto & points2D = image.Points2D();

        config_image_ids.insert(image_id);
        constant_camera_ids.insert(image.CameraId());

        std::vector<double*> local_qvec_data;
        std::vector<double*> local_tvec_data;
        std::vector<double*> local_camera_params_data;

        if (camera.NumLocalCameras() > 1) {
            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
            for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
                local_qvec_data.push_back(camera.LocalQvecsData() + 4 * i);
                local_tvec_data.push_back(camera.LocalTvecsData() + 3 * i);
                local_camera_params_data.push_back(camera.LocalIntrinsicParamsData() + i * local_param_size);
            }
        }

        // Reprojection error.
        for (size_t i = 0; i < points2D.size(); ++i) {
            uint32_t local_camera_id = local_camera_indices[i];
            Point2D point2D = points2D.at(i);
            
            if (!point2D.HasMapPoint()) continue;

            MapPoint &mappoint = reconstruction_->MapPoint(point2D.MapPointId());

            ceres::CostFunction *cost_function = nullptr;

            if (camera.NumLocalCameras() > 1) {
                if (camera.ModelName().compare("OPENCV_FISHEYE") == 0) {
                    double f = camera.LocalMeanFocalLength(local_camera_id);
                    Eigen::Vector3d bearing = camera.LocalImageToBearing(local_camera_id, point2D.XY());
                    cost_function = LargeFovRigBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                        image.Qvec(), image.Tvec(), bearing, f);

                    problem.AddResidualBlock(cost_function, loss_function, local_qvec_data[local_camera_id], 
                                             local_tvec_data[local_camera_id], mappoint.XYZ().data());
                }
            } else {
                // for monocular camera
                if (camera.ModelName().compare("SPHERICAL") == 0) {
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.FocalLength();

                    cost_function = SphericalBundleAdjustmentConstantPoseCostFunction<SphericalCameraModel>::Create(
                        image.Qvec(), image.Tvec(), bearing, f);

                    double *local_qvec2_data = camera.ParamsData() + 10;
                    double *local_tvec2_data = camera.ParamsData() + 14;

                    problem.AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(),
                                            local_qvec2_data, local_tvec2_data);
                } else if (camera.ModelName().compare("OPENCV_FISHEYE") == 0){
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.MeanFocalLength();
                    cost_function = LargeFovBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                        image.Qvec(), image.Tvec(), bearing, f);

                    problem.AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
                } else {
                    switch (camera.ModelId()) {
    #define CAMERA_MODEL_CASE(CameraModel)                                                                               \
        case CameraModel::kModelId:                                                                                      \
            cost_function =                                                                                              \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(image.Qvec(), image.Tvec(), point2D.XY()); \
            break;
                        CAMERA_MODEL_SWITCH_CASES
    #undef CAMERA_MODEL_CASE
                    }
                    problem.AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(), camera.ParamsData());
                }
            }
            if (variable_mappoint_ids.find(point2D.MapPointId()) == variable_mappoint_ids.end()) {
                constant_mappoint_ids.insert(point2D.MapPointId());
            }
        }
    }

    for (auto mappoint_id : constant_mappoint_ids) {
        problem.SetParameterBlockConstant(reconstruction_->MapPoint(mappoint_id).XYZ().data());
    }    

    // Parameterize cameras.
    for (auto camera_id : constant_camera_ids) {
        Camera &camera = reconstruction_->Camera(camera_id);
        if (camera.NumLocalCameras() > 1) {
            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
            for (int local_camera_id = 0; local_camera_id < camera.NumLocalCameras(); ++local_camera_id) {
                double *local_qvec_data = camera.LocalQvecsData() + 4 * local_camera_id;
                double *local_tvec_data = camera.LocalTvecsData() + 3 * local_camera_id;
                problem.SetParameterBlockConstant(local_qvec_data);
                problem.SetParameterBlockConstant(local_tvec_data);
            }
        } else if (camera.ModelName().compare("SPHERICAL") == 0) {
            double *local_qvec2_data = camera.ParamsData() + 10;
            double *local_tvec2_data = camera.ParamsData() + 14;
            problem.SetParameterBlockConstant(local_qvec2_data);
            problem.SetParameterBlockConstant(local_tvec2_data);
        } else if (camera.ModelName().compare("OPENCV_FISHEYE") != 0) {
            problem.SetParameterBlockConstant(camera.ParamsData());
        }
    }

    // Parameterize cameras inside updated map.
    {
        std::unordered_set<camera_t> camera_ids;
        for (auto image_id : image_ids) {
            camera_ids.insert(reconstruction_->Image(image_id).CameraId());
        }

        for (auto camera_id : camera_ids) {
            Camera &camera = reconstruction_->Camera(camera_id);

            if (camera.NumLocalCameras() > 1) {
                int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
                for (int local_camera_id = 0; local_camera_id < camera.NumLocalCameras(); ++local_camera_id) {
                    double *local_qvec_data = camera.LocalQvecsData() + 4 * local_camera_id;
                    double *local_tvec_data = camera.LocalTvecsData() + 3 * local_camera_id;
                    double *local_camera_params_data =
                        camera.LocalIntrinsicParamsData() + local_camera_id * local_param_size;

                    if(!(camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_camera)){
                        if (constant_camera) {
                            problem.SetParameterBlockConstant(local_camera_params_data);
                        } else {
                            std::vector<int> const_camera_params;
                            if (!ba_options.refine_focal_length) {
                                const std::vector<size_t> &params_idxs = camera.FocalLengthIdxs();
                                const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                        params_idxs.end());
                            }
                            if (!ba_options.refine_principal_point) {
                                const std::vector<size_t> &params_idxs = camera.PrincipalPointIdxs();
                                const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                        params_idxs.end());
                            }
                            if ((!ba_options.refine_extra_params) && (camera.ModelName().compare("SPHERICAL") != 0)) {
                                const std::vector<size_t> &params_idxs = camera.ExtraParamsIdxs();
                                const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                        params_idxs.end());
                            }

                            if (const_camera_params.size() > 0) {
                                ceres::SubsetParameterization *camera_params_parameterization =
                                    new ceres::SubsetParameterization(static_cast<int>(local_param_size),
                                                                    const_camera_params);
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
                                problem.SetManifold(local_camera_params_data, camera_params_parameterization);
#else
                                problem.SetParameterization(local_camera_params_data, camera_params_parameterization);
#endif
                            }
                        }
                    }

                    if (ba_options.refine_local_extrinsics && local_camera_id > 0) {
                        ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
                        problem.SetManifold(local_qvec_data, quaternion_parameterization);
#else
                        problem.SetParameterization(local_qvec_data, quaternion_parameterization);
#endif
                    } else {
                        problem.SetParameterBlockConstant(local_qvec_data);
                        problem.SetParameterBlockConstant(local_tvec_data);
                    }
                }
                continue;
            }

            if (constant_camera) {
                if(camera.ModelName().compare("SPHERICAL") != 0){
                    problem.SetParameterBlockConstant(camera.ParamsData());
                }
                if (camera.ModelName().compare("SPHERICAL") == 0) {
                    double *camera_params_data = camera.ParamsData();

                    double *local_qvec2_data = camera_params_data + 10;
                    double *local_tvec2_data = camera_params_data + 14;
                    problem.SetParameterBlockConstant(local_qvec2_data);
                    problem.SetParameterBlockConstant(local_tvec2_data);
                }
                continue;
            } else {
                if(camera.ModelName() != "SPHERICAL"){
                    std::vector<int> const_camera_params;

                    if (!ba_options.refine_focal_length) {
                        const std::vector<size_t> &params_idxs = camera.FocalLengthIdxs();
                        const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                    }
                    if (!ba_options.refine_principal_point) {
                        const std::vector<size_t> &params_idxs = camera.PrincipalPointIdxs();
                        const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                    }
                    if ((!ba_options.refine_extra_params) && (camera.ModelName().compare("SPHERICAL") != 0)) {
                        const std::vector<size_t> &params_idxs = camera.ExtraParamsIdxs();
                        const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                    }

                    for (size_t idx : camera.FocalLengthIdxs()) {
                        double est_focal = camera.ParamsData()[idx];
                        problem.SetParameterLowerBound(camera.ParamsData(), idx,
                                                        lower_bound_focal_length_factor * est_focal);
                        problem.SetParameterUpperBound(camera.ParamsData(), idx,
                                                        upper_bound_focal_length_factor * est_focal);
                    }
                    for (size_t idx : camera.PrincipalPointIdxs()) {
                        problem.SetParameterLowerBound(camera.ParamsData(), idx, 0);
                        problem.SetParameterUpperBound(camera.ParamsData(), idx,
                                                        std::max(camera.Width(), camera.Height()));
                    }

                    for (size_t idx : camera.ExtraParamsIdxs()) {
                        problem.SetParameterLowerBound(camera.ParamsData(), idx, -0.99);
                        problem.SetParameterUpperBound(camera.ParamsData(), idx, 0.99);
                    }

                    if (const_camera_params.size() > 0) {
                        ceres::SubsetParameterization *camera_params_parameterization =
                            new ceres::SubsetParameterization(static_cast<int>(camera.NumParams()),
                                                            const_camera_params);
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
                        problem.SetManifold(camera.ParamsData(), camera_params_parameterization);
#else
                        problem.SetParameterization(camera.ParamsData(), camera_params_parameterization);
#endif
                    }
                }

                if (camera.ModelName().compare("SPHERICAL") == 0) {
                    double *camera_params_data = camera.ParamsData();

                    double *local_qvec2_data = camera_params_data + 10;
                    double *local_tvec2_data = camera_params_data + 14;

                    if (ba_options.refine_extra_params) {
                        ceres::LocalParameterization *quaternion_parameterization2 = new ceres::QuaternionParameterization;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
                        problem.SetManifold(local_qvec2_data, quaternion_parameterization2);
#else
                        problem.SetParameterization(local_qvec2_data, quaternion_parameterization2);
#endif
                    } else {
                        problem.SetParameterBlockConstant(local_qvec2_data);
                        problem.SetParameterBlockConstant(local_tvec2_data);
                    }
                }
            }
        }
    }

    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.num_threads = GetEffectiveNumThreads(-1);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = GetEffectiveNumThreads(-1);
#endif  // CERES_VERSION_MAJOR

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (ba_options.print_summary) {
        PrintHeading2("AdjustUpdatedBundle");
        PrintSolverSummary(summary);
    }
}

bool IncrementalMapper::AdjustGlobalBundle(const Options& options, const BundleAdjustmentOptions& ba_options) {
    CHECK_NOTNULL(reconstruction_.get());

    const std::vector<image_t>& reg_image_ids = reconstruction_->RegisterImageIds();

    if (reg_image_ids.size() < 2) {
        return false;
    }

    CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                         "registered for global "
                                         "bundle-adjustment";
    
    int ba_count = GlobalAdjustmentCount();
    AccGlobalAdjustmentCount();
    BundleAdjustmentOptions custom_ba_options = ba_options;
    if (options.map_update){
        reconstruction_->b_aligned = true;
    }else if (custom_ba_options.use_prior_absolute_location) {
        CHECK(reconstruction_->has_gps_prior);
        reconstruction_->AlignWithPriorLocations(options.max_error_gps, options.max_error_horizontal_gps,
                                                 options.max_gps_time_offset);
    }

    if (custom_ba_options.plane_constrain) {
        reconstruction_->ComputePrimaryPlane(options.max_distance_to_plane, options.max_plane_count);
    }

    if (options.with_depth && options.rgbd_delayed_start && !reconstruction_->depth_enabled) {
        reconstruction_->depth_enabled = reconstruction_->TryScaleAdjustmentWithDepth(options.rgbd_delayed_start_weights);
    }

    // Avoid degeneracies in bundle adjustment.
    reconstruction_->FilterObservationsWithNegativeDepth();

    // if (options.map_update && !options.update_old_map) {
    //     const auto & modified_mappoints = GetModifiedMapPoints();
    //     for (const auto & mappoint_id : modified_mappoints) {
    //         fixed_mappoint_ids_.erase(mappoint_id);
    //     }
    // }

    const std::vector<sweep_t> & reg_sweep_ids = reconstruction_->RegisterSweepIds();
    std::cout << "reg_sweep_ids: " << reg_sweep_ids.size() << std::endl;
    std::unordered_map<sweep_t, Eigen::Matrix3x4d> sweep_old_poses;

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    if (options.map_update && !options.update_old_map) {
        const auto & correspondence_graph = scene_graph_container_->CorrespondenceGraph();
        std::unordered_set<image_t> const_image_ids;
        std::vector<image_t> variable_image_ids;
        variable_image_ids.reserve(reg_image_ids.size());
        for (const image_t image_id : reg_image_ids) {
            if (fixed_images_[image_id]) {
                continue;
            }
            variable_image_ids.push_back(image_id);

            const auto & image_neighbors = correspondence_graph->ImageNeighbor(image_id);
            for (auto neighbor_id : image_neighbors) {
                if (fixed_images_[neighbor_id] &&
                    const_image_ids.find(neighbor_id) == const_image_ids.end()) {
                    const_image_ids.insert(neighbor_id);
                }
            }
        }
        std::cout << "Fix " << const_image_ids.size() << " images in old map." << std::endl;

        for (const image_t image_id : variable_image_ids) {
            ba_config.AddImage(image_id);
            const Image& image = reconstruction_->Image(image_id);
            const Camera& camera = reconstruction_->Camera(image.CameraId());
            if (camera.IsCameraConstant()) {
                ba_config.SetConstantCamera(image.CameraId());
            }
        }
        for (const auto & image_id : const_image_ids) {
            ba_config.AddImage(image_id);
            const Image& image = reconstruction_->Image(image_id);
            ba_config.SetConstantPose(image_id);
            ba_config.SetConstantCamera(image.CameraId());
        }

        // for (const mappoint_t mappoint_id : fixed_mappoint_ids_) {
        //     if (reconstruction_->ExistsMapPoint(mappoint_id)) {
        //         ba_config.AddConstantPoint(mappoint_id);
        //     }
        // }
    } else {
        for (const image_t image_id : reg_image_ids) {
            ba_config.AddImage(image_id);
            const Image& image = reconstruction_->Image(image_id);
            const Camera& camera = reconstruction_->Camera(image.CameraId());
            if (camera.IsCameraConstant() || reg_image_ids.size() <= options.num_fix_camera_first) {
                ba_config.SetConstantCamera(image.CameraId());
            }
        }

        CHECK(reconstruction_->RegisterImageIds().size() > 0);
        const image_t first_image_id = reconstruction_->RegisterImageIds()[0];
        CHECK(reconstruction_->ExistsImage(first_image_id));
        const Image& image = reconstruction_->Image(first_image_id);
        const Camera& camera = reconstruction_->Camera(image.CameraId());

        if (!custom_ba_options.use_prior_absolute_location || !reconstruction_->b_aligned) {
            ba_config.SetConstantPose(reg_image_ids[0]);
            
            int max_t_index = -1;
            double max_t = -1;
            const Image& second_image = reconstruction_->Image(reg_image_ids[1]);
            for(int j = 0; j < 3; ++j){
                if(abs(second_image.Tvec()[j]) > max_t){
                    max_t = abs(second_image.Tvec()[j]);
                    max_t_index = j;
                }
            }   
            
            ba_config.SetConstantTvec(reg_image_ids[1], {max_t_index});
        }

        for (auto image_lidar_pair : image_to_lidar_map_) {
            sweep_t sweep_id = image_lidar_pair.second;
            if (reconstruction_->ExistsLidarSweep(sweep_id) && reconstruction_->LidarSweep(sweep_id).IsRegistered()) {
                ba_config.AddSweepImagePair(image_lidar_pair.second, image_lidar_pair.first);
            }
        }
        std::cout << "Collect " << ba_config.SweepImagePairs().size() << " seep image pairs" << std::endl;
        ba_config.SetLidar2CamMatrix(options.lidar_to_cam_matrix);

        for (const sweep_t sweep_id : reg_sweep_ids) {
            ba_config.AddSweep(sweep_id);
            sweep_old_poses[sweep_id] = reconstruction_->LidarSweep(sweep_id).InverseProjectionMatrix();
        }
    }
    
    // Run bundle adjustment.
    auto ba_begin_time = std::chrono::steady_clock::now();

    // Disable Block BA.
    custom_ba_options.force_full_ba = true;

    // Enable BlockBA when the number of registered images is larger than block_size.
    if (custom_ba_options.block_size != -1 &&
        reconstruction_->NumRegisterImages() > custom_ba_options.block_size) {
        custom_ba_options.force_full_ba = true;
    }

    if (custom_ba_options.force_full_ba) {
#ifdef USE_OPENBLAS
        openblas_set_num_threads(GetEffectiveNumThreads(-1));
#endif
        BundleAdjuster bundle_adjuster(custom_ba_options, ba_config);
        if(!bundle_adjuster.Solve(reconstruction_.get())) {
            return false;
        }
    } else {
        BlockBundleAdjuster bundle_adjuster(custom_ba_options, ba_config);
        // bundle_adjuster.SetGlobalBACount(ba_count);
        if (!bundle_adjuster.Solve(reconstruction_.get())) {
            return false;
        }
    }
    auto ba_end_time = std::chrono::steady_clock::now();
    printf("GBATime: %d images, %d points, %.3f sec.\n", 
        ba_config.NumImages(), reconstruction_->NumMapPoints(), 
        std::chrono::duration<float>(ba_end_time - ba_begin_time).count());
    //   // Normalize scene for numerical stability and
    //   // to avoid large scale changes in viewer.
    if(options.ba_normalize_reconstruction){
        reconstruction_->Normalize();
    }

    auto voxelmap_begin_time = std::chrono::steady_clock::now();

    std::shared_ptr<VoxelMap> & voxel_map = reconstruction_->VoxelMap();

    uint64_t num_total_lidar_point = 0, changed_total_num_point = 0;
    std::vector<lidar::OctoTree::Point> old_lidar_points, new_lidar_points;
    for (const sweep_t sweep_id : reg_sweep_ids) {
        const class LidarSweep & lidar_sweep = reconstruction_->LidarSweep(sweep_id);
        const LidarPointCloud & ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
        const LidarPointCloud & ref_less_corners = lidar_sweep.GetCornerPointsLessSharp();
        const size_t num_surf = ref_less_surfs.points.size();
        const size_t num_corner = ref_less_corners.points.size();
        num_total_lidar_point += num_surf + num_corner;
    }
    old_lidar_points.reserve(num_total_lidar_point);
    new_lidar_points.reserve(num_total_lidar_point);

    for (const sweep_t sweep_id : reg_sweep_ids) {
        const class LidarSweep & lidar_sweep = reconstruction_->LidarSweep(sweep_id);
        Eigen::Matrix3x4d old_inv_projmat = sweep_old_poses[sweep_id];
        Eigen::Matrix3x4d new_inv_projmat = reconstruction_->LidarSweep(sweep_id).InverseProjectionMatrix();
        LidarPointCloud pc;
        LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
        LidarPointCloud ref_less_corners = lidar_sweep.GetCornerPointsLessSharp();
        const size_t num_surf = ref_less_surfs.points.size();
        const size_t num_corner = ref_less_corners.points.size();
        pc = std::move(ref_less_surfs);
        pc += std::move(ref_less_corners);

        size_t changed_num_point = 0;
// #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < pc.points.size(); ++i) {
            Eigen::Vector4d hpoint(pc.points[i].x, pc.points[i].y, pc.points[i].z, 1.0);
            Eigen::Vector3d newX = new_inv_projmat * hpoint;
            lidar::OctoTree::Point new_point;
            new_point.x = newX[0];
            new_point.y = newX[1];
            new_point.z = newX[2];
            new_point.lifetime = pc.points[i].lifetime;
            // new_point.lifetime = lidar_sweep.timestamp_;
            lidar::OctoTree * new_loc = nullptr;
            new_loc = voxel_map->LocateRoot(new_point);

            Eigen::Vector3d oldX = old_inv_projmat * hpoint;
            lidar::OctoTree::Point old_point;
            old_point.x = oldX[0];
            old_point.y = oldX[1];
            old_point.z = oldX[2];
            old_point.lifetime = pc.points[i].lifetime;
            // old_point.lifetime = lidar_sweep.timestamp_;
            lidar::OctoTree * old_loc = nullptr;
            old_loc = voxel_map->LocateRoot(old_point);
            if (old_loc == new_loc && old_loc != nullptr) {
                uint64_t new_locate_code = new_loc->LocateCode(new_point);
                uint64_t old_locate_code = old_loc->LocateCode(old_point);
                if (old_locate_code == new_locate_code) {
                    continue;
                }
            }

            old_lidar_points.push_back(old_point);
            new_lidar_points.push_back(new_point);
            changed_num_point++;
        }
        changed_total_num_point += changed_num_point;
    }
    std::cout << "old_lidar_points.size(): " << old_lidar_points.size() << std::endl;
    std::cout << "new_lidar_points.size(): " << new_lidar_points.size() << std::endl;
    if (reg_sweep_ids.size() > 0) {
        double change_ratio = (double)changed_total_num_point / (double)num_total_lidar_point * 100;
        std::cout << StringPrintf("Lidar Sweep: %.2f%% points changed\n", change_ratio) << std::flush;
    }
    auto voxelmap_end_time = std::chrono::steady_clock::now();
    auto voxelmap_change_begin_time = std::chrono::steady_clock::now();
    if (old_lidar_points.size() > 0) {
        voxel_map->UpdateVoxelMapLazy(old_lidar_points, new_lidar_points);
        voxel_map->RebuildOctree();
    }
    auto voxelmap_change_end_time = std::chrono::steady_clock::now();

    if (reg_sweep_ids.size() > 0) {
        float elapsed_time = std::chrono::duration<float>(voxelmap_end_time - voxelmap_begin_time).count();
        float elapsed_time1 = std::chrono::duration<float>(voxelmap_change_end_time - voxelmap_change_begin_time).count();
        std::cout << StringPrintf("VoxelMap update in %.3f/%.3f sec\n", elapsed_time, elapsed_time1);
    }
    return true;
}

bool IncrementalMapper::AdjustGlobalBundleNonKeyFrames(const Options& options,
                                                       const BundleAdjustmentOptions& ba_options,
                                                       const std::unordered_set<mappoint_t>& const_mappoint_ids) {
    CHECK_NOTNULL(reconstruction_.get());
    auto& images = reconstruction_->Images();
    CHECK_GE(images.size(), 2) << "At least two images must be "
                                  "registered for global "
                                  "bundle-adjustment";

    if (ba_options.plane_constrain) {
        reconstruction_->ComputePrimaryPlane(options.max_distance_to_plane, options.max_plane_count);
    }

    // Avoid degeneracies in bundle adjustment.
    reconstruction_->FilterObservationsWithNegativeDepth();

    BundleAdjustmentConfig ba_config;
    for (auto& image : images) {
        ba_config.AddImage(image.first);
        if (image.second.IsKeyFrame()) {
            ba_config.SetConstantPose(image.first);
        }
        const Camera& camera = reconstruction_->Camera(image.second.CameraId());
        if (camera.IsCameraConstant()) {
            ba_config.SetConstantCamera(image.second.CameraId());
        }
    }
    // for (const auto& mappoint_id : const_mappoint_ids) {
    // 	if (reconstruction_->ExistsMapPoint(mappoint_id)) {
    // 		ba_config.AddConstantPoint(mappoint_id);
    // 	}
    // }

    if (ba_config.NumImages() > 1) {
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        if (!bundle_adjuster.Solve(reconstruction_.get())) {
            std::cout << "Bundle Adjustment Failed!" << std::endl;
            return false;
        }
    }
    return true;
}

bool IncrementalMapper::AdjustFrame2FrameBundle(
    const Options& options, 
    const BundleAdjustmentOptions& ba_options,
    const std::vector<sweep_t> sweep_ids, 
    const std::unordered_set<sweep_t>& fixed_sweep_ids,
    double & final_cost){
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());

    BundleAdjustmentOptions f2f_ba_options = ba_options;
    f2f_ba_options.lidarsweep_voxel_gnss = false;

    f2f_ba_options.refine_focal_length = false;
    f2f_ba_options.refine_principal_point = false;
    f2f_ba_options.refine_extra_params = false;
    f2f_ba_options.refine_local_extrinsics = false;
    
    int num_fix_sweep = 0;
    BundleAdjustmentConfig ba_config;
    for (size_t i = 0; i < sweep_ids.size(); i++){
        const auto sweep_id = sweep_ids.at(i);
        if (!reconstruction_->LidarSweep(sweep_id).IsRegistered()){
            continue;
        }
        ba_config.AddSweep(sweep_id);
        if (fixed_sweep_ids.count(sweep_id) > 0){
            ba_config.SetConstantSweep(sweep_id);
            num_fix_sweep++;
        }
    }
    std::cout << "ba_config: num_sweeps " << ba_config.NumSweeps() << std::endl;
    std::cout << "fixed sweeps: " << num_fix_sweep << std::endl;
    std::cout << "num_iteration_frame2frame: " << ba_options.max_num_iteration_frame2frame << std::endl;

    PrintHeading1("LidarSweep F2F Bundle adjustment");

    int num_iter_failed = 0, num_iter_converged = 0;
    double initial_cost = std::numeric_limits<double>::max();
    final_cost = initial_cost;
    for (int i = 0; i < ba_options.max_num_iteration_frame2frame && num_iter_converged < 7; i++){
        f2f_ba_options.save_path = workspace_path_ + "/lidar_ba_info/" + std::to_string(reconstruction_->RegisterSweepIds().size()) + "_"
                                 + GetPathBaseName(reconstruction_->LidarSweep(sweep_ids.at(0)).Name()) + "_"
                                 + GetPathBaseName(reconstruction_->LidarSweep(sweep_ids.at(1)).Name()) + "/" 
                                 + std::to_string(i);
        // std::cout << "BA iter Before" << i << ": \n \tframe " 
        //     << reconstruction_->LidarSweep(sweep_ids.at(0)).Name() << ": "
        //     << reconstruction_->LidarSweep(sweep_ids.at(0)).Qvec().transpose() << ", " 
        //     << reconstruction_->LidarSweep(sweep_ids.at(0)).Tvec().transpose() << " / " 
        //     << reconstruction_->LidarSweep(sweep_ids.at(1)).Name() << ": "
        //     << reconstruction_->LidarSweep(sweep_ids.at(1)).Qvec().transpose() << ", " 
        //     << reconstruction_->LidarSweep(sweep_ids.at(1)).Tvec().transpose() << std::endl;
        BundleAdjuster bundle_adjuster(f2f_ba_options, ba_config);
        if (!bundle_adjuster.Solve(reconstruction_.get())) {
            return false;
        }

        // std::cout << "BA iter After" << i << ": \n \tframe " << reconstruction_->LidarSweep(sweep_ids.at(0)).Name() << ": "
        //     << reconstruction_->LidarSweep(sweep_ids.at(0)).Qvec().transpose() << ", " 
        //     << reconstruction_->LidarSweep(sweep_ids.at(0)).Tvec().transpose() << " / " 
        //     << reconstruction_->LidarSweep(sweep_ids.at(1)).Name() << ": "
        //     << reconstruction_->LidarSweep(sweep_ids.at(1)).Qvec().transpose() << ", " 
        //     << reconstruction_->LidarSweep(sweep_ids.at(1)).Tvec().transpose() << std::endl;
        const ceres::Solver::Summary summary = bundle_adjuster.Summary();
        if (summary.termination_type == ceres::CONVERGENCE) {
            num_iter_converged++;
        }
        double ba_initial_cost = std::sqrt(summary.initial_cost / summary.num_residuals_reduced);
        double ba_final_cost = std::sqrt(summary.final_cost / summary.num_residuals_reduced);
        if (i == 0) initial_cost = ba_initial_cost;
        if (ba_final_cost > initial_cost) {
            num_iter_failed++;
        }
        // if (num_iter_failed > 2) {
        //     std::cout << "sove f2f frame failed" << std::endl;
        //     return false;
        // }
        initial_cost = ba_initial_cost;
    }
    final_cost = initial_cost;
    std::cout << "AdjustFrame2FrameBundle Done" << std::endl;
    return true;
}

bool IncrementalMapper::RefineSceneScale(const Options& options) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());

    clock_t start_time = clock();

    std::vector<double> scales;
    // std::vector<Eigen::Vector3d> points1, points2;
    // std::unordered_set<mappoint_t> overlap_mappoint_ids;
    const auto& image_ids = reconstruction_->RegisterImageIds();
    for (const auto& image_id : image_ids) {
        class Image& image = reconstruction_->Image(image_id);
        class Camera& camera = reconstruction_->Camera(image.CameraId());

        if (camera.NumLocalCameras() <= 1) {
            continue;
        }

        const auto& local_image_indices = image.LocalImageIndices();

        Eigen::Vector4d qvec = image.Qvec();
        Eigen::Vector3d tvec = image.Tvec();

        std::vector<Eigen::Matrix3x4d> proj_matrixs;
        for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
            Eigen::Vector4d local_qvec(camera.LocalQvecsData() + 4 * i);
            Eigen::Vector3d local_tvec(camera.LocalTvecsData() + 3 * i);
            Eigen::Matrix3x4d local_proj_matrix;
            local_proj_matrix = ComposeProjectionMatrix(local_qvec, local_tvec);
            proj_matrixs.emplace_back(local_proj_matrix);
        }

        FeatureMatches matches =
        scene_graph_container_->CorrespondenceGraph()->FindCorrespondencesBetweenImages(image_id, image_id);

        // std::string filename1 = StringPrintf("image%04d-1.obj", image_id);
        // std::string filename2 = StringPrintf("image%04d-2.obj", image_id);
        // FILE *fp1 = fopen(filename1.c_str(), "w");
        // FILE *fp2 = fopen(filename2.c_str(), "w");
        for (const auto & match : matches) {
            point2D_t point2D_idx1 = match.point2D_idx1;
            point2D_t point2D_idx2 = match.point2D_idx2;
            class Point2D& point2D1 = image.Point2D(point2D_idx1);
            class Point2D& point2D2 = image.Point2D(point2D_idx2);
            if (!point2D1.HasMapPoint() && !point2D2.HasMapPoint()) {
                continue;
            }
            if (point2D1.MapPointId() != point2D2.MapPointId()) {
                continue;
            }
            uint32_t local_camera_id1 = local_image_indices[point2D_idx1];
            uint32_t local_camera_id2 = local_image_indices[point2D_idx2];
            if (local_camera_id1 == local_camera_id2) {
                continue;
            }

            const Eigen::Vector2d point1_N = camera.LocalImageToWorld(local_camera_id1, point2D1.XY());
            const Eigen::Vector2d point2_N = camera.LocalImageToWorld(local_camera_id2, point2D2.XY());

            Eigen::Matrix3x4d proj_matrix1 = proj_matrixs.at(local_camera_id1);
            Eigen::Matrix3x4d proj_matrix2 = proj_matrixs.at(local_camera_id2);

            const Eigen::Vector3d& lX =TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);

            double d1 = proj_matrix1.row(2).dot(lX.homogeneous());
            double d2 = proj_matrix2.row(2).dot(lX.homogeneous());
            if (d1 > 1 || d2 > 1 || d1 <= 0 || d2 <= 0) {
                continue;
            }

            // Eigen::Vector3d hpoint1_N = point1_N.homogeneous().normalized();
            // double angle1 = std::acos(hpoint1_N.z()) / M_PI * 180.0;
            // Eigen::Vector3d hpoint2_N = point2_N.homogeneous().normalized();
            // double angle2 = std::acos(hpoint2_N.z()) / M_PI * 180.0;
            // if (angle1 > 45 || angle2 > 45) {
            //     continue;
            // }

            const Eigen::Vector3d gX = reconstruction_->MapPoint(point2D1.MapPointId()).XYZ();
            const Eigen::Vector3d gXc = QuaternionToRotationMatrix(qvec) * gX + tvec;
            if (gXc[2] > 1) {
                continue;
            }

            // overlap_mappoint_ids.insert(point2D1.MapPointId());
            // points1.emplace_back(lX);
            // points2.emplace_back(gXc);

            const double scale = lX.norm() / gXc.norm();
            scales.push_back(scale);

            // fprintf(fp1, "v %f %f %f\n", lX[0], lX[1], lX[2]);
            // fprintf(fp2, "v %f %f %f\n", gXc[0], gXc[1], gXc[2]);
        }
        // fclose(fp1);
        // fclose(fp2);
    }

    if (scales.size() <= 100) {
        std::cout << StringPrintf("too low observations(%d) that close to camera", scales.size()) << std::endl;
        return false;
    }

    double m_scale = 0.0;
    for (auto scale : scales) {
        m_scale += scale;
    }
    m_scale /= scales.size();
    double stdev = 0.0;
    for (auto scale : scales) {
        stdev += (scale - m_scale) * (scale - m_scale);
    }
    stdev = std::sqrt(stdev / scales.size());

    // {
    //     std::vector<int> bins(300, 0);
    //     for (double scale : scales) {
    //         int bin_id = std::min(scale * 100.0, 299.0);
    //         bins[bin_id]++;
    //     }
    //     FILE *file = fopen("scales.txt", "w");
    //     for (int i = 0; i < 299; ++i) {
    //         fprintf(file, "%d\n", bins[i]);
    //     }
    //     fclose(file);

    //     FILE *file1 = fopen("points_est.obj", "w");
    //     for (auto point : points1) {
    //         fprintf(file1, "v %f %f %f\n", point.x(), point.y(), point.z());
    //     }
    //     fclose(file1);
    //     FILE *file2 = fopen("points_sfm.obj", "w");
    //     for (auto point : points2) {
    //         fprintf(file2, "v %f %f %f\n", point.x(), point.y(), point.z());
    //     }
    //     fclose(file2);

    //     FILE *file3 = fopen("points_g.obj", "w");
    //     for (auto mappoint_id : overlap_mappoint_ids) {
    //         class MapPoint& mappoint = reconstruction_->MapPoint(mappoint_id);
    //         Eigen::Vector3ub color = mappoint.Color();
    //         fprintf(file3, "v %f %f %f %d %d %d\n", mappoint.X(), mappoint.Y(), mappoint.Z(), (int)color.x(), (int)color.y(), (int)color.z());
    //     }
    //     fclose(file3);
    // }

    size_t nth = scales.size() / 2;
    std::nth_element(scales.begin(), scales.begin() + nth, scales.end());

    std::cout << "reconstruction mean scale: " << m_scale << std::endl;
    std::cout << "reconstruction median scale: " << scales.at(nth) << std::endl;
    std::cout << "Scale info: " << stdev << std::endl;
    reconstruction_->Rescale(scales.at(nth));

    return true;
}

size_t IncrementalMapper::FilterImages(const Options& options) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());

    // Do not filter images in the early stage of the reconstruction, since the
    // calibration is often still refining a lot. Hence, the camera parameters
    // are not stable in the beginning.
    const size_t kMinNumImages = 20;
    if (reconstruction_->NumRegisterImages() < kMinNumImages) {
        return {};
    }

    const std::vector<image_t> image_ids = reconstruction_->FilterImages(
        options.min_focal_length_ratio, options.max_focal_length_ratio, options.max_extra_param);

    for (const image_t image_id : image_ids) {
        DeRegisterImageEvent(image_id);
        filtered_images_.insert(image_id);
    }
    if (image_ids.size() > 0) {
        std::cout << "  => Filtered image ids: ";
        for (const auto image_id : image_ids) {
            std::cout << " " << image_id;
        }
        std::cout << std::endl;
    }
    return image_ids.size();
}

size_t IncrementalMapper::FilterImages(const Options& options, const std::unordered_set<image_t>& addressed_images) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());

    // Do not filter images in the early stage of the reconstruction, since the
    // calibration is often still refining a lot. Hence, the camera parameters
    // are not stable in the beginning.
    const size_t kMinNumImages = 20;
    if (reconstruction_->NumRegisterImages() < kMinNumImages) {
        return {};
    }

    const std::vector<image_t> image_ids = reconstruction_->FilterImages(
        options.min_focal_length_ratio, options.max_focal_length_ratio, options.max_extra_param,addressed_images);

    for (const image_t image_id : image_ids) {
        DeRegisterImageEvent(image_id);
        filtered_images_.insert(image_id);
    }
    if(image_ids.size()>0){
        std::cout<<"  => Filtered image ids: ";
        for(const auto image_id: image_ids){
            std::cout<<" "<<image_id;
        }
        std::cout<<std::endl;
    }
    return image_ids.size();
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());
    return reconstruction_->FilterAllMapPoints(2, options.filter_max_reproj_error, options.filter_min_tri_angle);
}

size_t IncrementalMapper::FilterPoints(const Options& options, int min_track_length) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());
    return reconstruction_->FilterAllMapPoints(min_track_length, options.filter_max_reproj_error,
                                               options.filter_min_tri_angle);
}

size_t IncrementalMapper::FilterPoints(const Options& options,const std::unordered_set<mappoint_t>& addressed_points) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());
    return reconstruction_->FilterMapPoints(2, options.filter_max_reproj_error, options.filter_min_tri_angle,
        addressed_points);
}


size_t IncrementalMapper::FilterPointsFinal(const Options& options) {
    CHECK_NOTNULL(reconstruction_.get());
    CHECK(options.Check());
    std::cout << "Filter final params: " << options.filter_max_reproj_error_final << " "
              << options.filter_min_track_length_final << " " << options.filter_min_tri_angle_final << std::endl;
    return reconstruction_->FilterAllMapPoints(options.filter_min_track_length_final,
                                               options.filter_max_reproj_error_final,
                                               options.filter_min_tri_angle_final);
}

const Reconstruction& IncrementalMapper::GetReconstruction() const {
    CHECK_NOTNULL(reconstruction_.get());
    return *reconstruction_.get();
}

size_t IncrementalMapper::NumTotalRegImages() const { return num_total_reg_images_; }

size_t IncrementalMapper::NumSharedRegImages() const { return num_shared_reg_images_; }

const std::unordered_set<mappoint_t>& IncrementalMapper::GetModifiedMapPoints() {
    return triangulator_->GetModifiedMapPoints();
}

void IncrementalMapper::AddModifiedMapPoint(const mappoint_t mappoint_id) {
    triangulator_->AddModifiedMapPoint(mappoint_id);
}

void IncrementalMapper::ClearModifiedMapPoints() { triangulator_->ClearModifiedMapPoints(); }

std::vector<image_t> IncrementalMapper::FindFirstInitialImage(const Options& options) const {
    // Struct to hold meta-data for ranking images.
    struct ImageInfo {
        image_t image_id;
        bool prior_focal_length;
        image_t num_correspondences;
    };

    const size_t init_max_reg_trials = static_cast<size_t>(options.init_max_reg_trials);

    // Collect information of all not yet registered images with
    // correspondences.
    std::vector<ImageInfo> image_infos;
    image_infos.reserve(reconstruction_->NumImages());
    for (const auto& image : reconstruction_->Images()) {
        // Only images with correspondences can be registered.
        if (image.second.NumCorrespondences() == 0) {
            continue;
        }

        // Only use images for initialization a maximum number of times.
        if (init_num_reg_trials_.count(image.first) && init_num_reg_trials_.at(image.first) >= init_max_reg_trials) {
            continue;
        }

        // Only use images for initialization that are not registered in any
        // of the other reconstructions.
        if (num_registrations_.count(image.first) > 0 && 
            num_registrations_.at(image.first) > 0) {   
            continue;
        }

        const class Camera& camera = reconstruction_->Camera(image.second.CameraId());
        ImageInfo image_info;
        image_info.image_id = image.first;
        image_info.prior_focal_length = camera.HasPriorFocalLength();
        image_info.num_correspondences = image.second.NumCorrespondences();
        image_infos.push_back(image_info);
    }

    // Sort images such that images with a prior focal length and more
    // correspondences are preferred, i.e. they appear in the front of the list.
    std::sort(image_infos.begin(), image_infos.end(), [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
            return true;
        } else if (!image_info1.prior_focal_length && image_info2.prior_focal_length) {
            return false;
        } else {
            return image_info1.num_correspondences > image_info2.num_correspondences;
        }
    });

    // Extract image identifiers in sorted order.
    std::vector<image_t> image_ids;
    image_ids.reserve(image_infos.size());
    for (const ImageInfo& image_info : image_infos) {
        image_ids.push_back(image_info.image_id);
    }

    return image_ids;
}

std::vector<image_t> IncrementalMapper::FindSecondInitialImage(const Options& options, const image_t image_id1) const {
    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    // Collect images that are connected to the first seed image and have
    // not been registered before in other reconstructions.
    const class Image& image1 = reconstruction_->Image(image_id1);
    std::unordered_map<image_t, point2D_t> num_correspondences;
    std::unordered_map<image_t, std::vector<float>> image_disparitys;
    for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D(); ++point2D_idx) {
        const class Point2D& point2D1 = image1.Point2D(point2D_idx);
        for (const auto& corr : correspondence_graph->FindCorrespondences(image_id1, point2D_idx)) {
            if (num_registrations_.count(corr.image_id) == 0 || 
                num_registrations_.at(corr.image_id) == 0) {
                num_correspondences[corr.image_id] += 1;

                const class Image& corr_image = reconstruction_->Image(corr.image_id);
                const class Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
                float square_disparity = (point2D1.XY() - corr_point2D.XY()).squaredNorm();
                image_disparitys[corr.image_id].push_back(square_disparity);
            }
        }
    }

    // Struct to hold meta-data for ranking images.
    struct ImageInfo {
        image_t image_id;
        bool prior_focal_length;
        point2D_t num_correspondences;
    };

    const size_t init_min_num_inliers = static_cast<size_t>(options.init_min_num_inliers);
    const double init_min_disparity_squared = options.init_min_disparity * options.init_min_disparity;

    // Compose image information in a compact form for sorting.
    std::vector<ImageInfo> image_infos;
    image_infos.reserve(reconstruction_->NumImages());
    for (const auto elem : num_correspondences) {
        if (elem.second >= init_min_num_inliers) {
            double disparity = Median(image_disparitys[elem.first]);
            if (disparity < init_min_disparity_squared) {
                continue;
            }

            const class Image& image = reconstruction_->Image(elem.first);
            const class Camera& camera = reconstruction_->Camera(image.CameraId());
            ImageInfo image_info;
            image_info.image_id = elem.first;
            image_info.prior_focal_length = camera.HasPriorFocalLength();
            image_info.num_correspondences = elem.second;
            image_infos.push_back(image_info);
        }
    }

    // Sort images such that images with a prior focal length, greater
    // density and more correspondences are preferred, i.e. they
    // appear in the front of the list.
    std::sort(image_infos.begin(), image_infos.end(), [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
            return true;
        } else if (!image_info1.prior_focal_length && image_info2.prior_focal_length) {
            return false;
        }
        else {
            return image_info1.num_correspondences > image_info2.num_correspondences;
        }
    });

    // Extract image identifiers in sorted order.
    std::vector<image_t> image_ids;
    image_ids.reserve(image_infos.size());
    for (const ImageInfo& image_info : image_infos) {
        image_ids.push_back(image_info.image_id);
    }

    return image_ids;
}

std::vector<image_t> IncrementalMapper::FindLocalBundle(const Options& options, const image_t image_id) const {
    CHECK(options.Check());

    const Image& image = reconstruction_->Image(image_id);
    CHECK(image.IsRegistered());

    // Extract all images that have at least one Map Point with the query image
    // in common, and simultaneously count the number of common Map Points.
    std::unordered_map<image_t, size_t> shared_observations;

    std::unordered_set<mappoint_t> mappoint_ids;
    mappoint_ids.reserve(image.NumMapPoints());

    for (const Point2D& point2D : image.Points2D()) {
        if (point2D.HasMapPoint()) {
            mappoint_ids.insert(point2D.MapPointId());
            const MapPoint& mappoint = reconstruction_->MapPoint(point2D.MapPointId());
            for (const TrackElement& track_el : mappoint.Track().Elements()) {
                if (track_el.image_id != image_id) {
                    shared_observations[track_el.image_id] += 1;
                }
            }
        }
    }

    // Sort overlapping images according to number of shared observations.
    std::vector<std::pair<image_t, size_t>> overlapping_images(shared_observations.begin(), shared_observations.end());
    std::sort(overlapping_images.begin(), overlapping_images.end(),
              [](const std::pair<image_t, size_t>& image1, const std::pair<image_t, size_t>& image2) {
                  return image1.second > image2.second;
              });

    // The local bundle is composed of the given image and its most connected
    // neighbor images, hence the subtraction of 1.

    const size_t num_images = static_cast<size_t>(options.local_ba_num_images - 1);
    const size_t num_eff_images = std::min(num_images, overlapping_images.size());

    // Extract most connected images and ensure sufficient triangulation angle.

    std::vector<image_t> local_bundle_image_ids;
    local_bundle_image_ids.reserve(num_eff_images);

    // If the number of overlapping images equals the number of desired images in
    // the local bundle, then simply copy over the image identifiers.
    if (overlapping_images.size() == num_eff_images) {
        for (const auto& overlapping_image : overlapping_images) {
            local_bundle_image_ids.push_back(overlapping_image.first);
        }
        return local_bundle_image_ids;
    }

    // In the following iteration, we start with the most overlapping images and
    // check whether it has sufficient triangulation angle. If none of the
    // overlapping images has sufficient triangulation angle, we relax the
    // triangulation angle threshold and start from the most overlapping image
    // again. In the end, if we still haven't found enough images, we simply use
    // the most overlapping images.

    const double min_tri_angle_rad = DegToRad(options.local_ba_min_tri_angle);

    // The selection thresholds (minimum triangulation angle, minimum number of
    // shared observations), which are successively relaxed.
    const std::array<std::pair<double, double>, 8> selection_thresholds = {{
        std::make_pair(min_tri_angle_rad / 1.0, 0.6 * image.NumMapPoints()),
        std::make_pair(min_tri_angle_rad / 1.5, 0.6 * image.NumMapPoints()),
        std::make_pair(min_tri_angle_rad / 2.0, 0.5 * image.NumMapPoints()),
        std::make_pair(min_tri_angle_rad / 2.5, 0.4 * image.NumMapPoints()),
        std::make_pair(min_tri_angle_rad / 3.0, 0.3 * image.NumMapPoints()),
        std::make_pair(min_tri_angle_rad / 4.0, 0.2 * image.NumMapPoints()),
        std::make_pair(min_tri_angle_rad / 5.0, 0.1 * image.NumMapPoints()),
        std::make_pair(min_tri_angle_rad / 6.0, 0.1 * image.NumMapPoints()),
    }};

    const Eigen::Vector3d proj_center = image.ProjectionCenter();
    std::vector<Eigen::Vector3d> shared_points3D;
    shared_points3D.reserve(image.NumMapPoints());
    std::vector<double> tri_angles(overlapping_images.size(), -1.0);
    std::vector<char> used_overlapping_images(overlapping_images.size(), false);

    for (const auto& selection_threshold : selection_thresholds) {
        for (size_t overlapping_image_idx = 0; overlapping_image_idx < overlapping_images.size();
             ++overlapping_image_idx) {
            // Check if the image has sufficient overlap. Since the images are ordered
            // based on the overlap, we can just skip the remaining ones.
            if (overlapping_images[overlapping_image_idx].second < selection_threshold.second) {
                break;
            }

            // Check if the image is already in the local bundle.
            if (used_overlapping_images[overlapping_image_idx]) {
                continue;
            }

            const auto& overlapping_image = reconstruction_->Image(overlapping_images[overlapping_image_idx].first);
            const Eigen::Vector3d overlapping_proj_center = overlapping_image.ProjectionCenter();

            // In the first iteration, compute the triangulation angle. In later
            // iterations, reuse the previously computed value.
            double& tri_angle = tri_angles[overlapping_image_idx];
            if (tri_angle < 0.0) {
                // Collect the commonly observed Map Points.
                shared_points3D.clear();
                for (const Point2D& point2D : image.Points2D()) {
                    if (point2D.HasMapPoint() && mappoint_ids.count(point2D.MapPointId())) {
                        shared_points3D.push_back(reconstruction_->MapPoint(point2D.MapPointId()).XYZ());
                    }
                }

                // Calculate the triangulation angle at a certain percentile.
                const double kTriangulationAnglePercentile = 75;
                tri_angle =
                    Percentile(CalculateTriangulationAngles(proj_center, overlapping_proj_center, shared_points3D),
                               kTriangulationAnglePercentile);
            }

            // Check that the image has sufficient triangulation angle.
            if (tri_angle >= selection_threshold.first) {
                local_bundle_image_ids.push_back(overlapping_image.ImageId());
                used_overlapping_images[overlapping_image_idx] = true;
                // Check if we already collected enough images.
                if (local_bundle_image_ids.size() >= num_eff_images) {
                    break;
                }
            }
        }

        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
            break;
        }
    }

    // In case there are not enough images with sufficient triangulation angle,
    // simply fill up the rest with the most overlapping images.

    if (local_bundle_image_ids.size() < num_eff_images) {
        for (size_t overlapping_image_idx = 0; overlapping_image_idx < overlapping_images.size();
             ++overlapping_image_idx) {
            // Collect image if it is not yet in the local bundle.
            if (!used_overlapping_images[overlapping_image_idx]) {
                local_bundle_image_ids.push_back(overlapping_images[overlapping_image_idx].first);
                used_overlapping_images[overlapping_image_idx] = true;

                // Check if we already collected enough images.
                if (local_bundle_image_ids.size() >= num_eff_images) {
                    break;
                }
            }
        }
    }

    return local_bundle_image_ids;
}

void IncrementalMapper::RegisterImageEvent(const image_t image_id) {
    size_t& num_regs_for_image = num_registrations_[image_id];
    num_regs_for_image += 1;
    if (num_regs_for_image == 1) {
        num_total_reg_images_ += 1;
    } else if (num_regs_for_image > 1) {
        num_shared_reg_images_ += 1;
    }
}

void IncrementalMapper::DeRegisterImageEvent(const image_t image_id) {
    size_t& num_regs_for_image = num_registrations_[image_id];
    num_regs_for_image -= 1;
    if (num_regs_for_image == 0) {
        num_total_reg_images_ -= 1;
    } else if (num_regs_for_image > 0) {
        num_shared_reg_images_ -= 1;
    }
}

void IncrementalMapper::DecreaseNumRegTrials(const image_t image_id) {
    CHECK(num_reg_trials_.find(image_id) != num_reg_trials_.end());
    num_reg_trials_[image_id] -= 1;
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(const Options& options, const image_t image_id1,
                                                       const image_t image_id2) {
    const image_pair_t image_pair_id = sensemap::utility::ImagePairToPairId(image_id1, image_id2);

    if (prev_init_image_pair_id_ == image_pair_id) {
        return true;
    }

    const Image& image1 = scene_graph_container_->Image(image_id1);
    const Camera& camera1 = scene_graph_container_->Camera(image1.CameraId());
    const int num_local_camera1 = camera1.NumLocalCameras();

    const Image& image2 = scene_graph_container_->Image(image_id2);
    const Camera& camera2 = scene_graph_container_->Camera(image2.CameraId());
    const int num_local_camera2 = camera2.NumLocalCameras();

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();
    const FeatureMatches matches = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);

    std::vector<Eigen::Vector2d> points1;
    points1.reserve(image1.NumPoints2D());
    for (const auto& point : image1.Points2D()) {
        points1.push_back(point.XY());
    }

    std::vector<Eigen::Vector2d> points2;
    points2.reserve(image2.NumPoints2D());
    for (const auto& point : image2.Points2D()) {
        points2.push_back(point.XY());
    }

    TwoViewGeometry two_view_geometry;
    TwoViewGeometry::Options two_view_geometry_options;
    two_view_geometry_options.max_error = options.init_max_error;
    two_view_geometry_options.max_angle_error = options.init_max_angular_error;
    
    two_view_geometry_options.ransac_options.confidence = 0.9999;
    two_view_geometry_options.ransac_options.min_num_trials = 30;
    two_view_geometry_options.ransac_options.max_error = options.init_max_error;
    two_view_geometry_options.ransac_options.min_inlier_ratio_to_best_model = options.min_inlier_ratio_to_best_model;
    two_view_geometry_options.is_sphere = !options.lidar_sfm;

    if (num_local_camera1 <= 1 && num_local_camera2 <= 1) {
        if (camera1.ModelName().compare("SPHERICAL") == 0 || camera2.ModelName().compare("SPHERICAL") == 0) {
            if(camera1.ModelName().compare("SPHERICAL") == 0 && camera2.ModelName().compare("SPHERICAL") == 0)
                two_view_geometry.EstimateSpherical(camera1, points1, camera2, points2, matches, two_view_geometry_options);
            else
                two_view_geometry.EstimatePespectiveAndSpherical(camera1, points1, camera2, points2, matches, two_view_geometry_options);
        } else {
            two_view_geometry.EstimateCalibrated(camera1, points1, camera2, points2, matches, two_view_geometry_options);
        }
    } else if (num_local_camera1 > 1 && num_local_camera2 > 1) {
        two_view_geometry_options.ransac_options.max_error = options.init_max_angular_error;

        std::vector<uint32_t> local_image_indices1 = image1.LocalImageIndices();
        std::vector<uint32_t> local_image_indices2 = image2.LocalImageIndices();

        two_view_geometry.EstimateCalibratedRig(camera1, points1, camera2, points2, matches, local_image_indices1,
                                                local_image_indices2, two_view_geometry_options);
    } else {
        two_view_geometry_options.ransac_options.max_error = options.init_max_error;

        std::vector<uint32_t> local_image_indices1 = image1.LocalImageIndices();
        std::vector<uint32_t> local_image_indices2 = image2.LocalImageIndices();

        two_view_geometry.EstimateOneAndRig(camera1, points1, camera2, points2, matches, 
                                            local_image_indices1, local_image_indices2,
                                            two_view_geometry_options);
    }

    std::vector<double> tri_angles;

    if (num_local_camera1 <= 1 && num_local_camera2 <= 1) {
        if (!two_view_geometry.EstimateRelativePose(camera1, points1, camera2, points2, two_view_geometry_options,
                                                    &tri_angles, true)) {
            return false;
        }
    } else if (num_local_camera1 > 1 && num_local_camera2 > 1) {
        two_view_geometry_options.ransac_options.max_error = options.init_max_angular_error;

        CHECK_EQ(num_local_camera1, num_local_camera2);
        std::vector<uint32_t> local_image_indices1 = image1.LocalImageIndices();
        std::vector<uint32_t> local_image_indices2 = image2.LocalImageIndices();

        if (!two_view_geometry.EstimateRelativePoseRigGV(camera1, points1, camera2, points2, local_image_indices1,
                                                         local_image_indices2, two_view_geometry_options, &tri_angles,
                                                         true)) {
            return false;
        }
    } else {
        std::vector<uint32_t> local_image_indices1 = image1.LocalImageIndices();
        std::vector<uint32_t> local_image_indices2 = image2.LocalImageIndices();

        if (!two_view_geometry.EstimateRelativePoseOneAndRig(camera1, points1, camera2, points2, local_image_indices1,
                                                             local_image_indices2, &tri_angles, true)) {
            return false;
        }
    }

    if (num_local_camera1 > 1 || num_local_camera2 > 1) {
        std::cout << "Two view geometry infomation: " << tri_angles.size() << " "
                  << RadToDeg(two_view_geometry.tri_angle) << " " << two_view_geometry.tvec_rig.x() << " "
                  << two_view_geometry.tvec_rig.y() << " " << two_view_geometry.tvec_rig.z() << std::endl;

        if (static_cast<int>(tri_angles.size()) >= options.init_min_num_inliers &&
            two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
            prev_init_image_pair_id_ = image_pair_id;
            prev_init_two_view_geometry_ = two_view_geometry;

            if (options.sub_matching && options.self_matching && num_local_camera1 > 2 && num_local_camera2 > 2) {
                Eigen::Vector4d relative_qvec;
                Eigen::Vector3d relative_tvec;
                if (EstimateRelativePoseRig(options, image_id1, image_id2, relative_qvec, relative_tvec)) {
                    Eigen::Vector4d qvec_rig = prev_init_two_view_geometry_.qvec_rig;
                    Eigen::Vector3d tvec_rig = prev_init_two_view_geometry_.tvec_rig;
                    Eigen::Matrix3d R_rig = QuaternionToRotationMatrix(qvec_rig);
                    Eigen::Matrix3d R_relative = QuaternionToRotationMatrix(relative_qvec);
                    Eigen::Matrix3d R_diff = R_relative.transpose() * R_rig;
                    Eigen::AngleAxisd angle_axis(R_diff);
                    double R_angle = angle_axis.angle();
                    // if (RadToDeg(R_angle) > 15) {
                    //     std::cout << "R from two view geometry and intr-view not consistent" << std::endl;
                    //     return false;
                    // }
                    std::cout << "angle diff: " << RadToDeg(R_angle) << std::endl;
                    std::cout << relative_qvec.transpose() << " " 
                            << relative_tvec.transpose() << std::endl;
                    std::cout << qvec_rig.transpose() << " " 
                            << tvec_rig.transpose() << std::endl;

                    const float t_scale = relative_tvec.norm() / tvec_rig.norm();
                    std::cout << "t_scale: " << t_scale << std::endl;
                    prev_init_two_view_geometry_.tvec_rig *= t_scale;
                }
            }

            return true;
        }
        return false;

    } else {
        int tri_angle_larger_than_min = 0;
        for (auto tri_angle : tri_angles) {
            if (tri_angle > DegToRad(options.init_min_tri_angle)) {
                tri_angle_larger_than_min++;
            }
        }
        std::cout << "Two view geometry infomation: " << two_view_geometry.inlier_matches.size() << " "
                  << RadToDeg(two_view_geometry.tri_angle) << " " << two_view_geometry.tvec.x() << " "
                  << two_view_geometry.tvec.y() << " " << two_view_geometry.tvec.z() << std::endl;
        std::cout << "Triangles: " << tri_angles.size() << " " << tri_angle_larger_than_min << std::endl;
        std::cout << "Tri_angle: " << RadToDeg(two_view_geometry.tri_angle) << std::endl;

        if (static_cast<int>(two_view_geometry.inlier_matches.size()) >= options.init_min_num_inliers &&
            two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
            prev_init_image_pair_id_ = image_pair_id;
            prev_init_two_view_geometry_ = two_view_geometry;
            return true;
        }
        return false;
    }
}

bool IncrementalMapper::EstimateRelativePoseBy3D(const Options& options, const image_t image_id1,
                                                 const image_t image_id2, Eigen::Vector4d& qvec,
                                                 Eigen::Vector3d& tvec) {
    const Image & image1 = reconstruction_->Image(image_id1);
    const Camera & camera1 = reconstruction_->Camera(image1.CameraId());

    const Image & image2 = reconstruction_->Image(image_id2);
    const Camera & camera2 = reconstruction_->Camera(image2.CameraId());

    const std::shared_ptr<CorrespondenceGraph> correspondence_graph = 
                                scene_graph_container_->CorrespondenceGraph();
    const FeatureMatches matches = 
        correspondence_graph->FindCorrespondencesBetweenImages(image_id1, 
                                                               image_id2);
    
    const std::vector<class Point2D>& points1 = image1.Points2D();
    const std::vector<class Point2D>& points2 = image2.Points2D();

    std::vector<Eigen::Vector3d> points3D1, points3D2;
    for (size_t i = 0; i < matches.size(); ++i) {
        const point2D_t idx1 = matches[i].point2D_idx1;
        const point2D_t idx2 = matches[i].point2D_idx2;
        Eigen::Vector2d XY1 = points1[idx1].XY();
        Eigen::Vector2d XY2 = points2[idx2].XY();
        const float depth1 = points1[idx1].Depth();
        const float depth2 = points2[idx2].Depth();
        if (depth1 > 0 && depth2 > 0) {
            Eigen::Vector2d point1_normalized = camera1.ImageToWorld(XY1);
            Eigen::Vector2d point2_normalized = camera2.ImageToWorld(XY2);
            Eigen::Vector3d hpoint3D1(point1_normalized[0] * depth1, point1_normalized[1] * depth1, depth1);
            Eigen::Vector3d hpoint3D2(point2_normalized[0] * depth2, point2_normalized[1] * depth2, depth2);
            points3D1.emplace_back(hpoint3D1);
            points3D2.emplace_back(hpoint3D2);
        }
    }
    const size_t rel_pose_min_num_corrs = 30;

    if (points3D1.size() <= rel_pose_min_num_corrs) {
        return false;
    }

    RANSACOptions ransac_options;
    ransac_options.max_error = 0.005;
    ransac_options.min_inlier_ratio = 0.3;
    LORANSAC<SimilarityTransformEstimator<3, 0>, SimilarityTransformEstimator<3, 0>> ransac(ransac_options);

    const auto report = ransac.Estimate(points3D1, points3D2);
    Eigen::Matrix3d rot = report.model.block<3, 3>(0, 0);
    tvec = report.model.block<3, 1>(0, 3);
    qvec = RotationMatrixToQuaternion(rot);
    return true;
}

bool IncrementalMapper::EstimateRelativePoseRig(const Options& options,
                                                const image_t image_id1,
                                                const image_t image_id2,
                                                Eigen::Vector4d& qvec,
                                                Eigen::Vector3d& tvec) {
#if 1
    const auto& correspondence_graph = scene_graph_container_->CorrespondenceGraph();
    const FeatureMatches& corrs_between_images = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);
    const FeatureMatches& corrs1 = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id1);
    const FeatureMatches& corrs2 = correspondence_graph->FindCorrespondencesBetweenImages(image_id2, image_id2);

    class Image& image1 = reconstruction_->Image(image_id1);
    class Image& image2 = reconstruction_->Image(image_id2);
    class Camera& camera1 = reconstruction_->Camera(image1.CameraId());
    class Camera& camera2 = reconstruction_->Camera(image2.CameraId());
    std::vector<uint32_t> local_image_indices1 = image1.LocalImageIndices();
    std::vector<uint32_t> local_image_indices2 = image2.LocalImageIndices();

    uint32_t local_camera_id1, local_camera_id2;
    FeatureMatches create_corrs1, create_corrs2;
    for (const FeatureMatch& corr : corrs_between_images) {
        auto it1 = std::find_if(corrs1.begin(), corrs1.end(), 
            [&](const FeatureMatch& corr1) {
                return corr.point2D_idx1 == corr1.point2D_idx1 ||
                        corr.point2D_idx1 == corr1.point2D_idx2;
            });
        auto it2 = std::find_if(corrs2.begin(), corrs2.end(), 
            [&](const FeatureMatch& corr2) {
                return corr.point2D_idx2 == corr2.point2D_idx1 ||
                        corr.point2D_idx2 == corr2.point2D_idx2;
            });
        if (it1 != corrs1.end() && it2 != corrs2.end()) {
            local_camera_id1 = local_image_indices1[it1->point2D_idx1];
            local_camera_id2 = local_image_indices1[it1->point2D_idx2];
            if (local_camera_id1 == local_camera_id2) {
                continue;
            }
            local_camera_id1 = local_image_indices2[it2->point2D_idx1];
            local_camera_id2 = local_image_indices2[it2->point2D_idx2];
            if (local_camera_id1 == local_camera_id2) {
                continue;
            }
            create_corrs1.emplace_back(*it1);
            create_corrs2.emplace_back(*it2);
        }
    }

    std::vector<Eigen::Matrix3x4d> proj_matrixs1;
    proj_matrixs1.resize(camera1.NumLocalCameras());
    for (size_t i = 0; i < camera1.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera1.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        proj_matrixs1[i] = ComposeProjectionMatrix(local_qvec, local_tvec);
    }
    std::vector<Eigen::Matrix3x4d> proj_matrixs2;
    proj_matrixs2.resize(camera2.NumLocalCameras());
    for (size_t i = 0; i < camera2.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera2.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        proj_matrixs2[i] = ComposeProjectionMatrix(local_qvec, local_tvec);
    }
    
    auto TriangulateCorrespondence = [&](const FeatureMatch& corr,
        const class Image& image, const class Camera& camera,
        const std::vector<Eigen::Matrix3x4d>& proj_matrixs,
        const std::vector<uint32_t>& local_image_indices, Eigen::Vector3d *point3D, double *d1, double *d2) {
        local_camera_id1 = local_image_indices[corr.point2D_idx1];
        local_camera_id2 = local_image_indices[corr.point2D_idx2];
        const class Point2D& point2D1 = image.Point2D(corr.point2D_idx1);
        const class Point2D& point2D2 = image.Point2D(corr.point2D_idx2);
        Eigen::Vector2d point1_N = camera.LocalImageToWorld(local_camera_id1, point2D1.XY());
        // Eigen::Vector3d hpoint1_N = point1_N.homogeneous().normalized();
        // double angle1 = std::acos(hpoint1_N.z()) / M_PI * 180.0;
        // if (angle1 < 45) {
        //     return false;
        // }

        Eigen::Vector2d point2_N = camera.LocalImageToWorld(local_camera_id2, point2D2.XY());
        // Eigen::Vector3d hpoint2_N = point1_N.homogeneous().normalized();
        // double angle2 = std::acos(hpoint2_N.z()) / M_PI * 180.0;
        // if (angle2 < 45) {
        //     return false;
        // }

        Eigen::Matrix3x4d proj_matrix1 = proj_matrixs.at(local_camera_id1);
        Eigen::Matrix3x4d proj_matrix2 = proj_matrixs.at(local_camera_id2);

        const Eigen::Vector3d& xyz = TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);

        *d1 = proj_matrix1.row(2).dot(xyz.homogeneous());
        *d2 = proj_matrix2.row(2).dot(xyz.homogeneous());

        if (*d1 > 0 && *d2 > 0) {
            *point3D = xyz;
            return true;
        }
        return false;
    };

    std::cout << StringPrintf("Find %d intra-view correspondence\n", create_corrs1.size());

    struct CorrPair {
        Eigen::Vector3d xyz1;
        Eigen::Vector3d xyz2;
    };

    std::map<int, std::vector<CorrPair> > corrs_pairs;

    for (size_t i = 0; i < create_corrs1.size(); ++i) {
        const FeatureMatch& corr1 = create_corrs1.at(i);
        const FeatureMatch& corr2 = create_corrs2.at(i);

        double d11, d12, d21, d22;
        Eigen::Vector3d xyz1, xyz2;
        bool b1 = TriangulateCorrespondence(corr1, image1, camera1, 
                    proj_matrixs1, local_image_indices1, &xyz1, &d11, &d12);
        bool b2 = TriangulateCorrespondence(corr2, image2, camera2, 
                    proj_matrixs2, local_image_indices2, &xyz2, &d21, &d22);
        if (b1 && b2) {
            double max_d1 = std::max(d11, d12);
            double max_d2 = std::max(d21, d22);
            double max_d = std::max(max_d1, max_d2) + 1;

            CorrPair corr_pair;
            corr_pair.xyz1 = xyz1;
            corr_pair.xyz2 = xyz2;
            corrs_pairs[max_d].emplace_back(corr_pair);
        }
    }

    int num_inlier = 0, best_depth_level = -1;
    std::vector<Eigen::Vector3d> points3D1, points3D2;

    for (auto & corrs_pair : corrs_pairs) {
        std::cout << StringPrintf("%d: %d, ", corrs_pair.first, corrs_pair.second.size());
        for (auto & corr_pair : corrs_pair.second) {
            points3D1.emplace_back(corr_pair.xyz1);
            points3D2.emplace_back(corr_pair.xyz2);
        }
        num_inlier += corrs_pair.second.size();
        if (num_inlier >= options.init_min_corrs_intra_view) {
            best_depth_level = corrs_pair.first;
            break;
        }
    }
    std::cout << std::endl;
    if (num_inlier < options.init_min_corrs_intra_view) {
        std::cout << StringPrintf("%d inliers, less than %d\n", num_inlier, options.init_min_corrs_intra_view);
        return false;
    }
    if (best_depth_level > options.init_max_depth) {
        std::cout << StringPrintf("depth level %d larger than %f\n", best_depth_level, options.init_max_depth);
        return false;
    }

    std::cout << StringPrintf("%d correspondences are triangulated\n", points3D1.size());

#if 0
    RANSACOptions ransac_options;
    ransac_options.max_error = 0.01;
    ransac_options.min_inlier_ratio = 0.3;
    LORANSAC<SimilarityTransformEstimator<3, 0>, SimilarityTransformEstimator<3, 0>> ransac(ransac_options);

    const auto report = ransac.Estimate(points3D1, points3D2);
    Eigen::Matrix3d rot = report.model.block<3, 3>(0, 0);
    tvec = report.model.block<3, 1>(0, 3);
    qvec = RotationMatrixToQuaternion(rot);
#else
    auto model = SimilarityTransformEstimator<3, 0>::Estimate(points3D1, points3D2).at(0);
    Eigen::Matrix3d rot = model.block<3, 3>(0, 0);
    tvec = model.block<3, 1>(0, 3);
    qvec = RotationMatrixToQuaternion(rot);
#endif

    // {
    //     FILE *fp1 = fopen("points3D1.obj", "w");
    //     FILE *fp1_align = fopen("points3D1_aligned.obj", "w");
    //     for (auto point3D : points3D1) {
    //         fprintf(fp1, "v %f %f %f\n", point3D.x(), point3D.y(), point3D.z());
    //         auto point3D_aligned = rot * point3D + tvec;
    //         fprintf(fp1_align, "v %f %f %f\n", point3D_aligned.x(), point3D_aligned.y(), point3D_aligned.z());
    //     }
    //     fclose(fp1);
    //     fclose(fp1_align);
    //     FILE *fp2 = fopen("points3D2.obj", "w");
    //     for (auto point3D : points3D2) {
    //         fprintf(fp2, "v %f %f %f\n", point3D.x(), point3D.y(), point3D.z());
    //     }
    //     fclose(fp2);
    // }
#else
    const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);
    double max_residuals = options.init_max_error * options.init_max_error;

    const auto& correspondence_graph = scene_graph_container_->CorrespondenceGraph();
    const FeatureMatches& corrs_between_images = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);
    const FeatureMatches& corrs1 = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id1);
    const FeatureMatches& corrs2 = correspondence_graph->FindCorrespondencesBetweenImages(image_id2, image_id2);

    class Image& image1 = reconstruction_->Image(image_id1);
    class Image& image2 = reconstruction_->Image(image_id2);
    class Camera& camera1 = reconstruction_->Camera(image1.CameraId());
    class Camera& camera2 = reconstruction_->Camera(image2.CameraId());
    std::vector<uint32_t> local_image_indices1 = image1.LocalImageIndices();
    std::vector<uint32_t> local_image_indices2 = image2.LocalImageIndices();
    uint32_t num_local_camera1 = camera1.NumLocalCameras();
    uint32_t num_local_camera2 = camera2.NumLocalCameras();

    FeatureMatches create_corrs1, create_corrs2;
    for (const FeatureMatch& corr : corrs_between_images) {
        auto it1 = std::find_if(corrs1.begin(), corrs1.end(), 
            [&](const FeatureMatch& corr1) {
                return corr.point2D_idx1 == corr1.point2D_idx1 ||
                        corr.point2D_idx1 == corr1.point2D_idx2;
            });
        auto it2 = std::find_if(corrs2.begin(), corrs2.end(), 
            [&](const FeatureMatch& corr2) {
                return corr.point2D_idx2 == corr2.point2D_idx1 ||
                        corr.point2D_idx2 == corr2.point2D_idx2;
            });
        if (it1 != corrs1.end() && it2 != corrs2.end()) {
            uint32_t local_camera_id1 = local_image_indices1[it1->point2D_idx1];
            uint32_t local_camera_id2 = local_image_indices1[it1->point2D_idx2];
            if (local_camera_id1 == local_camera_id2) {
                continue;
            }
            local_camera_id1 = local_image_indices2[it2->point2D_idx1];
            local_camera_id2 = local_image_indices2[it2->point2D_idx2];
            if (local_camera_id1 == local_camera_id2) {
                continue;
            }
            create_corrs1.emplace_back(*it1);
            create_corrs2.emplace_back(*it2);
        }
    }

    std::vector<Eigen::Matrix3x4d> proj_matrixs1;
    std::vector<Eigen::Vector3d> proj_centers1;
    proj_matrixs1.resize(camera1.NumLocalCameras());
    proj_centers1.resize(camera1.NumLocalCameras());
    for (size_t i = 0; i < camera1.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera1.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        
        const Eigen::Matrix3d local_camera_R =
            QuaternionToRotationMatrix(local_qvec) * QuaternionToRotationMatrix(image1.Qvec());
        const Eigen::Vector3d local_camera_T =
            QuaternionToRotationMatrix(local_qvec) * image1.Tvec() + local_tvec;

        proj_matrixs1[i] = ComposeProjectionMatrix(local_camera_R, local_camera_T);
        proj_centers1[i] = -local_camera_R.transpose() * local_camera_T;
    }
    std::vector<Eigen::Matrix3x4d> proj_matrixs2;
    std::vector<Eigen::Vector3d> proj_centers2;
    proj_matrixs2.resize(camera2.NumLocalCameras());
    proj_centers2.resize(camera2.NumLocalCameras());
    for (size_t i = 0; i < camera2.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera2.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);

        const Eigen::Matrix3d local_camera_R =
            QuaternionToRotationMatrix(local_qvec) * QuaternionToRotationMatrix(image2.Qvec());
        const Eigen::Vector3d local_camera_T =
            QuaternionToRotationMatrix(local_qvec) * image2.Tvec() + local_tvec;

        proj_matrixs2[i] = ComposeProjectionMatrix(local_camera_R, local_camera_T);
        proj_centers2[i] = -local_camera_R.transpose() * local_camera_T;
    }
    
    auto TriangulateTwoViewCorrespondence = [&](const point2D_t& point2D_idx1,
        const point2D_t& point2D_idx2, Eigen::Vector3d* X) {
        uint32_t local_camera_id1 = local_image_indices1[point2D_idx1];
        uint32_t local_camera_id2 = local_image_indices2[point2D_idx2];
        const class Point2D& point2D1 = image1.Point2D(point2D_idx1);
        const class Point2D& point2D2 = image2.Point2D(point2D_idx2);
        Eigen::Vector2d point1_N = camera1.LocalImageToWorld(local_camera_id1, point2D1.XY());
        Eigen::Vector2d point2_N = camera2.LocalImageToWorld(local_camera_id2, point2D2.XY());

        Eigen::Matrix3x4d proj_matrix1 = proj_matrixs1.at(local_camera_id1);
        Eigen::Matrix3x4d proj_matrix2 = proj_matrixs2.at(local_camera_id2);
        Eigen::Vector3d proj_center1 = proj_centers1.at(local_camera_id1);
        Eigen::Vector3d proj_center2 = proj_centers2.at(local_camera_id2);

        const Eigen::Vector3d& xyz = TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);

        const double tri_angle = CalculateTriangulationAngle(proj_center1, proj_center2, xyz);

        double error1 = CalculateSquaredReprojectionErrorRig(point2D1.XY(), xyz,
            proj_matrix1, local_camera_id1, camera1);
        double error2 = CalculateSquaredReprojectionErrorRig(point2D2.XY(), xyz,
            proj_matrix2, local_camera_id2, camera2);

        // std::cout << "error: " << tri_angle << ", " << error1 << ", " << error2 << std::endl;

        if (tri_angle >= min_tri_angle_rad &&
            error1 < max_residuals && error2 < max_residuals &&
            HasPointPositiveDepth(proj_matrix1, xyz) &&
            HasPointPositiveDepth(proj_matrix2, xyz)) {
            *X = xyz;
            return true;
        }
        return false;
    };

    std::cout << StringPrintf("Find %d intra-view correspondence\n", create_corrs1.size());

    size_t num_triangulated = 0;
    std::unordered_map<uint32_t, std::vector<Eigen::Vector3d> > corrs3D1;
    std::unordered_map<uint32_t, std::vector<Eigen::Vector3d> > corrs3D2;
    for (size_t i = 0; i < create_corrs1.size(); ++i) {
        const FeatureMatch& corr1 = create_corrs1.at(i);
        const FeatureMatch& corr2 = create_corrs2.at(i);

        uint32_t local_camera_id1 = local_image_indices1[corr1.point2D_idx1];
        uint32_t local_camera_id2 = local_image_indices1[corr1.point2D_idx2];
        if (local_camera_id1 == local_camera_id2) {
            continue;
        }

        Eigen::Vector3d xyz1, xyz2;
        bool b1 = TriangulateTwoViewCorrespondence(corr1.point2D_idx1, corr2.point2D_idx1, &xyz1);
        if (!b1) {
            b1 = TriangulateTwoViewCorrespondence(corr1.point2D_idx1, corr2.point2D_idx2, &xyz1);
        }
        bool b2 = TriangulateTwoViewCorrespondence(corr1.point2D_idx2, corr2.point2D_idx2, &xyz2);
        if (!b2) {
            b2 = TriangulateTwoViewCorrespondence(corr1.point2D_idx2, corr2.point2D_idx1, &xyz2);
        }
        if (b1 && b2) {
            if (local_camera_id1 > local_camera_id2) {
                uint32_t pair_id = local_camera_id2 * num_local_camera1 + local_camera_id1;
                corrs3D1[pair_id].emplace_back(xyz2);
                corrs3D2[pair_id].emplace_back(xyz1);
            } else {
                uint32_t pair_id = local_camera_id1 * num_local_camera1 + local_camera_id2;
                corrs3D1[pair_id].emplace_back(xyz1);
                corrs3D2[pair_id].emplace_back(xyz2);
            }
            num_triangulated++;
        }
    }
    std::cout << StringPrintf("%d correspondence are triangulated!\n", num_triangulated);

    std::vector<double> scales;
    for (const auto& corr3D : corrs3D1) {
        std::vector<Eigen::Vector3d>& points3D1 = corrs3D1.at(corr3D.first);
        std::vector<Eigen::Vector3d>& points3D2 = corrs3D2.at(corr3D.first);

        uint32_t local_camera_id1 = corr3D.first / num_local_camera1;
        uint32_t local_camera_id2 = corr3D.first % num_local_camera1;

        std::cout << corr3D.first << " " << num_local_camera1 << std::endl;

        std::cout << StringPrintf("local_camera %d -> %d: %d correspondence\n", 
            local_camera_id1, local_camera_id2, points3D1.size());

        if (points3D1.size() < 20) {
            continue;
        }

        auto model = SimilarityTransformEstimator<3, 0>::Estimate(points3D1, points3D2).at(0);
        Eigen::Matrix3d rot = model.block<3, 3>(0, 0);
        Eigen::Vector3d tvec = model.block<3, 1>(0, 3);
        Eigen::Vector4d qvec = RotationMatrixToQuaternion(rot);

        Eigen::Vector4d local_qvec1, local_qvec2;
        Eigen::Vector3d local_tvec1, local_tvec2;
        camera1.GetLocalCameraExtrinsic(local_camera_id1, local_qvec1, local_tvec1);
        camera1.GetLocalCameraExtrinsic(local_camera_id2, local_qvec2, local_tvec2);
        Eigen::Matrix3d R1 = QuaternionToRotationMatrix(local_qvec1);
        Eigen::Matrix3d R2 = QuaternionToRotationMatrix(local_qvec2);
        Eigen::Matrix3d relative_R = R2 * R1.transpose();
        Eigen::Vector3d relative_t = local_tvec2 - relative_R * local_tvec1;

        double scale = relative_t.norm() / tvec.norm();
        scales.push_back(scale);

        std::cout << StringPrintf("local_camera(%d)%d -> %d: %f\n", 
            points3D1.size(), local_camera_id1, local_camera_id2, scale);

        // {
        //     FILE *fp1 = fopen(StringPrintf("points3D%d-%d-1.obj", local_camera_id1, local_camera_id2).c_str(), "w");
        //     FILE *fp1_align = fopen(StringPrintf("points3D%d-%d-1-aligned.obj", local_camera_id1, local_camera_id2).c_str(), "w");
        //     for (auto point3D : points3D1) {
        //         fprintf(fp1, "v %f %f %f\n", point3D.x(), point3D.y(), point3D.z());
        //         auto point3D_aligned = rot * point3D + tvec;
        //         fprintf(fp1_align, "v %f %f %f\n", point3D_aligned.x(), point3D_aligned.y(), point3D_aligned.z());
        //     }
        //     fclose(fp1);
        //     fclose(fp1_align);
        //     FILE *fp2 = fopen(StringPrintf("points3D%d-%d-2.obj", local_camera_id1, local_camera_id2).c_str(), "w");
        //     for (auto point3D : points3D2) {
        //         fprintf(fp2, "v %f %f %f\n", point3D.x(), point3D.y(), point3D.z());
        //     }
        //     fclose(fp2);
        // }
    }
    if (scales.empty()) {
        std::cout << "Estimate Scale Failed!" << std::endl;
        return false;
    }
#endif
    return true;
}

void IncrementalMapper::ComputeDepthInfo(const Options& options) {
    const int num_eff_threads = sensemap::GetEffectiveNumThreads(-1);
    sensemap::ThreadPool thread_pool(num_eff_threads);
    std::cout << "ComputeDepthInfo" << std::flush;

    std::vector<image_t> images;
    for (auto & pair : reconstruction_->Images()) {
        images.emplace_back(pair.first);
    }

    for (size_t index = 0; index < images.size(); ++index)
    thread_pool.AddTask([&](size_t i) {
        const auto image_id = images[i];
        ComputeDepthInfo(options, image_id);
    }, index);
    thread_pool.Wait();

    std::cout << " done" << std::endl;
}

void IncrementalMapper::ComputeDepthInfo(const Options& options, const image_t image_id) {
    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());
    if (!IsFileRGBD(image.Name())) return;

    if (image.DepthFlag()) return;
    image.SetDepthFlag(true);

    const int width = camera.Width();
    const int height = camera.Height();

    RGBDData data;
    ExtractRGBDData(JoinPaths(options.image_path, image.Name()), RGBDReadOption::NoColor(), data);
    image.timestamp_ = data.timestamp;
    data.ReadRGBDCameraParams(options.rgbd_camera_params);
    if (!data.HasRGBDCalibration()) {
        std::cerr << "No RGBD calibration information found" << std::endl;
        std::abort();
    }

    MatXf warped_depthmap(width, height, 1);
    UniversalWarpDepthMap(warped_depthmap, data.depth, data.color_camera, data.depth_camera, data.depth_RT.cast<float>());

    const float window_step = std::max(1.0f, (warped_depthmap.GetWidth() + warped_depthmap.GetHeight()) * 0.0025f);
    for (size_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); 
        ++point2D_idx) {
        class Point2D& point2D = image.Point2D(point2D_idx);

        float scale_pt_x = point2D.X()/float(width) * float(warped_depthmap.GetWidth());
        float scale_pt_y = point2D.Y()/float(height) * float(warped_depthmap.GetHeight());

        float covariance = -1;
        int x = std::round(scale_pt_x);
        int y = std::round(scale_pt_y);

        std::vector<float> vals;
        float mean(0), stdev(0);
        
        int total_count = 0;
        for (int dy = -2; dy <= 2; dy++) {
            int iy = y + dy * window_step;
            for (int dx = -2; dx <= 2; dx++) {
                int ix = x + dx * window_step;
                if (iy >= 0 && iy < warped_depthmap.GetHeight() &&
                    ix >= 0 && ix < warped_depthmap.GetWidth()
                ) {
                    const float depth = warped_depthmap.Get(iy, ix);
                    if (depth > 0) {
                        vals.push_back(depth);
                        mean += depth;
                    }
                }
                total_count++;
            }
        }
        if (vals.size() > 0) {
            mean /= vals.size();
            std::for_each(vals.begin(), vals.end(), [&](float val) {
                stdev += (val - mean) * (val - mean);
            });
            covariance = std::sqrt(stdev / vals.size());
            point2D.Covariance() = covariance;
        }
        // TODO: Check if linear interpolation improves results or not.
        if (y >= 0 && y < warped_depthmap.GetHeight() &&
            x >= 0 && x < warped_depthmap.GetWidth()) {

            const float depth = warped_depthmap.Get(y, x);

            // Plain distance weight is better in practice
            float weight_distance = 1.0;

            // If a point has insufficient valid neighbors
            // increase the covariance due to its uncertainty
            float covar = covariance * std::pow(1.0f * (total_count + 1) / (vals.size() + 1), 2.0f);

            // For a point with a large covariance,
            // reduce the weight due to its uncertainty
            float weight_covariance = std::exp(-100.0f * covar * covar);

            point2D.Depth() = depth;
            point2D.DepthWeight() = weight_covariance * weight_distance;
        } else {
            point2D.Depth() = 0.0f;
            point2D.DepthWeight() = 0.0f;
        }
    }
}

void IncrementalMapper::GetSmallGrayAndDepth(const Options& options, const image_t image_id, cv::Mat &gray, cv::Mat &depth){
    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());


    RGBDData data;
    ExtractRGBDData(JoinPaths(options.image_path, image.Name()), data);
    int width = data.depth.GetWidth();
    int height = data.depth.GetHeight();
    data.ReadRGBDCameraParams(options.rgbd_camera_params);

    if (!data.HasRGBDCalibration() || 
         data.color_camera.ModelName() != "PINHOLE" || 
         data.depth_camera.ModelName() != "PINHOLE"
    ) {
        std::cerr << "ICP requires RGBD calibration with PINHOLE model" << std::endl;
        std::abort();
    }

    Eigen::Matrix3d color_K = data.color_camera.CalibrationMatrix();
    Eigen::Matrix3d depth_K = data.depth_camera.CalibrationMatrix();
    color_K.row(0) *= 1.0 * width / data.color.Width();
    color_K.row(1) *= 1.0 * height / data.color.Height();

    cv::Mat cv_gray;
    FreeImage2Mat(&data.color, cv_gray);
    if(cv_gray.cols!=width || cv_gray.rows!=height)
        cv::resize(cv_gray, cv_gray, cv::Size(width, height));

    gray = cv_gray;

    MatXf warped_depthmap(width, height, 1);
    FastWarpDepthMap(warped_depthmap, data.depth, color_K.cast<float>(), depth_K.cast<float>(), data.depth_RT.cast<float>());
    Mat2CvMat(warped_depthmap, depth);
    {
        std::lock_guard<std::mutex> lk(k_mtx_);
        if(small_warped_rgb_K_.isIdentity()) small_warped_rgb_K_ = color_K.cast<float>();
    }
}


bool IncrementalMapper::ComputeICPLink(const Options& options, const image_t src_id, const image_t dst_id){

    const int icp_level = 3;
    const std::vector<int> icp_iter = {5, 6, 6};
    const std::vector<int> icp_sample_step = {3, 2, 2};

    auto rgbd_align = std::make_shared<RGBDRegistration>(icp_level, icp_sample_step, 0.01, 56 / 255.0);

    auto pointcloud_align = std::make_shared<PointCloudAlign>(0.05, 7, 3);

    auto st = std::chrono::steady_clock::now();

    cv::Mat src_gray, dst_gray, src_depth, dst_depth;
    Image& src_image = reconstruction_->Image(src_id);
    Camera& src_camera = reconstruction_->Camera(src_image.CameraId());

    Image& dst_image = reconstruction_->Image(dst_id);
    Camera& dst_camera = reconstruction_->Camera(dst_image.CameraId());

//    Eigen::Matrix3d src_k_mat = src_camera.CalibrationMatrix();
//    Eigen::Vector4f src_K(src_k_mat(0, 0), src_k_mat(1, 1), src_k_mat(0, 2), src_k_mat(1, 2));
//    Eigen::Matrix3d dst_k_mat = dst_camera.CalibrationMatrix();
//    Eigen::Vector4f dst_K(dst_k_mat(0, 0), dst_k_mat(1, 1), dst_k_mat(0, 2), dst_k_mat(1, 2));

//    rgbd_align->SetInput(src_gray, src_depth, src_K, dst_gray, dst_depth, dst_K);
//    rgbd_align->enable_color_ = false;

    cv::Mat small_src_gray, small_dst_gray, small_src_depth, small_dst_depth;
    GetSmallGrayAndDepth(options, src_id, small_src_gray, small_src_depth);
    GetSmallGrayAndDepth(options, dst_id, small_dst_gray, small_dst_depth);

    if (options.rgbd_max_reproj_depth > 0) {
        for (int y = 0; y < small_src_depth.rows; y++) {
            for (int x = 0; x < small_src_depth.cols; x++) {
                if (small_src_depth.at<float>(y, x) > options.rgbd_max_reproj_depth) {
                    small_src_depth.at<float>(y, x) = 0;
                }
            }
        }
        for (int y = 0; y < small_dst_depth.rows; y++) {
            for (int x = 0; x < small_dst_depth.cols; x++) {
                if (small_dst_depth.at<float>(y, x) > options.rgbd_max_reproj_depth) {
                    small_dst_depth.at<float>(y, x) = 0;
                }
            }
        }
    }

    auto md1 = std::chrono::steady_clock::now();


    if(small_src_depth.cols>300) pointcloud_align->SetSampleStep(6);

//    cv::Mat depth_8;
//    small_src_depth.convertTo(depth_8, CV_8UC1, 255/5);
//    cv::imwrite(icp_path_ + std::to_string(src_id)+"d.jpg", depth_8 );
//    cv::imwrite(icp_path_ + std::to_string(src_id)+"g.jpg", small_src_gray );


    float scale = small_src_depth.cols/float(src_camera.Width());
    Eigen::Vector4f warped_rgb_K_vec(small_warped_rgb_K_(0, 0), small_warped_rgb_K_(1, 1),
                                     small_warped_rgb_K_(0, 2), small_warped_rgb_K_(1, 2));

    pointcloud_align->SetInput(small_src_depth, warped_rgb_K_vec, small_dst_depth, warped_rgb_K_vec);

//    pointcloud_align->SaveDstCloud(icp_path_ + std::to_string(src_id)+".obj", false);
//
//    pointcloud_align->SaveDstCloud(icp_path_ + std::to_string(src_id)+"n.obj", true);

    auto CorrespondenceGraph = reconstruction_->GetCorrespondenceGraph();
    FeatureMatches matches = CorrespondenceGraph->FindCorrespondencesBetweenImages(src_id, dst_id);

    auto src_points2d = src_image.Points2D();
    auto dst_points2d = dst_image.Points2D();

    std::vector<Eigen::Vector3d> src_pts, dst_pts;

    for(auto it = matches.begin(); it!=matches.end();){
        auto &src_pt = src_points2d[it->point2D_idx1];
        auto &dst_pt = dst_points2d[it->point2D_idx2];
        if(src_pt.Depth()<0.1 || dst_pt.Depth()<0.1){
            it = matches.erase(it);
        }
        else {
            src_pts.push_back(Eigen::Vector3d(src_pt.X(), src_pt.Y(), src_pt.Depth()));
            dst_pts.push_back(Eigen::Vector3d(dst_pt.X(), dst_pt.Y(), dst_pt.Depth()));
            it++;
        }
    }

    if(src_pts.size()<10) return false;

    FeatureAlign feat_align(0.05);
    feat_align.SetInput(src_pts, small_warped_rgb_K_.cast<double>()/scale,
                        dst_pts, small_warped_rgb_K_.cast<double>()/scale);

    Eigen::Matrix4d dst_pose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d src_pose = Eigen::Matrix4d::Identity();

    src_pose.block<3,4>(0,0) = src_image.ProjectionMatrix();//world to camera
    dst_pose.block<3,4>(0,0) = dst_image.ProjectionMatrix();

    Eigen::Matrix4d odometry = dst_pose * src_pose.inverse();

    int icp_corres_num = 0, feat_corres_num = 0;
    float icp_residual = 0, feat_residual = 0;

    ///icp func
    Eigen::Matrix6d icp_JtJ, fea_JtJ;
    Eigen::Vector6d icp_Jtr, fea_Jtr;
    ICP_Solver<6> icp_solver;
    for (int level = 2; level >= 0; level--) {
        for (int iter = 0; iter < icp_iter[level]; iter++) {

            pointcloud_align->SetSearchRadius(2 * level + 3);

//            rgbd_align->ComputeJacobian(level, odometry, icp_corres_num, icp_residual, icp_JtJ, icp_Jtr);

            pointcloud_align->ComputeJacobian(odometry, icp_corres_num, icp_residual, icp_JtJ, icp_Jtr);
//            printf("ICP align for kf %d %d, level %d error is: %d %f\n", src_id, dst_id, level,
//                     icp_corres_num, icp_residual);

//            std::cout<<"JTR: "<<icp_Jtr.transpose()<<std::endl;


            feat_align.ComputeJacobian(odometry, feat_corres_num, feat_residual, fea_JtJ, fea_Jtr);
            float feat_weight = 0.1 * icp_corres_num/float(std::max(feat_corres_num, 30));
            if(level==2) feat_weight *= 4;
            if(level==1) feat_weight *= 3;
            if(level==0) feat_weight *= 2;

            icp_JtJ+=feat_weight*feat_weight * fea_JtJ;
            icp_Jtr+=feat_weight*feat_weight * fea_Jtr;

            ///solve the problem
            icp_solver.UpdateState(odometry, icp_corres_num, icp_residual);
            if (icp_solver.roll_back_) {
                printf("icp_solver.roll_back_\n");
                odometry = icp_solver.RollBackSolve();
            } else {
                bool solve_sucess;
                Eigen::Matrix4d new_pose;
                std::tie(solve_sucess, new_pose) = icp_solver.Solve(icp_JtJ, -icp_Jtr);
                if (!solve_sucess) break;
                else {
                    odometry = new_pose;
                }
            }
        }
    }

    ///to do: check icp result and add to icp link
    ICP_Check icp_checker(0.05, 40.0, rgbd_align->enable_color_, rgbd_align->enable_bright_,
                          rgbd_align->bright_bias_);
//    float inlier_ratio = icp_checker.CheckAlignResult(rgbd_align->GetSrcPyr(), rgbd_align->GetDstPyr(), odometry);

    float inlier_ratio = icp_checker.CheckAlignResult(pointcloud_align->GetSrcXYZ(), pointcloud_align->GetDstXYZ(), odometry);

    auto md2 = std::chrono::steady_clock::now();


    float feat_weight = 0.2 * icp_corres_num/float(std::max(feat_corres_num, 30));
    // printf("ICP align %d  ==>> %d inlier is %lf, feat num is %d, %d, error is %f %f,  weight is: %f, weight_feat is: %f\n", 
    //         src_id, dst_id, inlier_ratio, src_pts.size(), feat_corres_num, icp_residual, feat_residual, 
    //         feat_weight,feat_weight * feat_residual);
    std::string name = std::to_string(src_id)+"_"+std::to_string(dst_id);
    std::string suf = inlier_ratio < 0.75 ? "f" : "s";

//    cv::Mat cat_img;
//    cv::hconcat(src_gray, dst_gray, cat_img);
//    cat_img.convertTo(cat_img, CV_GRAY2BGR);
//    cv::hconcat(cat_img,  icp_checker.show_result_, cat_img);
//    cv::imwrite(icp_path_+name+suf+".jpg", cat_img);

    //void SaveDepthAsObj(std::string fileName, cv::Mat depth, cv::Mat color, Eigen::Matrix3f K, Eigen::Matrix4f trans){
//    SaveDepthAsObj(icp_path_+name+"s" + suf + ".obj", rgbd_align->GetSrcPyr(), odometry.cast<float>());
//    SaveDepthAsObj(icp_path_+name+"d" + suf + ".obj", rgbd_align->GetDstPyr(), Eigen::Matrix4f::Identity());


    if(feat_corres_num<10) return false;

    if (inlier_ratio < 0.75) {
        return false;
    }
    ICP_Info icp_info(0.01);
//    int matched_cnt = icp_info.ComputeInfo(rgbd_align->GetSrcPyr(), rgbd_align->GetDstPyr(), odometry);
    int matched_cnt = icp_info.ComputeInfo(pointcloud_align->GetSrcXYZ(), pointcloud_align->GetDstXYZ(), odometry);

    ICPLink icp_link(dst_id, matched_cnt, odometry, icp_info.GetInfo());


    auto ed = std::chrono::steady_clock::now();
    printf("ICP time %f  %f  %f\n", std::chrono::duration<float, std::milli>(md1 - st).count(),
           std::chrono::duration<float, std::milli>(md2 - md1).count(),
           std::chrono::duration<float, std::milli>(ed - md2).count());


    src_image.icp_links_.push_back(icp_link);
    return true;
}


std::pair<bool, ICPLink> IncrementalMapper::ComputeICPLink2(const Options& options, const image_t src_id, const image_t dst_id){

    const int icp_level = 3;
    const std::vector<int> icp_iter = {5, 6, 6};
    const std::vector<int> icp_sample_step = {3, 2, 2};

    auto rgbd_align = std::make_shared<RGBDRegistration>(icp_level, icp_sample_step, 0.01, 56 / 255.0);

    auto pointcloud_align = std::make_shared<PointCloudAlign>(0.05, 7, 3);

    auto st = std::chrono::steady_clock::now();

    cv::Mat src_gray, dst_gray, src_depth, dst_depth;
    Image& src_image = reconstruction_->Image(src_id);
    Camera& src_camera = reconstruction_->Camera(src_image.CameraId());

    Image& dst_image = reconstruction_->Image(dst_id);
    Camera& dst_camera = reconstruction_->Camera(dst_image.CameraId());

//    Eigen::Matrix3d src_k_mat = src_camera.CalibrationMatrix();
//    Eigen::Vector4f src_K(src_k_mat(0, 0), src_k_mat(1, 1), src_k_mat(0, 2), src_k_mat(1, 2));
//    Eigen::Matrix3d dst_k_mat = dst_camera.CalibrationMatrix();
//    Eigen::Vector4f dst_K(dst_k_mat(0, 0), dst_k_mat(1, 1), dst_k_mat(0, 2), dst_k_mat(1, 2));
//    rgbd_align->SetInput(src_gray, src_depth, src_K, dst_gray, dst_depth, dst_K);
//    rgbd_align->enable_color_ = false;

    cv::Mat small_src_gray, small_dst_gray, small_src_depth, small_dst_depth;
    GetSmallGrayAndDepth(options, src_id, small_src_gray, small_src_depth);
    GetSmallGrayAndDepth(options, dst_id, small_dst_gray, small_dst_depth);

    if (options.rgbd_max_reproj_depth > 0) {
        for (int y = 0; y < small_src_depth.rows; y++) {
            for (int x = 0; x < small_src_depth.cols; x++) {
                if (small_src_depth.at<float>(y, x) > options.rgbd_max_reproj_depth) {
                    small_src_depth.at<float>(y, x) = 0;
                }
            }
        }
        for (int y = 0; y < small_dst_depth.rows; y++) {
            for (int x = 0; x < small_dst_depth.cols; x++) {
                if (small_dst_depth.at<float>(y, x) > options.rgbd_max_reproj_depth) {
                    small_dst_depth.at<float>(y, x) = 0;
                }
            }
        }
    }

    auto md1 = std::chrono::steady_clock::now();


    if(small_src_depth.cols>300) pointcloud_align->SetSampleStep(6);

//    cv::Mat depth_8;
//    small_src_depth.convertTo(depth_8, CV_8UC1, 255/5);
//    cv::imwrite(icp_path_ + std::to_string(src_id)+"d.jpg", depth_8 );
//    cv::imwrite(icp_path_ + std::to_string(src_id)+"g.jpg", small_src_gray );


    float scale = small_src_depth.cols/float(src_camera.Width());
    Eigen::Vector4f warped_rgb_K_vec(small_warped_rgb_K_(0, 0), small_warped_rgb_K_(1, 1),
                                     small_warped_rgb_K_(0, 2), small_warped_rgb_K_(1, 2));

    pointcloud_align->SetInput(small_src_depth, warped_rgb_K_vec, small_dst_depth, warped_rgb_K_vec);

//    pointcloud_align->SaveDstCloud(icp_path_ + std::to_string(src_id)+".obj", false);
//    pointcloud_align->SaveDstCloud(icp_path_ + std::to_string(src_id)+"n.obj", true);

    auto CorrespondenceGraph = reconstruction_->GetCorrespondenceGraph();
    FeatureMatches matches = CorrespondenceGraph->FindCorrespondencesBetweenImages(src_id, dst_id);

    auto src_points2d = src_image.Points2D();
    auto dst_points2d = dst_image.Points2D();

    std::vector<Eigen::Vector3d> src_pts, dst_pts;

    for(auto it = matches.begin(); it!=matches.end();){
        auto &src_pt = src_points2d[it->point2D_idx1];
        auto &dst_pt = dst_points2d[it->point2D_idx2];
        if(src_pt.Depth()<0.1 || dst_pt.Depth()<0.1){
            it = matches.erase(it);
        }
        else {
            src_pts.push_back(Eigen::Vector3d(src_pt.X(), src_pt.Y(), src_pt.Depth()));
            dst_pts.push_back(Eigen::Vector3d(dst_pt.X(), dst_pt.Y(), dst_pt.Depth()));
            it++;
        }
    }

    if(src_pts.size()<10) return std::make_pair(false, ICPLink());

    FeatureAlign feat_align(0.05);
    feat_align.SetInput(src_pts, small_warped_rgb_K_.cast<double>()/scale,
                        dst_pts, small_warped_rgb_K_.cast<double>()/scale);


    Eigen::Matrix4d dst_pose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d src_pose = Eigen::Matrix4d::Identity();

    src_pose.block<3,4>(0,0) = src_image.ProjectionMatrix();//world to camera
    dst_pose.block<3,4>(0,0) = dst_image.ProjectionMatrix();

    Eigen::Matrix4d odometry = dst_pose * src_pose.inverse();

    int icp_corres_num = 0, feat_corres_num = 0;
    float icp_residual = 0, feat_residual = 0;

    ///icp func
    Eigen::Matrix6d icp_JtJ, fea_JtJ;
    Eigen::Vector6d icp_Jtr, fea_Jtr;
    ICP_Solver<6> icp_solver;
    for (int level = 2; level >= 0; level--) {
        for (int iter = 0; iter < icp_iter[level]; iter++) {

            pointcloud_align->SetSearchRadius(2 * level + 3);

//            rgbd_align->ComputeJacobian(level, odometry, icp_corres_num, icp_residual, icp_JtJ, icp_Jtr);

            pointcloud_align->ComputeJacobian(odometry, icp_corres_num, icp_residual, icp_JtJ, icp_Jtr);
//            printf("ICP align for kf %d %d, level %d error is: %d %f\n", src_id, dst_id, level,
//                     icp_corres_num, icp_residual);

//            std::cout<<"JTR: "<<icp_Jtr.transpose()<<std::endl;

            feat_align.ComputeJacobian(odometry, feat_corres_num, feat_residual, fea_JtJ, fea_Jtr);
            float feat_weight = 0.1 * icp_corres_num/float(std::max(feat_corres_num, 30));
            if(level==2) feat_weight *= 4;
            if(level==1) feat_weight *= 3;
            if(level==0) feat_weight *= 2;

            icp_JtJ+=feat_weight*feat_weight * fea_JtJ;
            icp_Jtr+=feat_weight*feat_weight * fea_Jtr;

            ///solve the problem
            icp_solver.UpdateState(odometry, icp_corres_num, icp_residual);
            if (icp_solver.roll_back_) {
                printf("icp_solver.roll_back_\n");
                odometry = icp_solver.RollBackSolve();
            } else {
                bool solve_sucess;
                Eigen::Matrix4d new_pose;
                std::tie(solve_sucess, new_pose) = icp_solver.Solve(icp_JtJ, -icp_Jtr);
                if (!solve_sucess) break;
                else {
                    odometry = new_pose;
                }
            }
        }
    }

    ///to do: check icp result and add to icp link
    ICP_Check icp_checker(0.05, 40.0, rgbd_align->enable_color_, rgbd_align->enable_bright_,
                          rgbd_align->bright_bias_);
//    float inlier_ratio = icp_checker.CheckAlignResult(rgbd_align->GetSrcPyr(), rgbd_align->GetDstPyr(), odometry);

    float inlier_ratio = icp_checker.CheckAlignResult(pointcloud_align->GetSrcXYZ(), pointcloud_align->GetDstXYZ(), odometry);

    auto md2 = std::chrono::steady_clock::now();


    float feat_weight = 0.2 * icp_corres_num/float(std::max(feat_corres_num, 30));
    // printf("ICP align %d  ==>> %d inlier is %lf, feat num is %d, %d, error is %f %f,  weight is: %f, weight_feat is: %f\n",
    //        src_id, dst_id, inlier_ratio, src_pts.size(), feat_corres_num, icp_residual, feat_residual,
    //        feat_weight,feat_weight * feat_residual);
    std::string name = std::to_string(src_id)+"_"+std::to_string(dst_id);
    std::string suf = inlier_ratio < 0.75 ? "f" : "s";

//    cv::Mat cat_img;
//    cv::hconcat(src_gray, dst_gray, cat_img);
//    cat_img.convertTo(cat_img, CV_GRAY2BGR);
//    cv::hconcat(cat_img,  icp_checker.show_result_, cat_img);
//    cv::imwrite(icp_path_+name+suf+".jpg", cat_img);

    //void SaveDepthAsObj(std::string fileName, cv::Mat depth, cv::Mat color, Eigen::Matrix3f K, Eigen::Matrix4f trans){
//    SaveDepthAsObj(icp_path_+name+"s" + suf + ".obj", rgbd_align->GetSrcPyr(), odometry.cast<float>());
//    SaveDepthAsObj(icp_path_+name+"d" + suf + ".obj", rgbd_align->GetDstPyr(), Eigen::Matrix4f::Identity());


    if(feat_corres_num<10) return std::make_pair(false, ICPLink());

    if (inlier_ratio < 0.75) {
        return std::make_pair(false, ICPLink());
    }
    ICP_Info icp_info(0.01);
//    int matched_cnt = icp_info.ComputeInfo(rgbd_align->GetSrcPyr(), rgbd_align->GetDstPyr(), odometry);
    int matched_cnt = icp_info.ComputeInfo(pointcloud_align->GetSrcXYZ(), pointcloud_align->GetDstXYZ(), odometry);

    ICPLink icp_link(dst_id, matched_cnt, odometry, icp_info.GetInfo());

    auto ed = std::chrono::steady_clock::now();
    return std::make_pair(true, icp_link);
}

void IncrementalMapper::LidarSetUp(const Options &options) {
    reconstruction_->LoadLidar(options.lidar_path);

    const auto & lidarsweeps = reconstruction_->LidarSweeps();
    for (const auto & sweep :  lidarsweeps) {
        sweep_timestamps_.emplace_back(sweep.first, sweep.second.timestamp_);
    }
    std::sort(sweep_timestamps_.begin(), sweep_timestamps_.end(), 
        [&](const std::pair<sweep_t, long long> & a, const std::pair<sweep_t, long long> & b){
            return a.second < b.second;
        });

    for (auto & image : reconstruction_->Images()) {
        auto & rec_image = reconstruction_->Image(image.first); 
        auto image_name = GetPathBaseName(rec_image.Name());
        image_name = image_name.substr(0, image_name.rfind("."));
        long long image_timestamp = std::atof(image_name.c_str()) * 1e9;
        rec_image.timestamp_ = image_timestamp;
        image_timestamps_.emplace_back(image.first, image_timestamp);
    }
    std::sort(image_timestamps_.begin(), image_timestamps_.end(), 
        [&](const std::pair<image_t, long long> & a, const std::pair<image_t, long long> & b){
            return a.second < b.second;
        });

    // load lidar prior pose.
    if (!options.lidar_prior_pose_file.empty()) {
        std::vector<RigNames> rig_names;
        ReadRigList(options.lidar_prior_pose_file, rig_names);

        const auto & sweep_name_to_id = reconstruction_->sweep_name_to_id;

        for (const auto & rig_name : rig_names) {
            // std::string lidar_name = GetPathBaseName(rig_name.pcd);
            std::string lidar_name = rig_name.name;
            // std::cout << "lidar prior: " << lidar_name << " " << sweep_name_to_id.count(lidar_name) << std::endl;
            if (sweep_name_to_id.find(lidar_name) != sweep_name_to_id.end()) {
                sweep_t sweep_id = sweep_name_to_id.at(lidar_name);
                LidarSweep & sweep = reconstruction_->LidarSweep(sweep_id);
                sweep.SetQvecPrior(InvertQuaternion(rig_name.q));
                sweep.SetTvecPrior(rig_name.t);
                // std::cout << rig_name.t.transpose() << " " << rig_name.q.transpose() << std::endl;
            }
        }
    }
}

void IncrementalMapper::AppendToVoxelMap(const Options& options, const sweep_t sweep_id) {
    auto octree_start_time = std::chrono::steady_clock::now();

    auto & lidar_sweep = reconstruction_->LidarSweep(sweep_id);

    // Eigen::Matrix4d htrans = Eigen::Matrix4d::Identity();
    // htrans.topRows(3) = lidar_sweep.ProjectionMatrix();
    // Eigen::Matrix4d T = htrans.inverse();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topRows(3) = lidar_sweep.InverseProjectionMatrix();

    LidarPointCloud ref_less_points, ref_less_points_t;
    LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
    LidarPointCloud ref_less_corners = lidar_sweep.GetCornerPointsLessSharp();
    ref_less_points = std::move(ref_less_surfs);
    ref_less_points += std::move(ref_less_corners);
    LidarPointCloud::TransfromPlyPointCloud (ref_less_points, ref_less_points_t, T);

    std::vector<lidar::OctoTree::Point> points;
    points.reserve(ref_less_points_t.points.size());
    for (auto point : ref_less_points_t.points) {
        lidar::OctoTree::Point X;
        X.x = point.x;
        X.y = point.y;
        X.z = point.z;
        X.intensity = point.intensity;
        X.lifetime = point.lifetime;
        // X.lifetime = lidar_sweep.timestamp_;
        X.type = 0;
        points.push_back(X);
    }

    reconstruction_->VoxelMap()->AppendToVoxelMap(points);
    auto octree_end_time = std::chrono::steady_clock::now();
    float elapsed_time = std::chrono::duration<float>(octree_end_time - octree_start_time).count();
    std::cout << StringPrintf("AppendToVoxelMap in %.3f sec\n", elapsed_time);
}

void IncrementalMapper::AbstractFeatureVoxels(const Options& options, const std::string plane_path, const std::string line_path, const bool force) {
    std::cout << "Abstract all feature voxels" << std::endl;

    std::srand(0);

    std::vector<Eigen::Vector3d> plane_points;
    std::vector<Eigen::Vector3i> plane_colors;
    std::vector<Eigen::Vector3d> line_points;
    std::vector<Eigen::Vector3i> line_colors;

    std::vector<lidar::OctoTree*> octree_list = reconstruction_->VoxelMap()->AbstractFeatureVoxels(force);
    for (auto octree : octree_list) {
        int r = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        int g = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        int b = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        std::vector<Eigen::Vector3d> points;
        octree->GetGridPoints(points);
        if (octree->voxel_->FeatureType() == Voxel::FeatureType::PLANE) {
            for (size_t i = 0; i < points.size(); ++i) {
                double w = 1.0 - octree->voxel_->Error(points[i]);
                // double w = 1.0;
                plane_colors.push_back(Eigen::Vector3i(r * w, g * w, b * w));
            }
            plane_points.insert(plane_points.end(), points.begin(), points.end());
        } else if (octree->voxel_->FeatureType() == Voxel::FeatureType::LINE) {
            line_points.insert(line_points.end(), points.begin(), points.end());
            line_colors.insert(line_colors.end(), points.size(), Eigen::Vector3i(r, g, b));
        }
        if (force) {
            for (size_t i = 0; i < points.size(); ++i) {
                double w = 1.0 - octree->voxel_->Error(points[i]);
                // double w = 1.0;
                plane_colors.push_back(Eigen::Vector3i(r * w, g * w, b * w));
            }
            plane_points.insert(plane_points.end(), points.begin(), points.end());
        }
    }

    if (!plane_path.empty()) {
        std::ofstream file(plane_path, std::ios::out);
        for (size_t i = 0; i < plane_points.size(); ++i) {
            auto & p = plane_points[i];
            auto & c = plane_colors[i];
            file << "v " << p[0] << " " << p[1] << " " << p[2] << " " << c[0] << " " << c[1] << " " << c[2] << std::endl;
        }
        file.close();
    }
    
    if (!line_path.empty()) {
        std::ofstream file2(line_path, std::ios::out);
        for (size_t i = 0; i < line_points.size(); ++i) {
            auto & p = line_points[i];
            auto & c = line_colors[i];
            file2 << "v " << p[0] << " " << p[1] << " " << p[2] << " " << c[0] << " " << c[1] << " " << c[2] << std::endl;
        }
        file2.close();
    }
}

void IncrementalMapper::AbstractPlaneFeatureVoxels(const std::vector<lidar::OctoTree::Point> & points,
    const Options& options, const std::string plane_path, const std::string line_path) {
    // std::cout << "Abstract all plane feature voxels about " << points.size() << " points" << std::endl;

    std::srand(0);

    std::vector<Eigen::Vector3d> plane_points;
    std::vector<Eigen::Vector3d> plane_colors;
    std::vector<size_t> plane_id;

    // std::vector<lidar::OctoTree*> octree_list = reconstruction_->VoxelMap()->AbstractFeatureVoxels();
    std::vector<lidar::OctoTree*> octree_list = reconstruction_->VoxelMap()->AbstractSweepFeatureVoxels(points);
    // std::cout << "AbstractSweepFeatureVoxels: " << octree_list.size() << std::endl;
    size_t num_octree = 0;
    for (auto octree : octree_list) {
        int r = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        int g = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        int b = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        std::vector<Eigen::Vector3d> points;
        octree->GetGridPoints(points);
        if (octree->voxel_->FeatureType() == Voxel::FeatureType::PLANE) {
            for (size_t i = 0; i < points.size(); ++i) {
                // double w = 1.0 - octree->voxel_->Error(points[i]);
                double w = 1.0 / 255;
                plane_colors.push_back(Eigen::Vector3d(r * w, g * w, b * w));
                plane_id.push_back(num_octree);
            }
            plane_points.insert(plane_points.end(), points.begin(), points.end());
            num_octree++;
        } 
    }

    if (!plane_path.empty()) {
        std::ofstream file(plane_path, std::ios::out);
        for (size_t i = 0; i < plane_points.size(); ++i) {
            auto & p = plane_points[i];
            auto & c = plane_colors[i];
            file << "v " << p[0] << " " << p[1] << " " << p[2] << " " << c[0] << " " << c[1] << " " << c[2] << " " << plane_id[i] << std::endl;
        }
        file.close();
        
    }
    
}

}  // namespace sensemap
