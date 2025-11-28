//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_OPTIM_BUNDLE_ADJUSTMENT_H_
#define SENSEMAP_OPTIM_BUNDLE_ADJUSTMENT_H_

#include <memory>
#include <unordered_set>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "util/alignment.h"
#include "util/ceres_types.h"
#include "base/reconstruction.h"
#include "lidar/lidar_correspondence.h"

namespace sensemap {

class Reconstruction;

struct BundleAdjustmentOptions {
    // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
    enum class LossFunctionType { TRIVIAL, SOFT_L1, Huber, CAUCHY };
    LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

    // Scaling factor determines residual at which robustification takes place.
    double loss_function_scale = 1.0;

    // Lower bound for focal length
    double lower_bound_focal_length_factor = 0.5;
    
    // Upper bound for focal length
    double upper_bound_focal_length_factor = 1.5;

    // Whether to refine the focal length parameter group.
    bool refine_focal_length = true;

    // Whether to refine the principal point parameter group.
    bool refine_principal_point = false;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = true;

    // Whether to refine the local extrinsics for camera rig
    bool refine_local_extrinsics = false;

    // Whether to refine the rig camera to the same position.
    bool local_relative_translation_constraint = false;

    // Whether to refine the extrinsic parameter group.
    bool refine_extrinsics = true;

    // Whether to print a final summary.
    bool print_summary = true;

    bool plane_constrain = false;

    bool gba_weighted = false;

    double plane_weight = 0.1;

    double ba_min_tri_angle = 0.0;
    // Ceres-Solver options.
    ceres::Solver::Options solver_options;

    // Whether to use prior relative pose
    bool use_prior_relative_pose = false;
    bool use_prior_distance_only = false;
    bool use_prior_translation_only = false;
    bool use_prior_aggressively = false;
    double prior_pose_weight = 1.0;

    // Whether to use absolute location, e.g. the gps prior
    bool use_prior_absolute_location = false;
    double prior_absolute_location_weight = 1.0;
    double prior_absolute_orientation_weight = 0.0;

    // Gnss2Camera extrinc prior
    double prior_gnss2camera_extri_weight = 1;
	// Depth weight in BA
    double rgbd_ba_depth_weight = 10.0;

    // icp
    bool use_icp_relative_pose = false;
    double icp_base_weight = 10.0;

    // gravity
    bool use_gravity = false;
    double gravity_base_weight = 10.0;

    // time domain smoothing
    bool use_time_domain_smoothing = false;
    double time_domain_smoothing_weight = 2.0;

    // block ba
    bool parameterize_points_with_track = true;
    bool debug_info = false;
    int block_ba_frequency = 10;
    int block_size = -1;
    int block_common_image_num = 20;
    std::string workspace_path = "";
    int min_connected_points_for_common_images = 100;
    
    // Only use latitude and longtitude for optimization or not
    bool optimization_use_horizontal_gps_only = false;

    bool refine_points_only = false;
    bool force_full_ba = true;

    // lidar
    double lidar_weight = 100;
    bool refine_lidar2cam_params = false;
    bool lidarsweep_voxel_gnss = true;
    int refine_min_numsweeps = std::numeric_limits<int>::max();
    int max_num_iteration_frame2frame = 10;

    BundleAdjustmentOptions() {
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 0.0;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
    }

    std::string save_path = "";

    // Create a new loss function based on the specified options. The caller
    // takes ownership of the loss function.
    ceres::LossFunction* CreateLossFunction() const;

    bool Check() const;
};

// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentConfig {
public:
    BundleAdjustmentConfig();

    size_t NumImages() const;
    size_t NumPoints() const;
    size_t NumConstantCameras() const;
    size_t NumConstantPoses() const;
    size_t NumConstantTvecs() const;
    size_t NumVariablePoints() const;
    size_t NumConstantPoints() const;
    size_t NumConstantGNSS() const;
    size_t NumSweeps() const;

    // Determine the number of residuals for the given reconstruction. The number
    // of residuals equals the number of observations times two.
    size_t NumResiduals(const Reconstruction& reconstruction) const;

    // Add / remove images from the configuration.
    void AddImage(const image_t image_id);
    bool HasImage(const image_t image_id) const;
    void RemoveImage(const image_t image_id);

    // Add / remove lidar sweep from the configuration.
    void AddSweep(const sweep_t sweep_id);
    bool HasSweep(const sweep_t sweep_id) const;
    void RemoveSweep(const sweep_t sweep_id);
    void AddSweepImagePair(const sweep_t sweep_id, const image_t image_id);
    void SetLidar2CamMatrix(const Eigen::Matrix3x4d lidar_2_cam);

    // Set cameras of added images as constant or variable. By default all
    // cameras of added images are variable. Note that the corresponding images
    // have to be added prior to calling these methods.
    void SetConstantCamera(const camera_t camera_id);
    void SetVariableCamera(const camera_t camera_id);
    bool IsConstantCamera(const camera_t camera_id) const;

    // Set the pose of added images as constant. The pose is defined as the
    // rotational and translational part of the projection matrix.
    void SetConstantPose(const image_t image_id);
    void SetVariablePose(const image_t image_id);
    bool HasConstantPose(const image_t image_id) const;

    // Set the pose of added scans as constant. The pose is defined as the
    // rotational and translational part of the projection matrix.
    void SetConstantSweep(const sweep_t sweep_id);
    void SetVariableSweep(const sweep_t sweep_id);
    bool HasConstantSweep(const sweep_t sweep_id) const;

    // Set the translational part of the pose, hence the constant pose
    // indices may be in [0, 1, 2] and must be unique. Note that the
    // corresponding images have to be added prior to calling these methods.
    void SetConstantTvec(const image_t image_id, const std::vector<int>& idxs);
    void RemoveConstantTvec(const image_t image_id);
    bool HasConstantTvec(const image_t image_id) const;

    // Add / remove points from the configuration. Note that points can either
    // be variable or constant but not both at the same time.
    void AddVariablePoint(const mappoint_t mappoint_id);
    void AddConstantPoint(const mappoint_t mappoint_id);
    bool HasPoint(const mappoint_t mappoint_id) const;
    bool HasVariablePoint(const mappoint_t mappoint_id) const;
    bool HasConstantPoint(const mappoint_t mappoint_id) const;
    void RemoveVariablePoint(const mappoint_t mappoint_id);
    void RemoveConstantPoint(const mappoint_t mappoint_id);

    // Access configuration data.
    const std::unordered_set<image_t>& Images() const;
    const std::unordered_set<sweep_t>& Sweeps() const;
    const std::unordered_map<sweep_t, image_t> SweepImagePairs() const;
    const std::unordered_set<mappoint_t>& VariablePoints() const;
    const std::unordered_set<mappoint_t>& ConstantPoints() const;
    const std::vector<int>& ConstantTvec(const image_t image_id) const;
    const std::unordered_set<camera_t>& ConstantCameraIds() const;
    const std::unordered_set<image_t>& ConstantPoses() const;
    const std::unordered_map<image_t, std::vector<int>>& ConstantTVecs() const;

    // Add / remove gnss_images from the configuration.
    const std::unordered_set<image_t>& GNSSImages() const;
    void AddGNSS(const image_t image_id);    
    bool HasGNSS(const image_t image_id) const;
    void RemoveGNSS(const image_t image_id);

    Eigen::Matrix3x4d lidar_to_cam_matrix_;
private:
    std::unordered_set<camera_t> constant_camera_ids_;
    std::unordered_set<image_t> image_ids_;
    std::unordered_set<mappoint_t> variable_mappoint_ids_;
    std::unordered_set<mappoint_t> constant_mappoint_ids_;
    std::unordered_set<image_t> constant_poses_;
    std::unordered_map<image_t, std::vector<int>> constant_tvecs_;

    std::unordered_set<sweep_t> sweep_ids_;
    std::unordered_set<sweep_t> const_sweeps_;
    std::unordered_map<image_t, sweep_t> sweep_image_pairs_;

    std::unordered_set<image_t> constant_gnss_ids_;
    
    // specical points that have prior locations
    std::unordered_set<mappoint_t> landmark_mappoint_ids_;
    std::unordered_map<mappoint_t,std::vector<std::pair<mappoint_t,double>>> relative_landmark_distances_;
};

// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
class BundleAdjuster {
public:
    BundleAdjuster(const BundleAdjustmentOptions& options,
                    const BundleAdjustmentConfig& config);

    bool Solve(Reconstruction* reconstruction);

    bool PyramidSolve(Reconstruction* reconstruction);

    // Fast version of GBA
    bool FastSolve(Reconstruction* reconstruction, 
                   std::vector<mappoint_t> &const_mappoint_ids, 
                   std::vector<mappoint_t> &ba_mappoint_ids,
                   const BundleAdjustmentOptions& options);

    size_t PrecomputeResiduals(Reconstruction* reconstruction);

    // Get the Ceres solver summary for the last call to `Solve`.
    const ceres::Solver::Summary& Summary() const;

protected:
    void SetUp(Reconstruction* reconstruction,
                ceres::LossFunction* loss_function);
    // customized version according to the fast GBA
    void SetUp(Reconstruction* reconstruction,
               std::vector<mappoint_t>& const_mappoint_ids,
               std::vector<mappoint_t>& ba_mappoint_ids,
               ceres::LossFunction* loss_function,
               const BundleAdjustmentOptions& options);
    void TearDown(Reconstruction* reconstruction);

    void AddImageToProblem(const image_t image_id, Reconstruction* reconstruction,
                            ceres::LossFunction* loss_function);
    void AddImageToStructOnlyProblem(const image_t image_id, Reconstruction* reconstruction,
                            ceres::LossFunction* loss_function);


    void AddPointToStructOnlyProblem(std::unique_ptr<ceres::Problem>& problem, const mappoint_t mappoint_id,
                            Reconstruction* reconstruction,
                            ceres::LossFunction* loss_function);

    void AddPointToProblem(const mappoint_t mappoint_id,
                            Reconstruction* reconstruction,
                            ceres::LossFunction* loss_function);
    
    void AddNovatelToProblem(const image_t image_id, Reconstruction* reconstruction,
                            ceres::LossFunction* loss_function);

    void AddGnssExtriToProble(const camera_t camera_id, Reconstruction *reconstruction, 
                              ceres::LossFunction *loss_function);

    void AddLidarToProblem(const sweep_t sweep_id, const image_t image_id, const uint32_t m_num_visible_point, 
                           Reconstruction* reconstruction, ceres::LossFunction* loss_function);

    void AddMapPointToVoxelMap(Reconstruction* reconstruction, ceres::LossFunction* loss_function);

    void AddSfMToLidarConstraint(Reconstruction* reconstruction);
    
    void AddLidarFrame2FrameToProblem(const sweep_t sweep_id,
                                  Reconstruction *reconstruction,
                                  ceres::LossFunction *loss_function,
                                  const double loss_weight);

    void AddExtraLidar2CameraToProblem(const camera_t camera_id,
                                       Reconstruction *reconstruction,
                                       ceres::LossFunction *loss_function,
                                       const double loss_weight,
                                       const Eigen::Matrix4d absolute_pose);
                     

protected:
    void ParameterizeCameras(Reconstruction* reconstruction);
    void ParameterizePoints(Reconstruction* reconstruction);
    void ParameterizePointsWithoutTrack(Reconstruction *reconstruction);
    void ParameterizePoses(Reconstruction* reconstruction);
    
    inline void SetParameterization(ceres::Problem *problem, double* values, ceres::LocalParameterization *local_parameterization);

    const BundleAdjustmentOptions options_;
    BundleAdjustmentConfig config_;
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
    std::unordered_set<camera_t> camera_ids_;
    std::unordered_set<camera_t> gnss_camera_ids_;
    std::unordered_set<camera_t> lidar_camera_ids_;
    std::unordered_map<camera_t, std::unordered_set<int> >local_camera_ids_;
    std::unordered_map<mappoint_t, size_t> mappoint_num_observations_;
    std::unordered_set<image_t> invovled_image_extrinsics_;
    std::unordered_set<image_t> constant_pose_images_;
    Eigen::Vector3d gravity_;
};

class BlockBundleAdjuster {
public:

    struct BlockBundleAdjustmentOptions {
        size_t max_block_images = -1;
        size_t block_images_num = -1;
        size_t common_images_num = -1;
        size_t link_edges = 3;
        size_t maximum_threads_num = -1;
        size_t min_connected_points_for_common_images = 100;
        BlockBundleAdjustmentOptions() {}
        BlockBundleAdjustmentOptions(
            const size_t in_total_images,
            const size_t in_images_per_block,
            const size_t in_common_images_num,
            const size_t in_min_connected_points_for_common_images) 
        {
            max_block_images = in_images_per_block;
            common_images_num = in_common_images_num;
            min_connected_points_for_common_images = in_min_connected_points_for_common_images;
            if (max_block_images <= 0 || common_images_num <= 0 || common_images_num >= max_block_images) {
                block_images_num = -1;
            }
            else {
                size_t target_block_num = (in_total_images - 1) / max_block_images + 1;
                size_t target_images_thres = in_total_images / target_block_num;
                size_t remain_num = in_total_images % target_images_thres;

                block_images_num = target_images_thres + remain_num / target_block_num + 1;
            }
        }
    };

    struct Block {
        block_t id;
        
        // used in relative pose constraint.
        std::unordered_set<image_t> invovled_image_extrinsics;

        std::unordered_set<mappoint_t> mappoints_in_mask;

        std::unordered_set<camera_t> camera_ids;
        std::unordered_map<camera_t, std::unordered_set<int> >local_camera_ids;
        std::unordered_map<mappoint_t, size_t> mappoint_num_observations;
        BundleAdjustmentConfig config;
    };

    struct DisjointSet {
        std::vector<size_t> roots_;

        DisjointSet(size_t size) {
            roots_.resize(size);
            for (int i = 0; i < roots_.size(); ++ i) roots_[i] = i;
        }

        size_t Find(size_t x) {
            if (roots_[x] == x) return roots_[x];
            return roots_[x] = Find(roots_[x]);
        }

        void Merge(size_t x, size_t y) {
            int fx = Find(x);
            int fy = Find(y);
            roots_[fy] = fx;
        }

        std::vector<size_t> MaxComponent() {
            std::unordered_map<size_t, size_t> component_mapper;

            for (int i = 0; i < roots_.size(); ++ i) Find(i);

            size_t maximum_group_id = -1;
            size_t maximum_group_num = 0;
            for (int i = 0; i < roots_.size(); ++ i) {
                size_t group_id = roots_[i];
                int cnt = 0;
                if (component_mapper.count(group_id)) cnt = component_mapper[group_id];

                ++ cnt;
                component_mapper[group_id] = cnt;

                if (cnt > maximum_group_num) {
                    maximum_group_num = cnt;
                    maximum_group_id = group_id;
                }
            }

            std::vector<size_t> res;
            for (int i = 0; i < roots_.size(); ++ i) {
                if (roots_[i] == maximum_group_id) {
                    res.emplace_back(i);
                }
            }

            return res;
        }
    };

public:
    BlockBundleAdjuster(const BundleAdjustmentOptions& options,
                        const BundleAdjustmentConfig& config);

    bool Solve(Reconstruction* reconstruction);

private:
    void DivideBlocks(Reconstruction* reconstruction, std::vector<Block>* blocks,
                      std::unordered_map<image_pair_t, int>* corres_map);
    bool SolveBlock(Reconstruction* reconstruction, Block* block);

    void SetUp(Reconstruction* reconstruction, 
               ceres::Problem* problem, 
               ceres::LossFunction *loss_function, 
               Block* block);

private:
    void AddImageToProblem(const image_t image_id, 
                           Reconstruction* reconstruction, 
                           ceres::Problem* problem,
                           ceres::LossFunction* loss_function, 
                           Block* block);
    
    void AddPointToProblem(const mappoint_t mappoint_id,
                            Reconstruction* reconstruction,
                            ceres::Problem* problem,
                            ceres::LossFunction* loss_function, 
                            Block* block);
    
    void AddNovatelToProblem(const image_t image_id, 
                             Reconstruction* reconstruction,
                             ceres::Problem* problem,
                             ceres::LossFunction* loss_function, 
                             Block* block);

    void AddGnssExtriToProble(const camera_t camera_id, 
                              Reconstruction *reconstruction, 
                              ceres::Problem* problem,
                              ceres::LossFunction *loss_function, 
                              Block* block);

    void ParameterizeCameras(Reconstruction* reconstruction, ceres::Problem* problem, Block* block);
    void ParameterizePoints(Reconstruction* reconstruction, ceres::Problem* problem, Block* block);
    void ParameterizePointsWithoutTrack(Reconstruction *reconstruction, ceres::Problem* problem, Block* block);
    void ParameterizePoses(Reconstruction* reconstruction, ceres::Problem* problem, Block* block);

    inline void SetParameterization(ceres::Problem *problem, double* values, ceres::LocalParameterization *local_parameterization);

private:
    const BundleAdjustmentOptions options_;
    BlockBundleAdjustmentOptions block_options_;

    BundleAdjustmentConfig config_;

    std::unordered_set<image_t> constant_pose_images_;
    Eigen::Vector3d gravity_;
};

void PrintSolverSummary(const ceres::Solver::Summary& summary);

}

#endif
