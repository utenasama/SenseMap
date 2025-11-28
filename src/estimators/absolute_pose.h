//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_ABSOLUTE_POSE_H_
#define SENSEMAP_ESTIMATORS_ABSOLUTE_POSE_H_

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "util/alignment.h"
#include "util/types.h"
#include "util/logging.h"
#include "util/threading.h"
#include "base/camera.h"
#include "optim/ransac/loransac.h"

namespace sensemap {

// Analytic solver for the P3P (Perspective-Three-Point) problem.
//
// The algorithm is based on the following paper:
//
//    X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang. Complete Solution
//    Classification for the Perspective-Three-Point Problem.
//    http://www.mmrc.iss.ac.cn/~xgao/paper/ieee.pdf
class P3PEstimator {
 public:
  // The 2D image feature observations.
  typedef Eigen::Vector2d X_t;
  // The observed 3D features in the world frame.
  typedef Eigen::Vector3d Y_t;
  // The transformation from the world to the camera frame.
  typedef Eigen::Matrix3x4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate the most probable solution of the P3P problem from a set of
  // three 2D-3D point correspondences.
  //
  // @param points2D   Normalized 2D image points as 3x2 matrix.
  // @param points3D   3D world points as 3x3 matrix.
  //
  // @return           Most probable pose as length-1 vector of a 3x4 matrix.
  std::vector<M_t> Estimate(const std::vector<X_t>& points2D,
                            const std::vector<Y_t>& points3D);

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D     Normalized 2D image points as Nx2 matrix.
  // @param points3D     3D world points as Nx3 matrix.
  // @param proj_matrix  3x4 projection matrix.
  // @param residuals    Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& proj_matrix, std::vector<double>* residuals);

};

// P3P estimator for spherical camera
class P3PEstimatorSpherical {
 public:
  // The 2D image feature observations.
  typedef Eigen::Vector3d X_t;
  // The observed 3D features in the world frame.
  typedef Eigen::Vector3d Y_t;
  // The transformation from the world to the camera frame.
  typedef Eigen::Matrix3x4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate the most probable solution of the P3P problem from a set of
  // three 2D-3D point correspondences.
  //
  // @param points2D   Normalized 2D image points as 3x2 matrix.
  // @param points3D   3D world points as 3x3 matrix.
  //
  // @return           Most probable pose as length-1 vector of a 3x4 matrix.
  std::vector<M_t> Estimate(const std::vector<X_t>& points2D_bearing,
                            const std::vector<Y_t>& points3D);

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D     Normalized 2D image points as Nx2 matrix.
  // @param points3D     3D world points as Nx3 matrix.
  // @param proj_matrix  3x4 projection matrix.
  // @param residuals    Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D_bearing,
                 const std::vector<Y_t>& points3D,
                 const M_t& proj_matrix, std::vector<double>* residuals);
};




// EPNP solver for the PNP (Perspective-N-Point) problem. The solver needs a
// minimum of 4 2D-3D correspondences.
//
// The algorithm is based on the following paper:
//
//    Lepetit, Vincent, Francesc Moreno-Noguer, and Pascal Fua.
//    "Epnp: An accurate o (n) solution to the pnp problem."
//    International journal of computer vision 81.2 (2009): 155-166.
//
// The implementation is based on their original open-source release, but is
// ported to Eigen and contains several improvements over the original code.
class EPNPEstimator {
 public:
  // The 2D image feature observations.
  typedef Eigen::Vector2d X_t;
  // The observed 3D features in the world frame.
  typedef Eigen::Vector3d Y_t;
  // The transformation from the world to the camera frame.
  typedef Eigen::Matrix3x4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 4;
 
  // Estimate the most probable solution of the P3P problem from a set of
  // three 2D-3D point correspondences.
  //
  // @param points2D   Normalized 2D image points as 3x2 matrix.
  // @param points3D   3D world points as 3x3 matrix.
  //
  // @return           Most probable pose as length-1 vector of a 3x4 matrix.
  std::vector<M_t> Estimate(const std::vector<X_t>& points2D,
                                   const std::vector<Y_t>& points3D);

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D     Normalized 2D image points as Nx2 matrix.
  // @param points3D     3D world points as Nx3 matrix.
  // @param proj_matrix  3x4 projection matrix.
  // @param residuals    Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& proj_matrix, std::vector<double>* residuals);

 private:
  bool ComputePose(const std::vector<Eigen::Vector2d>& points2D,
                   const std::vector<Eigen::Vector3d>& points3D,
                   Eigen::Matrix3x4d* proj_matrix);

  void ChooseControlPoints();
  bool ComputeBarycentricCoordinates();

  Eigen::Matrix<double, Eigen::Dynamic, 12> ComputeM();
  Eigen::Matrix<double, 6, 10> ComputeL6x10(
      const Eigen::Matrix<double, 12, 12>& Ut);
  Eigen::Matrix<double, 6, 1> ComputeRho();

  void FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);

  void RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L_6x10,
                      const Eigen::Matrix<double, 6, 1>& rho,
                      Eigen::Vector4d* betas);

  double ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut,
                   const Eigen::Vector4d& betas, Eigen::Matrix3d* R,
                   Eigen::Vector3d* t);

  void ComputeCcs(const Eigen::Vector4d& betas,
                  const Eigen::Matrix<double, 12, 12>& Ut);
  void ComputePcs();

  void SolveForSign();

  void EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t);

  double ComputeTotalReprojectionError(const Eigen::Matrix3d& R,
                                       const Eigen::Vector3d& t);

  const std::vector<Eigen::Vector2d>* points2D_ = nullptr;
  const std::vector<Eigen::Vector3d>* points3D_ = nullptr;
  std::vector<Eigen::Vector3d> pcs_;
  std::vector<Eigen::Vector4d> alphas_;
  std::array<Eigen::Vector3d, 4> cws_;
  std::array<Eigen::Vector3d, 4> ccs_;
   
};

//For spherical camera

class EPNPEstimatorSpherical {
 public:
  // The 2D image feature observations.
  typedef Eigen::Vector3d X_t;
  // The observed 3D features in the world frame.
  typedef Eigen::Vector3d Y_t;
  // The transformation from the world to the camera frame.
  typedef Eigen::Matrix3x4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 4;
 
  // Estimate the most probable solution of the P3P problem from a set of
  // three 2D-3D point correspondences.
  //
  // @param points2D   Normalized 2D image points as 3x2 matrix.
  // @param points3D   3D world points as 3x3 matrix.
  //
  // @return           Most probable pose as length-1 vector of a 3x4 matrix.
  std::vector<M_t> Estimate(const std::vector<X_t>& points2D_bearing,
                            const std::vector<Y_t>& points3D);

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D     Normalized 2D image points as Nx2 matrix.
  // @param points3D     3D world points as Nx3 matrix.
  // @param proj_matrix  3x4 projection matrix.
  // @param residuals    Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D_bearing,
                 const std::vector<Y_t>& points3D,
                 const M_t& proj_matrix, std::vector<double>* residuals);
  const std::vector<Eigen::Vector3d>* points2D_bearing_ = nullptr;
 private:
  bool ComputePose(const std::vector<Eigen::Vector2d>& points2D,
                   const std::vector<Eigen::Vector3d>& points3D,
                   Eigen::Matrix3x4d* proj_matrix);

  void ChooseControlPoints();
  bool ComputeBarycentricCoordinates();

  Eigen::Matrix<double, Eigen::Dynamic, 12> ComputeM();
  Eigen::Matrix<double, 6, 10> ComputeL6x10(
      const Eigen::Matrix<double, 12, 12>& Ut);
  Eigen::Matrix<double, 6, 1> ComputeRho();

  void FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);

  void RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L_6x10,
                      const Eigen::Matrix<double, 6, 1>& rho,
                      Eigen::Vector4d* betas);

  double ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut,
                   const Eigen::Vector4d& betas, Eigen::Matrix3d* R,
                   Eigen::Vector3d* t);

  void ComputeCcs(const Eigen::Vector4d& betas,
                  const Eigen::Matrix<double, 12, 12>& Ut);
  void ComputePcs();

  void SolveForSign();

  void EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t);

  double ComputeTotalReprojectionError(const Eigen::Matrix3d& R,
                                       const Eigen::Vector3d& t);

  const std::vector<Eigen::Vector2d>* points2D_ = nullptr;
  const std::vector<Eigen::Vector3d>* points3D_ = nullptr;

  std::vector<Eigen::Vector3d> pcs_;
  std::vector<Eigen::Vector4d> alphas_;
  std::array<Eigen::Vector3d, 4> cws_;
  std::array<Eigen::Vector3d, 4> ccs_;
   
};




struct AbsolutePoseEstimationOptions {
    // Whether to estimate the focal length.
    bool estimate_focal_length = false;

    // Number of discrete samples for focal length estimation.
    size_t num_focal_length_samples = 10;

    // Minimum focal length ratio for discrete focal length sampling
    // around focal length of given camera.
    double min_focal_length_ratio = 0.5;

    // Maximum focal length ratio for discrete focal length sampling
    // around focal length of given camera.
    double max_focal_length_ratio = 2;

    // Number of threads for parallel estimation of focal length.
    int num_threads = ThreadPool::kMaxNumThreads;

    // Options used for P3P RANSAC.
    RANSACOptions ransac_options;

    void Check() const {
        CHECK_GT(num_focal_length_samples, 0);
        CHECK_GT(min_focal_length_ratio, 0);
        CHECK_GT(max_focal_length_ratio, 0);
        CHECK_LT(min_focal_length_ratio, max_focal_length_ratio);
        ransac_options.Check();
    }
};

struct AbsolutePoseRefinementOptions {
    // Convergence criterion.
    double gradient_tolerance = 1.0;

    // Maximum number of solver iterations.
    int max_num_iterations = 100;

    // Scaling factor determines at which residual robustification takes place.
    double loss_function_scale = 1.0;

    // Lower bound for focal length
    double lower_bound_focal_length_factor = 0.5;
    
    // Upper bound for focal length
    double upper_bound_focal_length_factor = 1.5;

    // Whether to refine the focal length parameter group.
    bool refine_focal_length = true;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = true;

    // Whether to print final summary.
    bool print_summary = true;

    // Vector of point depths
    std::vector<float> point_depths;

    // Vector of point depths weights
    std::vector<float> point_depths_weights;

    void Check() const {
        CHECK_GE(gradient_tolerance, 0.0);
        CHECK_GE(max_num_iterations, 0);
        CHECK_GE(loss_function_scale, 0.0);
    }
};

// Estimate absolute pose (optionally focal length) from 2D-3D correspondences.
//
// Focal length estimation is performed using discrete sampling around the
// focal length of the given camera. The focal length that results in the
// maximal number of inliers is assigned to the given camera.
//
// @param options              Absolute pose estimation options.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param qvec                 Estimated rotation component as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component.
// @param camera               Camera for which to estimate pose. Modified
//                             in-place to store the estimated focal length.
// @param num_inliers          Number of inliers in RANSAC.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
//
// @return                     Whether pose is estimated successfully.
bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                          Camera* camera, size_t* num_inliers,
                          std::vector<char>* inlier_mask);

bool EstimateAbsolutePoses(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          const Camera* camera,
                          std::vector<double>& estimated_focal_length_factors,
                          std::vector<Eigen::Vector4d>& qvecs,
                          std::vector<Eigen::Vector3d>& tvecs);

// Refine absolute pose (optionally focal length) from 2D-3D correspondences.
//
// @param options              Refinement options.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param qvec                 Estimated rotation component as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component.
// @param camera               Camera for which to estimate pose. Modified
//                             in-place to store the estimated focal length.
//
// @return                     Whether the solution is usable.
bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        const uint64_t num_reg_images,
                        std::vector<uint64_t> mappoints_create_time,
                        const std::vector<double>& mappoint_weights,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        Camera* camera);



//for camera-rig
bool RefineAbsolutePoseRig(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        const uint64_t num_reg_images, 
                        std::vector<uint64_t> mappoints_create_time, 
                        const std::vector<double>& mappoint_weights,
                        const std::vector<int>& local_camera_indices,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        Camera* camera);




}  // namespace sensemap

#endif  // SENSEMAP_ESTIMATORS_ABSOLUTE_POSE_H_
