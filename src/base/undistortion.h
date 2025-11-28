//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_UNDISTORTION_H_
#define SENSEMAP_BASE_UNDISTORTION_H_

#include "util/threading.h"
#include "util/bitmap.h"
#include "base/reconstruction.h"

#include "base/pose.h"
#include "util/rgbd_helper.h"

namespace sensemap {

struct UndistortOptions {
    double blank_pixels = 0;

    // // Whether keep origin image size. If max_image_size is setted, the variable
    // // is invalid and the image size is set according to max_image_size.
    // bool as_origin = true;

    int max_image_size = -1;
    int max_rig_image_size = -1;

    double fov_w = -1.0;
    double fov_h = -1.0;

    double min_scale = 0.2;
    double max_scale = 2.0;

    bool verbose = false;

    // Whether keep original resolution. If as_orig_res is true,
    // max_image_size is set to -1 to ensure original resolution,
    // else image size is set according to max_image_size.
    bool as_orig_res = false;

    // Extract prior depths
    bool as_rgbd = false;

    // Intrinsic infomation for prior depth->camera warps
    std::string rgbd_camera_params;

    // Depth limit for prior depth
    int warp_depth_sparsely = 0;
    int reverse_scale_recovery = 0;
    float min_prior_depth = 0.0f;
    float max_prior_depth = 0.0f;

    // Whether to enable cross-warp between subpaths
    int cross_warp_num_images = 8;
    std::vector<std::string> cross_warp_subpath;

    // Whether to filter images in certain subpaths
    std::vector<std::string> filter_subpath;

    // Mask path to perform undistortion
    std::string mask_path;
    bool postprocess = false;
};

struct PriorDepthInfo {
    MatXf depthmap; // depth data
    Eigen::Matrix3f K;  // depth intrinsic
    Eigen::Matrix4f RT; // depth RT to color
};
class CrossWarpHelper;

class Undistorter : public Thread {
public:
    static void WarpImageBetweenCameras(const Camera& source_camera,
                                 const Camera& target_camera,
                                 const Bitmap& source_image,
                                 Bitmap* target_image);
    static void WarpMaskBetweenCameras(const Camera& source_camera,
                                 const Camera& target_camera,
                                 const Bitmap& source_image,
                                 Bitmap* target_image);
    static void UndistortCamera(const UndistortOptions& options,
                         const Camera& camera,
                         Camera* undistorted_camera);
    static void UndistortImage(const UndistortOptions& options,
                        const Bitmap& distorted_bitmap,
                        const Camera& distorted_camera,
                        Bitmap* undistorted_bitmap,
                        Camera* undistorted_camera);
    static void UndistortReconstruction(const UndistortOptions& options,
                                 Reconstruction* reconstruction);
    static std::map<image_t, std::shared_ptr<PriorDepthInfo>> InitPriorDepths(
                        const std::string & image_path,
                        const Reconstruction & reconstruction,
                        const UndistortOptions & options);
    static void WarpPriorDepth(
                        image_t image_id,
                        const Camera & undistorted_camera,
                        const Reconstruction & reconstruction,
                        const std::map<image_t, std::shared_ptr<PriorDepthInfo>> & prior_depths,
                        const std::string & undistort_depths_path,
                        const UndistortOptions & options);
    static void CrossWarpPriorDepth(
                        image_t image_id,
                        const Camera & undistorted_camera,
                        const Reconstruction & reconstruction,
                        const CrossWarpHelper & cross_warp_helper,
                        const std::map<image_t, std::shared_ptr<PriorDepthInfo>> & prior_depths,
                        const std::string & undistort_depths_path,
                        const UndistortOptions & options);
    static double ReverseScaleRecovery(
                        std::map<image_t, std::shared_ptr<PriorDepthInfo>> & prior_depths,
                        const Reconstruction & reconstruction);

public:
    Undistorter(const UndistortOptions& options,
                const std::string& image_path,
                const std::string& workspace_path);
private:
    void Run();
    void Undistort(const size_t reg_image_idx) const;
    void CreateWorkspace();
    void WritePatchMatchConfig() const;
    void WriteFusionConfig() const;

private:
    UndistortOptions options_;
    std::string image_path_;
    std::string workspace_path_;
    Reconstruction reconstruction_;

    std::string undistort_images_path_;
    std::string undistort_sparse_path_;
    std::string undistort_depths_path_;
    std::string undistort_masks_path_;

    std::shared_ptr<CrossWarpHelper> cross_warp_helper_;
    std::map<image_t, std::shared_ptr<PriorDepthInfo>> prior_depths_;
    std::unordered_set<image_t> filter_image_ids_;
};

}

#endif