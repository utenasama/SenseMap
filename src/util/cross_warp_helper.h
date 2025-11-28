#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "base/image.h"
#include "base/camera.h"
#include "base/matrix.h"
#include "controllers/patch_match_controller.h"

namespace sensemap {
class CrossWarpHelper {
public:
    virtual std::vector<image_t> GetNeighboringImages(image_t ref_image_id) const = 0;
};

class PatchMatchCrossWarpHelper : public mvs::PatchMatchController, public CrossWarpHelper {
public:
    PatchMatchCrossWarpHelper(const std::string &workspace_path, const std::unordered_set<image_t> &prior_depth_images, int num_images = 8);
    std::vector<image_t> GetNeighboringImages(image_t ref_image_id) const override;

private:
    const int num_images_;
    std::unordered_set<int> prior_depth_images_;

    bool SelectSpatialNeighborViews(mvs::Problem &problem) const;
};
}