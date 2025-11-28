//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_TEXTURE_MESH_TEXTURE_H_
#define SENSEMAP_TEXTURE_MESH_TEXTURE_H_

#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "util/alignment.h"
#include "util/types.h"
#include "util/math.h"
#include "util/ply.h"
#include "util/threading.h"
#include "util/bitmap.h"
#include "base/common.h"

#include "texture/utils.h"

namespace sensemap {
namespace texture {

class TextureMapping : public Thread {
public:
    struct Options {
        std::string workspace_path;
        std::string image_folder = JoinPaths(DENSE_DIR, IMAGES_DIR);
        std::string depth_folder = JoinPaths(DENSE_DIR, STEREO_DIR, DEPTHS_DIR);
        std::string model_name = "";

        double maximum_allowable_depth_ = 2.5;
        double depth_threshold_for_visiblity_check_ = 0.03;
        int image_boundary_margin_ = 10;
    };

    TextureMapping(const Options& options);


protected:

    void Run();
    void Mapping();

    std::vector<std::shared_ptr<RGBDImage>> rgbd_images_;
    CameraTrajectory camera_;
    TriangleMesh mesh_;

    Options options_;

};

}  // namespace texture
}  // namespace sensemap

#endif  // SENSEMAP_TEXTURE_MESH_TEXTURE_H_
