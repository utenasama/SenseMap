// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef UTIL_LARGEFOVIMAGE_H_
#define UTIL_LARGEFOVIMAGE_H_

#include "base/camera.h"
#include "bitmap.h"
#include "util/types.h"

namespace sensemap {

class LargeFovImage {
public:
    LargeFovImage(){
        image_width_ = 0;
        image_height_ = 0;
    }
    void SetCamera(const Camera& camera);
    void ParamPreprocess(int perspective_width, int perspective_height, double fov_w, int image_width,
                         int image_height);

    bool ToPerspective(const Bitmap& img_in, Bitmap& img_out, const int perspective_id);
    void ConvertPerspectiveCoordToOriginal(const double u_in, const double v_in, const int perspective_id,
                                           double& u_out, double& v_out);

    int GetImageWidth();
    int GetImageHeight();

private:
    int image_width_, image_height_;
    int perspective_width_, perspective_height_;

    // Output image focal length
    double focal_length_;

    std::vector<std::vector<double>> map_xs_;
    std::vector<std::vector<double>> map_ys_;

    Camera camera_;
};

}  // namespace sensemap
#endif