// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef UTIL_PIECEWISEIMAGE_H_
#define UTIL_PIECEWISEIMAGE_H_

#include "base/camera.h"
#include "bitmap.h"
#include "util/types.h"

namespace sensemap {

class PiecewiseImage {
public:
    PiecewiseImage(){
        image_width_ = 0;
        image_height_ = 0;
    }
    void SetCamera(const Camera& camera);
    void ParamPreprocess(int perspective_width, int perspective_height, 
                         double fov_w, int image_width, int image_height, 
                         int piece_num = 3);
    void ParamPreprocess(int perspective_width, double fov_w, double fov_h, 
                         int image_width, int image_height, int piece_num = 3);

    // Get pixel map from panorama image to perspective image
    std::vector<Eigen::RowMatrixXi> GetPiecewiseRmapId() const;

    std::vector<double> GetPiecewiseRMapX() const;

    std::vector<double> GetPiecewiseRMapY() const;

    // bool ToPerspective(const Bitmap& img_in, Bitmap& img_out, const int perspective_id);
    // void ConvertPerspectiveCoordToOriginal(const double u_in, const double v_in, const int perspective_id,
    //                                        double& u_out, double& v_out);



    bool ToSplitedPerspectives(const Bitmap& img_in, std::vector<Bitmap>& imgs_out, const int local_camera_id);
    void ConvertSplitedPerspectiveCoordToOriginal(const double u_in, const double v_in, const int local_camera_id, const int piece_id,
                                           double& u_out, double& v_out);
    

    int GetImageWidth();
    int GetImageHeight();

private:
    int image_width_, image_height_;
    int perspective_width_, perspective_height_;

    // Output image focal length
    double focal_length_;

    // Create the camera transform matrix
    std::vector<Eigen::Matrix3d> transforms_;
    std::vector<Eigen::Vector2d> offsets_;


    // coordinate map from piecewise image to the original image 
    std::vector<Eigen::RowMatrixXi> rmap_idxs_;

    // coordinate map from the splited images to the original image
    std::vector<std::vector<double>> splited_map_xs_;
    std::vector<std::vector<double>> splited_map_ys_;

    // coordinate map from origin image to the splited images.
    std::vector<double> rmap_x_;
    std::vector<double> rmap_y_;

    Camera camera_;

    int piece_count_;
};

}  // namespace sensemap
#endif