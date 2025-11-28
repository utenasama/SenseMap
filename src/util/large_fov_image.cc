// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "large_fov_image.h"
#include "util/threading.h"
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace sensemap {

void LargeFovImage::SetCamera(const Camera& camera) { camera_ = camera; }

void LargeFovImage::ParamPreprocess(int perspective_width, int perspective_height, double fov_w, int image_width,
                                    int image_height) {
    image_width_ = image_width;
    image_height_ = image_height;

    perspective_width_ = perspective_width;
    perspective_height_ = perspective_height;

    focal_length_ = perspective_width * 0.5 / tan(fov_w / 360.0 * M_PI);


    map_xs_.clear();
    map_ys_.clear();

    std::vector<double> cur_map_x, cur_map_y;

    std::unique_ptr<ThreadPool> thread_pool;
    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    auto Remapping = [&](int local_camera_id, int thread_id) {
        int height_slice = (perspective_height_ + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (thread_id + 1), perspective_height_);
        for (int v = height_slice * thread_id; v < height_limit; ++v) {
            for (int u = 0; u < perspective_width_; u++) {
                Eigen::Vector2d point_in;
                point_in(0) = static_cast<double>(u - perspective_width_ / 2) / focal_length_;
                point_in(1) = static_cast<double>(v - perspective_height_ / 2) / focal_length_;
                Eigen::Vector2d point_out;
                if(camera_.NumLocalCameras()>1){
                    point_out = camera_.WorldToLocalImage(local_camera_id, point_in);
                }
                else{
                    point_out = camera_.WorldToImage(point_in);
                }
                cur_map_x[v * perspective_width_ + u] = point_out(0);
                cur_map_y[v * perspective_width_ + u] = point_out(1);
            }
        }
    };

    for (int local_camera_id = 0; local_camera_id < camera_.NumLocalCameras(); ++local_camera_id) {
        cur_map_x.resize(perspective_height_ * perspective_width_);
        cur_map_y.resize(perspective_height_ * perspective_width_);

        for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
            thread_pool->AddTask(Remapping, local_camera_id, thread_idx);
        };
        thread_pool->Wait();
        map_xs_.emplace_back(cur_map_x);
        map_ys_.emplace_back(cur_map_y);
    }
}

bool LargeFovImage::ToPerspective(const Bitmap& img_in, Bitmap& img_out, const int perspective_id) {
    if (img_in.Width() != image_width_ || img_in.Height() != image_height_) {
        std::cout << "Error! Input image size error ... " << std::endl;
        return false;
    }

    if (img_in.Channels() == 3) {
        img_out.Allocate(perspective_width_, perspective_height_, true);
    } else {
        img_out.Allocate(perspective_width_, perspective_height_, false);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    // Apply remap to update the output image
    for (int y = 0; y < perspective_height_; ++y) {
        for (int x = 0; x < perspective_width_; ++x) {
            BitmapColor<float> color;

            img_in.InterpolateBilinear(map_xs_[perspective_id][y * perspective_width_ + x],
                                       map_ys_[perspective_id][y * perspective_width_ + x], &color);

            BitmapColor<uint8_t> color_uint;
            color_uint.r = color.r > 255.0 ? 255 : static_cast<uint8_t>(color.r);
            color_uint.g = color.g > 255.0 ? 255 : static_cast<uint8_t>(color.g);
            color_uint.b = color.b > 255.0 ? 255 : static_cast<uint8_t>(color.b);
            img_out.SetPixel(x, y, color_uint);
        }
    }
    return true;
}

void LargeFovImage::ConvertPerspectiveCoordToOriginal(const double u_in, const double v_in, const int perspective_id,
                                                      double& u_out, double& v_out) {

    Eigen::Vector2d point_in;
    point_in(0) = static_cast<double>(u_in - perspective_width_ / 2) / focal_length_;
    point_in(1) = static_cast<double>(v_in - perspective_height_ / 2) / focal_length_;
    
    Eigen::Vector2d point_out;
    if (camera_.NumLocalCameras() > 1) {
        point_out = camera_.WorldToLocalImage(perspective_id, point_in);
    } else {
        point_out = camera_.WorldToImage(point_in);
    }

    u_out = point_out(0);
    v_out = point_out(1);
}


int LargeFovImage::GetImageHeight(){
    return image_height_;
}

int LargeFovImage::GetImageWidth(){
    return image_width_;
}

}  // namespace sensemap
