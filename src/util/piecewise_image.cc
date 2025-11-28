// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "piecewise_image.h"
#include "util/threading.h"

#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace sensemap {

void PiecewiseImage::SetCamera(const Camera& camera) { camera_ = camera; }

void PiecewiseImage::ParamPreprocess(int perspective_width, int perspective_height, double fov_w, int image_width,
                                    int image_height, int piece_num) {
    image_width_ = image_width;
    image_height_ = image_height;

    perspective_width_ = perspective_width;
    perspective_height_ = perspective_height;

    focal_length_ = perspective_width * 0.5 / tan(fov_w / 360.0 * M_PI);
    piece_count_ = piece_num;

    int cx = (perspective_width_ >> 1);
    int cy = (perspective_height_ >> 1);
    int perspective_size = perspective_height_ * perspective_width_;

    // double roll[5] = {0,0,0,0,0};
    // double yaw[5] = {0,0,0,-60,60};
    // double pitch[5] = {60,0,-60,0,0};
    
    double roll[5] = {0,0,0,0,0};
    double yaw[5]  =  {-60,0,60,0,0};
    double pitch[5] = {0,0,0,-60,60};

    // get piecewise transforms    
    for (size_t i = 0; i < piece_count_ ; ++i) {
        Eigen::Matrix3d transform;
        transform = Eigen::AngleAxisd(yaw[i] / 180 * M_PI, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(pitch[i] / 180 * M_PI, Eigen::Vector3d::UnitX()) *
                    Eigen::AngleAxisd(roll[i] / 180 * M_PI, Eigen::Vector3d::UnitZ());
        transforms_.emplace_back(transform);
    }

    const local_camera_t num_local_cameras = camera_.NumLocalCameras();

    splited_map_ys_.clear();
    splited_map_xs_.clear();

    const int num_piece = num_local_cameras * piece_count_;
    std::vector<std::vector<double> > cur_splited_map_x(num_piece), cur_splited_map_y(num_piece);
    for (int i = 0; i < num_piece; ++i) {
        cur_splited_map_x[i].resize(perspective_size, -1.0);
        cur_splited_map_y[i].resize(perspective_size, -1.0);
    }

    std::unique_ptr<ThreadPool> thread_pool;
    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    const double inv_focal_length = 1.0 / focal_length_;

    auto GenerateSplittedMap = [&](int local_camera_id, int piece_id, int thread_id) {
        int height_slice = (perspective_height_ + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (thread_id + 1), perspective_height_);
        const int splited_map_id = local_camera_id * piece_count_ + piece_id;
        for (int v = height_slice * thread_id; v < height_limit; ++v) {
            int pitch = v * perspective_width_;
            for (int u = 0; u < perspective_width_; u++) {
                Eigen::Vector2d point_in;
                point_in(0) = static_cast<double>(u - cx) * inv_focal_length;
                point_in(1) = static_cast<double>(v - cy) * inv_focal_length;

                Eigen::Vector3d point = point_in.homogeneous();
                Eigen::Vector3d bearing = transforms_[piece_id] * point;

                Eigen::Vector2d point_out;
                if (camera_.NumLocalCameras() > 1) {
                    point_out = camera_.BearingToLocalImage(local_camera_id, bearing);
                } else {
                    point_out = camera_.BearingToImage(bearing);
                }

                cur_splited_map_x[splited_map_id][pitch + u] = point_out(0);
                cur_splited_map_y[splited_map_id][pitch + u] = point_out(1);
            }
        }
    };

    double u, v;
    std::vector<Eigen::RowMatrixXi> rmap_idxs(num_local_cameras);
    for (int i = 0; i < num_local_cameras; ++i) {
        rmap_idxs[i] = Eigen::RowMatrixXi::Constant(image_height, image_width, -1);
    }

    rmap_x_.resize(image_height * image_width * num_local_cameras);
    rmap_y_.resize(image_height * image_width * num_local_cameras);
    std::fill(rmap_x_.begin(), rmap_x_.end(), -1.0);
    std::fill(rmap_y_.begin(), rmap_y_.end(), -1.0);

    auto GenerateRemapIdx = [&](int local_camera_id, int piece_id, int thread_id, 
                                Eigen::Matrix3d transform_inv) {
        
        int height_slice = (image_height + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (thread_id + 1), image_height);
        const int local_image_idx = local_camera_id * piece_count_ + piece_id;
        const size_t local_remap_base = local_camera_id * image_height * image_width;
        for (int y = height_slice * thread_id; y < height_limit; ++y) {
            for (int x = 0; x < image_width; ++x) {
                Eigen::Vector2d image_point(x, y);
                Eigen::Vector3d bearing;
                if (num_local_cameras > 1) {
                    bearing = camera_.LocalImageToBearing(local_camera_id, image_point);
                } else {
                    bearing = camera_.ImageToBearing(image_point);
                }
                bearing = transform_inv * bearing;
                Eigen::Vector2d point = (bearing / bearing.z()).head<2>();
                u = point.x() * focal_length_ + cx;
                v = point.y() * focal_length_ + cy;
                if (u < 0 || u >= perspective_width_ ||
                    v < 0 || v >= perspective_height_) {
                    continue;
                }
                rmap_x_[local_remap_base + y * image_width + x] = u;
                rmap_y_[local_remap_base + y * image_width + x] = v;
                rmap_idxs[local_camera_id](y, x) = local_image_idx;
            }
        }
    };

    for (int local_camera_id = 0; local_camera_id < num_local_cameras; ++local_camera_id) {
        for (size_t piece_id = 0; piece_id < piece_count_; ++piece_id) {
            for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
              thread_pool->AddTask(GenerateSplittedMap, local_camera_id, piece_id, thread_idx);
            };
            // Update rmap_x and rmap_y
            Eigen::Matrix3d transform_inv = transforms_.at(piece_id).inverse();
            for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
                thread_pool->AddTask(GenerateRemapIdx, local_camera_id, piece_id, thread_idx, transform_inv);
            };
        }
    }
    thread_pool->Wait();
    splited_map_xs_.swap(cur_splited_map_x);
    splited_map_ys_.swap(cur_splited_map_y);
    rmap_idxs_.swap(rmap_idxs);
}

void PiecewiseImage::ParamPreprocess(int perspective_width, 
                                     double fov_w, double fov_h, 
                                     int image_width, int image_height,
                                     int piece_num) {
    image_width_ = image_width;
    image_height_ = image_height;

    focal_length_ = perspective_width * 0.5 / tan(fov_w / 360.0 * M_PI);
    int perspective_height = focal_length_ * tan(fov_h / 360.0 * M_PI) * 2;
    perspective_width_ = perspective_width;
    perspective_height_ = perspective_height;
    
    int cx = (perspective_width_ >> 1);
    int cy = (perspective_height_ >> 1);
    int perspective_size = perspective_height_ * perspective_width_;

    piece_count_ = piece_num;
    
    double roll[5] = {0,0,0,0,0};
    double yaw[5]  =  {-60,0,60,0,0};
    double pitch[5] = {0,0,0,-60,60};

    // get piecewise transforms    
    for (size_t i = 0; i < piece_count_ ; ++i) {
        Eigen::Matrix3d transform;
        transform = Eigen::AngleAxisd(yaw[i] / 180 * M_PI, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(pitch[i] / 180 * M_PI, Eigen::Vector3d::UnitX()) *
                    Eigen::AngleAxisd(roll[i] / 180 * M_PI, Eigen::Vector3d::UnitZ());
        transforms_.emplace_back(transform);
    }

    const local_camera_t num_local_cameras = camera_.NumLocalCameras();

    splited_map_ys_.clear();
    splited_map_xs_.clear();
    rmap_idxs_.resize(num_local_cameras);

    const int num_piece = num_local_cameras * piece_count_;
    std::vector<std::vector<double> > cur_splited_map_x(num_piece), cur_splited_map_y(num_piece);
    for (int i = 0; i < num_piece; ++i) {
        cur_splited_map_x[i].resize(perspective_size, -1.0);
        cur_splited_map_y[i].resize(perspective_size, -1.0);
    }
    
    const double inv_focal_length = 1.0 / focal_length_;

    std::unique_ptr<ThreadPool> thread_pool;
    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    auto GenerateSplittedMap = [&](int local_camera_id, int piece_id, int thread_id) {
        int height_slice = (perspective_height_ + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (thread_id + 1), perspective_height_);
        const int splited_map_id = local_camera_id * piece_count_ + piece_id;
        for (int v = height_slice * thread_id; v < height_limit; ++v) {
            int pitch = v * perspective_width_;
            for (int u = 0; u < perspective_width_; u++) {
                Eigen::Vector2d point_in;
                point_in(0) = static_cast<double>(u - cx) * inv_focal_length;
                point_in(1) = static_cast<double>(v - cy) * inv_focal_length;

                Eigen::Vector3d bearing = transforms_[piece_id] * point_in.homogeneous();

                Eigen::Vector2d point_out;
                if (num_local_cameras > 1) {
                    point_out = camera_.BearingToLocalImage(local_camera_id, bearing);
                } else {
                    point_out = camera_.BearingToImage(bearing);
                }

                cur_splited_map_x[splited_map_id][pitch + u] = point_out(0);
                cur_splited_map_y[splited_map_id][pitch + u] = point_out(1);
            }
        }
    };

    double u, v;
    std::vector<Eigen::RowMatrixXi> rmap_idxs(num_local_cameras);
    for (int i = 0; i < num_local_cameras; ++i) {
        rmap_idxs[i] = Eigen::RowMatrixXi::Constant(image_height, image_width, -1);
    }

    rmap_x_.resize(image_height * image_width * num_local_cameras);
    rmap_y_.resize(image_height * image_width * num_local_cameras);
    std::fill(rmap_x_.begin(), rmap_x_.end(), -1.0);
    std::fill(rmap_y_.begin(), rmap_y_.end(), -1.0);

    auto GenerateRemapIdx = [&](int local_camera_id, int piece_id, int thread_id, 
                                Eigen::Matrix3d transform_inv) {
        
        int height_slice = (image_height + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (thread_id + 1), image_height);
        const int local_image_idx = local_camera_id * piece_count_ + piece_id;
        const size_t local_remap_base = local_camera_id * image_height * image_width;
        for (int y = height_slice * thread_id; y < height_limit; ++y) {
            for (int x = 0; x < image_width; ++x) {
                Eigen::Vector2d image_point(x, y);
                Eigen::Vector3d bearing;
                if (num_local_cameras > 1) {
                    bearing = camera_.LocalImageToBearing(local_camera_id, image_point);
                } else {
                    bearing = camera_.ImageToBearing(image_point);
                }
                bearing = transform_inv * bearing;
                Eigen::Vector2d point = (bearing / bearing.z()).head<2>();
                u = point.x() * focal_length_ + cx;
                v = point.y() * focal_length_ + cy;
                if (u < 0 || u >= perspective_width_ ||
                    v < 0 || v >= perspective_height_) {
                    continue;
                }
                rmap_x_[local_remap_base + y * image_width + x] = u;
                rmap_y_[local_remap_base + y * image_width + x] = v;
                rmap_idxs[local_camera_id](y, x) = local_image_idx;
            }
        }
    };

    for (int local_camera_id = 0; local_camera_id < num_local_cameras; ++local_camera_id) {
        for (size_t piece_id = 0; piece_id < piece_count_; ++piece_id) {
            for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
              thread_pool->AddTask(GenerateSplittedMap, local_camera_id, piece_id, thread_idx);
            };
            // Update rmap_x and rmap_y
            Eigen::Matrix3d transform_inv = transforms_.at(piece_id).inverse();
            for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
                thread_pool->AddTask(GenerateRemapIdx, local_camera_id, piece_id, thread_idx, transform_inv);
            };
        }
    }
    thread_pool->Wait();

    splited_map_xs_.swap(cur_splited_map_x);
    splited_map_ys_.swap(cur_splited_map_y);
    rmap_idxs_.swap(rmap_idxs);
}

std::vector<Eigen::RowMatrixXi> PiecewiseImage::GetPiecewiseRmapId() const {
    return rmap_idxs_;
}

std::vector<double> PiecewiseImage::GetPiecewiseRMapX() const {
    return rmap_x_;
}

std::vector<double> PiecewiseImage::GetPiecewiseRMapY() const {
    return rmap_y_;
}

bool PiecewiseImage::ToSplitedPerspectives(const Bitmap& img_in, std::vector<Bitmap>& imgs_out, const int local_camera_id) {
    if (img_in.Width() != image_width_ || img_in.Height() != image_height_) {
        std::cout << "Error! Input image size error ... " << std::endl;
        return false;
    }

    // the piece number is fixed 
    imgs_out.resize(piece_count_);

    for(int piece_id = 0; piece_id < piece_count_; ++piece_id){

        if (img_in.Channels() == 3){
            imgs_out[piece_id].Allocate(perspective_width_ , perspective_height_ , true);
        }
        else{
            imgs_out[piece_id].Allocate(perspective_width_ , perspective_height_, false);
        }

        int local_image_idx = local_camera_id * piece_count_ + piece_id;
        // Apply remap to update the output image
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int y = 0; y < perspective_height_; ++y){
            int pitch = y * perspective_width_;
            for (int x = 0; x < perspective_width_ ; ++x){
                BitmapColor<float> color;
                color.r = color.g = color.b = 0;

                img_in.InterpolateBilinear(splited_map_xs_[local_image_idx][pitch + x],
                                           splited_map_ys_[local_image_idx][pitch + x], &color);

                BitmapColor<uint8_t> color_uint;
                color_uint.r = color.r > 255.0 ? 255 : static_cast<uint8_t>(color.r);
                color_uint.g = color.g > 255.0 ? 255 : static_cast<uint8_t>(color.g);
                color_uint.b = color.b > 255.0 ? 255 : static_cast<uint8_t>(color.b);
                imgs_out[piece_id].SetPixel(x, y, color_uint);
            }
        }
    }
    return true;
}

void PiecewiseImage::ConvertSplitedPerspectiveCoordToOriginal(const double u_in, const double v_in, 
                                        const int local_camera_id, const int piece_id, double& u_out, double& v_out) {


    Eigen::Vector2d point_in;
    point_in(0) = static_cast<double>(u_in - perspective_width_ / 2) / focal_length_;
    point_in(1) = static_cast<double>(v_in - perspective_height_ / 2) / focal_length_;
    
    Eigen::Vector3d point = point_in.homogeneous();
    Eigen::Vector3d bearing = transforms_[piece_id] * point;

    Eigen::Vector2d point_out;
    if(camera_.NumLocalCameras()>1){
        point_out = camera_.BearingToLocalImage(local_camera_id, bearing);
    }
    else{
        point_out = camera_.BearingToImage(bearing);
    }
    u_out = point_out(0);
    v_out = point_out(1);
}


int PiecewiseImage::GetImageHeight(){
    return image_height_;
}

int PiecewiseImage::GetImageWidth(){
    return image_width_;
}

}  // namespace sensemap
