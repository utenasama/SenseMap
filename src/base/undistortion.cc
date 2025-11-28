//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"
#include "util/threading.h"
#include "util/imageconvert.h"
#include "util/depth_mesh.h"
#include "util/rgbd_helper.h"
#include "util/cross_warp_helper.h"
#include "base/camera_models.h"
#include "base/undistortion.h"
#include "base/projection.h"
#include "base/common.h"
#include "mvs/depth_map.h"
#include <opencv2/core/eigen.hpp>

namespace sensemap {

void Undistorter::WarpImageBetweenCameras(const Camera& source_camera,
                             const Camera& target_camera,
                             const Bitmap& source_image,
                             Bitmap* target_image) {
    CHECK_EQ(source_camera.Width(), source_image.Width());
    CHECK_EQ(source_camera.Height(), source_image.Height());

    target_image->Allocate(target_camera.Width(),
                           target_camera.Height(),
                           source_image.IsRGB());
    
    #pragma omp parallel for
    for (int y = 0; y < target_image->Height(); ++y) {
        Eigen::Vector2d image_point;
        image_point.y() = y + 0.5;
        for (int x = 0; x < target_image->Width(); ++x) {
            image_point.x() = x + 0.5;

            const Eigen::Vector2d world_point = 
                target_camera.ImageToWorld(image_point);
            const Eigen::Vector2d source_point =
                source_camera.WorldToImage(world_point);
            
            BitmapColor<float> color;
            if (source_image.InterpolateBilinear(source_point.x() - 0.5,
                                                 source_point.y() - 0.5, &color)) {
                target_image->SetPixel(x, y, color.Cast<uint8_t>());
            } else {
                target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
            }
        }
    }
}

void Undistorter::WarpMaskBetweenCameras(const Camera& source_camera,
                             const Camera& target_camera,
                             const Bitmap& source_image,
                             Bitmap* target_image) {
    CHECK_EQ(source_camera.Width(), source_image.Width());
    CHECK_EQ(source_camera.Height(), source_image.Height());

    target_image->Allocate(target_camera.Width(),
                           target_camera.Height(),
                           source_image.IsRGB());
    
    #pragma omp parallel for
    for (int y = 0; y < target_image->Height(); ++y) {
        Eigen::Vector2d image_point;
        image_point.y() = y + 0.5;
        for (int x = 0; x < target_image->Width(); ++x) {
            image_point.x() = x + 0.5;

            const Eigen::Vector2d world_point = 
                target_camera.ImageToWorld(image_point);
            const Eigen::Vector2d source_point =
                source_camera.WorldToImage(world_point);
            
            BitmapColor<uint8_t> color;
            if (source_image.InterpolateNearestNeighbor(source_point.x() - 0.5,
                                                        source_point.y() - 0.5, &color)) {
                target_image->SetPixel(x, y, color);
            } else {
                target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
            }
        }
    }
}

void Undistorter::UndistortCamera(const UndistortOptions& options,
                     const Camera& camera,
                     Camera* undistorted_camera) {
    CHECK_GE(options.blank_pixels, 0);
    CHECK_LE(options.blank_pixels, 1);
    CHECK_GT(options.min_scale, 0.0);
    CHECK_LE(options.min_scale, options.max_scale);
    CHECK_NE(options.max_image_size, 0);

    undistorted_camera->SetModelId(PinholeCameraModel::model_id);
    undistorted_camera->SetWidth(camera.Width());
    undistorted_camera->SetHeight(camera.Height());

    const size_t orig_undistorted_camera_width = undistorted_camera->Width();
    const size_t orig_undistorted_camera_height = undistorted_camera->Height();

    // Copy focal length parameters.
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    CHECK_LE(focal_length_idxs.size(), 2)
        << "Not more than two focal length parameters supported.";
    if (focal_length_idxs.size() == 1) {
        undistorted_camera->SetFocalLengthX(camera.FocalLength());
        undistorted_camera->SetFocalLengthY(camera.FocalLength());
    } else if (focal_length_idxs.size() == 2) {
        undistorted_camera->SetFocalLengthX(camera.FocalLengthX());
        undistorted_camera->SetFocalLengthY(camera.FocalLengthY());
    }

    // Copy principal point parameters.
    undistorted_camera->SetPrincipalPointX(camera.PrincipalPointX());
    undistorted_camera->SetPrincipalPointY(camera.PrincipalPointY());

    size_t roi_min_x = 0;
    size_t roi_min_y = 0;
    size_t roi_max_x = camera.Width();
    size_t roi_max_y = camera.Height();

    if (camera.ModelId() == UnifiedCameraModel::model_id ||
        camera.ModelId() == OpenCVFisheyeCameraModel::model_id) {
        double scale = 0.4;
        double fx = undistorted_camera->FocalLengthX() * scale;
        double fy = undistorted_camera->FocalLengthY() * scale;
        if (options.fov_w > 0.0 && options.fov_w < 180.0 &&
            options.fov_h > 0.0 && options.fov_h < 180.0) {
            double radx = options.fov_w / 360.0 * M_PI;
            fx = undistorted_camera->Width() * 0.5 / std::tan(radx);
            double rady = options.fov_h / 360.0 * M_PI;
            fy = undistorted_camera->Height() * 0.5 / std::tan(rady);       
        } else if (options.fov_w > 0.0 && options.fov_w < 180.0) {
            double rad = options.fov_w / 360.0 * M_PI;
            double efx = undistorted_camera->Width() * 0.5 / std::tan(rad);
            fy = efx * fy / fx;
            fx = efx;
        } else if (options.fov_h > 0.0 && options.fov_h < 180.0) {
            double rad = options.fov_h / 360.0 * M_PI;
            double efy = undistorted_camera->Height() * 0.5 / std::tan(rad);
            fx = efy * fx / fy;
            fy = efy;
        }
        undistorted_camera->SetFocalLengthX(fx);
        undistorted_camera->SetFocalLengthY(fy);
        undistorted_camera->SetFromRIG(true);
    } else if (camera.ModelId() != SimplePinholeCameraModel::model_id &&
               camera.ModelId() != PinholeCameraModel::model_id) {
        // Determine min/max coordinates along top / bottom image border.

        double left_min_x = std::numeric_limits<double>::max();
        double left_max_x = std::numeric_limits<double>::lowest();
        double right_min_x = std::numeric_limits<double>::max();
        double right_max_x = std::numeric_limits<double>::lowest();

#pragma omp parallel for
        for (size_t y = roi_min_y; y < roi_max_y; ++y) {
            // Left border.
            const Eigen::Vector2d world_point1 =
                camera.ImageToWorld(Eigen::Vector2d(0.5, y + 0.5));
            const Eigen::Vector2d undistorted_point1 =
                undistorted_camera->WorldToImage(world_point1);

            // Right border.
            const Eigen::Vector2d world_point2 =
                camera.ImageToWorld(Eigen::Vector2d(camera.Width() - 0.5, y + 0.5));
            const Eigen::Vector2d undistorted_point2 =
                undistorted_camera->WorldToImage(world_point2);
#pragma omp critical
            {
                left_min_x = std::min(left_min_x, undistorted_point1(0));
                left_max_x = std::max(left_max_x, undistorted_point1(0));
                right_min_x = std::min(right_min_x, undistorted_point2(0));
                right_max_x = std::max(right_max_x, undistorted_point2(0));
            }
        }

        // Determine min, max coordinates along left / right image border.
        double top_min_y = std::numeric_limits<double>::max();
        double top_max_y = std::numeric_limits<double>::lowest();
        double bottom_min_y = std::numeric_limits<double>::max();
        double bottom_max_y = std::numeric_limits<double>::lowest();

#pragma omp parallel for
        for (size_t x = roi_min_x; x < roi_max_x; ++x) {
            // Top border.
            const Eigen::Vector2d world_point1 =
                camera.ImageToWorld(Eigen::Vector2d(x + 0.5, 0.5));
            const Eigen::Vector2d undistorted_point1 =
                undistorted_camera->WorldToImage(world_point1);
            // Bottom border.
            const Eigen::Vector2d world_point2 =
                camera.ImageToWorld(Eigen::Vector2d(x + 0.5, camera.Height() - 0.5));
            const Eigen::Vector2d undistorted_point2 =
                undistorted_camera->WorldToImage(world_point2);
#pragma omp critical
            {
                top_min_y = std::min(top_min_y, undistorted_point1(1));
                top_max_y = std::max(top_max_y, undistorted_point1(1));
                bottom_min_y = std::min(bottom_min_y, undistorted_point2(1));
                bottom_max_y = std::max(bottom_max_y, undistorted_point2(1));
            }
        }

        const double cx = undistorted_camera->PrincipalPointX();
        const double cy = undistorted_camera->PrincipalPointY();

        // Scale such that undistorted image contains all pixels of distorted image.
        const double min_scale_x = std::min(cx / (cx - left_min_x),
                    (undistorted_camera->Width() - 0.5 - cx) / (right_max_x - cx));
        const double min_scale_y = std::min(cy / (cy - top_min_y),
            (undistorted_camera->Height() - 0.5 - cy) / (bottom_max_y - cy));

        // Scale such that there are no blank pixels in undistorted image.
        const double max_scale_x = std::max(cx / (cx - left_max_x),
                    (undistorted_camera->Width() - 0.5 - cx) / (right_min_x - cx));
        const double max_scale_y = std::max(cy / (cy - top_max_y),
            (undistorted_camera->Height() - 0.5 - cy) / (bottom_min_y - cy));

        // Interpolate scale according to blank_pixels.
        double scale_x = 1.0 / (min_scale_x * options.blank_pixels +
                                max_scale_x * (1.0 - options.blank_pixels));
        double scale_y = 1.0 / (min_scale_y * options.blank_pixels +
                                max_scale_y * (1.0 - options.blank_pixels));

        // Clip the scaling factors.
        scale_x = Clip(scale_x, options.min_scale, options.max_scale);
        scale_y = Clip(scale_y, options.min_scale, options.max_scale);

        // Scale undistorted camera dimensions.
        undistorted_camera->SetWidth(static_cast<size_t>(
            std::max(1.0, scale_x * undistorted_camera->Width())));
        undistorted_camera->SetHeight(static_cast<size_t>(
            std::max(1.0, scale_y * undistorted_camera->Height())));

        // Scale the principal point according to the new dimensions of the camera.
        undistorted_camera->SetPrincipalPointX(
            undistorted_camera->PrincipalPointX() *
            static_cast<double>(undistorted_camera->Width()) /
            static_cast<double>(orig_undistorted_camera_width));
        undistorted_camera->SetPrincipalPointY(
            undistorted_camera->PrincipalPointY() *
            static_cast<double>(undistorted_camera->Height()) /
            static_cast<double>(orig_undistorted_camera_height));
    }
    if (options.max_image_size > 0) {
        int max_image_size = std::min(options.max_image_size, 
                                     (int)std::max(undistorted_camera->Width(), undistorted_camera->Height()));
        const double max_image_scale_x = max_image_size /
            static_cast<double>(undistorted_camera->Width());
        const double max_image_scale_y = max_image_size /
            static_cast<double>(undistorted_camera->Height());
        const double max_image_scale =
            std::min(max_image_scale_x, max_image_scale_y);
        if (max_image_scale < 1.0) {
            undistorted_camera->Rescale(max_image_scale);
        }
    } else if (orig_undistorted_camera_width < undistorted_camera->Width() ||
               orig_undistorted_camera_height < undistorted_camera->Height()) {
        const double max_image_scale_x = orig_undistorted_camera_width /
            static_cast<double>(undistorted_camera->Width());
        const double max_image_scale_y = orig_undistorted_camera_height /
            static_cast<double>(undistorted_camera->Height());
        const double max_image_scale =
            std::min(max_image_scale_x, max_image_scale_y);
        if (max_image_scale < 1.0) {
            undistorted_camera->Rescale(max_image_scale);
        }
    }
}

void Undistorter::UndistortImage(const UndistortOptions& options,
                    const Bitmap& distorted_bitmap,
                    const Camera& distorted_camera,
                    Bitmap* undistorted_bitmap,
                    Camera* undistorted_camera) {
    CHECK_EQ(distorted_camera.Width(), distorted_bitmap.Width());
    CHECK_EQ(distorted_camera.Height(), distorted_bitmap.Height());

    UndistortCamera(options, distorted_camera, undistorted_camera);

    WarpImageBetweenCameras(distorted_camera, *undistorted_camera,
                            distorted_bitmap, undistorted_bitmap);
}

void Undistorter::UndistortReconstruction(const UndistortOptions& options,
                             Reconstruction* reconstruction) {
    const auto distorted_cameras = reconstruction->Cameras();
    for (auto& camera : distorted_cameras) {
        Camera& undistorted_camera = reconstruction->Camera(camera.first);
        UndistortCamera(options, camera.second, &undistorted_camera);
    }

    for (const auto& distorted_image : reconstruction->Images()) {
        auto& image = reconstruction->Image(distorted_image.first);
        const auto& distorted_camera = distorted_cameras.at(image.CameraId());
        const auto& undistorted_camera = reconstruction->Camera(image.CameraId());
        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
            ++point2D_idx) {
            auto& point2D = image.Point2D(point2D_idx);
            point2D.SetXY(undistorted_camera.WorldToImage(
                distorted_camera.ImageToWorld(point2D.XY())));
        }
    }
}

std::map<image_t, std::shared_ptr<PriorDepthInfo>> Undistorter::InitPriorDepths(
    const std::string & image_path,
    const Reconstruction & reconstruction,
    const UndistortOptions & options
) {
    std::cout << "Init prior depths" << std::endl;

    std::map<image_t, std::shared_ptr<PriorDepthInfo>> prior_depths;
    auto image_ids = reconstruction.RegisterImageIds();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < image_ids.size(); i++) {
        const auto & image_id = image_ids[i];
        const auto & image = reconstruction.Image(image_id);

        auto prior_depth = std::make_shared<PriorDepthInfo>();
        if (IsFileRGBD(image.Name())) {
            // Bin RGBD
            RGBDData data;
            if (ExtractRGBDData(JoinPaths(image_path, image.Name()),
                RGBDReadOption::NoColor(), data
            )) {
                data.ReadRGBDCameraParams(options.rgbd_camera_params);
                if (data.HasRGBDCalibration()) {

                    if (data.depth_camera.ModelName() == "PINHOLE" || 
                        data.depth_camera.ModelName() == "SIMPLE_PINHOLE"
                    ) {
                        prior_depth->depthmap = data.depth;
                        prior_depth->K = data.depth_camera.CalibrationMatrix().cast<float>();
                    } else {
                        Camera undistorted_depth_camera;
                        Undistorter::UndistortCamera(options, data.depth_camera, &undistorted_depth_camera);

                        MatXf undistorted_depth_map(undistorted_depth_camera.Width(), undistorted_depth_camera.Height(), 1);
                        UniversalWarpDepthMap(undistorted_depth_map, data.depth, undistorted_depth_camera, data.depth_camera, Eigen::Matrix4f::Identity());
                    
                        prior_depth->depthmap = undistorted_depth_map;
                        prior_depth->K = undistorted_depth_camera.CalibrationMatrix().cast<float>();
                    }
                    
                    prior_depth->RT = data.depth_RT.cast<float>();
                }
            } else {
                continue;
            }
        } else {
            continue;
        }

        // Filter depth values
        for (int y = 0; y < prior_depth->depthmap.GetHeight(); y++) {
            for (int x = 0; x < prior_depth->depthmap.GetWidth(); x++) {
                if (options.max_prior_depth > 0 && prior_depth->depthmap.Get(y, x) > options.max_prior_depth || 
                    options.min_prior_depth > 0 && prior_depth->depthmap.Get(y, x) < options.min_prior_depth
                ) {
                    prior_depth->depthmap.Set(y, x, 0.0f);
                }
            }
        }

        #pragma omp critical
        {
            prior_depths[image_id] = prior_depth;
        }
    }
    std::cout << "Init " << prior_depths.size() << " prior depths" << std::endl;

    return prior_depths;
}

void Undistorter::WarpPriorDepth(
    image_t image_id,
    const Camera & undistorted_camera,
    const Reconstruction & reconstruction,
    const std::map<image_t, std::shared_ptr<PriorDepthInfo>> & prior_depths,
    const std::string & undistort_depths_path,
    const UndistortOptions & options
) {
    const Image& image = reconstruction.Image(image_id);
    auto find = prior_depths.find(image_id);

    if (find != prior_depths.end()) {
        auto prior_depth = find->second;
        Eigen::Matrix3f undistorted_K = undistorted_camera.CalibrationMatrix().cast<float>();
        Eigen::Matrix3f depth_K = prior_depth->K;
        MatXf warped_depthmap(undistorted_camera.Width(), undistorted_camera.Height(), 1);
        if (options.warp_depth_sparsely) {
            WarpDepthMap2RGB(warped_depthmap, prior_depth->depthmap, undistorted_K, depth_K, prior_depth->RT);
        } else {
            MeshWarpDepthMap(warped_depthmap, prior_depth->depthmap, undistorted_K, depth_K, prior_depth->RT);
        }

        const std::string output_depths_path =
            JoinPaths(undistort_depths_path, image.Name() + ".jpg." + DEPTH_EXT);
        const std::string output_depths_parent_path = GetParentDir(output_depths_path);
        if (!boost::filesystem::exists(output_depths_parent_path)) {
            boost::filesystem::create_directories(output_depths_parent_path);
        }
        mvs::DepthMap depth_map_wrapper(warped_depthmap, 0, MAX_VALID_DEPTH_IN_M);
        depth_map_wrapper.Write(output_depths_path);
        if (options.verbose) {
            depth_map_wrapper.ToBitmap().Write(output_depths_path + "_color.jpg");
        }
    }
}

void Undistorter::CrossWarpPriorDepth(
    image_t image_id,
    const Camera & undistorted_camera,
    const Reconstruction & reconstruction,
    const CrossWarpHelper & cross_warp_helper,
    const std::map<image_t, std::shared_ptr<PriorDepthInfo>> & prior_depths,
    const std::string & undistort_depths_path,
    const UndistortOptions & options
) {
    const Image& image = reconstruction.Image(image_id);
    bool enable_cross_warp = false;
    for (auto & subpath : options.cross_warp_subpath) {
        if (IsInsideSubpath(image.Name(), subpath)) {
            enable_cross_warp = true;
            break;
        }
    }

    if (enable_cross_warp) {
        auto src_image_ids = cross_warp_helper.GetNeighboringImages(image.ImageId());
        std::vector<image_t> src_image_candidate_ids; 
        for (auto id : src_image_ids)
        {
            if (prior_depths.count(id) == 0)
                continue;
            src_image_candidate_ids.push_back(id);
        }
        if (src_image_candidate_ids.size())
        {
            MatXf warped_depthmap(undistorted_camera.Width(), undistorted_camera.Height(), 1);

            std::vector<Eigen::Vector3f> vertices;
            std::vector<Eigen::Vector3i> faces;
            Eigen::Matrix3d R = QuaternionToRotationMatrix(image.Qvec());
            Eigen::Vector3d t = image.Tvec();
            Eigen::Matrix3d K = undistorted_camera.CalibrationMatrix();
            for(int i = 0; i < src_image_candidate_ids.size(); i++)
            {
                image_t src_image_id = src_image_candidate_ids[i];
                const auto &src_image = reconstruction.Image(src_image_id);
                auto prior_depth = prior_depths.find(src_image_id)->second;

                // pose = RT-1 * A
                Eigen::Matrix4d color_to_depth_RT = prior_depth->RT.cast<double>().inverse();
                Eigen::Matrix3d color_R = QuaternionToRotationMatrix(src_image.Qvec());
                Eigen::Vector3d color_T = src_image.Tvec();
                Eigen::Matrix3d depth_R = color_to_depth_RT.block<3, 3>(0, 0) * color_R;    // Rd*Ri
                Eigen::Vector3d depth_T = color_to_depth_RT.block<3, 3>(0, 0) * color_T +   // Rd*ti + td
                                          color_to_depth_RT.block<3, 1>(0, 3) / 1000.0;

                DepthMesh mesh;
                GenerateMesh(mesh, prior_depth->depthmap.GetPtr(), prior_depth->depthmap.GetWidth(), prior_depth->depthmap.GetWidth(), prior_depth->depthmap.GetHeight(),
                             prior_depth->K, depth_R.cast<float>(), depth_T.cast<float>(), 2, 0.05, VertexFormat::POSITION, std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::max());
    
                const int vertex_offset = vertices.size();
                for (auto v : mesh.vertices_) {
                    const Eigen::Vector3d proj_point3D = R * v.cast<double>() + t;
                    if (proj_point3D.z() > 1e-5) {
                        Eigen::Vector2d point2D = (K * proj_point3D).hnormalized();
                        vertices.emplace_back(point2D.x(), point2D.y(), proj_point3D.z());
                    } else {
                        vertices.emplace_back(Eigen::Vector3f::Zero());
                    }
                }
                for (auto f : mesh.faces_) {
                    faces.emplace_back(f.x() + vertex_offset, f.y() + vertex_offset, f.z() + vertex_offset);
                }
            }

            cv::Mat warped_depth_map(warped_depthmap.GetHeight(), warped_depthmap.GetWidth(), CV_32FC1, warped_depthmap.GetPtr());
            WarpFrameBuffer(warped_depth_map, vertices, faces, 0, MAX_VALID_DEPTH_IN_M, -1);

            const std::string output_depths_path =
                JoinPaths(undistort_depths_path, image.Name() + ".jpg." + DEPTH_EXT);
            const std::string output_depths_parent_path = GetParentDir(output_depths_path);
            if (!boost::filesystem::exists(output_depths_parent_path)) {
                boost::filesystem::create_directories(output_depths_parent_path);
            }
            mvs::DepthMap depth_map_wrapper(warped_depthmap, 0, MAX_VALID_DEPTH_IN_M);
            depth_map_wrapper.Write(output_depths_path);
            if (options.verbose) {
                depth_map_wrapper.ToBitmap().Write(output_depths_path + "_color.jpg");
            }
        }
    }
}

double Undistorter::ReverseScaleRecovery(
    std::map<image_t, std::shared_ptr<PriorDepthInfo>> & prior_depths,
    const Reconstruction & reconstruction
) {
    std::vector<image_t> rgbd_ids;
    for (auto & prior_depth : prior_depths) {
        rgbd_ids.push_back(prior_depth.first);
    }

    std::vector<std::pair<double, double>> scale_candidates;
    #pragma omp parallel
    {
        std::vector<std::pair<double, double>> _scale_candidates;

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < rgbd_ids.size(); i++)
        {
            auto & prior_depth = prior_depths[rgbd_ids[i]];
            const image_t image_id = rgbd_ids[i];
            const auto & image = reconstruction.Image(image_id);

            Camera depth_camera;
                   depth_camera.SetModelIdFromName("PINHOLE");
                   depth_camera.SetFocalLengthX(prior_depth->K(0, 0));
                   depth_camera.SetFocalLengthY(prior_depth->K(1, 1));
                   depth_camera.SetPrincipalPointX(prior_depth->K(0, 2));
                   depth_camera.SetPrincipalPointY(prior_depth->K(1, 2));
                   depth_camera.SetWidth (prior_depth->depthmap.GetWidth());
                   depth_camera.SetHeight(prior_depth->depthmap.GetHeight());
            Camera color_camera = reconstruction.Camera(image.CameraId());
            
            const int width = color_camera.Width();
            const int height = color_camera.Height();
            MatXf warped_depthmap(width, height, 1);
            UniversalWarpDepthMap(warped_depthmap, prior_depth->depthmap, color_camera, depth_camera, prior_depth->RT);

            std::vector<std::pair<double, double>> image_scale_candidates;
            for (const auto & point2d : image.Points2D())
            {
                if (!point2d.HasMapPoint()) continue;
                auto mappoint_id = point2d.MapPointId();

                auto pt_3d = reconstruction.MapPoint(mappoint_id).XYZ();
                auto pt_cam = QuaternionRotatePoint(image.Qvec(), pt_3d) + image.Tvec();
                if (pt_cam.z() <= 0) continue;
                
                auto image_coord = color_camera.WorldToImage(pt_cam.hnormalized());
                int nx = image_coord(0) + 0.5f;
                int ny = image_coord(1) + 0.5f;
                if (nx < 0 || nx >= width ||
                    ny < 0 || ny >= height
                ) {
                    continue;    
                }

                double rgbd_depth = warped_depthmap.Get(ny, nx);
                if (rgbd_depth > 0) {
                    image_scale_candidates.emplace_back(
                        pt_cam.z() / rgbd_depth, 
                        1.0
                    );
                }
            }

            const size_t image_scale_count = image_scale_candidates.size();
            if (image_scale_count > 10) {
                std::sort(image_scale_candidates.begin(), image_scale_candidates.end(), ScaleCandidateComparer);
                _scale_candidates.insert(_scale_candidates.end(), 
                    image_scale_candidates.begin() + image_scale_count * 0.2,
                    image_scale_candidates.begin() + image_scale_count * 0.8);
            }
        }

        #pragma omp critical
        {
            scale_candidates.reserve(scale_candidates.size() + _scale_candidates.size());
            scale_candidates.insert(scale_candidates.end(), _scale_candidates.begin(), _scale_candidates.end());
        }
    }

    double scale = GetBestScaleByStatitics(scale_candidates);
    if (scale > 0) {
        float inv_scale = 1.0 / scale;
        for (auto & prior_depth : prior_depths) {
            auto & depthmap = prior_depth.second->depthmap;
            for (int y = 0; y < depthmap.GetHeight(); y++) {
                for (int x = 0; x < depthmap.GetWidth(); x++) {
                    float depth = depthmap.Get(y, x);
                    if (depth > 0) {
                        depth *= inv_scale;
                        depthmap.Set(y, x, depth);
                    }
                }
            }
        }
    }

    std::cout << "ReverseScaleRecovery: " << scale << std::endl;
    return scale;
}

Undistorter::Undistorter(const UndistortOptions& options,
                         const std::string& image_path,
                         const std::string& workspace_path)
    : options_(options),
      image_path_(image_path),
      workspace_path_(workspace_path) {}

void Undistorter::CreateWorkspace() {
    std::cout << "Creating workspace..." << std::endl;

    auto dense_workspace_path = JoinPaths(workspace_path_, DENSE_DIR);
    CreateDirIfNotExists(dense_workspace_path);
    undistort_images_path_ = JoinPaths(dense_workspace_path, options_.as_orig_res ? IMAGES_ORIG_RES_DIR : IMAGES_DIR);
    CreateDirIfNotExists(undistort_images_path_);
    undistort_sparse_path_ = JoinPaths(dense_workspace_path, options_.as_orig_res ? SPARSE_ORIG_RES_DIR : SPARSE_DIR);
    CreateDirIfNotExists(undistort_sparse_path_);
    CreateDirIfNotExists(JoinPaths(dense_workspace_path, STEREO_DIR));
    CreateDirIfNotExists(JoinPaths(dense_workspace_path, STEREO_DIR, DEPTHS_DIR));
    CreateDirIfNotExists(JoinPaths(dense_workspace_path, STEREO_DIR, NORMALS_DIR));
    CreateDirIfNotExists(JoinPaths(dense_workspace_path, STEREO_DIR, CONSISTENCY_DIR));

    if (!options_.as_orig_res) {
        if (options_.as_rgbd) {
            undistort_depths_path_ = JoinPaths(dense_workspace_path, DEPTHS_DIR);
            CreateDirIfNotExists(undistort_depths_path_);
            prior_depths_ = InitPriorDepths(image_path_, reconstruction_, options_);

            if (options_.reverse_scale_recovery) {
                double scale = ReverseScaleRecovery(prior_depths_, reconstruction_);
                std::ofstream(JoinPaths(dense_workspace_path, "scale.txt")) << scale;
            }

            if (!options_.cross_warp_subpath.empty()) {
                std::unordered_set<image_t> prior_depth_images;
                for (auto & item : prior_depths_) {
                    prior_depth_images.insert(item.first);
                }
                cross_warp_helper_ = std::make_shared<PatchMatchCrossWarpHelper>(workspace_path_, prior_depth_images, options_.cross_warp_num_images);
            }
        }

        if (!options_.mask_path.empty()) {
            undistort_masks_path_ = JoinPaths(dense_workspace_path, MASKS_DIR);
            CreateDirIfNotExists(undistort_masks_path_);
        }
    }
}

void Undistorter::Run() {
    PrintHeading1("Image undistortion");

    auto dense_workspace_path = JoinPaths(workspace_path_, DENSE_DIR);
    auto undistort_images_path = JoinPaths(dense_workspace_path, options_.as_orig_res ? IMAGES_ORIG_RES_DIR : IMAGES_DIR);
    auto undistort_sparse_path = JoinPaths(dense_workspace_path, options_.as_orig_res ? SPARSE_ORIG_RES_DIR : SPARSE_DIR);
    if (ExistsDir(dense_workspace_path) &&
        ExistsDir(undistort_images_path) &&
        ExistsDir(undistort_sparse_path)) {
        return;
    }

    reconstruction_.ReadBinary(workspace_path_);

    if (ExistsFile(JoinPaths(workspace_path_, ALIGNMENT_POSE_NAME))) {
        auto align_image_ids = reconstruction_.ReadAlignmentBinary(JoinPaths(workspace_path_, ALIGNMENT_POSE_NAME));
        filter_image_ids_.insert(align_image_ids.begin(), align_image_ids.end());
    }
    if (!options_.filter_subpath.empty()) {
        for (const auto &image : reconstruction_.Images()) {
            for (const auto &subpath : options_.filter_subpath) {
                if (IsInsideSubpath(image.second.Name(), subpath)) {
                    filter_image_ids_.insert(image.second.ImageId());
                    break;
                }
            }
        }
    }

    CreateWorkspace();

    auto image_ids = reconstruction_.GetNewImageIds();
    if (image_ids.size() == 0) {
        image_ids = reconstruction_.RegisterImageIds();
    }

    ThreadPool thread_pool;
    std::vector<std::future<void> > futures;
    futures.reserve(image_ids.size());
    for (size_t i = 0; i < image_ids.size(); ++i) {
        futures.emplace_back(
            thread_pool.AddTask(&Undistorter::Undistort, this, image_ids.at(i))
        );
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        if (IsStopped()) {
            break;
        }
        std::cout << StringPrintf("Undistorting image [%d/%d]", i + 1, 
                                   futures.size())
                  << std::endl;
        futures[i].get();
    }

    std::cout << "Writing reconstruction..." << std::endl;
    Reconstruction undistorted_reconstruction = reconstruction_;
    UndistortReconstruction(options_, &undistorted_reconstruction);

    if (!filter_image_ids_.empty()) {
        for (auto image_id : filter_image_ids_) {
            undistorted_reconstruction.DeleteImage(image_id);
        }
        std::cout << "Filter " << filter_image_ids_.size() << " Images" << std::endl;
    }

    int num_filter = undistorted_reconstruction.FilterMapPointsWithSpatialDistribution(
        undistorted_reconstruction.MapPointIds());
    std::cout << "Filter " << num_filter << " MapPoints with spatial distribution " << std::endl;
    std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
    undistorted_reconstruction.FilterUselessPoint2D(filtered_reconstruction);
    filtered_reconstruction->WriteReconstruction(undistort_sparse_path_);
    // undistorted_reconstruction.WriteReconstruction(undistort_sparse_path_);

    WritePatchMatchConfig();
    WriteFusionConfig();
    
    GetTimer().PrintMinutes();
}

void Undistorter::Undistort(const size_t reg_image_idx) const {
    // const image_t image_id = reconstruction_.RegisterImageIds().at(reg_image_idx);
    const image_t image_id = reg_image_idx;
    const Image& image = reconstruction_.Image(image_id);
    const Camera& camera = reconstruction_.Camera(image.CameraId());
    if (filter_image_ids_.count(image_id)) return;

    std::string output_image_path =
        JoinPaths(undistort_images_path_, image.Name());

    Bitmap distorted_bitmap;
    const std::string input_image_path = JoinPaths(image_path_, image.Name());
    const std::string output_depths_path =
        JoinPaths(undistort_depths_path_, image.Name() + ".jpg." + DEPTH_EXT);
    const std::string output_depths_parent_path = GetParentDir(output_depths_path);
    if (options_.as_rgbd) {
        output_image_path = JoinPaths(undistort_images_path_, image.Name() + ".jpg");

        if (IsFileRGBD(input_image_path)) {
            MatXf depthmap;
            if (!ExtractRGBDData(input_image_path, distorted_bitmap, true)) {
                std::cerr << "ERROR: Cannot read image at path" << input_image_path 
                        << std::endl;
            }
        } else if (!distorted_bitmap.Read(input_image_path)) {
            std::cerr << "ERROR: Cannot read image at path" << input_image_path 
                    << std::endl;
            return;
        }
    } else if (!distorted_bitmap.Read(input_image_path)) {
        std::cerr << "ERROR: Cannot read image at path" << input_image_path 
                  << std::endl;
        return;
    }

    Bitmap undistorted_bitmap;
    Camera undistorted_camera;
    UndistortImage(options_, distorted_bitmap, camera, &undistorted_bitmap, 
                   &undistorted_camera);

    if (options_.postprocess) {
        cv::Mat undistorted_mat;
        FreeImage2Mat(&undistorted_bitmap, undistorted_mat);
        cv::Mat undistorted_lab;
        cv::cvtColor(undistorted_mat, undistorted_lab, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> lab_planes(3);
        cv::split(undistorted_lab, lab_planes);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2);
        cv::Mat dst;
        clahe->apply(lab_planes[0], dst);
        dst.copyTo(lab_planes[0]);
        cv::merge(lab_planes, undistorted_lab);

        cv::Mat dst_rgb;
        cv::cvtColor(undistorted_lab, dst_rgb, cv::COLOR_Lab2BGR);
        Mat2FreeImage(dst_rgb, &undistorted_bitmap);
    }

    std::string parent_path = GetParentDir(output_image_path);
    if (!ExistsPath(parent_path)) {
        boost::filesystem::create_directories(parent_path);
    }

    undistorted_bitmap.Write(output_image_path);

    if (!options_.as_orig_res) {
        if (options_.as_rgbd && prior_depths_.count(reg_image_idx) != 0) {
            WarpPriorDepth(image_id, undistorted_camera, reconstruction_, prior_depths_, undistort_depths_path_, options_);
        }

        if (options_.as_rgbd && prior_depths_.count(reg_image_idx) == 0 && !options_.cross_warp_subpath.empty()) {
            CrossWarpPriorDepth(image_id, undistorted_camera, reconstruction_, *cross_warp_helper_, prior_depths_, undistort_depths_path_, options_);
        }

        if (!options_.mask_path.empty()) {
            const std::string input_masks_path =
                JoinPaths(options_.mask_path, image.Name() + "." + MASK_EXT);
            const std::string output_masks_path =
                JoinPaths(undistort_masks_path_, image.Name() + "." + MASK_EXT);
            
            Bitmap distorted_mask;
            Bitmap undistorted_mask;
            if (distorted_mask.Read(input_masks_path, false)) {
                CHECK_EQ(distorted_mask.Width(), camera.Width()) << "Mask size mismatch!";
                CHECK_EQ(distorted_mask.Height(), camera.Height()) << "Mask size mismatch!";
                WarpMaskBetweenCameras(camera, undistorted_camera, distorted_mask, &undistorted_mask);
                undistorted_mask.Write(output_masks_path);

                distorted_mask.Deallocate();
                undistorted_mask.Deallocate();
            }
        }
    }
}

void Undistorter::WritePatchMatchConfig() const {
    const auto path = JoinPaths(workspace_path_, DENSE_DIR, STEREO_DIR, "patch-match.cfg");
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;
    for (const auto& image : reconstruction_.Images()) {
        file << image.second.Name() << std::endl;
        file << "__auto__, 20" << std::endl;
    }
}

void Undistorter::WriteFusionConfig() const {
    const auto path = JoinPaths(workspace_path_, DENSE_DIR, STEREO_DIR, "fusion.cfg");
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;
    for (const auto& image : reconstruction_.Images()) {
        file << image.second.Name() << std::endl;
    }
}

}