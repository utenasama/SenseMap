//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"
#include "mesh_texture.h"


namespace sensemap {
namespace texture {
namespace {

std::tuple<std::vector<std::shared_ptr<cv::Mat>>,
        std::vector<std::shared_ptr<cv::Mat>>,
        std::vector<std::shared_ptr<cv::Mat>>,
        std::vector<std::shared_ptr<cv::Mat>>,
        std::vector<std::shared_ptr<cv::Mat>>> CreateGradientImages(
        const std::vector<std::shared_ptr<RGBDImage>> &images_rgbd) {
    std::vector<std::shared_ptr<cv::Mat>> images_gray;
    std::vector<std::shared_ptr<cv::Mat>> images_dx;
    std::vector<std::shared_ptr<cv::Mat>> images_dy;
    std::vector<std::shared_ptr<cv::Mat>> images_color;
    std::vector<std::shared_ptr<cv::Mat>> images_depth;
    for (const auto &rgbd : images_rgbd) {
        cv::Mat gray_image, gray_image_filtered, dx_image, dy_image;

        cv::cvtColor(rgbd->color_, gray_image, cv::COLOR_BGR2GRAY);
        gray_image.convertTo(gray_image, CV_32FC1, 1.0f / 255);

        cv::GaussianBlur(gray_image, gray_image_filtered, cv::Size(3, 3), 0);
        images_gray.push_back(std::make_shared<cv::Mat>(gray_image_filtered));

        cv::Sobel(gray_image_filtered, dx_image, CV_32FC1, 1, 0);
        images_dx.push_back(std::make_shared<cv::Mat>(dx_image));

        cv::Sobel(gray_image_filtered, dy_image, CV_32FC1, 0, 1);
        images_dy.push_back(std::make_shared<cv::Mat>(dy_image));

        auto color = std::make_shared<cv::Mat>(rgbd->color_);
        images_color.push_back(color);

        auto depth = std::make_shared<cv::Mat>(rgbd->depth_);
//        auto depth = std::make_shared<cv::Mat>(rgbd->depth_.rows,
//                                               rgbd->depth_.cols, CV_32FC1);


//        for (int y = 0; y < depth->rows; y++)
//        {
//            for (int x = 0; x < depth->cols; x++)
//            {
//                auto &p = depth->at<float>(y, x);
//                p = rgbd->depth_.at<unsigned short>(y, x) / 1000.0f;
//                //std::cout<<rgbd->depth_.at<unsigned short>(y,x)<<" ";
//                if (p >= 3.0f)
//                    p = 0.0f;
//            }
//            //std::cout<<std::endl;
//        }

        images_depth.push_back(depth);
    }
    return std::make_tuple(images_gray, images_dx, images_dy,
                           images_color, images_depth);
}

void SetGeometryColorAverage(
        TriangleMesh& mesh,
        const std::vector<std::shared_ptr<cv::Mat>>& images_color,
        const CameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        int image_boundary_margin/*= 10*/)
{
    auto n_vertex = mesh.vertices_.size();
    mesh.vertex_colors_.clear();
    mesh.vertex_colors_.resize(n_vertex);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n_vertex; i++) {
        mesh.vertex_colors_[i] = Eigen::Vector3d::Zero();
        double sum = 0.0;
        for (auto iter = 0; iter < visiblity_vertex_to_image[i].size();
             iter++) {
            int j = visiblity_vertex_to_image[i][iter];
            unsigned char r_temp, g_temp, b_temp;
            bool valid = false;
            std::tie(valid, r_temp) = MeshUtilities::QueryImageIntensity
                    <unsigned char>(*images_color[j], mesh.vertices_[i], camera,
                                    j, 0, image_boundary_margin);
            std::tie(valid, g_temp) = MeshUtilities::QueryImageIntensity
                    <unsigned char>(*images_color[j], mesh.vertices_[i], camera,
                                    j, 1, image_boundary_margin);
            std::tie(valid, b_temp) = MeshUtilities::QueryImageIntensity
                    <unsigned char>(
                    *images_color[j], mesh.vertices_[i], camera,
                    j, 2, image_boundary_margin);
            float r = (float)r_temp / 255.0f;
            float g = (float)g_temp / 255.0f;
            float b = (float)b_temp / 255.0f;
            if (valid) {
                mesh.vertex_colors_[i] += Eigen::Vector3d(r, g, b);
                sum += 1.0;
            }
        }
        if (sum > 0.0) {
            mesh.vertex_colors_[i] /= sum;
        }
    }
}

}

TextureMapping::TextureMapping(const TextureMapping::Options &options)
: options_(options){

}

void TextureMapping::Run() {
    for (int rec_idx = 0; ; ++rec_idx) {
        const std::string reconstruction_path = 
                JoinPaths(options_.workspace_path, std::to_string(rec_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }

        //Read K & frames_number
        std::map<uint32_t, Eigen::Matrix3d> intrinsic_map;
        auto name = JoinPaths(reconstruction_path, DENSE_DIR, SPARSE_DIR, 
                                "cameras.bin");
        ReadIntrinsicMatrixFromCOLMAPBinary(name, intrinsic_map);
        std::cout<<intrinsic_map.size()<<std::endl;

        //Read camera pose
        name = JoinPaths(reconstruction_path, DENSE_DIR, SPARSE_DIR, 
                                "images.bin");
        std::vector<Eigen::Matrix4d> poses;
        std::vector<std::string> names;
        std::vector<uint32_t > camera_ids;
        ReadExtrinsicMatrixFromCOLMAPBinary(name, poses, names, camera_ids);

        int n_frames = names.size();
        int n_keyframes = n_frames;
        int step = n_frames / n_keyframes;
        std::cout<<n_frames<<" "<<step<<std::endl;

        for (int i = 0, j =0; i < n_keyframes; ++i, j = j + step)
        {
                std::cout << StringPrintf("\rLoad Registered Images...(%d%)", 
                                          j * 100 / (n_keyframes - 1));
                name = JoinPaths(reconstruction_path, options_.image_folder, 
                                 names[j]);
                if (!ExistsFile(name)) {
                        continue;
                }
                //Read rgb image
                // std::cout << name << std::endl;
                cv::Mat rgb_image = cv::imread(name);

                name = JoinPaths(reconstruction_path, options_.depth_folder, 
                                 StringPrintf("%s.%s.%s", names[j].c_str(), 
                                 GEOMETRIC_TYPE, DEPTH_EXT).c_str());
                // std::cout << name << std::endl;
                if (!ExistsFile(name)) {
                        continue;
                }
                //Read depth image
                cv::Mat depth_image;
                ReadDepthFromCOLMAP(name, depth_image);
        //        cv::imshow("test d", depth_image / 10);
        //        cv::imshow("test rgb", rgb_image);
        //        cv::waitKey(0);
                cv::resize(depth_image, depth_image, rgb_image.size());
                rgbd_images_.push_back(std::make_shared<RGBDImage>(
                        rgb_image,depth_image));

                camera_.parameters_.push_back(
                        std::make_shared<CameraParameters>(
                                intrinsic_map[camera_ids[j]], poses[j]));
        }
        std::cout << std::endl;

        //Read obj model
        ReadTriangleMeshObj(options_.model_name, mesh_, true);

        Mapping();

        name = JoinPaths(reconstruction_path, DENSE_DIR, TEX_MODEL_NAME);
        WriteTriangleMeshObj(name, mesh_);
    }
}

void TextureMapping::Mapping() {
    std::vector<std::shared_ptr<cv::Mat>> images_gray,
            images_dx, images_dy, images_color, images_depth;
    std::tie(images_gray, images_dx, images_dy, images_color, images_depth) =
            CreateGradientImages(rgbd_images_);



    std::vector<std::vector<int>> visiblity_vertex_to_image;
    std::vector<std::vector<int>> visiblity_image_to_vertex;
    std::tie(visiblity_vertex_to_image, visiblity_image_to_vertex) =
            MeshUtilities::CreateVertexAndImageVisibility(
                    mesh_, images_depth, camera_,
                    options_.maximum_allowable_depth_,
                    options_.depth_threshold_for_visiblity_check_);

    SetGeometryColorAverage(
            mesh_, images_color, camera_,
            visiblity_vertex_to_image, options_.image_boundary_margin_);
}

} // namespace sensemap
} // namespace texture
