
//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "util/types.h"
#include "util/bitmap.h"
#include "util/mat.h"
#include "util/misc.h"
#include "util/rgbd_helper.h"
#include "util/imageconvert.h"
#include "util/threading.h"

#include "base/common.h"
#include "base/image.h"
#include "base/undistortion.h"
#include "base/reconstruction.h"
#include "base/reconstruction_manager.h"

#include "mvs/depth_map.h"
#include "mvs/normal_map.h"

#include "../Configurator_yaml.h"

using namespace sensemap;

int main(int argc, char *argv[]) {

	std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string image_path = param.GetArgument("image_path", "");
    bool use_tof = param.GetArgument("image_type", "") == "rgbd";
	bool geom_consistency = param.GetArgument("geom_consistency", 1);
    std::string input_type = geom_consistency ? GEOMETRIC_TYPE : PHOTOMETRIC_TYPE;

    for (int reconstruction_idx = 0; ;
        ++reconstruction_idx) {

        std::string component_path = 
            JoinPaths(workspace_path, std::to_string(reconstruction_idx));

        if (!boost::filesystem::exists(component_path)) break;

        std::string dense_path = JoinPaths(component_path, DENSE_DIR);
        std::string stereo_path = JoinPaths(dense_path, STEREO_DIR);
        std::string sparse_path = JoinPaths(dense_path, SPARSE_DIR);
        std::string mask_path = JoinPaths(dense_path, MASKS_DIR);
        std::string tof_path = JoinPaths(dense_path, DEPTHS_DIR);
        std::string depths_path = JoinPaths(stereo_path, DEPTHS_DIR);
        std::string normals_path = JoinPaths(stereo_path, NORMALS_DIR);

        if (!ExistsDir(dense_path)) break;
        if (!ExistsDir(stereo_path)) break;
        if (!ExistsDir(sparse_path)) break;
        if (!ExistsDir(depths_path)) break;
        if (!ExistsDir(normals_path)) break;
        if (use_tof && !ExistsDir(tof_path)) break;

        std::shared_ptr<Reconstruction> reconstruction(new Reconstruction());
        reconstruction->ReadBinary(sparse_path);

        std::vector<std::string> image_names;
        std::vector<image_t> images = reconstruction->RegisterImageIds();
        for (auto image_id : images) {
            const auto& image = reconstruction->Image(image_id);
            std::string image_name = JoinPaths(image_path, image.Name());
            image_names.push_back(image_name);
        }

        std::vector<std::vector<Eigen::Vector3f>> image_plane_pts(images.size());
        std::vector<std::vector<Eigen::Vector3f>> image_plane_nms(images.size());

        const int num_eff_threads = sensemap::GetEffectiveNumThreads(-1);
        sensemap::ThreadPool thread_pool(num_eff_threads);
        for (size_t index = 0; index < images.size(); ++index)
        thread_pool.AddTask([&](size_t i) {
            auto & image = reconstruction->Image(images[i]);
            Camera camera = reconstruction->Camera(image.CameraId());

            cv::Mat mask = cv::imread(JoinPaths(mask_path, image.Name()) + "." + MASK_EXT, cv::IMREAD_GRAYSCALE);
            if (mask.empty()) return;
            CHECK_EQ(camera.Width(), mask.cols);
            CHECK_EQ(camera.Height(), mask.rows);

            const int element_size = std::max(1, (int)((mask.cols + mask.rows) * 0.0025));
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * element_size + 1, 2 * element_size + 1));
            cv::Mat plane_mask = (mask == 128);
            cv::erode(plane_mask, plane_mask, element);

            std::string depth_file_name = StringPrintf("%s.%s.%s", 
                image.Name().c_str(), input_type.c_str(), DEPTH_EXT);
            std::string normal_file_name = StringPrintf("%s.%s.%s", 
                image.Name().c_str(), input_type.c_str(), NORMAL_EXT);
            std::string depth_map_path =
                JoinPaths(depths_path, depth_file_name);
            std::string normal_map_path =
                JoinPaths(normals_path, normal_file_name);

            if (!ExistsFile(depth_map_path) || !ExistsFile(normal_map_path)) {
                depth_file_name = StringPrintf("%s.jpg.%s.%s", 
                    image.Name().c_str(), input_type.c_str(), DEPTH_EXT);
                normal_file_name = StringPrintf("%s.jpg.%s.%s", 
                    image.Name().c_str(), input_type.c_str(), NORMAL_EXT);
                depth_map_path =
                    JoinPaths(depths_path, depth_file_name);
                normal_map_path =
                    JoinPaths(normals_path, normal_file_name);
            }
            if (!ExistsFile(depth_map_path) || !ExistsFile(normal_map_path)) {
                std::cout << "Depth/normal map for " << image.Name() << " not found" << std::endl;
                return;
            }

            mvs::DepthMap depth_map;
            mvs::NormalMap normal_map;
            depth_map.Read(depth_map_path);
            normal_map.Read(normal_map_path);

            if (use_tof) {
                std::string tof_file_name = StringPrintf("%s.jpg.%s", 
                    image.Name().c_str(), DEPTH_EXT);
                std::string tof_map_path =
                    JoinPaths(tof_path, tof_file_name);

                mvs::DepthMap raw_depth_map;
                raw_depth_map.Read(tof_map_path);

                ResizeDepthMap(depth_map, raw_depth_map);
            }

            if (!depth_map.IsValid() || !normal_map.IsValid()) {
                std::cout << "Depth/normal map for " << image.Name() << " invalid" << std::endl;
                return;
            }

            const Eigen::Matrix3f R = image.RotationMatrix().cast<float>();
            const Eigen::Vector3f t = image.Tvec().cast<float>();
            for (int y = 0; y < depth_map.GetHeight(); y++) {
                for (int x = 0; x < depth_map.GetWidth(); x++) {
                    if (plane_mask.at<uchar>(y, x)) {
                        const float z = depth_map.Get(y, x);
                        if (z > 0.0f) {
                            Eigen::Vector3f pt = camera.ImageToWorld(Eigen::Vector2d(x, y)).homogeneous().cast<float>();
                            pt *= z;
                            pt = R.transpose() * (pt - t);

                            image_plane_pts[i].emplace_back(pt.cast<float>());
                        }

                        Eigen::Vector3f n = Eigen::Vector3f(normal_map.Get(y, x, 0), 
                                                            normal_map.Get(y, x, 1), 
                                                            normal_map.Get(y, x, 2));
                        if (n.squaredNorm() > std::numeric_limits<float>::epsilon()) {
                            n = R.transpose() * n;
                            image_plane_nms[i].emplace_back(n.cast<float>());
                        }
                    }
                }
            }

            std::cout << "\rRead " << image.Name() << std::flush;
        }, index); 
        thread_pool.Wait();
        std::cout << std::endl;

        std::vector<Eigen::Vector3f> merged_plane_pts;
        for (auto & plane_pts : image_plane_pts) {
            merged_plane_pts.insert(merged_plane_pts.end(), plane_pts.begin(), plane_pts.end());
            plane_pts.clear();
            plane_pts.shrink_to_fit();
        }
        image_plane_pts.clear();
        image_plane_pts.shrink_to_fit();

        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        for (const auto & plane_nms : image_plane_nms) {
            for (const auto & nm : plane_nms) {
                normal += nm.cast<double>();
            }
        }
        image_plane_nms.clear();
        image_plane_nms.shrink_to_fit();

        Eigen::Vector4f plane = Eigen::Vector4f::Zero();
        if (merged_plane_pts.size() > 10 && !normal.isZero()) {

            constexpr int num_iters = 3;
            for (int iter = 0; iter < num_iters; iter++) {
                Eigen::Vector3d center = Eigen::Vector3d::Zero();
                for (const auto & pt : merged_plane_pts) {
                    center += pt.cast<double>();
                }
                center /= merged_plane_pts.size();

                Eigen::Vector3f centerf = center.cast<float>();
                Eigen::MatrixXf A(merged_plane_pts.size(), 3);
                for (int i = 0; i < merged_plane_pts.size(); i++) {
                    const auto & pt = merged_plane_pts[i];
                    A(i, 0) = pt[0] - centerf[0];
                    A(i, 1) = pt[1] - centerf[1];
                    A(i, 2) = pt[2] - centerf[2];
                }

                Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinV);
                float a = svd.matrixV()(0, 2);
                float b = svd.matrixV()(1, 2);
                float c = svd.matrixV()(2, 2);
                float d = -(a * center[0] + b * center[1] + c * center[2]);

                plane = Eigen::Vector4f(a, b, c, d);
                std::cout << "Iteration " << iter << ": " << std::endl;
                std::cout << plane.transpose() << std::endl;

                // FILE * fp = fopen((std::string("plane_") + std::to_string(iter) + ".obj").c_str(), "w");
                // for (const auto & pt : merged_plane_pts) {
                //     fprintf(fp, "v %f %f %f\n", pt[0], pt[1], pt[2]);
                // }
                // fclose(fp);

                if (iter != num_iters - 1) {
                    std::sort(merged_plane_pts.begin(), merged_plane_pts.end(), [&]
                    (const Eigen::Vector3f & a, const Eigen::Vector3f & b) {
                        return std::abs(plane[0] * a[0] + plane[1] * a[1] + plane[2] * a[2] + plane[3]) <
                               std::abs(plane[0] * b[0] + plane[1] * b[1] + plane[2] * b[2] + plane[3]);
                    });

                    // std::cout << merged_plane_pts.front().homogeneous().dot(plane) << std::endl;
                    // std::cout << merged_plane_pts.back().homogeneous().dot(plane) << std::endl;
                    merged_plane_pts.resize(merged_plane_pts.size() * 0.9);
                }
            }

            normal /= normal.head<3>().norm();
            std::cout << "Average normal :" << std::endl;
            std::cout << normal.transpose() << std::endl;
            if (plane.head<3>().dot(normal.cast<float>()) < 0) {
                plane *= -1;
            }
        }

        std::ofstream ofs(JoinPaths(dense_path, "plane.txt"));
        std::cout << "Global plane: " << std::endl;
        std::cout << plane.transpose() << std::endl;
        ofs << plane[0] << " " << plane[1] << " " << plane[2] << " " << plane[3] << std::endl;
    }

    return 0;
}