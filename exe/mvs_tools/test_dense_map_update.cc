//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <memory>
#include <future>
#include <utility>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <boost/iterator/zip_iterator.hpp>

#include "base/common.h"
#include "base/image.h"
#include "base/camera.h"
#include "base/camera_models.h"
#include "base/reconstruction.h"
#include "base/reconstruction_manager.h"

#include "util/types.h"
#include "util/timer.h"
#include "util/string.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/mat.h"
#include "util/bb.h"
#include "util/threading.h"
#include "util/obj.h"
#include "util/panorama.h"

#include "mvs/workspace.h"
#include "mvs/model.h"
#include "mvs/depth_map.h"

#include "../Configurator_yaml.h"

typedef CGAL::Simple_cartesian<double>                       Kernel;
typedef CGAL::Polyhedron_3<Kernel>                           Polyhedron;
typedef Polyhedron::HalfedgeDS                               HalfedgeDS;
typedef Kernel::Point_3                                      Point;
typedef Kernel::Vector_3                                     Vector;
typedef Kernel::Segment_3                                    Segment;
typedef Kernel::Ray_3                                        Ray;
typedef Kernel::Triangle_3                                   Triangle;
typedef std::list<Triangle>::iterator                        Iterator;
typedef CGAL::AABB_triangle_primitive<Kernel, Iterator>      Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive>                 Traits;
typedef CGAL::AABB_tree<Traits>                              Tree;
typedef Tree::Primitive_id                                   Primitive_id;

using namespace sensemap;

const int num_perspective_per_image = 6;
const double fov_w = 60.0;
const double fov_h = 90.0;

std::string configuration_file_path;

float max_depth_error;
float max_intensity_diff;

float RGB2Gray(float r, float g, float b) {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

void GeneratePanoramaMask(const std::string& workspace_path,
                          const Reconstruction& reconstruction,
                          const std::vector<mvs::Image>& perspective_images,
                          const std::vector<std::string>& image_names,
                          const std::vector<image_t>& image_ids,
                          const Panorama& panorama) {
    
    const int r = 5;

    std::cout << "Generate Panorama Mask" << std::endl;
    auto mask_path = JoinPaths(workspace_path, "masks");
    CreateDirIfNotExists(mask_path);

    auto perspective_mask_path = JoinPaths(workspace_path, DENSE_DIR, "masks");
    CreateDirIfNotExists(perspective_mask_path);

    std::vector<std::vector<double> > map_xs = panorama.GetPanoramaMapX();
    std::vector<std::vector<double> > map_ys = panorama.GetPanoramaMapY();

    int num_image = image_ids.size();
    std::unordered_map<image_t, std::vector<size_t> > image_ids_map;
    for (size_t i = 0; i < num_image; ++i) {
        image_ids_map[image_ids.at(i)].push_back(i);
    }

    int progress = 0;
    auto WarpMaskSingleImage = [&](image_t image_id, std::vector<size_t>& subimage_ids) {
        const class Image& image = reconstruction.Image(image_id);
        const class Camera& camera = reconstruction.Camera(image.CameraId());
        const int width = camera.Width();
        const int height = camera.Height();
        Bitmap bitmap;
        bitmap.Allocate(width, height, false);
        BitmapColor<uint8_t> bk_color(0);
        BitmapColor<uint8_t> fg_color(255);
        bitmap.Fill(bk_color);

        for (int j = 0; j < subimage_ids.size(); ++j) {
            std::vector<double>& map_x = map_xs.at(j);
            std::vector<double>& map_y = map_ys.at(j);

            Bitmap mask_map;
            mask_map.Read(JoinPaths(perspective_mask_path, image_names.at(subimage_ids[j]) + ".mask.jpg"), false);

            const int perspective_width = mask_map.Width();
            const int perspective_height = mask_map.Height();
            
            for (int y = 0; y < perspective_height; ++y) {
                for (int x = 0; x < perspective_width; ++x) {
                    BitmapColor<uint8_t> val = mask_map.GetPixel(x, y);
                    if (!val.r) {
                        continue;
                    }
                    double fx = map_x.at(y * perspective_width + x);
                    double fy = map_y.at(y * perspective_width + x);

                    int min_x = fx - r;
                    int min_y = fy - r;
                    int max_x = fx + r;
                    int max_y = fy + r;
                    min_x = std::max(0, std::min(width - 1, min_x));
                    min_y = std::max(0, std::min(height - 1, min_y));
                    max_x = std::min(width - 1, std::max(0, max_x));
                    max_y = std::min(height - 1, std::max(0, max_y));
                    for (int y1 = min_y; y1 <= max_y; ++y1) {
                        for (int x1 = min_x; x1 <= max_x; ++x1) {
                            bitmap.SetPixel(x1, y1, fg_color);
                        }
                    }
                }
            }
            mask_map.Deallocate();
        }

        std::string mask_name = JoinPaths(mask_path, image.Name());
        std::string parent_dir = GetParentDir(mask_name);
        if (!boost::filesystem::exists(parent_dir)) {
            boost::filesystem::create_directories(parent_dir);
        }
        bitmap.Write(mask_name);

        progress++;
        if (progress % 100 == 0) {
            std::cout << StringPrintf("\rGeneratePanoramaMask %d / %d", progress, num_image);
        }
    };
    
    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    for (auto image_id_map : image_ids_map) {
        thread_pool->AddTask(WarpMaskSingleImage, image_id_map.first, image_id_map.second);    
    }
    thread_pool->Wait();
    std::cout << std::endl;
}

void DetectRedundantImages(
    const std::string& workspace_path, 
    const Tree& tree1, const Tree& tree2,
    const std::vector<std::string>& images_name1,
    const std::vector<std::string>& images_name2,
    const std::vector<mvs::Image>& images1,
    const std::vector<PlyPoint>& fused_points1,
    const std::vector<std::vector<uint32_t> >& points_visibility1,
    const std::vector<mvs::Image>& images2,
    const std::vector<PlyPoint>& fused_points2,
    const std::vector<std::vector<uint32_t> >& points_visibility2,
    std::vector<MatXu>& masks_map,
    std::vector<image_t>& deleted_image_ids) {
    
    const int win_r = 2;
    const int patch_size = (2 * win_r + 1) * (2 * win_r + 1);
    const float angle_diff_thres = std::cos(80 / 180.0 * M_PI);
    const double estep = 1e-4;

    std::vector<std::vector<size_t> > image_point_idxs;
    image_point_idxs.resize(images1.size());
    for (size_t i = 0; i < fused_points1.size(); ++i) {
        auto visibility = points_visibility1.at(i);
        for (auto vis : visibility) {
            image_point_idxs.at(vis).push_back(i);
        }
    }

    // Estimate Bounding Box.
    std::vector<BoundingBox> bboxes;
    for (size_t i = 0; i < images1.size(); ++i) {
        if (image_point_idxs.at(i).size() == 0) {
            bboxes.emplace_back(BoundingBox());
            continue;
        }

        Eigen::Vector3f lt, rb;
        lt.x() = lt.y() = lt.z() = std::numeric_limits<float>::max();
        rb.x() = rb.y() = rb.z() = std::numeric_limits<float>::lowest();
        for (auto idx : image_point_idxs.at(i)) {
            auto point = fused_points1.at(idx);
            lt.x() = std::min(lt.x(), point.x);
            lt.y() = std::min(lt.y(), point.y);
            lt.z() = std::min(lt.z(), point.z);
            rb.x() = std::max(rb.x(), point.x);
            rb.y() = std::max(rb.y(), point.y);
            rb.z() = std::max(rb.z(), point.z);
        }
        bboxes.emplace_back(lt, rb);
    }

    for (size_t i = 0; i < images1.size(); ++i) {
        image_point_idxs.at(i).clear();
    }

    std::cout << "Collect points inside image" << std::endl;
    for (size_t image_idx = 0; image_idx < images1.size(); ++image_idx) {
        const mvs::Image& image1 = images1.at(image_idx);
        const int width = image1.GetWidth();
        const int height = image1.GetHeight();
        
        Eigen::Vector3f C(image1.GetC());
        Eigen::Vector3f cam_ray(image1.GetViewingDirection());
        const Eigen::RowMatrix3x4f P(image1.GetP());

        const BoundingBox& bb = bboxes.at(image_idx);
        auto& image_point_idx = image_point_idxs.at(image_idx);

        for (size_t i = 0; i < fused_points1.size(); ++i) {
            auto point = fused_points1.at(i);
            const Eigen::Vector3f X(&point.x);
            if (!bb.Contains(X)) {
                continue;
            }
            
            Eigen::Vector3f point_ray = (X - C).normalized();
            // if (cam_ray.dot(point_ray) < angle_diff_thres) {
            //     continue;
            // }

            const Eigen::Vector3f Xn(&point.nx);
            if (point_ray.dot(Xn.normalized()) > 0) {
                continue;
            }

            const Eigen::Vector3f proj = P * X.homogeneous();
            const float z = proj.z();
            if (z <= 0) {
                continue;
            }
            int u = proj.x() / z;
            int v = proj.y() / z;
            if (u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }
            image_point_idx.push_back(i);
        }
        if (image_idx % 100 == 0) {
            std::cout << "\rProcess Image#" << image_idx + 1 << "/" << images1.size();
        }
    }
    std::cout << std::endl;

    auto updated_workspace_path = workspace_path + "-update";
    CreateDirIfNotExists(updated_workspace_path);
    auto updated_dense_path = JoinPaths(updated_workspace_path, DENSE_DIR);
    CreateDirIfNotExists(updated_dense_path);
    auto masks_path = JoinPaths(updated_dense_path, "masks");
    CreateDirIfNotExists(masks_path);

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));
    
    std::vector<std::future<bool> > futures(images1.size());
    masks_map.resize(images1.size());

    std::cout << "Detect Changed region" << std::endl;
    int progress = 0;
    auto ProcessImage = [&](const size_t i) {

        const mvs::Image& image1 = images1.at(i);

        const int width = image1.GetWidth();
        const int height = image1.GetHeight();

        MatXu& mask_map = masks_map.at(i);
        mask_map = MatXu(width, height, 1);
        mask_map.Fill(0);

        if (image_point_idxs.at(i).size() == 0) {
            return false;
        }

        const Eigen::RowMatrix3x4f P(image1.GetP());
        const Eigen::RowMatrix3x4f invP(image1.GetInvP());
        const Eigen::Vector3f C(image1.GetC());
        const Eigen::Vector3f cam_ray(image1.GetViewingDirection());

        MatXf depth_map = MatXf(width, height, 1);
        depth_map.Fill(0);
        MatXf color_map = MatXf(width, height, 1);
        color_map.Fill(0);
        MatXi point_idx_map = MatXi(width, height, 1);
        point_idx_map.Fill(-1);
        MatXu cover_map = MatXu(width, height, 1);
        cover_map.Fill(0);

        size_t num_image_point = 0;
        for (auto idx : image_point_idxs.at(i)) {
            auto point = fused_points1.at(idx);
            const Eigen::Vector3f X(&point.x);
            const Eigen::Vector3ub color(&point.r);
            const Eigen::Vector3f proj = P * X.homogeneous();
            const float z = proj.z();
            int u = proj.x() / z;
            int v = proj.y() / z;

            float d = depth_map.Get(v, u);
            if (d == 0 || d > z) {
                // Construct segment query.
                Eigen::Vector3f point_ray = (X - C).normalized();
                Eigen::Vector3f query_point = X - point_ray * estep;
                Point a(query_point[0], query_point[1], query_point[2]);
                Point b(C[0], C[1], C[2]);
                Segment segment_query(a, b);

                // Test intersection with segment query.
                if (tree1.do_intersect(segment_query)) {
                    continue;
                }

                if (d == 0) {
                    num_image_point++;
                }
                depth_map.Set(v, u, z);

                float gray = RGB2Gray(color.x(), color.y(), color.z());
                color_map.Set(v, u, gray);
                point_idx_map.Set(v, u, idx);
            }
        }

        for (size_t j = 0; j < fused_points2.size(); ++j) {
            auto point2 = fused_points2.at(j);
            auto visibility2 = points_visibility2.at(j);
            const Eigen::Vector3f X(&point2.x);
            const Eigen::Vector3f Xn(&point2.nx);
            const Eigen::Vector3ub color(&point2.r);
            const float gray2 = RGB2Gray(color.x(), color.y(), color.z());

            Eigen::Vector3f point_ray = (X - C).normalized();
            // if (cam_ray.dot(point_ray) < angle_diff_thres) {
            //     continue;
            // }
            if (point_ray.dot(Xn.normalized()) > 0) {
                continue;
            }

            const Eigen::Vector3f proj = P * X.homogeneous();
            float z = proj.z();
            if (z <= 0) {
                continue;
            }

            int u = proj.x() / z;
            int v = proj.y() / z;
            if (u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }

            // Construct segment query.
            Eigen::Vector3f query_point = X - point_ray * estep;
            Point a(query_point[0], query_point[1], query_point[2]);
            Point b(C[0], C[1], C[2]);
            Segment segment_query(a, b);

            // Test intersection with segment query.
            if (tree2.do_intersect(segment_query)) {
                continue;
            }

            int u_min = std::max(u - win_r, 0);
            int v_min = std::max(v - win_r, 0);
            int u_max = std::min(u + win_r, width - 1);
            int v_max = std::min(v + win_r, height - 1);

            for (int y = v_min; y <= v_max; ++y) {
                for (int x = u_min; x <= u_max; ++x) {
                    cover_map.Set(y, x, 1);
                    const float d = depth_map.Get(y, x);
                    if (d <= 0) {
                        continue;
                    }
                    const float gray1 = color_map.Get(y, x);
                    const float gray_diff = std::fabs(gray1 - gray2);
                    if (gray_diff > max_intensity_diff) {
                        continue;
                    }

                    const float ratio = std::fabs(z - d) / std::max(z, d);
                    if (ratio > max_depth_error) {
                        continue;
                    }
                    point_idx_map.Set(y, x, -1);
                    depth_map.Set(y, x, 0);
                }
            }
        }

        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                if (!cover_map.Get(r, c)) {
                    continue;
                }
                int point_idx = point_idx_map.Get(r, c);
                if (point_idx != -1) {
                    int min_r = std::max(0, r - win_r);
                    int min_c = std::max(0, c - win_r);
                    int max_r = std::min(height - 1, r + win_r);
                    int max_c = std::min(width - 1, c + win_r);
                    int num_fill_pixel = 0;
                    for (int y = min_r; y <= max_r; ++y) {
                        for (int x = min_c; x <= max_c; ++x) {
                            int npoint_idx = point_idx_map.Get(y, x);
                            if (npoint_idx != -1 && y != r && x != c) {
                                num_fill_pixel++;
                            }
                        }
                    }
                    if (num_fill_pixel * 5 < patch_size) {
                        continue;
                    }
                    for (int y = min_r; y <= max_r; ++y) {
                        for (int x = min_c; x <= max_c; ++x) {
                            mask_map.Set(y, x, 255);
                        }
                    }
                }
            }
        }
        {
            std::string mask_path = JoinPaths(masks_path, images_name1.at(i) + ".mask.jpg");
            std::string parent_path = GetParentDir(mask_path);
            if (!boost::filesystem::exists(parent_path)) {
                boost::filesystem::create_directories(parent_path);
            }
            Bitmap bitmap;
            bitmap.Allocate(width, height, false);
            for (int r = 0; r < height; ++r) {
                for (int c = 0; c < width; ++c) {
                    BitmapColor<uint8_t> color;
                    color.r = mask_map.Get(r, c);
                    bitmap.SetPixel(c, r, color);
                }
            }
            bitmap.Write(mask_path);
            bitmap.Deallocate();
        }
        progress++;
        if (progress % 100 == 0) {
            std::cout << "\rProcess Image#" << progress << "/" << images1.size() << std::flush;
        }
        return false;
    };

    deleted_image_ids.clear();
    for (size_t i = 0; i < images1.size(); ++i) {
        futures[i] = thread_pool->AddTask(ProcessImage, i);
    }
    for (size_t i = 0; i < images1.size(); ++i) {
        if (futures[i].get()) {
            deleted_image_ids.push_back(i);
        }
    }
    std::cout << std::endl;
}

void LoadSceneGraph(const std::string& workspace_path,
                    const bool camera_rig,
                    SceneGraphContainer& scene_graph) {
    FeatureDataContainer feature_data_container;
    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.txt"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.txt"))) {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), false);
            feature_data_container.ReadLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
        } else {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), camera_rig);
        }
        feature_data_container.ReadImagesData(JoinPaths(workspace_path, "/features.txt"));
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.txt"))) {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), false);
            feature_data_container.ReadLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
        } else {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), camera_rig);
        }
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        exist_feature_file = true;
    }

    // If the camera model is Spherical
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.txt"))) {
            feature_data_container.ReadSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));
        } else {
            std::cout << " Error! Existing feature data do not contain sub_panorama data" << std::endl;
            exit(-1);
        }
    }

    std::cout << JoinPaths(workspace_path, "/scene_graph.bin") << std::endl;
    bool load_scene_graph = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
        load_scene_graph = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.txt"))) {
        scene_graph.ReadSceneGraphData(JoinPaths(workspace_path, "/scene_graph.txt"));
        load_scene_graph = true;
    }
    if (!load_scene_graph) {
        return;
    }

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph.Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph.Cameras();

    std::vector<image_t> image_ids = feature_data_container.GetImageIds();

    for (const auto image_id : image_ids) {
        const Image &image = feature_data_container.GetImage(image_id);
        if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        images[image_id] = image;

        const FeatureKeypoints &keypoints = feature_data_container.GetKeypoints(image_id);
        images[image_id].SetPoints2D(keypoints);
        const PanoramaIndexs & panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);

        const Camera &camera = feature_data_container.GetCamera(image.CameraId());

        std::vector<uint32_t> local_image_indices(keypoints.size());
        for(size_t i = 0; i<keypoints.size(); ++i){
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        images[image_id].SetLocalImageIndices(local_image_indices);

        if (!scene_graph.ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
        }

        if (scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(
                scene_graph.CorrespondenceGraph()->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(
                scene_graph.CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
        } else {
            std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
        }
    }

    scene_graph.CorrespondenceGraph()->Finalize();
    std::cout << "LoadSceneGraph Done!" << std::endl;
}

void UpdataSparseMap(const std::string& workspace_path,
               const int reconstruction_idx,
               const std::vector<image_t>& image_ids1,
               const std::vector<MatXu>& masks_map1) {
    SceneGraphContainer scene_graph;
    LoadSceneGraph(workspace_path, false, scene_graph);

    auto rec_path = JoinPaths(workspace_path, std::to_string(reconstruction_idx));
    Reconstruction reconstruction;
    reconstruction.ReadBinary(rec_path);

    std::cout << "Update Sparse Reconstruction" << std::endl;
    // Update Sparse Map.
    for (auto image_id : image_ids1) {
        const MatXu& mask = masks_map1.at(image_id);
        class Image& image = reconstruction.Image(image_id);
        size_t num_point2D = image.NumPoints2D();
        for (point2D_t point2D_idx = 0; point2D_idx < num_point2D; ++point2D_idx) {
            const class Point2D& point2D = image.Point2D(point2D_idx);
            if (!point2D.HasMapPoint()) {
                continue;
            }
            int x = point2D.X();
            int y = point2D.Y();
            if (mask.Get(y, x)) {
                reconstruction.DeleteMapPoint(point2D.MapPointId());
            }
        }
    }

    // std::cout << "Update SceneGraph" << std::endl;
    // // Update SceneGraph.
    // for (auto image_id : image_ids1) {
    //     if (scene_graph.ExistsImage(image_id)) {
    //         std::cout << "Image#" << image_id << std::endl;
    //         const MatXu& mask = masks_map1.at(image_id);
    //         scene_graph.DeleteCorrespondences(image_id, mask);
    //     }
    // }

    rec_path = rec_path + "-update";
    CreateDirIfNotExists(rec_path);
    reconstruction.WriteBinary(rec_path);

    // auto update_scene_path = JoinPaths(workspace_path, "updated-scene");
    // CreateDirIfNotExists(update_scene_path);

    // feature_data_container.WriteImagesBinaryData(JoinPaths(update_scene_path, "/features.bin"));
    // feature_data_container.WriteCamerasBinaryData(JoinPaths(update_scene_path, "/cameras.bin"));
    // feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(update_scene_path, "/local_cameras.bin"));
    // feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(update_scene_path, "/sub_panorama.bin"));
    // scene_graph.WriteSceneGraphBinaryData(JoinPaths(update_scene_path, "/scene_graph.bin"));
    // scene_graph.CorrespondenceGraph()->ExportToGraph(update_scene_path + "/scene_graph.png");
}

void UpdateDenseMap(const std::string& workspace_path,
               const int reconstruction_idx,
               const std::vector<PlyPoint>& fused_points1,
               const std::vector<std::vector<uint32_t> >& points_visibility1,
               const std::vector<PlyPoint>& fused_points2,
               const std::vector<std::vector<uint32_t> >& points_visibility2,
               const std::vector<mvs::Image>& images1,
               const std::vector<MatXu>& masks_map1) {
    std::cout << "Update DenseMap" << std::endl;

    auto rec_path = JoinPaths(workspace_path, std::to_string(reconstruction_idx));
    // Reconstruction reconstruction;
    // reconstruction.ReadBinary(rec_path);

    rec_path = rec_path + "-update";
    CreateDirIfNotExists(rec_path);

    // Update Dense Map.
    std::vector<PlyPoint> fused_points = fused_points2;
    fused_points.reserve(fused_points1.size() + fused_points2.size());
    // std::vector<PlyPoint> deleted_points;
    // std::vector<std::vector<uint32_t> > deleted_visibilitys;

    std::vector<std::vector<uint32_t> > points_visibility;
    for (size_t j = 0; j < fused_points1.size(); ++j) {
        Eigen::Map<const Eigen::Vector3f> X(&fused_points1.at(j).x);
        auto visibility = points_visibility1.at(j);
        bool removal = false;
        for (auto vis : visibility) {
            const mvs::Image& image = images1.at(vis);
            Eigen::Map<const Eigen::RowMatrix3x4f> P(image.GetP());
            Eigen::Vector3f proj = P * X.homogeneous();
            if (proj.z() <= 0) {
                continue;
            }
            int u = proj.x() / proj.z();
            int v = proj.y() / proj.z();
            if (u >= 0 && u < image.GetWidth() && 
                v >= 0 && v < image.GetHeight() &&
                masks_map1.at(vis).Get(v, u)) {
                removal = true;
                break;
            }
        }
        if (!removal) {
            fused_points.emplace_back(fused_points1.at(j));
            points_visibility.emplace_back(visibility);
        }
    }

    auto dense_path = JoinPaths(rec_path, "dense");
    CreateDirIfNotExists(dense_path);
    std::cout << dense_path << std::endl;
    auto fused_path = JoinPaths(dense_path, FUSION_NAME);
    WriteBinaryPlyPoints(fused_path, fused_points, false, true);
    WritePointsVisibility(fused_path + ".vis", points_visibility);
}

void Fix(TriangleMesh& mesh) {
    const int valid_label = 1;
    std::unordered_map<size_t, std::vector<size_t> > adj_facets_per_vert;
    for (size_t i = 0; i < mesh.faces_.size(); ++i) {
        auto facet = mesh.faces_[i];
        if (mesh.vertex_labels_[facet[0]] == valid_label) {
            adj_facets_per_vert[facet[0]].push_back(i);
        }
        if (mesh.vertex_labels_[facet[1]] == valid_label) {
            adj_facets_per_vert[facet[1]].push_back(i);
        }
        if (mesh.vertex_labels_[facet[2]] == valid_label) {
            adj_facets_per_vert[facet[2]].push_back(i);
        }
    }

    std::vector<bool> visited(mesh.vertices_.size(), false);
    for (auto adj_facets : adj_facets_per_vert) {
        if (visited[adj_facets.first]) {
            continue;
        }
        std::queue<size_t> Q;
        Q.push(adj_facets.first);
        visited.at(adj_facets.first) = true;

        std::vector<size_t> connected_verts;
        connected_verts.push_back(adj_facets.first);
        while(!Q.empty()) {
            int vert_id = Q.front();
            Q.pop();
            auto adj_facet_ids = adj_facets_per_vert.at(vert_id);
            for (auto facet_id : adj_facet_ids) {
                auto facet = mesh.faces_[facet_id];
                if (facet[0] != vert_id && !visited[facet[0]]) {
                    if (mesh.vertex_labels_[facet[0]] == valid_label) {
                        Q.push(facet[0]);
                        visited.at(facet[0]) = true;
                        connected_verts.push_back(facet[0]);
                    }
                }
                if (facet[1] != vert_id && !visited[facet[1]]) {
                    if (mesh.vertex_labels_[facet[1]] == valid_label) {
                        Q.push(facet[1]);
                        visited.at(facet[1]) = true;
                        connected_verts.push_back(facet[1]);
                    }
                }
                if (facet[2] != vert_id && !visited[facet[2]]) {
                    if (mesh.vertex_labels_[facet[2]] == valid_label) {
                        Q.push(facet[2]);
                        visited.at(facet[2]) = true;
                        connected_verts.push_back(facet[2]);
                    }
                }
            }
        }
        if (connected_verts.size() < 100) {
            for (auto vert_id : connected_verts) {
                mesh.vertex_labels_.at(vert_id) = 0;
            
                Eigen::Vector3d rgb;
                rgb[0] = adepallete[3];
                rgb[1] = adepallete[4];
                rgb[2] = adepallete[5];
                mesh.vertex_colors_.at(vert_id) = rgb;
            }
        }
    }

    auto FixNonConsistentLabel = [&]() {
        std::vector<Eigen::Vector3i> facets;
        for (size_t i = 0; i < mesh.faces_.size(); ++i) {
            auto facet = mesh.faces_[i];
            if ((mesh.vertex_labels_[facet[0]] == valid_label &&
                mesh.vertex_labels_[facet[1]] == valid_label &&
                mesh.vertex_labels_[facet[2]] == valid_label) ||
                (mesh.vertex_labels_[facet[0]] != valid_label &&
                mesh.vertex_labels_[facet[1]] != valid_label &&
                mesh.vertex_labels_[facet[2]] != valid_label)) {
                continue;
            }
            facets.emplace_back(facet);
        }

        Eigen::Vector3d rgb;
        rgb[0] = adepallete[6];
        rgb[1] = adepallete[7];
        rgb[2] = adepallete[8];
        for (auto facet : facets) {
            mesh.vertex_labels_.at(facet[0]) = 1;
            mesh.vertex_labels_.at(facet[1]) = 1;
            mesh.vertex_labels_.at(facet[2]) = 1;

            mesh.vertex_colors_.at(facet[0]) = rgb;
            mesh.vertex_colors_.at(facet[1]) = rgb;
            mesh.vertex_colors_.at(facet[2]) = rgb;
        }
        return facets.size();
    };

    int i, iter = 0;
    while((i = FixNonConsistentLabel()) && iter < 3) {
        iter++;
        std::cout << StringPrintf("Fix %d / %d vertex labels",
            i, mesh.vertex_labels_.size()) << std::endl;
    }
}

void ModelChangeDetection(const std::vector<mvs::Image>& images,
                          const std::vector<MatXu>& masks_map,
                          const Tree& tree, TriangleMesh& mesh) {
    const double thresh_fov = std::cos(85.0);
    const double estep = 1e-6;

    size_t num_vert = mesh.vertices_.size();
    int progress = 0;
    auto ComputeVertexLabel = [&](const int i, int8_t* vert_label) {
        const auto& vert = mesh.vertices_.at(i);
        const auto& vert_normal = mesh.vertex_normals_.at(i);

        int samps_per_label[2] = { 0, 0 };
        for (size_t cam_id = 0; cam_id < images.size(); ++cam_id) {
            const mvs::Image& image = images.at(cam_id);
            Eigen::Vector3d cam_ray = 
                Eigen::Vector3f(image.GetViewingDirection()).cast<double>();
            Eigen::Vector3d C = Eigen::Vector3f(image.GetC()).cast<double>();
            Eigen::Vector3d point_ray = (vert - C).normalized();
            double cam_angle = cam_ray.dot(point_ray);
            double ray_angle = point_ray.dot(-vert_normal);
            if (cam_angle < thresh_fov || ray_angle < 0.15) {
                continue;
            }

            Eigen::RowMatrix3d K = 
                Eigen::RowMatrix3f(image.GetK()).cast<double>();
            Eigen::RowMatrix3d R = 
                Eigen::RowMatrix3f(image.GetR()).cast<double>();
            Eigen::Vector3d T = 
                Eigen::Vector3f(image.GetT()).cast<double>();

            Eigen::Vector3d proj = K * (R * vert + T);
            int u = proj[0] / proj[2];
            int v = proj[1] / proj[2];
            if (proj[2] <= 0 ||
                u < 0 || u >= image.GetWidth() || 
                v < 0 || v >= image.GetHeight()) {
                continue;
            }
            
            // Construct segment query.
            Eigen::Vector3d query_point = vert - point_ray * estep;
            Point a(query_point[0], query_point[1], query_point[2]);
            Point b(C[0], C[1], C[2]);
            Segment segment_query(a, b);

            // Test intersection with segment query.
            if (tree.do_intersect(segment_query)) {
                continue;
            }

            if (masks_map.at(cam_id).Get(v, u)) {
                samps_per_label[1]++;
            } else {
                samps_per_label[0]++;
            }
        }
        *vert_label = 0;
        if (samps_per_label[1] > samps_per_label[0]) {
            *vert_label = 1;
        }
        ++progress;
        if (progress % 100 == 0) {
            std::cout << StringPrintf("\rProcess Point %d/%d", progress, num_vert);
        }
    };

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    // const int num_eff_threads = 1;
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    mesh.vertex_labels_.resize(num_vert);
    std::fill(mesh.vertex_labels_.begin(), mesh.vertex_labels_.end(), 0);

    for (size_t i = 0; i < num_vert; ++i) {
        thread_pool->AddTask(ComputeVertexLabel, i, &mesh.vertex_labels_[i]); 
    }
    thread_pool->Wait();
    std::cout << std::endl;

    mesh.vertex_colors_.resize(num_vert);
    for (size_t i = 0; i < num_vert; ++i) {
        int best_label = mesh.vertex_labels_[i];
        best_label = (best_label + 256) % 256 + 1;
        Eigen::Vector3d rgb;
        rgb[0] = adepallete[best_label * 3];
        rgb[1] = adepallete[best_label * 3 + 1];
        rgb[2] = adepallete[best_label * 3 + 2];
        mesh.vertex_colors_.at(i) = rgb;
    }
}

void GenerateMask(const std::string& workspace_path, 
                  const Tree& tree, const TriangleMesh& mesh, 
                  const Reconstruction& reconstruction) {
    const double thresh_fov = std::cos(85.0);
    const double estep = 1e-6;

    auto image_ids = reconstruction.RegisterImageIds();

    int progress = 0;
    auto GenerateMaskSingleImage = [&](image_t image_id) {
        const class Image& image = reconstruction.Image(image_id);
        const class Camera& camera = reconstruction.Camera(image.CameraId());

        const Eigen::Vector3d cam_ray = image.ViewingDirection();
        const Eigen::Vector3d C = image.ProjectionCenter();
        const Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();

        const int width = camera.Width();
        const int height = camera.Height();

        std::vector<bool> vert_visibility(mesh.vertices_.size(), false);
        for (size_t v_idx = 0; v_idx < mesh.vertices_.size(); ++v_idx) {
            if (mesh.vertex_labels_.at(v_idx) == 0) {
                continue;
            }
            auto vert = mesh.vertices_.at(v_idx);
            auto vert_normal = mesh.vertex_normals_.at(v_idx);
            Eigen::Vector3d point_ray = (vert - C).normalized();
            double cam_angle = cam_ray.dot(point_ray);
            double ray_angle = point_ray.dot(-vert_normal);
            if (cam_angle < thresh_fov || ray_angle < 0.15) {
                continue;
            }

            Eigen::Vector3d Xc = proj_matrix * vert.homogeneous();
            Eigen::Vector2d world_point(Xc[0] / Xc[2], Xc[1] / Xc[2]);
            Eigen::Vector2d proj = camera.WorldToImage(world_point);
            int u = proj[0];
            int v = proj[1];
            if (Xc[2] <= 0 || u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }
            
            // Construct segment query.
            Eigen::Vector3d query_point = vert - point_ray * estep;
            Point a(query_point[0], query_point[1], query_point[2]);
            Point b(C[0], C[1], C[2]);
            Segment segment_query(a, b);

            // Test intersection with segment query.
            if (tree.do_intersect(segment_query)) {
                continue;
            }
            vert_visibility.at(v_idx) = true;
        }

        Bitmap bitmap;
        bitmap.Allocate(width, height, false);
        BitmapColor<uint8_t> bk_color(0);
        BitmapColor<uint8_t> fg_color(255);
        bitmap.Fill(bk_color);

        for (size_t i = 0; i < mesh.faces_.size(); ++i) {
            const Eigen::Vector3i& facet = mesh.faces_[i];
            int num_vis = (int)vert_visibility.at(facet[0]) +
                          (int)vert_visibility.at(facet[1]) +
                          (int)vert_visibility.at(facet[2]);
            if (num_vis < 2) {
                continue;
            }

            Eigen::Vector3d Xc;
            Eigen::Vector2d uv[3];
            Xc = proj_matrix * mesh.vertices_[facet[0]].homogeneous();
            if (Xc[2] <= 0) continue;
            uv[0] = camera.WorldToImage(Eigen::Vector2d(Xc[0] / Xc[2], Xc[1] / Xc[2]));

            Xc = proj_matrix * mesh.vertices_[facet[1]].homogeneous();
            if (Xc[2] <= 0) continue;
            uv[1] = camera.WorldToImage(Eigen::Vector2d(Xc[0] / Xc[2], Xc[1] / Xc[2]));

            Xc = proj_matrix * mesh.vertices_[facet[2]].homogeneous();
            if (Xc[2] <= 0) continue;
            uv[2] = camera.WorldToImage(Eigen::Vector2d(Xc[0] / Xc[2], Xc[1] / Xc[2]));
            
            int u_min = std::min(uv[0][0], std::min(uv[1][0], uv[2][0]));
            int u_max = std::max(uv[0][0], std::max(uv[1][0], uv[2][0]));
            int v_min = std::min(uv[0][1], std::min(uv[1][1], uv[2][1]));
            int v_max = std::max(uv[0][1], std::max(uv[1][1], uv[2][1]));
            u_min = std::max(0, u_min);
            v_min = std::max(0, v_min);
            u_max = std::min(width - 1, u_max);
            v_max = std::min(height - 1, v_max);
            if (u_min >= u_max || v_min >= v_max) {
                continue;
            }

            Eigen::Vector2d v1 = (uv[1] - uv[0]);
            Eigen::Vector2d v2 = (uv[2] - uv[1]);
            Eigen::Vector2d v3 = (uv[0] - uv[2]);

            for (int v = v_min; v <= v_max; ++v) {
                for (int u = u_min; u <= u_max; ++u) {
                    double m1 = v1.x() * (v - uv[0][1]) - (u - uv[0][0]) * v1.y();
                    double m2 = v2.x() * (v - uv[1][1]) - (u - uv[1][0]) * v2.y();
                    double m3 = v3.x() * (v - uv[2][1]) - (u - uv[2][0]) * v3.y();
                    if (m1 < 0 && m2 < 0 && m3 < 0) {
                       bitmap.SetPixel(u, v, fg_color);
                    }
                }
            }
        }

        std::string mask_path = JoinPaths(workspace_path, "masks", image.Name() + ".mask.jpg");
        std::string parent_path = GetParentDir(mask_path);
        if (!boost::filesystem::exists(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        bitmap.Write(mask_path);
        bitmap.Deallocate();

        progress++;
        if (progress % 100 == 0) {
            std::cout << StringPrintf("\rGenerateMask %d / %d", progress, image_ids.size());
        }
    };

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    for (auto image_id : image_ids) {
        thread_pool->AddTask(GenerateMaskSingleImage, image_id);    
    }
    thread_pool->Wait();

    std::cout << std::endl;
}

void GeneratePerspectiveMask(const std::string& workspace_path, 
                             const Tree& tree, const TriangleMesh& mesh, 
                             const std::vector<mvs::Image>& images,
                             const std::vector<std::string>& images_name) {
    const double thresh_fov = std::cos(85.0);
    const double estep = 1e-6;

    auto GenerateMaskSingleImage = [&](size_t image_idx) {
        mvs::Image image = images.at(image_idx);
        const Eigen::Vector3d cam_ray = Eigen::Vector3f(image.GetViewingDirection()).cast<double>();
        const Eigen::Vector3d C = Eigen::Vector3f(image.GetC()).cast<double>();
        const Eigen::Matrix3x4d proj_matrix(Eigen::RowMatrix3x4f(image.GetP()).cast<double>());

        const int width = image.GetWidth();
        const int height = image.GetHeight();

        std::vector<bool> vert_visibility(mesh.vertices_.size(), false);
        for (size_t v_idx = 0; v_idx < mesh.vertices_.size(); ++v_idx) {
            if (mesh.vertex_labels_.at(v_idx) == 0) {
                continue;
            }
            auto vert = mesh.vertices_.at(v_idx);
            auto vert_normal = mesh.vertex_normals_.at(v_idx);
            Eigen::Vector3d point_ray = (vert - C).normalized();
            double cam_angle = cam_ray.dot(point_ray);
            double ray_angle = point_ray.dot(-vert_normal);
            if (cam_angle < thresh_fov || ray_angle < 0.15) {
                continue;
            }

            Eigen::Vector3d proj = proj_matrix * vert.homogeneous();
            int u = proj[0] / proj[2];
            int v = proj[1] / proj[2];
            if (proj[2] <= 0 || u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }
            
            // Construct segment query.
            Eigen::Vector3d query_point = vert - point_ray * estep;
            Point a(query_point[0], query_point[1], query_point[2]);
            Point b(C[0], C[1], C[2]);
            Segment segment_query(a, b);

            // Test intersection with segment query.
            if (tree.do_intersect(segment_query)) {
                continue;
            }
            vert_visibility.at(v_idx) = true;
        }

        Bitmap bitmap;
        bitmap.Allocate(width, height, false);
        BitmapColor<uint8_t> bk_color(0);
        BitmapColor<uint8_t> fg_color(255);
        bitmap.Fill(bk_color);

        for (size_t i = 0; i < mesh.faces_.size(); ++i) {
            const Eigen::Vector3i& facet = mesh.faces_[i];
            int num_vis = (int)vert_visibility.at(facet[0]) +
                          (int)vert_visibility.at(facet[1]) +
                          (int)vert_visibility.at(facet[2]);
            if (num_vis < 2) {
                continue;
            }

            Eigen::Vector3d Xc;
            Eigen::Vector2d uv[3];
            Xc = proj_matrix * mesh.vertices_[facet[0]].homogeneous();
            if (Xc[2] <= 0) continue;
            uv[0] = (Xc / Xc[2]).head<2>();

            Xc = proj_matrix * mesh.vertices_[facet[1]].homogeneous();
            if (Xc[2] <= 0) continue;
            uv[1] = (Xc / Xc[2]).head<2>();

            Xc = proj_matrix * mesh.vertices_[facet[2]].homogeneous();
            if (Xc[2] <= 0) continue;
            uv[2] = (Xc / Xc[2]).head<2>();
            
            int u_min = std::min(uv[0][0], std::min(uv[1][0], uv[2][0]));
            int u_max = std::max(uv[0][0], std::max(uv[1][0], uv[2][0]));
            int v_min = std::min(uv[0][1], std::min(uv[1][1], uv[2][1]));
            int v_max = std::max(uv[0][1], std::max(uv[1][1], uv[2][1]));
            u_min = std::max(0, u_min);
            v_min = std::max(0, v_min);
            u_max = std::min(width - 1, u_max);
            v_max = std::min(height - 1, v_max);
            if (u_min >= u_max || v_min >= v_max) {
                continue;
            }

            Eigen::Vector2d v1 = (uv[1] - uv[0]);
            Eigen::Vector2d v2 = (uv[2] - uv[1]);
            Eigen::Vector2d v3 = (uv[0] - uv[2]);

            for (int v = v_min; v <= v_max; ++v) {
                for (int u = u_min; u <= u_max; ++u) {
                    double m1 = v1.x() * (v - uv[0][1]) - (u - uv[0][0]) * v1.y();
                    double m2 = v2.x() * (v - uv[1][1]) - (u - uv[1][0]) * v2.y();
                    double m3 = v3.x() * (v - uv[2][1]) - (u - uv[2][0]) * v3.y();
                    if (m1 < 0 && m2 < 0 && m3 < 0) {
                       bitmap.SetPixel(u, v, fg_color);
                    }
                }
            }
        }

        std::string mask_path = JoinPaths(workspace_path, DENSE_DIR, "masks", images_name.at(image_idx) + ".mask.jpg");
        std::string parent_path = GetParentDir(mask_path);
        if (!boost::filesystem::exists(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        bitmap.Write(mask_path);
        bitmap.Deallocate();

        if (image_idx % 100 == 0) {
            std::cout << StringPrintf("\rGeneratePerspectiveMask %d / %d", 
                                        image_idx + 1, images.size());
        }
    };

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
        thread_pool->AddTask(GenerateMaskSingleImage, image_idx);    
    }
    thread_pool->Wait();
    std::cout << std::endl;
}

void LoadSparseModel(const std::string workspace_path,
                     const std::string format,
                     std::vector<mvs::Image>& images,
                     std::vector<image_t>& image_ids,
                     std::vector<std::string>& images_name) {
    images.clear();
    image_ids.clear();
    images_name.clear();

    auto dense_reconstruction_path = JoinPaths(workspace_path, DENSE_DIR);
    auto undistort_image_path = JoinPaths(dense_reconstruction_path, IMAGES_DIR);
    if (format.compare("panorama") == 0) {
        std::vector<std::vector<int> > overlapping_images;
        std::vector<std::pair<float, float> > depth_ranges;
        ImportPanoramaWorkspace(dense_reconstruction_path, images_name, 
            images, image_ids, overlapping_images, depth_ranges, false);
    } else {
        mvs::Model model;
        model.Read(dense_reconstruction_path, undistort_image_path, format);
        images = model.images;
        for (size_t i = 0; i < images.size(); ++i) {
            images_name.push_back(model.GetImageName(i));
            image_ids.push_back(model.GetImageId(i));
        }
    }
}

int main(int argc, char *argv[]) {
    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
	std::string image_type = param.GetArgument("image_type", "perspective");
    max_depth_error = param.GetArgument("max_depth_error", 0.01f);
    max_intensity_diff = param.GetArgument("max_intensity_diff", 30.0f);

	PrintHeading("SenseMap.  Copyright(c) 2020, SenseTime Group.");

    int num_reconstruction = 0;
    for (size_t reconstruction_idx = 0; ; reconstruction_idx++) {
        
        Timer timer;
        timer.Start();

        auto rec_path = JoinPaths(workspace_path, std::to_string(reconstruction_idx));
        if (!ExistsDir(rec_path)) {
            break;
        }
        auto dense_path = JoinPaths(rec_path, DENSE_DIR);
        if (!ExistsDir(dense_path)) {
            continue;
        }
        
        auto fused_path = JoinPaths(dense_path, FUSION_NAME);
        auto fused_vis_path = fused_path + ".vis";
        std::cout << fused_path << std::endl;
        if (!ExistsFile(fused_path) || !ExistsFile(fused_vis_path)) {
            continue;
        }

        auto model_path = JoinPaths(dense_path, MODEL_NAME);
        if (!ExistsFile(model_path)) {
            continue;
        }

        auto update_path = JoinPaths(workspace_path, "map_update");
        std::cout << update_path << std::endl;
        if (!ExistsDir(update_path)) {
            continue;
        }
        auto update_rec_path = JoinPaths(update_path, std::to_string(reconstruction_idx));
        if (!ExistsDir(update_rec_path)) {
            continue;
        }
        auto update_dense_path = JoinPaths(update_rec_path, DENSE_DIR);
        if (!ExistsDir(update_dense_path)) {
            continue;
        }
        auto update_fused_path = JoinPaths(update_dense_path, FUSION_NAME);
        auto update_fused_vis_path = update_fused_path + ".vis";
        std::cout << update_fused_path << std::endl;
        if (!ExistsFile(update_fused_path) || 
            !ExistsFile(update_fused_vis_path)) {
            continue;
        }

        auto update_model_path = JoinPaths(update_dense_path, MODEL_NAME);
        if (!ExistsFile(update_model_path)) {
            continue;
        }

        std::cout << "Load Cameras" << std::endl;
        std::vector<mvs::Image> images1;
        std::vector<image_t> image_ids1;
        std::vector<std::string> images_name1;
        LoadSparseModel(rec_path, image_type, images1, image_ids1, images_name1);

        Reconstruction reconstruction1;
        reconstruction1.ReadBinary(rec_path);

        std::cout << "Load Map dense points" << std::endl;
        const std::vector<PlyPoint>& fused_points1 = ReadPly(fused_path);
        std::vector<std::vector<uint32_t> > points_visibility1;
        ReadPointsVisibility(fused_vis_path, points_visibility1);

        std::cout << "Load Update Cameras" << std::endl;
        std::vector<mvs::Image> images2;
        std::vector<image_t> image_ids2;
        std::vector<std::string> images_name2;
        LoadSparseModel(update_rec_path, image_type, images2, image_ids2, images_name2);

        Reconstruction reconstruction2;
        reconstruction2.ReadBinary(update_rec_path);

        std::cout << "Load Update Map dense points" << std::endl;
        const std::vector<PlyPoint>& fused_points2 = ReadPly(update_fused_path);
        std::vector<std::vector<uint32_t> > points_visibility2;
        ReadPointsVisibility(update_fused_vis_path, points_visibility2);

        TriangleMesh mesh;
        ReadTriangleMeshObj(model_path, mesh);

        std::list<Triangle> triangles;
        for (auto facet : mesh.faces_) {
            auto a = mesh.vertices_.at(facet[0]);
            auto b = mesh.vertices_.at(facet[1]);
            auto c = mesh.vertices_.at(facet[2]);
            if ((a - b).norm() < 1e-6 || (b - c).norm() < 1e-6 || (a - c).norm() < 1e-6) {
                continue;
            }
            Triangle tri(Point(a[0], a[1], a[2]), Point(b[0], b[1], b[2]), 
                        Point(c[0], c[1], c[2]));
            triangles.emplace_back(tri);
        }
        std::cout << "Construct Origin Model AABB Tree" << std::endl;
        // Contruct AABB tree.
        Tree tree(triangles.begin(), triangles.end());

        TriangleMesh mesh2;
        ReadTriangleMeshObj(update_model_path, mesh2);

        std::list<Triangle> triangles2;
        for (auto facet : mesh2.faces_) {
            auto a = mesh2.vertices_.at(facet[0]);
            auto b = mesh2.vertices_.at(facet[1]);
            auto c = mesh2.vertices_.at(facet[2]);
            if ((a - b).norm() < 1e-6 || (b - c).norm() < 1e-6 || (a - c).norm() < 1e-6) {
                continue;
            }
            Triangle tri(Point(a[0], a[1], a[2]), Point(b[0], b[1], b[2]), 
                        Point(c[0], c[1], c[2]));
            triangles2.emplace_back(tri);
        }
        std::cout << "Construct Updated Model AABB Tree" << std::endl;
        // Contruct AABB tree.
        Tree tree2(triangles2.begin(), triangles2.end());
        
        std::vector<image_t> deleted_image_ids;
        std::vector<MatXu> masks_map1;
        DetectRedundantImages(update_rec_path, tree, tree2, 
                              images_name1, images_name2,
                              images1, fused_points1, points_visibility1,
                              images2, fused_points2, points_visibility2,
                              masks_map1, deleted_image_ids);

        UpdateDenseMap(update_path, reconstruction_idx, 
                       fused_points1, points_visibility1,
                       fused_points2, points_visibility2, 
                       images1, masks_map1);

        ModelChangeDetection(images1, masks_map1, tree, mesh);
        Fix(mesh);

        masks_map1.clear();
        masks_map1.shrink_to_fit();
        
        auto changed_model_path = JoinPaths(dense_path, "model_changed.obj");
        WriteTriangleMeshObj(changed_model_path, mesh);

        if (image_type.compare("panorama") == 0) {
            GeneratePerspectiveMask(rec_path, tree, mesh, images1, images_name1);

            const mvs::Image& dimage1 = images1.at(0);
            auto registered_ids1 = reconstruction1.RegisterImageIds();
            const class Image& simage1 = reconstruction1.Image(registered_ids1.at(0));
            const class Camera& scamera1 = reconstruction1.Camera(simage1.CameraId());
            Panorama panorama1;
            panorama1.PerspectiveParamsProcess(dimage1.GetWidth(), fov_w, fov_h, 
                num_perspective_per_image, scamera1.Width(), scamera1.Height());

            GeneratePanoramaMask(rec_path, reconstruction1, 
                images1, images_name1, image_ids1, panorama1);
        } else {        
            GenerateMask(rec_path, tree, mesh, reconstruction1);
        }

        timer.PrintMinutes();

        num_reconstruction++;
    }
    
    return 0;
}