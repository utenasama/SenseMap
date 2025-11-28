//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"
#include "util/obj.h"
#include "util/threading.h"
#include "texture/mesh_texture.h"
#include "mvs/image.h"
#include "mvs/workspace.h"

#include "../Configurator_yaml.h"

#include <unordered_set>

using namespace sensemap;
using namespace sensemap::mvs;

std::string configuration_file_path;

void Coloring(TriangleMesh& model, const mvs::Image& image,
              std::unordered_set<size_t>* visible_vertices_map) {
    std::cout << image.GetPath() << std::endl;
    
    const int width = image.GetWidth();
    const int height = image.GetHeight();

    const Eigen::RowMatrix3d K = Eigen::RowMatrix3f(image.GetK()).cast<double>();
    const Eigen::RowMatrix3d R = Eigen::RowMatrix3f(image.GetR()).cast<double>();
    const Eigen::Vector3d T = Eigen::Vector3f(image.GetT()).cast<double>();

    const double thresh_fov = std::cos(90.0);

    const Eigen::Vector3d C = -R.transpose() * T;
    const Eigen::Vector3d ray = R.row(2);
    const Eigen::RowMatrix3d Kinv = K.inverse();

    std::vector<std::vector<float> > depth(height, 
        std::vector<float>(width, 0.0f));
    std::vector<std::vector<size_t> > visible_facets(height,
        std::vector<size_t>(width, -1));

    for (size_t i = 0; i < model.faces_.size(); ++i) {
        if (ray.dot(model.face_normals_[i]) > 0.4) {
            continue;
        }
        const Eigen::Vector3i& facet = model.faces_[i];
        const Eigen::Vector3d p0 = model.vertices_[facet[0]];
        const Eigen::Vector3d p1 = model.vertices_[facet[1]];
        const Eigen::Vector3d p2 = model.vertices_[facet[2]];
        const Eigen::Vector3d centroid = (p0 + p1 + p2) / 3;
        
        const double dev_angle = ray.dot((centroid - C).normalized());
        // Outside of FOV.
        if (dev_angle < thresh_fov) {
            continue;
        }

        Eigen::Vector3d uv[3];
        Eigen::Vector3d Xc[3];

        Xc[0] = R * p0 + T;
        uv[0] = K * Xc[0]; uv[0] /= uv[0][2];
        Xc[1] = R * p1 + T;
        uv[1] = K * Xc[1]; uv[1] /= uv[1][2];
        Xc[2] = R * p2 + T;
        uv[2] = K * Xc[2]; uv[2] /= uv[2][2];
        
        int u_min = std::min(uv[0][0], std::min(uv[1][0], uv[2][0]));
        int u_max = std::max(uv[0][0], std::max(uv[1][0], uv[2][0]));
        int v_min = std::min(uv[0][1], std::min(uv[1][1], uv[2][1]));
        int v_max = std::max(uv[0][1], std::max(uv[1][1], uv[2][1]));
        u_min = std::max(0, u_min);
        v_min = std::max(0, v_min);
        u_max = std::min(width - 1, u_max);
        v_max = std::min(height - 1, v_max);
        if (Xc[0][2] < 0 || Xc[1][2] < 0 || Xc[2][2] < 0 ||
            u_min > u_max || v_min > v_max) {
            continue;
        }

        const float z = (Xc[0][2] + Xc[1][2] + Xc[2][2]) / 3;
        // Eigen::Vector3d edge1(Xc[1] - Xc[0]);
        // Eigen::Vector3d edge2(Xc[2] - Xc[0]);

        // const Eigen::Vector3d normal = edge1.cross(edge2).normalized();
		// const Eigen::Vector3d normalPlane = normal * (1.0 / normal.dot(Xc[0]));

        for (int v = v_min; v <= v_max; ++v) {
            for (int u = u_min; u <= u_max; ++u) {
                int t1 = (u - uv[0][0]) * (v - uv[1][1]) - (u - uv[1][0]) * (v - uv[0][1]);
                int t2 = (u - uv[1][0]) * (v - uv[2][1]) - (u - uv[2][0]) * (v - uv[1][1]);
                int t3 = (u - uv[2][0]) * (v - uv[0][1]) - (u - uv[0][0]) * (v - uv[2][1]);
                if ((t1 >= 0 && t2 >= 0 && t3 >= 0) ||
                    (t1 <= 0 && t2 <= 0 && t3 <= 0)) {
                    // Eigen::Vector3d xc = Kinv * Eigen::Vector3d(u, v, 1.0);
                    // const double z = 1.0 / normalPlane.dot(xc);
                    const float nd = depth[v][u];
                    if (nd == 0 || z < nd) {
                        depth[v][u] = z;
                        visible_facets[v][u] = i;
                    }
                }
            }
        }
    }

    std::unordered_set<size_t> visible_facets_map;
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            if (depth[r][c] <= 0) {
                continue;
            }
            size_t facet_idx = visible_facets[r][c];
            if (visible_facets_map.find(facet_idx) == visible_facets_map.end()) {
                visible_facets_map.insert(facet_idx);
                const Eigen::Vector3i& facet = model.faces_[facet_idx];
                visible_vertices_map->insert(facet[0]);
                visible_vertices_map->insert(facet[1]);
                visible_vertices_map->insert(facet[2]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    using namespace sensemap;
    using namespace texture;

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string in_model_path = param.GetArgument("in_model_path", "");
	std::string image_type = param.GetArgument("image_type", "perspective");

    std::cout << in_model_path << std::endl;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    #if 0
    TextureMapping::Options texture_options;
    texture_options.workspace_path = workspace_path;
    texture_options.model_name = in_model_path;
    TextureMapping mapper(texture_options);
    mapper.Start();
    mapper.Wait();
    #else
    std::string dense_reconstruction_path = 
        JoinPaths(workspace_path, "0", DENSE_DIR);
    std::string undistort_image_path =
        JoinPaths(dense_reconstruction_path, IMAGES_DIR);

    std::vector<mvs::Image> images;
    if (image_type.compare("panorama") == 0) {
        std::vector<std::string> image_names;
        std::vector<image_t> image_ids;
        std::vector<std::vector<int> > overlapping_images;
        std::vector<std::pair<float, float> > depth_ranges;
        ImportPanoramaWorkspace(dense_reconstruction_path, image_names, 
            images, image_ids, overlapping_images, depth_ranges, false);
    } else if (image_type.compare("perspective") == 0) {
        Workspace::Options workspace_options;
        workspace_options.max_image_size = -1;
        workspace_options.image_as_rgb = true;
        workspace_options.workspace_path = dense_reconstruction_path;
        workspace_options.workspace_format = "perspective";
        workspace_options.image_path = undistort_image_path;

        std::unique_ptr<Workspace> workspace_;
        workspace_.reset(new Workspace(workspace_options));
        const Model& model = workspace_->GetModel();
        images = model.images;
    }

    TriangleMesh model;
    ReadTriangleMeshObj(in_model_path, model, false);

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    std::chrono::high_resolution_clock::time_point start_time = 
        std::chrono::high_resolution_clock::now();

    std::vector<std::unordered_set<size_t> > visible_vertices_maps(images.size());

    model.vertex_colors_.resize(model.vertices_.size());
    for (size_t i = 0; i < images.size(); ++i) {
        const mvs::Image& image = images.at(i);
        // Coloring(model, image, visible_vertices_maps[i]);
        thread_pool->AddTask(Coloring, model, image, &visible_vertices_maps[i]);
    }
    thread_pool->Wait();

    for (size_t i = 0; i < images.size(); ++i) {
        const mvs::Image& image = images.at(i);
        const int width = image.GetWidth();
        const int height = image.GetHeight();

        const Eigen::RowMatrix3d K = Eigen::RowMatrix3f(image.GetK()).cast<double>();
        const Eigen::RowMatrix3d R = Eigen::RowMatrix3f(image.GetR()).cast<double>();
        const Eigen::Vector3d T = Eigen::Vector3f(image.GetT()).cast<double>();

        Bitmap bitmap;
        bitmap.Read(image.GetPath(), true);
        std::unordered_set<size_t>::iterator it = visible_vertices_maps[i].begin();
        for (; it != visible_vertices_maps[i].end(); ++it) {
            size_t vertex_idx = *it;
            Eigen::Vector3d& p = model.vertices_[vertex_idx];

            Eigen::Vector3d proj = K * (R * p + T);
            float fu = proj[0] / proj[2];
            float fv = proj[1] / proj[2];
            if (fu < 1 || fv < 0 || fu >= width - 1 || fv >= height - 1) {
                continue;
            }
            BitmapColor<float> val = bitmap.InterpolateBilinear(fu, fv);
            Eigen::Vector3d &color = model.vertex_colors_[vertex_idx];
            color[0] = (color[0] + val.r) * 0.5f;
            color[1] = (color[1] + val.g) * 0.5f;
            color[2] = (color[2] + val.b) * 0.5f;
        }
    }

    std::chrono::high_resolution_clock::time_point end_time = 
        std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                    end_time - start_time).count() / 1e6;
    std::cout << StringPrintf("Coloring: %lf [seconds]", duration) 
                << std::endl;

    std::string output_path = in_model_path;
        // JoinPaths(GetParentDir(in_model_path), TEX_MODEL_NAME);
    std::cout << output_path << std::endl;
    WriteTriangleMeshObj(output_path, model);

    #endif

    return 0;
}
