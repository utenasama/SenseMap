// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_MVS_MVS_CLUSTER_H_
#define SENSEMAP_MVS_MVS_CLUSTER_H_

#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <boost/filesystem/path.hpp>
#include <boost/tuple/tuple.hpp>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "base/common.h"
#include "controllers/patch_match_options.h"
#include "mvs/workspace.h"
#include "mvs/fusion.h"

#include "util/misc.h"
#include "util/types.h"
#include "util/threading.h"
#include "util/obj.h"
#include "util/ply.h"
#include "util/octree.h"
#include "util/roi_box.h"
#include "util/semantic_table.h"
#include "util/kmeans.h"
#include "util/proc.h"

// #define CLUSTER_INFO

namespace sensemap {
namespace mvs {

class MVSCluster : public Thread {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct Options {
        std::string image_type = "perspective";
        float max_ram = -1.0;
        float ram_eff_factor = 0.7;
        int cluster_step = 5;

        std::string gpu_index = "-1";
        float max_gpu_memory = 6.0f;
        float gpu_memory_factor = 0.7;

        bool geom_consistency = true;
        int num_iter_geom_consistency = 1;

        bool filter = true;
        bool geo_filter = false;

        size_t max_num_src_images = 8; // openmvs
        double min_triangulation_angle = 1.0f;
        bool refine_with_semantic = false;
        bool has_prior_depth = false;

        float fuse_common_persent = 0.06;
        float fuse_with_normal = true;
        int fuse_check_num_images = 50;

        bool map_update = false;

        PatchMatchOptions ConverToPM(){
            PatchMatchOptions pm_options;
            pm_options.image_type = image_type;
            pm_options.max_ram = max_ram;

            pm_options.gpu_index = gpu_index;
            pm_options.max_gpu_memory = max_gpu_memory;
            pm_options.gpu_memory_factor = gpu_memory_factor;

            pm_options.geom_consistency = geom_consistency;
            pm_options.num_iter_geom_consistency = num_iter_geom_consistency;
            pm_options.filter = filter;
            pm_options.geo_filter = geo_filter;
            pm_options.max_num_src_images = max_num_src_images; 
            pm_options.min_triangulation_angle = min_triangulation_angle;
            pm_options.refine_with_semantic = refine_with_semantic;
            pm_options.has_prior_depth = has_prior_depth;

            return pm_options;
        };
   
        StereoFusion::Options ConverToFusion() {
            StereoFusion::Options fusion_options;
            fusion_options.max_ram = max_ram;
            fusion_options.ram_eff_factor = ram_eff_factor;
            fusion_options.fuse_common_persent = fuse_common_persent;
            fusion_options.with_normal = fuse_with_normal;
            fusion_options.check_num_images = fuse_check_num_images;

            return fusion_options;
        };

        void Print() const{
            PrintHeading2("MVSCluster::Options");
            PrintOption(max_ram);
            PrintOption(ram_eff_factor);
            PrintOption(gpu_index);
            PrintOption(max_gpu_memory);
            PrintOption(gpu_memory_factor);

            PrintOption(geom_consistency);
            PrintOption(num_iter_geom_consistency);
            PrintOption(max_num_src_images);

            PrintOption(min_triangulation_angle);
            PrintOption(refine_with_semantic);
            PrintOption(has_prior_depth);
            PrintOption(fuse_common_persent);
            PrintOption(fuse_with_normal);
            PrintOption(fuse_check_num_images);

            PrintOption(map_update);
        };

    };


public:
    MVSCluster();
    MVSCluster(Options& options, const std::string& reconstruction_path);
    
    bool GetRoiBox(const PatchMatchOptions& options,
                                    const Model& model,
                                    const std::string& ori_box_path, 
                                    Box& roi_box);

    int PatchMatchImageCluster(const PatchMatchOptions& options,
                std::vector<std::vector<int>> &cluster_image_map,
                uint64_t& max_images_num, int num_thread,
                const std::vector<mvs::Image> &images, 
                const std::vector<std::vector<int> >& overlapping_images_,
                const std::unordered_set<int>& update_image_idxs = std::unordered_set<int>(),
                const bool has_mask = false);
    
    int FusionImageCluster( const StereoFusion::Options& options,
                 std::vector<std::vector<int>> &cluster_image_map,
                 std::vector<int>& common_image_ids,
                 uint64_t& max_images_num,
                 const std::vector<mvs::Image> &images,
                 const std::vector<std::vector<int> >& overlapping_images_,
                 const std::unordered_set<int>& update_image_idxs = std::unordered_set<int>());

    bool SaveImageClusterYaml(const std::string path, const int step,
                const std::vector<std::vector<int>> &cluster_image_map,
                const std::vector<int>& common_image_ids = std::vector<int>());
    bool ReadImageClusterYaml(const std::string path, 
                std::vector<std::vector<int>> &cluster_image_map, int & step);
    bool ReadImageClusterYaml(const std::string path, 
                std::vector<std::vector<int>> &cluster_image_map,
                std::vector<int>& common_image_ids, int & step);

protected:
    void Run() override;

    void GetGpuProp();
    int EstimatePatchMatchThreadsPerGPU();
    void ReadGpuIndices();

    int FusionImageCluster();

private:
    Options options_;
    std::unique_ptr<Workspace> workspace_;

    std::vector<int> threads_per_gpu_;
    std::vector<float> max_gpu_memory_array_;
    std::vector<int> max_gpu_cudacore_array_;
    std::vector<uint64_t > max_gpu_texture_layered_0_;
    std::vector<uint64_t > max_gpu_texture_layered_1_;
    std::vector<uint64_t > max_gpu_texture_layered_2_;
    const std::string reconstruction_path_;
    int num_out_reconstructions_;
};

}  // namespace mvs
}  // namespace sensemap
#endif //SENSEMAP_MVS_MVS_CLUSTER_H_