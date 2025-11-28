//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_OPTIM_BLOCK_BUNDLE_ADJUSTMENT_H_
#define SENSEMAP_OPTIM_BLOCK_BUNDLE_ADJUSTMENT_H_

#include <memory>
#include <unordered_set>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include "util/alignment.h"
#include "optim/bundle_adjustment.h"
#include "pose_graph_optimizer.h"

namespace sensemap {

class Reconstruction;
class SimilarityTransform3;
struct BundleAdjustmentOptions;
struct BundleAdjustmentConfig;

struct BlockBundleAdjustmentOptions {
    size_t max_block_images = -1;
    size_t block_images_num = -1;
    size_t common_images_num = -1;
    size_t link_edges = 3;
    size_t maximum_threads_num = -1;
    size_t min_connected_points_for_common_images = 100;
    BlockBundleAdjustmentOptions() {

    }

    BlockBundleAdjustmentOptions(
        const size_t in_total_images,
        const size_t in_images_per_block,
        const size_t in_common_images_num,
        const size_t in_min_connected_points_for_common_images) 
    {
        max_block_images = in_images_per_block;
        common_images_num = in_common_images_num;
        min_connected_points_for_common_images = in_min_connected_points_for_common_images;
        if (max_block_images <= 0 || common_images_num <= 0 || common_images_num >= max_block_images) {
            block_images_num = -1;
        }
        else {
            size_t target_block_num = (in_total_images - 1) / max_block_images + 1;
            size_t target_images_thres = in_total_images / target_block_num;
            size_t remain_num = in_total_images % target_images_thres;

            block_images_num = target_images_thres + remain_num / target_block_num + 1;
        }
    }
};

struct Block {
    block_t id;
    BundleAdjustmentConfig config;
    std::shared_ptr<Reconstruction> reconstruction;
    std::unordered_set<image_t> common_images;
    std::unordered_set<mappoint_t> mappoints;
    std::unordered_set<mappoint_t> constant_mappoints;
    
    Eigen::Vector3d image_sum_location = Eigen::Vector3d::Zero();
    Block() {};
    Block(const block_t block_id) {
        id = block_id;
        reconstruction = std::make_shared<Reconstruction>();
    }
};

struct DisjointSet {
    std::vector<size_t> roots_;

    DisjointSet(size_t size) {
        roots_.resize(size);
        for (int i = 0; i < roots_.size(); ++ i) roots_[i] = i;
    }

    size_t Find(size_t x) {
        if (roots_[x] == x) return roots_[x];
        return roots_[x] = Find(roots_[x]);
    }

    void Merge(size_t x, size_t y) {
        int fx = Find(x);
        int fy = Find(y);
        roots_[fy] = fx;
    }

    std::vector<size_t> MaxComponent() {
        std::unordered_map<size_t, size_t> component_mapper;

	    for (int i = 0; i < roots_.size(); ++ i) Find(i);

        size_t maximum_group_id = -1;
        size_t maximum_group_num = 0;
        for (int i = 0; i < roots_.size(); ++ i) {
            size_t group_id = roots_[i];
            int cnt = 0;
	        if (component_mapper.count(group_id)) cnt = component_mapper[group_id];

            ++ cnt;
            component_mapper[group_id] = cnt;

            if (cnt > maximum_group_num) {
                maximum_group_num = cnt;
                maximum_group_id = group_id;
            }
        }

        std::vector<size_t> res;
        for (int i = 0; i < roots_.size(); ++ i) {
            if (roots_[i] == maximum_group_id) {
                res.emplace_back(i);
            }
        }

        return res;
    }
};

class BlockBundleAdjuster {
public:
    BlockBundleAdjuster(
        const BundleAdjustmentOptions& options,
        const BundleAdjustmentConfig& config);

    bool Solve(Reconstruction* reconstruction);

    void SetGlobalBACount(int count) {global_ba_count = count;}
private:
    // block GBA: divide blocks with kmeans
    std::unordered_map<block_t, std::shared_ptr<Block>>
    DivideBlocks(Reconstruction *reconstruction, const BlockBundleAdjustmentOptions &option) const;
    
    // block GBA: divide blocks with kmeans
    std::unordered_map<block_t, std::shared_ptr<Block>>
    DivideBlocks2(Reconstruction *reconstruction, const BlockBundleAdjustmentOptions &option) const;

    // pose graph
    void PoseGraph(Reconstruction *reconstruction, 
                std::unordered_map<block_t, std::shared_ptr<Block>> &blocks,
                const BlockBundleAdjustmentOptions &option);

    // block sim3 posegraph
    void BlockPoseGraph(Reconstruction *reconstruction, const std::unordered_map<block_t, std::shared_ptr<Block>> &blocks);

    size_t CalculateThreadsNum(const size_t blocks_count, const size_t threads_thres) const;
    
    size_t Triangulate(Reconstruction* reonstruction, const mappoint_t &mappoint_id);

    std::shared_ptr<SimilarityTransform3> EstimateAffine(
        const Reconstruction *reconstruction,
        const std::shared_ptr<Block> src_block, 
        const std::shared_ptr<Block> dst_block);
private:
    BundleAdjustmentOptions options_;
    BundleAdjustmentConfig config_;
    
    BlockBundleAdjustmentOptions block_options_;

    int global_ba_count = 0;
};

}

#endif
