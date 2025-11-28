//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_GRAPH_SCENE_CLUSTERING_H_
#define SENSEMAP_GRAPH_SCENE_CLUSTERING_H_

#include <vector>
#include <memory>

#include "util/types.h"

namespace sensemap {
    
// Scene clustering approach using normalized cuts on the scene graph. The scene
// is hierarchically partitioned into overlapping clusters until a maximum
// number of images is in a leaf node.
class SceneClustering {
public:
    struct Options {
        // The branching factor of the hierarchical clustering.
        int branching = 2;

        // The number of overlapping images between child clusters.
        int image_overlap = 0;

        int image_dist_seq_overlap = 10;

        // The maximum number of images in a leaf node cluster, otherwise the
        // cluster is further partitioned using the given branching factor. Note
        // that a cluster leaf node will have at most `leaf_max_num_images +
        // overlap` images to satisfy the overlap constraint.
        int leaf_max_num_images = 400;

        // cluster partition parameter.
        double min_modularity_thres = 0.3;
        int min_modularity_count = 4000;
        int max_modularity_count = 8000;

        int community_image_overlap = 10;
        int community_transitivity = 1;

        void Print() const;
        bool Check() const;
    };

    struct Cluster {
        std::vector<image_t> image_ids;
        std::vector<Cluster> child_clusters;
    };

    SceneClustering(const Options& options);

    void Partition(const std::vector<std::pair<image_t, image_t>>& image_pairs,
                   const std::vector<int>& num_inliers);

    const Cluster* GetRootCluster() const;
    std::vector<const Cluster*> GetLeafClusters() const;

private:
    void PartitionCluster(const std::vector<std::pair<int, int>>& edges,
                          const std::vector<int>& weights,
                          Cluster* cluster);

    const Options options_;
    std::unique_ptr<Cluster> root_cluster_;
};
} // namespace sensemap

#endif