//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <set>
#include <unordered_map>

#include "util/logging.h"
#include "util/misc.h"
#include "graph/graph_cut.h"

#include "scene_clustering.h"

namespace sensemap {

void SceneClustering::Options::Print() const {
    PrintHeading2("SceneClusteringOptions");
    PrintOption(branching);
    PrintOption(image_overlap);
    PrintOption(image_dist_seq_overlap);
    PrintOption(leaf_max_num_images);
    PrintOption(min_modularity_thres);
    PrintOption(min_modularity_count);
    PrintOption(max_modularity_count);
    PrintOption(community_image_overlap);
    PrintOption(community_transitivity);
}

bool SceneClustering::Options::Check() const {
    CHECK_OPTION_GT(branching, 0);
    CHECK_OPTION_GE(image_overlap, 0);
    return true;
}

SceneClustering::SceneClustering(const Options& options) : options_(options) {
    CHECK(options_.Check());
}

void SceneClustering::Partition(
    const std::vector<std::pair<image_t, image_t> > &image_pairs,
    const std::vector<int> &num_inliers) {
    CHECK(!root_cluster_);
    CHECK_EQ(image_pairs.size(), num_inliers.size());

    std::set<image_t> image_ids;
    std::vector<std::pair<int, int> > edges;
    for (const auto & image_pair : image_pairs) {
        image_ids.insert(image_pair.first);
        image_ids.insert(image_pair.second);
        edges.emplace_back(image_pair.first, image_pair.second);
    }

    root_cluster_.reset(new Cluster());
    root_cluster_->image_ids.insert(
        root_cluster_->image_ids.end(), 
        image_ids.begin(), 
        image_ids.end());
    PartitionCluster(edges, num_inliers, root_cluster_.get());
}

void SceneClustering::PartitionCluster(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights,
    Cluster* cluster) {
    CHECK_EQ(edges.size(), weights.size());

    if (edges.size() == 0 ||
        cluster->image_ids.size() <= options_.leaf_max_num_images) {
        return ;
    }

    const std::unordered_map<int, int> labels =
        ComputeNormalizedMinGraphCut(edges, weights, options_.branching);

    // Assign the images to the clustered child clusters.
    cluster->child_clusters.resize(options_.branching);
    for (const auto image_id : cluster->image_ids) {
        if (labels.count(image_id)) {
            Cluster & child_cluster = 
                cluster->child_clusters.at(labels.at(image_id));
            child_cluster.image_ids.push_back(image_id);

        }
    }

    // Collect the edges based on whether they are inter or 
    // intra child clusters.
    typedef std::pair<int, int> PAIR;
    typedef std::pair<PAIR, int> PAIR2;
    typedef std::vector<std::vector<int> > VEC2;
    typedef std::vector<std::vector<PAIR> > VEC2_PAIR;
    typedef std::vector<std::vector<PAIR2> > VEC2_PAIR2;

    VEC2_PAIR child_edges(options_.branching);
    VEC2 child_weights(options_.branching);
    VEC2_PAIR2 overlapping_edges(options_.branching);

    for (size_t i = 0; i < edges.size(); ++i) {
        const int label1 = labels.at(edges[i].first);
        const int label2 = labels.at(edges[i].second);
        if (label1 == label2) {
            child_edges.at(label1).push_back(edges[i]);
            child_weights.at(label1).push_back(weights[i]);
        } else {
            overlapping_edges.at(label1).emplace_back(edges[i], weights[i]);
            overlapping_edges.at(label2).emplace_back(edges[i], weights[i]);
        }
    }

    // Recursively partition all the child clusters.
    for (int i = 0; i < options_.branching; ++i) {
        PartitionCluster(child_edges[i],
                         child_weights[i],
                         &cluster->child_clusters[i]);
    }

    if (options_.image_overlap > 0) {
        for (int i = 0; i < options_.branching; ++i) {
            // Sort the overlapping edges by the number of inlier matches, such
            // that we add overlapping images with many common observations.
            std::sort(overlapping_edges[i].begin(), overlapping_edges[i].end(),
                        [](const std::pair<std::pair<int, int>, int>& edge1,
                           const std::pair<std::pair<int, int>, int>& edge2) {
                        return edge1.second > edge2.second;
                        });

            // Select overlapping edges at random and add image to cluster.
            std::set<int> overlapping_image_ids;
            for (const auto& edge : overlapping_edges[i]) {
                if (labels.at(edge.first.first) == i) {
                    overlapping_image_ids.insert(edge.first.second);
                } else {
                    overlapping_image_ids.insert(edge.first.first);
                }
                if (overlapping_image_ids.size() >=
                    static_cast<size_t>(options_.image_overlap)) {
                    break;
                }
            }

            // Recursively append the overlapping images to cluster and its children.
            // TODO ?
            std::function<void(Cluster*)> InsertOverlappingImageIds =
                [&](Cluster* cluster) {
                    cluster->image_ids.insert(cluster->image_ids.end(),
                                            overlapping_image_ids.begin(),
                                            overlapping_image_ids.end());
                    for (auto& child_cluster : cluster->child_clusters) {
                        InsertOverlappingImageIds(&child_cluster);
                    }
                };

            InsertOverlappingImageIds(&cluster->child_clusters[i]);
        }
    }
}

const SceneClustering::Cluster* SceneClustering::GetRootCluster() const {
    return root_cluster_.get();
}

std::vector<const SceneClustering::Cluster*>
SceneClustering::GetLeafClusters() const {
    CHECK(root_cluster_);

    std::vector<const Cluster*> leaf_clusters;

    if (!root_cluster_) {
        return leaf_clusters;
    } else if (root_cluster_->child_clusters.empty()) {
        leaf_clusters.push_back(root_cluster_.get());
        return leaf_clusters;
    }

    std::vector<const Cluster*> non_leaf_clusters;
    non_leaf_clusters.push_back(root_cluster_.get());
    
    // level-order traversal
    while(!non_leaf_clusters.empty()) {
        const auto cluster = non_leaf_clusters.back();
        non_leaf_clusters.pop_back();

        for (const auto & child_cluster : cluster->child_clusters) {
            if (child_cluster.child_clusters.empty()) {
                leaf_clusters.push_back(&child_cluster);
            } else {
                non_leaf_clusters.push_back(&child_cluster);
            }
        }
    }

    return leaf_clusters;
}


} // namespace sensemap