//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_GRAPH_MAXIMUM_SPANNING_TREE_GRAPH_H_
#define SENSEMAP_GRAPH_MAXIMUM_SPANNING_TREE_GRAPH_H_

#include <Eigen/Eigen>
#include <unordered_map>

#include "correspondence_graph.h"

// #define TWO_VIEW_CONFIDENCE
namespace sensemap {


// Returns the views ids participating in the largest connected component in
// the view graph.
void GetLargestConnectedComponentIds(const CorrespondenceGraph& view_graph,
                                     std::unordered_set<image_t>* largest_cc);

void GetAllConnectedComponentIds(const CorrespondenceGraph& view_graph,
                                 std::unordered_map<image_t, std::unordered_set<image_t> >& cc);

// Computes orientations of each view in the view graph by computing the maximum
// spanning tree (by edge weight) and solving for the global orientations by
// chaining rotations. Orientations are estimated for only the largest connected
// component of the viewing graph.
std::vector<image_t> OrientationsFromMaximumSpanningTree(
        const CorrespondenceGraph& view_graph,
        std::unordered_map<image_t , Eigen::Vector3d>* orientations);

std::vector<image_t> OrientationsFromMaximumSpanningTree(
        const CorrespondenceGraph &view_graph,
        std::unordered_map<image_t, Eigen::Vector3d> *orientations,
        std::unordered_set<std::pair<image_t , image_t> > &mst);

std::unordered_set<image_pair_t>  ViewGraphFilteringFromMaximumSpanningTree(
        std::unordered_set<std::pair<image_t , image_t> > &mst,
        CorrespondenceGraph &view_graph, const int max_iter = 3,  const double threshold = 5.0);

EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair)
ViewGraphFiltering(
    CorrespondenceGraph &view_graph,
    EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) &image_pairs,
    double ratio);

// Extract a subgraph from this view graph which contains only the input
// views. Note that this means that only edges between the input views will be
// preserved in the subgraph.
void ExtractSubgraph(const CorrespondenceGraph& view_graph,
                     const std::unordered_set<image_t>& views_in_subgraph,
                     CorrespondenceGraph* subgraph);

} // namespace sensemap

#endif //SENSEMAP_GRAPH_MAXIMUM_SPANNING_TREE_GRAPH_H_
