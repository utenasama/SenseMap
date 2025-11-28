//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <unordered_map>

#include "util/types.h"

#ifndef SENSEMAP_GRAPH_GRAPH_CUT_H_
#define SENSEMAP_GRAPH_GRAPH_CUT_H_

namespace sensemap {

// Compute the min-cut of a undirected graph using the Stoer Wagner algorithm.
void ComputeMinGraphCutStoerWagner(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights, int* cut_weight,
    std::vector<char>* cut_labels);

// Compute the normalized min-cut of an undirected graph using Graclus.
// Partitions the graph into clusters and returns the cluster labels per vertex.
std::unordered_map<int, int> ComputeNormalizedMinGraphCut(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights, 
    const int num_parts);

} // namespace sensemap

#endif