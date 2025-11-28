//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_GRAPH_MINIMUM_SPANNING_TREE_H_
#define SENSEMAP_GRAPH_MINIMUM_SPANNING_TREE_H_

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "graph/connected_components.h"

namespace sensemap {

// A class for extracting the minimum spanning tree of a graph using Kruskal's
// greedy algorithm. The minimum spanning tree is a subgraph that contains all
// nodes in the graph and only the edges that connect these nodes with a minimum
// edge weight summation. The algorithm runs in O(E * log (V) ) where E is the
// number of edges and V is the number of nodes in the graph. For more details
// on the algorithm please see:
//   https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
template <typename T, typename V>
class MinimumSpanningTree {
public:
    MinimumSpanningTree() {}

    // Add an edge in the graph.
    void AddEdge(const T& node1, const T& node2, const V& weight) {
        edges_.emplace_back(weight, std::pair<T, T>(node1, node2));
    }

    bool SetWeight(const T& node1, const T& node2, const V& weight){
        bool success = false;
        for(auto &edge : edges_){
            if(edge.second == std::pair<T, T>(node1, node2) || edge.second == std::pair<T, T>(node2, node1)){
                edge.first = weight;
                success = true;
                break;
            }
        }
        return success;
    }

    // Extracts the minimum spanning tree. Returns true on success and false upon
    // failure. If true is returned, the output variable contains the edge list of
    // the minimum spanning tree.
    bool Extract(std::unordered_set<std::pair<T, T> >* minimum_spanning_tree) {
        if (edges_.size() == 0) {
            VLOG(2) << "No edges were passed to the minimum spanning tree extractor!";
            return false;
        }

        // Determine the number of nodes in the graph.
        const int num_nodes =  CountNodesInGraph() ;

        // Reserve space in the MST since we know it will have exactly N - 1 edges.
        minimum_spanning_tree->reserve(num_nodes - 1);

        // Order all edges by their weights.
        std::sort(edges_.begin(), edges_.end());

        // For each edge in the graph, add it to the minimum spanning tree if it
        // does not create a cycle.
        ConnectedComponents<T> cc;
        for (int i = 0;
             i < edges_.size() && minimum_spanning_tree->size() < num_nodes - 1;
             i++) {
            const auto& edge = edges_[i];
            if (!cc.NodesInSameConnectedComponent(edge.second.first,
                                                  edge.second.second)) {
                cc.AddEdge(edge.second.first, edge.second.second);
                minimum_spanning_tree->emplace(edge.second.first, edge.second.second);
            }
        }

        return minimum_spanning_tree->size() == num_nodes - 1;
    }

    bool ExtractLocalNodes(std::set<T>* local_spanning_tree_node, int num) {
        if (edges_.size() == 0) {
            VLOG(2) << "No edges were passed to the minimum spanning tree extractor!";
            return false;
        }

        if(num < 2){
            return false;
        }

        // Order all edges by their weights.
        std::sort(edges_.begin(), edges_.end());

        // For each edge in the graph, add it to the minimum spanning tree if it
        // does not create a cycle.

        local_spanning_tree_node->insert(edges_[0].second.first);
        local_spanning_tree_node->insert(edges_[0].second.second);

        while( local_spanning_tree_node->size() < num ) {
            bool is_found = false;
            for(auto i = 1; i < edges_.size(); ++i) {
                const auto &edge = edges_[i];
                bool has_first = !local_spanning_tree_node->count(edge.second.first) == 0;
                bool has_second = !local_spanning_tree_node->count(edge.second.second) == 0;
                if ( has_first && !has_second) {
                    local_spanning_tree_node->insert(edge.second.second);
                    is_found = true;
                    break;
                } else if ( !has_first && has_second) {
                    local_spanning_tree_node->insert(edge.second.first);
                    is_found = true;
                    break;
                }
            }
            if(!is_found){
                break;
            }
        }

        return local_spanning_tree_node->size() == num ;
    }


private:
    // Counts the number of nodes in the graph by counting the number of unique
    // node values we have received from AddEdge.
    int CountNodesInGraph() {
        std::vector<T> nodes;
        nodes.reserve(edges_.size() * 2);
        for (const auto& edge : edges_) {
            nodes.emplace_back(edge.second.first);
            nodes.emplace_back(edge.second.second);
        }
        std::sort(nodes.begin(), nodes.end());
        auto unique_end = std::unique(nodes.begin(), nodes.end());
        return std::distance(nodes.begin(), unique_end);
    }

    std::vector<std::pair<V, std::pair<T, T> > > edges_;

    // Each node is mapped to a Root node. If the node is equal to the root id
    // then the node is a root and the size of the root is the size of the
    // connected component.
    std::unordered_map<T, T> disjoint_set_;

};

} // namespace sensemap

#endif //SENSEMAP_GRAPH_MINIMUM_SPANNING_TREE_H_
