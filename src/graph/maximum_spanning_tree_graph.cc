//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "maximum_spanning_tree_graph.h"

#include <base/pose.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <chrono>
#include <fstream>

#include "graph/connected_components.h"
#include "graph/minimum_spanning_tree.h"
#include "util/math.h"

namespace sensemap {

namespace {

typedef std::pair<struct CorrespondenceGraph::ImagePair, image_pair_t >
        HeapElement;

bool SortHeapElement(const HeapElement &h1,
                     const HeapElement &h2) {
    return h1.first.num_correspondences < h2.first.num_correspondences;
}

// Computes the orientation of the neighbor camera based on the orientation of
// the source camera and the relative rotation between the cameras.
Eigen::Vector3d ComputeOrientation(
        const Eigen::Vector3d& source_orientation,
        const struct CorrespondenceGraph::ImagePair& pair,
        const image_t source_view_id, const image_t neighbor_view_id) {
    Eigen::Vector3d relative_orientation;
    Eigen::Matrix3d source_rotation_mat, relative_rotation;
    ceres::AngleAxisToRotationMatrix(
            source_orientation.data(),
            ceres::ColumnMajorAdapter3x3(source_rotation_mat.data()));
    // ceres::QuaternionToAngleAxis(pair.qvec.data(), relative_orientation.data());
    ceres::QuaternionToAngleAxis(pair.two_view_geometry.qvec.data(), 
                                                             relative_orientation.data());
    ceres::AngleAxisToRotationMatrix(
            relative_orientation.data(),
            ceres::ColumnMajorAdapter3x3(relative_rotation.data()));

    const Eigen::Matrix3d neighbor_orientation =
            (source_view_id < neighbor_view_id)
            ? (relative_rotation * source_rotation_mat).eval()
            : (relative_rotation.transpose() * source_rotation_mat).eval();

    Eigen::Vector3d orientation;
    ceres::RotationMatrixToAngleAxis(
            ceres::ColumnMajorAdapter3x3(neighbor_orientation.data()),
            orientation.data());
    return orientation;
}

// Adds all the edges of view_id to the heap. Only edges that do not already
// have an orientation estimation are added.
void AddEdgesToHeap(
        const CorrespondenceGraph &view_graph,
        const std::unordered_map<image_t , Eigen::Vector3d> &orientations,
        const image_t image_id, std::vector<HeapElement> *heap) {
    const std::unordered_set<image_t> neighbor_ids =
            view_graph.ImageNeighbor(image_id);
    for (const image_t neighbor_id : neighbor_ids) {
        // Only add edges to the heap that contain a vertex that has not been seen.
        if (ContainsKey(orientations, neighbor_id)) {
            continue;
        }

        auto pair_id = sensemap::utility::ImagePairToPairId(image_id, neighbor_id);
        heap->emplace_back(view_graph.ImagePair(image_id, neighbor_id), pair_id);
        std::push_heap(heap->begin(), heap->end(), SortHeapElement);
    }
}

} // unnamed namespace

std::vector<image_t> OrientationsFromMaximumSpanningTree(
        const CorrespondenceGraph &view_graph,
        std::unordered_map<image_t, Eigen::Vector3d> *orientations){
    std::unordered_set<std::pair<image_t , image_t> > mst;
    return OrientationsFromMaximumSpanningTree(view_graph, orientations, mst);
}

std::vector<image_t> OrientationsFromMaximumSpanningTree(
        const CorrespondenceGraph &view_graph,
        std::unordered_map<image_t, Eigen::Vector3d> *orientations,
    std::unordered_set<std::pair<image_t , image_t> > &mst) {

    CHECK_NOTNULL(orientations);

    std::cout<<"Original graph pairs:          "
             <<view_graph.NumImagePairs()<<std::endl;

    // Compute the largest connected component of the input view graph since the
    // MST is only valid on a single connected component.
    std::unordered_set<image_t > largest_cc;
    GetLargestConnectedComponentIds(view_graph, &largest_cc);

    auto pairs = view_graph.ImagePairs();
    for(auto pair : pairs)
    {
            image_t image_id1;
            image_t image_id2;
            sensemap::utility::PairIdToImagePair(pair.first,
                                                 &image_id1, &image_id2);
//        std::cout<<image_id1<<"  "<<image_id2<<" "
//                 <<pair.second.num_correspondences<<std::endl;
    }

    std::cout<<"Largest connected graph mode num: "
             <<largest_cc.size()<<std::endl;


    CorrespondenceGraph largest_cc_subgraph;

    if(largest_cc.size() == view_graph.NumImages()){
        largest_cc_subgraph = view_graph;
    } else {
        ExtractSubgraph(view_graph, largest_cc, &largest_cc_subgraph);
     }


    // Compute maximum spanning tree.
    const auto& image_pairs = largest_cc_subgraph.ImagePairs();
    MinimumSpanningTree<image_t , int> mst_extractor;
    for (const auto& image_pair : image_pairs) {
            // Since we want the *maximum* spanning tree, we negate all of the edge
            // weights in the *minimum* spanning tree extractor.
            image_t image_id1;
            image_t image_id2;
            sensemap::utility::PairIdToImagePair(image_pair.first,
                                                 &image_id1, &image_id2);
            mst_extractor.AddEdge(
#ifdef TWO_VIEW_CONFIDENCE
                            image_id1, image_id2,  -image_pair.second.two_view_geometry.confidence);
#else
                            image_id1, image_id2,  -image_pair.second.num_correspondences);
#endif
    }

    if (!mst_extractor.Extract(&mst)) {
            VLOG(2)
            << "Could not extract the maximum spanning tree from the view graph";
            return {};
    }

    // Create an MST view graph.
    CorrespondenceGraph mst_view_graph;
    std::pair<image_t , image_t> best_pair;
#ifdef TWO_VIEW_CONFIDENCE
    double max_confidence = 0.0;
#else
    int max_num = 0;
#endif
    for (const auto &image_id : largest_cc) {
        mst_view_graph.AddImage(image_id, largest_cc_subgraph.Image(image_id));
    }

    for (const auto& pair : mst) {
#ifdef TWO_VIEW_CONFIDENCE
        auto confidence = largest_cc_subgraph.ImagePair(pair.first, pair.second).two_view_geometry.confidence;
        if (confidence > max_confidence) {
            max_confidence = confidence;
            best_pair = pair;
        }
#else
        auto num = largest_cc_subgraph.ImagePair(pair.first, pair.second).num_correspondences;
        if( num > max_num){
            max_num = num;
            best_pair = pair;
        }
#endif

//        mst_view_graph.AddImage(pair.first, largest_cc_subgraph.Image(pair.first));
//        mst_view_graph.AddImage(pair.second, largest_cc_subgraph.Image(pair.second));
        mst_view_graph.AddCorrespondences(
                pair.first,
                pair.second,
                largest_cc_subgraph.ImagePair(pair.first, pair.second));
    }
    mst_view_graph.CalculateImageNeighbors();
    std::cout<<"MST graph pairs:               "
             <<mst_view_graph.NumImagePairs()<<std::endl;

#ifdef TWO_VIEW_CONFIDENCE
    std::cout<<"Best pairs:               "
            <<best_pair.first<<" "<<best_pair.second<<" : "<<max_confidence<<std::endl;
#else
    std::cout<<"Best pairs:               "
            <<best_pair.first<<" "<<best_pair.second<<" : "<<max_num<<std::endl;
#endif
    pairs = mst_view_graph.ImagePairs();

    // Chain the relative rotations together to compute orientations.  We use a
    // heap to determine the next edges to add to the minimum spanning tree.
    std::vector<HeapElement> heap;

    // Set the root value.
    const image_t root_image_id = best_pair.first;
    (*orientations)[root_image_id] = Eigen::Vector3d::Zero();
    AddEdgesToHeap(mst_view_graph, *orientations, root_image_id, &heap);

        std::vector<image_t> tree_nodes;
        tree_nodes.reserve(mst_view_graph.NumImages());
        tree_nodes.push_back(root_image_id);

    while (!heap.empty()) {
        const HeapElement next_element = heap.front();
        // Remove the best edge.
        std::pop_heap(heap.begin(), heap.end(), SortHeapElement);
        heap.pop_back();

        // If the edge contains two vertices that have already been added then do
        // nothing.
        image_t image_id1;
        image_t image_id2;
        sensemap::utility::PairIdToImagePair(next_element.second,
                                             &image_id1, &image_id2);

        if (!ContainsKey(*orientations, image_id1)) {
            // Compute the orientation for the vertex.
            (*orientations)[image_id1] =
                    ComputeOrientation(FindOrDie(*orientations,image_id2),
                                       next_element.first, image_id2, image_id1);

        // Add all edges to the heap.
            AddEdgesToHeap(
                    mst_view_graph, *orientations, image_id1, &heap);

                     tree_nodes.push_back(image_id1);
        }

        else {
            // Compute the orientation for the vertex.
            (*orientations)[image_id2] =
                    ComputeOrientation(FindOrDie(*orientations,image_id1),
                                       next_element.first, image_id1, image_id2);

            // Add all edges to the heap.
            AddEdgesToHeap(
                    mst_view_graph, *orientations, image_id2, &heap);

                     tree_nodes.push_back(image_id2);
        }
    }
    return tree_nodes;
}


std::unordered_set<image_pair_t>  ViewGraphFilteringFromMaximumSpanningTree(
        std::unordered_set<std::pair<image_t , image_t> > &mst,
        CorrespondenceGraph &view_graph, const int max_iter,  const double threshold) {


    std::unordered_set<std::pair<image_t , image_t> > strong_pairs(mst);
    std::unordered_set<std::pair<image_t , image_t> > current_strong_pairs(mst);


    view_graph.CalculateImageNeighbors();

    auto image_pairs = view_graph.ImagePairs();
    int num_pairs = image_pairs.size();

    const double pi = 3.141592653;


    for(int iter = 0; iter < max_iter; ++iter){
        std::cout<<iter<<std::endl;
        if(strong_pairs.size() == num_pairs)
            break;
        std::unordered_set<std::pair<image_t , image_t> > new_strong_pairs(mst);
        for(auto& strong_pair : current_strong_pairs){
            auto node1 = strong_pair.first;
            auto node2 = strong_pair.second;
            auto neighbors_1 = view_graph.ImageNeighbor(node1);
            auto neighbors_2 = view_graph.ImageNeighbor(node2);

            for(auto & neighbor1 : neighbors_1){
                //std::cout<<node1<<" "<<node2<<" "<<neighbor1<<std::endl;

                if(strong_pairs.count(std::make_pair(node1, neighbor1)) == 0
                   && strong_pairs.count(std::make_pair(neighbor1, node1)) == 0){
                    continue;
                }

                if(view_graph.ExistImagePair(node2, neighbor1)
                   && strong_pairs.count(std::make_pair(node2, neighbor1)) == 0
                   && strong_pairs.count(std::make_pair(neighbor1, node2)) == 0 ){

                    std::vector<image_t> triplet_nodes{node1, node2, neighbor1};
                    std::sort(triplet_nodes.begin(), triplet_nodes.end());

                    auto id0 = triplet_nodes[0];
                    auto id1 = triplet_nodes[1];
                    auto id2 = triplet_nodes[2];

                    image_pair_t p01 = sensemap::utility::ImagePairToPairId(id0, id1);
                    auto qvec_01 = image_pairs[p01].two_view_geometry.qvec;
                    image_pair_t p02 = sensemap::utility::ImagePairToPairId(id0, id2);
                    auto qvec_02 = image_pairs[p02].two_view_geometry.qvec;
                    image_pair_t p12 =sensemap::utility::ImagePairToPairId(id1, id2);
                    auto qvec_12 = image_pairs[p12].two_view_geometry.qvec;


                    auto delta_qvec = ConcatenateQuaternions(
                            ConcatenateQuaternions(qvec_01, qvec_12),
                            InvertQuaternion(qvec_02));
                    auto delta_mat = QuaternionToRotationMatrix(delta_qvec);
                    auto theta_dis = acos((delta_mat.trace() - 1) * 0.5) / pi * 90.0;
                    //std::cout<<id1<<" "<<id2<<" "<<id0<<": "<<theta_dis<<std::endl;
                    if(theta_dis < threshold){
                        new_strong_pairs.emplace(node2, neighbor1);
                    }
                }
            }
            for(auto & neighbor2 : neighbors_2){
                //std::cout<<node1<<" "<<node2<<" "<<neighbor2<<std::endl;

                if(strong_pairs.count(std::make_pair(node2, neighbor2)) == 0
                   && strong_pairs.count(std::make_pair(neighbor2, node2)) == 0){
                    continue;
                }

                if(view_graph.ExistImagePair(node1, neighbor2)
                   && strong_pairs.count(std::make_pair(node1, neighbor2)) == 0
                   && strong_pairs.count(std::make_pair(neighbor2, node1)) == 0 ){

                    std::vector<image_t> triplet_nodes{node1, node2, neighbor2};
                    std::sort(triplet_nodes.begin(), triplet_nodes.end());

                    auto id0 = triplet_nodes[0];
                    auto id1 = triplet_nodes[1];
                    auto id2 = triplet_nodes[2];

                    image_pair_t p01 = sensemap::utility::ImagePairToPairId(id0, id1);
                    auto qvec_01 = image_pairs[p01].two_view_geometry.qvec;
                    image_pair_t p02 = sensemap::utility::ImagePairToPairId(id0, id2);
                    auto qvec_02 = image_pairs[p02].two_view_geometry.qvec;
                    image_pair_t p12 =sensemap::utility::ImagePairToPairId(id1, id2);
                    auto qvec_12 = image_pairs[p12].two_view_geometry.qvec;


                    auto delta_qvec = ConcatenateQuaternions(
                            ConcatenateQuaternions(qvec_01, qvec_12),
                            InvertQuaternion(qvec_02));
                    auto delta_mat = QuaternionToRotationMatrix(delta_qvec);
                    auto theta_dis = acos((delta_mat.trace() - 1) * 0.5) / pi * 90.0;
                    //std::cout<<id1<<" "<<id2<<" "<<id0<<": "<<theta_dis<<std::endl;
                    if(theta_dis < threshold){
                        new_strong_pairs.emplace(node1, neighbor2);
                    }
                }
            }
            strong_pairs.insert(new_strong_pairs.begin(), new_strong_pairs.end());
            current_strong_pairs = new_strong_pairs;
        }
    }

    std::unordered_set<image_pair_t> good_pairs;
    for(auto &pair : strong_pairs){
        auto good_pair = sensemap::utility::ImagePairToPairId(pair.first, pair.second);
        good_pairs.insert(good_pair);
    }

    return good_pairs;
}


void GetLargestConnectedComponentIds(const CorrespondenceGraph &view_graph,
                                     std::unordered_set<image_t> *largest_cc) {

    ConnectedComponents<image_t> cc_extractor;
    // Add all edges to the connected components extractor.
    auto image_pairs = view_graph.ImagePairs();
    for(auto image_pair : image_pairs){
        image_t image_id1;
        image_t image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first,
                                             &image_id1, &image_id2);
        cc_extractor.AddEdge(image_id1, image_id2);
    }
    // Extract all connected components.
    std::unordered_map<image_t, std::unordered_set<image_t> > connected_components;
    cc_extractor.Extract(&connected_components);
    // Search for the largest CC in the viewing graph.
    image_t largest_cc_id = kInvalidImageId;
    int largest_cc_size = 0;
    for (const auto& connected_component : connected_components) {
        if (connected_component.second.size() > largest_cc_size) {
            largest_cc_size = connected_component.second.size();
            largest_cc_id = connected_component.first;
        }
    }
    CHECK_NE(largest_cc_id, kInvalidImageId);
    // Swap the largest connected component to the output.
    std::swap(*largest_cc, connected_components[largest_cc_id]);
}

EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair)
ViewGraphFiltering(CorrespondenceGraph &view_graph,
                   EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) &image_pairs,
                   double ratio = 0.55){

    view_graph.CalculateImageNeighbors();
    MinimumSpanningTree<image_t, float> mst_extractor;

    for (auto image_pair : image_pairs) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);

#ifdef TWO_VIEW_CONFIDENCE
        mst_extractor.AddEdge(image_id1, image_id2, -image_pair.second.two_view_geometry.confidence);
#else
        mst_extractor.AddEdge(image_id1, image_id2, -image_pair.second.num_correspondences);
#endif
    }
    std::unordered_set<std::pair<image_t, image_t> > minimum_spanning_tree;
    mst_extractor.Extract(&minimum_spanning_tree);


    const double threshold = 5.0;

    EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) new_pairs;

    // std::cout<<image_pairs.size()<<std::endl;
    for(auto pair : image_pairs){
        image_t node1;
        image_t node2;
        sensemap::utility::PairIdToImagePair(pair.first,
                                             &node1, &node2);

        if(minimum_spanning_tree.find(std::make_pair(node1, node2)) != minimum_spanning_tree.end()){
            new_pairs.insert(pair);
            continue;
        }

        auto neighbors_1 = view_graph.ImageNeighbor(node1);

        int triplet_count = 0, valid_count = 0;
        for(auto & neighbor1 : neighbors_1){
            if(view_graph.ExistImagePair(node2, neighbor1)){

                std::vector<image_t> triplet_nodes{node1, node2, neighbor1};
                std::sort(triplet_nodes.begin(), triplet_nodes.end());

                auto id0 = triplet_nodes[0];
                auto id1 = triplet_nodes[1];
                auto id2 = triplet_nodes[2];

                image_pair_t p01 = sensemap::utility::ImagePairToPairId(id0, id1);
                if(image_pairs.find(p01) == image_pairs.end()) {
                    continue;
                }
                auto qvec_01 = image_pairs[p01].two_view_geometry.qvec;

                image_pair_t p02 = sensemap::utility::ImagePairToPairId(id0, id2);
                if(image_pairs.find(p02) == image_pairs.end()) {
                    continue;
                }
                auto qvec_02 = image_pairs[p02].two_view_geometry.qvec;

                image_pair_t p12 =sensemap::utility::ImagePairToPairId(id1, id2);
                if(image_pairs.find(p12) == image_pairs.end()) {
                    continue;
                }
                auto qvec_12 = image_pairs[p12].two_view_geometry.qvec;


                auto delta_qvec = ConcatenateQuaternions(
                    ConcatenateQuaternions(qvec_01, qvec_12),
                    InvertQuaternion(qvec_02));
                auto delta_mat = QuaternionToRotationMatrix(delta_qvec);
                Eigen::AngleAxisd angle_axis(delta_mat);
                double R_angle = angle_axis.angle();
                //std::cout<<id1<<" "<<id2<<" "<<id0<<": "<<RadToDeg(R_angle)<<std::endl;
                if(RadToDeg(R_angle) < threshold){
                    valid_count++;
                }
                triplet_count++;
            }
        }
        //std::cout<< valid_count<<" / "<<  triplet_count << std::endl;
        if(triplet_count * ratio < valid_count){
            new_pairs.insert(pair);
        }
    }


    return new_pairs;


}


void GetAllConnectedComponentIds(const CorrespondenceGraph& view_graph,
                                 std::unordered_map<image_t, std::unordered_set<image_t> >& cc) {

    ConnectedComponents<image_t> cc_extractor;
    // Add all edges to the connected components extractor.
    auto image_pairs = view_graph.ImagePairs();
    for(auto image_pair : image_pairs){
        image_t image_id1;
        image_t image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first,
                                             &image_id1, &image_id2);
        cc_extractor.AddEdge(image_id1, image_id2);
    }
    // Extract all connected components.
    cc.clear();
    cc_extractor.Extract(&cc);
}

void ExtractSubgraph(const CorrespondenceGraph &view_graph,
                     const std::unordered_set<image_t> &views_in_subgraph,
                     CorrespondenceGraph *subgraph) {
    CHECK_NOTNULL(subgraph);
        for (const auto &image_id : views_in_subgraph) {
            subgraph->AddImage(image_id, view_graph.Image(image_id));
        }

        const auto &image_pairs = view_graph.ImagePairs();
        for (auto image_pair : image_pairs) {
            image_t image_id1, image_id2;
            utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
            if (views_in_subgraph.count(image_id1) && views_in_subgraph.count(image_id2)) {
                subgraph->AddCorrespondences(image_id1, image_id2, view_graph.ImagePair(image_id1, image_id2));
            }
        }
//    auto image_neighbors = view_graph.ImageNeighbors();
//    // Iterate over each vertex and add all edges to that vertex that are part of
//    // the subgraph.
//    for (const auto& vertex : image_neighbors) {
//        // If the vertex is not contained in the subgraph, skip it.
//        if (!ContainsKey(views_in_subgraph, vertex.first)) {
//            continue;
//        }
//
//        // Iterate over all edges to the current vertex and find edges to add to the
//        // subgraph. An edge will be added to the subgraph only if both vertices are
//        // in the subgraph.
//        for (const image_t & second_vertex : vertex.second) {
//            // Skip this vertex (and thus, edge) if it is not in the subgraph. Also,
//            // only consider edges where vertex1 < vertex2 so as not to add redundant
//            // to the subgraph.
//            if (!ContainsKey(views_in_subgraph, second_vertex) ||
//                second_vertex < vertex.first) {
//                continue;
//            }
//
//            // Add the edge to the subgraph.
//            auto pair_id = sensemap::utility::ImagePairToPairId(
//                    vertex.first, second_vertex);
//            auto pairs = view_graph.ImagePairs();
//            const struct CorrespondenceGraph::ImagePair& pair =
//                    FindOrDie(pairs, pair_id);
//
//            //subgraph->AddImage(vertex.first, view_graph.Image(vertex.first));
//            //subgraph->AddImage(second_vertex, view_graph.Image(second_vertex));
//            subgraph->AddCorrespondences(vertex.first, second_vertex, pair);
//        }
//    }
}

} // namespace sensemap
