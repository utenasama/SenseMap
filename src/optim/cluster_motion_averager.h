//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_OPTIMIZER_CLUSTER_MOTION_AVERAGER_H_
#define SENSEMAP_OPTIMIZER_CLUSTER_MOTION_AVERAGER_H_

#include <vector>
#include "util/types.h"
#include "base/reconstruction.h"
#include "util/alignment.h"

namespace sensemap{

class ClusterMotionAverager{

public:

    //optimize the global motions of all the clusters based on an initial guess
    void ClusterMotionAverage(
     const std::vector<std::shared_ptr<Reconstruction> >& reconstructions,
     EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms,
     EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) &relative_transforms,
     std::vector<std::vector<cluster_t>>& clusters_ordered);


    //Estimate relative motions between each connected cluster pair, and 
    //initialize the global motions of all the clusters 
    void InitializeMotionAverage( 
           const std::vector<std::shared_ptr<Reconstruction> >& reconstructions,
           EIGEN_STL_UMAP(cluster_pair_t,Eigen::Matrix3x4d)& relative_trans,
           EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms,
           std::vector<std::vector<cluster_t>>& clusters_ordered);
    
    void SetGraphs(const CorrespondenceGraph* correspondence_graph);

    ClusterMotionAverager(const bool debug_info);

    ClusterMotionAverager(const bool debug_info, 
                        const bool save_strong_pairs, 
                        const std::string strong_pairs_path, 
                        const size_t candidate_strong_pairs_num, 
                        const bool load_strong_pairs);

    ~ClusterMotionAverager(){};
private:

    //build a max spanning tree of the cluster graph, in which vertices are 
    //composed of the clusters, and two clusters are connected by an edge if
    //their relative transformations have been estimated. The edge weight is the
    //reliability of the transform, here the number of inlier 3D point 
    //correspondences. 
    //Maybe not all the clusters are connected into a single component, 
    //so this function may generate several different trees. Each of 
	//them defines a connected component.
	
	//@param vertices  All the input vertices
	//@param weights     The edge weights of the graph
	//@param vertices_ordered  Each element of the vector is a tree. 
	//						   Only the edges between two consecutive ordered
	//                         vertices are kept.
	//@param edges_ordered     the reserved edges.      
    
	void BuildMaxSpanningTree(
                    const std::vector<cluster_t>& vetices,
                    const std::unordered_map<cluster_pair_t,size_t>& weights,
                    std::vector<std::vector<cluster_t>>& vertices_ordered,
                    std::vector<std::vector<cluster_pair_t>>& edges_ordered);
    
    //The initial global transforms are obtained using the edges on the max 
    //spanning tree, they are close to the optimals.  
    //So relative transforms which are inconsistent with the initial global  
    //should be filtered out to avoid contaminating the optimization.  
    void FilterRelativeTransforms(
           EIGEN_STL_UMAP(cluster_pair_t,Eigen::Matrix3x4d)& 
           relative_transforms,
           EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms);


    bool ScaleAverage(
     const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& 
     relative_transforms,
     const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
     const cluster_t constant_cluster,
     std::unordered_map<cluster_t, double>& global_scales);

    bool RotationAverage(
     const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& 
     relative_transforms,
     const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
     const cluster_t constant_cluster,
     EIGEN_STL_UMAP(cluster_t, Eigen::Vector3d)& global_rotations);

    bool TranslationAverage(
     const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& 
     relative_transforms,
     const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
     const std::unordered_map<cluster_t,double>& global_scales,
     const EIGEN_STL_UMAP(cluster_t,Eigen::Vector3d)& global_rotations,
     const cluster_t constant_cluster,
     EIGEN_STL_UMAP(cluster_t, Eigen::Vector3d)& global_translations);


	// Write the intemediate results for debuging
	// write the result of merging two clusters using the relative transforms
	void WriteTwoClusterMergeResult(
     const std::vector<std::shared_ptr<Reconstruction> >& reconstructions,
     const EIGEN_STL_UMAP(cluster_pair_t,Eigen::Matrix3x4d)& 
	 relative_transforms, bool after_filter = false);
    
	// Write the result of merging all the clusters using the initial global
	// transforms and after motion average
	void WriteMergeResult(
     const std::vector<std::shared_ptr<Reconstruction> >& reconstructions,
	 const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
	 const std::vector<std::vector<cluster_t>>& clusters_ordered,
     bool after_motion_average = false);  

    
    //const ViewGraph* view_graph_;
    const CorrespondenceGraph* full_correspondence_graph_;
    bool debug_info_ = false;

    bool save_strong_pairs_ = false;
    std::string strong_pairs_path_ = "";
    size_t candidate_strong_pairs_num_ = 0;
    bool load_strong_pairs_ = false;
};

}//namespace sensemap
#endif //SENSEMAP_OPTIMIZER_CLUSTER_MOTION_AVERAGER_H_