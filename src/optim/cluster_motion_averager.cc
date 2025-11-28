//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "cluster_motion_averager.h"
#include "estimators/reconstruction_aligner.h"
#include "util/image_pair.h"
#include "base/similarity_transform.h"
#include "global_motions/L1_scale_optimizer.h"
#include "global_motions/L1_translation_optimizer.h"
#include "global_motions/robust_rotation_optimizer.h"
#include "global_motions/utils.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

namespace sensemap{
ClusterMotionAverager::ClusterMotionAverager(const bool debug_info):
    debug_info_(debug_info){
}

ClusterMotionAverager::ClusterMotionAverager(const bool debug_info, 
                        const bool save_strong_pairs, 
                        const std::string strong_pairs_path, 
                        const size_t candidate_strong_pairs_num, 
                        const bool load_strong_pairs):
                        debug_info_(debug_info),
                        save_strong_pairs_(save_strong_pairs),
                        strong_pairs_path_(strong_pairs_path),
                        candidate_strong_pairs_num_(candidate_strong_pairs_num),
                        load_strong_pairs_(load_strong_pairs){
}

bool ClusterMotionAverager::ScaleAverage(
    const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& 
    relative_transforms,
    const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms,
    const cluster_t constant_cluster,
    std::unordered_map<cluster_t, double>& global_scales){
    
    std::cout<<"[Scale Average]*********************" <<std::endl; 

    //feed the relative scales to the optimizer
    std::map<ViewIdPair, double> relative_scales;
    for(const auto& relative_transform:relative_transforms){
        SimilarityTransform3 similarity_trans(relative_transform.second);
        double s_ij=similarity_trans.Scale();

        cluster_t cluster_i, cluster_j;
        utility::PairIdToImagePairOrdered(relative_transform.first,
                                          &cluster_i, &cluster_j);

        ViewIdPair cluster_id_pair=ViewIdPair(cluster_i,cluster_j);
        relative_scales.emplace(cluster_id_pair,s_ij);
    }

    //feed the initial global scales to the optimizer
    for(const auto& global_transform: global_transforms ){
        SimilarityTransform3 similarity_trans(global_transform.second);
        double s=similarity_trans.Scale();
        cluster_t cluster_id= global_transform.first;  
        global_scales.emplace(cluster_id,s);
    }


    for(const auto& relative_scale:relative_scales){
        cluster_t cluster1 = relative_scale.first.first;
        cluster_t cluster2 = relative_scale.first.second;
        std::cout<<"S_"<<cluster1<<"_"<<cluster2<<": "<<relative_scale.second
                 <<" ";
        std::cout<<"S_"<<cluster1<<": "<<global_scales.at(cluster1)<<" "
                 <<"S_"<<cluster2<<": "<<global_scales.at(cluster2)<<std::endl;   
    }
    
    
    L1ScaleOptimizer::Options scale_optimizer_options;
    L1ScaleOptimizer scale_optimizer(scale_optimizer_options);
    if(scale_optimizer.OptimizeScales(relative_scales, constant_cluster, 
                                      &global_scales)) {
        std::cout<<"[ScaleAverage] After optimzation"<<std::endl;
        for(const auto& global_scale:global_scales){
            std::cout<<"S_"<<global_scale.first<<" "<<global_scale.second
                     <<std::endl;

        }
        return true;
    }
    else{
        return false;
    }
}

bool ClusterMotionAverager::RotationAverage(
    const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& 
    relative_transforms,
    const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
    const cluster_t constant_cluster,
    EIGEN_STL_UMAP(cluster_t, Eigen::Vector3d)& global_rotations){
    
    std::cout<<"[Rotation Average]*********************" <<std::endl;    

    //feed relative rotations    
    EIGEN_STL_MAP(ViewIdPair, Eigen::Vector3d) relative_rotations;
    for(const auto& relative_transform:relative_transforms){
        SimilarityTransform3 similarity_trans(relative_transform.second);
        
        Eigen::Matrix3d r_ij = similarity_trans.Matrix().block<3,3>(0,0)/
                            similarity_trans.Scale();
        Eigen::Vector3d r_ij_vec = globalmotion::RotationMatrixToVector(r_ij);
        
        cluster_t cluster_i, cluster_j;
        utility::PairIdToImagePairOrdered(relative_transform.first,
                                          &cluster_i, &cluster_j);
        ViewIdPair cluster_id_pair=ViewIdPair(cluster_i,cluster_j);

        relative_rotations.emplace(cluster_id_pair,r_ij_vec);
    }

    //feed initial global rotations
    for(const auto& global_transform: global_transforms ){
        SimilarityTransform3 similarity_trans(global_transform.second);

        Eigen::Matrix3d r=similarity_trans.Matrix().block<3,3>(0,0)/
                          similarity_trans.Scale();
        Eigen::Vector3d r_vec = globalmotion::RotationMatrixToVector(r);
        
        cluster_t cluster_id= global_transform.first; 
        global_rotations.emplace(cluster_id,r_vec);
    }
    
    
    RobustRotationOptimizer::Options rotation_optimizer_options;
    RobustRotationOptimizer rotation_optimizer(rotation_optimizer_options);
    if(rotation_optimizer.OptimizeRotations(relative_rotations,
                                            constant_cluster,
                                            &global_rotations)){
        return true;                                        
    }
    else{
        return false;
    }
}

bool ClusterMotionAverager::TranslationAverage(
    const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& 
    relative_transforms,
    const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
    const std::unordered_map<cluster_t,double>& global_scales,
    const EIGEN_STL_UMAP(cluster_t,Eigen::Vector3d)& global_rotations,
    const cluster_t constant_cluster,
    EIGEN_STL_UMAP(cluster_t, Eigen::Vector3d)& global_translations){

    std::cout<<"[Translation Average]*********************" <<std::endl;
         
    //feed relative translations
    //not t_ij, but t_ij' = 1/s_j*R_j^T*t_ij is actually used in the optimzation
    //such that t_j'-t_i' = t_ij'
    
    EIGEN_STL_MAP(ViewIdPair, Eigen::Vector3d) relative_translations;

    for(const auto& relative_transform:relative_transforms){

        SimilarityTransform3 similarity_trans(relative_transform.second);
        Eigen::Vector3d t_ij=similarity_trans.Translation();
    
        cluster_t cluster_i, cluster_j;
        utility::PairIdToImagePairOrdered(relative_transform.first,
                                   &cluster_i, &cluster_j);
        ViewIdPair cluster_id_pair=ViewIdPair(cluster_i,cluster_j);
        
        double scale_j = global_scales.at(cluster_j);  
        Eigen::Vector3d rotation_j = global_rotations.at(cluster_j);
        Eigen::Matrix3d rotation_j_mat = 
                            globalmotion::VectorToRotationMatrix(rotation_j);    

        Eigen::Vector3d t_ij_transformed = 
                            rotation_j_mat.transpose()*t_ij/scale_j;

        relative_translations.emplace(cluster_id_pair,t_ij_transformed);
    }
    
    //feed initial global translations
    // t' = 1/s*R^T*t is actually used in the optimization
    for(const auto& global_transform: global_transforms ){
        
        SimilarityTransform3 similarity_trans(global_transform.second);
        
        Eigen::Vector3d t=similarity_trans.Translation();
        cluster_t cluster_id= global_transform.first;    

        double scale= global_scales.at(cluster_id);
        Eigen::Vector3d rotation = global_rotations.at(cluster_id);
        Eigen::Matrix3d rotation_mat = 
                            globalmotion::VectorToRotationMatrix(rotation);          
        Eigen::Vector3d t_transformed = rotation_mat.transpose()*t/scale;
        global_translations.emplace(cluster_id, t_transformed);
    }
    
    L1TranslationOptimizer::Options translation_optimizer_options; 
    L1TranslationOptimizer translation_optimizer(translation_optimizer_options);
    if(!translation_optimizer.OptimizeTranslations(relative_translations,
                                                   constant_cluster,
                                                   &global_translations)){
        return false;
    }
    else{
        //transform the estimated t' back to t via t = s*R*t'
        for(auto& global_translation: global_translations ){

            cluster_t cluster_id=global_translation.first;
            Eigen::Vector3d t_transformed=global_translation.second;

            double scale= global_scales.at(cluster_id);
            Eigen::Vector3d rotation = global_rotations.at(cluster_id);
            Eigen::Matrix3d rotation_mat = 
                            globalmotion::VectorToRotationMatrix(rotation);

            Eigen::Vector3d t = rotation_mat*t_transformed*scale;
            global_translation.second= t;
        }
        return true;
    }
}


void ClusterMotionAverager::ClusterMotionAverage(
    const std::vector<std::shared_ptr<Reconstruction> >& reconstructions,
    EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
    EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& relative_transforms,
    std::vector<std::vector<cluster_t>>& clusters_ordered){
   
    InitializeMotionAverage(reconstructions,relative_transforms, 
                            global_transforms,clusters_ordered);

    std::cout<<"[ClusterMotionAverager]: Connected components: "
             <<clusters_ordered.size()<<std::endl;


    for(auto& recon: reconstructions){
        CHECK_GE(recon->NumImages(),recon->NumRegisterImages());
    }
    
    FilterRelativeTransforms(relative_transforms, global_transforms);
    
    // if(debug_info_){
    //     std::cout<<"Write Initial merge result"<<std::endl;
    //     WriteMergeResult(reconstructions,global_transforms,clusters_ordered, false);
    // }
    

    // std::cout<<"Optimize transforms"<<std::endl;
    // for(size_t component_idx=0; component_idx< clusters_ordered.size(); 
    //     ++component_idx){

    //     if(clusters_ordered[component_idx].size()<2){
    //         continue;
    //     }

    //     std::unordered_map<view_t, double> global_scales;
    //     EIGEN_STL_UMAP(view_t, Eigen::Vector3d) global_rotations;
    //     EIGEN_STL_UMAP(view_t, Eigen::Vector3d) global_translations;

    //     std::unordered_set<cluster_t> clusters_in_this_component;
    //     EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms_in_this_component;

    //     for(auto cluster_id: clusters_ordered[component_idx]){
    //         clusters_in_this_component.insert(cluster_id);
    //         global_transforms_in_this_component.emplace(cluster_id, 
    //                                         global_transforms.at(cluster_id));
    //     }

    //     EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) 
    //                             relative_transforms_in_this_component;

    //     for(auto relative_transform: relative_transforms){
    //         cluster_pair_t pair_id = relative_transform.first;
    //         cluster_t cluster1, cluster2;
    //         utility::PairIdToImagePair(pair_id, &cluster1, &cluster2);
    //         if(!clusters_in_this_component.count(cluster1)||
    //            !clusters_in_this_component.count(cluster2)){
    //             continue;       
    //         }
    //         relative_transforms_in_this_component.emplace(pair_id,
    //                                                 relative_transform.second);
    //     }                
        
    //     if(!ScaleAverage(relative_transforms_in_this_component, 
    //                      global_transforms_in_this_component,
    //                      clusters_ordered[component_idx][0], 
    //                      global_scales)){
    //         continue;
    //     }
        
    //     if(!RotationAverage(relative_transforms_in_this_component, 
    //                        global_transforms_in_this_component,
    //                        clusters_ordered[component_idx][0], 
    //                        global_rotations)){
    //         continue;
    //     }
    //     if(!TranslationAverage(relative_transforms_in_this_component, 
    //                            global_transforms_in_this_component,
    //                            global_scales, 
    //                            global_rotations,
    //                            clusters_ordered[component_idx][0],
    //                            global_translations)){
    //         continue;                   
    //     }
        
    //     for (auto &global_scale : global_scales){
    //         cluster_t cluster_id = global_scale.first;

    //         Eigen::Vector3d r = global_rotations.at(cluster_id);
    //         Eigen::Vector3d t = global_translations.at(cluster_id);
    //         double s = global_scales.at(cluster_id);

    //         Eigen::Matrix3d r_mat = globalmotion::VectorToRotationMatrix(r);

    //         Eigen::Matrix3x4d similarity_trans_mat;
    //         similarity_trans_mat.block<3, 3>(0, 0) = r_mat * s;
    //         similarity_trans_mat.block<3, 1>(0, 3) = t;

    //         global_transforms[cluster_id] = similarity_trans_mat;
    //     }
    // }

    // if(debug_info_){
    //     std::cout<<"Write Motion Average merge result"<<std::endl;
    //     WriteMergeResult(reconstructions,global_transforms,clusters_ordered, true);
    // }
}


void ClusterMotionAverager::BuildMaxSpanningTree(
                    const std::vector<cluster_t>& vertices,
                    const std::unordered_map<cluster_pair_t,size_t>& weights,
                    std::vector<std::vector<cluster_t>>& vertices_ordered,
                    std::vector<std::vector<cluster_pair_t>>& edges_ordered){

    CHECK(vertices.size()!=0);
    std::unordered_set<cluster_t> remained_vertices;
    for(size_t i=1; i<vertices.size(); ++i){
        remained_vertices.insert(vertices[i]);
    }

    size_t connected_component_idx = 0; // -- As not all the cluster connected, we may create several connected components
    vertices_ordered.push_back(std::vector<cluster_t>());
    vertices_ordered[connected_component_idx].push_back(vertices[0]); // -- Use the first vertex as origin

    edges_ordered.push_back(std::vector<cluster_pair_t>());

    while(!remained_vertices.empty()){ 
        size_t max_weight = 0;
        size_t next_vertex;
        cluster_pair_t next_edge;
        bool is_connected=false;
        for(auto selected_vertex: vertices_ordered[connected_component_idx]){
            for(auto remained_vertex:remained_vertices){
                cluster_pair_t pair_id=
                   utility::ImagePairToPairId(selected_vertex,remained_vertex);

                if(weights.find(pair_id)!=weights.end()&&
                   weights.at(pair_id)>max_weight){
                    
                    max_weight=weights.at(pair_id);        
                    next_vertex=remained_vertex;
                    is_connected=true;
                    next_edge=pair_id;
                }
            }
        }
        if(is_connected){
            vertices_ordered[connected_component_idx].push_back(next_vertex);
            remained_vertices.erase(next_vertex);
            edges_ordered[connected_component_idx].push_back(next_edge);    
        }
        else{
            connected_component_idx++;
            vertices_ordered.push_back(std::vector<cluster_t>());
            vertices_ordered[connected_component_idx].push_back(*remained_vertices.begin());                                
            edges_ordered.push_back(std::vector<cluster_pair_t>());

            remained_vertices.erase(*remained_vertices.begin());
        }
    }

    // FIXME: Print the Max Spanning Tree Result
    std::cout <<"[InitializeMotionAverage] Max Spanning Tree Result" <<std::endl;
    size_t counter = 0;
    for(const auto current_edges : edges_ordered){
        std::cout << "Max Spanning Tree "<<  counter << "\n Edge:" <<std::endl; 
        for(auto current_edge : current_edges){
            image_t cluster_1, cluster_2;
            utility::PairIdToImagePair(current_edge, &cluster_1, &cluster_2);
            std::cout << "\t"<<cluster_1 << " - "<<  cluster_2 <<std::endl; 
        }
        std::cout << "Vertex: " << std::endl;
        for (auto current_vertex : vertices_ordered[counter]){
            std::cout << "\t" << current_vertex << std::endl;
        }
        counter++;
    }
    std::cout << std::endl;
    
}

void WriteStrongPairs(const std::string& path, 
                    const std::map<cluster_pair_t, std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>>>& whole_image_pairs, 
                    std::map<cluster_pair_t, size_t>& whole_max_num_image_pair){
    std::ofstream file(path, std::ios::trunc);
	CHECK(file.is_open()) << path;

    size_t whole_image_pairs_size = whole_image_pairs.size();
    std::cout << "write whole_image_pairs_size: " << whole_image_pairs_size << std::endl;
    file.write((char*)&whole_image_pairs_size, sizeof(size_t));
    bool chosed_pair = true;
    for(auto image_pairs : whole_image_pairs){
        size_t counter = 0;
        size_t max_num_image_pair = whole_max_num_image_pair[image_pairs.first];
        std::cout << "write image_pairs.first: " << image_pairs.first << std::endl;   
        file.write((char*)&image_pairs.first, sizeof(cluster_pair_t));

        size_t image_pairs_size = image_pairs.second.size();
        std::cout << "write image_pairs_size: " << image_pairs_size << std::endl; 
        file.write((char*)&image_pairs_size, sizeof(size_t));
        for (size_t i = 0; i < image_pairs_size; ++i) {
            std::cout << "write image_pair: " << image_pairs.second[i].first.first << " "
                      << image_pairs.second[i].first.second << " " << image_pairs.second[i].second << " " << chosed_pair
                      << std::endl;
            file.write((char*)&image_pairs.second[i].first.first, sizeof(image_t));
            file.write((char*)&image_pairs.second[i].first.second, sizeof(image_t));
            file.write((char*)&image_pairs.second[i].second, sizeof(point2D_t));
            if(counter < max_num_image_pair){
                chosed_pair = true;
                counter++;
            }else{
                chosed_pair = false;
            }
            file.write((char*)&chosed_pair, sizeof(bool));
        }
        
    }
}

void ReadStrongPairs(const std::string& path, 
                    std::map<cluster_pair_t, std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>>>& whole_image_pairs,
                    std::map<cluster_pair_t, size_t>& whole_max_num_image_pair){
    std::ifstream file(path, std::ios::binary);
	CHECK(file.is_open()) << path;

    size_t num_whole_image_pairs;
    file.read(reinterpret_cast<char*>(&num_whole_image_pairs), sizeof(size_t));
    std::cout << "read whole_image_pairs_size: " << num_whole_image_pairs << std::endl;
    for (size_t i = 0; i < num_whole_image_pairs; ++i) {
        cluster_pair_t cluster_pair_id;
        file.read(reinterpret_cast<char*>(&cluster_pair_id), sizeof(cluster_pair_t));
        std::cout << "read image_pairs.first: " << cluster_pair_id << std::endl;  
        size_t num_image_pairs;
        file.read(reinterpret_cast<char*>(&num_image_pairs), sizeof(size_t));
        std::cout << "read image_pairs_size: " << num_image_pairs << std::endl; 
        std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> image_pairs;
        int counter = 0;
        for (size_t j = 0; j < num_image_pairs; ++j) {
            std::pair<std::pair<image_t, image_t>, point2D_t> image_pair;
            file.read(reinterpret_cast<char*>(&image_pair.first.first), sizeof(image_t));
            file.read(reinterpret_cast<char*>(&image_pair.first.second), sizeof(image_t));
            file.read(reinterpret_cast<char*>(&image_pair.second), sizeof(point2D_t));
            bool chosed_pair;
            file.read(reinterpret_cast<char*>(&chosed_pair), sizeof(bool));
            std::cout << "read image_pair: " << image_pair.first.first << " "
                      << image_pair.first.second << " " << image_pair.second << " " << chosed_pair
                      << std::endl;
            if(chosed_pair){
                image_pairs.insert(image_pairs.begin()+counter, image_pair);
                counter++;
            } else {
                image_pairs.push_back(image_pair);
            }
        }
        whole_image_pairs[cluster_pair_id] = image_pairs;
        whole_max_num_image_pair[cluster_pair_id] = counter;
    }
}

void ClusterMotionAverager::InitializeMotionAverage(
        const std::vector<std::shared_ptr<Reconstruction> >& reconstructions,
        EIGEN_STL_UMAP(cluster_pair_t,Eigen::Matrix3x4d)& relative_trans,
        EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
        std::vector<std::vector<cluster_t>>& clusters_ordered){
    global_transforms.clear();
    // CHECK_NOTNULL(view_graph_);
    CHECK_NOTNULL(full_correspondence_graph_);

    //Estimate the relative transforms of all the possible cluster pairs
    ReconstructionAlignerOptions options;
    options.save_strong_pairs = save_strong_pairs_;
    options.candidate_strong_pairs_num = candidate_strong_pairs_num_;
    options.load_strong_pairs = load_strong_pairs_;
    ReconstructionAligner reconstruction_aligner(options);

    reconstruction_aligner.SetGraphs(full_correspondence_graph_);

    std::unordered_map<cluster_pair_t,size_t> num_corres_between_clusters;
    
    std::cout<<"[InitializeMotionAverage] reconstruction size: "
            <<reconstructions.size()<<std::endl<<std::endl;

    std::map<cluster_pair_t, std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>>> whole_image_pairs;
    std::map<cluster_pair_t, size_t> whole_max_num_image_pair;

    if(load_strong_pairs_){
        ReadStrongPairs(strong_pairs_path_, whole_image_pairs, whole_max_num_image_pair);
    }

    for (size_t i = 0; i < reconstructions.size(); ++i) {
        for(size_t j=i+1; j<reconstructions.size(); ++j){
            std::cout<<"[InitializeMotionAverage] align reconstruction "<<i
                     <<" to "<<j<<std::endl;
            //set the cluster with smaller id  as the src cluster and the
            //cluster with bigger id as the dst cluster
            Eigen::Matrix3x4d transform;

            std::vector<std::pair<std::pair<image_t, image_t>, point2D_t>> image_pairs;
            size_t max_num_image_pair = 0;
            cluster_pair_t pair_id=utility::ImagePairToPairId(i,j);// Convert cluster id to cluster pair id      
            if (load_strong_pairs_) {
                if(whole_image_pairs.count(pair_id)){                                                                                                                                       
                    image_pairs = whole_image_pairs[pair_id];
                    max_num_image_pair = whole_max_num_image_pair[pair_id];
                }
            }

            bool save_image_pairs = false;

            size_t num_corres=
                  reconstruction_aligner.RelativeTransformFrom3DCorrespondences(
                                            *(reconstructions[i].get()),
                                            *(reconstructions[j].get()),
                                            transform,
                                            image_pairs,
                                            max_num_image_pair,
                                            save_image_pairs, 0);
            std::cout << "save_image_pairs " << save_image_pairs<<" " <<image_pairs.size()<< std::endl;

            if(num_corres){// If the correspondence larger than 0, we will save the cluster pair id  with it transform                           
                relative_trans.emplace(pair_id,transform);
                num_corres_between_clusters.emplace(pair_id,num_corres);
                if(save_strong_pairs_){
                    whole_image_pairs[pair_id] = image_pairs;
                    whole_max_num_image_pair[pair_id] = max_num_image_pair;
                }
            }
            if(save_image_pairs && save_strong_pairs_){
                whole_image_pairs[pair_id] = image_pairs;
                whole_max_num_image_pair[pair_id] = max_num_image_pair;
            }
            std::cout<<"[InitializeMotionAverage] inlier point correspondences:"
                     <<num_corres<<std::endl
                     <<std::endl;
            std::cout << "transform " << std::endl << transform << std::endl;
        }
    }

    if(save_strong_pairs_){
        WriteStrongPairs(strong_pairs_path_, whole_image_pairs, whole_max_num_image_pair);
    }

    //Initialize the global motions. Firstly, generate a max spanning tree of 
    //the graph, then concatenate the relative transforms on the edges of this
    //tree. 
    
    //generate the max spanning tree
    std::vector<cluster_t> clusters(reconstructions.size()); // -- Vertex in order 
    for(size_t i=0; i<clusters.size(); ++i){
        clusters[i]=i;
    }
    std::vector<std::vector<cluster_pair_t>> edges_ordered; // -- Edge

    BuildMaxSpanningTree(clusters, num_corres_between_clusters, // -- Use corresondence to construct max spanning tree
                         clusters_ordered, edges_ordered);

    //compute the initial transformations for each connected component
    for(size_t connected_component_idx = 0; 
        connected_component_idx < clusters_ordered.size(); 
        ++connected_component_idx){
        std::vector<cluster_t> clusters_ordered_in_this_component =        
                                    clusters_ordered[connected_component_idx];
        std::vector<cluster_pair_t> edges_ordered_in_this_component =
                                    edges_ordered[connected_component_idx];

        // Convert the relative transform to global transform
        Eigen::Matrix4d matrixI = Eigen::MatrixXd::Identity(4, 4);
        global_transforms.emplace(clusters_ordered_in_this_component[0], // -- The first cluster has an identity global transform
                                  matrixI.topLeftCorner<3, 4>());
        std::cout << "clusters_ordered_in_this_component[0] " <<clusters_ordered_in_this_component[0]<< std::endl;

        CHECK(edges_ordered_in_this_component.size() ==
              clusters_ordered_in_this_component.size() - 1); // -- Verifiy the MST graph 

        for (size_t i = 1; i < clusters_ordered_in_this_component.size(); ++i){

            cluster_pair_t pair_id = edges_ordered_in_this_component[i - 1];
            CHECK(relative_trans.find(pair_id) != relative_trans.end());

            SimilarityTransform3 trans_pre_cur(relative_trans.at(pair_id));// Convert the relative transform to sim3

            cluster_t cluster1, cluster2;
            utility::PairIdToImagePair(pair_id, &cluster1, &cluster2);// Convert the cluster pair id to cluster id

            std::cout << "clusters_ordered_in_this_component[i] " <<clusters_ordered_in_this_component[i]<< std::endl;
            std::cout << "cluster1 " <<cluster1<< " cluster2 " <<cluster2<< std::endl;
            std::cout << "trans_pre_cur " << std::endl << relative_trans.at(pair_id) << std::endl;

            cluster_t current_cluster = clusters_ordered_in_this_component[i];
            cluster_t pre_cluster;
            CHECK(cluster1 == current_cluster || cluster2 == current_cluster);

            if (cluster1 == current_cluster){
                pre_cluster = cluster2;
            }
            else{
                pre_cluster = cluster1;
            }
            CHECK(global_transforms.find(pre_cluster) != 
                  global_transforms.end()); // Check the pre_cluster has global transform or not
            
            SimilarityTransform3 trans_pre(global_transforms.at(pre_cluster));

            Eigen::Matrix4d trans_current;

            //the relative transformation is always from the cluster with 
            //smaller id to that with larger id.
            if (current_cluster > pre_cluster){
                trans_current = trans_pre_cur.Matrix() * trans_pre.Matrix();
            }
            else{
                trans_current = trans_pre_cur.Inverse().Matrix() * 
                                trans_pre.Matrix();
            }
            global_transforms.emplace(current_cluster,
                                      trans_current.topLeftCorner<3, 4>());
            std::cout << "current_cluster " << current_cluster << std::endl
                      << global_transforms[current_cluster] << std::endl;
        }
    }
}


void ClusterMotionAverager::FilterRelativeTransforms(
        EIGEN_STL_UMAP(cluster_pair_t,Eigen::Matrix3x4d)& 
        relative_transforms,
        EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms){

    EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) 
            relative_transforms_copy = relative_transforms;

    for(const auto& relative_transform:relative_transforms_copy){
        SimilarityTransform3 similarity_ij(relative_transform.second);
        
        Eigen::Matrix3d r_ij = similarity_ij.Matrix().block<3,3>(0,0)/
                            similarity_ij.Scale();
        Eigen::Vector3d r_ij_vec = globalmotion::RotationMatrixToVector(r_ij);
        
        cluster_t cluster_i, cluster_j;
        utility::PairIdToImagePairOrdered(relative_transform.first,
                                          &cluster_i, &cluster_j);

        SimilarityTransform3 similarity_i(global_transforms.at(cluster_i));
        Eigen::Matrix3d r_i = similarity_i.Matrix().block<3,3>(0,0)/
                              similarity_i.Scale();
        Eigen::Vector3d r_i_vec = globalmotion::RotationMatrixToVector(r_i);

        SimilarityTransform3 similarity_j(global_transforms.at(cluster_j));
        Eigen::Matrix3d r_j = similarity_j.Matrix().block<3,3>(0,0)/
                              similarity_j.Scale();
        Eigen::Vector3d r_j_vec = globalmotion::RotationMatrixToVector(r_j);

        //Check the consistency, consider the both the rotations and 
        //translations

        //The rotation error
        Eigen::Vector3d rotation_error = 
        globalmotion::MultiplyRotations(-r_j_vec,
                        globalmotion::MultiplyRotations(r_ij_vec, 
                                                        r_i_vec));
        
        // the translation error
        double scale_j = similarity_j.Scale();
        Eigen::Vector3d t_ij=similarity_ij.Translation();
        Eigen::Vector3d t_ij_transformed = r_j.transpose()*t_ij/scale_j;

        double scale_i = similarity_i.Scale();
        Eigen::Vector3d t_i = similarity_i.Translation();
        Eigen::Vector3d t_i_transformed = r_i.transpose()*t_i/scale_i;

        Eigen::Vector3d t_j = similarity_j.Translation();   
        Eigen::Vector3d t_j_transformed = r_j.transpose()*t_j/scale_j;

        Eigen::Vector3d translation_error = t_j_transformed - t_i_transformed - 
                                            t_ij_transformed;
        Eigen::Vector3d d_ij= t_j_transformed - t_i_transformed;

        // remove the relative transforms which are largely inconsistent with
        // the inital global transforms
        if(rotation_error.norm()/M_PI*180>60||(d_ij.norm()>0&&
           translation_error.norm()/(d_ij.norm())>0.5)){
            relative_transforms.erase(relative_transform.first);
        }
    }           
}

void ClusterMotionAverager::WriteTwoClusterMergeResult(
     const std::vector<std::shared_ptr<Reconstruction> >& reconstructions,
     const EIGEN_STL_UMAP(cluster_pair_t,Eigen::Matrix3x4d)&  
     relative_transforms, bool after_filter){

     for(auto relative_transform: relative_transforms){

        cluster_t cluster_i, cluster_j;
        utility::PairIdToImagePairOrdered(relative_transform.first,
                                   &cluster_i, &cluster_j);
        std::shared_ptr<Reconstruction> merged_recons =
                                    std::make_shared<Reconstruction>();

        *(merged_recons.get()) = *(reconstructions[cluster_i].get());

        merged_recons->Merge(*(reconstructions[cluster_j].get()),
                            relative_transform.second,
                            8.0);
        if(after_filter){
            std::string rec_path = StringPrintf("./points%04d_%04d",
                                                cluster_i, 
                                                cluster_j);
            if (boost::filesystem::exists(rec_path)) {
                boost::filesystem::remove_all(rec_path);
            }
            boost::filesystem::create_directories(rec_path);
            merged_recons->WriteReconstruction(rec_path, true);
        }
        else{
            
            std::string rec_path = StringPrintf("./org_points%04d_%04d",
                                                cluster_i, 
                                                cluster_j);
            if (boost::filesystem::exists(rec_path)) {
                boost::filesystem::remove_all(rec_path);
            }
            boost::filesystem::create_directories(rec_path);
            merged_recons->WriteReconstruction(rec_path, true);
        }
    }
}


void ClusterMotionAverager::WriteMergeResult(
     const std::vector<std::shared_ptr<Reconstruction> >& reconstructions,
     const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
     const std::vector<std::vector<cluster_t>>& clusters_ordered,
     bool after_motion_average){

    for(size_t component_idx = 0; component_idx< clusters_ordered.size();
        ++component_idx){

        std::shared_ptr<Reconstruction> merged_recons_initial =
            std::make_shared<Reconstruction>();
        CHECK(clusters_ordered[component_idx][0]>=0&&
              clusters_ordered[component_idx][0]<reconstructions.size());

        *(merged_recons_initial.get()) =
            *(reconstructions[clusters_ordered[component_idx][0]].get());

        for (size_t i = 1; i < clusters_ordered[component_idx].size(); ++i){
            CHECK(clusters_ordered[component_idx][i]>=0&&
                  clusters_ordered[component_idx][i]<reconstructions.size());

            merged_recons_initial->Merge(
                    *reconstructions[clusters_ordered[component_idx][i]],
                    global_transforms.at(clusters_ordered[component_idx][i]),
                    8.0);
        }
        std::string rec_path;
        if(after_motion_average){
            rec_path = StringPrintf("./motion_average_merged_%d", component_idx);
        }else{
            rec_path = StringPrintf("./initial_merged_%d", component_idx);
        }
        if (boost::filesystem::exists(rec_path)){
            boost::filesystem::remove_all(rec_path);
        }
        boost::filesystem::create_directories(rec_path);
        merged_recons_initial->WriteReconstruction(rec_path, true);
    }
}


void ClusterMotionAverager::SetGraphs( 
                    const CorrespondenceGraph* correspondence_graph){
    full_correspondence_graph_=correspondence_graph;
}

}//namespace sensempa
