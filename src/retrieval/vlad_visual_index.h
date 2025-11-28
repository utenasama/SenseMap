//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_SRC_RETRIEVAL_VLAD_VISUAL_INDEX_H_
#define SENSEMAP_SRC_RETRIEVAL_VLAD_VISUAL_INDEX_H_

#include "VLFeat/kmeans.h"
#include "VLFeat/vlad.h"
#include "util/alignment.h"
#include <vector>
#include <map>
#include "utils.h"

namespace sensemap {

class VladVisualIndex{

public:
    
    VladVisualIndex(){
        kmeans = NULL;
    }
    ~VladVisualIndex(){
        if(kmeans!=NULL){vl_kmeans_delete(kmeans);}
    }
    struct CodeBookCreateOptions {
        // The number of vocabularies in the code book
        int num_vocabulary = 256;
        int feature_dim = 128;
    };

    struct QueryOptions{
        int max_num_images = 256;
    };


    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Descriptors;
    typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VLAD;

    // create code book from a set of training features
    void CreateCodeBook(const CodeBookCreateOptions& option, const Descriptors& training_features);

    // save code book
    void SaveCodeBook(const std::string code_book_path);

    // load code book
    void LoadCodeBook(const std::string code_book_path);

    // extract the vlad for an image and add it to the database  
    void ADD(const Descriptors& training_features, const int image_id);

    void LocalFeaturesToVlad(const Descriptors& training_features, VLAD& vlad);
    void ADD(const VLAD& vlad, const int image_id);
    
    // get vlad for a specific image
    //const VLAD & VladForImage(const int image_id){return vlad_vectors.at(image_id);};


    // find nearest neigbors for  the query image

    void Query(const QueryOptions& option, const VLAD& query_vlad, std::vector<retrieval::ImageScore>* image_scores);
    
private:

    VlKMeans * kmeans;

    //std::map<int, VLAD> vlad_vectors; 
    std::vector<std::pair<int, VLAD>> vlad_vectors;
};


} //namespace
#endif