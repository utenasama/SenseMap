//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "vlad_visual_index.h"
#include "VLFeat/mathop.h"
#include "VLFeat/generic.h"
#include <fstream>
#include "util/logging.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace sensemap{


void VladVisualIndex::CreateCodeBook(const CodeBookCreateOptions& option, const Descriptors& training_features) {
    
    if(kmeans!=NULL){
        vl_kmeans_delete(kmeans);
    }
    kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlVectorComparisonType::VlDistanceL2);
    
    vl_kmeans_cluster(kmeans, training_features.data(), training_features.cols(), training_features.rows(),
                      option.num_vocabulary);
}

void VladVisualIndex::ADD(const Descriptors& training_features, const int image_id) {

    VLAD vlad;
    LocalFeaturesToVlad(training_features,vlad);
    vlad_vectors.emplace_back(image_id, vlad);
}

void VladVisualIndex::LocalFeaturesToVlad(const Descriptors& training_features, VLAD& vlad){

    size_t num_data = training_features.rows();
    int dimension = kmeans->dimension;
    int num_centers = kmeans->numCenters;

    std::vector<vl_uint32> indexes(training_features.rows());
    std::vector<float> distances;
    vl_kmeans_quantize(kmeans, indexes.data(), distances.data(), training_features.data(), num_data);

    std::vector<float> assignments(num_data * num_centers, 0);
    for (int i = 0; i < num_data; ++i) {
        assignments[i * num_centers + indexes[i]] = 1.0;
    }

    vlad.resize(kmeans->numCenters * kmeans->dimension,1);

    vl_vlad_encode(vlad.data(), VL_TYPE_FLOAT, vl_kmeans_get_centers(kmeans), kmeans->dimension, kmeans->numCenters,
                   training_features.data(), training_features.rows(), assignments.data(), 0);

    vlad.normalize();
}

void VladVisualIndex::ADD(const VLAD& vlad, const int image_id){
    vlad_vectors.emplace_back(image_id,vlad);
}

void VladVisualIndex::Query(const QueryOptions& option, const VLAD& query_vlad,
                            std::vector<retrieval::ImageScore>* image_scores) {
    
    image_scores->clear();
    image_scores->resize(vlad_vectors.size());
    
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif

    for(size_t idx = 0; idx <vlad_vectors.size(); ++idx){
        double score = query_vlad.dot(vlad_vectors[idx].second);
        retrieval::ImageScore image_score;
        image_score.image_id = vlad_vectors[idx].first;
        image_score.score = score;
        (*image_scores)[idx] = image_score;
    }

    auto SortFunc = [](const retrieval::ImageScore& score1, const retrieval::ImageScore& score2) {
        return score1.score > score2.score;
    };

    size_t num_images = image_scores->size();
    if (option.max_num_images >= 0) {
        num_images = std::min<size_t>(image_scores->size(), option.max_num_images);
    }

    if (num_images == image_scores->size()) {
        std::sort(image_scores->begin(), image_scores->end(), SortFunc);
    } else {
        std::partial_sort(image_scores->begin(), image_scores->begin() + num_images, image_scores->end(), SortFunc);
        image_scores->resize(num_images);
    }
}


void VladVisualIndex::SaveCodeBook(const std::string code_book_path){

    std::ofstream file(code_book_path, std::ios::binary);
    CHECK(file.is_open()) << code_book_path;

    file.write((char*)&(kmeans->dataType), sizeof(vl_type));
    file.write((char*)&(kmeans->dimension), sizeof(vl_size));
    file.write((char*)&(kmeans->numCenters), sizeof(vl_size));
    file.write((char*)&(kmeans->numTrees), sizeof(vl_size));
    file.write((char*)&(kmeans->maxNumComparisons), sizeof(vl_size));
    
    file.write((char*)&(kmeans->maxNumIterations), sizeof(vl_size));
    file.write((char*)&(kmeans->minEnergyVariation), sizeof(double));
    file.write((char*)&(kmeans->numRepetitions), sizeof(vl_size));

    file.write((char*)&(kmeans->verbosity), sizeof(int));
    file.write((char*)&(kmeans->energy), sizeof(double));


    vl_size data_size = vl_get_type_size(kmeans->dataType) * kmeans->dimension * kmeans->numCenters;
    
    file.write((char*)&data_size, sizeof(vl_size)); 
    file.write((char*)kmeans->centers, data_size);



    bool have_center_distances;
    if (kmeans->centerDistances) {
        have_center_distances = true;
        file.write((char*)&have_center_distances, sizeof(bool));

        data_size = vl_get_type_size(kmeans->dataType) * kmeans->numCenters * kmeans->numCenters;
        file.write((char*)&data_size, sizeof(vl_size)); 
        file.write((char*)kmeans->centerDistances, data_size);
    }
    else{
        have_center_distances = false;
        file.write((char*)&have_center_distances, sizeof(bool));
    }
    file.close();
}


void VladVisualIndex::LoadCodeBook(const std::string code_book_path){
    std::cout<<"Load vlad code book"<<std::endl;
    std::ifstream file(code_book_path, std::ios::binary);
    CHECK(file.is_open()) << code_book_path;

    if(kmeans!=NULL){
        vl_kmeans_delete(kmeans);
    }
    kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlVectorComparisonType::VlDistanceL2);


    file.read(reinterpret_cast<char*>(&(kmeans->dataType)), sizeof(vl_type));
    file.read(reinterpret_cast<char*>(&(kmeans->dimension)), sizeof(vl_size));
    file.read(reinterpret_cast<char*>(&(kmeans->numCenters)), sizeof(vl_size));
    file.read(reinterpret_cast<char*>(&(kmeans->numTrees)), sizeof(vl_size));
    file.read(reinterpret_cast<char*>(&(kmeans->maxNumComparisons)), sizeof(vl_size));
    
    file.read(reinterpret_cast<char*>(&(kmeans->maxNumIterations)), sizeof(vl_size));
    file.read(reinterpret_cast<char*>(&(kmeans->minEnergyVariation)), sizeof(double));
    file.read(reinterpret_cast<char*>(&(kmeans->numRepetitions)), sizeof(vl_size));

    file.read(reinterpret_cast<char*>(&(kmeans->verbosity)), sizeof(int));
    file.read(reinterpret_cast<char*>(&(kmeans->energy)), sizeof(double));


    vl_size data_size;
    file.read(reinterpret_cast<char*>(&data_size), sizeof(vl_size));
    kmeans->centers = vl_malloc(data_size);
    file.read(reinterpret_cast<char*>(kmeans->centers), data_size);

    
    bool have_center_distances;
    file.read((char*)&have_center_distances, sizeof(bool));

    if(have_center_distances){
        file.read(reinterpret_cast<char*>(&data_size), sizeof(vl_size));
        kmeans->centerDistances = vl_malloc(data_size);
        file.read(reinterpret_cast<char*>(kmeans->centerDistances), data_size);
    }
    file.close();
}

}
