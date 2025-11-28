//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "graph/correspondence_graph.h"

#include <unordered_set>
#include <fstream>
#include <functional>
#include <Eigen/Core>

#include "util/string.h"
#include "util/graph.h"
#include "util/timer.h"
#include "util/bitmap.h"
#include "base/pose.h"
#include "base/track.h"

namespace sensemap {

CorrespondenceGraph::CorrespondenceGraph() {}

void CorrespondenceGraph::Copy(CorrespondenceGraph &correspondence_graph) {
    correspondence_graph = *this;
}

std::unordered_map<image_pair_t, point2D_t>
CorrespondenceGraph::NumCorrespondencesBetweenImages() const {
    std::unordered_map<image_pair_t, point2D_t> num_corrs_between_images;
    num_corrs_between_images.reserve(image_pairs_.size());
    for (const auto& image_pair : image_pairs_) {
        num_corrs_between_images.emplace(image_pair.first,
                                        image_pair.second.num_correspondences);
    }
    return num_corrs_between_images;
}

EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair)
CorrespondenceGraph::ImagePairs() const {
    return image_pairs_;
}

bool CorrespondenceGraph::ExistImagePair(
    const image_t image_id1, const image_t image_id2) const {
    const image_pair_t pair_id =
            sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    return image_pairs_.find(pair_id) != image_pairs_.end();
}

struct CorrespondenceGraph::ImagePair CorrespondenceGraph::ImagePair(
        const image_t image_id1, const image_t image_id2) const {
    const image_pair_t pair_id =
            sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    return image_pairs_.at(pair_id);
}

struct CorrespondenceGraph::ImagePair CorrespondenceGraph::ImagePair(
    const image_pair_t pair_id) const {
    return image_pairs_.at(pair_id);
}

struct CorrespondenceGraph::Image
CorrespondenceGraph::Image(const image_t image_id) const {
    CHECK(ExistsImage(image_id));
    return images_.at(image_id);
}

struct CorrespondenceGraph::Image& CorrespondenceGraph::ImageRef(const image_t image_id) {
    CHECK(ExistsImage(image_id));
    return images_.at(image_id);
}


void CorrespondenceGraph::CalculateImageNeighbors() {
    if (image_neighbors_.empty()) {
        for (auto image_pair : image_pairs_) {
            image_t image_id1;
            image_t image_id2;
            sensemap::utility::PairIdToImagePair(image_pair.first,
                                                 &image_id1, &image_id2);
            image_neighbors_[image_id1].insert(image_id2);
            image_neighbors_[image_id2].insert(image_id1);
        }
    }
}

void CorrespondenceGraph::UpdateImageNeighbors(
        const image_t image_id1, const image_t image_id2){
    image_neighbors_[image_id1].insert(image_id2);
    image_neighbors_[image_id2].insert(image_id1);
}

std::unordered_map<image_t, std::unordered_set<image_t> >
CorrespondenceGraph::ImageNeighbors() const {
    CHECK(!image_neighbors_.empty());
    return image_neighbors_;
}

std::unordered_set<image_t>
CorrespondenceGraph::ImageNeighbor(const image_t image_id) const {
    CHECK(!image_neighbors_.empty());
    if(image_neighbors_.find(image_id) != image_neighbors_.end()){
        return image_neighbors_.at(image_id);
    }else{
        // std::cout<<"image_id do not exist, return empty set"<<std::endl;
        return std::unordered_set<image_t>();
    }
}

void CorrespondenceGraph::UpdateImagePairsConfidence() {
    size_t max_num_matches = 0;
    for (auto & image_pair : image_pairs_) {
        max_num_matches = std::max(max_num_matches, (size_t)image_pair.second.num_correspondences);
    }

    for (auto & image_pair : image_pairs_) {
        float match_ratio = std::sqrt((float)image_pair.second.num_correspondences / max_num_matches);
        // image_pair.second.two_view_geometry.confidence = match_ratio * 0.4 + image_pair.second.two_view_geometry.confidence * 0.6;
        image_pair.second.two_view_geometry.confidence = std::sqrt(match_ratio * image_pair.second.two_view_geometry.confidence);
    }
}

void CorrespondenceGraph::Finalize() {
    for (auto it = images_.begin(); it != images_.end();) {
        it->second.num_observations = 0;
        for (auto& vcorr : it->second.corrs) {
            vcorr.shrink_to_fit();
            if (vcorr.size() > 0) {
                it->second.num_observations += 1;
            }
        }
        if (it->second.num_observations == 0) {
            std::cout << "delete image " << it->first << ", cause of no observations" << std::endl;
            images_.erase(it++);
        } else {
            ++it;
        }
    }

    std::vector<image_pair_t> deleted_image_pair_ids;
    for (auto & image_pair : image_pairs_) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        if (!ExistsImage(image_id1) || !ExistsImage(image_id2)) {
            deleted_image_pair_ids.push_back(image_pair.first);
        }
        image_pair.second.two_view_geometry.inlier_matches.clear();
        image_pair.second.two_view_geometry.inlier_matches.shrink_to_fit();
    }

    for (auto pair_id : deleted_image_pair_ids) {
        image_pairs_.erase(pair_id);
    }
    std::cout << "delete " << deleted_image_pair_ids.size() << " image pairs" << std::endl;

    image_neighbors_.clear();
    CalculateImageNeighbors();
}

void CorrespondenceGraph::AddImage(const image_t image_id,
                                   const size_t num_points) {
    CHECK(!ExistsImage(image_id));
    images_[image_id].corrs.resize(num_points);
}

void CorrespondenceGraph::AddImage(const image_t image_id,
                                   const struct CorrespondenceGraph::Image image) {
    if (!ExistsImage(image_id)) {
        images_[image_id] = image;
    }
}

void CorrespondenceGraph::DeleteCorrespondences(
        const image_t image_id1, const image_t image_id2){
    const image_pair_t pair_id =
            sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    if(image_pairs_.count(pair_id) == 0){
        return;
    }
    auto fun = [](Correspondence c, image_t id){ return c.image_id == id;};
    const FeatureMatches& matches = FindCorrespondencesBetweenImages(image_id1, image_id2);
    struct Image& image1 = images_.at(image_id1);
    struct Image& image2 = images_.at(image_id2);
    for(auto &corr1 : image1.corrs){
        auto it = std::find_if(corr1.begin(), corr1.end(),
                std::bind(fun,std::placeholders::_1, image_id2));
        while (it != corr1.end()) { // hold ambiguous match.
            corr1.erase(it);
            it = std::find_if(corr1.begin(), corr1.end(), std::bind(fun,std::placeholders::_1, image_id2));
        }
    }
    for(auto &corr2 : image2.corrs){
        auto it = std::find_if(corr2.begin(), corr2.end(),
                std::bind(fun,std::placeholders::_1, image_id1));
        while(it != corr2.end()) { // hold ambiguous match.
            corr2.erase(it);
            it = std::find_if(corr2.begin(), corr2.end(), std::bind(fun,std::placeholders::_1, image_id1));
        }
    }
    image1.num_correspondences -= matches.size();
    image2.num_correspondences -= matches.size();

    image_pairs_.erase(pair_id);

    image_neighbors_[image_id1].erase(image_id2);
    image_neighbors_[image_id2].erase(image_id1);
}

void CorrespondenceGraph::DeleteCorrespondences(const image_t image_id,
                                                const std::vector<point2D_t>& feature_outliers) {
    auto fun = [](Correspondence c, image_t id, point2D_t point2D_idx){ 
        return c.image_id == id && c.point2D_idx == point2D_idx;};
    
    struct Image& image = images_.at(image_id);
    for (auto point2D_idx : feature_outliers) {
        auto& corrs1 = image.corrs.at(point2D_idx);
        for (auto corr1 : corrs1) {
            struct Image& corr_image = images_.at(corr1.image_id);
            auto& corrs2 = corr_image.corrs.at(corr1.point2D_idx);
            auto it = std::find_if(corrs2.begin(), corrs2.end(),
                std::bind(fun, std::placeholders::_1, image_id, point2D_idx));
            if(it != corrs2.end()){
                corrs2.erase(it);
                corr_image.num_correspondences--;

                auto pair_id = utility::ImagePairToPairId(image_id, corr1.image_id);
                image_pairs_.at(pair_id).num_correspondences--;
            }
        }
        image.num_correspondences -= corrs1.size();
        corrs1.clear();
    }
}

void CorrespondenceGraph::AddCorrespondences(
        const image_t image_id1, const image_t image_id2,
        const struct CorrespondenceGraph::ImagePair &pair) {
    // Avoid self-matches - should only happen, if user provides custom matches.
    if (image_id1 == image_id2) {
        std::cout << "WARNING: Cannot use self-matches for image_id=" << image_id1
                  << std::endl;
        return;
    }
    // Corresponding images.
    struct Image &image1 = images_.at(image_id1);
    struct Image &image2 = images_.at(image_id2);

    // Set the number of all correspondences for this image pair. Further below,
    // we will make sure that only unique correspondences are counted.
    const image_pair_t pair_id =
            sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    auto &image_pair = image_pairs_[pair_id];
    // image_pair.qvec = pair.qvec;
    image_pair.two_view_geometry = pair.two_view_geometry;
    image_pair.num_correspondences = pair.num_correspondences;

    image_pair.two_view_geometry.inlier_matches.clear();
    image_pair.two_view_geometry.inlier_matches.shrink_to_fit();
}

void CorrespondenceGraph::AddCorrespondences(
    const image_t image_id1,
    const image_t image_id2,
    const TwoViewGeometry& two_view_geometry,
    const bool remove_redundant) {
    // Avoid self-matches - should only happen, if user provides custom matches.
    if (image_id1 == image_id2 && remove_redundant) {
        std::cout << "WARNING: Cannot use self-matches for image_id=" << image_id1 << std::endl;
        return;
    }

    const FeatureMatches& matches = two_view_geometry.inlier_matches;

    // Corresponding images.
    struct Image& image1 = images_.at(image_id1);
    struct Image& image2 = images_.at(image_id2);

    // Store number of correspondences for each image to find good initial pair.
    image1.num_correspondences += matches.size();
    image2.num_correspondences += matches.size();

    // Set the number of all correspondences for this image pair. Further below,
    // we will make sure that only unique correspondences are counted.
    const image_pair_t pair_id =
        sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    auto& image_pair = image_pairs_[pair_id];
    image_pair.image_id1 = image_id1;
    image_pair.image_id2 = image_id2;
    image_pair.two_view_geometry = two_view_geometry;
    image_pair.num_correspondences += static_cast<point2D_t>(matches.size());

    // Store all matches in correspondence graph data structure. This data-
    // structure uses more memory than storing the raw match matrices, but is
    // significantly more efficient when updating the correspondences in case an
    // observation is triangulated.

    for (const auto& match : matches) {
        const bool valid_idx1 = match.point2D_idx1 < image1.corrs.size();
        const bool valid_idx2 = match.point2D_idx2 < image2.corrs.size();

        if (valid_idx1 && valid_idx2) {
            auto& corrs1 = image1.corrs[match.point2D_idx1];
            auto& corrs2 = image2.corrs[match.point2D_idx2];

            const bool duplicate1 =
                std::find_if(corrs1.begin(), corrs1.end(),
                            [image_id2](const Correspondence& corr) {
                              return corr.image_id == image_id2;
                            }) != corrs1.end();
            const bool duplicate2 =
                std::find_if(corrs2.begin(), corrs2.end(),
                            [image_id1](const Correspondence& corr) {
                              return corr.image_id == image_id1;
                            }) != corrs2.end();

            if (remove_redundant && (duplicate1 || duplicate2)) {
                image1.num_correspondences -= 1;
                image2.num_correspondences -= 1;
                image_pair.num_correspondences -= 1;
                std::cout << StringPrintf(
                                "WARNING: Duplicate correspondence between "
                                "point2D_idx=%d in image_id=%d and point2D_idx=%d in "
                                "image_id=%d",
                                match.point2D_idx1, image_id1, match.point2D_idx2,
                                image_id2)
                          << std::endl;
            } else {
                corrs1.emplace_back(image_id2, match.point2D_idx2);
                corrs2.emplace_back(image_id1, match.point2D_idx1);
            }
        } else {
            image1.num_correspondences -= 1;
            image2.num_correspondences -= 1;
            image_pair.num_correspondences -= 1;
            if (!valid_idx1) {
                std::cout
                    << StringPrintf(
                          "WARNING: point2D_idx=%d in image_id=%d does not exist",
                          match.point2D_idx1, image_id1)
                    << std::endl;
            }
            if (!valid_idx2) {
                std::cout
                    << StringPrintf(
                          "WARNING: point2D_idx=%d in image_id=%d does not exist",
                          match.point2D_idx2, image_id2)
                    << std::endl;
            }
        }
    }
    image_pair.two_view_geometry.inlier_matches.clear();
    image_pair.two_view_geometry.inlier_matches.shrink_to_fit();
}


void CorrespondenceGraph::UpdateCorrespondence(
    const image_t image_id1,
    const image_t image_id2,
    const point2D_t point_id1,
    const point2D_t point_id2){
    // Avoid self-matches - should only happen, if user provides custom matches.
    if (image_id1 == image_id2) {
        std::cout << "WARNING: Cannot use self-matches for image_id=" << image_id1 << std::endl;
        return;
    }

    // Corresponding images.
    struct Image& image1 = images_.at(image_id1);
    struct Image& image2 = images_.at(image_id2);

    const image_pair_t pair_id =
        sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    auto& image_pair = image_pairs_[pair_id];


    // Resize corrs
    image1.corrs.resize(image1.corrs.size() + 1);
    image2.corrs.resize(image2.corrs.size() + 1);

    // Update correspondence number
    image1.num_correspondences += 1;
    image2.num_correspondences += 1;
    image_pair.num_correspondences += 1;


    auto& corrs1 = image1.corrs[point_id1];
    auto& corrs2 = image2.corrs[point_id2];

    corrs1.emplace_back(image_id2, point_id2);
    corrs2.emplace_back(image_id1, point_id1);

}

void CorrespondenceGraph::FindTransitiveCorrespondences(
    const image_t image_id, 
    const point2D_t point2D_idx,
    const size_t transitivity,
    std::vector<Correspondence>* found_corrs) const {

    if (!HasCorrespondences(image_id, point2D_idx)) {
        return ;
    }

    // if (transitivity == 1) {
    //     std::vector<CorrespondenceGraph::Correspondence> corrs = FindCorrespondences(image_id, point2D_idx);
    //     found_corrs = &corrs;
    //     return ;
    // }

    found_corrs->emplace_back(image_id, point2D_idx);

    std::unordered_map<image_t, std::unordered_set<point2D_t>> image_corrs;
    image_corrs[image_id].insert(point2D_idx);

    size_t corr_queue_begin = 0;
    size_t corr_queue_end = 1;

    for (size_t t = 0; t < transitivity; ++t) {
        // Collect correspondences at transitive level t to all
        // correspondences that were collected at transitive level t - 1.
        for (size_t i = corr_queue_begin; i < corr_queue_end; ++i) {
            const Correspondence ref_corr = (*found_corrs)[i];

            const struct Image& image = images_.at(ref_corr.image_id);
            const std::vector<Correspondence>& ref_corrs =
                image.corrs[ref_corr.point2D_idx];

            for (const Correspondence corr : ref_corrs) {
                // Check if correspondence already collected, otherwise collect.
                auto& corr_image_corrs = image_corrs[corr.image_id];
                // if (corr_image_corrs.count(corr.point2D_idx) == 0) {
                //     corr_image_corrs.insert(corr.point2D_idx);
                //     found_corrs->emplace_back(corr.image_id, corr.point2D_idx);
                // }
                if (corr_image_corrs.insert(corr.point2D_idx).second) {
                    found_corrs->emplace_back(corr.image_id, corr.point2D_idx);
                }
            }
        }

        // Move on to the next block of correspondences at next transitive level.
        corr_queue_begin = corr_queue_end;
        corr_queue_end = found_corrs->size();

        // No new correspondences collected in last transitivity level.
        if (corr_queue_begin == corr_queue_end) {
            break;
        }
    }

    // Remove first element, which is the given observation by swapping it
    // with the last collected correspondence.
    if (found_corrs->size() > 1) {
        found_corrs->front() = found_corrs->back();
    }
    found_corrs->pop_back();

    // return found_corrs;
}


void CorrespondenceGraph::UpdateCorrespondence(const image_t image_id1,
                                               const image_t image_id2,
                                               const TwoViewGeometry& two_view_geometry){
    const image_pair_t pair_id =
        sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    auto& image_pair = image_pairs_[pair_id];
    image_pair.two_view_geometry = two_view_geometry;
    image_pair.two_view_geometry.inlier_matches.clear();
    image_pair.two_view_geometry.inlier_matches.shrink_to_fit();
}

FeatureMatches CorrespondenceGraph::FindCorrespondencesBetweenImages(
    const image_t image_id1, 
    const image_t image_id2) const {
    const auto num_correspondences =
        NumCorrespondencesBetweenImages(image_id1, image_id2);

    if (num_correspondences == 0) {
        return {};
    }

    FeatureMatches found_corrs;
    found_corrs.reserve(num_correspondences);

    const struct Image& image1 = images_.at(image_id1);

    for (point2D_t point2D_idx1 = 0; point2D_idx1 < image1.corrs.size();
        ++point2D_idx1) {
        for (const Correspondence& corr1 : image1.corrs[point2D_idx1]) {
            if (corr1.image_id == image_id2) {
                found_corrs.emplace_back(point2D_idx1, corr1.point2D_idx);
            }
        }
    }

    return found_corrs;
}

void CorrespondenceGraph::FindTransitiveCorrespondencesBetweenImages(
    const image_t image_id1,
    const image_t image_id2,
    const size_t transitivity,
    FeatureMatches* corrs) const{

    // FeatureMatches found_corrs;
    const struct Image& image1 = images_.at(image_id1);

    corrs->reserve(image1.corrs.size());

    for(point2D_t point2D_idx1 = 0; point2D_idx1 < image1.corrs.size(); 
       ++point2D_idx1) {
        std::vector<CorrespondenceGraph::Correspondence> transitive_corrs;
        FindTransitiveCorrespondences(image_id1,point2D_idx1,transitivity, &transitive_corrs);
        
        for (const Correspondence& corr1 : transitive_corrs) {
            if (corr1.image_id == image_id2) {
                corrs->emplace_back(point2D_idx1, corr1.point2D_idx);
                //break;
            }
        }
    }
}

bool CorrespondenceGraph::IsTwoViewObservation(
    const image_t image_id,
    const point2D_t point2D_idx) const {
    const struct Image& image = images_.at(image_id);
    const std::vector<Correspondence>& corrs = image.corrs.at(point2D_idx);
    if (corrs.size() != 1) {
        return false;
    }
    const struct Image& other_image = images_.at(corrs[0].image_id);
    const std::vector<Correspondence>& other_corrs =
        other_image.corrs.at(corrs[0].point2D_idx);
    return other_corrs.size() == 1;
}

std::vector<class Track> CorrespondenceGraph::GenerateTracks(
    int track_degree,
    bool remove_ambiguous,
    bool verbose) {
    std::vector<class Track> tracks;

    //set collected flags
    image_t max_image_id = 0;
    std::vector<image_t> image_ids;
    image_ids.reserve(images_.size());
    for(auto &image : images_) {
        max_image_id = std::max(max_image_id, image.first);
        image_ids.push_back(image.first);
    }
    std::vector<std::vector<bool> > image_corrs_flag(max_image_id + 1);
    for(auto image_id : image_ids) {
        image_corrs_flag[image_id].resize(images_.at(image_id).corrs.size(), false);
    }

    int delete_num = 0;
    for(auto image_id : image_ids){
        auto & image = images_.at(image_id);
        auto & corrs = image.corrs;
        for (point2D_t point2D_idx = 0; point2D_idx < corrs.size();
        ++point2D_idx){

            // Check if correspondence already collected, otherwise collect.
            if (image_corrs_flag[image_id][point2D_idx] == true) {
                continue;
            }

            image_corrs_flag[image_id][point2D_idx] = true;
            
            if (corrs[point2D_idx].size() == 0) {
                continue;
            }

            std::vector<CorrespondenceGraph::Correspondence> found_corrs;
            found_corrs.emplace_back(image_id, point2D_idx);

            size_t corr_queue_begin = 0;
            size_t corr_queue_end = found_corrs.size();
            while(corr_queue_begin != corr_queue_end) {
                for (size_t i = corr_queue_begin; i < corr_queue_end; ++i) {
                    auto &ref_corr = found_corrs[i];

                    auto &image_i = images_.at(ref_corr.image_id);
                    auto &ref_corrs = image_i.corrs[ref_corr.point2D_idx];
                    for (const auto& corr : ref_corrs) {
                        // Check if correspondence already collected, otherwise collect.
                        if (image_corrs_flag[corr.image_id][corr.point2D_idx] == false) {
                            image_corrs_flag[corr.image_id][corr.point2D_idx] = true;
                            found_corrs.emplace_back(corr.image_id, corr.point2D_idx);
                        }
                    }
                }

                // Move on to the next block of correspondences at next transitive level.
                corr_queue_begin = corr_queue_end;
                corr_queue_end = found_corrs.size();
            }

            if (found_corrs.size() <= 1) {
                continue;
            }

            //delete the [track_degree] views tracks and related data
            bool ambiguous = false;
            // if (remove_ambiguous) {
            //     for(int i = 0; i < found_corrs.size(); ++i){
            //         for(int j = 0; j < found_corrs.size(); ++j){
            //             if(i == j){
            //                 continue;
            //             }
            //             if(found_corrs[i].image_id == found_corrs[j].image_id){
            //                 //a track with two points in the same image
            //                 ambiguous = true;
            //                 break;
            //             }
            //         }
            //     }
            // }
            if(ambiguous || found_corrs.size() < track_degree){
                std::unordered_set<image_pair_t > pair_ids;
                for(const auto & corr : found_corrs){
                    //collect image_pairs
                    auto& ref_image = images_[corr.image_id];
                    auto& corr_images = ref_image.corrs[corr.point2D_idx];
                    for(auto& image2 : corr_images){
                        image_pair_t pair_id =
                                sensemap::utility::ImagePairToPairId(
                                        corr.image_id, image2.image_id);
                        pair_ids.insert(pair_id);
                    }
                    //delete corrs & observations
                    ref_image.num_correspondences -= corr_images.size();
                    corr_images.clear();
                    if (found_corrs.size() > 1) {
                        ref_image.num_observations--;
                    }
                }
                //delete image_pairs
                for (auto pair_id : pair_ids) {
                    if (image_pairs_.find(pair_id) != image_pairs_.end()) {
                        image_pairs_[pair_id].num_correspondences--;
                    }
                }
                delete_num++;
                continue;
            }

            Track track;
            track.Reserve(found_corrs.size());
            for(auto corr : found_corrs){
                track.AddElement(corr.image_id, corr.point2D_idx);
            }
            tracks.emplace_back(track);
    
        }
    }

    if(verbose){
        printf("delete %d corresspondences while generating tracks!\n",
                delete_num);
        printf("find %lu tracks ! (min length is %d)\n",
                tracks.size(), track_degree);
    }

    image_neighbors_.clear();
    Finalize();

    return tracks;
}

void CorrespondenceGraph::TrackSelection(std::vector<class Track>& tracks,
                                         std::vector<unsigned char>& inliers,
                                         const int max_cover_per_view,
                                         const std::unordered_set<image_t> &const_image_ids) {
    Timer timer;
    timer.Start();
    if (max_cover_per_view > 0) {
        image_t max_image_id = 0;
        for(const auto &image : images_) {
            max_image_id = std::max(max_image_id, image.first);
        }
        std::vector<int> cover_view(max_image_id + 1, 0);
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (!inliers.at(i)) {
                continue;
            }
            const auto & track = tracks.at(i);
            bool selected = false;
            for (const auto & track_elem : track.Elements()) {
                if (cover_view[track_elem.image_id] < max_cover_per_view) {
                    selected = true;
                    break;
                }
            }

            inliers[i] = selected;

            if (selected) {
                std::unordered_set<image_t> track_images;
                for (const auto & track_elem : track.Elements()) {
                    track_images.insert(track_elem.image_id);
                }
                for (auto image_id : track_images) {
                    cover_view[image_id]++;
                }
            }
        }
    }

    if (!const_image_ids.empty()){
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (inliers.at(i)) {
                continue;
            }
            auto track = tracks.at(i);
            bool selected = false;
            for (const auto & track_elem : track.Elements()) {
                if (const_image_ids.find(track_elem.image_id) != const_image_ids.end()) {
                    selected = true;
                    break;
                }
            }
            inliers[i] = selected;
        }
    }

    size_t num_selected = 0;
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (inliers[i]) {
            num_selected++;
            continue;
        }

        auto& track_elems = tracks.at(i).Elements();

        // std::set<std::pair<image_t, point2D_t>> track_elem_set;
        // for (auto& corr : track_elems) {
        //     track_elem_set.insert(std::pair<image_t, point2D_t>(corr.image_id, corr.point2D_idx));
        // }

        std::unordered_set<image_pair_t> pair_ids;
        for (const auto & corr : track_elems) {
            // collect image_pairs
            auto& ref_image = images_[corr.image_id];
            auto& corr_images = ref_image.corrs[corr.point2D_idx];
            for (const auto& image2 : corr_images) {
                image_pair_t pair_id = sensemap::utility::ImagePairToPairId(corr.image_id, image2.image_id);
                pair_ids.insert(pair_id);

                // std::pair<image_t, point2D_t> corr_elem =
                //         std::pair<image_t, point2D_t>(image2.image_id, image2.point2D_idx);
                // CHECK(track_elem_set.find(corr_elem) != track_elem_set.end())
                // << "track_elem_set don't contain corr elem:" << corr_elem.first << " " << corr_elem.second;
            }
            // delete corrs & observations
            ref_image.num_correspondences -= corr_images.size();

            corr_images.clear();
            if (track_elems.size() > 1) {
                ref_image.num_observations--;
            }
        }
        // delete image_pairs
        for (auto pair_id : pair_ids) {
            image_pairs_[pair_id].num_correspondences--;
        }
    }
    std::cout << StringPrintf("TrackSelection: %d tracks are selected\n", num_selected);

    image_neighbors_.clear();
    Finalize();
    timer.PrintSeconds();
}

const std::unordered_set<image_pair_t>& CorrespondenceGraph::GetStrongLoopPairs() const {
    return strong_loop_pairs_;
}

const std::unordered_map<image_pair_t,double>& CorrespondenceGraph::GetLoopPairsInfo() const {
    return loop_pairs_info_;
}

const std::unordered_set<image_pair_t>& CorrespondenceGraph::GetNormalPairs() const {
    return normal_pairs_;
}
const std::unordered_set<image_pair_t>& CorrespondenceGraph::GetAllPairs() const {
    std::unordered_set<image_pair_t> image_pairs_ids;
    for (auto image_pair : image_pairs_) {
        image_pairs_ids.insert(image_pair.first);
    }
    return image_pairs_ids;
}

void CorrespondenceGraph::WriteCorrespondenceData(
    const std::string &path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open());

    file << "# NUM_IMAGE_PAIRS" << std::endl;
    file << "## Image pair list:" << std::endl;
    file << "# IMAGE_PAIR_ID, NUM_CORRESPONDENCE" << std::endl;
    file << "# NUM_IMAGE" << std::endl;
    file << "## Image list with correspondence of each feature point:"
         << std::endl;
    file << "# IMAGE_ID, NUM_OBSERVATIONS, NUM_CORRESPONDENCES" << std::endl;
    file << "## Correspondence list of all of feature points:" << std::endl;
    file << "# NUM_CORRESPONDENCE" << std::endl;
    file << "## Correspondence list of each feature point:" << std::endl;
    file << "# IMAGE_ID, POINT2D_IDX" << std::endl;

    // write image_pairs_
    uint32_t num_image_pair = static_cast<uint32_t>(image_pairs_.size());
    file << num_image_pair << std::endl;

    std::ostringstream image_pair_line;
    for (const auto & image_pair : image_pairs_) {
        image_pair_line << image_pair.first << " " 
                        << image_pair.second.num_correspondences << " "
                        << image_pair.second.two_view_geometry.qvec.x() << " "
                        << image_pair.second.two_view_geometry.qvec.y() << " "
                        << image_pair.second.two_view_geometry.qvec.z() << " "
                        << image_pair.second.two_view_geometry.qvec.w() << " "
                        << image_pair.second.image_id1<<" "
                        << image_pair.second.image_id2<<" "
                        << image_pair.second.two_view_geometry.tvec.x() << " "
                        << image_pair.second.two_view_geometry.tvec.y() << " "
                        << image_pair.second.two_view_geometry.tvec.z() << " ";
    }
    std::string image_pair_line_string = image_pair_line.str();
    image_pair_line_string = 
        image_pair_line_string.substr(0, image_pair_line_string.size() - 1);
    file << image_pair_line_string << std::endl;

    // write images_
    uint32_t num_image = static_cast<uint32_t>(images_.size());
    file << num_image << std::endl;

    for (const auto & image : images_) {
        std::ostringstream line;
        std::string line_string;

        line << image.first << " " 
             << image.second.num_observations << " " 
             << image.second.num_correspondences << " "
             << image.second.corrs.size();
        line_string = line.str();
        file << line_string << std::endl;

        for (const auto & vcorr : image.second.corrs) {
            line.str("");
            line.clear();

            uint32_t num_corr = static_cast<uint32_t>(vcorr.size());
            line << num_corr << " ";
            for (const auto & corr : vcorr) {
                line << corr.image_id << " " << corr.point2D_idx << " ";
            }
            line_string = line.str();
            line_string = line_string.substr(0, line_string.size() - 1);
            file << line_string << std::endl;
        }
    }

    file.close();
}

void CorrespondenceGraph::ReadCorrespondenceData(const std::string &path) {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        } else{
            break;
        }
    }

    std::stringstream header_line_stream(line);
    // read image_pairs_
    std::getline(header_line_stream >> std::ws, item, ' ');
    const uint32_t num_image_pair = std::stoul(item);

    std::cout << "num_image_pair: " << num_image_pair << std::endl;
    
    std::getline(file, line);
    std::stringstream image_pair_line_stream(line);
    for (uint32_t i = 0; i < num_image_pair; ++i) {
        struct ImagePair image_pair;
        // image_pair.two_view_geometry = std::make_shared<TwoViewGeometry>();

        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair_t image_pair_id = std::stoull(item);
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.num_correspondences = std::stoul(item); 
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.two_view_geometry.qvec.x() = std::stod(item);
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.two_view_geometry.qvec.y() = std::stod(item);
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.two_view_geometry.qvec.z() = std::stod(item);
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.two_view_geometry.qvec.w() = std::stod(item);

        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.image_id1 = std::stoul(item);
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.image_id2 = std::stoul(item);
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.two_view_geometry.tvec.x() = std::stod(item);
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.two_view_geometry.tvec.y() = std::stod(item);
        std::getline(image_pair_line_stream >> std::ws, item, ' ');
        image_pair.two_view_geometry.tvec.z() = std::stod(item);

        image_pairs_.emplace(image_pair_id, image_pair);
    }

    std::getline(file, line);
    std::stringstream image_header_line_stream(line);
    std::getline(image_header_line_stream >> std::ws, item, ' ');
    // read images_
    const uint32_t num_image = std::stoul(item);
    std::cout << "num_image: " << num_image << std::endl;

    for (uint32_t i = 0; i < num_image; ++i) {
        image_t image_id;
        uint32_t image_num_correspondence;
        struct Image image;

        std::getline(file, line);
        std::stringstream image_line_stream(line);
        std::getline(image_line_stream >> std::ws, item, ' ');
        image_id = std::stoul(item);
        std::getline(image_line_stream >> std::ws, item, ' ');
        image.num_observations = std::stoul(item);
        std::getline(image_line_stream >> std::ws, item, ' ');
        image.num_correspondences = std::stoul(item);
        std::getline(image_line_stream >> std::ws, item, ' ');
        image_num_correspondence = std::stoul(item);

        for (uint32_t j = 0; j < image_num_correspondence; ++j) {
            point2D_t num_corr;
            Correspondence corr;
            std::vector<Correspondence> vCorr;

            std::getline(file, line);
            std::stringstream corres_line_stream(line);
            std::getline(corres_line_stream >> std::ws, item, ' ');
            num_corr = std::stoul(item);
            vCorr.reserve(num_corr);
            for (uint32_t k = 0; k < num_corr; ++k) {
                std::getline(corres_line_stream >> std::ws, item, ' ');
                corr.image_id = std::stoul(item);
                std::getline(corres_line_stream >> std::ws, item, ' ');
                corr.point2D_idx = std::stoul(item);
                vCorr.emplace_back(corr);
            }
            image.corrs.emplace_back(vCorr);
        }
        images_.emplace(image_id, image);
    }

    file.close();
}

void CorrespondenceGraph::WriteBlueToothPairsInfoBinaryData(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open());

    uint32_t num_pair = static_cast<uint32_t>(pairs_bluetooth_info_.size());
    file.write((char*)&num_pair, sizeof(uint32_t));

    for(const auto& pair_bluetooth_info : pairs_bluetooth_info_){
        file.write((char*)&pair_bluetooth_info.first, sizeof(image_pair_t));
        file.write((char*)&pair_bluetooth_info.second, sizeof(double));
    }
    file.close();
}

void CorrespondenceGraph::ReadBlueToothPairsInfoBinaryData(const std::string& path) {
    pairs_bluetooth_info_.clear();
    
    std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    uint32_t num_pair;
    file.read(reinterpret_cast<char*>(&num_pair), sizeof(uint32_t));

    for(uint32_t i = 0; i < num_pair; ++i){
        image_pair_t pair_id;
        double bluetooth_distance;
        file.read(reinterpret_cast<char*>(&pair_id), sizeof(image_pair_t));
        file.read(reinterpret_cast<char*>(&bluetooth_distance), sizeof(double));
        pairs_bluetooth_info_.emplace(pair_id, bluetooth_distance);
    }
    file.close();
}

void CorrespondenceGraph::WriteStrongLoopsBinaryData(const std::string& path) const{
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open());

    uint32_t num_strong_loop_pair = static_cast<uint32_t>(strong_loop_pairs_.size());
    std::cout << "num_strong_loop_pair = " << num_strong_loop_pair << std::endl;
    file.write((char*)&num_strong_loop_pair, sizeof(uint32_t));

    for(const auto& strong_loop_pair:strong_loop_pairs_){
        file.write((char*)&strong_loop_pair,sizeof(image_pair_t));
    }
    file.close();
}

void CorrespondenceGraph::ReadStrongLoopsBinaryData(const std::string& path){
    
    strong_loop_pairs_.clear();
    
    std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    uint32_t num_strong_loop_pair;
    file.read(reinterpret_cast<char*>(&num_strong_loop_pair), sizeof(uint32_t));
    std::cout << "num_strong_loop_pair: " << num_strong_loop_pair << std::endl;

    for(uint32_t i = 0; i < num_strong_loop_pair; ++i){
        image_pair_t strong_loop_pair;
        file.read(reinterpret_cast<char*>(&strong_loop_pair), sizeof(image_pair_t));
        strong_loop_pairs_.insert(strong_loop_pair);
    }
    file.close();
}

void CorrespondenceGraph::WriteLoopPairsInfoBinaryData(const std::string& path) const{
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open());

    uint32_t num_loop_pair = static_cast<uint32_t>(loop_pairs_info_.size());
    std::cout << "num_loop_pair = " << num_loop_pair << std::endl;
    file.write((char*)&num_loop_pair, sizeof(uint32_t));

    for(const auto& loop_pair:loop_pairs_info_){
        file.write((char*)&loop_pair.first,sizeof(image_pair_t));
        file.write((char*)&loop_pair.second,sizeof(double));

    }
    file.close();
}

void CorrespondenceGraph::ReadLoopPairsInfoBinaryData(const std::string& path){
    
    loop_pairs_info_.clear();
    
    std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    uint32_t num_loop_pair;
    file.read(reinterpret_cast<char*>(&num_loop_pair), sizeof(uint32_t));
    std::cout << "num_loop_pair: " << num_loop_pair << std::endl;

    for(uint32_t i = 0; i < num_loop_pair; ++i){
        image_pair_t pair_id;
        double match_score;
        file.read(reinterpret_cast<char*>(&pair_id), sizeof(image_pair_t));
        file.read(reinterpret_cast<char*>(&match_score), sizeof(double));
        loop_pairs_info_.emplace(pair_id,  match_score);
    }
    file.close();
}


void CorrespondenceGraph::WriteNormalPairsBinaryData(const std::string& path) const{
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open());

    uint32_t num_pair = static_cast<uint32_t>(normal_pairs_.size());
    std::cout << "num_pair = " << num_pair << std::endl;
    file.write((char*)&num_pair, sizeof(uint32_t));

    for(const auto& pair:normal_pairs_){
        file.write((char*)&pair,sizeof(image_pair_t));
    }
    file.close();
}

void CorrespondenceGraph::ReadNormalPairsBinaryData(const std::string& path){
    
    normal_pairs_.clear();
    
    std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    uint32_t num_pair;
    file.read(reinterpret_cast<char*>(&num_pair), sizeof(uint32_t));
    std::cout << "num_pair: " << num_pair << std::endl;

    for(uint32_t i = 0; i < num_pair; ++i){
        image_pair_t pair;
        file.read(reinterpret_cast<char*>(&pair), sizeof(image_pair_t));
        normal_pairs_.insert(pair);
    }
    file.close();
}

void CorrespondenceGraph::WriteCorrespondenceBinaryData(
    const std::string &path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open());
    
    std::string version_name("version1.1");
    version_name += '\0';
    std::cout << version_name.c_str() << " " << version_name.length() << std::endl;
    file.write(version_name.c_str(), version_name.length());

    // write image_pairs_
    uint32_t num_image_pair = static_cast<uint32_t>(image_pairs_.size());
    std::cout << "num_image_pair = " << num_image_pair << std::endl;
    file.write((char*)&num_image_pair, sizeof(uint32_t));

    for (const auto & image_pair : image_pairs_) {
        file.write((char*)&image_pair.first, sizeof(image_pair_t));
        // std::cout<<"num corr: "<<image_pair.second.num_correspondences<<std::endl;
        file.write((char*)&image_pair.second.num_correspondences, sizeof(point2D_t));
        file.write((char*)&image_pair.second.two_view_geometry.qvec.x(), sizeof(double));
        file.write((char*)&image_pair.second.two_view_geometry.qvec.y(), sizeof(double));
        file.write((char*)&image_pair.second.two_view_geometry.qvec.z(), sizeof(double));
        file.write((char*)&image_pair.second.two_view_geometry.qvec.w(), sizeof(double));
        file.write((char*)&image_pair.second.two_view_geometry.tvec.x(), sizeof(double));
        file.write((char*)&image_pair.second.two_view_geometry.tvec.y(), sizeof(double));
        file.write((char*)&image_pair.second.two_view_geometry.tvec.z(), sizeof(double));
    }
   
    // write images_
    uint32_t num_image = static_cast<uint32_t>(images_.size());
    file.write((char*)&num_image, sizeof(uint32_t));

    for (const auto & image : images_) {
        file.write((char*)&image.first, sizeof(image_t));
        file.write((char*)&image.second.num_observations, sizeof(point2D_t));
        file.write((char*)&image.second.num_correspondences, sizeof(point2D_t));
        size_t corres_size = image.second.corrs.size();
        file.write((char*)&corres_size, sizeof(size_t));

        for (const auto & vcorr : image.second.corrs) {
            point2D_t num_corr = static_cast<uint32_t>(vcorr.size());
            file.write((char*)&num_corr, sizeof(point2D_t));
            for (const auto & corr : vcorr) {
                file.write((char*)&corr.image_id, sizeof(image_t));
                file.write((char*)&corr.point2D_idx, sizeof(point2D_t));
            }
        }
    }

    file.close();
}

void CorrespondenceGraph::ReadCorrespondenceBinaryData(const std::string &path) {
    std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    std::string version_name = "\0";

    char name_char;
    do {
        file.read(&name_char, 1);
        if (name_char != '\0') {
            version_name += name_char;
        }
    } while (name_char != '\0');
    bool old_version = false;
    if (version_name.find("version") == std::string::npos) {
        old_version = true;
        file.seekg(0);
    }
    std::cout << version_name.c_str() << " " << version_name.length() << " " << old_version << std::endl;

    // read image_pairs_
    uint32_t num_image_pair;
    file.read(reinterpret_cast<char*>(&num_image_pair), sizeof(uint32_t));
    std::cout << "num_image_pair: " << num_image_pair << std::endl;
    
    for (uint32_t i = 0; i < num_image_pair; ++i) {
        struct ImagePair image_pair;
        // image_pair.two_view_geometry = std::make_shared<TwoViewGeometry>();

        image_pair_t image_pair_id;
        file.read(reinterpret_cast<char*>(&image_pair_id), sizeof(image_pair_t));

        point2D_t num_correspondences;
        file.read(reinterpret_cast<char*>(&num_correspondences), sizeof(point2D_t));
        image_pair.num_correspondences = num_correspondences; 

        double x, y, z, w;
        file.read(reinterpret_cast<char*>(&x), sizeof(double));
        file.read(reinterpret_cast<char*>(&y), sizeof(double));
        file.read(reinterpret_cast<char*>(&z), sizeof(double));
        file.read(reinterpret_cast<char*>(&w), sizeof(double));
        image_pair.two_view_geometry.qvec.x() = x;
        image_pair.two_view_geometry.qvec.y() = y;
        image_pair.two_view_geometry.qvec.z() = z;
        image_pair.two_view_geometry.qvec.w() = w;
        if (!old_version) {
            double tx, ty, tz;
            file.read(reinterpret_cast<char*>(&tx), sizeof(double));
            file.read(reinterpret_cast<char*>(&ty), sizeof(double));
            file.read(reinterpret_cast<char*>(&tz), sizeof(double));
            image_pair.two_view_geometry.tvec.x() = tx;
            image_pair.two_view_geometry.tvec.y() = ty;
            image_pair.two_view_geometry.tvec.z() = tz;
        }

        // image_t image_id1, image_id2;
        // utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);
        image_pairs_.emplace(image_pair_id, image_pair);
    }

    // read images_
    uint32_t num_image;
    file.read(reinterpret_cast<char*>(&num_image), sizeof(uint32_t));
    std::cout << "num_image: " << num_image << std::endl;

    for (uint32_t i = 0; i < num_image; ++i) {
        image_t image_id;
        size_t image_num_correspondence;
        struct Image image;

        file.read(reinterpret_cast<char*>(&image_id), sizeof(image_t));
        file.read(reinterpret_cast<char*>(&image.num_observations), sizeof(point2D_t));
        file.read(reinterpret_cast<char*>(&image.num_correspondences), sizeof(point2D_t));
        file.read(reinterpret_cast<char*>(&image_num_correspondence), sizeof(size_t));
        image.corrs.reserve(image_num_correspondence);
        for (uint32_t j = 0; j < image_num_correspondence; ++j) {
            point2D_t num_corr;
            Correspondence corr;
            file.read(reinterpret_cast<char*>(&num_corr), sizeof(point2D_t));

            std::vector<Correspondence> vCorr;
            vCorr.reserve(num_corr);

            for (uint32_t k = 0; k < num_corr; ++k) {
                file.read(reinterpret_cast<char*>(&corr.image_id), sizeof(image_t));
                file.read(reinterpret_cast<char*>(&corr.point2D_idx), sizeof(point2D_t));
                vCorr.emplace_back(corr);
            }
            image.corrs.emplace_back(vCorr);
        }
        images_.emplace(image_id, image);
    }

    file.close();
}


std::vector<std::pair<image_t, image_t>> CorrespondenceGraph::ReadCorrespondencePairBinaryData(const std::string &path){
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    // read image_pairs_
    uint32_t num_image_pair;
    file.read(reinterpret_cast<char *>(&num_image_pair), sizeof(uint32_t));
    std::cout << "num_image_pair: " << num_image_pair << std::endl;

    std::vector<std::pair<image_t, image_t>> image_pairs(num_image_pair);
    for (uint32_t i = 0; i < num_image_pair; ++i) {
        uint64_t image_pair_id;
        file.read(reinterpret_cast<char *>(&image_pair_id), sizeof(uint64_t));
        auto image_id2 = static_cast<image_t>(image_pair_id % std::numeric_limits<int>::max());
        auto image_id1 = static_cast<image_t>((image_pair_id - image_id2) / std::numeric_limits<int>::max());

        image_pairs[i] = {image_id1, image_id2};

        point2D_t num_correspondences;
        file.read(reinterpret_cast<char *>(&num_correspondences), sizeof(point2D_t));

        double x, y, z, w;
        file.read(reinterpret_cast<char *>(&x), sizeof(double));
        file.read(reinterpret_cast<char *>(&y), sizeof(double));
        file.read(reinterpret_cast<char *>(&z), sizeof(double));
        file.read(reinterpret_cast<char *>(&w), sizeof(double));
    }
    file.close();

    std::cout << image_pairs.size() << std::endl;
    return image_pairs;
}

void CorrespondenceGraph::WriteImagePairsBinaryData(const std::string &path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open());

    // write image_pairs_
    uint32_t num_image_pair = static_cast<uint32_t>(image_pairs_.size());
    std::cout << "num_image_pair = " << num_image_pair << std::endl;
    file.write((char*)&num_image_pair, sizeof(uint32_t));
    for (const auto & image_pair : image_pairs_) {
        file.write((char*)&image_pair.first, sizeof(image_pair_t));
        file.write((char*)&image_pair.second.two_view_geometry.confidence, sizeof(double));
    }

    file.close();
}

void CorrespondenceGraph::ReadImagePairsBinaryData(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    // read image_pairs_
    uint32_t num_image_pair;
    file.read(reinterpret_cast<char*>(&num_image_pair), sizeof(uint32_t));
    std::cout << "num_image_pair: " << num_image_pair << std::endl;

    for (uint32_t i = 0; i < num_image_pair; ++i) {
        uint64_t image_pair_id;
        file.read(reinterpret_cast<char *>(&image_pair_id), sizeof(uint64_t));

        double confidence;
        file.read(reinterpret_cast<char *>(&confidence), sizeof(double));

        image_pairs_.at(image_pair_id).two_view_geometry.confidence = confidence;        
    }
    file.close();
}

void CorrespondenceGraph::ExportToGraph(const std::string& path) const {
    // image_t max_capacity = 0;
    // for (const auto & image : images_) {
    //     max_capacity = std::max(max_capacity, image.first);
    // }
    // Eigen::MatrixXi graph_data = 
    //     Eigen::MatrixXi::Zero(max_capacity, max_capacity);

    // for (auto & image_pair : image_pairs_) {
    //     image_t image_id1, image_id2;
    //     sensemap::utility::PairIdToImagePair(image_pair.first, 
    //                                          &image_id1, 
    //                                          &image_id2);
    //     point2D_t num_corres =
    //     NumCorrespondencesBetweenImages(image_id1, image_id2);
    //     //TODO: Convert the image id to the image name index
    //     if(!image_index_map_.empty()){
    //         image_id1 = image_index_map_.find(image_id1)->second;
    //         image_id2 = image_index_map_.find(image_id2)->second;
    //     }
    //     graph_data(image_id1 - 1, image_id2 - 1) = num_corres;
    //     graph_data(image_id2 - 1, image_id1 - 1) = num_corres;
    // }
    // sensemap::graph::ExportToGraph(path, graph_data);

    const int max_graph_node = 50000;

    size_t min_val = std::numeric_limits<size_t>::max();
    size_t max_val = 0;
    for (auto & image_pair : image_pairs_) {
        size_t num_corres = NumCorrespondencesBetweenImages(image_pair.first);
        min_val = std::min(min_val, num_corres);
        max_val = std::max(max_val, num_corres);
    }

    size_t num_images = images_.size();
    std::vector<image_t> image_ids;
    image_ids.reserve(num_images);
    image_t max_image_id = 0;
    for (const auto & image : images_) {
        image_ids.push_back(image.first);
        max_image_id = std::max(max_image_id, image.first);
    }
    std::sort(image_ids.begin(), image_ids.end());

    int step = 1;
    if (num_images > max_graph_node) {
        step = (num_images - 1) / max_graph_node + 1;
    }

    std::unordered_map<image_t, int> image_id_to_idx;
    std::vector<bool> selected(max_image_id + 1, false);
    int num_image_pick = 0;
    for (int i = 0; i < num_images; i += step) {
        selected[image_ids[i]] = true;
        image_id_to_idx[image_ids[i]] = num_image_pick++;
    }

    if (num_images > max_graph_node) {
        std::cout << StringPrintf("Compress Graph Node from %d to %d\n", num_images, num_image_pick);
    }

    Bitmap bitmap;
    bitmap.Allocate(num_image_pick, num_image_pick, true);

    const double max_value = std::log1p(max_val);
    const double bk_value = std::log1p(0) / max_value;
    const BitmapColor<float> bk_color(255 * JetColormap::Red(bk_value),
                                      255 * JetColormap::Green(bk_value),
                                      255 * JetColormap::Blue(bk_value));

    bitmap.Fill(bk_color.Cast<uint8_t>());
    for (auto & image_pair : image_pairs_) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, 
                                             &image_id1, 
                                             &image_id2);
        if (!selected[image_id1] || !selected[image_id2]) {
            continue;
        }
        int idx1 = image_id_to_idx.at(image_id1);
        int idx2 = image_id_to_idx.at(image_id2);

        size_t num_corres = NumCorrespondencesBetweenImages(image_pair.first);

        const double value = std::log1p(num_corres) / max_value;
        const BitmapColor<float> color(255 * JetColormap::Red(value),
                                        255 * JetColormap::Green(value),
                                        255 * JetColormap::Blue(value));
        bitmap.SetPixel(idx1, idx2, color.Cast<uint8_t>());
        bitmap.SetPixel(idx2, idx1, color.Cast<uint8_t>());
    }
    bitmap.Write(path);
}

 void CorrespondenceGraph::SetImageIndexMap(const std::unordered_map<image_t, image_t> image_index_map){
     image_index_map_ = image_index_map;
 }

 void CorrespondenceGraph::DeleteImage(image_t image_id){

    if (image_neighbors_.find(image_id) != image_neighbors_.end()) {
        std::unordered_set<image_t> neighbor_images = image_neighbors_.at(image_id);
        for (const auto neighbor_image : neighbor_images) {
            DeleteCorrespondences(image_id, neighbor_image);
        }
        image_neighbors_.erase(image_id);
    }
    // CHECK(images_.find(image_id)!=images_.end());
    if (images_.find(image_id) == images_.end()) {
        return;
    }
    images_.erase(image_id);
    Finalize();
}

void CorrespondenceGraph::Clear() {
    images_.clear();
    image_pairs_.clear();
    image_neighbors_.clear();
    image_index_map_.clear();
}

}  // namespace sensemap
