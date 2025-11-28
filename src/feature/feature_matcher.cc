// Copyright (c) 2019, SenseTime Group.
// All rights reserved.
#include "feature/feature_matcher.h"

#include "base/pose.h"
#include "base/track.h"
#include "base/projection.h"
#include "feature/utils.h"
#include "retrieval/visual_index.h"
#include "retrieval/vlad_visual_index.h"
#include "util/proc.h"
#include "util/cuda.h"
#include "util/image_pair.h"
#include "util/misc.h"
#include "util/gps_reader.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if CUDA_ENABLED
#include "SiftGPU/SiftGPU.h"
#endif

namespace sensemap {

int GPUPairsMatcherThread::obj_count = 0;

ExhaustiveMatchingOptions FeatureMatchingOptions::ExhaustiveMatching() const {
    ExhaustiveMatchingOptions options = exhaustive_matching_;
    return options;
}

SequentialMatchingOptions FeatureMatchingOptions::SequentialMatching() const {
    SequentialMatchingOptions options = sequential_matching_;
    return options;
}

VocabTreeMatchingOptions FeatureMatchingOptions::VocabTreeMatching() const {
    VocabTreeMatchingOptions options = vocabtree_matching_;
    return options;
}

SpatialMatchingOptions FeatureMatchingOptions::SpatialMatching() const {
    SpatialMatchingOptions options = spatial_matching_;
    return options;
}

HybridMatchingOptions FeatureMatchingOptions::HybridMatching() const {
    HybridMatchingOptions options = hybrid_matching_;
    return options;
}

ManualMatchingOptions FeatureMatchingOptions::ManualMatching() const {
    ManualMatchingOptions options = manual_matching_;
    return options;
}

PairMatchingOptions FeatureMatchingOptions::PairMatching() const {
    PairMatchingOptions options = pair_matching_;
    return options;
}

FeatureMatchingOptions::MatchMethod FeatureMatchingOptions::Method() const { return method_; }

bool ExhaustiveMatchingOptions::Check() const {
    CHECK_OPTION_GT(block_size, 1);
    return true;
}

bool SequentialMatchingOptions::Check() const {
    CHECK_OPTION_GT(overlap, 0);
    CHECK_OPTION_GT(loop_detection_period, 0);
    CHECK_OPTION_GT(loop_detection_num_images, 0);
    CHECK_OPTION_GT(loop_detection_num_nearest_neighbors, 0);
    CHECK_OPTION_GT(loop_detection_num_checks, 0);
    return true;
}

bool VocabTreeMatchingOptions::Check() const {
    CHECK_OPTION_GT(num_images, 0);
    CHECK_OPTION_GT(num_nearest_neighbors, 0);
    CHECK_OPTION_GT(num_checks, 0);
    return true;
}

bool SpatialMatchingOptions::Check() const {
    CHECK_OPTION_GT(max_num_neighbors, 0);
    CHECK_OPTION_GT(max_distance, 0.0);
    return true;
}

bool HybridMatchingOptions::Check() const {
    // TODO:

    return true;
}

bool ManualMatchingOptions::Check() const {
    // TODO:

    return true;
}

bool FeatureMatchingOptions::Check() const {
    return exhaustive_matching_.Check() && sequential_matching_.Check() && vocabtree_matching_.Check() &&
           spatial_matching_.Check() && pair_matching_.Check();
}

PairsGenerator::PairsGenerator(const FeatureMatchingOptions& options, FeatureDataContainer* container)
    : options_(options), feature_data_container_(container) {
    CHECK(options_.Check());
}

void PairsGenerator::Run() {
    image_pairs_.clear();
    // -- Disable the hybrid status as defult
    hybrid_status_ = HYBRID_STATUS::DISABLE;
    
    std::cout << "Generate pairs for new images" << std::endl;
    switch (options_.Method()) {
        case FeatureMatchingOptions::MatchMethod::EXHAUSTIVE:
            GeneratePairsExhaustively();
            break;
        case FeatureMatchingOptions::MatchMethod::SEQUENTIAL:
            GeneratePairsSequentially();
            break;
        case FeatureMatchingOptions::MatchMethod::VOCABTREE:
            GeneratePairsVocabTree();
            break;
        case FeatureMatchingOptions::MatchMethod::HYBRID:
            GeneratePairsHybridly();
            break;
        case FeatureMatchingOptions::MatchMethod::HYBRID_INPUT:
            GeneratePairsWithHybridInput();
            break;
        case FeatureMatchingOptions::MatchMethod::SPATIAL:
            GeneratePairsSpatially();
            break;
        case FeatureMatchingOptions::MatchMethod::MANUAL:
            GeneratePairsManually();
            break;
        default:
            break;
    }

    // Generate match pair by April Tag detection
    if (options_.detect_apriltag_) {
        //GeneratePairsByAprilTag();
    }

    // Vocabulary match between reconstruction
    if (options_.match_between_reconstructions_) {
        GeneratePairsBetweenReconVocabTree();
    }

    std::stable_sort(image_pairs_.begin(), image_pairs_.end(),
                     [](const std::pair<image_t, image_t>& p1, const std::pair<image_t, image_t>& p2) {
                         return p1.first < p2.first;
                     });
}

const std::vector<std::pair<image_t, image_t>>& PairsGenerator::GetPairs() { return image_pairs_; }
const std::vector<std::pair<image_t, image_t>>& PairsGenerator::GetLoopPairs() { return loop_image_pairs_; }

void PairsGenerator::GeneratePairsExhaustively(const std::vector<image_t>& input_image_id) {
    ExhaustiveMatchingOptions em_options = options_.ExhaustiveMatching();

    const std::vector<image_t> image_ids = hybrid_status_ ? input_image_id : feature_data_container_->GetNewImageIds();

    CHECK(!image_ids.empty()) << "The image id set in GeneratePairsExhaustively is empty!!!";

    const size_t block_size = static_cast<size_t>(em_options.block_size);
    const size_t num_blocks = static_cast<size_t>(std::ceil(static_cast<double>(image_ids.size()) / block_size));
    const size_t num_pairs_per_block = block_size * (block_size - 1) / 2;

    if (!hybrid_status_) {
        image_pairs_.clear();
    }

    for (size_t start_idx1 = 0; start_idx1 < image_ids.size(); start_idx1 += block_size) {
        const size_t end_idx1 = std::min(image_ids.size(), start_idx1 + block_size) - 1;

        for (size_t start_idx2 = 0; start_idx2 < image_ids.size(); start_idx2 += block_size) {
            const size_t end_idx2 = std::min(image_ids.size(), start_idx2 + block_size) - 1;

            for (size_t idx1 = start_idx1; idx1 <= end_idx1; ++idx1) {
                for (size_t idx2 = start_idx2; idx2 <= end_idx2; ++idx2) {
                    const size_t block_id1 = idx1 % block_size;
                    const size_t block_id2 = idx2 % block_size;
                    if ((idx1 > idx2 && block_id1 <= block_id2) || (idx1 < idx2 && block_id1 < block_id2)) {
                        if (image_ids[idx1] > image_ids[idx2]) {
                            image_pairs_.emplace_back(image_ids[idx2], image_ids[idx1]);
                        } else {
                            image_pairs_.emplace_back(image_ids[idx1], image_ids[idx2]);
                        }
                    }
                }
            }
        }
    }
}

void PairsGenerator::GeneratePairsSequentially(const std::vector<image_t>& input_image_id) {
    SequentialMatchingOptions sqm_options = options_.SequentialMatching();
    
    const std::vector<image_t> unorder_image_ids =
        hybrid_status_ ? input_image_id : feature_data_container_->GetNewImageIds();

    CHECK(!unorder_image_ids.empty()) << "The image id set in GeneratePairsSequentially is empty!!!";

    const std::vector<image_t> image_ids = GetOrderedImageIds(feature_data_container_, unorder_image_ids);

    std::unordered_map<image_t, image_t> image_orders;
    for (size_t order_idx = 0; order_idx < image_ids.size(); ++order_idx) {
        image_orders.emplace(image_ids[order_idx], order_idx);
    }

    if (!hybrid_status_) {
        image_pairs_.clear();
    }
    for (size_t image_idx1 = 0; image_idx1 < image_ids.size(); ++image_idx1) {
        const auto image_id1 = image_ids.at(image_idx1);
        for (int i = 0; i < sqm_options.overlap; ++i) {
            const size_t image_idx2 = image_idx1 + i + 1;

            if (image_idx2 < image_ids.size()) {
                const auto image_id2 = image_ids.at(image_idx2);
                if (image_id1 > image_id2) {
                    image_pairs_.emplace_back(image_id2, image_id1);
                } else {
                    image_pairs_.emplace_back(image_id1, image_id2);
                }
                if (sqm_options.quadratic_overlap) {
                    const size_t image_idx2_quadratic = image_idx1 + (1 << i);
                    if (image_idx2_quadratic < image_ids.size()) {
                        const auto image_id2_q = image_ids.at(image_idx2_quadratic);
                        if (image_id1 > image_id2_q) {
                            image_pairs_.emplace_back(image_id2_q, image_id1);
                        } else {
                            image_pairs_.emplace_back(image_id1, image_id2_q);
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    std::vector<image_t> loop_detection_image_ids;
    for (size_t i = 0; i < image_ids.size(); i += sqm_options.loop_detection_period) {
        loop_detection_image_ids.push_back(image_ids[i]);
    }
    std::vector<std::vector<image_t>> neighbors;

    // loop detection is required
    if (!sqm_options.robust_loop_detection && sqm_options.loop_detection) {
        if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::SIFT) {
            retrieval::VisualIndex<> visual_index;
            visual_index.Read(sqm_options.vocab_tree_path);

            IndexImagesInVisualIndex(options_.VocabTreeMatching().num_threads, sqm_options.loop_detection_num_checks,
                                    options_.VocabTreeMatching().vocab_tree_max_num_features, image_ids,
                                    feature_data_container_, &visual_index);
            FindNearestNeighborsInVisualIndex(
                options_.VocabTreeMatching().num_threads, sqm_options.loop_detection_num_images + sqm_options.overlap * 2,
                options_.VocabTreeMatching().num_nearest_neighbors, sqm_options.loop_detection_num_checks,
                sqm_options.loop_detection_num_images_after_verification,
                options_.VocabTreeMatching().vocab_tree_max_num_features, loop_detection_image_ids, feature_data_container_,
                &visual_index, neighbors, false, options_.VocabTreeMatching().max_score_factor);
        } else if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::VLAD) {
            VladVisualIndex vlad_visual_index;
            vlad_visual_index.LoadCodeBook(sqm_options.vlad_code_book_path);

            Timer timer;
            timer.Start();
            std::cout << "Load vlad vectors from data container to vlad visual index:" << std::endl;
            // Index all the image
            for (int i = 0; i < image_ids.size(); ++i) {
                image_t current_id = image_ids[i];
                const auto& vlad = feature_data_container_->GetVladVector(current_id);
                vlad_visual_index.ADD(vlad, current_id);
            }
            std::cout << StringPrintf("Load in %.3f seconds", timer.ElapsedSeconds()) << std::endl;
            FindNearestNeighborsInVladVisualIndex(sqm_options.loop_detection_num_images + sqm_options.overlap * 2,
                                                loop_detection_image_ids, feature_data_container_, &vlad_visual_index,
                                                neighbors, false, options_.VocabTreeMatching().max_score_factor);
        }
        CHECK_EQ(loop_detection_image_ids.size(), neighbors.size());

        for (size_t i = 0; i < neighbors.size(); ++i) {
            image_t image_id1 = loop_detection_image_ids[i];

            if (options_.have_prior_pose_) {
                CHECK(options_.prior_pose_validations.find(image_id1) != options_.prior_pose_validations.end());
                if (!options_.prior_pose_validations.at(image_id1)) {
                    std::cout << "invalid prior pose, skip the match" << std::endl;
                    continue;
                }
            }

            CHECK(image_orders.find(image_id1) != image_orders.end());
            image_t image_id1_order = image_orders.at(image_id1);

            size_t loop_image_count = 0;
            for (size_t j = 0; j < neighbors[i].size(); ++j) {
                image_t image_id2 = neighbors[i][j];
                if (options_.have_prior_pose_) {
                    CHECK(options_.prior_pose_validations.find(image_id2) != options_.prior_pose_validations.end());
                    if (!options_.prior_pose_validations.at(image_id2)) {
                        std::cout << "invalid prior pose, skip the match" << std::endl;
                        continue;
                    }
                }

                bool match_from_far_place = false;
                if (options_.have_prior_pose_) {
                    if (options_.prior_rotations.find(image_id1) != options_.prior_rotations.end() &&
                        options_.prior_rotations.find(image_id2) != options_.prior_rotations.end()) {
                        Eigen::Vector4d qvec1 = options_.prior_rotations.at(image_id1);
                        Eigen::Vector3d tvec1 = options_.prior_translations.at(image_id1);

                        Eigen::Vector4d qvec2 = options_.prior_rotations.at(image_id2);
                        Eigen::Vector3d tvec2 = options_.prior_translations.at(image_id2);

                        Eigen::Vector3d C1 = ProjectionCenterFromPose(qvec1, tvec1);
                        Eigen::Vector3d C2 = ProjectionCenterFromPose(qvec2, tvec2);

                        double distance = (C1 - C2).norm();
                        if (distance > options_.max_match_distance * options_.prior_neighbor_distance) {
                            match_from_far_place = true;
                        }
                    }
                }
                if (match_from_far_place) {
                    std::cout << "match from far place" << std::endl;
                    continue;
                }

                CHECK(image_orders.find(image_id2) != image_orders.end());
                image_t image_id2_order = image_orders.at(image_id2);

                if (fabs(static_cast<float>(image_id1_order) - static_cast<float>(image_id2_order)) <=
                    sqm_options.overlap) {
                    continue;
                }
                if (image_id1 > image_id2) {
                    image_pairs_.emplace_back(image_id2, image_id1);
                } else {
                    image_pairs_.emplace_back(image_id1, image_id2);
                }
                loop_image_count++;
                if (loop_image_count >= sqm_options.loop_detection_num_images) {
                    break;
                }
            }
        }
    } else if (!hybrid_status_ && sqm_options.robust_loop_detection &&
               sqm_options.loop_detection_before_sequential_matching) {
        GeneratePairsRobustLoopDetection();
    }

    // add prior matches
    if (options_.have_prior_image_pairs_) {
        std::cout << "ADD prior image pairs " << options_.prior_image_pairs_.size() << std::endl;
        for (auto image_pair : options_.prior_image_pairs_) {
            CHECK_LE(image_pair.first, image_ids.size());
            CHECK_LE(image_pair.second, image_ids.size());
            image_pairs_.push_back(image_pair);
        }
    }
}

void PairsGenerator::GeneratePairsVocabTree(const std::vector<image_t>& input_image_id) {
    VocabTreeMatchingOptions vtm_options = options_.VocabTreeMatching();

    const std::vector<image_t> image_ids =
        hybrid_status_ == HYBRID_STATUS::INSIDE_MATCHING ? input_image_id : feature_data_container_->GetNewImageIds();

    CHECK(!image_ids.empty()) << "The image id set in GeneratePairsVocabTree is empty!!!";

    if (hybrid_status_ == HYBRID_STATUS::DISABLE) {
        image_pairs_.clear();
    }

    // Find the nearest neighbors for each image and obtain the image pairs.
    std::vector<std::vector<image_t>> neighbors;

    if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::SIFT) {
        retrieval::VisualIndex<> visual_index;
        visual_index.Read(vtm_options.vocab_tree_path);
        Timer timer;
        timer.Start();
        // Index all the images
        IndexImagesInVisualIndex(vtm_options.num_threads, vtm_options.num_checks, vtm_options.vocab_tree_max_num_features,
                                image_ids, feature_data_container_, &visual_index);
        
        std::cout << StringPrintf("IndexImagesInVisualIndex in %.3f seconds", timer.ElapsedSeconds()) << std::endl;
        
        timer.Start();
        FindNearestNeighborsInVisualIndex(
        vtm_options.num_threads, vtm_options.num_images * 1.4, vtm_options.num_nearest_neighbors, vtm_options.num_checks,
        vtm_options.num_images_after_verification, vtm_options.vocab_tree_max_num_features, image_ids,
        feature_data_container_, &visual_index, neighbors, hybrid_status_ == HYBRID_STATUS::OUTSIDE_MATCHING,
        options_.VocabTreeMatching().max_score_factor);

        std::cout << StringPrintf("FindNearestNeighborsInVisualIndex in %.3f seconds", timer.ElapsedSeconds()) << std::endl;

    } else if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::VLAD) {
        VladVisualIndex vlad_visual_index;
        vlad_visual_index.LoadCodeBook(vtm_options.vlad_code_book_path);
        Timer timer;
        timer.Start();
        std::cout<<"Load vlad vectors from data container to vlad visual index:"<<std::endl;
        // Index all the images
        for(int i = 0; i < image_ids.size(); ++i){    
            image_t current_id = image_ids[i];        
            const auto& vlad = feature_data_container_->GetVladVector(current_id);
            vlad_visual_index.ADD(vlad,current_id);
        }
        std::cout << StringPrintf("Load in %.3f seconds", timer.ElapsedSeconds()) << std::endl;
        FindNearestNeighborsInVladVisualIndex(
            vtm_options.num_images, image_ids, feature_data_container_, &vlad_visual_index, neighbors,
            hybrid_status_ == HYBRID_STATUS::OUTSIDE_MATCHING, vtm_options.max_score_factor);
    }

    CHECK_EQ(image_ids.size(), neighbors.size());

    auto ValidateMatchPairWithPrior = [&](const size_t i, const double max_match_distance) {
        int num_match_i = 0;
        int num_match_from_near = 0;
        int num_match_from_far = 0;
        std::vector<std::pair<image_t,image_t>> valid_image_pairs;
        const float rtk_loop_factor = 0.1;

        image_t image_id1 = image_ids[i];
        auto image1 = feature_data_container_->GetImage(image_id1);
        for (size_t j = 0; j < neighbors[i].size(); ++j) {
            image_t image_id2 = neighbors[i][j];
            auto image2 = feature_data_container_->GetImage(image_id2);

            if (options_.have_prior_pose_) {
                CHECK(options_.prior_pose_validations.find(image_id2) != options_.prior_pose_validations.end());
                if (!options_.prior_pose_validations.at(image_id2)) {
                    std::cout << "invalid prior pose, skip the match" << std::endl;
                    continue;
                }
            }

            bool match_from_far_place = false;
            if (options_.have_prior_pose_) {
                if (options_.prior_rotations.find(image_id1) != options_.prior_rotations.end() &&
                    options_.prior_rotations.find(image_id2) != options_.prior_rotations.end()) {
                    Eigen::Vector4d qvec1 = options_.prior_rotations.at(image_id1);
                    Eigen::Vector3d tvec1 = options_.prior_translations.at(image_id1);

                    Eigen::Vector4d qvec2 = options_.prior_rotations.at(image_id2);
                    Eigen::Vector3d tvec2 = options_.prior_translations.at(image_id2);

                    Eigen::Vector3d C1 = ProjectionCenterFromPose(qvec1, tvec1);
                    Eigen::Vector3d C2 = ProjectionCenterFromPose(qvec2, tvec2);

                    double distance = (C1 - C2).norm();
                    if (distance > options_.max_match_distance * options_.prior_neighbor_distance) {
                        match_from_far_place = true;
                    }
                } else if (options_.prior_translations.find(image_id1) != options_.prior_translations.end() &&
                           options_.prior_translations.find(image_id2) != options_.prior_translations.end()) {
                    Eigen::Vector3d C1 = options_.prior_translations.at(image_id1);
                    Eigen::Vector3d C2 = options_.prior_translations.at(image_id2);
                    double distance = (C1 - C2).norm();
                    if (distance > max_match_distance) {
                        match_from_far_place = true;
                    }
                }
            } else if (image1.HasTvecPrior() && image2.HasTvecPrior()) {
                double distance = (image1.TvecPrior() - image2.TvecPrior()).norm();
                if (distance > max_match_distance) {
                    match_from_far_place = true;
                }
            }

            if (!match_from_far_place && num_match_i < vtm_options.num_images) {
                // std::cout << "match from far place" << std::endl;
                if (num_match_i < vtm_options.num_images * (1.0f - rtk_loop_factor)){
                    if (image_id1 > image_id2) {
                        image_pairs_.emplace_back(image_id2, image_id1);
                    } else {
                        image_pairs_.emplace_back(image_id1, image_id2);
                    }
                    num_match_from_near++;
                } else {
                    if (image_id1 > image_id2) {
                        valid_image_pairs.emplace_back(image_id2, image_id1);
                    } else {
                        valid_image_pairs.emplace_back(image_id1, image_id2);
                    }
                }
                num_match_i++;
                continue;
            }

            bool match_weak_overlap = false;
            if (image1.HasQvecPrior() && image2.HasQvecPrior()){
                Eigen::Vector4d prior_qvec1 = image1.QvecPrior();
                Eigen::Vector4d prior_qvec2 = image2.QvecPrior();
                Eigen::Vector4d prior_qvec12 = ConcatenateQuaternions(prior_qvec1, InvertQuaternion(prior_qvec2));

                double delt_theat = 2 * std::acos(prior_qvec12[0]);
                if (delt_theat > M_PI / 4){
                    match_weak_overlap = true;
                } else {
                    Eigen::Vector3d prior_tvec1 = QuaternionToRotationMatrix(prior_qvec1) * -image1.TvecPrior();
                    Eigen::Vector3d prior_tvec2 = QuaternionToRotationMatrix(prior_qvec2) * -image2.TvecPrior();

                    Eigen::Vector2d pnt(-1, -1), view(0,0);
                    const auto proj1 = ComposeProjectionMatrix(prior_qvec1, prior_tvec1);
                    const auto proj2 = ComposeProjectionMatrix(prior_qvec2, prior_tvec2);
                    if (HasPointPositiveDepth(proj1, prior_tvec2)){
                        const auto& camera1 = feature_data_container_->GetCamera(image1.CameraId());
                        pnt = ProjectPointToImage(image2.TvecPrior(), proj1, camera1);
                        view.x() = camera1.Width();
                        view.y() = camera1.Height();
                    } else if (HasPointPositiveDepth(proj2, prior_tvec1)) {
                        const auto& camera2 = feature_data_container_->GetCamera(image2.CameraId());
                        pnt = ProjectPointToImage(image1.TvecPrior(), proj2, camera2);
                        view.x() = camera2.Width();
                        view.y() = camera2.Height();
                    }

                    if (pnt.x() < 0 || pnt.x() > view.x() || pnt.y() < 0 || pnt.y() > view.y()){
                        match_weak_overlap = true;
                    } else {
                        std::cout << "[ValidateRtkLoopPair]: " << image1.Name() << " - " 
                                << image2.Name() << "(" << match_weak_overlap << ")"
                                << ": delat theat=" << delt_theat << "(rad), " 
                                << pnt.transpose() << " / " << view.transpose() << std::endl;
                    }
                }
            }

            if (!match_weak_overlap && num_match_from_far < vtm_options.num_images * rtk_loop_factor) {
                // std::cout << "match from far place" << std::endl;
                if (image_id1 > image_id2) {
                    image_pairs_.emplace_back(image_id2, image_id1);
                } else {
                    image_pairs_.emplace_back(image_id1, image_id2);
                }
                num_match_from_far++;
                continue;
            }
            
            if (num_match_i > vtm_options.num_images && num_match_from_far > vtm_options.num_images * rtk_loop_factor){
                break;
            }
        }
        
        const auto delt_num = std::min(size_t(vtm_options.num_images - num_match_from_near - num_match_from_far), valid_image_pairs.size());
        for (size_t idx = 0; idx < delt_num; idx++){
            image_pairs_.emplace_back(valid_image_pairs.at(idx));
            num_match_from_near++;
        }

        std::cout << "[ValidateMatchPairWithPrior]: " << image1.Name() << ": " <<  num_match_from_near << ", " << num_match_from_far << std::endl;

        return num_match_from_near + num_match_from_far;
    };

    for (size_t i = 0; i < image_ids.size(); ++i) {
        image_t image_id1 = image_ids[i];

        if (options_.have_prior_pose_) {
            CHECK(options_.prior_pose_validations.find(image_id1) != options_.prior_pose_validations.end());
            if (!options_.prior_pose_validations.at(image_id1)) {
                std::cout << "invalid prior pose, skip the match" << std::endl;
                continue;
            }
        }

        double max_match_distance = options_.max_match_distance;

        auto image = feature_data_container_->GetImage(image_id1);
        if (image.HasRelativeAltitude()) {
            max_match_distance = std::max(max_match_distance, int((image.RelativeAltitude() + 50 - 1) / 50) * 50.0);
        }

        std::cout << StringPrintf("max_match_distance#%d: %f\n", image_id1, max_match_distance);

        int iter = 0;
        while(iter++ < 3) {
            int num_match = ValidateMatchPairWithPrior(i, max_match_distance);
            if (num_match > 0) break;
            max_match_distance = options_.max_match_distance * iter;
        }
    }
}

void PairsGenerator::GeneratePairsRobustLoopDetection(const std::vector<image_t>& input_image_id) {

    SequentialMatchingOptions sqm_options = options_.SequentialMatching();
    std::vector<image_t> image_ids =
        GetOrderedImageIds(feature_data_container_, feature_data_container_->GetNewImageIds());

    CHECK(!image_ids.empty()) << "The image id set in GeneratePairsRobustLoopDetection is empty!!!";

    if (image_ids.size() <= sqm_options.overlap) {
        return;
    }

    Timer timer;

    std::vector<std::vector<retrieval::ImageScore>> image_image_scores(image_ids.size());

    if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::VLAD) {
        VladVisualIndex vlad_visual_index;
        vlad_visual_index.LoadCodeBook(sqm_options.vlad_code_book_path);
        
        timer.Start();
        std::cout<<"Load vlad vectors from data container to vlad visual index:"<<std::endl;
        //Load vlad vectors from data container to vlad visual index
        for(int i = 0; i < image_ids.size(); ++i){
            image_t current_id = image_ids[i];        
            const auto& vlad = feature_data_container_->GetVladVector(current_id);
            vlad_visual_index.ADD(vlad,current_id);
        }
        std::cout << StringPrintf("Indexing in %.3f seconds", timer.ElapsedSeconds()) << std::endl;

        VladVisualIndex::QueryOptions query_option;
        query_option.max_num_images = sqm_options.loop_detection_num_images * 10;
        
        timer.Start();
        std::cout << "Query " << std::endl;
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            const Image& image = feature_data_container_->GetImage(current_id);
            /// query by voc tree
            std::cout<<"query ["<<i<<"/"<<image_ids.size()<<"] image"<<std::endl;
            std::vector<retrieval::ImageScore> image_scores;
            const VladVisualIndex::VLAD& current_vlad = feature_data_container_->GetVladVector(current_id); 
            vlad_visual_index.Query(query_option,current_vlad,&image_scores);

            std::vector<retrieval::ImageScore> hybrid_image_scores(sqm_options.loop_detection_num_images);
            size_t retrieved_num = 0;

            // loop cross the sequence
            for (size_t j = 1; j < image_scores.size(); ++j) {
                const Image& image_neighbor = feature_data_container_->GetImage(image_scores[j].image_id);
                if (image_neighbor.LabelId() != image.LabelId()) {
                    hybrid_image_scores[retrieved_num++] = image_scores[j];
                }
                if (retrieved_num >= sqm_options.loop_detection_num_images * 2 / 5) {
                    break;
                }
            }

            // loop in the same sequence
            for (size_t j = 1; j < image_scores.size(); ++j) {
                const Image& image_neighbor = feature_data_container_->GetImage(image_scores[j].image_id);
                if (image_neighbor.LabelId() == image.LabelId()) {
                    hybrid_image_scores[retrieved_num++] = image_scores[j];
                }
                if (retrieved_num >= sqm_options.loop_detection_num_images) {
                    break;
                }
            }
            hybrid_image_scores.resize(retrieved_num);

            { image_image_scores[i].swap(hybrid_image_scores); }
        }

        std::cout << StringPrintf("Query in %.3fmin", timer.ElapsedMinutes()) << std::endl;
    } else if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::SIFT) {
        timer.Start();

        int max_num_features = sqm_options.loop_detection_max_num_features;

        retrieval::VisualIndex<> visual_index;
        visual_index.Read(sqm_options.vocab_tree_path);

        IndexImagesInVisualIndex(sqm_options.loop_detection_num_threads, sqm_options.loop_detection_num_checks,
                                 sqm_options.loop_detection_max_num_features, image_ids, feature_data_container_,
                                 &visual_index);

        std::cout << StringPrintf("Indexing in %.3f min", timer.ElapsedMinutes()) << std::endl;

        //--- need to test
        retrieval::VisualIndex<>::QueryOptions query_options;
        query_options.max_num_images = sqm_options.overlap * 5;
        query_options.num_neighbors = 3;
        query_options.num_images_after_verification = 0;  // query_options.max_num_images;

        timer.Start();
        std::cout << "Query " << std::endl;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            /// query by voc tree

            std::vector<retrieval::ImageScore> image_scores;
            auto keypoints = feature_data_container_->GetKeypoints(current_id);
            auto descriptors = feature_data_container_->GetCompressedDescriptors(current_id);
            if (max_num_features > 0 && descriptors.rows() > max_num_features) {
                ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
            }

            visual_index.Query(query_options, keypoints, descriptors, &image_scores);
            image_image_scores[i].swap(image_scores);
        }
        std::cout << StringPrintf("Query in %.3fmin", timer.ElapsedMinutes()) << std::endl;
    }
    
    std::unordered_set<image_pair_t> existed_loop_pairs;


    // find candidates in valid loop
    int num_connected_images = sqm_options.overlap;
    typedef std::pair<std::set<image_t>, int> ConsistentGroup;
    std::vector<ConsistentGroup> consistent_groups;

    for (int i = num_connected_images; i < image_ids.size(); ++i) {
        image_t current_id = image_ids[i];

        std::cout << "Frame#" << current_id << std::endl;

        std::vector<image_t> recent_ids;
        recent_ids.reserve(num_connected_images*2 + 1);
        for (int j = -num_connected_images; j <= num_connected_images; ++j) {
            if((i + j >= 0) && (i + j < image_ids.size()) && (j!=0)){
                recent_ids.push_back(image_ids[i + j]);
            }
        }

        image_t max_loop_id = image_ids[i - num_connected_images];

        /// query by voc tree
        std::list<retrieval::ImageScore> candidate_scores;

        /// get ids in searching area
        float max_recent_score = -1;
        int max_recent_id = -1;
        for (auto& image_score : image_image_scores[i]) {
            image_t image_id = image_score.image_id;
            if (image_id >= current_id) {
                continue;
            } else if (image_id <= max_loop_id) {  // is candidate
                candidate_scores.push_back(image_score);
            } else if (std::find(recent_ids.begin(), recent_ids.end(), image_id) !=
                       recent_ids.end()) {  // is recent image, compute max_recent_score
                if (image_score.score > max_recent_score) {
                    max_recent_score = image_score.score;
                    max_recent_id = image_id;
                }
            }
        }

        std::cout << "  => Query candidate num: " << candidate_scores.size() << std::endl;
        if (candidate_scores.empty()) {
            consistent_groups.clear();
            continue;
        }
        std::cout<<"max recent score: "<<max_recent_score<<std::endl;
        std::cout<<"max recent id: "<<max_recent_id<<std::endl;
        // filter by max_recent_score * max_recent_score_factor
        float min_score = max_recent_score * sqm_options.max_recent_score_factor;
        candidate_scores.remove_if([&](retrieval::ImageScore s) { return s.score < min_score; });

        std::cout << "  => Filtered candidate nums " << candidate_scores.size() << std::endl;
        if (candidate_scores.empty()) {
            consistent_groups.clear();
            continue;
        }

        // filter by best_acc_score * best_acc_score_factor
        std::unordered_set<image_t> connected_candidates;
        std::list<std::pair<float, image_t>> acc_score_image;
        float best_acc_score = min_score;
        for (auto ii = candidate_scores.begin(); ii != candidate_scores.end(); ++ii) {
            image_t image_i = ii->image_id;

            label_t label_i = feature_data_container_->GetImage(image_i).LabelId();
            auto neighbors_i =
                feature_data_container_->GetSequentialNeighbor(image_i, label_i, num_connected_images);

            float best_score = ii->score;
            float acc_score = ii->score;
            image_t best_id = image_i;
            for (auto ij = candidate_scores.begin(); ij != candidate_scores.end(); ++ij) {
                if (ii == ij) {
                    continue;
                }
                if (neighbors_i.find(ij->image_id) != neighbors_i.end()) {
                    acc_score += ij->score;
                    if (ij->score > best_score) {
                        best_score = ij->score;
                        best_id = ij->image_id;
                    }
                }
            }
            acc_score_image.emplace_back(acc_score, best_id);
            if (acc_score > best_acc_score) {
                best_acc_score = acc_score;
            }
        }
        // float min_retained_score = best_acc_score * 0.75;
        float min_retained_score = best_acc_score * sqm_options.best_acc_score_factor;
        for (auto& asi : acc_score_image) {
            if (asi.first > min_retained_score) {
                connected_candidates.insert(asi.second);
            }
        }

        std::cout << "  => Connected candidate nums " << connected_candidates.size() << std::endl;
        if (connected_candidates.empty()) {
            consistent_groups.clear();
            continue;
        }

        /// consistency check
        std::vector<image_t> consistent_candidates;
        std::vector<ConsistentGroup> current_consistent_group;
        std::vector<bool> consistnet_group_flags(consistent_groups.size(), false);

        for (auto& candidate_id : connected_candidates) {
            label_t candidate_label = feature_data_container_->GetImage(candidate_id).LabelId();

            auto neighbors_j =
                feature_data_container_->GetSequentialNeighbor(candidate_id, candidate_label, num_connected_images);
            std::set<image_t> candidate_group;
            for (auto& neighbor : neighbors_j) {
                if (neighbor < candidate_id) {
                    candidate_group.insert(neighbor);
                }
            }
            candidate_group.insert(candidate_id);

            bool enough_consistent = false;
            bool consistent_for_some_group = false;

            for (int j = 0; j < consistent_groups.size(); ++j) {
                auto& prev_group = consistent_groups[j].first;

                bool consistent = false;
                for (auto& id_in_group : candidate_group) {
                    if (prev_group.count(id_in_group)) {
                        consistent = true;
                        consistent_for_some_group = true;
                        break;
                    }
                }

                if (consistent) {
                    int prev_consistency = consistent_groups[j].second;
                    int curr_consistency = prev_consistency + 1;
                    if (!consistnet_group_flags[j]) {
                        current_consistent_group.emplace_back(candidate_group, curr_consistency);
                        consistnet_group_flags[j] = true;
                    }
                    if (curr_consistency >= sqm_options.loop_consistency_threshold && !enough_consistent) {
                        consistent_candidates.push_back(candidate_id);
                        enough_consistent = true;
                    }
                }
            }
            if (!consistent_for_some_group) {
                current_consistent_group.emplace_back(candidate_group, 1);
            }
        }

        consistent_groups = current_consistent_group;

        std::cout << "  => Consistent candidate nums " << consistent_candidates.size() << std::endl;
        if (consistent_candidates.empty()) {
            continue;
        }
    
        loop_image_pairs_.reserve(loop_image_pairs_.size() + consistent_candidates.size());
        image_pairs_.reserve(image_pairs_.size() + consistent_candidates.size());
        for (auto& candidate_id : consistent_candidates) {

            image_pair_t image_pair_id = sensemap::utility::ImagePairToPairId(candidate_id, current_id);
            if(existed_loop_pairs.count(image_pair_id)>0){
                continue;
            }
            existed_loop_pairs.insert(image_pair_id);

            if (current_id > candidate_id) {
                image_pairs_.emplace_back(candidate_id, current_id);
                loop_image_pairs_.emplace_back(candidate_id, current_id);
            } else {
                image_pairs_.emplace_back(current_id, candidate_id);
                loop_image_pairs_.emplace_back(current_id, candidate_id);
            }
        }
    }
}
void PairsGenerator::GeneratePairsWithHybridInput(){
    std::map<int, std::vector<image_t> > hybrid_image_ids;

    const std::vector<image_t> image_ids =
            GetOrderedImageIds(feature_data_container_, feature_data_container_->GetNewImageIds());
    for (auto image_id : image_ids) {
        auto camera_id = feature_data_container_->GetImage(image_id).CameraId();
        auto model_id = feature_data_container_->GetCamera(camera_id).ModelId();
        if(hybrid_image_ids.count(model_id) == 0){
            std::vector<image_t> new_list {image_id};
            hybrid_image_ids.insert(std::make_pair(model_id, new_list));
        } else {
            hybrid_image_ids[model_id].push_back(image_id);
        }
    }

    hybrid_status_ = HYBRID_STATUS::INSIDE_MATCHING;
    for(auto sub_image_ids : hybrid_image_ids){
        GeneratePairsVocabTree(sub_image_ids.second);

        std::cout<<sub_image_ids.first<<std::endl;
    }


    for(auto it1 = hybrid_image_ids.begin(); it1 != hybrid_image_ids.end(); ++it1){
        auto it2 = it1;
        it2++;
        if(it2 == hybrid_image_ids.end())
            continue;
        std::cout<< it1->first <<" & "<<it2->first<<std::endl;

        for (auto idx1 : it1->second) {
            for (auto idx2 : it2->second) {

                if (idx1 > idx2) {
                    image_pairs_.emplace_back(idx2, idx1);
                } else {
                    image_pairs_.emplace_back(idx1, idx2);
                }
            }
        }
    }

}

void PairsGenerator::GeneratePairsHybridly() {
    // -- Reset the image pairs
    image_pairs_.clear();

    // -- Enable the hybrid inside cluster matching;
    hybrid_status_ = HYBRID_STATUS::INSIDE_MATCHING;

    HybridMatchingOptions hm_options = options_.HybridMatching();

    ////////////////////////////////////////////////////////////////////////////
    // Generate cluster using image label
    ////////////////////////////////////////////////////////////////////////////

    // Check the image index and image id
    const std::vector<image_t> image_ids =
        GetOrderedImageIds(feature_data_container_, feature_data_container_->GetNewImageIds());
    size_t image_index = 0;
    for (auto image_id : image_ids) {
        if (image_id != image_index + 1) {
            std::cout << "Image index : " << image_index + 1 << " Image id : " << image_id
                      << " Image Name: " << feature_data_container_->GetImage(image_id).Name() << std::endl;
        }
        image_index++;
    }

    LabelledImagePtrUmap labelled_image_clusters;

    for (auto image_id : image_ids) {
        // Check the image has lable or not
        auto image = feature_data_container_->GetImage(image_id);

        if (!image.HasLabel()) {  // -- If image do not has a label
            std::cout << "The current image does not have label !" << std::endl;
            std::cout << "The current image id = " << image_id << " , " << image.LabelId() << std::endl;
            continue;
        } else {  // -- If the image contain a label
            // Get the image lable and insert in the labled image map
            labelled_image_clusters[image.LabelId()].emplace_back(image_id);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Using the selected method to generate matching pairs inside the cluster
    ////////////////////////////////////////////////////////////////////////////

    for (auto labelled_image_cluster : labelled_image_clusters) {
        CHECK(!labelled_image_cluster.second.empty()) << "The current cluster is empty!!!";

        if (!labelled_image_cluster.second.empty()) {
            switch (hm_options.method_inside_cluster) {
                case FeatureMatchingOptions::MatchMethod::EXHAUSTIVE:
                    GeneratePairsExhaustively(labelled_image_cluster.second);
                    break;
                case FeatureMatchingOptions::MatchMethod::SEQUENTIAL:
                    GeneratePairsSequentially(labelled_image_cluster.second);
                    break;
                case FeatureMatchingOptions::MatchMethod::VOCABTREE:
                    GeneratePairsVocabTree(labelled_image_cluster.second);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Using the vocabulary tree method to generate matching pairs outside the
    // cluster
    ////////////////////////////////////////////////////////////////////////////

    // -- Enable the hybrid outside cluster matching;
    hybrid_status_ = HYBRID_STATUS::OUTSIDE_MATCHING;
    if (!options_.sequential_matching_.robust_loop_detection) {
        GeneratePairsVocabTree();
    }
    else if(options_.sequential_matching_.loop_detection_before_sequential_matching){
        GeneratePairsRobustLoopDetection();
    }

    // add prior matches
    // if (options_.have_prior_image_pairs_) {
    //     std::cout << "ADD prior image pairs " << options_.prior_image_pairs_.size() << std::endl;
    //     for (auto image_pair : options_.prior_image_pairs_) {
    //         CHECK_LE(image_pair.first, image_ids.size());
    //         CHECK_LE(image_pair.second, image_ids.size());
    //         image_pairs_.push_back(image_pair);
    //     }
    // }
}

void PairsGenerator::GeneratePairsSpatially(const std::vector<image_t>& input_image_id) {


    PrintHeading1("Spatial feature matching");


    auto  sp_options = options_.SpatialMatching();

    const std::vector<image_t> image_ids = hybrid_status_ ? input_image_id : feature_data_container_->GetNewImageIds();

    CHECK(!image_ids.empty()) << "The current input image id set is empty!!!";

    if (!hybrid_status_) {
        image_pairs_.clear();
    }



    //////////////////////////////////////////////////////////////////////////////
    // Spatial indexing
    //////////////////////////////////////////////////////////////////////////////

    Timer timer;
    timer.Start();

    std::cout << "Indexing images..." << std::flush;

    size_t num_locations = 0;
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix(
            image_ids.size(), 3);

    std::vector<size_t> location_idxs;
    location_idxs.reserve(image_ids.size());

    std::vector<Eigen::Vector3d> ells(1);

    for (size_t i = 0; i < image_ids.size(); ++i) {
        const auto image_id = image_ids[i];
        const auto& image = feature_data_container_->GetImage(image_id);

        //if ((image.TvecPrior(0) == 0 && image.TvecPrior(1) == 0 &&
        //     sp_options.ignore_z) ||
        //    (image.TvecPrior(0) == 0 && image.TvecPrior(1) == 0 &&
        //     image.TvecPrior(2) == 0 && !sp_options.ignore_z)) {
        //    continue;
        //}

        location_idxs.push_back(i);

        if (sp_options.is_gps) {

            location_matrix(num_locations, 0) =
                    static_cast<float>(image.TvecPrior(0));
            location_matrix(num_locations, 1) =
                    static_cast<float>(image.TvecPrior(1));
            location_matrix(num_locations, 2) =
                    static_cast<float>(sp_options.ignore_z ? 0 : image.TvecPrior(2));
        } else {
            Eigen::Vector3d prior_tvec = QuaternionToRotationMatrix(image.QvecPrior()) * -image.TvecPrior();

            location_matrix(num_locations, 0) = static_cast<float>(prior_tvec[0]);
            location_matrix(num_locations, 1) = static_cast<float>(prior_tvec[1]);
            location_matrix(num_locations, 2) = static_cast<float>(sp_options.ignore_z ? 0 : prior_tvec[2]);
        }

        num_locations += 1;
    }


    if (num_locations == 0) {
        std::cout << " => No images with location data." << std::endl;
        return;
    }

    //////////////////////////////////////////////////////////////////////////////
    // Building spatial index
    //////////////////////////////////////////////////////////////////////////////

    timer.Restart();

    std::cout << "Building search index..." << std::flush;

    flann::Matrix<float> locations(location_matrix.data(), num_locations,
                                   location_matrix.cols());

    flann::LinearIndexParams index_params;
    flann::LinearIndex<flann::L2<float>> search_index(index_params);
    search_index.buildIndex(locations);


    //////////////////////////////////////////////////////////////////////////////
    // Searching spatial index
    //////////////////////////////////////////////////////////////////////////////

    timer.Restart();

    std::cout << "Searching for nearest neighbors..." << std::flush;

    const int knn = std::min<int>(sp_options.max_num_neighbors, num_locations);

    Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            index_matrix(num_locations, knn);
    flann::Matrix<size_t> indices(index_matrix.data(), num_locations, knn);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            distance_matrix(num_locations, knn);
    flann::Matrix<float> distances(distance_matrix.data(), num_locations, knn);

    flann::SearchParams search_params(flann::FLANN_CHECKS_AUTOTUNED);
    if (sp_options.num_threads == ThreadPool::kMaxNumThreads) {
        search_params.cores = std::thread::hardware_concurrency();
    } else {
        search_params.cores = sp_options.num_threads;
    }
    if (search_params.cores <= 0) {
        search_params.cores = 1;
    }

    search_index.knnSearch(locations, indices, distances, knn, search_params);


    //////////////////////////////////////////////////////////////////////////////
    // Matching
    //////////////////////////////////////////////////////////////////////////////

    const float max_distance =
            static_cast<float>(sp_options.max_distance * sp_options.max_distance);

    std::vector<std::pair<image_t, image_t>> image_pairs;
    image_pairs.reserve(knn);

    for (size_t i = 0; i < num_locations; ++i) {

        timer.Restart();

        image_pairs.clear();

        for (int j = 0; j < knn; ++j) {
            // Check if query equals result.
            if (index_matrix(i, j) == i) {
                continue;
            }

            // Since the nearest neighbors are sorted by distance, we can break.
            if (distance_matrix(i, j) > max_distance) {
                break;
            }

            const size_t idx = location_idxs[i];
            const image_t image_id = image_ids.at(idx);
            const size_t nn_idx = location_idxs.at(index_matrix(i, j));
            const image_t nn_image_id = image_ids.at(nn_idx);

            if(image_id < nn_image_id) {
                image_pairs_.emplace_back(image_id, nn_image_id);
            } else {
                image_pairs_.emplace_back(nn_image_id, image_id);
            }
        }
    }

}

void PairsGenerator::GeneratePairsManually() {
    ManualMatchingOptions man_options = options_.ManualMatching();

    // Check the image index and image id
    const std::vector<image_t> image_ids =
        GetOrderedImageIds(feature_data_container_, feature_data_container_->GetNewImageIds());

    std::unordered_set<image_pair_t> image_pairs;
    size_t image_index = 0;
    for (auto image_id : image_ids) {
        for (auto pair_image_id : man_options.generate_pairs_for_image(image_id, *feature_data_container_)) {
            image_pairs.insert(utility::ImagePairToPairId(image_id, pair_image_id));
        }
    }

    image_pairs_.clear();
    for (auto pair : image_pairs) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(pair, &image_id1, &image_id2);
        image_pairs_.emplace_back(image_id1, image_id2);
    }
}

void PairsGenerator::GeneratePairsByAprilTag() {
    // Generate AprilTag Matching pairs
    std::unordered_map<int, std::vector<image_t>> detected_id_map;
    const std::vector<image_t> image_ids = feature_data_container_->GetNewImageIds();
    for (const auto& image_id : image_ids) {
        auto cur_detections = feature_data_container_->GetAprilTagDetections(image_id);
        for (const auto& cur_detection : cur_detections) {
            detected_id_map[cur_detection.id].emplace_back(image_id);
        }
    }

    // For each pair images which has the same AprilTag id may have match
    int counter = 0;
    for (const auto& detected_id_pair : detected_id_map) {
        for (int i = 0; i < detected_id_pair.second.size(); i++) {
            for (int j = 0; j < detected_id_pair.second.size(); j++) {
                if (i == j || i > j) {
                    continue;
                }
                // Add match pair
                image_pairs_.push_back({detected_id_pair.second[i], detected_id_pair.second[j]});
                counter++;
            }
        }
    }
    std::cout << "AprilTag Add Match number = " << counter << std::endl;
}

void PairsGenerator::GeneratePairsBetweenReconVocabTree() {
    std::cout << "Generate Pairs Between Reconstructions ..." << std::endl;
    VocabTreeMatchingOptions vtm_options = options_.VocabTreeMatching();

    const std::vector<image_t> old_image_ids = feature_data_container_->GetOldImageIds();
    const std::vector<image_t> new_image_ids = feature_data_container_->GetNewImageIds();
    const std::vector<image_t> all_image_ids = feature_data_container_->GetImageIds();
    const std::vector<image_t> old_image_with_pose_ids = feature_data_container_->GetOldPoseImageIds();

    CHECK(!new_image_ids.empty() && !old_image_ids.empty())
        << "The current Exhaustively input image id set is empty!!!";
    
    // Find the nearest neighbors for each image and obtain the image pairs.
    std::vector<std::vector<image_t>> neighbors;

    if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::SIFT) {
        retrieval::VisualIndex<> visual_index;
        visual_index.Read(vtm_options.vocab_tree_path);
        // Index all the images
        IndexImagesInVisualIndex(vtm_options.num_threads, vtm_options.num_checks, vtm_options.vocab_tree_max_num_features,
                                old_image_ids, feature_data_container_, &visual_index);

        FindNearestNeighborsInVisualIndex(
            vtm_options.num_threads, vtm_options.num_images, vtm_options.num_nearest_neighbors, vtm_options.num_checks,
            vtm_options.num_images_after_verification, vtm_options.vocab_tree_max_num_features, new_image_ids,
            feature_data_container_, &visual_index, neighbors, true, vtm_options.max_score_factor);

    } else if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::VLAD) {
        VladVisualIndex vlad_visual_index;
        vlad_visual_index.LoadCodeBook(vtm_options.vlad_code_book_path);
        
        Timer timer;
        timer.Start();
        std::cout<<"Load vlad vectors from data container to vlad visual index:"<<std::endl;
        //Index all the image
        for(int i = 0; i < old_image_ids.size(); ++i){    
            image_t current_id = old_image_ids[i];        
            const auto& vlad = feature_data_container_->GetVladVector(current_id);
            vlad_visual_index.ADD(vlad,current_id);
        }
        std::cout << StringPrintf("Load in %.3f seconds", timer.ElapsedSeconds()) << std::endl;

        FindNearestNeighborsInVladVisualIndex(
            vtm_options.num_images, new_image_ids, feature_data_container_, &vlad_visual_index, neighbors,
            false, vtm_options.max_score_factor);
    }

    CHECK_EQ(new_image_ids.size(), neighbors.size());

    for (size_t i = 0; i < new_image_ids.size(); ++i) {
        image_t image_id1 = new_image_ids[i];
        for (size_t j = 0; j < neighbors[i].size(); ++j) {
            image_t image_id2 = neighbors[i][j];
            if (image_id1 > image_id2) {
                image_pairs_.emplace_back(image_id2, image_id1);
            } else {
                image_pairs_.emplace_back(image_id1, image_id2);
            }
        }
    }
}

PairsMatcherThread::PairsMatcherThread(const PairMatchingOptions& options, FeatureDataContainer* container)
    : options_(options), feature_data_container_(container) {}

void PairsMatcherThread::SetMaxNumMatches(const int max_num_matches) { options_.max_num_matches = max_num_matches; }

CPUPairsMatcherThread::CPUPairsMatcherThread(const PairMatchingOptions& options, FeatureDataContainer* container,
                                             JobQueue<Input>* input_queue, JobQueue<Output>* output_queue)
    : PairsMatcherThread(options, container), input_queue_(input_queue), output_queue_(output_queue) {
    CHECK(options_.Check());
}

void CPUPairsMatcherThread::Run() {
    SignalValidSetup();

    while (true) {
        if (IsStopped()) {
            break;
        }

        auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            auto data = input_job.Data();
            
            const FeatureDescriptors& descriptors1 = feature_data_container_->GetCompressedDescriptors(data.image_id1);
            const FeatureDescriptors& descriptors2 = feature_data_container_->GetCompressedDescriptors(data.image_id2);
            MatchSiftFeaturesCPU(options_, descriptors1, descriptors2, &data.matches);

            CHECK(output_queue_->Push(std::move(data)));
        }
    }
}

GPUPairsMatcherThread::GPUPairsMatcherThread(const PairMatchingOptions& options, FeatureDataContainer* container,
                                             JobQueue<Input>* input_queue, JobQueue<Output>* output_queue)
    : PairsMatcherThread(options, container), input_queue_(input_queue), output_queue_(output_queue) {
    CHECK(options_.Check());

    prev_uploaded_image_ids_[0] = kInvalidImageId;
    prev_uploaded_image_ids_[1] = kInvalidImageId;
    obj_count++;
}

void GPUPairsMatcherThread::Run() {
#ifdef CUDA_ENABLED
    SiftMatchGPU sift_match_gpu;
    if (!CreateSiftGPUMatcher(options_, &sift_match_gpu)) {
        std::cout << "ERROR: SiftGPU not fully supported" << std::endl;
        SignalInvalidSetup();
        return;
    }

    SignalValidSetup();

    while (true) {
        if (IsStopped()) {
            break;
        }
        const auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            auto data = input_job.Data();

            const CompressedFeatureDescriptors* descriptors1_ptr;
            GetDescriptorData(0, data.image_id1, &descriptors1_ptr);
            const CompressedFeatureDescriptors* descriptors2_ptr;
            GetDescriptorData(1, data.image_id2, &descriptors2_ptr);

            const Image& image1 = feature_data_container_->GetImage(data.image_id1);
            const Camera& camera1 = feature_data_container_->GetCamera(image1.CameraId());

            const Image& image2 = feature_data_container_->GetImage(data.image_id2);
            const Camera& camera2 = feature_data_container_->GetCamera(image2.CameraId());
            int num_local_camera1 = camera1.NumLocalCameras();
            int num_local_camera2 = camera2.NumLocalCameras();

            if (num_local_camera1 >= 2 && num_local_camera2 >= 2 && options_.sub_matching) {
                if (data.image_id1 != data.image_id2) {
                    const PanoramaIndexs& panorama_idxs1 = 
                        feature_data_container_->GetPanoramaIndexs(data.image_id1);
                    const PanoramaIndexs& panorama_idxs2 =
                        feature_data_container_->GetPanoramaIndexs(data.image_id2);
                    
                    std::vector<size_t> num_points_local_camera1;
                    std::vector<size_t> num_points_local_camera2;
                    num_points_local_camera1.resize(num_local_camera1);
                    num_points_local_camera2.resize(num_local_camera2);

                    size_t num_local_point = 0, local_camera_idx = 0;
                    for (size_t i = 0; ; ++i) {
                        if (i == panorama_idxs1.size()) {
                            num_points_local_camera1.at(local_camera_idx) = num_local_point;
                            break;
                        }
                        auto panorama_idx = panorama_idxs1.at(i);
                        if (local_camera_idx != panorama_idx.sub_image_id) {
                            num_points_local_camera1.at(local_camera_idx) = num_local_point;
                            local_camera_idx = panorama_idx.sub_image_id;
                            num_local_point = 1;
                        } else {
                            num_local_point++;
                        }
                    }

                    num_local_point = 0;
                    local_camera_idx = 0;
                    for (size_t i = 0; ; ++i) {
                        if (i == panorama_idxs2.size()) {
                            num_points_local_camera2.at(local_camera_idx) = num_local_point;
                            break;
                        }
                        auto panorama_idx = panorama_idxs2.at(i);
                        if (local_camera_idx != panorama_idx.sub_image_id) {
                            num_points_local_camera2.at(local_camera_idx) = num_local_point;
                            local_camera_idx = panorama_idx.sub_image_id;
                            num_local_point = 1;
                        } else {
                            num_local_point++;
                        }
                    }
                    
                    data.matches.reserve(panorama_idxs1.size());

                    size_t local_idx1 = 0, local_idx2 = 0;
                    for (size_t i = 0; i < num_points_local_camera1.size(); ++i) {
                        for (size_t j = 0; j < num_points_local_camera2.size(); ++j) {
                            FeatureMatches matches;
                            MatchSiftFeaturesGPU(options_, 
                                j == 0 ? descriptors1_ptr : NULL, 
                                local_idx1, num_points_local_camera1.at(i),
                                descriptors2_ptr, 
                                local_idx2, num_points_local_camera2.at(j),
                                &sift_match_gpu, &matches);
                                
                            std::for_each(matches.begin(), matches.end(),
                                [&](FeatureMatch& match) {
                                    match.point2D_idx1 += local_idx1;
                                    match.point2D_idx2 += local_idx2;
                                    data.matches.emplace_back(match);
                                });

                            local_idx2 += num_points_local_camera2.at(j);
                        }
                        local_idx1 += num_points_local_camera1.at(i);
                        local_idx2 = 0;
                    }
                } else if (options_.self_matching) {
                    std::cout << "self-matching" << std::endl;
                    const PanoramaIndexs& panorama_idxs = 
                        feature_data_container_->GetPanoramaIndexs(data.image_id1);
                    
                    std::vector<size_t> num_points_local_camera;
                    num_points_local_camera.resize(num_local_camera1);
                    std::vector<size_t> acc_num_points_local_camera;
                    acc_num_points_local_camera.resize(num_local_camera1);
                    
                    size_t num_local_point = 0, local_camera_idx = 0;
                    size_t acc_num_local_point = 0;
                    for (int i = 0; ; ++i) {
                        if (i == panorama_idxs.size()) {
                            num_points_local_camera.at(local_camera_idx) = num_local_point;
                            acc_num_points_local_camera.at(local_camera_idx) = acc_num_local_point;
                            break;
                        }
                        auto panorama_idx = panorama_idxs.at(i);
                        if (local_camera_idx != panorama_idx.sub_image_id) {
                            num_points_local_camera.at(local_camera_idx) = num_local_point;
                            acc_num_points_local_camera.at(local_camera_idx) = acc_num_local_point;
                            local_camera_idx = panorama_idx.sub_image_id;
                            num_local_point = 1;
                            acc_num_local_point++;
                        } else {
                            num_local_point++;
                            acc_num_local_point++;
                        }
                    }

                    data.matches.reserve(panorama_idxs.size());

                    for (int i = 0; i < num_local_camera1; ++i) {
                        int j = (i - 1 + num_local_camera1) % num_local_camera1;
                        size_t local_idx1 = acc_num_points_local_camera.at(i) - num_points_local_camera.at(i);
                        size_t local_idx2 = acc_num_points_local_camera.at(j) - num_points_local_camera.at(j);

                        FeatureMatches matches;
                        MatchSiftFeaturesGPU(options_, 
                            descriptors1_ptr, 
                            local_idx1, num_points_local_camera.at(i),
                            descriptors2_ptr, 
                            local_idx2, num_points_local_camera.at(j),
                            &sift_match_gpu, &matches);
                            
                        std::for_each(matches.begin(), matches.end(),
                            [&](FeatureMatch& match) {
                                match.point2D_idx1 += local_idx1;
                                match.point2D_idx2 += local_idx2;
                                data.matches.emplace_back(match);
                            });
                    }
                }
            } else {
                MatchSiftFeaturesGPU(options_, descriptors1_ptr, descriptors2_ptr, &sift_match_gpu, &data.matches);
            }

            CHECK(output_queue_->Push(std::move(data)));
        }
    }
#endif
}

void GPUPairsMatcherThread::GetDescriptorData(const int index, const image_t image_id,
                                              const CompressedFeatureDescriptors** descriptors_ptr) {
    CHECK_GE(index, 0);
    CHECK_LE(index, 1);
    if (options_.sub_matching) {
        *descriptors_ptr = &feature_data_container_->GetCompressedDescriptors(image_id);
    } else {
        if (prev_uploaded_image_ids_[index] == image_id) {
            *descriptors_ptr = nullptr;
        } else {
            prev_uploaded_descriptors_[index] = feature_data_container_->GetCompressedDescriptors(image_id);
            *descriptors_ptr = &prev_uploaded_descriptors_[index];
            prev_uploaded_image_ids_[index] = image_id;
        }
    }
}

GuidedCPUPairsMatcherThread::GuidedCPUPairsMatcherThread(const PairMatchingOptions& options,
                                                         FeatureDataContainer* container, JobQueue<Input>* input_queue,
                                                         JobQueue<Output>* output_queue)
    : PairsMatcherThread(options, container), input_queue_(input_queue), output_queue_(output_queue) {
    CHECK(options_.Check());
}

void GuidedCPUPairsMatcherThread::Run() {
    SignalValidSetup();

    while (true) {
        if (IsStopped()) {
            break;
        }

        const auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            auto data = input_job.Data();

            if (data.two_view_geometry.inlier_matches.size() < static_cast<size_t>(options_.min_num_inliers)) {
                CHECK(output_queue_->Push(std::move(data)));
                continue;
            }

            const auto& camera1 =
                feature_data_container_->GetCamera(feature_data_container_->GetImage(data.image_id1).CameraId());
            const auto& camera2 =
                feature_data_container_->GetCamera(feature_data_container_->GetImage(data.image_id2).CameraId());

            // Guided match will be skipped if the camera model is "SPHERICAL"
            if (camera1.ModelName().compare("SPHERICAL") == 0 || camera2.ModelName().compare("SPHERICAL") == 0) {
                CHECK(output_queue_->Push(std::move(data)));
                continue;
            }

            const FeatureKeypoints& keypoints1 = feature_data_container_->GetKeypoints(data.image_id1);
            const FeatureKeypoints& keypoints2 = feature_data_container_->GetKeypoints(data.image_id2);
            const FeatureDescriptors& descriptors1 = feature_data_container_->GetCompressedDescriptors(data.image_id1);
            const FeatureDescriptors& descriptors2 = feature_data_container_->GetCompressedDescriptors(data.image_id2);

            if (options_.guided_matching_multi_homography) {
                MatchWarpSiftFeaturesCPU(options_, keypoints1, keypoints2, descriptors1, descriptors2,
                                         &data.two_view_geometry);
            } else {
                MatchGuidedSiftFeaturesCPU(options_, keypoints1, keypoints2, descriptors1, descriptors2,
                                           &data.two_view_geometry);
            }
            CHECK(output_queue_->Push(std::move(data)));
        }
    }
}

GuidedGPUPairsMatcherThread::GuidedGPUPairsMatcherThread(const PairMatchingOptions& options,
                                                         FeatureDataContainer* container, JobQueue<Input>* input_queue,
                                                         JobQueue<Output>* output_queue)
    : PairsMatcherThread(options, container), input_queue_(input_queue), output_queue_(output_queue) {
    CHECK(options_.Check());
    options_.max_ratio = options_.guided_match_max_ratio;
    prev_uploaded_image_ids_[0] = kInvalidImageId;
    prev_uploaded_image_ids_[1] = kInvalidImageId;
}

void GuidedGPUPairsMatcherThread::Run() {
#ifdef CUDA_ENABLED
    SiftMatchGPU sift_match_gpu;
    if (!CreateSiftGPUMatcher(options_, &sift_match_gpu)) {
        std::cout << "ERROR: SiftGPU not fully supported" << std::endl;
        SignalInvalidSetup();
        return;
    }
    SignalValidSetup();

    while (true) {
        if (IsStopped()) {
            break;
        }

        const auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            auto data = input_job.Data();

            if (data.two_view_geometry.inlier_matches.size() < static_cast<size_t>(options_.min_num_inliers)) {
                CHECK(output_queue_->Push(std::move(data)));
                continue;
            }
            const auto& camera1 =
                feature_data_container_->GetCamera(feature_data_container_->GetImage(data.image_id1).CameraId());
            const auto& camera2 =
                feature_data_container_->GetCamera(feature_data_container_->GetImage(data.image_id2).CameraId());


            const CompressedFeatureDescriptors* descriptors1_ptr;
            const FeatureKeypoints* keypoints1_ptr;
            GetFeatureData(0, data.image_id1, &keypoints1_ptr, &descriptors1_ptr);
            
            const CompressedFeatureDescriptors* descriptors2_ptr;
            const FeatureKeypoints* keypoints2_ptr;
            GetFeatureData(1, data.image_id2, &keypoints2_ptr, &descriptors2_ptr);

            const auto panorama_indices1 = feature_data_container_->GetPanoramaIndexs(data.image_id1);
            const auto panorama_indices2 = feature_data_container_->GetPanoramaIndexs(data.image_id2);

            if (options_.guided_matching_multi_homography) {
                MatchWarpSiftFeaturesGPU(options_, keypoints1_ptr, keypoints2_ptr, descriptors1_ptr, descriptors2_ptr,
                                         &sift_match_gpu, &data.two_view_geometry);
            } else {
                if (camera1.ModelName().compare("SPHERICAL") == 0 ||
                    camera2.ModelName().compare("SPHERICAL") == 0) {

                    if (camera1.ModelName().compare("SPHERICAL") == 0 &&
                        camera2.ModelName().compare("SPHERICAL") == 0) {
                        MatchGuidedSiftFeaturesGPUSpherical(camera1, camera2,
                                                            options_, keypoints1_ptr, keypoints2_ptr,
                                                            descriptors1_ptr, descriptors2_ptr,
                                                            &sift_match_gpu, &data.two_view_geometry);
                    } else {
                        //FIXME
                    }
                } else if (camera1.NumLocalCameras() > 1 ||
                           camera2.NumLocalCameras() > 1) {
                    std::vector<uint32_t> local_image_indices1;
                    std::vector<uint32_t> local_image_indices2;

                    for (size_t i = 0; i < panorama_indices1.size(); ++i) {
                        local_image_indices1.push_back(panorama_indices1[i].sub_image_id);
                    }

                    for (size_t i = 0; i < panorama_indices2.size(); ++i) {
                        local_image_indices2.push_back(panorama_indices2[i].sub_image_id);
                    }

                    MatchGuidedSiftFeaturesGPURig(camera1, camera2,
                        options_, keypoints1_ptr, keypoints2_ptr,
                        descriptors1_ptr, descriptors2_ptr,
                        local_image_indices1, local_image_indices2,
                        &sift_match_gpu, &data.two_view_geometry);
                } else {
                    MatchGuidedSiftFeaturesGPU(options_, keypoints1_ptr,
                        keypoints2_ptr, descriptors1_ptr, descriptors2_ptr,
                        &sift_match_gpu, &data.two_view_geometry);
                }
            }
            std::cout << "inlier matches after guided_match: " << data.two_view_geometry.inlier_matches.size()
                      << std::endl;

            CHECK(output_queue_->Push(std::move(data)));
        }
    }
#endif
}

void GuidedGPUPairsMatcherThread::GetFeatureData(const int index, const image_t image_id,
                                                 const FeatureKeypoints** keypoints_ptr,
                                                 const CompressedFeatureDescriptors** descriptors_ptr) {
    CHECK_GE(index, 0);
    CHECK_LE(index, 1);
    if (prev_uploaded_image_ids_[index] == image_id) {
        *keypoints_ptr = nullptr;
        *descriptors_ptr = nullptr;
    } else {
        prev_uploaded_keypoints_[index] = feature_data_container_->GetKeypoints(image_id);
        prev_uploaded_descriptors_[index] = feature_data_container_->GetCompressedDescriptors(image_id);

        *keypoints_ptr = &prev_uploaded_keypoints_[index];
        *descriptors_ptr = &prev_uploaded_descriptors_[index];
        prev_uploaded_image_ids_[index] = image_id;
    }
}

TwoViewGeometryVerifier::TwoViewGeometryVerifier(const PairMatchingOptions& options, FeatureDataContainer* container,
                                                 JobQueue<Input>* input_queue, JobQueue<Output>* output_queue)
    : options_(options), feature_data_container_(container), input_queue_(input_queue), output_queue_(output_queue) {
    CHECK(options_.Check());

    two_view_geometry_options_.min_num_inliers = static_cast<size_t>(options_.min_num_inliers);
    two_view_geometry_options_.max_num_matches = options.max_num_matches;
    two_view_geometry_options_.ransac_options.max_error = options_.max_error;
    two_view_geometry_options_.ransac_options.confidence = options_.confidence;
    two_view_geometry_options_.ransac_options.min_num_trials = static_cast<size_t>(options_.min_num_trials);
    two_view_geometry_options_.ransac_options.max_num_trials = 1000;//static_cast<size_t>(options_.max_num_trials);
    two_view_geometry_options_.ransac_options.min_inlier_ratio = options_.min_inlier_ratio;
    two_view_geometry_options_.is_sphere = options_.is_sphere;
    std::cout << "is_sphere: " << options_.is_sphere << std::endl;
}

void TwoViewGeometryVerifier::Run() {
    while (true) {
        if (IsStopped()) {
            break;
        }
        const auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            auto data = input_job.Data();

            if (data.matches.size() < static_cast<size_t>(options_.min_num_inliers)) {
                CHECK(output_queue_->Push(std::move(data)));
                continue;
            }
            const auto& camera1 =
                feature_data_container_->GetCamera(feature_data_container_->GetImage(data.image_id1).CameraId());
            const auto& camera2 =
                feature_data_container_->GetCamera(feature_data_container_->GetImage(data.image_id2).CameraId());

            const auto& keypoints1 = feature_data_container_->GetKeypoints(data.image_id1);
            const auto& keypoints2 = feature_data_container_->GetKeypoints(data.image_id2);
            const auto& panorama_indices1 = feature_data_container_->GetPanoramaIndexs(data.image_id1);
            const auto& panorama_indices2 = feature_data_container_->GetPanoramaIndexs(data.image_id2);
            const auto points1 = FeatureKeypointsToPointsVector(keypoints1);
            const auto points2 = FeatureKeypointsToPointsVector(keypoints2);
            const int num_local_camera1 = camera1.NumLocalCameras();
            const int num_local_camera2 = camera2.NumLocalCameras();

            bool mix = num_local_camera1 != num_local_camera2;

            if (mix) {
                std::cout << StringPrintf("Original matches(Mix, [%d %d]): %d\n", data.image_id1, data.image_id2, data.matches.size());
            } else {
                std::cout << "Original matches: " << data.matches.size() << " " << std::endl;
            }

            data.two_view_geometry.inlier_matches = data.matches;

#if 1
            if (num_local_camera1 <= 1 && num_local_camera2 <= 1) {
                auto two_view_geometry_options = two_view_geometry_options_;
                two_view_geometry_options.ransac_options.max_error = options_.max_error;
                data.two_view_geometry.Estimate(camera1, points1, camera2, points2, data.matches,
                                                two_view_geometry_options);
                if (data.two_view_geometry.inlier_matches.size() > options_.min_num_inliers) {
                    data.two_view_geometry.EstimateRelativePose(camera1, points1, camera2, points2,
                                                                two_view_geometry_options);
                }
            } else {
                std::vector<uint32_t> local_image_indices1;
                std::vector<uint32_t> local_image_indices2;

                for (size_t i = 0; i < panorama_indices1.size(); ++i) {
                    local_image_indices1.push_back(panorama_indices1[i].sub_image_id);
                }

                for (size_t i = 0; i < panorama_indices2.size(); ++i) {
                    local_image_indices2.push_back(panorama_indices2[i].sub_image_id);
                }

                if (num_local_camera1 > 1 && num_local_camera2 > 1) {
                    auto two_view_geometry_options = two_view_geometry_options_;
                    two_view_geometry_options.ransac_options.max_error = options_.max_error / 2;
                    data.two_view_geometry.EstimateRig(camera1, points1, camera2, points2, data.matches,
                                                        local_image_indices1, local_image_indices2,
                                                        two_view_geometry_options);
                    if (data.two_view_geometry.inlier_matches.size() > options_.min_num_inliers) {
                        std::vector<double> tri_angles;
                        data.two_view_geometry.EstimateRelativePoseRigGV(camera1, points1, camera2, points2,
                            local_image_indices1, local_image_indices2, two_view_geometry_options, &tri_angles, true);
                    }
                } else {
                    auto two_view_geometry_options = two_view_geometry_options_;
                    two_view_geometry_options.ransac_options.max_error = options_.max_error;
                    data.two_view_geometry.EstimateOneAndRig(camera1, points1, camera2, points2, data.matches, 
                                                            local_image_indices1, local_image_indices2,
                                                            two_view_geometry_options);
                }
            }
#else
            if (num_local_camera1 > 1 || num_local_camera2 > 1) {
                two_view_geometry_options_.ransac_options.max_error = options_.max_angular_error;
                two_view_geometry_options_.ransac_options.min_num_trials = 100;
                std::vector<uint32_t> local_image_indices1;
                std::vector<uint32_t> local_image_indices2;

                for (size_t i = 0; i < panorama_indices1.size(); ++i) {
                    local_image_indices1.push_back(panorama_indices1[i].sub_image_id);
                }

                for (size_t i = 0; i < panorama_indices2.size(); ++i) {
                    local_image_indices2.push_back(panorama_indices2[i].sub_image_id);
                }

                if (num_local_camera1 > 1 && num_local_camera2 > 1) {
                    data.two_view_geometry.EstimateRig(camera1, points1,
                        camera2, points2, data.matches,
                        local_image_indices1, local_image_indices2,
                        two_view_geometry_options_);
                } else {
                    data.two_view_geometry.EstimateMix(camera1, points1,
                        camera2, points2, data.matches,
                        local_image_indices1, local_image_indices2,
                        two_view_geometry_options_);
                }
                if (num_local_camera1 == num_local_camera2 &&
                    data.two_view_geometry.inlier_matches.size() > options_.min_num_inliers) {
                    std::vector<double> tri_angles;
                    data.two_view_geometry.EstimateRelativePoseRigGV(camera1, points1, camera2, points2,
                        local_image_indices1, local_image_indices2, two_view_geometry_options_, &tri_angles, true);
                }
                if (mix) {
                    std::cout << StringPrintf("Inlier matches(Mix, [%d %d]): %d\n", data.image_id1, data.image_id2, data.two_view_geometry.inlier_matches.size());
                } else {
                    std::cout << "Inlier matches: " << data.two_view_geometry.inlier_matches.size() << std::endl;
                }
                CHECK(output_queue_->Push(std::move(data)));
                continue;
            }

            data.two_view_geometry.inlier_matches = data.matches;

            if (camera1.ModelName().compare("SPHERICAL") == 0 || camera2.ModelName().compare("SPHERICAL") == 0) {
                two_view_geometry_options_.ransac_options.max_error = options_.max_angular_error;
            }

            if (options_.multiple_models) {
                data.two_view_geometry.EstimateMultiple(camera1, points1, camera2, points2, data.matches,
                                                        two_view_geometry_options_);
            } else {
                data.two_view_geometry.Estimate(camera1, points1, camera2, points2, data.matches,
                                                two_view_geometry_options_);
            }
            if (data.two_view_geometry.inlier_matches.size() > options_.min_num_inliers) {
                data.two_view_geometry.EstimateRelativePose(camera1, points1, camera2, points2,
                                                            two_view_geometry_options_);
            }
#endif
            std::cout << "Inlier matches(" << feature_data_container_->GetImage(data.image_id1).Name() << "<->"
                << feature_data_container_->GetImage(data.image_id2).Name() << "): " << data.two_view_geometry.inlier_matches.size() << std::endl;
            CHECK(output_queue_->Push(std::move(data)));
        }
    }
}

PairsMatcher::PairsMatcher(const PairMatchingOptions& options, FeatureDataContainer* container)
    : options_(options), feature_data_container_(container), is_setup_(false) {
    CHECK(options_.Check());

    int num_threads = GetEffectiveNumThreads(options_.num_threads);
    CHECK_GT(num_threads, 0);

    std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
    CHECK_GT(gpu_indices.size(), 0);

#ifdef CUDA_ENABLED
    if (options_.use_gpu && gpu_indices.size() == 1 && gpu_indices[0] == -1) {
        const int num_cuda_devices = GetNumCudaDevices();
        CHECK_GT(num_cuda_devices, 0);
        gpu_indices.resize(num_cuda_devices);
        std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
    }
#else   // CUDA_ENABLED
    options_.use_gpu = false;
#endif  // CUDA_DISABLE

    // create normal pair matching threads
    if (options_.use_gpu) {
        auto gpu_options = options_;
        match_threads_.reserve(gpu_indices.size());
        for (const auto& gpu_index : gpu_indices) {
            gpu_options.gpu_index = std::to_string(gpu_index);
            match_threads_.emplace_back(
                new GPUPairsMatcherThread(gpu_options, feature_data_container_, &matcher_queue_, &verifier_queue_));
        }
    } else {
        match_threads_.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            match_threads_.emplace_back(
                new CPUPairsMatcherThread(options_, feature_data_container_, &matcher_queue_, &verifier_queue_));
        }
    }

    verifiers_.reserve(num_threads);
    if (options_.guided_matching) {
        // create verifiers first if guided matching is enabled
        for (int i = 0; i < num_threads; ++i) {
            verifiers_.emplace_back(new TwoViewGeometryVerifier(options_, feature_data_container_, &verifier_queue_,
                                                                &guided_matcher_queue_));
        }
        // create guided pair matching threads
        if (options_.use_gpu) {
            auto gpu_options = options_;
            guided_match_threads_.reserve(gpu_indices.size());
            for (const auto& gpu_index : gpu_indices) {
                gpu_options.gpu_index = std::to_string(gpu_index);

                guided_match_threads_.emplace_back(new GuidedGPUPairsMatcherThread(
                    gpu_options, feature_data_container_, &guided_matcher_queue_, &output_queue_));
            }
        } else {
            guided_match_threads_.reserve(num_threads);
            for (int i = 0; i < num_threads; ++i) {
                guided_match_threads_.emplace_back(new GuidedCPUPairsMatcherThread(
                    options_, feature_data_container_, &guided_matcher_queue_, &output_queue_));
            }
        }
    } else {  
        // create verifiers directly if guided matching is disabled
        std::cout<<"Verifier num threads: " << num_threads << std::endl;
        for (int i = 0; i < num_threads; ++i) {
            verifiers_.emplace_back(
                new TwoViewGeometryVerifier(options_, feature_data_container_, &verifier_queue_, &output_queue_));
        }
    }
}

PairsMatcher::~PairsMatcher() {
    matcher_queue_.Wait();
    verifier_queue_.Wait();
    guided_matcher_queue_.Wait();
    output_queue_.Wait();

    for (auto& matcher : match_threads_) {
        matcher->Stop();
    }

    for (auto& verifier : verifiers_) {
        verifier->Stop();
    }

    for (auto& guided_matcher : guided_match_threads_) {
        guided_matcher->Stop();
    }

    matcher_queue_.Stop();
    verifier_queue_.Stop();
    guided_matcher_queue_.Stop();
    output_queue_.Stop();

    for (auto& matcher : match_threads_) {
        matcher->Wait();
    }

    for (auto& verifier : verifiers_) {
        verifier->Wait();
    }

    for (auto& guided_matcher : guided_match_threads_) {
        guided_matcher->Wait();
    }
}

bool PairsMatcher::Setup() {
    for (auto& matcher : match_threads_) {
        matcher->SetMaxNumMatches(options_.max_num_matches);
        matcher->Start();
    }
    for (auto& verifier : verifiers_) {
        verifier->Start();
    }
    for (auto& guided_matcher : guided_match_threads_) {
        guided_matcher->SetMaxNumMatches(options_.max_num_matches);
        guided_matcher->Start();
    }

    for (auto& matcher : match_threads_) {
        if (!matcher->CheckValidSetup()) {
            return false;
        }
    }
    for (auto& guided_matcher : guided_match_threads_) {
        if (!guided_matcher->CheckValidSetup()) {
            return false;
        }
    }

    is_setup_ = true;
    return true;
}

void PairsMatcher::Match(const std::vector<std::pair<image_t, image_t>>& image_pairs,
                         MatchDataContainer* match_data_container) {
    CHECK_NOTNULL(feature_data_container_);
    CHECK(is_setup_);

    if (image_pairs.empty()) {
        return;
    }
    size_t pair_count = image_pairs.size();

    std::unordered_set<image_pair_t> image_pair_ids;
    image_pair_ids.reserve(image_pairs.size());

    size_t num_outputs = 0;
    for (const auto image_pair : image_pairs) {
        // Avoid self-matches.
        if (image_pair.first == image_pair.second) {
            continue;
        }

        // Avoid duplicate image pairs.
        const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_pair.first, image_pair.second);
        if (image_pair_ids.count(pair_id) > 0) {
            continue;
        }
        image_pair_ids.insert(pair_id);

        num_outputs += 1;

        FeatureMatcherData data;
        data.image_id1 = image_pair.first;
        data.image_id2 = image_pair.second;
        CHECK(matcher_queue_.Push(std::move(data)));
    }

    if (options_.sub_matching && options_.self_matching) {
        const auto & image_ids = feature_data_container_->GetImageIds();
        for (auto image_id : image_ids) {
            const auto& image = feature_data_container_->GetImage(image_id);
            const auto& camera = feature_data_container_->GetCamera(image.CameraId());
            if (camera.NumLocalCameras() > 2) {
                num_outputs += 1;

                FeatureMatcherData data;
                data.image_id1 = image_id;
                data.image_id2 = image_id;
                CHECK(matcher_queue_.Push(data));
            }
        }
    }

    // output results
    std::cout << std::endl;
    for (size_t i = 0; i < num_outputs; ++i) {
        // std::cout << "\r";
        std::cout << "Matching Pair [" << i + 1 << " / " << num_outputs << "]" << std::flush;
        std::cout << std::endl;
        const auto output_job = output_queue_.Pop();
        CHECK(output_job.IsValid());
        auto output = output_job.Data();

        if (output.matches.size() < static_cast<size_t>(options_.min_num_inliers)) {
            output.matches = {};
        }

        if (output.two_view_geometry.inlier_matches.size() < static_cast<size_t>(options_.min_num_inliers)) {
            output.two_view_geometry = TwoViewGeometry();
        }

        match_data_container->AddMatch(output);
    }
    CHECK_EQ(output_queue_.Size(), 0);
}

FeatureMatcher::FeatureMatcher(const FeatureMatchingOptions& options, FeatureDataContainer* container,
                               MatchDataContainer* md_container, SceneGraphContainer* graph)
    : generator_(options, container),
      matcher_(options.PairMatching(), container),
      options_(options),
      feature_data_container_(container),
      match_data_container_(md_container),
      scene_graph_(graph) {
    CHECK(options.Check());
}

void FeatureMatcher::Run() {
    generator_.Run();
    const std::vector<std::pair<image_t, image_t>>& image_pairs = generator_.GetPairs();

    CHECK(matcher_.Setup());

    Timer timer;
    timer.Start();

    matcher_.Match(image_pairs, match_data_container_);

    std::cout << StringPrintf("Descriptor match in %.3fs", timer.ElapsedSeconds()) << std::endl;

    const std::vector<std::pair<image_t, image_t>>& loop_image_pairs = generator_.GetLoopPairs();

    timer.Start();
    CheckLoopMatches(loop_image_pairs, options_.sequential_matching_.loop_detection_before_sequential_matching);
    std::cout << StringPrintf("Check loop match in %.3fs", timer.ElapsedSeconds()) << std::endl;

}

int FeatureMatcher::BluetoothBeaconDistance(const double& timestamp1, const double& timestamp2, double& distance) {
    double timestamp1ms, timestamp2ms;
    if (timestamp1 < 2000000000 && timestamp1 > 1000000000)
        timestamp1ms = timestamp1 * 1000;
    else if (timestamp1 < 2000000000000 && timestamp1 > 1000000000000)
        timestamp1ms = timestamp1;
    else {
        std::cout << "Wrong timestamp input : timestamp1: " << timestamp1 << std::endl;
        return -1;
    }
    if (timestamp2 < 2000000000 && timestamp2 > 1000000000)
        timestamp2ms = timestamp2 * 1000;
    else if (timestamp2 < 2000000000000 && timestamp2 > 1000000000000)
        timestamp2ms = timestamp2;
    else {
        std::cout << "Wrong timestamp input : timestamp2: " << timestamp2 << std::endl;
        return -1;
    }

    ////gather timestamp signal
    std::map<std::string, std::vector<int>> rssimap1, rssimap2;
    std::map<std::string, std::vector<double>> timestamp_map1, timestamp_map2;
    double time_shift = 3000;
    for (double timestamp = timestamp1ms - time_shift; timestamp < timestamp1ms + time_shift; timestamp++) {
        if (options_.prior_bluetooth_time_signal.count(timestamp)) {
            for (auto bluetooth_signal : options_.prior_bluetooth_time_signal[timestamp]) {
                std::string unic_majorminor = bluetooth_signal.first;
                rssimap1[unic_majorminor].push_back(bluetooth_signal.second);
                timestamp_map1[unic_majorminor].push_back(timestamp);
            }
        }
    }
    for (double timestamp = timestamp2ms - time_shift; timestamp < timestamp2ms + time_shift; timestamp++) {
        if (options_.prior_bluetooth_time_signal.count(timestamp)) {
            for (auto bluetooth_signal : options_.prior_bluetooth_time_signal[timestamp]) {
                std::string unic_majorminor = bluetooth_signal.first;
                rssimap2[unic_majorminor].push_back(bluetooth_signal.second);
                timestamp_map2[unic_majorminor].push_back(timestamp);
            }
        }
    }

    ////compare signal map
    int good_signal_num1 = 0, good_signal_num2 = 0;
    int signal_num1 = 0, signal_num2 = 0;
    int max_rssi1 = -100, max_rssi2 = -100;

    for (auto& beacon_data : rssimap1) {
        if (beacon_data.second.size() == 0) continue;
        
        // TODO: 
        // int rssi_mean = (int)(std::accumulate(std::begin(beacon_data.second), std::end(beacon_data.second), 0.0) /
        //                       beacon_data.second.size());
        double rssi_sum = 0;
        double weight_sum = 0;
        for (int i = 0; i < beacon_data.second.size(); i++) {
            double time_diff = std::abs(timestamp_map1[beacon_data.first][i] - timestamp1);
            // std::cout << "1 / (time_diff / time_shift) = " << 1 / (time_diff / time_shift) << std::endl;
            double weight = 1 / (time_diff / time_shift);
            rssi_sum += weight * beacon_data.second[i];
            weight_sum += weight;
        }

        double rssi_mean = rssi_sum / weight_sum;


        beacon_data.second.push_back(rssi_mean);

        if (rssi_mean > -85) {
            good_signal_num1++;
            signal_num1++;
        } else if (rssi_mean > -93){
            signal_num1++;
        } 


        if (rssi_mean > max_rssi1) {
            max_rssi1 = rssi_mean;
        }
    }

    for (auto& beacon_data : rssimap2) {
        if (beacon_data.second.size() == 0) continue;

        // TODO: 
        // int rssi_mean = (int)(std::accumulate(std::begin(beacon_data.second), std::end(beacon_data.second), 0.0) /
        //                       beacon_data.second.size());
        double rssi_sum = 0;
        double weight_sum = 0;
        for (int i = 0; i < beacon_data.second.size(); i++) {
            double time_diff = std::abs(timestamp_map2[beacon_data.first][i] - timestamp2);
            double weight = 1 / (time_diff / time_shift);
            rssi_sum += weight * beacon_data.second[i];
            weight_sum += weight;
        }

        double rssi_mean = rssi_sum / weight_sum;

        beacon_data.second.push_back(rssi_mean);

        if (rssi_mean > -85) {
            good_signal_num2++;
            signal_num2++;
        } else if (rssi_mean > -93){
            signal_num2++;
        } 

        if (rssi_mean > max_rssi2) {
            max_rssi2 = rssi_mean;
        }
    }

    // for (auto& beacon_data : rssimap1) {
    //     std::cout << "rssimap1 : " << beacon_data.first << " " << beacon_data.second.back() << std::endl;
    // }
    // for (auto& beacon_data : rssimap2) {
    //     std::cout << "rssimap2 : " << beacon_data.first << " " << beacon_data.second.back() << std::endl;
    // }

    // if (signal_num1 < 3 && signal_num2 < 3 && max_rssi1 < -90 && max_rssi2 < -90) {
    //     std::cout << "Error: both signal is too weak to compare" << std::endl;
    //     return -1;
    // }

    if (signal_num1 == 0 || signal_num2 == 0) {
        return -1;
    }

    // std::map<std::string,std::vector<int>> rssimap_stronger,rssimap_other;
    std::vector<double> Eucldis_vec;
    double rssi_distance = -100;

    // std::cout << "good_signal_num1 " << good_signal_num1 << " good_signal_num2: " << good_signal_num2 << std::endl;

    if (good_signal_num1 > good_signal_num2) {
        Eucldis_vec.clear();
        for (const auto& beacon_data : rssimap1) {
            if (beacon_data.second.back() < -95) continue;

            if (rssimap2.count(beacon_data.first)) {
                double Eucldis = (beacon_data.second.back() - rssimap2[beacon_data.first].back()) *
                                 (beacon_data.second.back() - rssimap2[beacon_data.first].back());
                // std::cout << " dis of Beacon " << beacon_data.first << " " << sqrt(Eucldis) << std::endl;
                Eucldis_vec.push_back(Eucldis);
            } else {
                double Eucldis = (-95 - beacon_data.second.back()) * (-95 - beacon_data.second.back());
                Eucldis_vec.push_back(Eucldis);
                // std::cout << " dis of Beacon " << beacon_data.first << " " << Eucldis << std::endl;
            }
        }
        rssi_distance =
            sqrt((std::accumulate(std::begin(Eucldis_vec), std::end(Eucldis_vec), 0.0)) / Eucldis_vec.size());

    } else {
        Eucldis_vec.clear();
        for (const auto& beacon_data : rssimap2) {
            if (beacon_data.second.back() < -95) continue;
            
            if (rssimap1.count(beacon_data.first)) {
                double Eucldis = (beacon_data.second.back() - rssimap1[beacon_data.first].back()) *
                                 (beacon_data.second.back() - rssimap1[beacon_data.first].back());
                // std::cout << " dis of Beacon " << beacon_data.first << " " << sqrt(Eucldis) << std::endl;
                Eucldis_vec.push_back(Eucldis);
            } else {
                double Eucldis = (-95 - beacon_data.second.back()) * (-95 - beacon_data.second.back());
                Eucldis_vec.push_back(Eucldis);
                // std::cout << " dis of Beacon " << beacon_data.first << " " << Eucldis << std::endl;
            }
        }
        rssi_distance =
            sqrt((std::accumulate(std::begin(Eucldis_vec), std::end(Eucldis_vec), 0.0)) / Eucldis_vec.size());
    }

    distance = rssi_distance;
    return 0;
}

void FeatureMatcher::CheckLoopMatches(const std::vector<std::pair<image_t,image_t>>& loop_image_pairs_, bool check){
    std::cout<<"CheckLoopMatches"<<std::endl;
    
    std::unordered_set<image_pair_t> loop_image_pair_ids;
    loop_image_pair_ids.reserve(loop_image_pairs_.size());
    
    for(const auto& image_pair: loop_image_pairs_){
        const image_pair_t loop_pair_id = sensemap::utility::ImagePairToPairId(image_pair.first, image_pair.second);
        loop_image_pair_ids.insert(loop_pair_id);
    }
    
    const std::vector<FeatureMatcherData>& results_ = match_data_container_->GetMatchResults();
    inlier_matches_ = std::vector<bool>(results_.size(),true);
    
    if(!check){
        //skip check in this stage, check in RobustLoopDetection
        return;
    }

    size_t min_num_matches = options_.pair_matching_.min_num_inliers;
    auto UseInlierMatchesCheck = [min_num_matches](const TwoViewGeometry& two_view_geometry) {
        return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >= min_num_matches;
    };

    std::unordered_map<image_pair_t, double> match_strength;
    std::unordered_map<image_pair_t, double> bluetooth_info;

    // The first round check according to the match distribution
    std::cout<<"results_.size() "<<results_.size()<<std::endl;
    for(size_t fm_i = 0; fm_i < results_.size(); ++fm_i){
        const FeatureMatcherData& fmdata = results_[fm_i];

        image_t image_id1 = fmdata.image_id1;
        image_t image_id2 = fmdata.image_id2;

        const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id1, image_id2);
        double match_covered_area_ratio = 0.0;
        if(!UseInlierMatchesCheck(fmdata.two_view_geometry) || !(loop_image_pair_ids.count(pair_id)>0)){
            continue;
        }
      
        const PanoramaIndexs& panorama_indices1 = feature_data_container_->GetPanoramaIndexs(image_id1);
        const PanoramaIndexs& panorama_indices2 = feature_data_container_->GetPanoramaIndexs(image_id2);

        const PieceIndexs& piece_indices1 = feature_data_container_->GetPieceIndexs(image_id1);
        const PieceIndexs& piece_indices2 = feature_data_container_->GetPieceIndexs(image_id2);

        std::cout << "loop pair: ";
        std::cout << image_id1<<"-"<<feature_data_container_->GetImage(image_id1).Name() << " "
                  << image_id2<<"-"<<feature_data_container_->GetImage(image_id2).Name() << " ";

        const Image& image1 = feature_data_container_->GetImage(image_id1);
        const Camera& Camera1 = feature_data_container_->GetCamera(image1.CameraId());

        const Image& image2 = feature_data_container_->GetImage(image_id2);
        const Camera& Camera2 = feature_data_container_->GetCamera(image2.CameraId());

        if (Camera1.NumLocalCameras() > 1 || Camera1.ModelName().compare("SPHERICAL") == 0 ||
            Camera2.NumLocalCameras() > 1 || Camera2.ModelName().compare("SPHERICAL") == 0) {
            bool piecewise = false;
            int sub_image_count1,sub_image_count2;
            if (Camera1.NumLocalCameras() == 2 && options_.pair_matching_.convert_to_perspective_image) {
                piecewise = true;
                sub_image_count1 = options_.pair_matching_.perspective_image_count;
            } else if (Camera1.ModelName().compare("SPHERICAL") == 0 &&
                       options_.pair_matching_.convert_to_perspective_image) {
                sub_image_count1 = options_.pair_matching_.perspective_image_count;
            } else {
                sub_image_count1 = Camera1.NumLocalCameras();
            }

            if (Camera2.NumLocalCameras() == 2 && options_.pair_matching_.convert_to_perspective_image) {
                piecewise = true;
                sub_image_count2 = options_.pair_matching_.perspective_image_count;
            } else if (Camera1.ModelName().compare("SPHERICAL") == 0 &&
                       options_.pair_matching_.convert_to_perspective_image) {
                sub_image_count2 = options_.pair_matching_.perspective_image_count;
            } else {
                sub_image_count2 = Camera2.NumLocalCameras();
            }

            if (MatchFromRepetitivePattern(fmdata.two_view_geometry.inlier_matches, panorama_indices1,
                                           panorama_indices2, piece_indices1, piece_indices2, match_covered_area_ratio,
                                           options_.pair_matching_.min_covered_sub_image_ratio, sub_image_count1,sub_image_count2,
                                           options_.pair_matching_.min_matched_feature_per_piece,
                                           piecewise)) {
                std::cout << "have repetitive pattern" << std::endl;
                inlier_matches_[fm_i] = false;
                continue;
            }
        }

        if (options_.have_prior_pose_) {
            CHECK(options_.prior_pose_validations.find(image_id1) != options_.prior_pose_validations.end());
            CHECK(options_.prior_pose_validations.find(image_id2) != options_.prior_pose_validations.end());

            if (!options_.prior_pose_validations.at(image_id1) || !options_.prior_pose_validations.at(image_id2)) {
                std::cout << "invalid prior pose, skip the match" << std::endl;
                inlier_matches_[fm_i] = false;
                continue;
            }
        }

        bool match_from_far_place = false;
        if (options_.have_prior_pose_) {
            if (options_.prior_rotations.find(image_id1) != options_.prior_rotations.end() &&
                options_.prior_rotations.find(image_id2) != options_.prior_rotations.end()) {
                Eigen::Vector4d qvec1 = options_.prior_rotations.at(image_id1);
                Eigen::Vector3d tvec1 = options_.prior_translations.at(image_id1);

                Eigen::Vector4d qvec2 = options_.prior_rotations.at(image_id2);
                Eigen::Vector3d tvec2 = options_.prior_translations.at(image_id2);

                Eigen::Vector3d C1 = ProjectionCenterFromPose(qvec1, tvec1);
                Eigen::Vector3d C2 = ProjectionCenterFromPose(qvec2, tvec2);

                double distance = (C1 - C2).norm();
                if (distance > options_.max_match_distance * options_.prior_neighbor_distance) {
                    match_from_far_place = true;
                }
            }
        }

        if (options_.have_prior_bluetooth_) {

            std::string image_1_name = feature_data_container_->GetImage(image_id1).Name();
            std::string image_2_name = feature_data_container_->GetImage(image_id2).Name();


            std::string video_name_1 = GetVideoTimeStampFromPath(image_1_name);
            std::string video_name_2 = GetVideoTimeStampFromPath(image_2_name);

            // Get the time stamp 
            double image_1_timestamp = std::stod(GetImageTimeStampFromPath(image_1_name));
            double image_2_timestamp = std::stod(GetImageTimeStampFromPath(image_2_name));

            double distance = -100;
            int bluetooth_status = BluetoothBeaconDistance(image_1_timestamp, image_2_timestamp, distance);
            bluetooth_info[pair_id] = distance;

            if (distance > (options_.prior_bluetooth_threshold_outlier)) {
                match_from_far_place = true;
            }

        }

        if (options_.have_prior_location_) {
            if (options_.prior_locations.find(image_id1) != options_.prior_locations.end() &&
                options_.prior_locations.find(image_id2) != options_.prior_locations.end()) {
                

                Eigen::Vector3d C1 = options_.prior_locations.at(image_id1).first;
                Eigen::Vector3d C2 = options_.prior_locations.at(image_id2).first;

                double distance = (C1 - C2).norm();
                if (distance > options_.max_distance_for_loop) {
                    match_from_far_place = true;
                }

            }
        }

        if (match_from_far_place) {
            std::cout << "match from far place" << std::endl;
            inlier_matches_[fm_i] = false;
            continue;
        }
        std::cout << std::endl;

        match_strength.emplace(pair_id,  match_covered_area_ratio);
    }


    std::unordered_map<image_pair_t,double> loop_pairs_info;
    std::unordered_set<image_pair_t> normal_pairs;
    // extract loop pairs with match scores
    loop_pairs_info.clear();
    loop_pairs_info = match_strength;

    // extract normal pairs from all image pairs
    std::cout<<"all pairs count: "<<generator_.GetPairs().size()<<std::endl;
    std::vector<image_pair_t> test_pairs;
    std::vector<image_pair_t> loop_pairs;
    for(auto& pair: generator_.GetPairs()){
        const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(pair.first, pair.second);
        if(!loop_pairs_info.count(pair_id)){
            normal_pairs.emplace(pair_id);
        }
    }
    std::cout<<"all normal pairs count: "<<normal_pairs.size()<<std::endl;

    scene_graph_->CorrespondenceGraph()->SetLoopPairsInfo(loop_pairs_info);
    scene_graph_->CorrespondenceGraph()->SetNoramlPairs(normal_pairs);
    if (options_.have_prior_bluetooth_) {
        scene_graph_->CorrespondenceGraph()->SetBluetoothPairsInfo(bluetooth_info);
    }

    std::cout<<"all loop pairs count: "<<match_strength.size()<<std::endl;
    // The second round check, decide whether a loop is strong enough. Only the matches from strong loop will be used for
    // localizing a camera
    std::cout << "Bluetooth Prior Filter Enable    : " << options_.have_prior_bluetooth_ << std::endl;
    std::cout << "Bluetooth Prior Filter Threshold Inside: " << options_.prior_bluetooth_threshold_inside << std::endl;
    std::cout << "Bluetooth Prior Filter Threshold Outside: " << options_.prior_bluetooth_threshold_outside << std::endl;
    std::cout << "Bluetooth Prior Filter Threshold Outlier: " << options_.prior_bluetooth_threshold_outlier << std::endl;

    std::unordered_map<image_t, std::unordered_set<image_t>> strong_graph_neighbors;
    strong_loop_pairs_.clear();
    for(const auto& image_pair: match_strength){
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePairOrdered(image_pair.first, &image_id1, &image_id2);

        label_t label_id1 = feature_data_container_->GetImage(image_id1).LabelId();
        auto neighbors_id1 = feature_data_container_->GetSequentialNeighbor(
            image_id1, label_id1, options_.pair_matching_.strong_loop_check_neighbor_count_src);

        label_t label_id2 = feature_data_container_->GetImage(image_id2).LabelId();
        auto neighbors_id2 = feature_data_container_->GetSequentialNeighbor(
            image_id2, label_id2, options_.pair_matching_.strong_loop_check_neighbor_count_dst);

        bool consistent_loop = true;

        for(const auto neighbor_id1: neighbors_id1){

            bool looped = false;
            for(const auto neighbor_id2: neighbors_id2){

                image_pair_t image_pair_id = sensemap::utility::ImagePairToPairId(neighbor_id1, neighbor_id2);

                if(match_strength.find(image_pair_id) != match_strength.end()){
                    looped = true;
                    break;
                }
            }

            if(!looped){
                consistent_loop = false;
                break;
            }
        }

        // Filter the base strong loop pair with bluetooth prior
        if (options_.have_prior_bluetooth_) {
            bool filter_pair = false;
            std::string image_1_name = feature_data_container_->GetImage(image_id1).Name();
            std::string image_2_name = feature_data_container_->GetImage(image_id2).Name();

            std::string video_name_1 = GetVideoTimeStampFromPath(image_1_name);
            std::string video_name_2 = GetVideoTimeStampFromPath(image_2_name);

            double distance = bluetooth_info[image_pair.first];

            // std::cout << std::setprecision(20)  << "image_1_timestamp = " << image_1_timestamp << std::endl;
            // std::cout << std::setprecision(20)  << "image_2_timestamp = " << image_2_timestamp << std::endl;

            // std::cout << "distance = " << distance << std::endl;
            // std::cout << "video_name_1 = " << video_name_1 << std::endl;
            // std::cout << "video_name_2 = " << video_name_2 << std::endl;
            if (video_name_1 == video_name_2) {
                // std::cout << "Video Name is Equal" << std::endl;
                bool distance_larger = distance > (options_.prior_bluetooth_threshold_inside);
                // std::cout << "Distance is larger than threshold " << distance_larger << std::endl;
                // std::cout << "Threshold = " << (options_.prior_bluetooth_threshold - 2) << std::endl;
                if (distance_larger) {
                    filter_pair= true;
                }
            } else {
                // std::cout << "Video Name is Not Equal" << std::endl;
                bool distance_larger = distance > (options_.prior_bluetooth_threshold_outside);
                // std::cout << "Distance is larger than threshold " << distance_larger << std::endl;
                // std::cout << "Threshold = " << (options_.prior_bluetooth_threshold + 2) << std::endl;
                if (distance_larger) {
                    filter_pair= true;
                }
            }

            // if (distance == -100) {
            //     filter_pair= true;
            // }

            if (filter_pair){
                continue;
            }
        }

        if(consistent_loop && image_pair.second > options_.pair_matching_.covered_sub_image_ratio_strong_loop){
            strong_loop_pairs_.insert(image_pair.first);
            std::cout<<"strong loop pair: "<<feature_data_container_->GetImage(image_id1).Name()<<" "
                     <<feature_data_container_->GetImage(image_id2).Name()<<std::endl;
            strong_graph_neighbors[image_id1].insert(image_id2);
            strong_graph_neighbors[image_id2].insert(image_id1);            
        }
    }


    std::cout<<"base strong loop pair count: "<<strong_loop_pairs_.size()<<std::endl;

    for(int transitivity = 0; transitivity<options_.pair_matching_.strong_loop_transitivity; transitivity ++){
        std::unordered_set<image_pair_t> transitive_strong_loop_pairs;

        for (const auto& image_pair : match_strength) {
            if(strong_loop_pairs_.count(image_pair.first) > 0){
                continue;
            }
            
            image_t image_id1, image_id2;
            sensemap::utility::PairIdToImagePairOrdered(image_pair.first, &image_id1, &image_id2);

            label_t label_id1 = feature_data_container_->GetImage(image_id1).LabelId();
            auto sequential_neighbors_id1 = feature_data_container_->GetSequentialNeighbor(
                image_id1, label_id1, options_.pair_matching_.transitive_strong_loop_neighbor_count);

            label_t label_id2 = feature_data_container_->GetImage(image_id2).LabelId();
            auto sequential_neighbors_id2 = feature_data_container_->GetSequentialNeighbor(
                image_id2, label_id2, options_.pair_matching_.transitive_strong_loop_neighbor_count);


            bool sequential_transitive_strong_loop = false;
            for (const auto neighbor_id1 : sequential_neighbors_id1) {
                for (const auto neighbor_id2 : sequential_neighbors_id2) {
                    image_pair_t image_pair_id = sensemap::utility::ImagePairToPairId(neighbor_id1, neighbor_id2);
                    if(strong_loop_pairs_.count(image_pair_id) > 0){
                        sequential_transitive_strong_loop = true;
                        break;
                    }
                }
                if(sequential_transitive_strong_loop){
                    break;
                }  
            }

            
            bool graph_transitive_strong_loop = false;
            if (strong_graph_neighbors.find(image_id1) != strong_graph_neighbors.end() &&
                strong_graph_neighbors.find(image_id2) != strong_graph_neighbors.end()){

                std::unordered_set<image_t> strong_graph_neighbors_id1 = strong_graph_neighbors.at(image_id1);
                std::unordered_set<image_t> strong_graph_neighbors_id2 = strong_graph_neighbors.at(image_id2);
        
                for (const auto neighbor_id1 : strong_graph_neighbors_id1) {
                    for (const auto neighbor_id2 : strong_graph_neighbors_id2) {
                        image_pair_t image_pair_id = sensemap::utility::ImagePairToPairId(neighbor_id1, neighbor_id2);
                        if (strong_loop_pairs_.count(image_pair_id) > 0) {
                            graph_transitive_strong_loop = true;
                            break;
                        }
                    }
                    if (graph_transitive_strong_loop) {
                        break;
                    }
                }
            }

            if (sequential_transitive_strong_loop || graph_transitive_strong_loop) {
                transitive_strong_loop_pairs.insert(image_pair.first);
                if (sequential_transitive_strong_loop) {
                    std::cout << "sequential_transitive_strong loop pair: "
                              << feature_data_container_->GetImage(image_id1).Name() << " "
                              << feature_data_container_->GetImage(image_id2).Name() << std::endl;
                } else if (graph_transitive_strong_loop) {
                    std::cout << "graph_transitive_strong loop pair: "
                              << feature_data_container_->GetImage(image_id1).Name() << " "
                              << feature_data_container_->GetImage(image_id2).Name() << std::endl;
                }
            }
        }

        for(const auto& pair_id: transitive_strong_loop_pairs){
            strong_loop_pairs_.insert(pair_id);
            image_t image_id1, image_id2;
            sensemap::utility::PairIdToImagePairOrdered(pair_id, &image_id1, &image_id2);
            strong_graph_neighbors[image_id1].insert(image_id2);
            strong_graph_neighbors[image_id2].insert(image_id1);
        }

        std::cout<<"transitive strong loop pair count: "<<transitive_strong_loop_pairs.size()<<std::endl;
        if(transitive_strong_loop_pairs.size() == 0){
            break;
        }
    }

    std::cout<<"final strong loop pair count: "<<strong_loop_pairs_.size()<<std::endl;

}

void FeatureMatcher::BuildSceneGraph() {
    Timer timer;
    timer.Start();
    std::cout << "Building Scene graph..." << std::endl << std::flush;
    std::shared_ptr<class CorrespondenceGraph> correspondence_graph = scene_graph_->CorrespondenceGraph();
    // preprocessing
    const std::vector<FeatureMatcherData>& results_ = match_data_container_->GetMatchResults();

    size_t min_num_matches = options_.pair_matching_.min_num_inliers;
    auto UseInlierMatchesCheck = [min_num_matches](const TwoViewGeometry& two_view_geometry) {
        return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >= min_num_matches;
    };
    std::vector<image_t> image_ids = feature_data_container_->GetImageIds();
    std::cout << "size = " << image_ids.size() << std::endl;
    image_t max_image_id = 0;
    std::unordered_set<image_t> image_id_set;
    for (const auto& image_id : image_ids) {
        image_id_set.insert(image_id);
        max_image_id = std::max(max_image_id, image_id);
    }

    // collect connected images in the correspondence graph
    std::unordered_set<image_t> connected_image_ids;
    connected_image_ids.reserve(image_ids.size());
    for (size_t i = 0; i < results_.size(); ++i) {
        const FeatureMatcherData& fmdata = results_[i];

        if (inlier_matches_[i] && UseInlierMatchesCheck(fmdata.two_view_geometry)) {
            image_t image_id1 = fmdata.image_id1;
            image_t image_id2 = fmdata.image_id2;

            if (image_id_set.count(image_id1) > 0 && image_id_set.count(image_id2) > 0) {
                connected_image_ids.insert(image_id1);
                connected_image_ids.insert(image_id2);
            }
        }
    }

    Timer timer1;
    timer1.Start();
    // add images and cameras
    std::vector<class Image> images;
    images.reserve(image_ids.size());
    for (const auto& image_id : image_ids) {
        if (!(connected_image_ids.count(image_id) > 0)) {
            continue;
        }
        // add image to image map
        const auto& image = feature_data_container_->GetImage(image_id);
        const Camera &camera = feature_data_container_->GetCamera(image.CameraId());
        images.emplace_back(image);

        const FeatureKeypoints& keypoints = feature_data_container_->GetKeypoints(image_id);
        const PanoramaIndexs& panorama_indices = feature_data_container_->GetPanoramaIndexs(image_id);
        
        images.back().SetPoints2D(keypoints);
        std::vector<uint32_t> local_image_indices(keypoints.size());
        for(size_t i = 0; i<keypoints.size(); ++i){
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                    local_image_indices[i] = image_id;
            }
            else{
			    local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
		}
        images.back().SetLocalImageIndices(local_image_indices);
    }

    for (const auto& image : images) {
        if (!scene_graph_->ExistsImage(image.ImageId())) {
            scene_graph_->AddImage(image);
        }

        const auto& camera = feature_data_container_->GetCamera(image.CameraId());
        if (!scene_graph_->ExistsCamera(image.CameraId())) {
            scene_graph_->AddCamera(camera);
        }
    }
    std::cout << StringPrintf("Add Images and Cameras: %d, Time Cost: %fsec\n", scene_graph_->NumImages(), timer1.ElapsedSeconds());

    Timer prealloc_timer;
    prealloc_timer.Start();

    double mean_corrs_per_image = 0.0f;
    std::vector<int> max_corrs_per_image(max_image_id + 1, 0);
    std::vector<std::vector<unsigned short> > num_corrs_per_keypoint(max_image_id + 1);
    for (size_t i = 0; i < results_.size(); ++i) {
        const FeatureMatcherData& fmdata = results_[i];
        image_t image_id1 = fmdata.image_id1;
        image_t image_id2 = fmdata.image_id2;
        max_corrs_per_image[image_id1]++;
        max_corrs_per_image[image_id2]++;
        mean_corrs_per_image += 2;

        if (num_corrs_per_keypoint[image_id1].empty()) {
            const FeatureKeypoints& keypoints = feature_data_container_->GetKeypoints(image_id1);
            num_corrs_per_keypoint[image_id1].resize(keypoints.size(), 0);
        }
        if (num_corrs_per_keypoint[image_id2].empty()) {
            const FeatureKeypoints& keypoints = feature_data_container_->GetKeypoints(image_id2);
            num_corrs_per_keypoint[image_id2].resize(keypoints.size(), 0);
        }
        for (const auto & match : fmdata.matches) {
            num_corrs_per_keypoint[image_id1][match.point2D_idx1]++;
            num_corrs_per_keypoint[image_id2][match.point2D_idx2]++;
        }
    }
    if (image_ids.size() > 0) mean_corrs_per_image /= image_ids.size();
    int max_corrs = *std::max_element(max_corrs_per_image.begin(), max_corrs_per_image.end());

    std::cout << "Match Pairs:          " << results_.size() << std::endl;
    std::cout << "Mean Corrs Per Image: " << mean_corrs_per_image << std::endl;
    std::cout << "Max Corrs:            " << max_corrs << std::endl;

    const double G_byte = 1024 * 1024 * 1024;
    double consumed_memeory = 0;
    for (const auto& image_id : image_ids) {
        for (auto num_corr : num_corrs_per_keypoint[image_id]) {
            consumed_memeory += num_corr;
        }
    }
    consumed_memeory = consumed_memeory * 8 / G_byte;
    std::cout << "Estimate Memory Consumption: " << consumed_memeory << "GB" << std::endl;
    float avaliable_memeory = 0;
    GetAvailableMemory(avaliable_memeory);
    std::cout << "Available Memory: " << avaliable_memeory << "GB" << std::endl;

    // for (size_t i = 0; i < results_.size(); ++i) {
    //     const FeatureMatcherData& fmdata = results_[i];
    //     image_t image_id1 = fmdata.image_id1;
    //     image_t image_id2 = fmdata.image_id2;
    //     if (!scene_graph_->CorrespondenceGraph()->ExistsImage(image_id1) ||
    //         !scene_graph_->CorrespondenceGraph()->ExistsImage(image_id2)) {
    //         continue;
    //     }
    //     auto & image1 = scene_graph_->CorrespondenceGraph()->ImageRef(image_id1);
    //     auto & image2 = scene_graph_->CorrespondenceGraph()->ImageRef(image_id2);
    //     for (const auto & match : fmdata.matches) {
    //         image1.corrs[match.point2D_idx1].reserve(num_corrs_per_keypoint[image_id1][match.point2D_idx1]);
    //         image2.corrs[match.point2D_idx2].reserve(num_corrs_per_keypoint[image_id2][match.point2D_idx2]);
    //     }
    // }
    for (auto image_id : image_ids) {
        if (!scene_graph_->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }
        auto & image = scene_graph_->CorrespondenceGraph()->ImageRef(image_id);
        for (point2D_t point_idx = 0; point_idx < image.corrs.size(); ++point_idx) {
            if (!num_corrs_per_keypoint[image_id].empty() && 
                num_corrs_per_keypoint[image_id][point_idx] > 0) {
                image.corrs[point_idx].reserve(num_corrs_per_keypoint[image_id][point_idx]);
            }
        }
    }

    std::cout << "Prealloc Cost Time: " << prealloc_timer.ElapsedSeconds() << "sec" << std::endl;
    GetAvailableMemory(avaliable_memeory);
    std::cout << "Available Memory: " << avaliable_memeory << "GB" << std::endl;

    Timer timer2;
    timer2.Start();
    // add correspondences and edges in view graph
    size_t num_ignored_by_check_image_pairs = 0;
    size_t num_ignored_by_id_image_pairs = 0;
    for (size_t i = 0; i < results_.size(); ++i) {
        const FeatureMatcherData& fmdata = results_[i];
        if (inlier_matches_[i] && UseInlierMatchesCheck(fmdata.two_view_geometry)) {
            image_t image_id1 = fmdata.image_id1;
            image_t image_id2 = fmdata.image_id2;
            if (!scene_graph_->CorrespondenceGraph()->ExistsImage(image_id1) ||
                !scene_graph_->CorrespondenceGraph()->ExistsImage(image_id2)) {
                continue;
            }

            const auto& image1 = feature_data_container_->GetImage(image_id1);
            const auto& camera1 = feature_data_container_->GetCamera(image1.CameraId());
            const auto& image2 = feature_data_container_->GetImage(image_id2);
            const auto& camera2 = feature_data_container_->GetCamera(image2.CameraId());

            bool remove_redundant = !(camera1.NumLocalCameras() > 2 && camera2.NumLocalCameras() > 2);

            if (image_id_set.count(image_id1) > 0 && image_id_set.count(image_id2) > 0) {
                scene_graph_->CorrespondenceGraph()->AddCorrespondences(
                    image_id1, image_id2, fmdata.two_view_geometry, 
                    remove_redundant);
            } else {
                num_ignored_by_id_image_pairs += 1;
            }
        } else {
            num_ignored_by_check_image_pairs += 1;
        }
    }
    std::cout << "Add Correspondences Cost Time: " << timer2.ElapsedSeconds() << "sec" << std::endl;

    correspondence_graph->Finalize();


    SequentialMatchingOptions sqm_options = options_.SequentialMatching();
    if ((options_.Method() == FeatureMatchingOptions::MatchMethod::SEQUENTIAL || 
         options_.Method() == FeatureMatchingOptions::MatchMethod::HYBRID) && 
        sqm_options.robust_loop_detection && !sqm_options.loop_detection_before_sequential_matching) {
        SequentialLoopClosing();
    }

    // {
    //     for (const auto & image_pair : scene_graph_->CorrespondenceGraph()->ImagePairs()) {
    //         image_t image_id1, image_id2;
    //         utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
    //         std::string image_name1 = scene_graph_->Image(image_id1).Name();
    //         std::string image_name2 = scene_graph_->Image(image_id2).Name();
    //         std::string text = StringPrintf("%s+%s: %d %.3f\n", image_name1.c_str(), image_name2.c_str(), 
    //                                         image_pair.second.num_correspondences, image_pair.second.two_view_geometry.confidence);
    //         std::cout << text << std::endl;
    //     }
    // }

    scene_graph_->CorrespondenceGraph()->UpdateImagePairsConfidence();

    bool remove_ambiguous = !(options_.pair_matching_.self_matching && options_.pair_matching_.sub_matching) && !options_.map_update;

    Timer timer3;
    timer3.Start();
    auto tracks = scene_graph_->CorrespondenceGraph()->GenerateTracks(options_.min_track_degree, remove_ambiguous, true);

    std::cout << "GenerateTracks Cost Time: " << timer3.ElapsedSeconds() << "sec" << std::endl;

    // Perform track selection 
    if (options_.track_preoperation) {

        Timer timer4;
        timer4.Start();

        std::sort(tracks.begin(), tracks.end(), 
            [](const Track& t1, const Track& t2) {
                return t1.Length() > t2.Length();
            });
        std::cout << "Track Sort Time Cost: " << timer4.ElapsedSeconds() << "sec" << std::endl;

        const auto& old_image_ids = feature_data_container_->GetOldImageIds();
        std::unordered_set<image_t> const_image_set;
        if (!old_image_ids.empty()){
            const_image_set.insert(old_image_ids.begin(), old_image_ids.end());
            std::cout << "TrackSelection const_image_ids: " << const_image_set.size() << std::endl;
        }

        std::vector<unsigned char> inlier_masks(tracks.size(), 0);
        scene_graph_->DistributedTrackSelection(tracks, inlier_masks, options_.track_block_radius,
                options_.track_max_per_block, options_.track_min_per_block, options_.track_max_cover_per_view, const_image_set);

        std::cout << "TrackSelection Time Cost: " << timer4.ElapsedSeconds() << "sec" << std::endl;
        timer4.PrintSeconds();
        //    scene_graph_->DistributeTrack(tracks, inlier_masks,
        //                                  options_.select_range_match_point,
        //                                  const_image_set);

        // scene_graph_->CorrespondenceGraph()->TrackSelection(tracks, inlier_masks,
        //    options_.max_cover_per_view, const_image_set);
    }
    // Set number of observations and correspondences per image.
    auto & scene_images = scene_graph_->Images();
    for (auto& image : scene_images) {
        if (!correspondence_graph->ExistsImage(image.first)) {
            continue;
        }
        image.second.SetNumObservations(scene_graph_->CorrespondenceGraph()->NumObservationsForImage(image.first));
        image.second.SetNumCorrespondences(
            scene_graph_->CorrespondenceGraph()->NumCorrespondencesForImage(image.first));
    }

    Timer timer5;
    timer5.Start();
    // Set overlap flag of keypoints.
    for (auto& image : scene_images) {
        if (!correspondence_graph->ExistsImage(image.first)) {
            continue;
        }
        const FeatureMatches& corrs = 
        correspondence_graph->FindCorrespondencesBetweenImages(image.first, image.first);
        for (const FeatureMatch& corr : corrs) {
            image.second.Point2D(corr.point2D_idx1).SetOverlap(true);
            image.second.Point2D(corr.point2D_idx2).SetOverlap(true);
        }
    }
    std::cout << "Set Overlap Time Cost: " << timer5.ElapsedSeconds() << "sec" << std::endl;

    if (options_.delete_duplicated_images_) {
        Timer timer6;
        timer6.Start();
        const std::vector<image_t> new_image_ids = feature_data_container_->GetNewImageIds();
        // const std::unordered_set<image_t> new_image_set(new_image_ids.begin(), new_image_ids.end());
        std::vector<bool> new_image_flag(max_image_id + 1, false);
        for (auto image_id : new_image_ids) {
            new_image_flag[image_id] = true;
        }
        const auto & image_pairs = correspondence_graph->ImagePairs();
        std::unordered_set<image_t> deleted_images;
        for (const auto & image_pair : image_pairs) {
            image_t image_id1, image_id2;
            sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
            const auto& image1 = feature_data_container_->GetImage(image_id1);
            const auto& image2 = feature_data_container_->GetImage(image_id2);
            const auto& camera1 = feature_data_container_->GetCamera(image1.CameraId());
            const auto& camera2 = feature_data_container_->GetCamera(image2.CameraId());
            if (camera1.NumLocalCameras() != camera2.NumLocalCameras()) {
                continue;
            }
            if (!new_image_flag[image_id2]) {
                // only new images can be deleted
                continue;
            }

            point2D_t num_corrs12 = correspondence_graph->NumCorrespondencesBetweenImages(image_id1, image_id2);
            // std::cout << image1.Name() << " <-> " << image2.Name() << std::endl;

            auto two_view_geometry = image_pair.second.two_view_geometry;
            Eigen::Matrix3d R = QuaternionToRotationMatrix(two_view_geometry.qvec);
            Eigen::AngleAxisd angle_axis(R);
            double R_angle = RadToDeg(angle_axis.angle());
            // std::cout << "two view geometry: " << R_angle << ", " << two_view_geometry.tvec.transpose() << std::endl;
            if (R_angle <= 1e-6 && two_view_geometry.tvec.norm() <= 1e-6) {
                point2D_t max_num_corrs = 0;
                std::unordered_set<image_t> image_neighbors = correspondence_graph->ImageNeighbor(image_id1);
                for (auto neighbor_id : image_neighbors) {
                    point2D_t num_corrs = correspondence_graph->NumCorrespondencesBetweenImages(image_id1, neighbor_id);
                    max_num_corrs = std::max(max_num_corrs, num_corrs);
                }
                float ratio = num_corrs12 * 1.0 / max_num_corrs;
                if (ratio > 0.99) {
                    deleted_images.insert(image_id2);
                }
            }
        }
        for (auto image_id : deleted_images) {
            const auto image = feature_data_container_->GetImage(image_id);
            std::cout << image.Name() << std::endl;
            scene_graph_->DeleteImage(image_id);
            feature_data_container_->DeleteImage(image_id);
        }
        std::cout << StringPrintf("Delete %d duplicated images! Time Cost: %fsec\n", deleted_images.size(), timer6.ElapsedSeconds());
    }

    std::cout << StringPrintf(" in %.3fs (ignored %d by check, %d by id)", timer.ElapsedSeconds(),
                              num_ignored_by_check_image_pairs, num_ignored_by_id_image_pairs)
              << std::endl;
}



void FeatureMatcher::SequentialLoopClosing() {
    Timer timer;

    std::shared_ptr<class CorrespondenceGraph> correspondence_graph = scene_graph_->CorrespondenceGraph();

    std::vector<image_t> image_ids =
        GetOrderedImageIds(feature_data_container_, feature_data_container_->GetNewImageIds());

    image_ids.erase(std::remove_if(image_ids.begin(), image_ids.end(),
                                   [=](image_t id) { return !correspondence_graph->ExistsImage(id); }),
                    image_ids.end());

    SequentialMatchingOptions sqm_options = options_.SequentialMatching();

    if (image_ids.size() <= sqm_options.overlap) {
        return;
    }

    int num_connected_images = sqm_options.overlap;
    int max_num_features = sqm_options.loop_detection_max_num_features;

    std::vector<std::vector<retrieval::ImageScore>> image_image_scores(image_ids.size());

    if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::VLAD) {
        VladVisualIndex vlad_visual_index;
        vlad_visual_index.LoadCodeBook(sqm_options.vlad_code_book_path);
        
        timer.Start();
        std::cout<<"Load vlad vectors from data container to vlad visual index:"<<std::endl;
        //Load vlad vectors from data container to vlad visual index
        for(int i = 0; i < image_ids.size(); ++i){
            image_t current_id = image_ids[i];        
            const auto& vlad = feature_data_container_->GetVladVector(current_id);
            vlad_visual_index.ADD(vlad,current_id);
        }
        std::cout << StringPrintf("Indexing in %.3f seconds", timer.ElapsedSeconds()) << std::endl;

        VladVisualIndex::QueryOptions query_option;
        query_option.max_num_images = sqm_options.loop_detection_num_images * 10;
        
        std::cout << "Query " << std::endl;
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            const Image& image = feature_data_container_->GetImage(current_id);
            /// query by voc tree
            std::cout<<"query ["<<i<<"/"<<image_ids.size()<<"] image"<<std::endl;
            std::vector<retrieval::ImageScore> image_scores;
            const VladVisualIndex::VLAD& current_vlad = feature_data_container_->GetVladVector(current_id); 
            vlad_visual_index.Query(query_option,current_vlad,&image_scores);

            std::vector<retrieval::ImageScore> hybrid_image_scores(sqm_options.loop_detection_num_images);
            size_t retrieved_num = 0;

            // loop cross the sequence
            for (size_t j = 1; j < image_scores.size(); ++j) {
                const Image& image_neighbor = feature_data_container_->GetImage(image_scores[j].image_id);
                if (image_neighbor.LabelId() != image.LabelId()) {
                    hybrid_image_scores[retrieved_num++] = image_scores[j];
                }
                if (retrieved_num >= sqm_options.loop_detection_num_images * 2 / 5) {
                    break;
                }
            }

            // loop cross the sequence or in the same sequence
            for (size_t j = 1; j < image_scores.size(); ++j) {
                const Image& image_neighbor = feature_data_container_->GetImage(image_scores[j].image_id);
                if (image_neighbor.LabelId() == image.LabelId()) {
                    hybrid_image_scores[retrieved_num++] = image_scores[j];
                }
                if (retrieved_num >= sqm_options.loop_detection_num_images) {
                    break;
                }
            }
            hybrid_image_scores.resize(retrieved_num);

            { image_image_scores[i].swap(hybrid_image_scores); }
        }

        std::cout << StringPrintf("Query in %.3fmin", timer.ElapsedMinutes()) << std::endl;
    } else if (options_.retrieve_type == FeatureMatchingOptions::RetrieveType::SIFT) {
        timer.Start();

        retrieval::VisualIndex<> visual_index;
        visual_index.Read(sqm_options.vocab_tree_path);

        IndexImagesInVisualIndex(sqm_options.loop_detection_num_threads, sqm_options.loop_detection_num_checks,
                                 sqm_options.loop_detection_max_num_features, image_ids, feature_data_container_,
                                 &visual_index);

        std::cout << StringPrintf("Indexing in %.3f min", timer.ElapsedMinutes()) << std::endl;

        //--- need to test
        retrieval::VisualIndex<>::QueryOptions query_options;
        query_options.max_num_images = sqm_options.overlap * 5;
        query_options.num_neighbors = 3;
        query_options.num_images_after_verification = 0;  // query_options.max_num_images;

        timer.Start();
        std::cout << "Query " << std::endl;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            /// query by voc tree

            std::vector<retrieval::ImageScore> image_scores;
            auto keypoints = feature_data_container_->GetKeypoints(current_id);
            auto descriptors = feature_data_container_->GetCompressedDescriptors(current_id);
            if (max_num_features > 0 && descriptors.rows() > max_num_features) {
                ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
            }

            visual_index.Query(query_options, keypoints, descriptors, &image_scores);
            image_image_scores[i].swap(image_scores);
        }
        std::cout << StringPrintf("Query in %.3fmin", timer.ElapsedMinutes()) << std::endl;
    }

    /// matching the candidate pairs
    MatchDataContainer match_data_container;
    std::vector<std::pair<image_t, image_t>> image_pairs;

    typedef std::pair<std::set<image_t>, int> ConsistentGroup;
    std::vector<ConsistentGroup> consistent_groups;

    // -> * * * * * * * * * * * * * * * * * * * * * * * *  sequential ids
    //                                                  |  current_id
    //                                        | | | | |    recent_ids
    //                                |                    max_loop_id
    // -> * * * * * * * * * * * * * * *                    ids in searching area
    //    | | | |         | |                              candidates
    //      | | |                                          connected_candidates
    //      | |                                            consistent_candidates

    std::unordered_set<image_pair_t> existed_loop_pairs;

    for (int i = num_connected_images; i < image_ids.size(); ++i) {
        image_t current_id = image_ids[i];

        std::cout << "Frame#" << current_id << std::endl;

        std::vector<image_t> recent_ids;
        recent_ids.reserve(num_connected_images*2 + 1);
        for (int j = -num_connected_images; j <= num_connected_images; ++j) {
            if((i + j >= 0) && (i + j < image_ids.size()) && (j!=0)){
                recent_ids.push_back(image_ids[i + j]);
            }
        }

        image_t max_loop_id = image_ids[i - num_connected_images];

        /// query by voc tree
        std::list<retrieval::ImageScore> candidate_scores;

        /// get ids in searching area
        float max_recent_score = -1;
        int max_recent_id = -1;
        for (auto& image_score : image_image_scores[i]) {
            image_t image_id = image_score.image_id;
            if (image_id >= current_id) {
                continue;
            } else if (image_id <= max_loop_id) {  // is candidate
                candidate_scores.push_back(image_score);
            } else if (std::find(recent_ids.begin(), recent_ids.end(), image_id) !=
                       recent_ids.end()) {  // is recent image, compute max_recent_score
                if (image_score.score > max_recent_score) {
                    max_recent_score = image_score.score;
                    max_recent_id = image_id;
                }
            }
        }

        std::cout << "  => Query candidate num: " << candidate_scores.size() << std::endl;
        if (candidate_scores.empty()) {
            consistent_groups.clear();
            continue;
        }

        std::cout<<"max recent score: "<<max_recent_score<<std::endl;
        std::cout<<"max recent id: "<<max_recent_id<<std::endl;
        // filter by max_recent_score * max_recent_score_factor
        float min_score = max_recent_score * sqm_options.max_recent_score_factor;
        candidate_scores.remove_if([&](retrieval::ImageScore s) {
            return s.score < min_score;
        });

        std::cout << "  => Filtered candidate nums " << candidate_scores.size() << std::endl;
        if (candidate_scores.empty()) {
            consistent_groups.clear();
            continue;
        }

        // filter by best_score * 0.8
        std::unordered_set<image_t> connected_candidates;
        std::list<std::pair<float, image_t>> acc_score_image;
        float best_acc_score = min_score;
        for (auto ii = candidate_scores.begin(); ii != candidate_scores.end(); ++ii) {
            image_t image_i = ii->image_id;
            auto neighbors_i = correspondence_graph->ImageNeighbor(image_i);

            float best_score = ii->score;
            float acc_score = ii->score;
            image_t best_id = image_i;
            for (auto ij = candidate_scores.begin(); ij != candidate_scores.end(); ++ij) {
                if (ii == ij) {
                    continue;
                }
                if (neighbors_i.find(ij->image_id) != neighbors_i.end()) {
                    acc_score += ij->score;
                    if (ij->score > best_score) {
                        best_score = ij->score;
                        best_id = ij->image_id;
                    }
                }
            }
            acc_score_image.emplace_back(acc_score, best_id);
            if (acc_score > best_acc_score) {
                best_acc_score = acc_score;
            }
        }
        // float min_retained_score = best_acc_score * 0.75;
        float min_retained_score = best_acc_score * sqm_options.best_acc_score_factor;
        for (auto& asi : acc_score_image) {
            if (asi.first > min_retained_score) {
                connected_candidates.insert(asi.second);
            }
        }

        std::cout << "  => Connected candidate nums " << connected_candidates.size() << std::endl;
        if (connected_candidates.empty()) {
            consistent_groups.clear();
            continue;
        }

        /// consistency check
        std::vector<image_t> consistent_candidates;
        std::vector<ConsistentGroup> current_consistent_group;
        std::vector<bool> consistnet_group_flags(consistent_groups.size(), false);

        for (auto& candidate_id : connected_candidates) {
            auto neighbors_j = correspondence_graph->ImageNeighbor(candidate_id);
            std::set<image_t> candidate_group;
            for (auto& neighbor : neighbors_j) {
                if (neighbor < candidate_id) {
                    candidate_group.insert(neighbor);
                }
            }
            candidate_group.insert(candidate_id);

            bool enough_consistent = false;
            bool consistent_for_some_group = false;

            for (int j = 0; j < consistent_groups.size(); ++j) {
                auto& prev_group = consistent_groups[j].first;

                bool consistent = false;
                for (auto& id_in_group : candidate_group) {
                    if (prev_group.count(id_in_group)) {
                        consistent = true;
                        consistent_for_some_group = true;
                        break;
                    }
                }

                if (consistent) {
                    int prev_consistency = consistent_groups[j].second;
                    int curr_consistency = prev_consistency + 1;
                    if (!consistnet_group_flags[j]) {
                        current_consistent_group.emplace_back(candidate_group, curr_consistency);
                        consistnet_group_flags[j] = true;
                    }
                    if (curr_consistency >= sqm_options.loop_consistency_threshold && !enough_consistent) {
                        consistent_candidates.push_back(candidate_id);
                        enough_consistent = true;
                    }
                }
            }
            if (!consistent_for_some_group) {
                current_consistent_group.emplace_back(candidate_group, 1);
            }
        }

        consistent_groups = current_consistent_group;

        std::cout << "  => Consistent candidate nums " << consistent_candidates.size() << std::endl;
        if (consistent_candidates.empty()) {
            continue;
        }

        image_pairs.reserve(image_pairs.size() + consistent_candidates.size());
        for (auto& candidate_id : consistent_candidates) {

            image_pair_t image_pair_id = sensemap::utility::ImagePairToPairId(candidate_id, current_id);
            if(existed_loop_pairs.count(image_pair_id)>0){
                continue;
            }
            existed_loop_pairs.insert(image_pair_id);

            if (current_id > candidate_id) {
                image_pairs.emplace_back(candidate_id, current_id);
            } else {
                image_pairs.emplace_back(current_id, candidate_id);
            }
            // update neighbors temporarily
            correspondence_graph->UpdateImageNeighbors(current_id, candidate_id);
        }
    }

    matcher_.Match(image_pairs, &match_data_container);
    auto& results = match_data_container.GetMatchResults();

    /// add pairs to correspondence graph
    size_t min_num_matches = options_.pair_matching_.min_num_inliers;
    auto UseInlierMatchesCheck = [min_num_matches](const TwoViewGeometry& two_view_geometry) {
        return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >= min_num_matches;
    };

    for (const auto& fmdata : results) {
        if (UseInlierMatchesCheck(fmdata.two_view_geometry)) {
            image_t image_id1 = fmdata.image_id1;
            image_t image_id2 = fmdata.image_id2;

            if (!correspondence_graph->ExistsImage(image_id1) || !correspondence_graph->ExistsImage(image_id2)) {
                continue;
            }

            if (correspondence_graph->ExistImagePair(image_id1, image_id2)) {
                continue;
            }
            double match_covered_area_ratio;
            const PanoramaIndexs& panorama_indices1 = feature_data_container_->GetPanoramaIndexs(image_id1);
            const PanoramaIndexs& panorama_indices2 = feature_data_container_->GetPanoramaIndexs(image_id2);

            const PieceIndexs& piece_indices1 = feature_data_container_->GetPieceIndexs(image_id1);
            const PieceIndexs& piece_indices2 = feature_data_container_->GetPieceIndexs(image_id2);

            std::cout << "loop pair: ";
            std::cout << feature_data_container_->GetImage(image_id1).Name() << " "
                      << feature_data_container_->GetImage(image_id2).Name() << " ";

            const Image& image1 = feature_data_container_->GetImage(image_id1);
            const Camera& Camera1 = feature_data_container_->GetCamera(image1.CameraId());

            const Image& image2 = feature_data_container_->GetImage(image_id2);
            const Camera& Camera2 = feature_data_container_->GetCamera(image2.CameraId());

            if (Camera1.NumLocalCameras() > 1 || Camera1.ModelName().compare("SPHERICAL") == 0 ||
                Camera2.NumLocalCameras() > 1 || Camera2.ModelName().compare("SPHERICAL") == 0) {

                bool piecewise = false;
                int sub_image_count1,sub_image_count2;
                if (Camera1.NumLocalCameras() == 2 && options_.pair_matching_.convert_to_perspective_image) {
                    piecewise = true;
                    sub_image_count1 = options_.pair_matching_.perspective_image_count;
                } else if (Camera1.ModelName().compare("SPHERICAL") == 0 &&
                        options_.pair_matching_.convert_to_perspective_image) {
                    sub_image_count1 = options_.pair_matching_.perspective_image_count;
                } else {
                    sub_image_count1 = Camera1.NumLocalCameras();
                }

                if (Camera2.NumLocalCameras() == 2 && options_.pair_matching_.convert_to_perspective_image) {
                    piecewise = true;
                    sub_image_count2 = options_.pair_matching_.perspective_image_count;
                } else if (Camera1.ModelName().compare("SPHERICAL") == 0 &&
                        options_.pair_matching_.convert_to_perspective_image) {
                    sub_image_count2 = options_.pair_matching_.perspective_image_count;
                } else {
                    sub_image_count2 = Camera2.NumLocalCameras();
                }

                if (MatchFromRepetitivePattern(fmdata.two_view_geometry.inlier_matches, panorama_indices1,
                                            panorama_indices2, piece_indices1, piece_indices2, match_covered_area_ratio,
                                            options_.pair_matching_.min_covered_sub_image_ratio, sub_image_count1,sub_image_count2,
                                            options_.pair_matching_.min_matched_feature_per_piece,
                                            piecewise)) {
                    std::cout << "have repetitive pattern" << std::endl;
                    continue;
                }
            }

            if (options_.have_prior_pose_) {
                CHECK(options_.prior_pose_validations.find(image_id1) != options_.prior_pose_validations.end());
                CHECK(options_.prior_pose_validations.find(image_id2) != options_.prior_pose_validations.end());

                if (!options_.prior_pose_validations.at(image_id1) || !options_.prior_pose_validations.at(image_id2)) {
                    std::cout << "invalid prior pose, skip the match" << std::endl;
                    continue;
                }
            }

            bool match_from_far_place = false;
            if (options_.have_prior_pose_) {
                if (options_.prior_rotations.find(image_id1) != options_.prior_rotations.end() &&
                    options_.prior_rotations.find(image_id2) != options_.prior_rotations.end()) {
                    Eigen::Vector4d qvec1 = options_.prior_rotations.at(image_id1);
                    Eigen::Vector3d tvec1 = options_.prior_translations.at(image_id1);

                    Eigen::Vector4d qvec2 = options_.prior_rotations.at(image_id2);
                    Eigen::Vector3d tvec2 = options_.prior_translations.at(image_id2);

                    Eigen::Vector3d C1 = ProjectionCenterFromPose(qvec1, tvec1);
                    Eigen::Vector3d C2 = ProjectionCenterFromPose(qvec2, tvec2);

                    double distance = (C1 - C2).norm();
                    if (distance > options_.max_match_distance * options_.prior_neighbor_distance) {
                        match_from_far_place = true;
                    }
                }
            }
            if (match_from_far_place) {
                std::cout << "match from far place" << std::endl;
                continue;
            }
            std::cout<<std::endl;

            const bool remove_redundant = !(Camera1.NumLocalCameras() > 2);
            correspondence_graph->AddCorrespondences(image_id1, image_id2, fmdata.two_view_geometry, remove_redundant);
        }
    }
    correspondence_graph->Finalize();
}

}  // namespace sensemap
