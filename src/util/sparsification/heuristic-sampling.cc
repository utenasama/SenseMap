
#include "heuristic-sampling.h"
#include <algorithm>
#include <unordered_map>

namespace sensemap {
namespace sampling {

void LandmarkSamplingWithCostFunctions::registerScoringFunction(const ScoringFunction::ConstPtr &scoring) {
    scoring_functions_.push_back(scoring);
}

void LandmarkSamplingWithCostFunctions::registerCostFunction(const SamplingCostFunction::ConstPtr &cost) {
    cost_functions_.push_back(cost);
}

void LandmarkSamplingWithCostFunctions::registerDescriptorScoringFunction(
    const descriptorscoring::ORBDescriptorScoringFunction::ConstPtr &despscoring) {
    descriptor_scoring_functions_.push_back(despscoring);
}

void LandmarkSamplingWithCostFunctions::sampleMapSegment(
    const Reconstruction &map,
    const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature,
    unsigned int desired_num_landmarks, unsigned int /*time_limit_seconds*/,
    const std::unordered_set<mappoint_t> &segment_landmark_id_set, const std::vector<image_t> &segment_vertex_id_list,
    std::unordered_set<mappoint_t> &summary_landmark_ids) {
    CHECK(summary_landmark_ids.empty());
    summary_landmark_ids = segment_landmark_id_set;

    std::cout << "Mappoint score calculation ..." << std::endl;
    LandmarkScoreMap landmark_scores;
    for (const auto &landmark_id : summary_landmark_ids) {
        double landmark_score = 0.0;
        for (auto &scoring_function : scoring_functions_) {
            landmark_score += (*scoring_function)(landmark_id, map);
        }
        for (auto &scoring_function : descriptor_scoring_functions_) {
            landmark_score += (*scoring_function)(landmark_id, map, orb_feature);
        }
        CHECK(landmark_scores.emplace(landmark_id, landmark_score).second);
    }

    std::cout << "Image score calculation ..." << std::endl;
    KeyframeKeypointCountMap keyframe_keypoint_counts;
    for (const auto &vertex_id : segment_vertex_id_list) {
        auto vertex_all_observed_landmark_ids = map.Image(vertex_id).Points2D();

        unsigned int num_valid_keypoints = 0;
        for (const auto &landmark_id : vertex_all_observed_landmark_ids) {
            if (landmark_id.HasMapPoint()) {
                if (segment_landmark_id_set.count(landmark_id.MapPointId()) > 0) {
                    ++num_valid_keypoints;
                }
            }
        }
        CHECK(keyframe_keypoint_counts.emplace(vertex_id, num_valid_keypoints).second);
    }

    const size_t kNumInitialLandmarks = summary_landmark_ids.size();
    const int kNumLandmarksToRemove = summary_landmark_ids.size() - desired_num_landmarks;
    std::cout << "Will remove " << kNumLandmarksToRemove << " out of " << summary_landmark_ids.size() << " landmarks."
              << std::endl;

    while (summary_landmark_ids.size() > desired_num_landmarks) {
        // Remove half but no more than:
        // * number left to reach the desired number
        // * 10 percent of the initial landmark number
        const unsigned int num_landmarks_to_remove =
            std::min(std::min(summary_landmark_ids.size() / 2u, summary_landmark_ids.size() - desired_num_landmarks),
                     kNumInitialLandmarks / 10 + 1);

        std::cout << "Iteration: will remove " << num_landmarks_to_remove << " landmark IDs out of "
                  << summary_landmark_ids.size() << std::endl;

        unsigned int num_landmarks_removed_in_iter = 0;
        while (num_landmarks_removed_in_iter < num_landmarks_to_remove) {
            std::cout << "\r";
            std::cout << "\tInner iteration, already removed: [ " << num_landmarks_removed_in_iter << " / "
                      << num_landmarks_to_remove << " ]" << std::flush;

            LandmarkScoreMap landmark_costs;
            unsigned int num_zero_cost_landmarks = 0;
            // Evaluate cost for each (still present) landmark.
            for (const mappoint_t &landmark_id : summary_landmark_ids) {
                double landmark_cost_values = 0.0;
                for (auto &cost_function : cost_functions_) {
                    landmark_cost_values += (*cost_function)(landmark_id, map, keyframe_keypoint_counts);
                }
                CHECK(landmark_costs.emplace(landmark_id, landmark_cost_values).second);
            }

            std::vector<StoreLandmarkIdScorePair> sorted_scores_and_costs;
            for (const mappoint_t &landmark_id : summary_landmark_ids) {
                // Cost is added as it represents the cost of removal of a particular
                // landmark.
                double score_and_cost = landmark_scores[landmark_id] + landmark_costs[landmark_id];
                sorted_scores_and_costs.emplace_back(landmark_id, score_and_cost);
            }

            // Sort the vector with scores, descending.
            std::sort(sorted_scores_and_costs.begin(), sorted_scores_and_costs.end(),
                      [](const StoreLandmarkIdScorePair &lhs, const StoreLandmarkIdScorePair &rhs) {
                          return lhs.second > rhs.second;
                      });

            // Count zero cost landmarks within this inner iteration.
            for (unsigned int i = sorted_scores_and_costs.size() - 1;
                 i >= sorted_scores_and_costs.size() - num_landmarks_to_remove; --i) {
                const mappoint_t &store_landmark_id = sorted_scores_and_costs[i].first;
                if (landmark_costs[store_landmark_id] == 0.0) {
                    ++num_zero_cost_landmarks;
                }
            }

            std::unordered_set<image_t> sparsified_keyframes;
            // Cut kNumLandmarkToRemovePerIter with worst total score.
            for (unsigned int i = sorted_scores_and_costs.size() - 1;
                 i >= sorted_scores_and_costs.size() - num_landmarks_to_remove; --i) {
                const mappoint_t &store_landmark_id = sorted_scores_and_costs[i].first;

                // There are still some landmarks with zero cost of removal, so
                // no need to remove the current one. Let's continue.
                if (num_zero_cost_landmarks > 0u && landmark_costs[store_landmark_id] > 0.0) {
                    continue;
                }

                std::unordered_set<image_t> store_landmark_id_observers;
                auto cur_track_elements = map.MapPoint(store_landmark_id).Track().Elements();
                for (const auto &track_element : cur_track_elements) {
                    store_landmark_id_observers.insert(track_element.image_id);
                }

                bool are_landmark_observers_already_sparsified = false;
                for (const image_t &vertex_id : store_landmark_id_observers) {
                    if (sparsified_keyframes.count(vertex_id) > 0u) {
                        // At least one of the observer keyframes of the current landmark
                        // was already affected by previous landmark deletions.
                        are_landmark_observers_already_sparsified = true;
                        break;
                    }
                }

                if (!are_landmark_observers_already_sparsified) {
                    ++num_landmarks_removed_in_iter;

                    sparsified_keyframes.insert(store_landmark_id_observers.begin(), store_landmark_id_observers.end());

                    for (const image_t &vertex_id : store_landmark_id_observers) {
                        // Verify if the vertex belongs to the current segment.
                        if (keyframe_keypoint_counts.count(vertex_id) > 0u) {
                            --keyframe_keypoint_counts[vertex_id];
                        }
                    }
                    if (summary_landmark_ids.size() > desired_num_landmarks) {
                        summary_landmark_ids.erase(sorted_scores_and_costs[i].first);
                    } else {
                        break;
                    }
                }
            }
        }
        std::cout << std::endl;
    }
    std::cout << "Summary size = " << summary_landmark_ids.size() << std::endl;
}

}  // namespace sampling
}  // namespace sensemap
