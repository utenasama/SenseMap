#ifndef MAP_SPARSIFICATION_SAMPLER_BASE_H_
#define MAP_SPARSIFICATION_SAMPLER_BASE_H_

#include <string>
#include <util/types.h>
#include "base/reconstruction.h"
#include <opencv2/opencv.hpp>

namespace sensemap {

class SamplerBase {
public:
    SENSEMAP_POINTER_TYPEDEFS(SamplerBase);

    enum class Type {
        // Basically selects all landmarks.
                kNoSampling = 0,
        // Randomly selects a desired number of landmarks.
                kRandom = 1,
        // Uses heuristics such as the number of landmark observations and the
        // number of landmarks observations in keyframes to select a desired number
        // of landmarks.
                kHeuristic = 2,
        kLpsolveIlp = 3,
        // If the map is too large, partition the map using METIS and the solve
        // and ILP problem.
                kLpsolvePartitionIlp = 4,
    };

    virtual ~SamplerBase() {}

    void sample(const Reconstruction &reconstruction,
                const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature,
                unsigned int desired_num_landmarks,
                std::unordered_set<mappoint_t> &summary_store_landmark_ids);

    virtual void sampleMapSegment(
            const Reconstruction &reconstruction,
            const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature,
            unsigned int desired_num_landmarks,
            unsigned int time_limit_seconds,
            const std::unordered_set<mappoint_t> &segment_store_landmark_id_set,
            const std::vector<image_t> &segment_vertex_id_list,
            std::unordered_set<mappoint_t> &summary_store_landmark_ids) = 0;

    virtual std::string getTypeString() const = 0;
};

}  // namespace sensemap
#endif  // MAP_SPARSIFICATION_SAMPLER_BASE_H_
