
#include "sampler-base.h"
namespace sensemap {

void SamplerBase::sample(
        const Reconstruction &reconstruction,
        const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature,
        unsigned int desired_num_landmarks,
        std::unordered_set<mappoint_t>& summary_store_landmark_ids) {
    std::unordered_set<mappoint_t> all_store_landmark_ids = reconstruction.MapPointIds();

    std::vector<image_t> all_vertex_ids = reconstruction.RegisterImageIds();

    const unsigned int kGlobalTimeLimitSeconds = 30;
    sampleMapSegment(reconstruction, orb_feature, desired_num_landmarks, kGlobalTimeLimitSeconds,
                     all_store_landmark_ids, all_vertex_ids,
                     summary_store_landmark_ids);
}

}  // namespace sensemap
