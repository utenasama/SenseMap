#ifndef HEURISTIC_HEURISTIC_SAMPLING_H_
#define HEURISTIC_HEURISTIC_SAMPLING_H_

#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "cost-functions/sampling-cost.h"
#include "descriptor-scoring-functions/descriptor-scoring-function.h"
#include "sampler-base.h"
#include "scoring/scoring-function.h"

namespace sensemap {
namespace sampling {

class LandmarkSamplingWithCostFunctions : public SamplerBase {
public:
    SENSEMAP_POINTER_TYPEDEFS(LandmarkSamplingWithCostFunctions);

    typedef cost_functions::SamplingCostFunction SamplingCostFunction;
    typedef scoring::ScoringFunction ScoringFunction;
    typedef SamplingCostFunction::KeypointPerVertexCountMap KeyframeKeypointCountMap;

    typedef std::pair<mappoint_t, double> StoreLandmarkIdScorePair;
    typedef std::unordered_map<mappoint_t, double> LandmarkScoreMap;

    void registerScoringFunction(const ScoringFunction::ConstPtr &scoring);

    void registerDescriptorScoringFunction(
        const descriptorscoring::ORBDescriptorScoringFunction::ConstPtr &despscoring);

    void registerCostFunction(const SamplingCostFunction::ConstPtr &cost);

    virtual void sampleMapSegment(
        const Reconstruction &reconstruction,
        const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature,
        unsigned int desired_num_landmarks, unsigned int time_limit_seconds,
        const std::unordered_set<mappoint_t> &segment_store_landmark_id_set,
        const std::vector<image_t> &segment_vertex_id_list, std::unordered_set<mappoint_t> &summary_store_landmark_ids);

    virtual std::string getTypeString() const { return "greedy"; }

private:
    std::vector<ScoringFunction::ConstPtr> scoring_functions_;
    std::vector<descriptorscoring::ORBDescriptorScoringFunction::ConstPtr> descriptor_scoring_functions_;
    std::vector<SamplingCostFunction::ConstPtr> cost_functions_;
};

}  // namespace sampling
}  // namespace sensemap
#endif  // HEURISTIC_HEURISTIC_SAMPLING_H_
