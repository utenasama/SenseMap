#ifndef MAP_SPARSIFICATION_HEURISTIC_COST_FUNCTIONS_SAMPLING_COST_H_
#define MAP_SPARSIFICATION_HEURISTIC_COST_FUNCTIONS_SAMPLING_COST_H_

#include <unordered_map>
#include "util/types.h"
#include "base/reconstruction.h"

namespace sensemap {
namespace cost_functions {

class SamplingCostFunction {
public:
    SENSEMAP_POINTER_TYPEDEFS(SamplingCostFunction);

    typedef std::unordered_map<image_t, unsigned int> KeypointPerVertexCountMap;

    SamplingCostFunction() : weight_(1.0) {
        loss_function_ = [](double x) { return x; };  // NOLINT
    }

    virtual ~SamplingCostFunction() {}

    double operator()(const mappoint_t &landmark_id, const Reconstruction &map,
                      const KeypointPerVertexCountMap &keyframe_keypoint_counts) const {
        double raw_score = scoreImpl(landmark_id, map, keyframe_keypoint_counts);
        return weight_ * loss_function_(raw_score);
    }

    void setWeight(double weight) { weight_ = weight; }

    void setLossFunction(const std::function<double(double)> &loss_function) {  // NOLINT
        loss_function_ = loss_function;
    }

private:
    virtual double scoreImpl(const mappoint_t store_landmark_id, const Reconstruction &map,
                             const KeypointPerVertexCountMap &keyframe_keypoint_counts) const = 0;

    double weight_;
    std::function<double(double)> loss_function_;  // NOLINT
};

}  // namespace cost_functions
}  // namespace sensemap
#endif  // MAP_SPARSIFICATION_HEURISTIC_COST_FUNCTIONS_SAMPLING_COST_H_
