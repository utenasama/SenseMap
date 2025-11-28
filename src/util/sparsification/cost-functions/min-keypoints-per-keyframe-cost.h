#ifndef MAP_SPARSIFICATION_HEURISTIC_COST_FUNCTIONS_MIN_KEYPOINTS_PER_KEYFRAME_COST_H_
#define MAP_SPARSIFICATION_HEURISTIC_COST_FUNCTIONS_MIN_KEYPOINTS_PER_KEYFRAME_COST_H_

#include <util/types.h>
#include "sampling-cost.h"

namespace sensemap {
namespace cost_functions {

class IsRequiredToConstrainKeyframesCost : public SamplingCostFunction {
public:
    SENSEMAP_POINTER_TYPEDEFS(IsRequiredToConstrainKeyframesCost);

    explicit IsRequiredToConstrainKeyframesCost(int min_keypoints_per_keyframe)
        : min_keypoints_per_keyframe_(min_keypoints_per_keyframe) {}

    virtual ~IsRequiredToConstrainKeyframesCost() {}

private:
    virtual inline double scoreImpl(mappoint_t store_landmark_id, const Reconstruction &map,
                                    const KeypointPerVertexCountMap &keyframe_keypoint_counts) const {
        double total_keyframe_cost = 0.0;

        // Verify if each of the observer frames has already enough landmarks.
        const auto &landmark = map.MapPoint(store_landmark_id);
        for (const auto &observation : landmark.Track().Elements()) {
            const auto &vertex_id = observation.image_id;
            auto keypoint_count_it = keyframe_keypoint_counts.find(vertex_id);

            // Count only observer frames from the current segment.
            if (keypoint_count_it != keyframe_keypoint_counts.end()) {
                total_keyframe_cost += std::max(0.0, static_cast<double>(min_keypoints_per_keyframe_ -
                                                                         static_cast<int>(keypoint_count_it->second)));
            }
        }
        return total_keyframe_cost;
    }

    int min_keypoints_per_keyframe_;
};

}  // namespace cost_functions
}  // namespace sensemap
#endif  // MAP_SPARSIFICATION_HEURISTIC_COST_FUNCTIONS_MIN_KEYPOINTS_PER_KEYFRAME_COST_H_
