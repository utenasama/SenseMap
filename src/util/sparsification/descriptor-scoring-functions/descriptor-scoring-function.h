#ifndef ORB_DESCRIPTOR_SCORING_FUNCTION_H_
#define ORB_DESCRIPTOR_SCORING_FUNCTION_H_

#include "base/reconstruction.h"
#include "util/types.h"
#include <opencv2/opencv.hpp>

namespace sensemap {
namespace descriptorscoring {

class ORBDescriptorScoringFunction {
public:
    SENSEMAP_POINTER_TYPEDEFS(ORBDescriptorScoringFunction);

    ORBDescriptorScoringFunction() : weight_(1.0) {}

    virtual ~ORBDescriptorScoringFunction() {}

    double operator()(
        const mappoint_t store_landmark_id, const Reconstruction &map,
        const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature) const {
        double raw_score = scoreImpl(store_landmark_id, map, orb_feature);
        return weight_ * raw_score;
    }

    void setWeight(double weight) { weight_ = weight; }

private:
    virtual double scoreImpl(
        mappoint_t store_landmark_id, const Reconstruction &map,
        const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature) const = 0;

    double weight_;
};

}  // namespace descriptorscoring
}  // namespace sensemap
#endif  // MAP_SPARSIFICATION_HEURISTIC_SCORING_SCORING_FUNCTION_H_
