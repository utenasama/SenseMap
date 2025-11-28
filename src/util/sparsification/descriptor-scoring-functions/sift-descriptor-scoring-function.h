#ifndef SIFT_DESCRIPTOR_SCORING_FUNCTION_H_
#define SIFT_DESCRIPTOR_SCORING_FUNCTION_H_

#include <opencv2/core/types.hpp>
#include "base/reconstruction.h"
#include "util/types.h"

namespace sensemap {
namespace descriptorscoring {

class SIFTDescriptorScoringFunction {
public:
    SENSEMAP_POINTER_TYPEDEFS(SIFTDescriptorScoringFunction);

    SIFTDescriptorScoringFunction() : weight_(1.0) {}

    virtual ~SIFTDescriptorScoringFunction() {}

    double operator()(
        const mappoint_t store_landmark_id, const Reconstruction &map,
        const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature) const {
        //                double raw_score DescriptorScoringFunction= scoreImpl(store_landmark_id, map, orb_feature);
        //                return weight_ * raw_score;
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
