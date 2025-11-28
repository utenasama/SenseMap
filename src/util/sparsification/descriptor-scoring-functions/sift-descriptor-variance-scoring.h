#ifndef MAP_SPARSIFICATION_HEURISTIC_SCORING_DESCRIPTOR_VARIANCE_SCORING_H_
#define MAP_SPARSIFICATION_HEURISTIC_SCORING_DESCRIPTOR_VARIANCE_SCORING_H_

#include <algorithm>
#include "sift-descriptor-scoring-function.h"

namespace sensemap {
namespace descriptorscoring {

class SIFTDescriptorVarianceScoring : public SIFTDescriptorScoringFunction {
public:
    SENSEMAP_POINTER_TYPEDEFS(SIFTDescriptorVarianceScoring);

    explicit SIFTDescriptorVarianceScoring(double descriptor_dev_scoring_threshold)
        : descriptor_dev_scoring_threshold_(descriptor_dev_scoring_threshold) {}

    virtual ~SIFTDescriptorVarianceScoring() {}

private:
    virtual inline double scoreImpl(
        const mappoint_t store_landmark_id, const Reconstruction &map,
        const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature) const {
        //                CHECK(map.hasLandmark(landmark_id));
        //                vi_map::VIMap::DescriptorsType descriptors;
        //                map.getLandmarkDescriptors(landmark_id, &descriptors);
        //                const double descriptor_std_dev =
        //                        aslam::common::descriptor_utils::descriptorMeanStandardDeviation(
        //                                descriptors);
        //                return std::max(
        //                        descriptor_dev_scoring_threshold_ - descriptor_std_dev, 0.0);
    }

    double descriptor_dev_scoring_threshold_;
};

}  // namespace descriptorscoring
}  // namespace sensemap
#endif  // MAP_SPARSIFICATION_HEURISTIC_SCORING_DESCRIPTOR_VARIANCE_SCORING_H_
