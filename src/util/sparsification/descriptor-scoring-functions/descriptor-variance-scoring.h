#ifndef SCORING_DESCRIPTOR_VARIANCE_SCORING_H_
#define SCORING_DESCRIPTOR_VARIANCE_SCORING_H_

#include <Eigen/Core>
#include <algorithm>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include "descriptor-scoring-function.h"
#include "map-sparsification_macros.h"

namespace sensemap {
namespace descriptorscoring {

class DescriptorVarianceScoring : public ORBDescriptorScoringFunction {
public:
    SENSEMAP_POINTER_TYPEDEFS(DescriptorVarianceScoring);

    explicit DescriptorVarianceScoring(double descriptor_dev_scoring_threshold)
        : descriptor_dev_scoring_threshold_(descriptor_dev_scoring_threshold) {}

    virtual ~DescriptorVarianceScoring() {}

private:
    inline bool GetMappointDescriptors(
        const mappoint_t store_landmark_id, const Reconstruction &map,
        const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature,
        std::vector<cv::Mat> &vDescriptors) const {
        vDescriptors.clear();
        auto cur_mappoint = map.MapPoint(store_landmark_id);
        auto cur_tracks = cur_mappoint.Track().Elements();
        for (auto cur_track : cur_tracks) {
            auto desc = orb_feature.at(cur_track.image_id).second.row(cur_track.point2D_idx);
            vDescriptors.push_back(desc);
        }

        // Check vDescriptor size
        return !vDescriptors.empty();
    }

    virtual inline double scoreImpl(
        const mappoint_t store_landmark_id, const Reconstruction &map,
        const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &orb_feature) const {
        CHECK(map.ExistsMapPoint(store_landmark_id));
        std::vector<cv::Mat> vDescriptors;
        GetMappointDescriptors(store_landmark_id, map, orb_feature, vDescriptors);

        DescriptorsType descriptors;
        descriptors.resize(vDescriptors[0].cols, vDescriptors.size());
        for (int i = 0; i < vDescriptors.size(); i++) {
            cv::Mat vec_descriptors = vDescriptors[i].t();
            Eigen::Matrix<uchar, Eigen::Dynamic, 1> eigen_mat;
            cv2eigen(vec_descriptors, eigen_mat);
            descriptors.col(i) = eigen_mat;
        }

        const double descriptor_std_dev = DescriptorUtils::descriptorMeanStandardDeviation(descriptors);
        return std::max(descriptor_dev_scoring_threshold_ - descriptor_std_dev, 0.0);
    }

    double descriptor_dev_scoring_threshold_;
};

}  // namespace descriptorscoring
}  // namespace sensemap
#endif  // MAP_SPARSIFICATION_HEURISTIC_SCORING_DESCRIPTOR_VARIANCE_SCORING_H_
