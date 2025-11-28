#include "FullGCHelper.h"

#include "util/image_pair.h"

namespace sensemap {

GCoptimization::EnergyTermType FullGCOHelper::SmoothFunc(int node_id1, int node_id2, int label1, int label2, void* GCOHelperData) {
    FullGCOHelper* gco_helper_data = static_cast<FullGCOHelper *>(GCOHelperData);
    GCoptimization::EnergyTermType smooth_cost = 1;
    if (label1 == label2) {
        smooth_cost = 0;
    } else if (label1 != 0 && label2 != 0) {
        auto pair_id = utility::ImagePairToPairId(label1 - 1, label2 - 1);
        if (gco_helper_data->image_pair_scores_->find(pair_id) != gco_helper_data->image_pair_scores_->end()) {
            smooth_cost = gco_helper_data->image_pair_scores_->at(pair_id);
            // printf("image_pair: %d %d %d\n", label1 - 1, label2 - 1, smooth_cost);
        } else {
            smooth_cost = 1;
        }
    }
    // const std::vector<uint32_t>& pixel_list1 = gco_helper_data->super_idx_to_pixels_->at(node_id1);
    // const std::vector<uint32_t>& pixel_list2 = gco_helper_data->super_idx_to_pixels_->at(node_id2);
    // method1. 边界上没有明显的梯度变化或梯度误差最小化
    // method2. overlap区域的直方图一致或误差最小化
    return smooth_cost;
}

}