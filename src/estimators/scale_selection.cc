// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "scale_selection.h"

namespace sensemap {

std::vector<typename ScaleSelectionEstimator::M_t> ScaleSelectionEstimator::Estimate(const std::vector<X_t>& scales, const std::vector<Y_t>& scales_tmp) {
    double scale = scales[0];
    return {scale};
}

void ScaleSelectionEstimator::Residuals(const std::vector<X_t>& scales, const std::vector<Y_t>& scales_tmp, const M_t& scale,
                                        std::vector<double>* residuals) {

    for (size_t i = 0; i < scales.size(); ++i) {
        double error = std::abs(scales[i]-scale);
        (*residuals)[i] = error;
    }
}

}  // namespace sensemap
