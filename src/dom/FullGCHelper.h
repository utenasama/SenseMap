//Copyright (c) 2022, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_DOM_FULLGCHELPER_H_
#define SENSEMAP_DOM_FULLGCHELPER_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/opencv.hpp>

#include "util/mat.h"
#include "util/types.h"
#include "mvs/image.h"
#include "GCO/GCoptimization.h"

namespace sensemap {

class FullGCOHelper {
public:
    FullGCOHelper(const MatXf* zBuffer, 
                const std::vector<mvs::Image> *images, const std::vector<cv::Mat> *bitmaps,
                const std::vector<std::unordered_set<uint32_t> >* neighbors_per_super,
                const std::vector<std::vector<uint32_t> >* super_visibilities,
                const std::unordered_map<uint32_t, std::vector<uint32_t> >* super_idx_to_pixels,
                const std::unordered_map<image_pair_t, float>* image_pair_scores) 
        : zBuffer_(zBuffer),
          images_(images),
          bitmaps_(bitmaps),
          neighbors_per_super_(neighbors_per_super),
          super_visibilities_(super_visibilities),
          super_idx_to_pixels_(super_idx_to_pixels),
          image_pair_scores_(image_pair_scores) {}

    static GCoptimization::EnergyTermType SmoothFunc(int node_id1, int node_id2, int label1, int label2, void* GCOHelperData);

private:
    const MatXf* zBuffer_;
    const std::vector<mvs::Image> *images_;
    const std::vector<cv::Mat> *bitmaps_;
    const std::vector<std::unordered_set<uint32_t> >* neighbors_per_super_;
    const std::vector<std::vector<uint32_t> >* super_visibilities_;
    const std::unordered_map<uint32_t, std::vector<uint32_t> >* super_idx_to_pixels_;
    const std::unordered_map<image_pair_t, float>* image_pair_scores_;
};

}

#endif