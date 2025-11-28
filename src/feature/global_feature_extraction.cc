// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#include "feature/global_feature_extraction.h"
#include "feature/utils.h"
#include "retrieval/vlad_visual_index.h"
#include "util/timer.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace sensemap {

GlobalFeatureExtractor::GlobalFeatureExtractor(const GlobalFeatureExtractionOptions& options,
                                               FeatureDataContainer* container, bool extract_for_new_image)
    : feature_data_container_(container), options_(options), extract_for_new_image_(extract_for_new_image) {}

void GlobalFeatureExtractor::Run() {
    VladVisualIndex vlad_visual_index;
    vlad_visual_index.LoadCodeBook(options_.vlad_code_book_path);

    std::vector<image_t> image_ids =
        extract_for_new_image_ ? feature_data_container_->GetNewImageIds() : feature_data_container_->GetImageIds();

    Timer timer;
    timer.Start();
    std::cout << "Indexing image:" << std::endl;

    // Index all the image
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < image_ids.size(); ++i) {
        std::cout << "Extract global features for the " << i << "th image" << std::endl;
        image_t current_id = image_ids[i];
        const auto& descriptors = feature_data_container_->GetCompressedDescriptors(current_id);
        VladVisualIndex::Descriptors float_descriptors;
        CompressedFeatureDescriptorsTofloat(descriptors, float_descriptors);

        VladVisualIndex::VLAD vlad_vector;
        vlad_visual_index.LocalFeaturesToVlad(float_descriptors, vlad_vector);
        feature_data_container_->GetVladVector(current_id) = vlad_vector;
    }

    std::cout << StringPrintf("Indexing in %.3f min", timer.ElapsedMinutes()) << std::endl;
}

}  // namespace sensemap