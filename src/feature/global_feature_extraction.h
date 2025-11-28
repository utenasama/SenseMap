// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_GLOBAL_FEATURE_EXTRACTION_H_
#define SENSEMAP_GLOBAL_FEATURE_EXTRACTION_H_

#include <string>

#include "container/feature_data_container.h"

namespace sensemap {

struct GlobalFeatureExtractionOptions {
    // vlad code book path
    std::string vlad_code_book_path = "";
    int local_feature_dim = 128;
};

class GlobalFeatureExtractor {
public:
    GlobalFeatureExtractor(const GlobalFeatureExtractionOptions& options, FeatureDataContainer* container,
                           bool extract_for_new_image = false);

    void Run();

private:
    GlobalFeatureExtractionOptions options_;
    FeatureDataContainer* feature_data_container_;
    bool extract_for_new_image_;
};

}  // namespace sensemap
#endif
