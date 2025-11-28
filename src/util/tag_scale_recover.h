// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_UTIL_TAGSCALERECOVER_H_
#define SENSEMAP_UTIL_TAGSCALERECOVER_H_

#include "base/reconstruction.h"
#include "container/feature_data_container.h"
#include "container/scene_graph_container.h"


#include "util/misc.h"

namespace sensemap {

class TagScaleRecover {
public:
    struct TagScaleRecoverOptions {
        std::string workspace_path = "";

        double tag_size = 0.113;
    };

public:
    TagScaleRecover(TagScaleRecoverOptions options);

    double ComputeScale(std::shared_ptr<FeatureDataContainer> feature_data_container,
                        std::shared_ptr<SceneGraphContainer> scene_graph_container,
                        std::shared_ptr<Reconstruction> reconstruction);

    std::vector<std::pair<std::string, Eigen::Vector3d>> alignment_points_;
    std::vector<Eigen::Vector3d> alignment_tag_points_;

private:
    TagScaleRecoverOptions options_;

    double scale_result_;
    
};

}  // namespace sensemap
#endif  // SENSEMAP_UTIL_TAGSCALERECOVER_H_