//
// Created by sensetime on 2021/2/4.
//

#ifndef SENSEMAP_GLOBAL_POSITION_H_
#define SENSEMAP_GLOBAL_POSITION_H_

#include <unordered_map>

#include "graph/correspondence_graph.h"
#include "util/types.h"

namespace sensemap {

// A generic class defining the interface for global rotation estimation
// methods. These methods take in as input the relative pairwise orientations
// and output estimates for the global orientation of each view.
    class GlobalPositionEstimator {
    public:
        GlobalPositionEstimator() {}
        virtual ~GlobalPositionEstimator() {}
        // Input the view pairs containing relative rotations between matched
        // geometrically verified views and outputs a rotation estimate for each view.
        //
        // Returns true if the rotation estimation was a success, false if there was a
        // failure. If false is returned, the contents of rotations are undefined.
        virtual bool EstimatePositions(
                const EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) &image_pairs,
                const std::unordered_map<image_t , Eigen::Vector3d>& orientation,
                std::unordered_map<image_t , Eigen::Vector3d>* positions) = 0;

    };

} // namespace sensemap


#endif //SENSEMAP_GLOBAL_POSITION_H_
