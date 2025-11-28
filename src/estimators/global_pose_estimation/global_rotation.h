//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_GLOBAL_ESTIMATION_GLOBAL_ROTATION_H_
#define SENSEMAP_ESTIMATORS_GLOBAL_ESTIMATION_GLOBAL_ROTATION_H_

#include <unordered_map>

#include "graph/correspondence_graph.h"
#include "util/types.h"

namespace sensemap {

// A generic class defining the interface for global rotation estimation
// methods. These methods take in as input the relative pairwise orientations
// and output estimates for the global orientation of each view.
class GlobalRotationEstimator {
public:
	GlobalRotationEstimator() {}
	virtual ~GlobalRotationEstimator() {}
	// Input the view pairs containing relative rotations between matched
	// geometrically verified views and outputs a rotation estimate for each view.
	//
	// Returns true if the rotation estimation was a success, false if there was a
	// failure. If false is returned, the contents of rotations are undefined.
	virtual bool EstimateRotations(
			const EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair)& view_pairs,
			std::unordered_map<camera_t , Eigen::Vector3d>* rotations) = 0;

};

} // namespace sensemap

#endif //SENSEMAP_ESTIMATORS_GLOBAL_ESTIMATION_GLOBAL_ROTATION_H_
