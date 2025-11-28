
//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_OCTREE_H_
#define SENSEMAP_UTIL_OCTREE_H_

#include <vector>
#include <memory>

#include "bb.h"

namespace sensemap {

class Octree {
public:
	struct NodeInfo {
		size_t view_id;
		Eigen::Vector3f X;
	};

	struct Node {
		BoundingBox bb;
		std::shared_ptr<Node> children[8];
		std::shared_ptr<std::vector<NodeInfo> > vert_list;
	};
public:
	Octree(int level = 10, double voxel_size = 0.05);
	~Octree() {}

	inline void SetBB(BoundingBox bb) { root_->bb = bb_ = bb; }

	inline void SetLevel(int level) { max_level_ = level; }

	inline void SetVoxelSize(double voxel_size) { min_voxel_size_ = voxel_size; }

	const std::vector<Node>& GetLeafNodes() const {
		return leaf_nodes_;
	}

	void Insert(const NodeInfo& info);

	void Traverse();

private:

	void InsertImpl(int level, std::shared_ptr<Node>& node, const NodeInfo& info);

	void TraverseChildren(std::shared_ptr<Node> node);

private:
	int max_level_;

	double min_voxel_size_;
	
	std::shared_ptr<Node> root_;
	
	std::vector<Node> leaf_nodes_;

	BoundingBox bb_;
};
} // namespace sensemap

#endif
