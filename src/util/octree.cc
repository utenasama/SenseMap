
#include <numeric>
#include <iostream>
#include <iterator>
#include <cmath>

#include "octree.h"

namespace sensemap {

Octree::Octree(int level, double voxel_size)
	: max_level_(level),
	  min_voxel_size_(voxel_size) {
	root_ = std::shared_ptr<Node>(new Node);
}

void Octree::Insert(const Octree::NodeInfo& info) {
	InsertImpl(0, root_, info);
}

void Octree::InsertImpl(int level, std::shared_ptr<Node>& node, const NodeInfo& info) {
	BoundingBox bbox = node->bb;
	if (bbox.GetMinLength() < min_voxel_size_ || level > max_level_) {
		if (!node->vert_list) {
			node->vert_list = std::make_shared<std::vector<NodeInfo> >();
		}
		node->vert_list->emplace_back(info);
		// std::cout<< node->vert_list->size() << std::endl;
		return;
	}
	for (int i = 0; i < 8; ++i) {
		auto bb = bbox.Get(i);
		if (bb.Contains(info.X)) {
			if (!node->children[i]) {
				node->children[i] = std::shared_ptr<Node>(new Node);
				node->children[i]->bb = bb;
			}
			InsertImpl(level + 1, node->children[i], info);
			break;
		}
	}
}

void Octree::Traverse() {
	if (!root_) {
		return;
	}
	for (int i = 0; i < 8; ++i) {
		TraverseChildren(root_->children[i]);
	}
}

void Octree::TraverseChildren(std::shared_ptr<Node> node) {
	if (!node) {
		return;
	}
	bool is_leaf = true;
	for (int i = 0; i < 8; ++i) {
		if (node->children[i]) {
			TraverseChildren(node->children[i]);
			is_leaf = false;
		}
	}
	if (is_leaf && !node->vert_list->empty()) {
		leaf_nodes_.push_back(*node.get());
	}
}

} // namespace sensemap