//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CLUSTER_COMMUNITY_TREE_H_
#define SENSEMAP_CLUSTER_COMMUNITY_TREE_H_

#include <vector>
#include <memory>
#include <list>
#include <glog/logging.h>

namespace fastcommunity {

template <typename T>
class CommunityTree  {
public:

	CommunityTree(): id_(-1), leaf_(false), merged_(false) {};

	explicit CommunityTree(const std::vector<T> elements,
	        CommunityTree<T>* parent)
			: id_(-1), leaf_(false), merged_(false),
			elements_(elements),parent_tree_(parent) {};

	~CommunityTree(){
		std::vector<CommunityTree<T> *> non_leaves;
		non_leaves.push_back(this);
		while (!non_leaves.empty()) {
			auto & tree = non_leaves.back();
			non_leaves.pop_back();
			for (const auto& child_tree : tree->child_trees()) {
				non_leaves.push_back(child_tree);
			}
			delete[] tree;
		}
	}

	int Id(){
        return id_;
	};

	void setId(const int id){
	    id_ = id;
	};

	void setIds(){
	    int idx = 0;
        std::vector<CommunityTree<T> *> non_leaves;
        non_leaves.push_back(this);
        while (!non_leaves.empty()) {
            auto & tree = non_leaves.back();
            non_leaves.pop_back();
            tree->setId(idx++);
            for (const auto& child_tree : tree->child_trees()) {
                non_leaves.push_back(child_tree);
            }
        }
	};

	bool isLeaf(){
        return leaf_;
	};

	void setLeaf(const bool leaf){
	    leaf_ = leaf;
	};

	bool isMerged(){
        return merged_;
	}

	void setMerged(const bool merged){
        merged_ = merged;
	};

	bool addChildTree(CommunityTree<T>* child_tree) {
		//CHECK ...
		//return false;
		child_trees_.push_back(child_tree);
		return true;
	};

	void AddElements(std::vector<T> elements) {
		elements_.insert(elements_.end(), elements.begin(), elements.end());
	};

	void SetElements(std::vector<T> elements) {
		elements_.clear();
		elements_ = std::move(elements);
	};

	const std::vector<T>& elements() const{
		return elements_;
	};

	CommunityTree*  parenet_tree() const{
		return parent_tree_;
	};

	const std::list<CommunityTree<T> *> & child_trees() const{
		return child_trees_;
	};

	std::list<CommunityTree<T>* > GetLeaves() {

		std::list<CommunityTree<T>* > leaves;
		if(child_trees_.empty()) {
		    leaf_ = true;
			leaves.push_back(this);
			return leaves;
		}
		std::vector<CommunityTree<T> *> non_leaves;
		non_leaves.push_back(this);
		while (!non_leaves.empty()) {
			auto & tree = non_leaves.back();
			non_leaves.pop_back();
			for (const auto& child_tree : tree->child_trees()) {
				if (child_tree->child_trees().empty()) {
				    child_tree->setLeaf(true);
					leaves.push_back(child_tree);
				} else {
					non_leaves.push_back(child_tree);
				}
			}
		}
		return leaves;
	};

	void writeDot(const std::string &path) {

	    setIds();

		std::ofstream file(path, std::ios::app);

		int index = 0;
		if (child_trees_.empty()) {
			file << std::endl;
			file << "graph cluster_" << id_<< " { " << std::endl;
			file << "   label = \"community "<< id_  << "\";" <<std::endl;
            file << elements_.size() << std::endl;
			file << "}" <<std::endl;
			file.close();
			return;
		}

		std::vector<CommunityTree<T> *> non_leaves;
		non_leaves.push_back(this);
		while (!non_leaves.empty()) {
			auto tree = non_leaves.back();
			non_leaves.pop_back();

			if(tree->elements().empty()){ // if is guard
				file << std::endl;
				file << "}" <<std::endl;
				continue;
			}

            non_leaves.push_back(new CommunityTree<T>()); // add guard

			file << std::endl;
			file << "subgraph cluster_" << tree->Id() << " { " << std::endl;
			file << "   label = \"community "<< tree->Id()  << "\";" <<std::endl;
            if(tree->child_trees().empty()){
                file << tree->elements().size() << std::endl;
            } else {
                for (const auto &child_tree : tree->child_trees()) {
                    non_leaves.push_back(child_tree);
                }
            }
		}
		file.close();
	};

private:
    int id_;
    bool leaf_;
    bool merged_;
	std::vector<T> elements_;
	CommunityTree<T>* parent_tree_;
	std::list<CommunityTree<T>*> child_trees_;
};



} // namespace fastcommunity

#endif //SENSEMAP_CLUSTER_COMMUNITY_TREE_H_
