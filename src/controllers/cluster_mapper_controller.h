//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CLUSTER_MAPPER_CONTROLLER_H
#define SENSEMAP_CLUSTER_MAPPER_CONTROLLER_H

#include <malloc.h>

#include "controllers/mapper_options.h"
#include "controllers/mapper_controller.h"
#include "graph/scene_clustering.h"
#include "base/pose.h"
#include "util/proc.h"

namespace sensemap {

class ClusterMapperController : public MapperController {
public:

	ClusterMapperController(
			const std::shared_ptr<ClusterMapperOptions> options,
			const std::string& workspace_path,
			const std::string& image_path,
			//const std::shared_ptr<FeatureDataContainer> feature_data_container,
			const std::shared_ptr<SceneGraphContainer> scene_graph_container,
			std::shared_ptr<ReconstructionManager> reconstruction_manager);

private:
	void Run() override;
	
	// void ReconstructClusters();

	void ReconstructCommunities(
		const std::vector<std::vector<image_t> > &communities,
		const std::vector<std::unordered_set<image_t> >& overlap_image_ids,
		const std::vector<label_t>& communitie_ids);

	// // FIXME: TMP function for global bundle
	// void FinalBundleAdjustment(Reconstruction* reconstruction);	

	void ClusterFinalBundleAdjust(
		const sensemap::SceneGraphContainer &scene_graph_container,
		const std::unordered_map<image_t, std::set<image_t>> image_neighbor_between_cluster,
		std::shared_ptr<Reconstruction> reconstruction);

protected:
	std::shared_ptr<ClusterMapperOptions> options_;
	const std::string workspace_path_;
	const std::string image_path_;
	const std::shared_ptr<FeatureDataContainer> feature_data_container_;
	const std::shared_ptr<SceneGraphContainer> scene_graph_container_;

	std::shared_ptr<ReconstructionManager> reconstruction_manager_;
	std::shared_ptr<SceneClustering> scene_clustering_;
};

} // namespace sensemap


#endif //SENSEMAP_CLUSTER_MAPPER_CONTROLLER_H
