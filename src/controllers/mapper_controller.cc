//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "mapper_controller.h"

#include <glog/logging.h>

#include "util/misc.h"
#include "controllers/incremental_mapper_controller.h"
#include "controllers/global_mapper_controller.h"
#include "controllers/hybrid_mapper_controller.h"
#include "controllers/cluster_mapper_controller.h"
#include "controllers/directed_mapper_controller.h"

namespace sensemap {

MapperController *MapperController::Create(
		const std::shared_ptr<MapperOptions> options,
		const std::string& workspace_path,
		const std::string& image_path,
		//const std::shared_ptr<FeatureDataContainer> feature_data_container,
		const std::shared_ptr<SceneGraphContainer> scene_graph_container,
		std::shared_ptr<ReconstructionManager> reconstruction_manager) {
	switch (options->mapper_type) {
		case MapperType::INDEPENDENT: {
			auto independent_options = std::make_shared<IndependentMapperOptions>(
					options->independent_mapper_options);
			switch (independent_options->independent_mapper_type) {
				case IndependentMapperType::INCREMENTAL:
					return new IncrementalMapperController(independent_options,
					                                       image_path,
														   workspace_path,
					                                       scene_graph_container,
					                                       reconstruction_manager);
					break;
				case IndependentMapperType::GLOBAL:
					return new GlobalMapperController(independent_options,
					                                  image_path,
													  workspace_path,
					                                  scene_graph_container,
					                                  reconstruction_manager);
					break;
				case IndependentMapperType::HYBRID:
					return new HybridMapperController(independent_options,
					                                  image_path,
													  workspace_path,
					                                  scene_graph_container,
					                                  reconstruction_manager);
					break;
                case IndependentMapperType::DIRECTED:
                    return new DirectedMapperController(independent_options,
                                                      image_path,
                                                      workspace_path,
                                                      scene_graph_container,
                                                      reconstruction_manager);
                    break;
				default:
					LOG(FATAL) << "Invalid reconstruction estimator specified.";
			}
			break;
		}
		case MapperType::CLUSTER:
			return new ClusterMapperController(
										std::make_shared<ClusterMapperOptions>
											(options->cluster_mapper_options),
										workspace_path,
										image_path,
										//feature_data_container,
										scene_graph_container,
										reconstruction_manager);
			break;
		default:
			LOG(FATAL) << "Invalid mapper specified.";
	}
	return nullptr;
}


IndependentMapperController::IndependentMapperController(
	const std::shared_ptr<IndependentMapperOptions> options,
	const std::string& image_path,
	const std::string& workspace_path,
	const std::shared_ptr<SceneGraphContainer> scene_graph_container,
	std::shared_ptr<class ReconstructionManager> reconstruction_manager)
	: options_(options),
	  image_path_(image_path),
	  workspace_path_(workspace_path),
	  scene_graph_container_(scene_graph_container),
		reconstruction_manager_(reconstruction_manager){
	PrintHeading1("Independent IncrementalMapperOptions");

}

} // namespace sensemap
