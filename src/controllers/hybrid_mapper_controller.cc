//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "hybrid_mapper_controller.h"

namespace sensemap {


HybridMapperController::HybridMapperController(
		const std::shared_ptr<IndependentMapperOptions> options,
		const std::string &image_path,
		const std::string& workspace_path,
		const std::shared_ptr<SceneGraphContainer> scene_graph_container,
		std::shared_ptr<ReconstructionManager> reconstruction_manager)
		: IndependentMapperController(options, image_path, workspace_path, 
						scene_graph_container, reconstruction_manager) {
	CHECK(options_->HybridMapperCheck());
}

void HybridMapperController::Run() {

}

void HybridMapperController::Reconstruct() {

}
} // namespace sensemap