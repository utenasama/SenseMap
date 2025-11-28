//Copyright (c) 2019, SenseTime Group.
//All rights reserved.


#ifndef SENSEMAP_HYBRID_MAPPER_CONTROLLER_H
#define SENSEMAP_HYBRID_MAPPER_CONTROLLER_H

#include "controllers/mapper_controller.h"

namespace sensemap {

class HybridMapperController : public IndependentMapperController{
public:
	HybridMapperController(
			const std::shared_ptr<IndependentMapperOptions> options,
			const std::string& image_path,
			const std::string& workspace_path,
			const std::shared_ptr<SceneGraphContainer> scene_graph_container,
			std::shared_ptr<ReconstructionManager> reconstruction_manager);

private:
	void Run() override;
	void Reconstruct() override;
};

} // namespace sensemap



#endif // SENSEMAP_HYBRID_MAPPER_CONTROLLER_H