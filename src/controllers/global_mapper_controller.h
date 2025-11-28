//Copyright (c) 2019, SenseTime Group.
//All rights reserved.


#ifndef SENSEMAP_GLOBAL_MAPPER_CONTROLLER_H
#define SENSEMAP_GLOBAL_MAPPER_CONTROLLER_H


#include "controllers/mapper_controller.h"
#include "sfm/global_mapper.h"

namespace sensemap {

class GlobalMapperController : public IndependentMapperController{
public:
	 GlobalMapperController(
			const std::shared_ptr<IndependentMapperOptions> options,
			const std::string& image_path,
			const std::string& workspace_path,
			const std::shared_ptr<SceneGraphContainer> scene_graph_container,
			std::shared_ptr<ReconstructionManager> reconstruction_manager);

private:
	void Run() override;
	void Reconstruct() override;


    GlobalMapper::Options init_mapper_options_;
};

} // namespace sensemap



#endif //SENSEMAP_GLOBAL_MAPPER_CONTROLLER_H
