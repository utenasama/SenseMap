//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CONTROLLER_MAPPER_CONTROLLER_H
#define SENSEMAP_CONTROLLER_MAPPER_CONTROLLER_H

#include "controllers/mapper_options.h"
#include "util/threading.h"

namespace sensemap {

class MapperController : public Thread {
public:
	virtual ~MapperController() = default;

	static MapperController* Create(
			const std::shared_ptr<MapperOptions> options,
			const std::string& workspace_path,
			const std::string& image_path,
			//const std::shared_ptr<FeatureDataContainer> feature_data_container,
			const std::shared_ptr<SceneGraphContainer> scene_graph_container,
			std::shared_ptr<ReconstructionManager> reconstruction_manager);
};

class IndependentMapperController : public MapperController{
public:
	IndependentMapperController(
			const std::shared_ptr<IndependentMapperOptions> options,
			const std::string& image_path,
			const std::string& workspace_path,
			const std::shared_ptr<SceneGraphContainer> scene_graph_container,
			std::shared_ptr<ReconstructionManager> reconstruction_manager);
	virtual ~IndependentMapperController() = default;

	virtual void Reconstruct() = 0;

protected:
	const std::shared_ptr<IndependentMapperOptions> options_;
	const std::string image_path_;
	const std::string workspace_path_;
	std::shared_ptr<class ReconstructionManager> reconstruction_manager_;
	std::shared_ptr<SceneGraphContainer> scene_graph_container_;
};

} // namespace sensemap

#endif //SENSEMAP_CONTROLLER_MAPPER_CONTROLLER_H
