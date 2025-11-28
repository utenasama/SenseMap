//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CONTROLLERS_INCREMENTAL_MAPPER_CONTROLLER_H_
#define SENSEMAP_CONTROLLERS_INCREMENTAL_MAPPER_CONTROLLER_H_

#include <set>

#include "util/types.h"
#include "util/threading.h"
#include "sfm/incremental_mapper.h"
#include "base/image.h"
#include "base/reconstruction_manager.h"
#include "container/scene_graph_container.h"
#include "controllers/mapper_options.h"
#include "controllers/mapper_controller.h"
namespace sensemap {
// Class that controls the incremental mapping procedure by iteratively
// initializing reconstructions from the same scene graph.
class IncrementalMapperController : public IndependentMapperController{
public:
	enum {
		INITIAL_IMAGE_PAIR_REG_CALLBACK,
		NEXT_IMAGE_REG_CALLBACK,
		LAST_IMAGE_REG_CALLBACK,
	};
	IncrementalMapperController(
			const std::shared_ptr<IndependentMapperOptions> options,
			const std::string& image_path,
			const std::string& workspace_path,
			const std::shared_ptr<SceneGraphContainer> scene_graph_container,
			std::shared_ptr<class ReconstructionManager> reconstruction_manager);

private:
	void Run() override;
	void Reconstruct() override;

	IncrementalMapper::Options init_mapper_options_;
};

size_t FilterPoints(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper);

size_t FilterPoints(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper,
                    int min_track_length);

size_t FilterPointsFinal(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper);

size_t FilterImages(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper);

size_t CompleteAndMergeTracks(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper);

size_t TriangulateImage(const IndependentMapperOptions& options, const Image& image,
                        std::shared_ptr<IncrementalMapper> mapper);

void ExtractColors(const std::string& image_path, const image_t image_id,
                   std::shared_ptr<Reconstruction> reconstruction);

}

#endif
