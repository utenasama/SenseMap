//
// Created by sensetime on 2021/4/14.
//

#ifndef SENSEMAP_CONTROLLERS_DIRECTED_MAPPER_CONTROLLER_H_
#define SENSEMAP_CONTROLLERS_DIRECTED_MAPPER_CONTROLLER_H_

#include "util/types.h"
#include "util/threading.h"
#include "sfm/incremental_mapper.h"
#include "base/reconstruction_manager.h"
#include "container/scene_graph_container.h"
#include "controllers/mapper_options.h"
#include "controllers/mapper_controller.h"

namespace sensemap {

class DirectedMapperController : public IndependentMapperController {
public:

    DirectedMapperController(
            const std::shared_ptr<IndependentMapperOptions> options,
            const std::string &image_path,
            const std::string &workspace_path,
            const std::shared_ptr<SceneGraphContainer> scene_graph_container,
            std::shared_ptr<class ReconstructionManager> reconstruction_manager);

    bool IsSuccess(){
        return success_;
    };

private:
    void Run() override;

    void Reconstruct() override;

    IncrementalMapper::Options mapper_options_;

    bool success_;
};

}
#endif //SENSEMAP_CONTROLLERS_DIRECTED_MAPPER_CONTROLLER_H_
