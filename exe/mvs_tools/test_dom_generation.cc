//Copyright (c) 2021, SenseTime Group.
//All rights reserved.

#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>

#include "dom/tdom.h"

#include "base/version.h"
#include "util/exception_handler.h"
#include "../Configurator_yaml.h"

std::string configuration_file_path;

using namespace sensemap;
using namespace sensemap::tdom;

int main(int argc, char *argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
    int max_image_size = param.GetArgument("max_image_size", -1);
    // int max_resolution_per_block = param.GetArgument("max_resolution_per_block", 25000);
    bool color_harmonization = param.GetArgument("color_harmonization", 0);
	std::string image_type = param.GetArgument("image_type", "perspective");
    DOMOptimizer optimizer = (DOMOptimizer)param.GetArgument("dom_optimizer", 0);
    float input_gsd = param.GetArgument("gsd", 10.0f) / 100; // cm
    float max_oblique_angle = param.GetArgument("max_oblique_angle", 45.0f);

    TDOMOptions options;
    // options.max_resolution_per_block = max_resolution_per_block;
    options.color_harmonization = color_harmonization;
    options.gsd = input_gsd;
    options.optimizer = optimizer;
    options.max_oblique_angle = max_oblique_angle;
    TDOM tdom(options, workspace_path);
    tdom.Run();

    return StateCode::SUCCESS;
}