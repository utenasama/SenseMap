// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "util/misc.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

std::string configuration_file_path;
std::string workspace_path;

int main(int argc, char* argv[]) {
    if (argc == 2) {
        configuration_file_path = argv[1];
    }
    else {
        std::cout << "Usage: " << argv[0] << " <SFM_YAML>" << std::endl;
        return 1;
    }

    Configurator param;
    param.Load(configuration_file_path.c_str());

    workspace_path = param.GetArgument("workspace_path", "");

    for (int i = 0; ;i++) {
        std::stringstream path;
        path << workspace_path << "/" << i << "/";
        if (!boost::filesystem::exists(path.str())) break;

        auto reconstruction = std::make_shared<Reconstruction>();
        reconstruction->ReadBinary(path.str());
        std::cout << "Read reconstruction finished ..." << std::endl;

        reconstruction->WritePoseText(path.str() + "pose.txt");
    }

    return 0;
}
