// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "util/exception_handler.h"
#include "base/reconstruction_manager.h"
#include "util/misc.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

std::string sparse_path;
std::string export_path;

int main(int argc, char* argv[]) {
    if (argc == 2) {
        sparse_path = argv[1];
        export_path = JoinPaths(sparse_path, "calibration.yaml");
    } else if (argc == 3) {
        sparse_path = argv[1];
        export_path = argv[2];
    } else {
        std::cout << "Usage: " << argv[0] << " <SPARSE_PATH> [EXPORT_PATH]" << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->ReadBinary(sparse_path);

    if (!boost::filesystem::exists(boost::filesystem::path(export_path).parent_path())) {
        boost::filesystem::create_directories(boost::filesystem::path(export_path).parent_path());
    }

    std::vector<image_t> image_ids(reconstruction->RegisterImageIds().begin(), reconstruction->RegisterImageIds().end());
    std::sort(image_ids.begin(), image_ids.end(), [&](image_t a, image_t b) {
        return reconstruction->Image(a).Name() < reconstruction->Image(b).Name();
    });

    std::vector<camera_t> camera_ids;
    for (auto & camera : reconstruction->Cameras()) {
        camera_ids.emplace_back(camera.first);
    }
    std::sort(camera_ids.begin(), camera_ids.end());

    YAML::Node node;
    node["num_cameras"] = camera_ids.size();
    for (int i = 0; i < camera_ids.size(); i++) {
        auto & camera = reconstruction->Camera(camera_ids[i]);

        YAML::Node camera_i;
        camera_i["camera_id"] = camera_ids[i];
        camera_i["camera_size"] = std::vector<int>({ (int)camera.Width(), (int)camera.Height() });
        camera_i["camera_model"] = camera.ModelName();
        camera_i["camera_params"] = camera.ParamsToString();

        node["camera_" + std::to_string(i)] = camera_i;
    }
    node["num_images"] = image_ids.size();
    for (int i = 0; i < image_ids.size(); i++) {
        auto & image = reconstruction->Image(image_ids[i]);

        YAML::Node image_i;
        image_i["image_id"] = image_ids[i];
        image_i["camera_id"] = image.CameraId();
        image_i["image_name"] = image.Name();
        image_i["image_qvec"] = std::vector<double>({ image.Qvec()[0], image.Qvec()[1], image.Qvec()[2], image.Qvec()[3] });
        image_i["image_tvec"] = std::vector<double>({ image.Tvec()[0], image.Tvec()[1], image.Tvec()[2] });

        node["image_" + std::to_string(i)] = image_i;
    }

    std::ofstream ofs(export_path);
    ofs << YAML::Dump(node) << std::endl;

    return 0;
}
