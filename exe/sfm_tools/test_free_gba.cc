// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/pose.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/mat.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include "controllers/cluster_mapper_controller.h"

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"

#include <dirent.h>
#include <sys/stat.h>

#include "util/gps_reader.h"
#include <unordered_set>


using namespace sensemap;



bool FreeGBA(std::shared_ptr<Reconstruction> reconstruction_, const BundleAdjustmentOptions& ba_options){


    const std::vector<image_t>& reg_image_ids = reconstruction_->RegisterImageIds();

    CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                         "registered for global "
                                         "bundle-adjustment";


    // Avoid degeneracies in bundle adjustment.
    reconstruction_->FilterObservationsWithNegativeDepth();

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reg_image_ids) {
        ba_config.AddImage(image_id);
        const Image& image = reconstruction_->Image(image_id);
        const Camera& camera = reconstruction_->Camera(image.CameraId());
    }
    CHECK(reconstruction_->RegisterImageIds().size() > 0);
    const image_t first_image_id = reconstruction_->RegisterImageIds()[0];
    CHECK(reconstruction_->ExistsImage(first_image_id));
    const Image& image = reconstruction_->Image(first_image_id);
    const Camera& camera = reconstruction_->Camera(image.CameraId());

    if (!ba_options.use_prior_absolute_location || !reconstruction_->b_aligned) {
        ba_config.SetConstantPose(reg_image_ids[0]);
        if (camera.NumLocalCameras() == 1) {
            ba_config.SetConstantTvec(reg_image_ids[1], {0});
        } 
        else {
            // ba_config.SetConstantPose(reg_image_ids[1]);
            ba_config.SetConstantTvec(reg_image_ids[1], {0});
        }
    }
    
    // Run bundle adjustment.
    BundleAdjuster bundle_adjuster(ba_options, ba_config);

    if (!bundle_adjuster.Solve(reconstruction_.get())) {
        return false;
    }
   
    return true;
}


int main(int argc, char* argv[]) {

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";
    
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    std::string input_rec_path = workspace_path + "/0-export";
    std::string output_rec_path = workspace_path + "/0-export-free-gba";

    std::cout << "model path : " << input_rec_path << std::endl;
    reconstruction->ReadReconstruction(input_rec_path);

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::INCREMENTAL;

    OptionParser option_parser;
    option_parser.GetMapperOptions(options->independent_mapper_options,param);

    BundleAdjustmentOptions custom_options =  options->independent_mapper_options.GlobalBundleAdjustment();
    PrintHeading1("GBA with free camera rig");
    FreeGBA(reconstruction,custom_options);

    
    if (!boost::filesystem::exists(output_rec_path)) {
        boost::filesystem::create_directories(output_rec_path);
    }
    reconstruction->WriteReconstruction(output_rec_path,options->independent_mapper_options.write_binary_model);
}
