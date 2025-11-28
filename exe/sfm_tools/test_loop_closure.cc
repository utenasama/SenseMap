// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>

#include <boost/filesystem/path.hpp>
#include "container/feature_data_container.h"
#include "../Configurator_yaml.h"
#include "../option_parsing.h"

OptionParser option_parser;

using namespace sensemap;


size_t FilterPoints(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions());
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterImages(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_images = mapper->FilterImages(options.IncrementalMapperOptions());
    std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
    return num_filtered_images;
}

size_t CompleteAndMergeTracks(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_completed_observations = mapper->CompleteTracks(options.Triangulation());
    std::cout << "  => Completed observations: " << num_completed_observations << std::endl;
    const size_t num_merged_observations = mapper->MergeTracks(options.Triangulation());
    std::cout << "  => Merged observations: " << num_merged_observations << std::endl;
    return num_completed_observations + num_merged_observations;
}

void AdjustGlobalBundle(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    BundleAdjustmentOptions custom_options = options.GlobalBundleAdjustment();
    // custom_options.refine_extrinsics = false;
    const size_t num_reg_images = mapper->GetReconstruction().NumRegisterImages();

    if (options.single_camera && (num_reg_images > options.num_images_for_self_calibration || options.camera_fixed) ||
        num_reg_images < options.num_fix_camera_first) {
        custom_options.refine_focal_length = false;
        custom_options.refine_extra_params = false;
        custom_options.refine_principal_point = false;
        custom_options.refine_local_extrinsics = false;
    }

    custom_options.plane_constrain = false;
    custom_options.use_prior_absolute_location = false;
   
    // Use stricter convergence criteria for first registered images.
    const size_t kMinNumRegImages = 10;
    if (num_reg_images < kMinNumRegImages) {
        custom_options.solver_options.function_tolerance /= 10;
        custom_options.solver_options.gradient_tolerance /= 10;
        custom_options.solver_options.parameter_tolerance /= 10;
        custom_options.solver_options.max_num_iterations *= 3;
        custom_options.solver_options.max_linear_solver_iterations = 200;
    }

    PrintHeading1("Global bundle adjustment");
   
    custom_options.workspace_path = mapper->GetWorkspacePath();
    
    mapper->AdjustGlobalBundle(options.IncrementalMapperOptions(), custom_options);
}

void IterativeGlobalBA(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper,
                    std::string workspace_path = "") {
    
    PrintHeading1("CompleteAndMergeTracks");
    CompleteAndMergeTracks(options, mapper);
    PrintHeading1("Retriangulation");
    std::cout << "  => Retriangulated observations: " << mapper->Retriangulate(options.Triangulation()) << std::endl;

    for (int i = 0; i < options.ba_global_max_refinements; ++i) {
        const size_t num_observations = mapper->GetReconstruction().ComputeNumObservations();
        size_t num_changed_observations = 0;
        AdjustGlobalBundle(options, mapper);

        num_changed_observations += CompleteAndMergeTracks(options, mapper);
        num_changed_observations += FilterPoints(options, mapper);

        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < options.ba_global_max_refinement_change) {
            break;
        }
    }
    FilterImages(options, mapper);
}




int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading("Version: loop-closure-1.6.8");

    Timer timer;
    timer.Start();

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    
    //load feature_data_container
    auto feature_data_container = std::make_shared<FeatureDataContainer>();
    feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
    feature_data_container->ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
    feature_data_container->ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path, "/features.bin"));

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
        feature_data_container->ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
    } else {
        std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
    }

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
        feature_data_container->ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
    } else {
        std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
    }

    //load scene graph
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    scene_graph_container->ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/strong_loops.bin"))) {
        std::cout<<"read strong_loops file "<<std::endl;
        scene_graph_container->ReadStrongLoopsBinaryData(JoinPaths(workspace_path, "/strong_loops.bin"));
    }

    SceneGraphContainer &scene_graph = *(scene_graph_container.get());

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph.Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph.Cameras();

    std::vector<image_t> image_ids = feature_data_container->GetImageIds();

    for (const auto image_id : image_ids) {
        const Image &image = feature_data_container->GetImage(image_id);
        const Camera &camera = feature_data_container->GetCamera(image.CameraId());
        if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        images[image_id] = image;

        const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
        images[image_id].SetPoints2D(keypoints);

        const PanoramaIndexs &panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        images[image_id].SetLocalImageIndices(local_image_indices);

        if (!scene_graph.ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
        }

        if (scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(scene_graph.CorrespondenceGraph()->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(
                scene_graph.CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
        } else {
            std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
        }
    }

    scene_graph.CorrespondenceGraph()->Finalize();

    feature_data_container.reset();
    
    // Load reconstructions
    CHECK(boost::filesystem::exists(workspace_path + "/0"));
    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->ReadReconstruction(workspace_path + "/0");


    //create mapper 
    IndependentMapperOptions mapper_options;
    option_parser.GetMapperOptions(mapper_options, param);

    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);

    
    mapper->SetWorkspacePath(workspace_path);
    if (mapper_options.explicit_loop_closure) {
        PrintHeading1("Explicit loop closure");
        if (mapper->AdjustCameraByLoopClosure(mapper_options.IncrementalMapperOptions())) {
            mapper->RetriangulateAllTracks(mapper_options.Triangulation());
            if (mapper_options.debug_info) {
                std::string recon_path =
                    StringPrintf("%s/after_loop_%d/", workspace_path.c_str(), reconstruction->NumRegisterImages());
                boost::filesystem::create_directories(recon_path);
                mapper->GetReconstruction().WriteReconstruction(recon_path, true);
            }
        }
    }

    bool ba_after_loop = static_cast<bool>(param.GetArgument("ba_after_loop", 0));
    if(ba_after_loop){    
        IterativeGlobalBA(mapper_options,mapper,workspace_path);
        std::string recon_path =
                StringPrintf("%s/after_ba_%d/", workspace_path.c_str(), reconstruction->NumRegisterImages());
        boost::filesystem::create_directories(recon_path);
        mapper->GetReconstruction().WriteReconstruction(recon_path, true);
    }

    auto keyframe_reconstruction = std::make_shared<Reconstruction>();
    keyframe_reconstruction->ReadReconstruction(workspace_path + "/0/KeyFrames");
    std::unordered_set<image_t> keyframe_ids;
    const std::vector<image_t> keyframe_registered_image_ids =  keyframe_reconstruction->RegisterImageIds();
    for(size_t i = 0; i<keyframe_registered_image_ids.size(); ++i){
        CHECK(reconstruction->ExistsImage(keyframe_registered_image_ids[i]));
        keyframe_ids.insert(keyframe_registered_image_ids[i]);
    }

    std::unordered_set<mappoint_t> all_mappoints = reconstruction->MapPointIds();

    keyframe_reconstruction.reset();
    keyframe_reconstruction = std::make_shared<Reconstruction>();

    reconstruction->Copy(keyframe_ids,all_mappoints,keyframe_reconstruction);

    std::string looped_rec_path = StringPrintf("%s/0-loop", workspace_path.c_str());
    if (boost::filesystem::exists(looped_rec_path)) {
        boost::filesystem::remove_all(looped_rec_path);
    }
    boost::filesystem::create_directories(looped_rec_path);

    reconstruction->WriteBinary(looped_rec_path);

    std::string looped_keyframe_rec_path = StringPrintf("%s/0-loop/KeyFrames", workspace_path.c_str());
    if (boost::filesystem::exists(looped_keyframe_rec_path)) {
        boost::filesystem::remove_all(looped_keyframe_rec_path);
    }
    boost::filesystem::create_directories(looped_keyframe_rec_path);

    keyframe_reconstruction->WriteBinary(looped_keyframe_rec_path);


    std::cout << std::endl;
    timer.PrintMinutes();
    return 0;
}


