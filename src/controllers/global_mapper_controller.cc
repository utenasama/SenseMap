//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <util/misc.h>
#include "global_mapper_controller.h"


namespace sensemap {

namespace {

size_t FilterPoints(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions());
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterPoints(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper,
                    int min_track_length) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions(), min_track_length);
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterPointsFinal(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_observations = mapper->FilterPointsFinal(options.IncrementalMapperOptions());
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

size_t TriangulateImage(const IndependentMapperOptions& options, const Image& image,
                        std::shared_ptr<IncrementalMapper> mapper) {
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

    std::cout << "  => Continued observations: " << image.NumMapPoints() << std::endl;
    const size_t num_tris = mapper->TriangulateImage(options.Triangulation(), image.ImageId());
    std::cout << "  => Added observations: " << num_tris << std::endl;

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::cout << StringPrintf(
            "  => TriangulateImage Elapsed time: %.3f [second]",
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 6e4)
            .c_str()
              << std::endl;

    return num_tris;
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   std::shared_ptr<Reconstruction> reconstruction,
                   const bool as_rgbd = false) {
    if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
        std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                                  reconstruction->Image(image_id).Name().c_str(), image_path.c_str())
                  << std::endl;
    }
}

}


GlobalMapperController::GlobalMapperController(
		const std::shared_ptr<IndependentMapperOptions> options,
		const std::string &image_path,
		const std::string& workspace_path,
		const std::shared_ptr<SceneGraphContainer> scene_graph_container,
		std::shared_ptr<ReconstructionManager> reconstruction_manager)
		: IndependentMapperController(options, image_path, workspace_path,
				scene_graph_container, reconstruction_manager) {
	CHECK(options_->GlobalMapperCheck());
    PrintHeading1("Global Mapper");
}

void GlobalMapperController::Run() {
    init_mapper_options_ = options_->GlobalMapperOptions();
    init_mapper_options_.image_path = image_path_;
    Reconstruct();

    std::cout << std::endl;
    GetTimer().PrintMinutes();
}

void GlobalMapperController::Reconstruct() {

    auto mapper =  std::make_shared<GlobalMapper>(scene_graph_container_);

    // Is there a sub-model before we start the reconstruction? I.e. the user
    // has imported an existing reconstruction.
    const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
    CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                    "single reconstruction, but multiple are given.";

    size_t reconstruction_idx = 0;
    if (!initial_reconstruction_given > 0) {
        reconstruction_idx = reconstruction_manager_->Add();
    }

    std::shared_ptr<Reconstruction> reconstruction =
            reconstruction_manager_->Get(reconstruction_idx);

    LOG(INFO) << "Filtering the intial view graph.";
    mapper->BeginReconstruction(reconstruction);


    // Step 1. Filter the initial view graph and remove any bad two view
    // geometries.
    if (!mapper->FilterInitialImageGraph(init_mapper_options_)) {
        LOG(INFO) << "Insufficient view pairs to perform estimation.";
        return ;
    }

    // Step 2. Calibrate any uncalibrated cameras.
    //LOG(INFO) << "Calibrating any uncalibrated cameras.";

    // Step 3. Estimate global rotations.
    LOG(INFO) << "Estimating the global rotations of all cameras.";
    if (!mapper->EstimateGlobalRotations(init_mapper_options_)) {
        LOG(WARNING) << "Rotation estimation failed!";
        return;
    }

    // Step 4. Filter bad rotations.
    //  LOG(INFO) << "Filtering any bad rotation estimations.";
    //mapper->FilterRotations(init_mapper_options_);

    // Step 5. Optimize relative translations.
    LOG(INFO) << "Optimizing the pairwise translation estimations.";
    mapper->OptimizePairwiseTranslations(init_mapper_options_);

    // Step 6. Filter bad relative translations.
    LOG(INFO) << "Filtering any bad relative translations.";
    mapper->FilterRelativeTranslation(init_mapper_options_);

    // Step 7. Estimate global positions.
    LOG(INFO) << "Estimating the positions of all cameras.";
    if (!mapper->EstimatePosition(init_mapper_options_)) {
        LOG(WARNING) << "Position estimation failed!";
        return ;
    }

#if 0
    // Step 8. Triangulate features.

    PrintHeading1("Triangulation All");
    mapper->EsitimateStructure(options_->Triangulation());

    // Always triangulate once, then retriangulate and remove outliers depending
    // on the reconstruciton estimator options.
    for (int i = 0; i < options_->ba_global_max_refinements; i++) {

        // Do a single step of bundle adjustment where only the camera positions and
        // 3D points are refined. This is only done for the very first bundle
        // adjustment iteration.
        if (i == 0 &&
                init_mapper_options_.refine_camera_positions_and_points_after_position_estimation) {
            LOG(INFO) << "Performing partial bundle adjustment to optimize only the "
                         "camera positions and 3d points.";
            auto ba_options = options_->GlobalBundleAdjustment();
            ba_options.refine_focal_length = false;
            ba_options.refine_principal_point = false;
            ba_options.refine_extra_params = false;
            ba_options.refine_extrinsics = true;
            mapper->AdjustGlobalBundle(init_mapper_options_, ba_options);

        }


        {
            auto ba_options = options_->GlobalBundleAdjustment();
            ba_options.refine_focal_length = true;
            ba_options.refine_principal_point = false;
            ba_options.refine_extra_params = false;
            ba_options.refine_extrinsics = false;
            mapper->AdjustGlobalBundle(init_mapper_options_, ba_options);
        }


        // Step 9. Bundle Adjustment.
        auto ba_options = options_->GlobalBundleAdjustment();
        ba_options.refine_focal_length = true;
        ba_options.refine_principal_point = true;
        ba_options.refine_extra_params = true;
        ba_options.refine_extrinsics = true;
        mapper->AdjustGlobalBundle(init_mapper_options_, ba_options);



        for(auto camera : reconstruction->Cameras()) {
            for(auto param : camera.second.Params())
                 std::cout << param <<" "<< std::endl;
        }
        std::cout<<std::endl;
    }

#endif

    {
        std::string rec_path = StringPrintf("%s/%d-global", workspace_path_.c_str(), reconstruction_idx);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }
        reconstruction->WriteReconstruction(rec_path, true);


        std::string prior_rec_path = rec_path + "-tri-prior";
        if (!boost::filesystem::exists(prior_rec_path)) {
            boost::filesystem::create_directories(prior_rec_path);
        }
        auto reconstruction_copy = *reconstruction;
        reconstruction_copy.AlignWithPriorLocations(options_->max_error_gps);
        reconstruction_copy.AddPriorToResult();
        reconstruction_copy.NormalizeWoScale();
        reconstruction_copy.WriteReconstruction(prior_rec_path, true);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Bundle adjustment
    //////////////////////////////////////////////////////////////////////////////
    int min_track_length = 3;
    const auto tri_options = options_->Triangulation();
    std::shared_ptr<IncrementalMapper> incrematal_mapper = std::make_shared<IncrementalMapper>(scene_graph_container_);
    incrematal_mapper->BeginReconstruction(reconstruction);

    for(auto pointid : reconstruction->MapPointIds()){
        reconstruction->DeleteMapPoint(pointid);
    }

    int triangulated_image_count = 1;
    std::vector<image_t> image_ids = scene_graph_container_->GetImageIds();
    for (const auto image_id : image_ids) {
        Image &image_scene = scene_graph_container_->Image(image_id);
        if (reconstruction->ExistsImage(image_scene.ImageId()) &&
            reconstruction->IsImageRegistered(image_scene.ImageId())){
            reconstruction->DeRegisterImage(image_scene.ImageId());
            //continue;
        }
        if (!reconstruction->ExistsImage(image_id)){
            reconstruction->AddImage(image_scene);
        }
        reconstruction->RegisterImage(image_id);
        Image &image = reconstruction->Image(image_id);

        Camera &camera = reconstruction->Camera(image.CameraId());

        const size_t num_existing_points3D = image.NumMapPoints();

        incrematal_mapper->TriangulateImage(tri_options, image_id);

        std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points"
                  << std::endl;
    }

    CompleteAndMergeTracks(*options_, incrematal_mapper);

    std::vector<image_t> reg_image_ids = reconstruction->RegisterImageIds();

    auto ba_options = options_->GlobalBundleAdjustment();
    ba_options.refine_focal_length = true;
    ba_options.refine_principal_point = true;
    ba_options.refine_extra_params = true;
    ba_options.refine_extrinsics = true;

    if( options_->use_translation_prior_constrain){
        reconstruction->b_aligned = true;
        ba_options.use_prior_absolute_location = true;
        // ba_options.prior_absolute_location_weight = 0.1;
        std::cout << "prior_absolute_location_weight: " << ba_options.prior_absolute_location_weight
                  << "\t loss function: " << options_->ba_global_loss_function << std::endl;
    } else {
        reconstruction->b_aligned = false;
        ba_options.use_prior_absolute_location = false;
    }

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;

    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        const image_t image_id = reg_image_ids[i];
        if (!scene_graph_container_->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }
        ba_config.AddImage(image_id);
    }

    for (int i = 0; i < options_->ba_global_max_refinements + 2; ++i) {
        reconstruction->FilterObservationsWithNegativeDepth();

        const size_t num_observations = reconstruction->ComputeNumObservations();

        PrintHeading1("RTK Bundle adjustment");
        std::cout << "iter: " << i << std::endl;
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        CHECK(bundle_adjuster.Solve(reconstruction.get()));

        size_t num_changed_observations = 0;
        num_changed_observations += CompleteAndMergeTracks(*options_, incrematal_mapper);
        num_changed_observations += FilterPoints(*options_, incrematal_mapper, min_track_length);
        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;

        size_t num_retriangulate_observations = 0;
        num_retriangulate_observations = incrematal_mapper->Retriangulate(options_->Triangulation());
        std::cout << "\nnum_retri_observations / num_ori_observations: "
                  << num_observations << " / "
                  << num_retriangulate_observations << std::endl;

        if (changed < options_->ba_global_max_refinement_change) {
            break;
        }
    }

    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        if (options_->extract_colors) {
            ExtractColors(image_path_, reg_image_ids[i], reconstruction);
        }
    }

    std::string rec_path = StringPrintf("%s/%d", workspace_path_.c_str(), 0);
    if (!boost::filesystem::exists(rec_path)) {
        boost::filesystem::create_directories(rec_path);
    }

    Reconstruction rec = *reconstruction.get();
    reconstruction->WriteReconstruction(rec_path, true);

    rec.AddPriorToResult();
    rec.NormalizeWoScale();

    std::string trans_rec_path = rec_path + "-gps";
    if (!boost::filesystem::exists(trans_rec_path)) {
        boost::filesystem::create_directories(trans_rec_path);
    }
    rec.WriteBinary(trans_rec_path);
    Eigen::Matrix3x4d matrix_to_geo = reconstruction->NormalizeWoScale();
    rec.WriteReconstruction(trans_rec_path, true);
}


} // namespace sensemap
