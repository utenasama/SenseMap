//Copyright (c) 2021, SenseTime Group.
//All rights reserved.

#include <unordered_map>
#include <util/misc.h>
#include <base/pose.h>
#include "directed_mapper_controller.h"
#include "global_mapper_controller.h"
#include <sfm/global_mapper.h>
#include <base/similarity_transform.h>
#include <malloc.h>
#include "graph/minimum_spanning_tree.h"
#include "graph/maximum_spanning_tree_graph.h"
#include "util/proc.h"

namespace sensemap {

namespace {
size_t FilterPoints(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions());
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterPoints(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper,
                    int min_track_length) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions(), min_track_length);
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterPointsFinal(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_observations = mapper->FilterPointsFinal(options.IncrementalMapperOptions());
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterImages(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_images = mapper->FilterImages(options.IncrementalMapperOptions());
    std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
    return num_filtered_images;
}

size_t CompleteAndMergeTracks(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_completed_observations = mapper->CompleteTracks(options.Triangulation());
    std::cout << "  => Completed observations: " << num_completed_observations << std::endl;
    const size_t num_merged_observations = mapper->MergeTracks(options.Triangulation());
    std::cout << "  => Merged observations: " << num_merged_observations << std::endl;
    return num_completed_observations + num_merged_observations;
}

float EstimateMemoryFromResiduals(size_t num_residuals) {
    float max_ram = (0.554168 * num_residuals - 49347.664344) * 1e-6;
    return max_ram;
}

void IterativeTrackSelection(const Reconstruction *reconstruction, 
                             std::vector<class MapPoint *> &mappoints,
                             const size_t max_cover_per_view) {
    for (auto &mappoint : reconstruction->MapPoints()) {
        mappoints.emplace_back((class MapPoint *)&mappoint.second);
    }

    std::sort(mappoints.begin(), mappoints.end(), [](class MapPoint *mappoint1, class MapPoint *mappoint2) {
        const class Track *track1 = &mappoint1->Track();
        const class Track *track2 = &mappoint2->Track();
        if (track1->Elements().size() == track2->Elements().size()) {
            return mappoint1->Error() < mappoint2->Error();
        } else {
            return track1->Elements().size() > track2->Elements().size();
        }
    });

    std::vector<unsigned char> inlier_masks(mappoints.size(), 0);
    std::unordered_map<image_t, int> cover_view;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (inlier_masks.at(i)) {
            continue;
        }
        auto track = mappoints.at(i)->Track();
        bool selected = false;
        for (auto track_elem : track.Elements()) {
            image_t image_id = track_elem.image_id;
            std::unordered_map<image_t, int>::iterator it = 
                cover_view.find(image_id);
            if (it == cover_view.end() || it->second < max_cover_per_view) {
                cover_view[image_id]++;
                selected = true;
                break;
            }
        }

        inlier_masks[i] = selected;

        if (selected) {
            std::unordered_set<image_t> track_images;
            for (auto track_elem : track.Elements()) {
                track_images.insert(track_elem.image_id);
            }
            for (auto image_id : track_images) {
                cover_view[image_id]++;
            }
        } else {
            mappoints.at(i) = NULL;
        }
    }
    int i, j;
    for (i = 0, j = 0; i < mappoints.size(); ++i) {
        if (mappoints[i]) {
            mappoints[j] = mappoints[i];
            j = j + 1;
        }
    }
    mappoints.resize(j);
}

void IterativeLocalRefinement(const IndependentMapperOptions& options, const image_t image_id,
                              std::shared_ptr<IncrementalMapper> mapper) {
    auto ba_options = options.LocalBundleAdjustment();

    if (options.single_camera) {
        ba_options.refine_extra_params = false;
        ba_options.refine_focal_length = false;
        ba_options.refine_principal_point = false;
        ba_options.refine_local_extrinsics = false;
    }

    for (int i = 0; i < options.ba_local_max_refinements; ++i) {
        const auto report = mapper->AdjustLocalBundle(options.IncrementalMapperOptions(), ba_options, options.Triangulation(), image_id,
                                                      mapper->GetModifiedMapPoints());
        std::cout << "  => Merged observations: " << report.num_merged_observations << std::endl;
        std::cout << "  => Completed observations: " << report.num_completed_observations << std::endl;
        std::cout << "  => Filtered observations: " << report.num_filtered_observations << std::endl;

        const double changed =
                (report.num_merged_observations + report.num_completed_observations + report.num_filtered_observations) /
                static_cast<double>(report.num_adjusted_observations);
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < options.ba_local_max_refinement_change) {
            break;
        }

    }
    mapper->ClearModifiedMapPoints();
}

}

// #define GSFM

DirectedMapperController::DirectedMapperController(
        const std::shared_ptr<IndependentMapperOptions> options,
        const std::string &image_path,
        const std::string& workspace_path,
        const std::shared_ptr<SceneGraphContainer> scene_graph_container,
        std::shared_ptr<ReconstructionManager> reconstruction_manager)
        : IndependentMapperController(options, image_path, workspace_path,
                                      scene_graph_container, reconstruction_manager) {
    CHECK(options_->IncrementalMapperCheck());
    PrintHeading1("Directed Mapper");
}


bool RTKReady(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container, double threshold = 0.8) {
    auto image_ids = scene_graph_container->GetImageIds();
    double ready_num = 0;
    for (auto id : image_ids) {
        auto image = scene_graph_container->Image(id);
        if (image.RtkFlag() == 50) {
            ready_num++;
        }
    }
    return ready_num / image_ids.size() > threshold;
}

#define MIN_LOCAL_SIZE 40
#define MAX_LOCAL_SIZE 200

bool LocalGSFM(const std::shared_ptr<IndependentMapperOptions> &options,
               const IncrementalMapper::Options &mapper_options,
               const std::string &workspace_path,
               std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
               bool rtk_ready, bool use_block_ba = false){

    size_t local_size;
    if(rtk_ready && !use_block_ba){
        local_size = MIN_LOCAL_SIZE;
    } else {
        local_size = MAX_LOCAL_SIZE;
    }

    bool rtk_good = false;

    std::unordered_map<image_t, std::unordered_set<image_t> > cc;
    GetAllConnectedComponentIds(*scene_graph_container->CorrespondenceGraph(), cc);
    std::cout << StringPrintf("Get %d component\n", cc.size());

    std::shared_ptr<SceneGraphContainer> cluster_graph_container =
        std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
    if (cc.size() > 1) {
        std::vector<std::pair<image_t, int> > sorted_clustered_images;
        sorted_clustered_images.reserve(cc.size());
        for (auto & clustered_image_ids : cc) {
            sorted_clustered_images.emplace_back(clustered_image_ids.first, clustered_image_ids.second.size());
        }
        std::sort(sorted_clustered_images.begin(), sorted_clustered_images.end(), 
            [&](const auto & image1, const auto & image2) {
                return image1.second > image2.second;
            });

        // Get Max-connected component.
        std::set<image_t> unique_image_ids;
        for (auto & image_id : cc[sorted_clustered_images[0].first]) {
            unique_image_ids.insert(image_id);
        }
        scene_graph_container->ClusterSceneGraphContainer(unique_image_ids, *cluster_graph_container.get());
    } else {
        cluster_graph_container = scene_graph_container;
    }

    auto correspondence_graph = cluster_graph_container->CorrespondenceGraph();
    auto image_neighors = correspondence_graph->ImageNeighbors();
    auto image_pairs = correspondence_graph->ImagePairs();

    std::vector<std::pair<float, std::set<image_t>>> scores;

    ///construct mst_extractor satisfied the conditions
    MinimumSpanningTree<image_t, float> mst_extractor;
    for (const auto &image_pair : image_pairs) {
        image_t image_id1;
        image_t image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first,
                                             &image_id1, &image_id2);

        auto image1 = cluster_graph_container->Image(image_id1);
        auto image2 = cluster_graph_container->Image(image_id2);

        mst_extractor.AddEdge(
#ifdef TWO_VIEW_CONFIDENCE
            image_id1, image_id2, -image_pair.second.two_view_geometry.confidence);
#else
            image_id1, image_id2, (float)-image_pair.second.num_correspondences);
#endif
    }

    local_size = std::min(cluster_graph_container->NumImages(), (size_t) local_size);
    std::cout << "local_size: " << local_size << std::endl;

    for (auto &image : cluster_graph_container->Images()) {

        ///find best edge
        image_t best_id = -1;
#ifdef TWO_VIEW_CONFIDENCE
        float max_confidence = 0.0f;
        for (auto &neighbor_id : image_neighors[image.first]) {
            auto image_pair = correspondence_graph->ImagePair(image.first, neighbor_id);
            double confidence = image_pair.two_view_geometry.confidence;
            if (confidence > max_confidence) {
                best_id = neighbor_id;
                max_confidence = confidence;
            }
        }
#else
        int max_correspondences_num = 0;

        for (auto &neighbor_id : image_neighors[image.first]) {
            auto num = correspondence_graph->NumCorrespondencesBetweenImages(image.first, neighbor_id);
            if (num > max_correspondences_num) {
                best_id = neighbor_id;
                max_correspondences_num = num;
            }
        }
#endif

        if (best_id == -1 /*|| max_correspondences_num < min_correspondences_num*/) {
            //                printf("pair id: %d <-> %d, max_correspondences_num: %d\n", image.first, best_id, max_correspondences_num);
            //                std::cout << std::flush;
            continue;
        }

        ///extract MST
        std::set<image_t> mst_nodes;

        //make sure the best edge at the top of mst
        if (!mst_extractor.SetWeight(image.first, best_id, std::numeric_limits<float>::lowest())) {
            //                 printf("pair id: %d <-> %d, set weight failed\n", image.first, best_id);
            //                 std::cout << std::flush;
            continue;
        }

        if (!mst_extractor.ExtractLocalNodes(&mst_nodes, local_size)) {
            //                 printf("pair id: %d <-> %d, ExtractLocalNodes failed\n", image.first, best_id);
            //                 std::cout << std::flush;
            auto pair_id = sensemap::utility::ImagePairToPairId(image.first, best_id);
#ifdef TWO_VIEW_CONFIDENCE
            mst_extractor.SetWeight(image.first, best_id, -image_pairs[pair_id].two_view_geometry.confidence);
#else
            mst_extractor.SetWeight(image.first, best_id, (float)-image_pairs[pair_id].num_correspondences);
#endif
            continue;
        }

        ///calculate score
        float score = 0;
        std::vector<image_t> v_mst_nodes;
        v_mst_nodes.assign(mst_nodes.begin(), mst_nodes.end());
        for (int i = 0; i < v_mst_nodes.size(); ++i) {
            for (int j = i + 1; j < v_mst_nodes.size(); ++j) {
#ifdef TWO_VIEW_CONFIDENCE
                if (correspondence_graph->ExistImagePair(v_mst_nodes[i], v_mst_nodes[j])) {
                    float conf = correspondence_graph->ImagePair(v_mst_nodes[i], v_mst_nodes[j]).two_view_geometry.confidence;
                    score += conf;
                }
#else
                size_t num_corrs = correspondence_graph->NumCorrespondencesBetweenImages(v_mst_nodes[i], v_mst_nodes[j]);
                score += num_corrs;
#endif
            }
        }

        std::pair<float, std::set<image_t>> score_pair;
        score_pair.first = score;
        score_pair.second = mst_nodes;
        scores.emplace_back(std::move(score_pair));

        //restore mst_extractor
        auto pair_id = sensemap::utility::ImagePairToPairId(image.first, best_id);
#ifdef TWO_VIEW_CONFIDENCE
        mst_extractor.SetWeight(image.first, best_id, -image_pairs[pair_id].two_view_geometry.confidence);
#else
        mst_extractor.SetWeight(image.first, best_id, (float)-image_pairs[pair_id].num_correspondences);
#endif
    }


    std::cout << std::flush;
    std::sort(scores.begin(), scores.end(),
        [&](const std::pair<float, std::set<image_t>> & s1,
            const std::pair<float, std::set<image_t>> & s2) {
        return s1.first > s2.first;
    });

    if (scores.empty()) {
        std::cout << "error: cannot find local spanning trees" << std::endl;
        return false;
    }

    std::set<image_t> local_nodes = scores[0].second; //(mst_nodes.begin(), mst_nodes.begin() + local_num);

    if (local_nodes.size() < local_size) {
        std::cout << "error: not enough images for  gsfm locally " << std::endl;
        return false;
    }

    for(auto id : local_nodes){
        std::cout<<id<<" ";
    }
    std::cout<<std::endl;

    auto local_container = std::make_shared<SceneGraphContainer>();
    cluster_graph_container->ClusterSceneGraphContainer(local_nodes, *local_container.get());

    bool camera_rig = false;
    for(auto camera : cluster_graph_container->Cameras()) {
        if (camera.second.NumLocalCameras() > 2) {
            camera_rig = true;
        }
    }
    if (rtk_ready && camera_rig) {
        for (auto & image : local_container->Images()) {
            if (image.second.RtkFlag() == 50) {
                image.second.SetPriorQvecGood(true);
            }
        }
    }

    //global sfm locally

    auto global_options = options->GlobalMapperOptions();
    global_options.use_rotation_prior_constrain = camera_rig && options->has_gps_prior;
    global_options.use_translation_prior_constrain = camera_rig && options->has_gps_prior;
    global_options.rotation_prior_constrain_weight = 100;
    global_options.translation_prior_constrain_weight = 200;
    global_options.translation_prior_weak_constrain_weight = 1;
    global_options.two_steo_refinement_of_position = options->has_gps_prior;

    std::shared_ptr<GlobalMapper> local_mapper = std::make_shared<GlobalMapper>(local_container);
    auto local_reconstruction = std::make_shared<Reconstruction>();

    local_mapper->BeginReconstruction(local_reconstruction);

    local_mapper->EstimateGlobalRotations(global_options);

    local_mapper->OptimizePairwiseTranslations(global_options);

    local_mapper->FilterRelativeTranslation(global_options);

    if (!local_mapper->EstimatePosition(global_options)) {
        LOG(WARNING) << "Position estimation failed!";
        return false;
    }

    if(local_size > 100) {
        //////////////////////////////////////////////////////////////////////////////
        // Bundle adjustment
        //////////////////////////////////////////////////////////////////////////////
        int min_track_length = 3;
        if (local_size == MAX_LOCAL_SIZE) {
            auto tracks = local_container->CorrespondenceGraph()->GenerateTracks();
            std::sort(tracks.begin(), tracks.end(), 
                [](const Track& t1, const Track& t2) {
                    return t1.Length() > t2.Length();
                });
            std::vector<unsigned char> inlier_masks(tracks.size(), 0);
            std::unordered_set<image_t> const_image_set;
            local_container->DistributedTrackSelection(tracks, inlier_masks, 200, 25, 5, 2000, const_image_set);
        }

        const auto tri_options = options->Triangulation();
        std::shared_ptr<IncrementalMapper> incrematal_mapper = std::make_shared<IncrementalMapper>(local_container);
        incrematal_mapper->BeginReconstruction(local_reconstruction);

        for(auto pointid : local_reconstruction->MapPointIds()){
            local_reconstruction->DeleteMapPoint(pointid);
        }

        int triangulated_image_count = 1;
        std::vector<image_t> image_ids = local_container->GetImageIds();
        for (const auto image_id : image_ids) {
            Image &image_scene = local_container->Image(image_id);
            if (local_reconstruction->ExistsImage(image_scene.ImageId()) &&
                local_reconstruction->IsImageRegistered(image_scene.ImageId())){
                local_reconstruction->DeRegisterImage(image_scene.ImageId());
                //continue;
            }
            if (!local_reconstruction->ExistsImage(image_id)){
                local_reconstruction->AddImage(image_scene);
            }
            local_reconstruction->RegisterImage(image_id);
            Image &image = local_reconstruction->Image(image_id);

            Camera &camera = local_reconstruction->Camera(image.CameraId());

            const size_t num_existing_points3D = image.NumMapPoints();

            incrematal_mapper->TriangulateImage(tri_options, image_id);

            std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points"
                      << std::endl;
        }

        PrintHeading1("CompleteAndMergeTracks");
        CompleteAndMergeTracks(*options, incrematal_mapper);

        std::vector<image_t> reg_image_ids = local_reconstruction->RegisterImageIds();

        auto ba_options = options->GlobalBundleAdjustment();
        if (local_size <= MIN_LOCAL_SIZE){
            ba_options.refine_focal_length = false;
            ba_options.refine_principal_point = false;
            ba_options.refine_extra_params = false;
        } else {
            ba_options.refine_focal_length = true;
            ba_options.refine_principal_point = true;
            ba_options.refine_extra_params = true;
        }
        ba_options.refine_extrinsics = true;
        local_reconstruction->b_aligned = false;
        ba_options.use_prior_absolute_location = false;
        // ba_options.solver_options.max_linear_solver_iterations = 200;
        // ba_options.solver_options.minimizer_progress_to_stdout = true;

        // Configure bundle adjustment.
        BundleAdjustmentConfig ba_config;

        for (size_t i = 0; i < reg_image_ids.size(); ++i) {
            const image_t image_id = reg_image_ids[i];
            if (!local_container->CorrespondenceGraph()->ExistsImage(image_id)) {
                continue;
            }
            ba_config.AddImage(image_id);
        }

        for (int i = 0; i < 2 ; ++i) {
            local_reconstruction->FilterObservationsWithNegativeDepth();

            const size_t num_observations = local_reconstruction->ComputeNumObservations();

            PrintHeading1("RTK LocalGSfM Bundle adjustment");
            std::cout << "iter: " << i << std::endl;
            BundleAdjuster bundle_adjuster(ba_options, ba_config);
            CHECK(bundle_adjuster.Solve(local_reconstruction.get()));

            size_t num_changed_observations = 0;
            num_changed_observations += CompleteAndMergeTracks(*options, incrematal_mapper);
            num_changed_observations += FilterPoints(*options, incrematal_mapper, min_track_length);
            const double changed = static_cast<double>(num_changed_observations) / num_observations;
            std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;

            size_t num_retriangulate_observations = 0;
            num_retriangulate_observations = incrematal_mapper->Retriangulate(options->Triangulation());
            std::cout << "\nnum_retri_observations / num_ori_observations: "
                        << num_observations << " / "
                        << num_retriangulate_observations << std::endl;

            if (changed < options->ba_global_max_refinement_change) {
                break;
            }
        }
    }

    for (auto &camera: local_reconstruction->Cameras()) {
        scene_graph_container->Camera(camera.first).Params() =
            local_reconstruction->Camera(camera.first).Params();
    }
    if(rtk_ready) {
        // Align with gps
        Eigen::Matrix3x4d matrix_to_align =
            local_reconstruction->AlignWithPriorLocations(mapper_options.max_error_gps);

        if (options->debug_info) {
            for (auto & camera : local_reconstruction->Cameras()) {
                std::string gsfm_path = JoinPaths(workspace_path, StringPrintf("/lsfm-%d", camera.first));
                if (!boost::filesystem::exists(gsfm_path)) {
                    boost::filesystem::create_directories(gsfm_path);
                }

                Reconstruction rig_reconstruction;
                local_reconstruction->ConvertRigReconstruction(rig_reconstruction);

                rig_reconstruction.AddPriorToResult();

                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                rig_reconstruction.FilterUselessPoint2D(filtered_reconstruction);

                filtered_reconstruction->WriteBinary(gsfm_path);
            }
        }

        // // compare tvec
        // const double inlier_ratio_thresh = 0.9f;
        // int inlier_count = 0;
        // for (auto id : local_reconstruction->RegisterImageIds()) {
        //     auto image = local_reconstruction->Image(id);
        //     auto C1 = image.ProjectionCenter();
        //     if (image.HasTvecPrior()) {
        //         auto C2 = image.TvecPrior();
        //         double dist = (C1 - C2).norm();
        //         if (dist < mapper_options.max_error_gps) {
        //             inlier_count++;
        //         }
        //     }
        // }

        // compare qvec
        const double inlier_ratio_thresh = 0.7f;
        int inlier_count = 0;
        for (auto id : local_reconstruction->RegisterImageIds()) {
            auto qvec1 = local_reconstruction->Image(id).Qvec();
            auto qvec2 = local_reconstruction->Image(id).QvecPrior();

            auto R1 = QuaternionToRotationMatrix(qvec1);
            auto R2 = QuaternionToRotationMatrix(qvec2);
            Eigen::Matrix3d R_diff = (R1.transpose()) * R2;
            Eigen::AngleAxisd angle_axis(R_diff);
            double delta = std::abs(RadToDeg(angle_axis.angle()));
            // std::cout<<delta<<std::endl;
            if (delta < 30) inlier_count++;
        }

        double inlier_ratio = static_cast<double>(inlier_count) / local_reconstruction->RegisterImageIds().size();
        std::cout << "RTK inlier ratio : " << inlier_ratio << std::endl;
        if (inlier_ratio >= inlier_ratio_thresh) {
            rtk_good = true;
        } else {
            rtk_good = false;
        }
    }
    for (auto &camera: scene_graph_container->Cameras()) {
        if (camera.second.NumLocalCameras() > 1) {
            std::cout << "Camera#" << camera.first << ", param: " << VectorToCSV(camera.second.LocalParams()) << std::endl;
        } else {
            std::cout << "Camera#" << camera.first << ", param: " << camera.second.ParamsToString() << std::endl;
        }
    }
    return rtk_good;
}

void DirectedMapperController::Reconstruct() {
    Timer timer;
    timer.Start();

    switch (mapper_options_.direct_mapper_type) {
        case 1 :
            std::cout<<"Type : general GSFM"<<std::endl; //Auto choose using RTK or not.
            break;
        case 2:
            std::cout<<"Type : RTK GSFM"<<std::endl; //If not have RTK or not ready, return;
            break;
        case 3:
            std::cout<<"Type : no RTK GSFM"<<std::endl; //GSFM without RTK whether it has rtk or not
            break;
        case 4:
            std::cout<<"Type : RTK Directed"<<std::endl; //If has good RTK, directed use it (without gsfm). If not, return.
            break;
        default:
            mapper_options_.direct_mapper_type = 1;
            std::cout<<"Type : general GSFM"<<std::endl;
            break;
    }


    std::vector<image_t> orig_image_ids = scene_graph_container_->GetImageIds();

    bool use_block_ba = false;
    if (options_->ba_block_size != -1 && orig_image_ids.size() > options_->ba_block_size) {
        use_block_ba = true;
    }

    ///Local GSFM (check RTK or Refine Intrinsic)-------------------------------------------------
    auto st1 = std::chrono::steady_clock::now();

    auto minimum_spanning_tree = scene_graph_container_->GetTreeEdges();
    {
        size_t num_invalid_relative_pose = 0;
        auto image_pairs = scene_graph_container_->CorrespondenceGraph()->ImagePairs();
        std::cout << "Delete invalid relative pose." << std::endl;
        for (auto image_pair : image_pairs) {
            image_t image_id1, image_id2;
            utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
            auto two_view_geometry = scene_graph_container_->CorrespondenceGraph()->ImagePair(image_pair.first).two_view_geometry;
            if (minimum_spanning_tree->find(image_pair.first) == minimum_spanning_tree->end() && 
                two_view_geometry.qvec.norm() <= 1e-6) {
                scene_graph_container_->CorrespondenceGraph()->DeleteCorrespondences(image_id1, image_id2);
                num_invalid_relative_pose++;
            }
        }
        std::cout << StringPrintf("delete %d / %d pairs\n", num_invalid_relative_pose, image_pairs.size());
    }


    ///Check if RTK ready and good for each camera
    std::set<image_t> updated_intrinsic_ids;
    std::set<image_t> good_rtk_ids;
    const size_t num_cameras = scene_graph_container_->NumCameras();
    size_t process = 0;
    bool camera_rig = false;
    for (auto &camera: scene_graph_container_->Cameras()) {

        PrintHeading2(StringPrintf("Process Camera#%d/%d", ++process, num_cameras));

        if (camera.second.ModelName().compare("SPHERICAL") == 0) {
            continue;
        }
        if (camera.second.NumLocalCameras() > 2) {
            camera_rig = true;
            // continue;
        }

        std::set<image_t> seleted_image_ids;
        for (auto &image_id : scene_graph_container_->GetImageIds()) {
            if (scene_graph_container_->Image(image_id).CameraId() == camera.first) {
                seleted_image_ids.insert(image_id);
            }
        }

        std::shared_ptr<SceneGraphContainer> cluster_scene_graph_container =
            std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
        scene_graph_container_->ClusterSceneGraphContainer(seleted_image_ids, *cluster_scene_graph_container.get());

        if (cluster_scene_graph_container->CorrespondenceGraph()->NumImagePairs() == 0) {
            continue;
        }

        bool rtk_ready;
        if(mapper_options_.direct_mapper_type == 3){ //force no rtk mode
            rtk_ready = false;
        }else{
            rtk_ready = RTKReady(cluster_scene_graph_container, 0.4);
        }

        bool rtk_good;
        if (camera.second.NumLocalCameras() > 2) {
            rtk_good = LocalGSFM(options_, mapper_options_, workspace_path_, cluster_scene_graph_container, rtk_ready, false);
        } else {
            rtk_good = LocalGSFM(options_, mapper_options_, workspace_path_, cluster_scene_graph_container, rtk_ready, use_block_ba);
        }
        
        if(rtk_ready && !rtk_good){
            if(mapper_options_.direct_mapper_type == 2 || mapper_options_.direct_mapper_type == 4) {
                std::cout << "error: rtk not good enough for rtk sfm" << std::endl;
                success_ = false;
                return;
            } else if (!use_block_ba && camera.second.NumLocalCameras() <= 2) {
                LocalGSFM(options_, mapper_options_, workspace_path_, cluster_scene_graph_container, false);
            }
        }

        if(!rtk_good || use_block_ba) {
            scene_graph_container_->Camera(camera.first).Params() =
                cluster_scene_graph_container->Camera(camera.first).Params();
            updated_intrinsic_ids.insert(seleted_image_ids.begin(), seleted_image_ids.end());
        }
        if (rtk_good) {
            good_rtk_ids.insert(seleted_image_ids.begin(), seleted_image_ids.end());
        }
    }

    std::cout << "good_rtk_ids: " << good_rtk_ids.size() << std::endl;

    auto ed1 = std::chrono::steady_clock::now();
    std::cout << "Local gsfm in  " << std::chrono::duration<float>(ed1 - st1).count() << " sec." << std::endl;

    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }

    auto image_pairs = scene_graph_container_->CorrespondenceGraph()->ImagePairs();
    std::vector<std::pair<image_t, image_t>> pairs;
    for(auto pair : image_pairs){
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(pair.first, &image_id1, &image_id2);
        if(updated_intrinsic_ids.count(image_id1) != 0 || updated_intrinsic_ids.count(image_id2) != 0) {
            pairs.emplace_back(image_id1, image_id2);
        }
    }

    ///Refine two view geometries (If RTK not good) -------------------------------------------------
    st1 = std::chrono::steady_clock::now();
    if(!pairs.empty()){
        int bad_two_view = 0;
        std::vector<std::pair<image_t, image_t>> bad_pairs;

        std::shared_ptr<std::unordered_set<image_pair_t> > tree_image_pair_ids = scene_graph_container_->GetTreeEdges();

        TwoViewGeometry::Options two_view_geometry_options;
        // two_view_geometry_options.max_error = 8.0;
        // two_view_geometry_options.max_angle_error = 0.8;

        two_view_geometry_options.ransac_options.confidence = 0.9999;
        two_view_geometry_options.ransac_options.min_num_trials = 100;
        two_view_geometry_options.ransac_options.max_num_trials = 1000;
        two_view_geometry_options.ransac_options.max_error = 8.0;
        // two_view_geometry_options.ransac_options.min_inlier_ratio_to_best_model = 0.8;
        two_view_geometry_options.min_num_inliers_relative_pose = 15;

        two_view_geometry_options.loose_constraint = true;
        two_view_geometry_options.is_sphere = !mapper_options_.lidar_sfm;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < pairs.size(); ++i) {
            auto image_id1 = pairs[i].first;
            auto image_id2 = pairs[i].second;
            auto pair_id = utility::ImagePairToPairId(image_id1, image_id2);

            const Image &image1 = scene_graph_container_->Image(image_id1);
            const Camera &camera1 = scene_graph_container_->Camera(image1.CameraId());
            const int num_local_camera1 = camera1.NumLocalCameras();

            const Image &image2 = scene_graph_container_->Image(image_id2);
            const Camera &camera2 = scene_graph_container_->Camera(image2.CameraId());
            const int num_local_camera2 = camera2.NumLocalCameras();

            // Recompute for rig cameras which num_local_camera > 2 degenerate the two view geometry.
            if (num_local_camera1 > 2 || num_local_camera2 > 2) {
                continue;
            }

            const std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();
            const FeatureMatches matches = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);

            std::vector<Eigen::Vector2d> points1;
            points1.reserve(image1.NumPoints2D());
            for (const auto &point: image1.Points2D()) {
                points1.push_back(point.XY());
            }

            std::vector<Eigen::Vector2d> points2;
            points2.reserve(image2.NumPoints2D());
            for (const auto &point: image2.Points2D()) {
                points2.push_back(point.XY());
            }

            TwoViewGeometry two_view_geometry;
            two_view_geometry.inlier_matches = matches;
            
            const auto & local_image_indices1 = image1.LocalImageIndices();
            const auto & local_image_indices2 = image2.LocalImageIndices();

            bool success; 
            if (num_local_camera1 <= 1 && num_local_camera2 <= 1) {
                two_view_geometry.Estimate(camera1, points1, camera2, points2, matches, two_view_geometry_options);
                success = two_view_geometry.EstimateRelativePose(camera1, points1, camera2, points2, two_view_geometry_options);
            } else if (num_local_camera1 > 1 && num_local_camera2 > 1) {
                two_view_geometry.EstimateRig(camera1, points1, camera2, points2, matches, local_image_indices1, local_image_indices2,
                                              two_view_geometry_options);
                std::vector<double> tri_angles;
                success = two_view_geometry.EstimateRelativePoseRigGV(camera1, points1, camera2, points2, local_image_indices1, 
                                                                      local_image_indices2, two_view_geometry_options, &tri_angles, true);
            } else {
                two_view_geometry.EstimateOneAndRig(camera1, points1, camera2, points2, matches, local_image_indices1, 
                                                    local_image_indices2, two_view_geometry_options);
                std::vector<double> tri_angles;
                success = two_view_geometry.EstimateRelativePoseOneAndRig(camera1, points1, camera2, points2, local_image_indices1,
                                                                          local_image_indices2, &tri_angles, true);
            }
            if (success) {
                scene_graph_container_->CorrespondenceGraph()->UpdateCorrespondence(image_id1, image_id2, two_view_geometry);
            } else if (tree_image_pair_ids->find(pair_id) == tree_image_pair_ids->end()) {
#pragma omp critical
                {
                    bad_pairs.emplace_back(image_id1, image_id2);
                    bad_two_view++;
                }
            }
        }
        std::cout << "bad_two_view : " << bad_two_view << std::endl;
        int bad_two_view_from_backbone = 0;
        for(auto& bad_pair : bad_pairs) {
            if (minimum_spanning_tree->find(bad_pair.first) != minimum_spanning_tree->end()) {
                bad_two_view_from_backbone++;
            } else {
                scene_graph_container_->CorrespondenceGraph()->DeleteCorrespondences(bad_pair.first, bad_pair.second);
            }
        }
        std::cout << "bad_two_view_from_backbone: " << bad_two_view_from_backbone << std::endl;
        ed1 = std::chrono::steady_clock::now();
        std::cout << "Recompute 2VeiwGeo in  " << std::chrono::duration<float>(ed1 - st1).count() << " sec." << std::endl;
    }

    ///Graph edges filtering -----------------------------------------------------------------
    st1 = std::chrono::steady_clock::now();

    std::shared_ptr<SceneGraphContainer> compressed_graph_container = std::make_shared<SceneGraphContainer>();
    if (!camera_rig) {
        float compress_ratio = 1.0f;
        int connectivity = 0;
        scene_graph_container_->CorrespondenceGraph()->CalculateImageNeighbors();
        const std::vector<image_t>& scene_image_ids = scene_graph_container_->GetImageIds();
        for (const image_t image_id : scene_image_ids) {
            const auto& image_neighbors = scene_graph_container_->CorrespondenceGraph()->ImageNeighbor(image_id);
            connectivity += image_neighbors.size();
        }
        connectivity = connectivity * 1.0f / scene_image_ids.size();
        std::cout << "scene graph connectivity: " << connectivity << std::endl;
        if (connectivity < 10) {
            compress_ratio = 1.0f;
        } else {
            compress_ratio = std::max(10.0f / connectivity, 0.8f);
        }
        // compress_ratio += bad_two_view * 1.0f / image_pairs.size();
        scene_graph_container_->CompressGraph(*compressed_graph_container.get(), compress_ratio);
    } else {
        // compressed_graph_container = scene_graph_container_;
        scene_graph_container_->Copy(*compressed_graph_container.get());

        double filter_ratio = 0.25;
        const size_t min_num_image_pair = image_pairs.size() * 0.6;
    	auto view_graph = *compressed_graph_container->CorrespondenceGraph();
        image_pairs = view_graph.ImagePairs();
        auto good_pairs = image_pairs;
        for (int it = 0; it < 3; ++it) {
            good_pairs = ViewGraphFiltering(view_graph, image_pairs, filter_ratio);
            std::cout << "good pairs num : " << good_pairs.size() << "/" << image_pairs.size() << std::endl;
            if (good_pairs.size() > min_num_image_pair) break;
            filter_ratio *= 0.6;
        }
        for (auto &pair : image_pairs) {
            if (good_pairs.count(pair.first) != 0) {
                continue;
            }
            image_t image_id1, image_id2;
            sensemap::utility::PairIdToImagePair(pair.first, &image_id1, &image_id2);
            compressed_graph_container->CorrespondenceGraph()->DeleteCorrespondences(image_id1, image_id2);
        }

        auto image_ids = compressed_graph_container->GetImageIds();
        for (const auto image_id : image_ids) {
            if (compressed_graph_container->Image(image_id).NumCorrespondences() == 0) {
                compressed_graph_container->DeleteImage(image_id);
            }
        }
    }

    std::cout << StringPrintf("scene_graph: %d %d\n", scene_graph_container_->NumImages(), scene_graph_container_->CorrespondenceGraph()->NumImagePairs());
    std::cout << StringPrintf("compressd scene_graph: %d %d\n", compressed_graph_container->NumImages(), compressed_graph_container->CorrespondenceGraph()->NumImagePairs());

    ed1 = std::chrono::steady_clock::now();
    std::cout << "CompressGraph in  " << std::chrono::duration<float>(ed1 - st1).count() << " sec." << std::endl;
    st1 = std::chrono::steady_clock::now();

    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container_);
    mapper->SetWorkspacePath(workspace_path_);

    size_t reconstruction_idx = reconstruction_manager_->Add();
    auto reconstruction = reconstruction_manager_->Get(reconstruction_idx);
    mapper->BeginReconstruction(reconstruction);

    for (auto pointid : reconstruction->MapPointIds()) {
        reconstruction->DeleteMapPoint(pointid);
    }

    const auto & image_ids = compressed_graph_container->GetImageIds();
    for (auto image_id : image_ids) {
        if (good_rtk_ids.find(image_id) != good_rtk_ids.end()) {
            auto & image = compressed_graph_container->Image(image_id);
            if (image.RtkFlag() == 50) {
                image.SetPriorQvecGood(true);
            }
        }
    }

    ///Global GSFM --------------------------------------------------------------------------------
    std::unordered_set<image_t> gsfm_poses;
    if (mapper_options_.direct_mapper_type != 4) {
        auto global_options = options_->GlobalMapperOptions();
        // global_options.use_rotation_prior_constrain = camera_rig ? false : mapper_options_.has_gps_prior;
        // global_options.use_translation_prior_constrain = camera_rig ? false : mapper_options_.has_gps_prior;
        global_options.use_rotation_prior_constrain = mapper_options_.has_gps_prior;
        global_options.use_translation_prior_constrain = mapper_options_.has_gps_prior;
        if(good_rtk_ids.size() == 0) {
            global_options.use_rotation_prior_constrain = false;
            global_options.use_translation_prior_constrain = false;
        }

        global_options.rotation_prior_constrain_weight = 100;
        global_options.translation_prior_constrain_weight = 100;
        global_options.translation_prior_weak_constrain_weight = 1;
        global_options.two_steo_refinement_of_position = mapper_options_.has_gps_prior;

        std::shared_ptr<GlobalMapper> global_mapper =
                std::make_shared<GlobalMapper>(compressed_graph_container);

        auto global_reconstruction = std::make_shared<Reconstruction>();
        global_mapper->BeginReconstruction(global_reconstruction);

        //Estimate Global Rotations
        if (!global_mapper->EstimateGlobalRotations(global_options)) {
            LOG(ERROR) << "Could not estimate camera rotations for Global Directed SfM.";
            success_ = false;
            return;
        }

        // Step 5. Optimize relative translations.
        LOG(INFO) << "Optimizing the pairwise translation estimations.";
        global_mapper->OptimizePairwiseTranslations(global_options);

        // Step 6. Filter bad relative translations.
        LOG(INFO) << "Filtering any bad relative translations.";
        global_mapper->FilterRelativeTranslation(global_options);

        LOG(INFO) << "Estimating the positions of all cameras.";
        if (!global_mapper->EstimatePosition(global_options)) {
            LOG(WARNING) << "Position estimation failed!";
            return ;
        }

        Eigen::Matrix3x4d matrix_to_align =
            global_reconstruction->AlignWithPriorLocations(mapper_options_.max_error_gps);

        for (const auto image_id : image_ids) {
            auto qvec = global_reconstruction->Image(image_id).Qvec();
            auto t = global_reconstruction->Image(image_id).Tvec();
            const SimilarityTransform3 tform(matrix_to_align);
            tform.TransformPose(&qvec, &t);
            reconstruction->Image(image_id).SetQvec(qvec);
            reconstruction->Image(image_id).SetTvec(t);
            gsfm_poses.insert(image_id);
        }

        if (options_->debug_info) {
            std::string gsfm_path = JoinPaths(workspace_path_, "/gsfm-0");
            if (!boost::filesystem::exists(gsfm_path)) {
                boost::filesystem::create_directories(gsfm_path);
            }

            std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
            global_reconstruction->FilterUselessPoint2D(filtered_reconstruction);

            filtered_reconstruction->WriteBinary(gsfm_path);

            auto registered_image_ids = global_reconstruction->RegisterImageIds();
            if (registered_image_ids.size() > 0) {
                auto image = global_reconstruction->Image(registered_image_ids[0]);
                auto camera = global_reconstruction->Camera(image.CameraId());
                if (camera.NumLocalCameras() > 1) {
                    Reconstruction rig_reconstruction;
                    global_reconstruction->ConvertRigReconstruction(rig_reconstruction);
	                
                    std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                    rig_reconstruction.FilterUselessPoint2D(filtered_reconstruction);

                    std::string rig_path = JoinPaths(workspace_path_, "/gsfm-0-export");
                    if (!boost::filesystem::exists(rig_path)) {
                        boost::filesystem::create_directories(rig_path);
                    }
                    filtered_reconstruction->WriteBinary(rig_path);

                    filtered_reconstruction->AddPriorToResult();
                    CreateDirIfNotExists(JoinPaths(workspace_path_, "/gsfm-0-gps-export"));
                    filtered_reconstruction->WriteBinary(JoinPaths(workspace_path_, "/gsfm-0-gps-export"));
                }
            }
        }

        global_mapper->EndReconstruction(true);
    }

    ed1 = std::chrono::steady_clock::now();
    std::cout << "Global gsfm in  " << std::chrono::duration<float>(ed1 - st1).count() << " sec." << std::endl;

    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }


    //////////////////////////////////////////////////////////////////////////////
    // Triangulation
    //////////////////////////////////////////////////////////////////////////////



//    std::shared_ptr<IncrementalMapper> incrematal_mapper = std::make_shared<IncrementalMapper>(scene_graph_container_);
//    incrematal_mapper->BeginReconstruction(reconstruction);
    int triangulated_image_count = 1;
    for (const auto image_id : orig_image_ids) {

        if(!scene_graph_container_->CorrespondenceGraph()->ExistsImage(image_id)){
            continue ;
        }

        Image &image_scene = scene_graph_container_->Image(image_id);

        if (reconstruction->ExistsImage(image_scene.ImageId()) &&
            reconstruction->IsImageRegistered(image_scene.ImageId())) {
            reconstruction->DeRegisterImage(image_scene.ImageId());
        }

        if (!reconstruction->ExistsImage(image_id)) {
            reconstruction->AddImage(image_scene);
        }

        Image &image = reconstruction->Image(image_id);

        Eigen::Vector4d prior_qvec;
        Eigen::Vector3d prior_tvec;

        if (gsfm_poses.find(image_id) != gsfm_poses.end()) {
            prior_qvec = image.Qvec();
            prior_tvec = image.Tvec();
        } else {
            continue;
        }

        image.SetQvec(prior_qvec);
        image.SetTvec(prior_tvec);
        reconstruction->RegisterImage(image_id);

        PrintHeading1(StringPrintf("Triangulating image #%d - %s (%d / %d)",
                                   image_id, image.Name().c_str(), triangulated_image_count++,
                                   orig_image_ids.size()));
        const size_t num_existing_points3D = image.NumMapPoints();
        std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations()
                  << " points"
                  << std::endl;

        mapper->TriangulateImage(options_->Triangulation(), image_id);

        std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points"
                  << std::endl;
    }

    for (const auto image_id : orig_image_ids) {
        if(!scene_graph_container_->CorrespondenceGraph()->ExistsImage(image_id)){
            continue ;
        }
        if (gsfm_poses.find(image_id) != gsfm_poses.end()) {
            continue;
        }
        
        Image &image = reconstruction->Image(image_id);
        Camera &camera = reconstruction->Camera(image.CameraId());

        std::vector<std::pair<point2D_t, mappoint_t>> tri_corrs;
        std::vector<char> inlier_mask;
        bool reg_next_success = false;
        if(camera.NumLocalCameras()>1){
            reg_next_success = mapper->EstimateCameraPoseRig(
                                    options_->IncrementalMapperOptions(),
                                    image_id,
                                    tri_corrs,
                                    inlier_mask);
        }else{
            reg_next_success = mapper->EstimateCameraPose(
                                options_->IncrementalMapperOptions(),
                                image_id,
                                tri_corrs,
                                inlier_mask);
        }
        if (!reg_next_success && !(image.HasQvecPrior() && image.HasTvecPrior())) {
            continue;
        }
        if (!reg_next_success) {
            Eigen::Vector4d prior_qvec = image.QvecPrior();
            Eigen::Vector3d prior_tvec = QuaternionToRotationMatrix(prior_qvec) * -image.TvecPrior();
            image.SetQvec(prior_qvec);
            image.SetTvec(prior_tvec);
        }

        reconstruction->RegisterImage(image_id);

        PrintHeading1(StringPrintf("Triangulating image with incremental registration #%d - %s (%d / %d)",
                                   image_id, image.Name().c_str(), triangulated_image_count++,
                                   orig_image_ids.size()));
        const size_t num_existing_points3D = image.NumMapPoints();
        std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations()
                  << " points"
                  << std::endl;

        mapper->TriangulateImage(options_->Triangulation(), image_id);

        std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points"
                  << std::endl;
    }

    CompleteAndMergeTracks(*options_.get(), mapper);
    if (!camera_rig) {
        FilterPoints(*options_.get(), mapper, 2);
    }

    std::vector<image_t> reg_image_ids = reconstruction->RegisterImageIds();
    //////////////////////////////////////////////////////////////////////////////
    // Bundle adjustment
    //////////////////////////////////////////////////////////////////////////////

    size_t total_num_image = reconstruction->NumImages() - reconstruction->NumRegisterImages();

    reconstruction->AlignWithPriorLocations(mapper_options_.max_error_gps);

    if (mapper_options_.direct_mapper_type != 3 || total_num_image == 0) {
        auto ba_options = options_->GlobalBundleAdjustment();
        // ba_options.refine_focal_length = true;
        // ba_options.refine_principal_point = true;
        // ba_options.refine_extra_params = true;
        // ba_options.refine_extrinsics = true;
        ba_options.solver_options.minimizer_progress_to_stdout = true;

        if ((options_->single_camera && options_->camera_fixed) ||
            reconstruction->NumRegisterImages() < mapper_options_.num_fix_camera_first) {
            ba_options.refine_focal_length = false;
            ba_options.refine_extra_params = false;
            ba_options.refine_principal_point = false;
            ba_options.refine_local_extrinsics = false;
        }

        reconstruction->b_aligned = options_->has_gps_prior;
        ba_options.use_prior_absolute_location = options_->has_gps_prior;

        ba_options.force_full_ba = !use_block_ba;

        std::cout << "prior_absolute_location_weight: " << ba_options.prior_absolute_location_weight
                  << "\t loss function: " << options_->ba_global_loss_function << std::endl;

        // PrintHeading1("CompleteAndMergeTracks");
        // CompleteAndMergeTracks(*options_.get(), mapper);
        // PrintHeading1("Retriangulation");
        // std::cout << "  => Retriangulated observations: " << mapper->Retriangulate(options_->Triangulation()) << std::endl;

        // Configure bundle adjustment.
        BundleAdjustmentConfig ba_config;

        for (size_t i = 0; i < reg_image_ids.size(); ++i) {
            const image_t image_id = reg_image_ids[i];
            if (!scene_graph_container_->CorrespondenceGraph()->ExistsImage(image_id)) {
                continue;
            }
            //ba_config.AddGNSS(image_id);
            ba_config.AddImage(image_id);
        }


        {
            float mem;
            sensemap::GetAvailableMemory(mem);
            std::cout << "memory left : " << mem << std::endl;
            std::cout << malloc_trim(0) << std::endl;
            sensemap::GetAvailableMemory(mem);
            std::cout << "memory left(g) : " << mem << std::endl;
        }

        for (int i = 0; i < options_->ba_global_max_refinements; ++i) {
            reconstruction->FilterObservationsWithNegativeDepth();

            const size_t num_observations = reconstruction->ComputeNumObservations();

            PrintHeading1("RTK Bundle adjustment");
            std::cout << "iter: " << i << std::endl;

            if (!ba_options.force_full_ba) {
                BlockBundleAdjuster bundle_adjuster(ba_options, ba_config);
                bundle_adjuster.Solve(reconstruction.get());
            } else {
                BundleAdjuster bundle_adjuster(ba_options, ba_config);
                CHECK(bundle_adjuster.Solve(reconstruction.get()));
            }

            size_t num_changed_observations = 0;
            num_changed_observations += CompleteAndMergeTracks(*options_.get(), mapper);
            num_changed_observations += FilterPoints(*options_.get(), mapper, mapper_options_.min_track_length);
            const double changed = static_cast<double>(num_changed_observations) / num_observations;
            std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;

            size_t num_retriangulate_observations = 0;
            num_retriangulate_observations = mapper->Retriangulate(options_->Triangulation());
            std::cout << "\nnum_retri_observations / num_ori_observations: "
                      << num_observations << " / "
                      << num_retriangulate_observations << std::endl;

            for (const auto &camera : reconstruction->Cameras()) {
                if (camera.second.NumLocalCameras() > 1) {
                    std::cout << "Camera#" << camera.first << ", param: " << VectorToCSV(camera.second.LocalParams()) << std::endl;
                } else {
                    std::cout << "Camera#" << camera.first << ", param: " << camera.second.ParamsToString() << std::endl;
                }
            }

            if (changed < options_->ba_global_max_refinement_change) {
                break;
            }
        }
    }

    reconstruction->AlignWithPriorLocations(mapper_options_.max_error_gps);
    reconstruction->FilterAllFarawayImages();
    reconstruction->TearDown();

    std::cout << "Mean Track Length: " << reconstruction->ComputeMeanTrackLength() << std::endl;
    std::cout << "Mean Reprojection Error: " << reconstruction->ComputeMeanReprojectionError() << std::endl;
    std::cout << "Mean Observation Per Register Image: " << reconstruction->ComputeMeanObservationsPerRegImage() << std::endl;

    auto reconstructed_image_ids = reconstruction->RegisterImageIds();

    if (options_->refine_separate_cameras) {
        std::cout << "BundleAdjust Optimization of separate cameras" << std::endl;

        const auto & image_ids = reconstruction->RegisterImageSortIds();
        int num_local_camera = 1;
        for (auto image_id : image_ids) {
            const class Image & image = reconstruction->Image(image_id);
            const class Camera & camera = reconstruction->Camera(image.CameraId());
            num_local_camera = std::max(camera.NumLocalCameras(), num_local_camera);
        }
        if (num_local_camera > 1) {
            std::shared_ptr<SceneGraphContainer> scene_graph = std::make_shared<SceneGraphContainer>();
            scene_graph_container_->ConvertRigSceneGraphContainer(*scene_graph.get(), reconstruction->RegisterImageSortIds());

            std::cout << "Rig SceneGraph: " << scene_graph->NumImages() << std::endl;

            // scene_graph->CorrespondenceGraph()->ExportToGraph(workspace_path_ + "/scene_graph_rig.png");

            std::shared_ptr<IncrementalMapper> rig_mapper = std::make_shared<IncrementalMapper>(scene_graph);
            rig_mapper->SetWorkspacePath(workspace_path_);

            reconstruction_manager_->Delete(reconstruction_idx);
            size_t rec_idx = reconstruction_manager_->Add();
            std::shared_ptr<Reconstruction> rig_reconstruction = reconstruction_manager_->Get(rec_idx);

            reconstruction->ConvertRigReconstruction(*rig_reconstruction.get());
            rig_reconstruction->has_gps_prior = options_->has_gps_prior;
                
            std::cout << "Register Images: " << rig_reconstruction->NumRegisterImages() << std::endl;

            rig_mapper->BeginReconstruction(rig_reconstruction);

            IndependentMapperOptions options = *options_.get();
            options.complete_max_reproj_error = 12.0;
            options.merge_max_reproj_error = 12;

            for (int iter = 0; iter < 2; ++iter) {

                PrintHeading1("Retriangulation");
                CompleteAndMergeTracks(options, rig_mapper);
                std::cout << "  => Retriangulated observations: " << rig_mapper->Retriangulate(options.Triangulation())
                        << std::endl;

                PrintHeading1("BundleAdjust Optimization of Separate Cameras");
                std::cout << "iter: " << iter << std::endl;

                BundleAdjustmentOptions ba_options = options.GlobalBundleAdjustment();
                if(options.use_prior_align_only){
                    ba_options.use_prior_absolute_location = false;
                }
                // ba_options.prior_absolute_location_weight = 0.05;

                ba_options.solver_options.minimizer_progress_to_stdout = true;
                ba_options.solver_options.max_num_iterations = 20;
                ba_options.solver_options.max_linear_solver_iterations = 100;

                // if (rig_reconstruction->NumRegisterImages() < mapper_options_.num_fix_camera_first) {
                ba_options.refine_focal_length = false;
                ba_options.refine_principal_point = false;
                ba_options.refine_extra_params = false;
                ba_options.refine_local_extrinsics = false;
                // }

                std::cout << "Register Images: " << rig_reconstruction->NumRegisterImages() << std::endl;

                rig_mapper->AdjustGlobalBundle(mapper_options_, ba_options);

                if (mapper_options_.has_gps_prior && !mapper_options_.map_update) {
                    CHECK(rig_reconstruction->has_gps_prior);
                    rig_reconstruction->AlignWithPriorLocations(mapper_options_.max_error_gps,
                                                                mapper_options_.max_error_horizontal_gps,
                                                                mapper_options_.max_gps_time_offset);
                }
            }

            FilterPointsFinal(options, rig_mapper);
            FilterImages(options, rig_mapper);
            rig_reconstruction->FilterAllFarawayImages();

            std::cout << "Mean Track Length: " << rig_reconstruction->ComputeMeanTrackLength() << std::endl;
            std::cout << "Mean Reprojection Error: " << rig_reconstruction->ComputeMeanReprojectionError() << std::endl;
            std::cout << "Mean Observation Per Register Image: " << rig_reconstruction->ComputeMeanObservationsPerRegImage() << std::endl;

            // std::string rig_path = JoinPaths(workspace_path_, "0-rig");
            // CreateDirIfNotExists(rig_path);
            // rig_reconstruction->WriteBinary(rig_path);
        }
    }

    {
        for (auto & image_id : reconstructed_image_ids) {
            scene_graph_container_->Image(image_id).SetRegistered(true);
        }
    }

    ed1 = std::chrono::steady_clock::now();
    std::cout << "Full gsfm in  " << std::chrono::duration<float>(ed1 - st1).count() << " sec." << std::endl;

    if(0){
        std::cout<<"sunhan :  final "<<std::endl;

        std::string global_path = StringPrintf("%s/%d-base", workspace_path_.c_str(), reconstruction_idx);
        auto global_reconstruction_ba = std::make_shared<Reconstruction>();
        global_reconstruction_ba->ReadReconstruction(global_path);
        auto global_reconstruction_noba = std::make_shared<Reconstruction>();
        global_path = StringPrintf("%s/%d-base", workspace_path_.c_str(), reconstruction_idx);
        global_reconstruction_noba->ReadReconstruction(global_path);

        double mean_dist0 = 0, mean_dist1 =0;
        double mean_Rangle1 = 0, mean_tangle1 = 0, mean_Rangle2 = 0, mean_tangle2 = 0;
        for (const auto image_id : image_ids) {

            if(!global_reconstruction_ba->ExistsImage(image_id) ||
                !global_reconstruction_noba->ExistsImage(image_id) ||
                !reconstruction->ExistsImage(image_id)){
                continue;
            }

            auto qvec0 = reconstruction->Image(image_id).Qvec();
            auto tvec0 = reconstruction->Image(image_id).Tvec();
            auto qvec2 = global_reconstruction_noba->Image(image_id).Qvec();
            auto tvec2 = global_reconstruction_noba->Image(image_id).Tvec();
            auto qvec3 = global_reconstruction_ba->Image(image_id).Qvec();
            auto tvec3 = global_reconstruction_ba->Image(image_id).Tvec();


            {
                auto R1 = QuaternionToRotationMatrix(qvec0);
                auto R2 = QuaternionToRotationMatrix(qvec3);

                Eigen::Matrix3d R_diff = (R1.transpose()) * R2;
                Eigen::AngleAxisd angle_axis(R_diff);
                double R_angle = angle_axis.angle();

                Eigen::Vector3d C1 = -(R1.transpose() * tvec3);
                Eigen::Vector3d C2 = -(R2.transpose() * tvec3);

                double cos_t_angle = C1.dot(C2) / (C1.norm() * C2.norm());
                cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));

                double t_angle = std::acos(cos_t_angle);

                std::cout << (RadToDeg(R_angle) + RadToDeg(t_angle))/2.0<< " " ;// << std::endl;

                mean_Rangle1 += RadToDeg(R_angle);
                mean_tangle1 += RadToDeg(t_angle);
            }

            {
                auto R1 = QuaternionToRotationMatrix(qvec2);
                auto R2 = QuaternionToRotationMatrix(qvec3);

                Eigen::Matrix3d R_diff = (R1.transpose()) * R2;
                Eigen::AngleAxisd angle_axis(R_diff);
                double R_angle = angle_axis.angle();

                Eigen::Vector3d C1 = -(R1.transpose() * tvec3);
                Eigen::Vector3d C2 = -(R2.transpose() * tvec3);

                double cos_t_angle = C1.dot(C2) / (C1.norm() * C2.norm());
                cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));

                double t_angle = std::acos(cos_t_angle);

                //                std::cout << RadToDeg(R_angle) << " " << RadToDeg(t_angle) << std::endl;

                mean_Rangle2 += RadToDeg(R_angle);
                mean_tangle2 += RadToDeg(t_angle);
            }

            mean_dist0 += (tvec0 - tvec3).squaredNorm() ;
            mean_dist1 += (tvec2 - tvec3).squaredNorm() ;
            std::cout << (tvec0 - tvec3).squaredNorm() << std::endl;
            // std::cout <<(RadToDeg(R_angle) + RadToDeg(t_angle)) / 2.0 << std::endl;
        }

        mean_Rangle1 /= image_ids.size();
        mean_tangle1 /= image_ids.size();
        mean_Rangle2 /= image_ids.size();
        mean_tangle2 /= image_ids.size();
        mean_dist0 /= image_ids.size();
        mean_dist1 /= image_ids.size();

        std::cout<<(mean_Rangle2+mean_tangle2)/2.0<<"  "<<mean_dist1<<std::endl;
        std::cout<<(mean_Rangle1+mean_tangle1)/2.0<<"  "<<mean_dist0<<std::endl;
    }

    success_ = true;
    std::cout << StringPrintf("sfm :Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;
}

void DirectedMapperController::Run() {
    mapper_options_ = options_->IncrementalMapperOptions();
    mapper_options_.image_path = image_path_;
    Reconstruct();
    GetTimer().PrintMinutes();
}
} // namespace sensemap
