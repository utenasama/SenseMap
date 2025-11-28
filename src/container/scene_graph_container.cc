// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "container/scene_graph_container.h"

#include "base/pose.h"
#include "container/io.h"
#include "graph/maximum_spanning_tree_graph.h"
#include "util/logging.h"

namespace sensemap {

SceneGraphContainer::SceneGraphContainer() : correspondence_graph_(nullptr) {
    correspondence_graph_ = std::shared_ptr<class CorrespondenceGraph>(new sensemap::CorrespondenceGraph());
    // correspondence_graph_ = 
    //     std::allocate_shared<class CorrespondenceGraph>(Eigen::aligned_allocator<class CorrespondenceGraph>());
}

void SceneGraphContainer::Copy(SceneGraphContainer & scene_graph_contrainer) {
    scene_graph_contrainer.Cameras() = cameras_;
    scene_graph_contrainer.Images() = images_;
    CorrespondenceGraph()->Copy(*scene_graph_contrainer.CorrespondenceGraph().get());
}

void SceneGraphContainer::AddCamera(const class Camera &camera) {
    CHECK(!ExistsCamera(camera.CameraId()));
    cameras_.emplace(camera.CameraId(), camera);
}

void SceneGraphContainer::AddImage(const class Image &image) {
    CHECK(!ExistsImage(image.ImageId()));
    images_.emplace(image.ImageId(), image);
    correspondence_graph_->AddImage(image.ImageId(), image.NumPoints2D());
}

void SceneGraphContainer::ClusterSceneGraphContainer(const std::set<image_t> &image_ids,
                                                     SceneGraphContainer &scene_graph_container) {
    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph_container.Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph_container.Cameras();

    for (const auto &image_id : image_ids) {
        if (!ExistsImage(image_id)) {
            continue;
        }
        const class Image &image = Image(image_id);
        CHECK(ExistsCamera(image.CameraId()));
        const class Camera &camera = Camera(image.CameraId());

        scene_graph_container.AddImage(image);
        if (!scene_graph_container.ExistsCamera(image.CameraId())) {
            scene_graph_container.AddCamera(camera);
        }
    }

    std::shared_ptr<class CorrespondenceGraph> correspondence_graph = scene_graph_container.CorrespondenceGraph();
    std::shared_ptr<class CorrespondenceGraph> correspondence_graph_ = CorrespondenceGraph();

    const auto &image_pairs = correspondence_graph_->ImagePairs();
    for (auto image_pair : image_pairs) {
        image_t image_id1, image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        if (image_ids.count(image_id1) && image_ids.count(image_id2)) {
            const FeatureMatches &feature_matches =
                correspondence_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);
            image_pair.second.two_view_geometry.inlier_matches = feature_matches;

            const class Image &image1 = Image(image_id1);
            CHECK(ExistsCamera(image1.CameraId()));
            const class Camera &camera1 = Camera(image1.CameraId());
            const class Image &image2 = Image(image_id2);
            CHECK(ExistsCamera(image2.CameraId()));
            const class Camera &camera2 = Camera(image2.CameraId());

            const bool remove_redundant = !(camera1.NumLocalCameras() > 2 && camera2.NumLocalCameras());

            correspondence_graph->AddCorrespondences(image_id1, image_id2, 
                image_pair.second.two_view_geometry, remove_redundant);
        }
    }
    correspondence_graph->Finalize();

    for (const auto &image_id : image_ids) {
        if (images.find(image_id) == images.end() || !correspondence_graph->ExistsImage(image_id)) {
            continue;
        }
        images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
        images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
    }

    for (auto image_id : image_ids) {
        if (!correspondence_graph->ExistsImage(image_id)) {
            scene_graph_container.DeleteImage(image_id);
        }
    }
}

void SceneGraphContainer::ConvertRigSceneGraphContainer(SceneGraphContainer &scene_graph_container, 
                                                        const std::vector<image_t>& image_ids) {
    // Convert Camera.

    std::cout << "Convert SceneGraph" << std::endl;

    std::unordered_map<camera_t, std::vector<camera_t> > camera_ids_map;
    std::unordered_map<camera_t, class Camera> new_cameras;
    size_t new_camera_id = 1;
    for (const auto& camera : cameras_) {
        int num_local_cameras = camera.second.NumLocalCameras();
        for (local_camera_t camera_id = 0; camera_id < num_local_cameras; ++camera_id){
            std::string model_name = camera.second.ModelName();
            int width = camera.second.Width();
            int height = camera.second.Height();

            std::vector<double> params;
            if (num_local_cameras > 1) {
                camera.second.GetLocalCameraIntrisic(camera_id, params);
            } else {
                params = camera.second.Params();
            }

            class Camera local_camera;
            local_camera.SetCameraId(new_camera_id);
            local_camera.SetModelIdFromName(model_name);
            local_camera.SetWidth(width);
            local_camera.SetHeight(height);
            local_camera.SetNumLocalCameras(1);
            local_camera.SetParams(params);

            new_cameras[new_camera_id] = local_camera;

            camera_ids_map[camera.first].emplace_back(new_camera_id);
            new_camera_id++;
        }
    }

    std::vector<image_t> image_sort_ids = image_ids;
    if (image_sort_ids.empty()) {
        image_sort_ids = GetImageIds();
    }
    std::sort(image_sort_ids.begin(), image_sort_ids.end());

    const local_camera_t max_num_local_camera = 6;

    // Convert Image.
    std::unordered_map<image_t, image_t> image_id_map;
    std::unordered_map<point2D_t, point2D_t> point2D_id_map;
    std::unordered_map<image_t, class Image> new_images;
    image_t image_rig_id = 1;
    for (const auto image_id : image_sort_ids) {
        const auto& image = images_.at(image_id);
        const class Camera& camera = cameras_.at(image.CameraId());
        const size_t num_local_camera = camera.NumLocalCameras();

        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        point2D_t point2D_id_offset = 0;

        for (local_camera_t local_id = 0; local_id < num_local_camera; local_id++){
            Eigen::Vector4d normalized_qvec;
            Eigen::Vector3d normalized_tvec;
            if (num_local_camera > 1) {
                camera.GetLocalCameraExtrinsic(local_id, local_qvec, local_tvec);

                normalized_qvec =
                NormalizeQuaternion(RotationMatrixToQuaternion(
                QuaternionToRotationMatrix(local_qvec) * QuaternionToRotationMatrix(image.Qvec())));

                normalized_tvec =
                QuaternionToRotationMatrix(local_qvec)*image.Tvec() + local_tvec;
            } else {
                normalized_qvec = image.Qvec();
                normalized_tvec = image.Tvec();
            }

            std::string image_name = image.LocalName(local_id);

            std::vector<Point2D> local_point2Ds;
            point2D_t rig_point2D_idx = 0;
            for(size_t i = 0; i< image.NumPoints2D(); ++i){    
                Point2D point2D = image.Points2D()[i];
                local_camera_t local_camera_id = image.LocalImageIndices()[i];
                if (local_camera_id == local_id){
                    local_point2Ds.emplace_back(point2D);
                    rig_point2D_idx++;
                }
            }

            std::vector<uint32_t> new_image_indices(local_point2Ds.size());
            std::fill(new_image_indices.begin(), new_image_indices.end(), image_rig_id);

            size_t new_camera_id = camera_ids_map.at(camera.CameraId())[local_id];

            class Image new_image;
            new_image.SetCameraId(new_camera_id);
            new_image.SetImageId(image_rig_id);
            new_image.SetName(image_name);
            new_image.SetQvec(normalized_qvec);
            new_image.SetTvec(normalized_tvec);
            new_image.SetPoseFlag(true);
            new_image.SetPoints2D(local_point2Ds);
            new_image.SetLocalImageIndices(new_image_indices);
            new_image.SetLabelId(image.LabelId());
            new_images[image_rig_id] = new_image;

            image_id_map[image_id * max_num_local_camera + local_id] = image_rig_id;
            point2D_id_map[image_rig_id] = point2D_id_offset;
            point2D_id_offset += local_point2Ds.size();

            image_rig_id++;
        }
    }

    for (const auto & new_image : new_images) {
        const class Image &image = new_image.second;
        const class Camera &camera = new_cameras.at(image.CameraId());

        scene_graph_container.AddImage(image);
        if (!scene_graph_container.ExistsCamera(image.CameraId())) {
            scene_graph_container.AddCamera(camera);
        }
    }

    std::shared_ptr<class CorrespondenceGraph> correspondence_graph = scene_graph_container.CorrespondenceGraph();
    std::shared_ptr<class CorrespondenceGraph> correspondence_graph_ = CorrespondenceGraph();

    const auto &image_pairs = correspondence_graph_->ImagePairs();
    for (auto image_pair : image_pairs) {
        image_t image_id1, image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        const FeatureMatches &feature_matches =
            correspondence_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);

        const class Image &image1 = Image(image_id1);
        const class Camera &camera1 = Camera(image1.CameraId());
        const std::vector<uint32_t>& local_image_indices1 = image1.LocalImageIndices();

        const class Image &image2 = Image(image_id2);
        const class Camera &camera2 = Camera(image2.CameraId());
        const std::vector<uint32_t>& local_image_indices2 = image2.LocalImageIndices();
        
        std::unordered_map<image_pair_t, TwoViewGeometry> two_view_geometrys;
        for (const auto match : feature_matches) {
            local_camera_t local_camera_id1 = local_image_indices1.at(match.point2D_idx1);
            local_camera_t local_camera_id2 = local_image_indices2.at(match.point2D_idx2);
            image_t new_image_id1 = image_id_map.at(image_id1 * max_num_local_camera + local_camera_id1);
            image_t new_image_id2 = image_id_map.at(image_id2 * max_num_local_camera + local_camera_id2);

            FeatureMatch new_match;
            new_match.point2D_idx1 = match.point2D_idx1 - point2D_id_map[new_image_id1];
            new_match.point2D_idx2 = match.point2D_idx2 - point2D_id_map[new_image_id2];
            
            image_pair_t pair_id = utility::ImagePairToPairId(new_image_id1, new_image_id2);
            two_view_geometrys[pair_id].inlier_matches.emplace_back(new_match);
        }

        for (auto two_view_geometry : two_view_geometrys) {
            image_t image_id1, image_id2;
            utility::PairIdToImagePair(two_view_geometry.first, &image_id1, &image_id2);
            correspondence_graph->AddCorrespondences(image_id1, image_id2, two_view_geometry.second, false);
        }
    }
    correspondence_graph->Finalize();

    const auto & new_image_ids = scene_graph_container.GetImageIds();
    for (const auto &image_id : new_image_ids) {
        if (!correspondence_graph->ExistsImage(image_id)) {
            continue;
        }
        class Image & image = scene_graph_container.Image(image_id);
        image.SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
        image.SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
    }
}

std::shared_ptr<std::unordered_set<image_pair_t> > SceneGraphContainer::GetTreeEdges(bool force) {
    if (!force && tree_image_pair_ids_) return tree_image_pair_ids_;

    if (!tree_image_pair_ids_) {
        tree_image_pair_ids_ = std::make_shared<std::unordered_set<image_pair_t>>();
    }
    tree_image_pair_ids_->clear();

    MinimumSpanningTree<image_t, float> mst_extractor;

    auto const &image_pairs = CorrespondenceGraph()->ImagePairs();
    for (auto image_pair : image_pairs) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);

#ifdef TWO_VIEW_CONFIDENCE
        mst_extractor.AddEdge(image_id1, image_id2, -image_pair.second.two_view_geometry.confidence);
#else
        mst_extractor.AddEdge(image_id1, image_id2, -image_pair.second.num_correspondences);
#endif
    }
    std::unordered_set<std::pair<image_t, image_t> > minimum_spanning_tree;
    mst_extractor.Extract(&minimum_spanning_tree);

    for (auto image_pair : minimum_spanning_tree) {
        auto pair_id = utility::ImagePairToPairId(image_pair.first, image_pair.second);
        tree_image_pair_ids_->insert(pair_id);
    }
    return tree_image_pair_ids_;
}

void SceneGraphContainer::CompressGraph(SceneGraphContainer &compressed_scene_graph_container,
                                        const float compress_ratio) {
    std::cout << "Graph Edge Compression" << std::endl;

    auto const &image_pairs = CorrespondenceGraph()->ImagePairs();
    const int num_images = CorrespondenceGraph()->NumImages();

    const double conf_thres = 0.1;

    std::unordered_set<image_pair_t> minimum_spanning_tree = *GetTreeEdges(true).get();

    std::vector<std::pair<image_pair_t, float> > sorted_image_pairs;
    sorted_image_pairs.reserve(image_pairs.size());

    size_t bad_image_pair = 0;
    for (auto image_pair : image_pairs) {
        // sorted_image_pairs.emplace_back(image_pair.first, image_pair.second.num_correspondences);
        if (minimum_spanning_tree.find(image_pair.first) != minimum_spanning_tree.end()) {
            continue;
        }
#ifdef TWO_VIEW_CONFIDENCE
        if (image_pair.second.two_view_geometry.confidence < conf_thres) {
            bad_image_pair++;
        } else {
            sorted_image_pairs.emplace_back(image_pair.first, image_pair.second.two_view_geometry.confidence);
        }
#else
        sorted_image_pairs.emplace_back(image_pair.first, image_pair.second.num_correspondences);
#endif
    }

#ifdef TWO_VIEW_CONFIDENCE
    std::cout << StringPrintf("Delete %d edges from scenegraph with confidence < %f\n", bad_image_pair, conf_thres);
#endif

    size_t num_redundance_edges = std::min(sorted_image_pairs.size(), size_t(compress_ratio * (image_pairs.size() - num_images)));
    if (num_redundance_edges > 0) {
        std::nth_element(sorted_image_pairs.begin(), sorted_image_pairs.begin() + num_redundance_edges,
                        sorted_image_pairs.end(),
                        [&](const std::pair<image_pair_t, float>& a, const std::pair<image_pair_t, float>& b) {
                            return a.second > b.second;
                        });
    }

    std::vector<image_pair_t> image_pair_ids;
    image_pair_ids.insert(image_pair_ids.end(), minimum_spanning_tree.begin(), minimum_spanning_tree.end());

    int num_edge = 0;
    while(num_edge < num_redundance_edges) {
        auto pair_id = sorted_image_pairs.at(num_edge++).first;
        image_pair_ids.push_back(pair_id);
    }

    std::cout << StringPrintf("%d edges are preserved, %d edges are pruned!\n",
                              image_pair_ids.size(), image_pairs.size() - image_pair_ids.size());

    // Construct compressed scene graph.
    EIGEN_STL_UMAP(image_t, class Image) &images = compressed_scene_graph_container.Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = compressed_scene_graph_container.Cameras();

    const auto & image_ids = GetImageIds();

    for (const auto &image_id : image_ids) {
        if (!ExistsImage(image_id)) {
            continue;
        }
        const class Image &image = Image(image_id);
        CHECK(ExistsCamera(image.CameraId()));
        const class Camera &camera = Camera(image.CameraId());

        compressed_scene_graph_container.AddImage(image);
        if (!compressed_scene_graph_container.ExistsCamera(image.CameraId())) {
            compressed_scene_graph_container.AddCamera(camera);
        }
    }

    std::shared_ptr<class CorrespondenceGraph> correspondence_graph = compressed_scene_graph_container.CorrespondenceGraph();
    std::shared_ptr<class CorrespondenceGraph> correspondence_graph_ = CorrespondenceGraph();

    for (auto image_pair_id : image_pair_ids) {
        image_t image_id1, image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);

        auto image_pair = correspondence_graph_->ImagePair(image_id1, image_id2);

        const FeatureMatches &feature_matches =
            correspondence_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);
        image_pair.two_view_geometry.inlier_matches = feature_matches;

        const class Image &image1 = Image(image_id1);
        CHECK(ExistsCamera(image1.CameraId()));
        const class Camera &camera1 = Camera(image1.CameraId());
        const class Image &image2 = Image(image_id2);
        CHECK(ExistsCamera(image2.CameraId()));
        const class Camera &camera2 = Camera(image2.CameraId());

        const bool remove_redundant = !(camera1.NumLocalCameras() > 2 && camera2.NumLocalCameras());

        correspondence_graph->AddCorrespondences(image_id1, image_id2,
                                                 image_pair.two_view_geometry, remove_redundant);
    }
    correspondence_graph->Finalize();

    for (const auto &image_id : image_ids) {
        if (images.find(image_id) == images.end() || !correspondence_graph->ExistsImage(image_id)) {
            continue;
        }
        images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
        images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
    }

    std::unordered_set<image_t> image_ids_unordered;
    for (const auto image_id : image_ids) {
        image_ids_unordered.insert(image_id);
    }

    for (auto image_id : image_ids) {
        if (!correspondence_graph->ExistsImage(image_id)) {
            compressed_scene_graph_container.DeleteImage(image_id);
        }
    }
}

void SceneGraphContainer::CompressGraph(SceneGraphContainer &compressed_scene_graph_container,
                                        const float compress_ratio, const int min_corrs, const int min_mst_iter) {
    std::cout << "Graph Edge Compression" << std::endl;

    auto const &image_pairs = CorrespondenceGraph()->ImagePairs();
    const int num_images = CorrespondenceGraph()->NumImages();

    std::vector<std::pair<image_pair_t, float> > sorted_image_pairs;
    sorted_image_pairs.reserve(image_pairs.size());

    MinimumSpanningTree<image_t, float> mst_extractor;

    for (auto image_pair : image_pairs) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);

        mst_extractor.AddEdge(image_id1, image_id2, -image_pair.second.num_correspondences);
        sorted_image_pairs.emplace_back(image_pair.first, image_pair.second.num_correspondences);
    }
    std::unordered_set<std::pair<image_t, image_t> > minimum_spanning_tree;
    mst_extractor.Extract(&minimum_spanning_tree);
    auto good_pairs_ids = ViewGraphFilteringFromMaximumSpanningTree(
        minimum_spanning_tree, *CorrespondenceGraph(), min_mst_iter, 1.0);



    const int num_redundance_edges = compress_ratio * (image_pairs.size());
    std::nth_element(sorted_image_pairs.begin(), sorted_image_pairs.begin() + num_redundance_edges + num_images - 1,
                     sorted_image_pairs.end(),
                     [&](const std::pair<image_pair_t, float>& a, const std::pair<image_pair_t, float>& b) {
                         return a.second > b.second;
                     });


    std::vector<image_pair_t> image_pair_ids;
    image_pair_ids.reserve(image_pairs.size());
    std::unordered_set<image_pair_t> tree_image_pair_ids;
    for (auto pair_id : good_pairs_ids) {
        image_pair_ids.push_back(pair_id);
        tree_image_pair_ids.insert(pair_id);
    }
    int num_edge = 0;
    while(num_edge < num_redundance_edges) {
        auto pair_id = sorted_image_pairs.at(num_edge).first;
        if (tree_image_pair_ids.find(pair_id) == tree_image_pair_ids.end()) {
            image_pair_ids.push_back(pair_id);
        }
        if(sorted_image_pairs.at(num_edge).second < min_corrs){
            break;
        }
        num_edge++;
    }

    std::cout << StringPrintf("%d edges are preserved, %d edges are pruned!\n",
                              image_pair_ids.size(), image_pairs.size() - image_pair_ids.size());

    // Construct compressed scene graph.
    EIGEN_STL_UMAP(image_t, class Image) &images = compressed_scene_graph_container.Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = compressed_scene_graph_container.Cameras();

    const auto & image_ids = GetImageIds();

    for (const auto &image_id : image_ids) {
        if (!ExistsImage(image_id)) {
            continue;
        }
        const class Image &image = Image(image_id);
        CHECK(ExistsCamera(image.CameraId()));
        const class Camera &camera = Camera(image.CameraId());

        compressed_scene_graph_container.AddImage(image);
        if (!compressed_scene_graph_container.ExistsCamera(image.CameraId())) {
            compressed_scene_graph_container.AddCamera(camera);
        }
    }

    std::shared_ptr<class CorrespondenceGraph> correspondence_graph = compressed_scene_graph_container.CorrespondenceGraph();
    std::shared_ptr<class CorrespondenceGraph> correspondence_graph_ = CorrespondenceGraph();

    for (auto image_pair_id : image_pair_ids) {
        image_t image_id1, image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);

        auto image_pair = correspondence_graph_->ImagePair(image_id1, image_id2);

        const FeatureMatches &feature_matches =
            correspondence_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);
        image_pair.two_view_geometry.inlier_matches = feature_matches;

        const class Image &image1 = Image(image_id1);
        CHECK(ExistsCamera(image1.CameraId()));
        const class Camera &camera1 = Camera(image1.CameraId());
        const class Image &image2 = Image(image_id2);
        CHECK(ExistsCamera(image2.CameraId()));
        const class Camera &camera2 = Camera(image2.CameraId());

        const bool remove_redundant = !(camera1.NumLocalCameras() > 2 && camera2.NumLocalCameras());

        correspondence_graph->AddCorrespondences(image_id1, image_id2,
                                                 image_pair.two_view_geometry, remove_redundant);
    }
    correspondence_graph->Finalize();

    for (const auto &image_id : image_ids) {
        if (images.find(image_id) == images.end() || !correspondence_graph->ExistsImage(image_id)) {
            continue;
        }
        images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
        images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
    }

    std::unordered_set<image_t> image_ids_unordered;
    for (const auto image_id : image_ids) {
        image_ids_unordered.insert(image_id);
    }

    for (auto image_id : image_ids) {
        if (!correspondence_graph->ExistsImage(image_id)) {
            compressed_scene_graph_container.DeleteImage(image_id);
        }
    }
}



std::vector<image_t> SceneGraphContainer::GetImageIds() const {
    std::vector<image_t> image_ids;
    image_ids.reserve(images_.size());
    for (const auto& image_data : images_) {
        image_ids.push_back(image_data.first);
    }
    return image_ids;
}

std::vector<image_t> SceneGraphContainer::GetNewImageIds() const {
    std::vector<image_t> image_ids;
    image_ids.reserve(images_.size());
    for (const auto& image_data : images_) {
        if(image_data.second.LabelId() != 0 && 
           image_data.second.LabelId() != kInvalidLabelId){
            image_ids.push_back(image_data.first);
        }
    }
    return image_ids;
}

void SceneGraphContainer::WriteSceneGraphData(const std::string &path) const {
    std::cout << "Write SceneGraphData" << std::endl;
    correspondence_graph_->WriteCorrespondenceData(path);
}

void SceneGraphContainer::ReadSceneGraphData(const std::string &path) {
    std::cout << "Read SceneGraphData" << std::endl;
    correspondence_graph_ = std::shared_ptr<class CorrespondenceGraph>(new class CorrespondenceGraph());
    correspondence_graph_->ReadCorrespondenceData(path);
}

void SceneGraphContainer::WriteSceneGraphBinaryData(const std::string &path) const {
    std::cout << "Write SceneGraphData" << std::endl;
    correspondence_graph_->WriteCorrespondenceBinaryData(path);
}

void SceneGraphContainer::ReadSceneGraphBinaryData(const std::string &path) {
    std::cout << "Read SceneGraphData" << std::endl;
    correspondence_graph_ = std::shared_ptr<class CorrespondenceGraph>(new class CorrespondenceGraph());
    correspondence_graph_->ReadCorrespondenceBinaryData(path);
}

void SceneGraphContainer::WriteImagePairsBinaryData(const std::string &path) const {
    std::cout << "Write TwoViewGeometryData" << std::endl;
    correspondence_graph_->WriteImagePairsBinaryData(path);
}

void SceneGraphContainer::ReadImagePairsBinaryData(const std::string &path) {
    std::cout << "Read TwoViewGeometryData" << std::endl;
    correspondence_graph_->ReadImagePairsBinaryData(path);
}

void SceneGraphContainer::WriteBlueToothPairsInfoBinaryData(const std::string &path) const {
    std::cout << "Write Bluetooth Pair" << std::endl;
    correspondence_graph_->WriteBlueToothPairsInfoBinaryData(path);
}

void SceneGraphContainer::ReadBlueToothPairsInfoBinaryData(const std::string &path) {
    std::cout << "Read Bluetooth Pair" << std::endl;
    correspondence_graph_->ReadBlueToothPairsInfoBinaryData(path);
}

void SceneGraphContainer::WriteStrongLoopsBinaryData(const std::string &path) const {
    std::cout << "Write StrongLoopsData" << std::endl;
    correspondence_graph_->WriteStrongLoopsBinaryData(path);
}

void SceneGraphContainer::ReadStrongLoopsBinaryData(const std::string &path) {
    std::cout << "Read StrongLoopsData" << std::endl;
    correspondence_graph_->ReadStrongLoopsBinaryData(path);
}


void SceneGraphContainer::WriteLoopPairsInfoBinaryData(const std::string &path) const {
    std::cout << "Write LoopPairsInfoData" << std::endl;
    correspondence_graph_->WriteLoopPairsInfoBinaryData(path);
}

void SceneGraphContainer::ReadLoopPairsInfoBinaryData(const std::string &path) {
    std::cout << "Read LoopPairsInfoData" << std::endl;
    correspondence_graph_->ReadLoopPairsInfoBinaryData(path);
}

void SceneGraphContainer::WriteNormalPairsBinaryData(const std::string &path) const {
    std::cout << "Write NormalPairsData" << std::endl;
    correspondence_graph_->WriteNormalPairsBinaryData(path);
}

void SceneGraphContainer::ReadNormalPairsBinaryData(const std::string &path) {
    std::cout << "Read NormalPairsData" << std::endl;
    correspondence_graph_->ReadNormalPairsBinaryData(path);
}

void SceneGraphContainer::DeleteImage(const image_t image_id){
    if(images_.find(image_id) != images_.end()){
        images_.erase(image_id);
    }

    correspondence_graph_->DeleteImage(image_id);    
}


void SceneGraphContainer::DeleteCorrespondences(const image_t image_id,
                                                const MatXu& outlier_mask) {
    class Image& image = images_.at(image_id);
    std::vector<point2D_t> feature_outliers;
    
    const int width = outlier_mask.GetWidth();
    const int height = outlier_mask.GetHeight();

    for (size_t i = 0; i < image.NumPoints2D(); ++i) {
        class Point2D& point2D = image.Point2D(i);
        int x = point2D.X();
        int y = point2D.Y();
        if (x >= 0 && x < width && y >= 0 && y < height && 
            outlier_mask.Get(y, x)) {
            feature_outliers.push_back(i);
        }
    }
    correspondence_graph_->DeleteCorrespondences(image_id, feature_outliers);
    correspondence_graph_->Finalize();
}

void SceneGraphContainer::DistributedTrackSelection(std::vector<class Track>& tracks,
                                                    std::vector<unsigned char>& inliers, const int radius, const int max_per_block, const int min_per_block,
                                                    const int max_cover_per_view, const std::unordered_set<image_t> &const_image_ids) {
    int num_selected = 0;
    if (radius > 0) {
        const int grid_size = (2 * radius + 1);
        const float inv_grid_size = 1.0f / grid_size;
        
        const std::vector<image_t>& image_ids = GetImageIds();
        image_t max_image_id = 0;
        for (auto image_id : image_ids) {
            max_image_id = std::max(image_id, max_image_id);
        } 
        std::vector<MatXu> cover_mask(max_image_id + 1);
        std::vector<int> num_per_image(max_image_id + 1, 0);
        for (size_t i = 0; i < image_ids.size(); ++i) {
            image_t image_id = image_ids.at(i);
            class Image& image = Image(image_id);
            class Camera& camera = Camera(image.CameraId());

            const int g_width = camera.Width() * inv_grid_size + 1;
            const int g_height = camera.Height() * inv_grid_size + 1;
            cover_mask[image_id] = MatXu(g_width, g_height, 1);
            num_per_image[image_id] = 0;
        }

        for (size_t i = 0; i < tracks.size(); ++i) {
            Track& track = tracks.at(i);

            if(track.Length() < 3)
                continue;

            bool selected = false;
            for (auto track_elem : track.Elements()) {
                const class Image& image = images_.at(track_elem.image_id);
                const class Point2D& point2D = image.Point2D(track_elem.point2D_idx);
                MatXu& mask = cover_mask.at(track_elem.image_id);

                int x = point2D.X() * inv_grid_size;
                int y = point2D.Y() * inv_grid_size;
                int num = mask.Get(y, x);
                if ( num < min_per_block ||
                     (num > min_per_block &&  num < max_per_block &&
                      num_per_image[track_elem.image_id] < max_cover_per_view)) {
                    selected = true;
                    break;
                }
            }
            if (selected) {
                for (auto track_elem : track.Elements()) {
                    const class Image& image = images_.at(track_elem.image_id);
                    const class Point2D& point2D = image.Point2D(track_elem.point2D_idx);
                    MatXu& mask = cover_mask.at(track_elem.image_id);

                    int x = point2D.X() * inv_grid_size;
                    int y = point2D.Y() * inv_grid_size;
                    int num = mask.Get(y, x);

                    mask.Set(y, x, num + 1);
                    num_per_image[track_elem.image_id]++;
                }
                inliers.at(i) = 1;
                num_selected++;
            }
        }
        std::cout << StringPrintf("Select Match Point %d/%d(radius=%d)\n", num_selected, tracks.size(), radius);
        this->CorrespondenceGraph()->TrackSelection(tracks, inliers, -1, const_image_ids);
    } else {
        std::fill(inliers.begin(), inliers.end(), 1);
        num_selected = tracks.size();
        std::cout << StringPrintf("Select Match Point %d/%d(radius=%d)\n", num_selected, tracks.size(), radius);
        this->CorrespondenceGraph()->TrackSelection(tracks, inliers, max_cover_per_view, const_image_ids);
    }
}


void SceneGraphContainer::DistributeTrack(std::vector<class Track>& tracks, 
    std::vector<unsigned char>& inliers, const int radius,
    const std::unordered_set<image_t> &const_image_ids) {
    int num_selected = 0;
    if (radius > 0) {
        const int grid_size = (2 * radius + 1);
        const std::vector<image_t>& image_ids = GetImageIds();
        std::unordered_map<image_t, MatXu> cover_mask;
        for (size_t i = 0; i < image_ids.size(); ++i) {
            image_t image_id = image_ids.at(i);
            class Image& image = Image(image_id);
            class Camera& camera = Camera(image.CameraId());

            const int g_width = camera.Width() / grid_size + 1;
            const int g_height = camera.Height() / grid_size + 1;
            cover_mask[image_id] = MatXu(g_width, g_height, 1);
        }

        for (size_t i = 0; i < tracks.size(); ++i) {
            Track& track = tracks.at(i);
            bool selected = true;
            for (auto track_elem : track.Elements()) {
                const class Image& image = images_.at(track_elem.image_id);
                const class Point2D& point2D = image.Point2D(track_elem.point2D_idx);
                MatXu& mask = cover_mask.at(track_elem.image_id);

                int x = point2D.X() / grid_size;
                int y = point2D.Y() / grid_size;
                if (mask.Get(y, x) != 0) {
                    selected = false;
                    break;
                }
                mask.Set(y, x, 1);
            }
            if (selected) {
                inliers.at(i) = 1;
                num_selected++;
            } else {
                for (auto track_elem : track.Elements()) {
                    if (const_image_ids.find(track_elem.image_id) != const_image_ids.end()){
                        inliers.at(i) = 1;
                        num_selected++;
                        break;
                    }
                }
            }
        }
    } else {
        std::fill(inliers.begin(), inliers.end(), 1);
        num_selected = tracks.size();
    }

    std::cout << StringPrintf("Select Match Point %d/%d(radius=%d)\n", num_selected, tracks.size(), radius);
}

}  // namespace sensemap
