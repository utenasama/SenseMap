//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CONTAINER_SCENE_GRAPH_CONTAINER_H_
#define SENSEMAP_CONTAINER_SCENE_GRAPH_CONTAINER_H_

#include <set>
#include <memory>

#include "base/image.h"
#include "util/types.h"
#include "graph/correspondence_graph.h"
#include "graph/minimum_spanning_tree.h"

namespace sensemap {

class SceneGraphContainer {
public:
    SceneGraphContainer();

    void Copy(SceneGraphContainer & scene_graph_contrainer);

public:

    inline size_t NumCameras() const;
    inline size_t NumImages() const;

    // Get specific objects.
    inline class Camera& Camera(const camera_t camera_id);
    inline const class Camera& Camera(const camera_t camera_id) const;
    inline class Image& Image(const image_t image_id);
    inline const class Image& Image(const image_t image_id) const;

    // Get all objects.
    inline const EIGEN_STL_UMAP(camera_t, class Camera) & Cameras() const;
    inline const EIGEN_STL_UMAP(image_t, class Image) & Images() const;
    inline EIGEN_STL_UMAP(camera_t, class Camera) & Cameras();
    inline EIGEN_STL_UMAP(image_t, class Image) & Images();

    std::vector<image_t> GetImageIds() const;
    std::vector<image_t> GetNewImageIds() const;

    // Check whether specific object exists.
    inline bool ExistsCamera(const camera_t camera_id) const;
    inline bool ExistsImage(const image_t image_id) const;

    // Manually add data to container.
    void AddCamera(const class Camera& camera);
    void AddImage(const class Image& image);

    inline const std::shared_ptr<class CorrespondenceGraph>
        CorrespondenceGraph() const;

    // Generate cluster SceneGraphContainer from image_ids.
    void ClusterSceneGraphContainer(
        const std::set<image_t> &image_ids,
        SceneGraphContainer &scene_graph_container);

    void ConvertRigSceneGraphContainer(SceneGraphContainer &scene_graph_container, 
                                       const std::vector<image_t>& image_ids = std::vector<image_t>());

    std::shared_ptr<std::unordered_set<image_pair_t> > GetTreeEdges(bool force = false);

    void CompressGraph(SceneGraphContainer &compressed_scene_graph_container,
                       const float compress_ratio = 0.4f);

    void CompressGraph(SceneGraphContainer &compressed_scene_graph_container,
                       const float compress_ratio, const int min_corrs, const int min_mst_iter = 1);

    void WriteSceneGraphData(const std::string &path) const;
    void ReadSceneGraphData(const std::string &path);

    void WriteSceneGraphBinaryData(const std::string &path) const;
    void ReadSceneGraphBinaryData(const std::string &path);

    void WriteImagePairsBinaryData(const std::string &path) const;
    void ReadImagePairsBinaryData(const std::string &path);

    void WriteBlueToothPairsInfoBinaryData(const std::string &path) const;
    void ReadBlueToothPairsInfoBinaryData(const std::string &path);

    void WriteStrongLoopsBinaryData(const std::string& path) const;
    void ReadStrongLoopsBinaryData(const std::string& path);

    void WriteLoopPairsInfoBinaryData(const std::string& path) const;
    void ReadLoopPairsInfoBinaryData(const std::string& path);

    void WriteNormalPairsBinaryData(const std::string& path) const;
    void ReadNormalPairsBinaryData(const std::string& path);

    void DeleteImage(const image_t image_id);

    void DeleteCorrespondences(const image_t image_id,
                            const MatXu& outlier_mask);

    void DistributeTrack(std::vector<class Track>& tracks, 
        std::vector<unsigned char>& inliers, const int radius,
        const std::unordered_set<image_t> &const_image_ids = std::unordered_set<image_t>());

    void DistributedTrackSelection(std::vector<class Track>& tracks, std::vector<unsigned char>& inliers,
                                   const int radius, const int max_per_block, const int min_per_block, const int max_cover_per_view,
                                   const std::unordered_set<image_t> &const_image_ids = std::unordered_set<image_t>());


private:

    std::shared_ptr<class CorrespondenceGraph> correspondence_graph_;
    std::shared_ptr<std::unordered_set<image_pair_t> > tree_image_pair_ids_;

    EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
    EIGEN_STL_UMAP(image_t, class Image) images_;

};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////

size_t SceneGraphContainer::NumCameras() const { return cameras_.size(); }

size_t SceneGraphContainer::NumImages() const { return images_.size(); }

class Camera& SceneGraphContainer::Camera(const camera_t camera_id) { return cameras_.at(camera_id); }

const class Camera& SceneGraphContainer::Camera(const camera_t camera_id) const { return cameras_.at(camera_id); }

class Image& SceneGraphContainer::Image(const image_t image_id) { return images_.at(image_id); }

const class Image& SceneGraphContainer::Image(const image_t image_id) const { return images_.at(image_id); }

const EIGEN_STL_UMAP(camera_t, class Camera) & SceneGraphContainer::Cameras() const { return cameras_; }

const EIGEN_STL_UMAP(image_t, class Image) & SceneGraphContainer::Images() const { return images_; }

EIGEN_STL_UMAP(camera_t, class Camera) & SceneGraphContainer::Cameras() { return cameras_; }

EIGEN_STL_UMAP(image_t, class Image) & SceneGraphContainer::Images() { return images_; }

bool SceneGraphContainer::ExistsCamera(const camera_t camera_id) const { return cameras_.find(camera_id) != cameras_.end(); }

bool SceneGraphContainer::ExistsImage(const image_t image_id) const { return images_.find(image_id) != images_.end(); }

const std::shared_ptr<class CorrespondenceGraph> SceneGraphContainer::CorrespondenceGraph() const { return correspondence_graph_; }

// const std::shared_ptr<class ViewGraph> SceneGraphContainer::ViewGraph() const {return view_graph_;}

} // namespace sensemap

#endif
