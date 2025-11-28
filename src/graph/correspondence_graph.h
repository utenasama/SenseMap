//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_GRAPH_CORRESPONDENCE_GRAPH_H_
#define SENSEMAP_GRAPH_CORRESPONDENCE_GRAPH_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>

#include "util/types.h"
#include "util/image_pair.h"
#include "util/mat.h"
#include "feature/types.h"
#include "estimators/two_view_geometry.h"

namespace sensemap{

// Scene graph represents the graph of image to image and feature to feature
// correspondences of a scene.
class CorrespondenceGraph {
public:
    struct Correspondence {
        Correspondence()
            : image_id(kInvalidImageId), point2D_idx(kInvalidPoint2DIdx) {}
        Correspondence(const image_t image_id,
                       const point2D_t point2D_idx)
            : image_id(image_id), point2D_idx(point2D_idx) {}

        // The identifier of the corresponding image.
        image_t image_id;

        // The index of the corresponding point in the corresponding image.
        point2D_t point2D_idx;
    };

    struct Image {
        // Number of 2D points with at least one correspondence to another image.
        point2D_t num_observations = 0;

        // Total number of correspondences to other images. This measure is useful
        // to find a good initial pair, that is connected to many images.
        point2D_t num_correspondences = 0;

        // Correspondences to other images per image point.
        std::vector<std::vector<Correspondence>> corrs;
    };

    struct ImagePair {
        // The number of correspondences between pairs of images.
        point2D_t num_correspondences = 0;

        image_t image_id1;
        image_t image_id2;
        
        // Two View Geometry.
        TwoViewGeometry two_view_geometry;
    };
 public:

    CorrespondenceGraph();

    void Copy(CorrespondenceGraph &correspondence_graph);

    // Number of added images.
    inline size_t NumImages() const;

    // Number of added images.
    inline size_t NumImagePairs() const;

    // Check whether image exists.
    inline bool ExistsImage(const image_t image_id) const;

    // Get the number of observations in an image. An observation is an image
    // point that has at least one correspondence.
    inline point2D_t NumObservationsForImage(const image_t image_id) const;

    // Get the number of correspondences per image.
    inline point2D_t NumCorrespondencesForImage(const image_t image_id) const;

    // Get the number of correspondences between a pair of images.
    inline point2D_t NumCorrespondencesBetweenImages(const image_t image_id1,
                                                     const image_t image_id2) const;

    inline point2D_t NumCorrespondencesBetweenImages(const image_pair_t pair_id) const;

    // Get the number of correspondences between all images.
    std::unordered_map<image_pair_t, point2D_t> NumCorrespondencesBetweenImages() const;

    // Return ImagePairs
    EIGEN_STL_UMAP(image_pair_t, struct ImagePair) ImagePairs() const;

    bool ExistImagePair(const image_t image_id1, const image_t image_id2) const;
    // Return image pair between two images.
    struct ImagePair ImagePair(const image_t image_id1, const image_t image_id2) const;
    struct ImagePair ImagePair(const image_pair_t pair_id) const;

    // Return Image data of given image.
    CorrespondenceGraph::Image Image(const image_t image_id) const;
    struct CorrespondenceGraph::Image& ImageRef(const image_t image_id);

    // Calculate ImageNeighbors
    void CalculateImageNeighbors();

    void UpdateImageNeighbors(const image_t image_id1, const image_t image_id2);

    // Return map of image with a collection of its neighbors
    std::unordered_map<image_t, std::unordered_set<image_t>>
    ImageNeighbors() const;
    
    // Return a collection of neighbors of given image.
    std::unordered_set<image_t> ImageNeighbor(const image_t image_id) const;

    void UpdateImagePairsConfidence();

    // Finalize the database manager.
    //
    // - Calculates the number of observations per image by counting the number
    //   of image points that have at least one correspondence.
    // - Deletes images without observations, as they are useless for SfM.
    // - Shrinks the correspondence vectors to their size to save memory.
    void Finalize();

    // Add new image to the correspondence graph.
    void AddImage(const image_t image_id,
                  const size_t num_points2D);
    
    void AddImage(const image_t image_id,
                  const struct CorrespondenceGraph::Image image);


    void AddCorrespondences(const image_t image_id1,
                            const image_t image_id2,
                            const TwoViewGeometry& two_view_geometry,
                            const bool remove_redundant = true);

    // For Global rotation averaging.
    void AddCorrespondences(const image_t image_id1,
                            const image_t image_id2,
                            const struct ImagePair& pair);

    // For AprilTag triangulation
    void UpdateCorrespondence(const image_t image_id1,
                              const image_t image_id2,
                              const point2D_t point_id1,
                              const point2D_t point_id2);

    void UpdateCorrespondence(const image_t image_id1,
                              const image_t image_id2,
                              const TwoViewGeometry& two_view_geometry);

    void DeleteCorrespondences(const image_t image_id1,
                               const image_t image_id2);

    void DeleteCorrespondences(const image_t image_id,
                               const std::vector<point2D_t>& feature_outliers);

    // Find the correspondence of an image observation to all other images.
    inline const std::vector<Correspondence>& FindCorrespondences(
        const image_t image_id,
        const point2D_t point2D_idx) const;

    // Find correspondences to the given observation.
    //
    // Transitively collects correspondences to the given observation by first
    // finding correspondences to the given observation, then looking for
    // correspondences to the collected correspondences in the first step, and so
    // forth until the transitivity is exhausted or no more correspondences are
    // found. The returned list does not contain duplicates and contains
    // the given observation.
    void FindTransitiveCorrespondences(
        const image_t image_id,
        const point2D_t point2D_idx,
        const size_t transitivity,
        std::vector<Correspondence>* found_corrs) const;

    // Find all correspondences between two images.
    FeatureMatches FindCorrespondencesBetweenImages(const image_t image_id1,
                                                    const image_t image_id2) const;
    
    // Find all transitive correspondences between two images.
    void FindTransitiveCorrespondencesBetweenImages(
                                            const image_t image_id1,
                                            const image_t image_id2,
                                            const size_t transitivity,
                                            FeatureMatches* corrs) const;

    // Check whether the image point has correspondences.
    inline bool HasCorrespondences(const image_t image_id,
                                   const point2D_t point2D_idx) const;

    // Check whether the given observation is part of a two-view track, i.e.
    // it only has one correspondence and that correspondence has the given
    // observation as its only correspondence.
    bool IsTwoViewObservation(const image_t image_id,
                              const point2D_t point2D_idx) const;

    std::vector<class Track> GenerateTracks(int track_degree = 2,
                                            bool remove_ambiguous = true,
                                            bool verbose = false);

    void TrackSelection(std::vector<class Track>& tracks,
                        std::vector<unsigned char>& inliers,
                        const int max_cover_per_view,
                        const std::unordered_set<image_t> &const_image_ids 
                        = std::unordered_set<image_t>());

    // Serialization
    void WriteCorrespondenceData(const std::string &path) const;
    void ReadCorrespondenceData(const std::string &path);

    void WriteCorrespondenceBinaryData(const std::string &path) const;
    void ReadCorrespondenceBinaryData(const std::string &path);

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

    std::vector<std::pair<image_t, image_t>> ReadCorrespondencePairBinaryData(const std::string &path);

    // Visualization
    void ExportToGraph(const std::string& path) const;

    // Set the image index map for originized graph output
    void SetImageIndexMap(const std::unordered_map<image_t, image_t> image_index_map);

    // Delete one certain image 
    void DeleteImage(image_t image_id);

    inline void SetStrongLoopPairs(const std::unordered_set<image_pair_t>& strong_loop_pairs);
    inline void SetLoopPairsInfo(const std::unordered_map<image_pair_t,double>& loop_pairs_info);
    inline void SetNoramlPairs(const std::unordered_set<image_pair_t>& normal_pairs);
    inline void SetBluetoothPairsInfo(const std::unordered_map<image_pair_t,double>& loop_pairs_info);

    const std::unordered_set<image_pair_t>& GetStrongLoopPairs() const ;
    const std::unordered_map<image_pair_t,double>& GetLoopPairsInfo() const;
    const std::unordered_set<image_pair_t>& GetNormalPairs() const;
    const std::unordered_set<image_pair_t>& GetAllPairs() const;
    const std::unordered_map<image_pair_t,double>& GetBluetoothPairsInfo() const;

    void Clear();
    
 private:
    std::unordered_map<image_pair_t, double> pairs_bluetooth_info_;
    std::unordered_set<image_pair_t> strong_loop_pairs_;
    std::unordered_map<image_pair_t,double> loop_pairs_info_;
    std::unordered_set<image_pair_t> normal_pairs_;

    std::unordered_map<image_t, struct Image> images_;
    EIGEN_STL_UMAP(image_pair_t, struct ImagePair) image_pairs_;
    std::unordered_map<image_t, std::unordered_set<image_t> > image_neighbors_;
    // Store the map for image id and corresponding image name index
    std::unordered_map<image_t, image_t> image_index_map_;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

size_t CorrespondenceGraph::NumImages() const { return images_.size(); }

size_t CorrespondenceGraph::NumImagePairs() const { return image_pairs_.size(); }

bool CorrespondenceGraph::ExistsImage(const image_t image_id) const { return images_.find(image_id) != images_.end(); }

point2D_t CorrespondenceGraph::NumObservationsForImage(const image_t image_id) const { return images_.at(image_id).num_observations; }

point2D_t CorrespondenceGraph::NumCorrespondencesForImage(const image_t image_id) const { return images_.at(image_id).num_correspondences; }

point2D_t CorrespondenceGraph::NumCorrespondencesBetweenImages(
    const image_t image_id1,
    const image_t image_id2) const {
    const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id1, image_id2);
    const auto it = image_pairs_.find(pair_id);
    if (it == image_pairs_.end()) {
      return 0;
    } else {
      return static_cast<point2D_t>(it->second.num_correspondences);
    }
}

point2D_t CorrespondenceGraph::NumCorrespondencesBetweenImages(const image_pair_t pair_id) const {
    const auto it = image_pairs_.find(pair_id);
    if (it == image_pairs_.end()) {
      return 0;
    } else {
      return static_cast<point2D_t>(it->second.num_correspondences);
    }
}

const std::vector<CorrespondenceGraph::Correspondence>&
CorrespondenceGraph::FindCorrespondences(const image_t image_id,
                                         const point2D_t point2D_idx) const {
  return images_.at(image_id).corrs.at(point2D_idx);
}

bool CorrespondenceGraph::HasCorrespondences(const image_t image_id,
                                             const point2D_t point2D_idx) const {
    return !images_.at(image_id).corrs.empty()
        && !images_.at(image_id).corrs.at(point2D_idx).empty();
}

inline void CorrespondenceGraph::SetStrongLoopPairs(const std::unordered_set<image_pair_t>& strong_loop_pairs){
    strong_loop_pairs_ = strong_loop_pairs;
}

inline void CorrespondenceGraph::SetBluetoothPairsInfo(const std::unordered_map<image_pair_t,double>& pairs_bluetooth_info){
    pairs_bluetooth_info_ = pairs_bluetooth_info;
}


inline void CorrespondenceGraph::SetLoopPairsInfo(const std::unordered_map<image_pair_t,double>& loop_pairs_info){
    loop_pairs_info_ = loop_pairs_info;
}

inline void CorrespondenceGraph::SetNoramlPairs(const std::unordered_set<image_pair_t>& normal_pairs){
    normal_pairs_ = normal_pairs;
}

}//namespace sensemap

#endif //SENSEMAP_GRAPH_CORRESPONDENCE_GRAPH_H_
