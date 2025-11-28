//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_SFM_INCREMENTAL_TRIANGULATOR_H_
#define SENSEMAP_SFM_INCREMENTAL_TRIANGULATOR_H_

#include <memory>
#include <unordered_set>

#include "util/types.h"
#include "graph/correspondence_graph.h"
#include "base/reconstruction.h"

namespace sensemap {

// It holds the state and provides all functionality for triangulation.
class IncrementalTriangulator {
public:
    struct Options {
        // Maximum transitivity to search for correspondences.
        int max_transitivity = 1;

        // Maximum angular error to create new triangulations.
        double create_max_angle_error = 2.0;

        // Maximum angular error to continue existing triangulations.
        double continue_max_angle_error = 2.0;

        // Maximum reprojection error in pixels to merge triangulations.
        double merge_max_reproj_error = 4.0;

        // Maximum reprojection error to complete an existing triangulation.
        double complete_max_reproj_error = 4.0;

        // Maximum transitivity for track completion.
        int complete_max_transitivity = 5;

        // Maximum angular error to re-triangulate under-reconstructed image pairs.
        double re_max_angle_error = 5.0;

        // Minimum ratio of common triangulations between an image pair over the
        // number of correspondences between that image pair to be considered
        // as under-reconstructed.
        double re_min_ratio = 0.2;

        // Maximum number of trials to re-triangulate an image pair.
        int re_max_trials = 1;

        // Minimum pairwise triangulation angle for a stable triangulation.
        // Larger threshold should be in favor of accurate triangluation. 
        // Excessive value could cause under-reconstruction.
        double min_angle = 1.5;

        // Whether to ignore two-view tracks.
        bool ignore_two_view_tracks = true;

        // Thresholds for bogus camera parameters. Images with bogus camera
        // parameters are ignored in triangulation.
        double min_focal_length_ratio = 0.1;
        double max_focal_length_ratio = 10.0;
        double max_extra_param = 1.0;

        bool Check() const;
    };

    // Data for a correspondence / element of a track, used to store all
    // relevant data for triangulation, in order to avoid duplicate lookup
    // in the underlying unordered_map's in the Reconstruction
    struct CorrData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        image_t image_id;
        point2D_t point2D_idx;
        const class Image* image;
        const class Camera* camera;
        const class Point2D* point2D;
    };

    // Create new incremental triangulator. Note that both the correspondence
    // graph and the reconstruction objects must live as long as the triangulator.
    IncrementalTriangulator(const std::shared_ptr<CorrespondenceGraph> correspondence_graph,
                            std::shared_ptr<Reconstruction> reconstruction);

    // Triangulate observations of image.
    //
    // Triangulation includes creation of new points, continuation of existing
    // points, and merging of separate points if given image bridges tracks.
    //
    // Note that the given image must be registered and its pose must be set
    // in the associated reconstruction.
    size_t TriangulateImage(const Options& options, const image_t image_id);

    // Complete triangulations for image. Tries to create new tracks for not
    // yet triangulated observations and tries to complete existing tracks.
    // Returns the number of completed observations.
    size_t CompleteImage(const Options& options, const image_t image_id);

    // Complete tracks for specific Map Points.
    //
    // Completion tries to recursively add observations to a track that might
    // have failed to triangulate before due to inaccurate poses, etc.
    // Returns the number of completed observations.
    size_t CompleteTracks(const Options& options,
                            const std::unordered_set<mappoint_t>& mappoint_ids);

    // Complete tracks of all Map Points.
    // Returns the number of completed observations.
    size_t CompleteAllTracks(const Options& options);

    // Merge tracks of for specific Map Points.
    // Returns the number of merged observations.
    size_t MergeTracks(const Options& options,
                        const std::unordered_set<mappoint_t>& mappoint_ids);

    // Merge tracks of all Map Points.
    // Returns the number of merged observations.
    size_t MergeAllTracks(const Options& options);

    // Perform retriangulation for under-reconstructed image pairs. Under-
    // reconstruction usually occurs in the case of a drifting reconstruction.
    //
    // Image pairs are under-reconstructed if less than `Options::tri_re_min_ratio
    // > tri_ratio`, where `tri_ratio` is the number of triangulated matches over
    // inlier matches between the image pair.
    size_t Retriangulate(const Options& options);
    size_t Retriangulate(const Options& options,std::unordered_set<image_t>* image_set);

    // Perform triangulation for all mappoints. Just being invoked after GBA.
    size_t RetriangulateAllTracks(const Options& options);

    // Get changed Map Points, since the last call to `ClearModifiedMapPoints`.
    const std::unordered_set<mappoint_t>& GetModifiedMapPoints();
    
    void AddModifiedMapPoint(const mappoint_t mappoint_id);

    // Clear the collection of changed Map Points.
    void ClearModifiedMapPoints();

    size_t Recreate(const Options& options,
                    const mappoint_t mappoint_id);
                    
private:

    // Clear cache of bogus camera parameters and merge trials.
    void ClearCaches();

    // Find (transitive) correspondences to other images.
    size_t Find(const Options& options, const image_t image_id,
                const point2D_t point2D_idx, const size_t transitivity,
                std::vector<CorrData>* corrs_data);

    // Try to create a new Map Point from the given correspondences.
    size_t Create(const Options& options,
                    const std::vector<CorrData>& corrs_data);

    // Try to continue the Map Point with the given correspondences.
    size_t Continue(const Options& options, const CorrData& ref_corr_data,
                    const std::vector<CorrData>& corrs_data);

    // Try to merge Map Point with any of its corresponding Map Points.
    size_t Merge(const Options& options, const mappoint_t mappoint_id);

    // Try to transitively complete the track of a Map Point.
    size_t Complete(const Options& options, const mappoint_t mappoint_id);

    // Check if camera has bogus parameters and cache the result.
    bool HasCameraBogusParams(const Options& options, const Camera& camera);

private:
    // Database cache for the reconstruction. Used to retrieve correspondence
    // information for triangulation.
    const std::shared_ptr<CorrespondenceGraph> correspondence_graph_;

    // Reconstruction of the model. Modified when triangulating new points.
    std::shared_ptr<Reconstruction> reconstruction_;

    // Cache for cameras with bogus parameters.
    std::unordered_map<camera_t, bool> camera_has_bogus_params_;

    // Cache for tried track merges to avoid duplicate merge trials.
    std::unordered_map<mappoint_t, std::unordered_set<mappoint_t>> merge_trials_;

    // Number of trials to retriangulate image pair.
    std::unordered_map<image_pair_t, int> re_num_trials_;

    // Changed Map Points, i.e. if a Map Point is modified (created, continued,
    // deleted, merged, etc.). Cleared once `ModifiedMapPoints` is called.
    std::unordered_set<mappoint_t> modified_mappoint_ids_;
};

}

#endif