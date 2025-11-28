//Copyright (c) 2019, SenseTime Group.
//All rights reserved.


// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef SENSEMAP_BASE_RECONSTRUCTION_H_
#define SENSEMAP_BASE_RECONSTRUCTION_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <map>

#include <Eigen/Core>

#include "util/types.h"
#include "util/color_space.h"
#include "camera.h"
#include "image.h"
#include "point2d.h"
#include "mappoint.h"
#include "track.h"
#include "container/scene_graph_container.h"
#include "container/feature_data_container.h"
#include "util/alignment.h"
#include "lidar/lidar_sweep.h"
#include "lidar/utils.h"
#include "lidar/voxel_map.h"

namespace sensemap {
class Reconstruction {
public:
    Reconstruction();

    // Get number of objects.
    inline size_t NumCameras() const;
    inline size_t NumImages() const;
    inline size_t NumRegisterImages() const;
    inline size_t NumMapPoints() const;
    inline size_t NumImagePairs() const;
    inline size_t NumLidarSweep() const;
    inline size_t NumRegisterLidarSweep() const;

    // Get const objects.
    inline const class Camera& Camera(const camera_t camera_id) const; 
    inline const class Image& Image(const image_t image_id) const;
    inline const class LidarSweep& LidarSweep(const sweep_t sweep_id) const;
    inline const class MapPoint& MapPoint(const mappoint_t mappoint_id) const;
    inline const std::pair<size_t, size_t>& ImagePair(const image_pair_t pair_id) const;
    inline const std::pair<size_t, size_t>& ImagePair(const image_t image_id1, const image_t image_id2) const;

    inline class LidarPointCloud& CornerMap(); 
    inline class LidarPointCloud& SurfaceMap();

    inline const std::shared_ptr<class VoxelMap>& VoxelMap() const;
    inline std::shared_ptr<class VoxelMap>& VoxelMap();

    // Get mutable objects.
    inline class Camera& Camera(const camera_t camera_id);
    inline class Image& Image(const image_t image_id);
    inline class LidarSweep& LidarSweep(const sweep_t sweep_id);
    inline class MapPoint& MapPoint(const mappoint_t mappoint_id);
    inline std::pair<size_t, size_t>& ImagePair(const image_pair_t pair_id);
    inline std::pair<size_t, size_t>& ImagePair(const image_t image_id1, const image_t image_id2);

    // Get reference to all objects.
    inline const EIGEN_STL_UMAP(camera_t, class Camera) & Cameras() const;
    inline const EIGEN_STL_UMAP(image_t, class Image) & Images() const;
    std::unordered_map<std::string, image_t> GetImageNames() const;
    inline const std::vector<image_t> & RegisterImageIds() const;
    inline const std::vector<sweep_t> RegisterSweepIds() const;
    inline const EIGEN_STL_UMAP(sweep_t, class LidarSweep) & LidarSweeps() const;
    inline const EIGEN_STL_UMAP(mappoint_t, class MapPoint) & MapPoints() const;
    inline const std::unordered_map<image_pair_t, std::pair<size_t, size_t> > & ImagePairs() const;
    const std::vector<image_t> RegisterImageIds(const std::vector<image_t> exclude_image_ids) const;

    const std::vector<image_t> GetNewImageIds() const;
    const std::vector<image_t> RegisterImageSortIds() const;

    const std::vector<sweep_t> RegisterSweepSortIds() const;

    // Get correspondence graph
    inline const std::shared_ptr<CorrespondenceGraph> GetCorrespondenceGraph() const;

    // Identifiers of all Map Points
    std::unordered_set<mappoint_t> MapPointIds() const;

    // Check whether specific object exists.
    inline bool ExistsCamera(const camera_t camera_id) const;
    inline bool ExistsImage(const image_t image_id) const;
    inline bool ExistsLidarSweep(const sweep_t sweep_id) const;
    inline bool ExistsMapPoint(const mappoint_t mappoint_id) const;
    inline bool ExistsImagePair(const image_pair_t pair_id) const;

    // Set data
    inline bool SetCamera(const camera_t camera_id, const class Camera &camera);
    // // Load data from SceneGraphContainer
    // void Load();

    // // Setup all relevant data structures before reconstruction.
    // void SetUp(const std::shared_ptr<CorrespondenceGraph> correspondence_graph);
    void SetUp(const std::shared_ptr<SceneGraphContainer> scene_graph_container);

    std::unordered_map<image_t, std::vector<image_t>> 
        ConvertRigReconstruction(Reconstruction& reconstruction);


    // Finalize the Reconstruction after the reconstruction has finished.
    // Once a scene has been finalized, it cannot be used for reconstruction.
    // This removes all not yet registered images and unused cameras, in order to
    // save memory.
    void TearDown();

    // Add new camera.
    void AddCamera(const class Camera& camera);
    
    // Add new image.
    void AddImage(const class Image& image);

    // Add lidar sweeps.
    void AddLidarSweep(const class LidarSweep& sweep);

    // Add new lidar scans.
    sweep_t AddLidarData(const std::string& lidar_path, const std::string& lidar_name);

    // Add new image nohasmappoint.
    void AddImageNohasMapPoint(const class Image& image);

    //delete an image, mainly used in map-update
    void DeleteImage(const image_t image_id);

    // Add new 3D mappoint with specified mappoint id
    bool AddMapPoint(mappoint_t mappoint_id, const Eigen::Vector3d& xyz, Track track, 
                    const Eigen::Vector3ub &color = Eigen::Vector3ub::Zero(),
                    bool verbose = false);

    // Add new 3D object, and return its unique ID.
    mappoint_t AddMapPoint(const Eigen::Vector3d &xyz, Track track, 
                           const Eigen::Vector3ub &color = Eigen::Vector3ub::Zero(),
                           bool verbose = false);

    // Add new 3D mappoint with error
    mappoint_t AddMapPointWithError(const Eigen::Vector3d & xyz, Track track,
                                    const Eigen::Vector3ub & color = Eigen::Vector3ub::Zero(),
                                    double error = 0);
    
    // Add 3D mappoint with error
    bool AddMapPointWithError(mappoint_t mappoint_id, const Eigen::Vector3d &xyz, Track track, 
                              double error, const Eigen::Vector3ub &color = Eigen::Vector3ub::Zero(),
                              bool verbose = false);

    // Add observation to existing Map Point.
    void AddObservation(const mappoint_t mappoint_id, const TrackElement& track_el);

    // Merge two Map Points and return new identifier of new Map Point.
    // The location of the merged Map Point is a weighted average of the two
    // original Map Point's locations according to their track lengths.
    mappoint_t MergeMapPoints(const mappoint_t mappoint_id1, const mappoint_t mappoint_id2);

    // Delete a Map Point, and all its references in the observed images.
    void DeleteMapPoint(const mappoint_t mappoint_id);

    // Delete one observation from an image and the corresponding Map Point.
    // Note that this deletes the entire Map Point, if the track has two elements
    // prior to calling this method.
    void DeleteObservation(const image_t image_id, const point2D_t point2D_idx);

    // Delete all 2D points of all images and all 3D points.
    void DeleteAllPoints2DAndPoints3D();

    // Register an existing image.
    void RegisterImage(const image_t image_id);

    // De-register an existing image, and all its references.
    void DeRegisterImage(const image_t image_id);

    // Register an existing lidar scan.
    void RegisterLidarSweep(const sweep_t sweep_id);

    // De-register an existing lidar scan.
    void DeRegisterLidarSweep(const sweep_t sweep_id);

    // Check if image is registered.
    inline bool IsImageRegistered(const image_t image_id) const;
   
    // Normalize scene by scaling and translation to avoid degenerate
    // visualization after bundle adjustment and to improve numerical
    // stability of algorithms.
    //
    // Translates scene such that the mean of the camera centers or point
    // locations are at the origin of the coordinate system.
    //
    // Scales scene such that the minimum and maximum camera centers are at the
    // given `extent`, whereas `p0` and `p1` determine the minimum and
    // maximum percentiles of the camera centers considered.
    void Normalize(const double extent = 10.0, const double p0 = 0.1,
                    const double p1 = 0.9, const bool use_images = true);


    Eigen::Matrix3x4d NormalizeWoScale(const double extent = 10.0, 
                                       const double p0 = 0.1, 
                                       const double p1 = 0.9, 
                                       const bool use_images = true);

    Eigen::Matrix3x4d Centration(const double extent = 10.0, 
        const double p0 = 0.1, const double p1 = 0.9);

    void Decentration(const Eigen::Matrix3x4d& transform);

    void AddPriorToResult();

    // Merge the given reconstruction into this reconstruction by registering the
    // images registered in the given but not in this reconstruction and by
    // merging the two clouds and their tracks. The coordinate frames of the two
    // reconstructions are aligned using the projection centers of common
    // registered images. Return true if the two reconstructions could be merged.
    bool Merge(const Reconstruction& reconstruction,
               const double max_reproj_error);

    bool Merge(const Reconstruction& reconstruction,
               const CorrespondenceGraph& scene_correspondence_graph,
               const double max_reproj_error,
               const double min_tri_angle);

    // bool Merge(const Reconstruction& reconstruction,
    //            const CorrespondenceGraph* full_correspondence_graph,
    //            const ViewGraph* full_view_graph, 
    //            const double max_reproj_error);

    //Merge another reconstruction into this reconstruction using the given transform 
    bool Merge(const Reconstruction& reconstruction,
               const Eigen::Matrix3x4d transform,
               const double max_reproj_error);

    std::unordered_map<image_t, image_t>  
        Append(const Reconstruction& reconstruction);

    // Find images that are both present in this and the given reconstruction.
    std::vector<image_t> FindCommonRegImageIds(
        const Reconstruction& reconstruction) const;

    // Filter Map Points with large reprojection error, negative depth, or
    // insufficient triangulation angle.
    //
    // @param max_reproj_error    The maximum reprojection error.
    // @param min_tri_angle       The minimum triangulation angle.
    // @param mappoint_ids         The points to be filtered.
    // @return                    The number of filtered observations.
    size_t FilterMapPoints(const double max_reproj_error,
                            const double min_tri_angle,
                            const std::unordered_set<mappoint_t>& mappoint_ids);
    size_t FilterMapPointsInImages(const double max_reproj_error,
                                    const double min_tri_angle,
                                    const std::unordered_set<image_t>& image_ids);
    size_t FilterAllMapPoints(const int min_track_length,
                              const double max_reproj_error,
                              const double min_tri_angle);

    size_t FilterMapPoints(const int min_track_length,
                              const double max_reproj_error,
                              const double min_tri_angle,
                              const std::unordered_set<mappoint_t>& addressed_points);

    // Filter observations that have negative depth.
    // @return    The number of filtered observations.
    size_t FilterObservationsWithNegativeDepth();

    // Filter images without observations or bogus camera parameters.
    // @return    The identifiers of the filtered images.
    std::vector<image_t> FilterImages(const double min_focal_length_ratio,
                                        const double max_focal_length_ratio,
                                        const double max_extra_param);
    
    std::vector<image_t> FilterImages(const double min_focal_length_ratio,
                                        const double max_focal_length_ratio,
                                        const double max_extra_param,
                                        const std::unordered_set<image_t>& addressed_images);                                    

    // Filter images with faraway camera poses, which are definitely wrong. 
    std::vector<image_t> FilterAllFarawayImages();


    // Compute statistics for scene.
    size_t ComputeNumObservations() const;
    double ComputeMeanTrackLength() const;
    double ComputeMeanObservationsPerRegImage() const;
    double ComputeMeanReprojectionError() const;

    // Extract colors for Map Points of given image. Colors will be extracted
    // only for Map Points which are completely black.
    //
    // @param image_id      Identifier of the image for which to extract colors.
    // @param path          Absolute or relative path to root folder of image.
    //                      The image path is determined by concatenating the
    //                      root path and the name of the image.
    //
    // @return              True if image could be read at given path.
    bool ExtractColorsForImage(const image_t image_id, const std::string& path);

    // Extract colors for all Map Points by computing the mean color of all images.
    //
    // @param path          Absolute or relative path to root folder of image.
    //                      The image path is determined by concatenating the
    //                      root path and the name of the image.
    void ExtractColorsForAllImages(const std::string& path);

    void ColorHarmonization(const std::string& path);

    // Read data from text or binary file. Prefer binary data if it exists.
    void ReadReconstruction(const std::string& path, bool camera_rig = false);
    void WriteReconstruction(const std::string &path, bool write_binary = true) const;

    // Read data from binary/text file.
    void ReadText(const std::string& path);
    void ReadBinary(const std::string& path, bool camera_rig = false);


    // Write data from binary/text file.
    void WriteText(const std::string& path) const;
    void WriteBinary(const std::string& path) const;

    void WriteCamerasText(const std::string& path) const;
    void WriteImagesText(const std::string& path) const;
    void WritePoints3DText(const std::string& path) const;
    void WriteKeyFramesText(const std::string& path) const;
    void WritePoseText(const std::string& path) const;
    void WriteLocText(const std::string& path) const;
    void WriteUpdateImagesText(const std::string& path) const;
    void WriteCameraIsRigText(const std::string& path) const;
    void WriteRegisteredImagesInfo(const std::string& path) const;
    void WriteColorCorrectionText(const std::string& path) const;

    void WriteCamerasBinary(const std::string& path) const;
    void WriteLocalCamerasBinary(const std::string& path) const;
    void WriteImagesBinary(const std::string& path) const;
    void WriteLocalImagesBinary(const std::string& path) const;
    void WritePoints3DBinary(const std::string& path) const;
    void WriteCameraIsRigBinary(const std::string& path) const;
    void ExportMapPoints(const std::string& path) const;
    void WriteColorCorrectionBinary(const std::string& path) const;
    void WriteLidarBinary(const std::string& path) const;
    void WriteLocalLidarBinary(const std::string& path) const;

    int GetReconstructionIndex() const;

    void WriteRegistImageData(const std::string &path) const;

    void ReadCamerasText(const std::string& path);
    void ReadImagesText(const std::string& path);
    void ReadKeyFramesText(const std::string& path);
    void ReadPoints3DText(const std::string& path);
    void ReadPoseText(const std::string& path);
    void ReadColorCorrectionText(const std::string& path);

    std::vector<image_t> ReadUpdateImagesText(const std::string& path);
    void ReadCameraIsRigText(const std::string& path);

    void ReadCamerasBinary(const std::string& path, bool camera_rig = false);
    void ReadLocalCamerasBinary(const std::string& path);
    void ReadImagesBinary(const std::string& path);
    void ReadLocalImagesBinary(const std::string& path);
    void ReadPoints3DBinary(const std::string& path);
    void ReadCameraIsRigBinary(const std::string& path);
    void ReadColorCorrectionBinary(const std::string& path);
    void ReadLidarBinary(const std::string& path);
    void ReadLocalLidarBinary(const std::string& path);

    // For alignment pose
    void WriteAlignmentBinary(const std::string& path, const std::unordered_set<image_t>& image_ids) const;
    std::unordered_set<image_t> ReadAlignmentBinary(const std::string& path);

    //FIXME: Transform reconstruction
    void TransformReconstruction(const Eigen::Matrix3x4d transform, bool inverse = true);


    void TranscribeImageIdsToDatabase(const std::shared_ptr<FeatureDataContainer> feature_data_container);
    
    void ComputeTriAngles();
    std::unordered_set<image_t> FindImageForMapPoints(const std::unordered_set<mappoint_t> mappoints);

    void ComputePrimaryPlane(const double max_distance_to_plane,
                             const int max_plane_count);

    void ComputeBaselineDistance();

    bool IsolatedImage(int neighbor_scope,const image_t image_id, double baseline_distance);

    // The primary plane of the camera trajectory, a*x+b*y+c*z+d*w = 0
    Eigen::Vector4d primary_plane;
    double baseline_distance;
    EIGEN_STL_UMAP(image_t, Eigen::Vector4d) planes_for_images;
    void WritePlaneForCameras(const std::string& path, const double max_distance_to_plane, const int max_plane_count);
    EIGEN_STL_UMAP(int, std::vector<Eigen::Vector3d> ) images_for_planes;

    EIGEN_STL_UMAP(sweep_t, Eigen::Vector4d) planes_for_lidars;
    EIGEN_STL_UMAP(int, std::vector<Eigen::Vector3d> ) lidars_for_planes;

    // Prior camera poses from preview system
    bool have_prior_pose = false;
    bool prior_force_keyframe = false;
	std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
    std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

    // RGBD SFM
    bool depth_enabled = true;
    double rgbd_filter_depth_weight = 0.0;
    double rgbd_max_reproj_depth = 0.0;
    bool TryScaleAdjustmentWithDepth(double min_depth_weights);

    // Lidar SFM
    void LoadLidar(const std::vector<RigNames>& rig_list, const std::string& lidar_path);
    void LoadLidar(const std::string& lidar_path);

    void UpdateNeighborsRelatPose(std::unordered_set<image_t>& filter_image_ids);
    // // build global or local point cloud
    // bool GlobalPointCloud();
    // bool LocalPointCloud(const std::unordered_set<image_t>& local_images);
    // Delete all future points of all lidar sweeps.
    inline void ClearCornerMap(); 
    inline void ClearSurfaceMap(); 
    // output point cloud .ply
    bool OutputLidarPointCloud2World(
        const std::string& lidar_path, 
        bool output_frame = false);
    // bool OutputLocalLidarPointCloud2World(const std::string& lidar_path,
    //                                       std::vector<image_t> image_ids = std::vector<image_t>());    

    // Prior GPS/RTK locations for images 
    bool has_gps_prior = false;
    std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
    std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> original_gps_locations;

    // Wheter a gps is valid for aligning the reconstruction
    std::unordered_map<image_t, bool> prior_locations_gps_inlier;
    std::unordered_map<label_t, long> time_offsets_label_map;
    
    // Wheter the latitude and longtitude of a gps is valid 
    std::unordered_map<image_t, bool> prior_horizontal_locations_gps_inlier;

    // Only use latitude and longtitude for optimization or not
    bool optimization_use_horizontal_gps_only = false;

    // The projection of the camera centers on the horizontal plane, which is estimated by assuming the x axis of the
    // camera is horizontal
    std::unordered_map<image_t, Eigen::Vector3d> image_center_projections;
    // The estimated horizontal plane
    Eigen::Vector4d projection_plane;

    // Estimate the horizontal plane
    void ComputeImageCenterProjectionHorizontal();

    // Align the current reconstruction with the prior gps locations
    Eigen::Matrix3x4d AlignWithPriorLocations(double max_error=3.0, double max_error_horizontal = 3.0, long max_gps_time_offset = 60000);
    bool b_aligned = false;
    void OutputPriorResiduals();
    void OutputPriorResidualsTxt(std::string save_path);
    
    void GetGPSTimeOffset(long max_time_ofset= 60000);

    void FilterUselessPoint2D(std::shared_ptr<Reconstruction> reconstruction_out,
                              std::unordered_set<image_t> const_id_set  = std::unordered_set<image_t>());

    void ResetPointStatus();

    void Rescale(const double scale = 1.0);
    void RescaleAll(const double scale = 1.0);

    // copy reconstruction data
    bool Copy(
        const std::unordered_set<image_t> &target_images, 
        const std::unordered_set<mappoint_t> &target_mappoints,
        std::shared_ptr<Reconstruction> res);

    size_t FilterMapPointsWithSpatialDistribution(
        const std::unordered_set<mappoint_t>& mappoint_ids,
        const float spacing_factor  = 2.0);

    size_t FilterMapPointsWithBoundBox(
        const Eigen::Vector3f bb_min, 
        const Eigen::Vector3f bb_max, 
        Eigen::Matrix3f pivot = Eigen::Matrix3f::Identity());

    std::vector<YCrCbFactor> yrb_factors;
    std::unordered_map<image_t, std::vector<image_t>> common_view_map;

    std::unordered_map<std::string, sweep_t> sweep_name_to_id;

    Eigen::Matrix3x4d lidar_to_cam_matrix = Eigen::Matrix3x4d::Identity();

private:
    size_t FilterMapPointsWithSmallTrackLength(const int min_track_length, const std::unordered_set<mappoint_t>& mappoint_ids);
    size_t FilterMapPointsWithSmallTriangulationAngle(const double min_tri_angle, const std::unordered_set<mappoint_t>& mappoint_ids);
    size_t FilterMapPointsWithLargeReprojectionError(const double max_reproj_error, const std::unordered_set<mappoint_t>& mappoint_ids,
    bool verbose = false);
    
    void SetObservationAsTriangulated(const image_t image_id, const point2D_t point2D_idx, 
                                      const bool is_continued_mappoint,
                                      bool verbose = false);
    void ResetTriObservations(const image_t image_id, const point2D_t point2D_idx, const bool is_deleted_mappoint);

private:
    std::shared_ptr<CorrespondenceGraph> correspondence_graph_;
 
    EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
    EIGEN_STL_UMAP(image_t, class Image) images_;
    EIGEN_STL_UMAP(sweep_t, class LidarSweep) lidarsweeps_;
    EIGEN_STL_UMAP(mappoint_t, class MapPoint) mappoints_;

    std::unordered_map<image_pair_t, std::pair<size_t, size_t> > image_pairs_;

    std::vector<image_t> register_image_ids_;

    //  Total number of added Map Points, used to generate unique identifiers.
    mappoint_t num_added_mappoints_;

    std::shared_ptr<class VoxelMap> voxel_map_;

    // Total of lidar sweep.
    std::vector<sweep_t> added_sweep_ids_;
    std::vector<sweep_t> register_sweep_ids_;
    // std::unordered_map<image_t, sweep_t> map_image_2_sweep_;
    std::unordered_map<sweep_t, std::vector<sweep_t>> map_sweep_neighbors_;
    // global lidar point cloud
    LidarPointCloud global_corner_points_;
    LidarPointCloud global_surf_points_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t Reconstruction::NumCameras() const { return cameras_.size(); }

size_t Reconstruction::NumImages() const { return images_.size(); }

size_t Reconstruction::NumRegisterImages() const { return register_image_ids_.size(); }

size_t Reconstruction::NumMapPoints() const { return mappoints_.size(); }

size_t Reconstruction::NumImagePairs() const { return image_pairs_.size(); }

size_t Reconstruction::NumLidarSweep() const { return lidarsweeps_.size(); }

size_t Reconstruction::NumRegisterLidarSweep() const { return register_sweep_ids_.size(); }

const class Camera& Reconstruction::Camera(const camera_t camera_id) const { return cameras_.at(camera_id); }

const class Image& Reconstruction::Image(const image_t image_id) const { return images_.at(image_id); }

const class LidarSweep& Reconstruction::LidarSweep(const sweep_t sweep_id) const { return lidarsweeps_.at(sweep_id); }

const class MapPoint& Reconstruction::MapPoint(const mappoint_t mappoint_id) const { return mappoints_.at(mappoint_id); }

const std::pair<size_t, size_t>& Reconstruction::ImagePair(const image_pair_t pair_id) const { return image_pairs_.at(pair_id); }

const std::pair<size_t, size_t>& Reconstruction::ImagePair(const image_t image_id1, const image_t image_id2) const {
    assert(false);
    const image_pair_t pair_id = utility::ImagePairToPairId(image_id1, image_id2);
    return image_pairs_.at(pair_id);
}

class Camera& Reconstruction::Camera(const camera_t camera_id) { return cameras_.at(camera_id); }

class Image& Reconstruction::Image(const image_t image_id) { return images_.at(image_id); }

class LidarSweep& Reconstruction::LidarSweep(const sweep_t sweep_id) { return lidarsweeps_.at(sweep_id); }

class MapPoint& Reconstruction::MapPoint(const mappoint_t mappoint_id) { return mappoints_.at(mappoint_id); }

std::pair<size_t, size_t>& Reconstruction::ImagePair(const image_pair_t pair_id) { return image_pairs_.at(pair_id); }

std::pair<size_t, size_t>& Reconstruction::ImagePair(const image_t image_id1, const image_t image_id2) {
    assert(false);
    const auto pair_id = utility::ImagePairToPairId(image_id1, image_id2);
    return image_pairs_.at(pair_id);
}

const EIGEN_STL_UMAP(camera_t, class Camera) & Reconstruction::Cameras() const { return cameras_; }

const EIGEN_STL_UMAP(image_t, class Image) & Reconstruction::Images() const { return images_; }

const std::vector<image_t> & Reconstruction::RegisterImageIds() const { return register_image_ids_; }

const std::vector<sweep_t> Reconstruction::RegisterSweepIds() const { return register_sweep_ids_; }

const EIGEN_STL_UMAP(sweep_t, class LidarSweep) & Reconstruction::LidarSweeps() const { return  lidarsweeps_; }

const EIGEN_STL_UMAP(mappoint_t, class MapPoint) & Reconstruction::MapPoints() const { return mappoints_; }

const std::unordered_map<image_pair_t, std::pair<size_t, size_t> > & Reconstruction::ImagePairs() const { return image_pairs_; }

const std::shared_ptr<CorrespondenceGraph> Reconstruction::GetCorrespondenceGraph() const { return correspondence_graph_; }

bool Reconstruction::ExistsCamera(const camera_t camera_id) const { return cameras_.find(camera_id) != cameras_.end(); }

bool Reconstruction::ExistsImage(const image_t image_id) const { return images_.find(image_id) != images_.end(); }

bool Reconstruction::ExistsLidarSweep(const sweep_t sweep_id) const { return lidarsweeps_.find(sweep_id) != lidarsweeps_.end(); }

bool Reconstruction::ExistsMapPoint(const mappoint_t mappoint_id) const { return mappoints_.find(mappoint_id) != mappoints_.end(); }

bool Reconstruction::ExistsImagePair(const image_pair_t pair_id) const { return image_pairs_.find(pair_id) != image_pairs_.end(); }

bool Reconstruction::IsImageRegistered(const image_t image_id) const { return Image(image_id).IsRegistered(); }

bool Reconstruction::SetCamera(const camera_t camera_id, const class Camera &camera) 
{
    if (cameras_.count(camera_id) == 0) return false;

    cameras_[camera_id] = camera;
    return true;
}

class LidarPointCloud& Reconstruction::CornerMap() {return global_corner_points_;}

class LidarPointCloud& Reconstruction::SurfaceMap() {return global_surf_points_;}

void Reconstruction::ClearCornerMap() { std::vector<LidarPoint>().swap(global_corner_points_.points);}

void Reconstruction::ClearSurfaceMap() { std::vector<LidarPoint>().swap(global_surf_points_.points);}

const std::shared_ptr<class VoxelMap>& Reconstruction::VoxelMap() const { return voxel_map_; }

std::shared_ptr<class VoxelMap>& Reconstruction::VoxelMap() { return voxel_map_; }
}
#endif