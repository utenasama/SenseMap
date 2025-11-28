// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

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

#include <iomanip>
#include <algorithm>
#include <string>

#include "reconstruction.h"
#include "util/exception_handler.h"
#include "base/pose.h"
#include "base/similarity_transform.h"
#include "container/scene_graph_container.h"
#include "estimators/plane.h"
#include "estimators/reconstruction_aligner.h"
#include "optim/ransac/loransac.h"
#include "projection.h"
#include "triangulation.h"
#include "util/bitmap.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/rgbd_helper.h"
#include "util/ply.h"
#include "estimators/camera_alignment.h"
#include "util/gps_reader.h"
#include "util/histogram.h"
#include "graph/minimum_spanning_tree.h"
#include "base/cost_functions.h"
#include "util/threading.h"
#include "util/misc.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

namespace sensemap {

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_t;
typedef kernel_t::FT FT;
typedef kernel_t::Point_3 point_3_t;
typedef CGAL::Search_traits_3<kernel_t> tree_traits_3_t;
typedef CGAL::Orthogonal_k_neighbor_search<tree_traits_3_t> neighbor_search_3_t;
typedef neighbor_search_3_t::iterator search_3_iterator_t;
typedef neighbor_search_3_t::Tree tree_3_t;
typedef boost::tuple<int, point_3_t> indexed_point_3_tuple_t;

namespace {
void ComputeMeanStdevDistance(std::vector<FT>& point_spacings, 
                              FT& average_spacing, FT& stdev_spacing,
                              const std::vector<point_3_t> &points_3,
                              const unsigned int nb_neighbors = 6){
    average_spacing = (FT)0.0;
    stdev_spacing = (FT)0.0;
    std::size_t point_num = points_3.size();
    point_spacings.resize(point_num);

    // Instantiate a KD-tree search.
    tree_3_t tree(points_3.begin(), points_3.end());
#ifdef CGAL_LINKED_WITH_TBB
    tree.build<CGAL::Parallel_tag>();
#endif

#ifndef CGAL_LINKED_WITH_TBB
    std::cout << "Starting average spacing computation ..." << std::endl;
    // iterate over input points, compute and output point spacings
    for (std::size_t i = 0; i < point_num; i++) {
        const auto &query = points_3[i];
        // performs k + 1 queries (if unique the query point is
        // output first). search may be aborted when k is greater
        // than number of input points
        neighbor_search_3_t search(tree, query, nb_neighbors + 1);
        auto &point_spacing = point_spacings[i];
        point_spacing = (FT)0.0;
        std::size_t k = 0;
        for (search_3_iterator_t search_iterator = search.begin(); search_iterator != search.end() && k <= nb_neighbors; search_iterator++, k++)
        {
            point_spacing += std::sqrt(search_iterator->second);
        }
        // output point spacing
        if (k > 1) {
            point_spacing /= (FT)(k - 1);
        }

        average_spacing += point_spacing;
    }
#else
    std::cout << "Starting average spacing computation(Parallel) ..." << std::endl;
    tbb::parallel_for (tbb::blocked_range<std::size_t> (0, point_num),
                     [&](const tbb::blocked_range<std::size_t>& r) {
                       for (std::size_t s = r.begin(); s != r.end(); ++ s) {
                            const auto &query = points_3[s];
                            // performs k + 1 queries (if unique the query point is
                            // output first). search may be aborted when k is greater
                            // than number of input points
                            neighbor_search_3_t search(tree, query, nb_neighbors + 1);
                            auto &point_spacing = point_spacings[s];
                            point_spacing = (FT)0.0;
                            std::size_t k = 0;
                            for (search_3_iterator_t search_iterator = search.begin(); search_iterator != search.end() && k <= nb_neighbors; search_iterator++, k++)
                            {
                                point_spacing += std::sqrt(search_iterator->second);
                            }
                            // output point spacing
                            if (k > 1) {
                                point_spacing /= (FT)(k - 1);
                            }
                       }
                     });
    for (auto & point_spacing : point_spacings) {
        average_spacing += point_spacing;
    }
#endif
    average_spacing /= (FT)point_num;

    for (std::size_t i = 0; i < point_num; i++){
        stdev_spacing += (point_spacings[i] - average_spacing) * (point_spacings[i] - average_spacing);
    }
    stdev_spacing = std::sqrt(stdev_spacing / (point_num - 1));
    // std::cout << " =>Average spacing: " << average_spacing << std::endl;
    return;
}

Eigen::Matrix3f PovitMatrix(
    const std::vector<Eigen::Vector3f> &points){
    Eigen::Matrix3f pivot;
    Eigen::Vector3f centroid(Eigen::Vector3f::Zero());
    for (const auto &point : points) {
        centroid += point;
    }
    std::size_t point_num = points.size();
    centroid /= point_num;

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; ++i) {
        for (const auto &point : points) {
            M(i, 0) += (point[i] - centroid[i]) * (point[0] - centroid[i]);
            M(i, 1) += (point[i] - centroid[i]) * (point[1] - centroid[i]);
            M(i, 2) += (point[i] - centroid[i]) * (point[2] - centroid[i]);
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    pivot = svd.matrixU().transpose();
    return pivot;
}

void PrintSolverSummary(const ceres::Solver::Summary &summary) {
    std::cout << std::right << std::setw(16) << "Residuals : ";
    std::cout << std::left << summary.num_residuals_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Parameters : ";
    std::cout << std::left << summary.num_effective_parameters_reduced
              << std::endl;

    std::cout << std::right << std::setw(16) << "Iterations : ";
    std::cout << std::left
              << summary.num_successful_steps + summary.num_unsuccessful_steps
              << std::endl;

    std::cout << std::right << std::setw(16) << "Time : ";
    std::cout << std::left << summary.total_time_in_seconds << " [s]"
              << std::endl;

    std::cout << std::right << std::setw(16) << "Initial cost : ";
    std::cout << std::right << std::setprecision(6)
              << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
              << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "Final cost : ";
    std::cout << std::right << std::setprecision(6)
              << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
              << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "Termination : ";

    std::string termination = "";

    switch (summary.termination_type) {
        case ceres::CONVERGENCE:
            termination = "Convergence";
            break;
        case ceres::NO_CONVERGENCE:
            termination = "No convergence";
            break;
        case ceres::FAILURE:
            termination = "Failure";
            break;
        case ceres::USER_SUCCESS:
            termination = "User success";
            break;
        case ceres::USER_FAILURE:
            termination = "User failure";
            break;
        default:
            termination = "Unknown";
            break;
    }

    std::cout << std::right << termination << std::endl;
    std::cout << std::endl;
}

}

Reconstruction::Reconstruction() : correspondence_graph_(nullptr), num_added_mappoints_(0) {}

std::unordered_set<mappoint_t> Reconstruction::MapPointIds() const {
    std::unordered_set<mappoint_t> mappoint_ids;
    mappoint_ids.reserve(mappoints_.size());

    for (const auto& mappoint : mappoints_) {
        mappoint_ids.insert(mappoint.first);
    }

    return mappoint_ids;
}

const std::vector<image_t> Reconstruction::RegisterImageIds(const std::vector<image_t> image_ids) const {
    std::vector<image_t> include_image_ids;
    include_image_ids.reserve(image_ids.size());
    std::unordered_set<image_t> all_image_ids_set(register_image_ids_.begin(), register_image_ids_.end());
    for (const auto& image_id : image_ids) {
        if (all_image_ids_set.count(image_id)) {
            include_image_ids.emplace_back(image_id);
        }
    }
    return include_image_ids;
}

// void Reconstruction::SetUp(const std::shared_ptr<CorrespondenceGraph> correspondence_graph) {
//     CHECK_NOTNULL(correspondence_graph);
//     for (auto & image : images_) {
//         image.second.SetUp(Camera(image.second.CameraId()));
//     }
//     correspondence_graph_ = correspondence_graph;

//     for (const auto image_id : register_image_ids_) {
//         const class Image & image = Image(image_id);
//         for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
//             if (image.Point2D(point2D_idx).HasMapPoint()) {
//                 const bool kIsContinuedMapPoint = false;
//                 SetObservationAsTriangulated(image_id, point2D_idx, kIsContinuedMapPoint);
//             }
//         }
//     }
// }

void Reconstruction::SetUp(const std::shared_ptr<SceneGraphContainer> scene_graph_container) {
    CHECK_NOTNULL(scene_graph_container->CorrespondenceGraph().get());

    // Add cameras.
    cameras_.reserve(scene_graph_container->NumCameras());
    for (const auto& camera : scene_graph_container->Cameras()) {
        if (!ExistsCamera(camera.first)) {
            AddCamera(camera.second);
        }
    }

    // Add images.
    images_.reserve(scene_graph_container->NumImages());
    for (const auto& image : scene_graph_container->Images()) {
        if (ExistsImage(image.second.ImageId())) {
            class Image& existing_image = Image(image.second.ImageId());

            if (existing_image.NumPoints2D() == 0) {
                existing_image.SetPoints2D(image.second.Points2D());
                existing_image.SetLocalImageIndices(image.second.LocalImageIndices());
            } else {
                CHECK_EQ(image.second.NumPoints2D(), existing_image.NumPoints2D());
                existing_image.SetLocalImageIndices(image.second.LocalImageIndices());
            }
            if (!existing_image.HasLabel()) {
                existing_image.SetLabelId(image.second.LabelId());
            }
            existing_image.SetNumObservations(image.second.NumObservations());
            existing_image.SetNumCorrespondences(image.second.NumCorrespondences());

            existing_image.RtkFlag() = image.second.RtkFlag();
            if (image.second.HasQvecPrior()){
                existing_image.SetQvecPrior(image.second.QvecPrior());
            }
            if (image.second.HasTvecPrior()){
                existing_image.SetTvecPrior(image.second.TvecPrior());
            }
            if (image.second.HasRtkStd()){
                existing_image.SetRtkStd(image.second.RtkStdLon(), image.second.RtkStdLat(), image.second.RtkStdHgt());
            }
            if (image.second.HasOrientStd()){
                existing_image.SetOrientStd(image.second.OrientStd());
            }
        } else {
            AddImage(image.second);
        }
    }

    // Add image pairs.
    for (const auto& image_pair : scene_graph_container->CorrespondenceGraph()->NumCorrespondencesBetweenImages()) {
        image_pairs_[image_pair.first] = std::make_pair(0, image_pair.second);
    }

    correspondence_graph_ = scene_graph_container->CorrespondenceGraph();

    for (auto& image : images_) {
        image.second.SetUp(Camera(image.second.CameraId()));
    }

    for (const auto image_id : register_image_ids_) {
        const class Image& image = Image(image_id);
        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
            if (image.Point2D(point2D_idx).HasMapPoint()) {
                const bool kIsContinuedMapPoint = false;
                SetObservationAsTriangulated(image_id, point2D_idx, kIsContinuedMapPoint);
            }
        }
    }

    const auto & mappoint_ids = MapPointIds();
    mappoint_t max_mappoint_id = 0;
    for (auto mappoint_id : mappoint_ids) {
        max_mappoint_id = std::max(max_mappoint_id, mappoint_id);
    }
    num_added_mappoints_ = max_mappoint_id;
}

std::unordered_map<image_t, std::vector<image_t>> 
Reconstruction::ConvertRigReconstruction(Reconstruction& reconstruction) {
    // Convert Camera.
    std::unordered_map<camera_t, std::vector<camera_t> > camera_ids_map;
    size_t local_camera_id = 1;
    for (const auto& camera : cameras_) {
        int num_local_cameras = camera.second.NumLocalCameras();
        const bool exist_camera = camera_ids_map.count(camera.first) != 0 &&
            camera_ids_map.at(camera.first).size() == num_local_cameras;
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
            if (exist_camera) {
                local_camera.SetCameraId(camera_ids_map.at(camera.first)[camera_id]);
            } else {
                local_camera.SetCameraId(local_camera_id);
            }
            local_camera.SetModelIdFromName(model_name);
            local_camera.SetWidth(width);
            local_camera.SetHeight(height);
            local_camera.SetNumLocalCameras(1);
            local_camera.SetParams(params);

            reconstruction.AddCamera(local_camera);
            if (!exist_camera) {
                camera_ids_map[camera.first].emplace_back(local_camera_id);
                local_camera_id++;
            }
        }
    }

    std::unordered_map<image_t, std::vector<image_t>> image_ids_map;
    EIGEN_STL_UMAP(mappoint_t, class MapPoint) rig_mappoints_;
    // Convert Image.
    image_t image_rig_id = 1;
    const auto image_sort_ids = RegisterImageSortIds();
    for (const auto image_id : image_sort_ids) {
        const auto& image = images_.at(image_id);
        if (!image.IsRegistered()) {
            continue;
        }
        const class Camera& camera = cameras_.at(image.CameraId());
        const size_t num_local_camera = camera.NumLocalCameras();
        const std::vector<Eigen::Vector4d> & local_qvec_priors = image.LocalQvecsPrior();
        const std::vector<Eigen::Vector3d> & local_tvec_priors = image.LocalTvecsPrior();

        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        for (local_camera_t local_id = 0; local_id < num_local_camera; local_id++){
            Eigen::Vector4d normalized_qvec;
            Eigen::Vector3d normalized_tvec;
            Eigen::Vector4d qvec_prior;
            Eigen::Vector3d tvec_prior;
            if (num_local_camera > 1) {
                camera.GetLocalCameraExtrinsic(local_id, local_qvec, local_tvec);

                normalized_qvec =
                NormalizeQuaternion(RotationMatrixToQuaternion(
                QuaternionToRotationMatrix(local_qvec) * QuaternionToRotationMatrix(image.Qvec())));

                normalized_tvec =
                QuaternionToRotationMatrix(local_qvec)*image.Tvec() + local_tvec;

                if (local_tvec_priors.size() > 0) {
                    tvec_prior = local_tvec_priors[local_id];
                }
                if (local_qvec_priors.size() > 0) {
                    qvec_prior = local_qvec_priors[local_id];
                }
            } else {
                normalized_qvec = image.Qvec();
                normalized_tvec = image.Tvec();

                qvec_prior = image.QvecPrior();
                tvec_prior = image.TvecPrior();
            }

            std::string image_name = image.Name();
            if (num_local_camera > 1 && image.HasLocalName(local_id)) {
                image_name = image.LocalName(local_id);
            } else {
                auto pos = image_name.find("cam0", 0);
                if (pos != std::string::npos) {
                    image_name.replace(pos, 4, "cam" + std::to_string(local_id));
                }
            }

            std::vector<Point2D> local_point2Ds;
            point2D_t rig_point2D_idx = 0;
            for(size_t i = 0; i< image.Points2D().size(); ++i){    
                Point2D point2D = image.Points2D()[i];
                local_camera_t local_camera_id = image.LocalImageIndices()[i];
                if (local_camera_id == local_id){
                    if (point2D.HasMapPoint()) {
                        if (rig_mappoints_.find(point2D.MapPointId()) == rig_mappoints_.end()){
                            class MapPoint mappoint_rig = 
                                mappoints_.at(point2D.MapPointId());
                            class TrackElement rig_trackelement(image_rig_id, rig_point2D_idx);
                            class Track rig_track;
                            rig_track.AddElement(rig_trackelement);
                            mappoint_rig.SetTrack(rig_track);
                            rig_mappoints_.emplace(point2D.MapPointId(), mappoint_rig);
                        } else{
                            class TrackElement rig_trackelement(image_rig_id, rig_point2D_idx);
                            rig_mappoints_.at(point2D.MapPointId()).Track().AddElement(rig_trackelement);
                        }
                    }
                    point2D.SetMapPointId(kInvalidMapPointId);
                    local_point2Ds.push_back(point2D);
                    rig_point2D_idx++;
                }
            }

            std::vector<uint32_t> new_image_indices(local_point2Ds.size());
            std::fill(new_image_indices.begin(), new_image_indices.end(), image_rig_id);

            // size_t local_camera_id = (image.second.CameraId()-1) * num_local_camera + local_id + 1;
            size_t local_camera_id = camera_ids_map.at(camera.CameraId())[local_id];

            class Image local_image;
            local_image.SetCameraId(local_camera_id);
            local_image.SetImageId(image_rig_id);
            local_image.SetName(image_name);
            local_image.SetQvec(normalized_qvec);
            local_image.SetTvec(normalized_tvec);
            if (image.HasQvecPrior()) {
                local_image.SetQvecPrior(qvec_prior);
            }
            if (image.HasTvecPrior()) {
                local_image.SetTvecPrior(tvec_prior);
            }
            // local_image.SetRegistered(true);
            local_image.SetPoseFlag(true);
            local_image.SetPoints2D(local_point2Ds);
            local_image.SetLocalImageIndices(new_image_indices);
            local_image.SetLabelId(image.LabelId());
            reconstruction.AddImage(local_image);
            reconstruction.RegisterImage(image_rig_id);

            image_ids_map[image_id].emplace_back(image_rig_id);
            image_rig_id++;
        }
    }

    // Convert MapPoint.
    for (auto mappoint : rig_mappoints_) {
        reconstruction.AddMapPoint(mappoint.first, 
                                   mappoint.second.XYZ(), 
                                   mappoint.second.Track(),
                                   mappoint.second.Color());
    }

    reconstruction.has_gps_prior = has_gps_prior;
    return image_ids_map;
}

void Reconstruction::TearDown() {
    // Note: The correspondence graph has been clear during tear down
    correspondence_graph_ = std::shared_ptr<CorrespondenceGraph>();

    // Remove all not yet registered images.
    std::unordered_set<camera_t> keep_camera_ids;
    for (auto it = images_.begin(); it != images_.end();) {
        if (it->second.IsRegistered()) {
            keep_camera_ids.insert(it->second.CameraId());
            it->second.TearDown();
            ++it;
        } else {
            it = images_.erase(it);
        }
    }

    // Remove all unused cameras.
    for (auto it = cameras_.begin(); it != cameras_.end();) {
        if (keep_camera_ids.count(it->first) == 0) {
            it = cameras_.erase(it);
        } else {
            ++it;
        }
    }

    // Compress tracks.
    for (auto& mappoint : mappoints_) {
        mappoint.second.Track().Compress();
    }
}

void Reconstruction::AddCamera(const class Camera& camera) {
    CHECK(!ExistsCamera(camera.CameraId()));
    CHECK(camera.VerifyParams());
    // cameras_.emplace(camera.CameraId(), camera);
    cameras_[camera.CameraId()] = camera;
}

void Reconstruction::AddImage(const class Image& image) {
    CHECK(!ExistsImage(image.ImageId()));
    images_[image.ImageId()] = image;
}

void Reconstruction::AddImageNohasMapPoint(const class Image& image) {
    CHECK(!ExistsImage(image.ImageId()));
    images_[image.ImageId()] = image;
    
    for (point2D_t i = 0; i < images_[image.ImageId()].NumPoints2D(); i++){
        images_[image.ImageId()].ResetMapPointForPoint2D(i);
    }

    if (image.IsRegistered()){
        register_image_ids_.push_back(image.ImageId());
    }
}

bool Reconstruction::AddMapPoint(mappoint_t mappoint_id, const Eigen::Vector3d& xyz, Track track,
                                 const Eigen::Vector3ub& color, bool verbose) {
    if (ExistsMapPoint(mappoint_id)) {
        return false;
    }

    num_added_mappoints_ = std::max(num_added_mappoints_, mappoint_id);
    class MapPoint& mappoint = mappoints_[mappoint_id];

    for (const auto& track_el : track.Elements()) {
        class Image& image = Image(track_el.image_id);
        CHECK(!image.Point2D(track_el.point2D_idx).HasMapPoint());
        image.SetMapPointForPoint2D(track_el.point2D_idx, mappoint_id);
        CHECK_LE(image.NumMapPoints(), image.NumPoints2D());
    }

    const bool kIsContinuedMapPoint = false;

    for (const auto& track_el : track.Elements()) {
        SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx, kIsContinuedMapPoint, verbose);
    }

    mappoint.SetXYZ(xyz);
    mappoint.SetTrack(std::move(track));
    mappoint.SetColor(color);
    if (mappoint.CreateTime() == 0) {
        mappoint.SetCreateTime(NumRegisterImages());
    }

    return true;
}

mappoint_t Reconstruction::AddMapPoint(const Eigen::Vector3d& xyz, 
                                       Track track, 
                                       const Eigen::Vector3ub& color,
                                       bool verbose) {
#if 1
    const mappoint_t mappoint_id = ++num_added_mappoints_;
    CHECK(!ExistsMapPoint(mappoint_id));

    class MapPoint& mappoint = mappoints_[mappoint_id];

    for (const auto& track_el : track.Elements()) {
        class Image& image = Image(track_el.image_id);
        CHECK(!image.Point2D(track_el.point2D_idx).HasMapPoint());
        image.SetMapPointForPoint2D(track_el.point2D_idx, mappoint_id);
        CHECK_LE(image.NumMapPoints(), image.NumPoints2D());
    }

    const bool kIsContinuedMapPoint = false;

    for (const auto& track_el : track.Elements()) {
        SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx, kIsContinuedMapPoint, verbose);
    }

    mappoint.SetXYZ(xyz);
    mappoint.SetTrack(std::move(track));
    mappoint.SetColor(color);
    if (mappoint.CreateTime() == 0) {
        mappoint.SetCreateTime(NumRegisterImages());
    }
    return mappoint_id;
#else
    const std::vector<TrackElement>& elems = track.Elements();
    const bool exist_mappoint = std::find_if(elems.begin(), elems.end(), 
        [&](const TrackElement track_el) {
            class Image& image = Image(track_el.image_id);
            return image.Point2D(track_el.point2D_idx).HasMapPoint();
        }) != elems.end();

    if (!exist_mappoint) {
        const mappoint_t mappoint_id = ++num_added_mappoints_;
        CHECK(!ExistsMapPoint(mappoint_id));

        class MapPoint& mappoint = mappoints_[mappoint_id];

        for (const auto& track_el : track.Elements()) {
            class Image& image = Image(track_el.image_id);
            CHECK(!image.Point2D(track_el.point2D_idx).HasMapPoint());
            image.SetMapPointForPoint2D(track_el.point2D_idx, mappoint_id);
            CHECK_LE(image.NumMapPoints(), image.NumPoints2D());
        }

        const bool kIsContinuedMapPoint = false;

        for (const auto& track_el : track.Elements()) {
            SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx, kIsContinuedMapPoint, verbose);
        }

        mappoint.SetXYZ(xyz);
        mappoint.SetTrack(std::move(track));
        mappoint.SetColor(color);

        return mappoint_id;
    } else {
        mappoint_t exist_id = kInvalidMapPointId;
        for (const auto& track_el : track.Elements()) {
            class Image& image = Image(track_el.image_id);
            class Point2D& point2D = image.Point2D(track_el.point2D_idx);
            if (point2D.HasMapPoint()) {
                exist_id = point2D.MapPointId();
                break;
            }
        }
        
        class MapPoint& exist_mappoint = MapPoint(exist_id);
        class Track& exist_track = exist_mappoint.Track();
        for (const auto& track_el : track.Elements()) {
            class Image& image = Image(track_el.image_id);
            if (!image.Point2D(track_el.point2D_idx).HasMapPoint()) {
                exist_track.AddElement(track_el);

                image.SetMapPointForPoint2D(track_el.point2D_idx, exist_id);
                SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx, false);
            }
        }
        exist_mappoint.SetTrack(exist_track);
        return exist_id;
    }
#endif
}

mappoint_t Reconstruction::AddMapPointWithError(const Eigen::Vector3d& xyz, Track track,
                                                const Eigen::Vector3ub& color, double error) {
    const mappoint_t mappoint_id = ++num_added_mappoints_;
    CHECK(!ExistsMapPoint(mappoint_id));

    class MapPoint& mappoint = mappoints_[mappoint_id];

    for (const auto& track_el : track.Elements()) {
        class Image& image = Image(track_el.image_id);
        CHECK(!image.Point2D(track_el.point2D_idx).HasMapPoint());
        image.SetMapPointForPoint2D(track_el.point2D_idx, mappoint_id);
        CHECK_LE(image.NumMapPoints(), image.NumPoints2D());
    }

    const bool kIsContinuedMapPoint = false;

    for (const auto& track_el : track.Elements()) {
        SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx, kIsContinuedMapPoint);
    }

    mappoint.SetXYZ(xyz);
    mappoint.SetTrack(std::move(track));
    mappoint.SetColor(color);
    mappoint.SetError(error);
    if (mappoint.CreateTime() == 0) {
        mappoint.SetCreateTime(NumRegisterImages());
    }
    return mappoint_id;
}

bool Reconstruction::AddMapPointWithError(mappoint_t mappoint_id, 
                                 const Eigen::Vector3d& xyz, Track track,
                                 double error, const Eigen::Vector3ub& color, 
                                 bool verbose) {
    if (ExistsMapPoint(mappoint_id)) {
        return false;
    }

    class MapPoint& mappoint = mappoints_[mappoint_id];

    for (const auto& track_el : track.Elements()) {
        class Image& image = Image(track_el.image_id);
        CHECK(!image.Point2D(track_el.point2D_idx).HasMapPoint());
        image.SetMapPointForPoint2D(track_el.point2D_idx, mappoint_id);
        CHECK_LE(image.NumMapPoints(), image.NumPoints2D());
    }

    // const bool kIsContinuedMapPoint = false;

    // for (const auto& track_el : track.Elements()) {
    //     SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx, kIsContinuedMapPoint, verbose);
    // }
    mappoint.SetXYZ(xyz);
    mappoint.SetTrack(std::move(track));
    mappoint.SetColor(color);
    mappoint.SetError(error);
    if (mappoint.CreateTime() == 0) {
        mappoint.SetCreateTime(NumRegisterImages());
    }
    return true;
}

void Reconstruction::AddObservation(const mappoint_t mappoint_id, const TrackElement& track_el) {
    class Image& image = Image(track_el.image_id);
    CHECK(!image.Point2D(track_el.point2D_idx).HasMapPoint());

    image.SetMapPointForPoint2D(track_el.point2D_idx, mappoint_id);
    CHECK_LE(image.NumMapPoints(), image.NumPoints2D());

    class MapPoint& mappoint = MapPoint(mappoint_id);
    mappoint.Track().AddElement(track_el);

    const bool kIsContinuedMapPoint = true;
    SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx, kIsContinuedMapPoint);
}

mappoint_t Reconstruction::MergeMapPoints(const mappoint_t mappoint_id1, const mappoint_t mappoint_id2) {
    // Check the map point existance
    CHECK(ExistsMapPoint(mappoint_id1));
    CHECK(ExistsMapPoint(mappoint_id2));

    const class MapPoint& mappoint1 = MapPoint(mappoint_id1);
    const class MapPoint& mappoint2 = MapPoint(mappoint_id2);

    const size_t track_len1 = mappoint1.Track().Length();
    const size_t track_len2 = mappoint2.Track().Length();
    // TODO: Use the length of track as average weight
    const Eigen::Vector3d merged_xyz =
        (track_len1 * mappoint1.XYZ() + track_len2 * mappoint2.XYZ()) / (track_len1 + track_len2);
    const Eigen::Vector3d merged_rgb =
        (track_len1 * mappoint1.Color().cast<double>() + track_len2 * mappoint2.Color().cast<double>()) /
        (track_len1 + track_len2);

    Track merged_track;
    merged_track.Reserve(track_len1 + track_len2);
    merged_track.AddElements(mappoint1.Track().Elements());  // -- Merge tracker
    merged_track.AddElements(mappoint2.Track().Elements());

    DeleteMapPoint(mappoint_id1);
    DeleteMapPoint(mappoint_id2);

    const mappoint_t merged_mappoint_id = AddMapPoint(merged_xyz, std::move(merged_track), merged_rgb.cast<uint8_t>());
    return merged_mappoint_id;
}

void Reconstruction::DeleteMapPoint(const mappoint_t mappoint_id) {
    const class Track& track = MapPoint(mappoint_id).Track();

    const bool kIsDeletedMapPoint = true;

    for (const auto& track_el : track.Elements()) {
	    if (ExistsImage(track_el.image_id) == false) continue;
        ResetTriObservations(track_el.image_id, track_el.point2D_idx, kIsDeletedMapPoint);
    }
    for (const auto& track_el : track.Elements()) {
	    if (ExistsImage(track_el.image_id) == false) continue;
        class Image& image = Image(track_el.image_id);
        image.ResetMapPointForPoint2D(track_el.point2D_idx);
    }

    mappoints_.erase(mappoint_id);
}

void Reconstruction::DeleteObservation(const image_t image_id, const point2D_t point2D_idx) {
    class Image& image = Image(image_id);
    const mappoint_t mappoint_id = image.Point2D(point2D_idx).MapPointId();
    class MapPoint& mappoint = MapPoint(mappoint_id);

    if (mappoint.Track().Length() <= 2) {
        DeleteMapPoint(mappoint_id);
        return;
    }

    mappoint.Track().DeleteElement(image_id, point2D_idx);

    const bool kIsDeletedMapPoint = false;
    ResetTriObservations(image_id, point2D_idx, kIsDeletedMapPoint);

    image.ResetMapPointForPoint2D(point2D_idx);
}

void Reconstruction::DeleteAllPoints2DAndPoints3D() {
    mappoints_.clear();
    for (auto& image : images_) {
        class Image new_image;
        new_image.SetImageId(image.second.ImageId());
        new_image.SetName(image.second.Name());
        new_image.SetCameraId(image.second.CameraId());
        new_image.SetRegistered(image.second.IsRegistered());
        new_image.SetNumCorrespondences(image.second.NumCorrespondences());
        new_image.SetQvec(image.second.Qvec());
        new_image.SetQvecPrior(image.second.QvecPrior());
        new_image.SetTvec(image.second.Tvec());
        new_image.SetTvecPrior(image.second.TvecPrior());
        image.second = new_image;
    }
}

void Reconstruction::RegisterImage(const image_t image_id) {
    class Image& image = Image(image_id);
    if (!image.IsRegistered()) {
        image.SetRegistered(true);
        image.SetPoseFlag(true);
        register_image_ids_.push_back(image_id);
        image.create_time_ = this->NumRegisterImages();
    }
}

void Reconstruction::DeRegisterImage(const image_t image_id) {
    class Image& image = Image(image_id);

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        if (image.Point2D(point2D_idx).HasMapPoint()) {
            DeleteObservation(image_id, point2D_idx);
        }
    }

    image.SetRegistered(false);
    image.SetPoseFlag(false);

    register_image_ids_.erase(std::remove(register_image_ids_.begin(), register_image_ids_.end(), image_id),
                              register_image_ids_.end());
}

void Reconstruction::RegisterLidarSweep(const sweep_t sweep_id) {
    class LidarSweep & lidar_sweep = LidarSweep(sweep_id);
    if (!lidar_sweep.IsRegistered()) {
        lidar_sweep.SetRegistered(true);
        register_sweep_ids_.push_back(sweep_id);
    }
}

void Reconstruction::DeRegisterLidarSweep(const sweep_t sweep_id) {
    class LidarSweep & lidar_sweep = LidarSweep(sweep_id);
    lidar_sweep.SetRegistered(false);
    register_sweep_ids_.erase(std::remove(register_sweep_ids_.begin(), register_sweep_ids_.end(), sweep_id),
                              register_sweep_ids_.end());

}

void Reconstruction::Normalize(const double extent, const double p0, const double p1, const bool use_images) {
    CHECK_GT(extent, 0);
    CHECK_GE(p0, 0);
    CHECK_LE(p0, 1);
    CHECK_GE(p1, 0);
    CHECK_LE(p1, 1);
    CHECK_LE(p0, p1);

    if ((use_images && register_image_ids_.size() < 2) || (!use_images && mappoints_.size() < 2)) {
        return;
    }

    EIGEN_STL_UMAP(class Image*, Eigen::Vector3d)
    proj_centers;

    for (size_t i = 0; i < register_image_ids_.size(); ++i) {
        class Image& image = Image(register_image_ids_[i]);
        const Eigen::Vector3d proj_center = image.ProjectionCenter();
        proj_centers[&image] = proj_center;
    }

    // Coordinates of image centers or point locations.
    std::vector<float> coords_x;
    std::vector<float> coords_y;
    std::vector<float> coords_z;
    if (use_images) {
        coords_x.reserve(proj_centers.size());
        coords_y.reserve(proj_centers.size());
        coords_z.reserve(proj_centers.size());
        for (const auto& proj_center : proj_centers) {
            coords_x.push_back(static_cast<float>(proj_center.second(0)));
            coords_y.push_back(static_cast<float>(proj_center.second(1)));
            coords_z.push_back(static_cast<float>(proj_center.second(2)));
        }
    } else {
        coords_x.reserve(mappoints_.size());
        coords_y.reserve(mappoints_.size());
        coords_z.reserve(mappoints_.size());
        for (const auto& point3D : mappoints_) {
            coords_x.push_back(static_cast<float>(point3D.second.X()));
            coords_y.push_back(static_cast<float>(point3D.second.Y()));
            coords_z.push_back(static_cast<float>(point3D.second.Z()));
        }
    }

    // Determine robust bounding box and mean.

    std::sort(coords_x.begin(), coords_x.end());
    std::sort(coords_y.begin(), coords_y.end());
    std::sort(coords_z.begin(), coords_z.end());

    const size_t P0 = static_cast<size_t>((coords_x.size() > 3) ? p0 * (coords_x.size() - 1) : 0);
    const size_t P1 = static_cast<size_t>((coords_x.size() > 3) ? p1 * (coords_x.size() - 1) : coords_x.size() - 1);

    const Eigen::Vector3d bbox_min(coords_x[P0], coords_y[P0], coords_z[P0]);
    const Eigen::Vector3d bbox_max(coords_x[P1], coords_y[P1], coords_z[P1]);

    Eigen::Vector3d mean_coord(0, 0, 0);
    for (size_t i = P0; i <= P1; ++i) {
        mean_coord(0) += coords_x[i];
        mean_coord(1) += coords_y[i];
        mean_coord(2) += coords_z[i];
    }
    mean_coord /= (P1 - P0 + 1);

    // Calculate scale and translation, such that
    // translation is applied before scaling.
    const double old_extent = (bbox_max - bbox_min).norm();
    double scale;
    if (old_extent < std::numeric_limits<double>::epsilon()) {
        scale = 1;
    } else {
        scale = extent / old_extent;
    }

    const Eigen::Vector3d translation = mean_coord;

    // Transform images.
    for (auto& image_proj_center : proj_centers) {
        image_proj_center.second -= translation;
        image_proj_center.second *= scale;
        const Eigen::Quaterniond quat(image_proj_center.first->Qvec(0), image_proj_center.first->Qvec(1),
                                      image_proj_center.first->Qvec(2), image_proj_center.first->Qvec(3));
        image_proj_center.first->SetTvec(quat * -image_proj_center.second);
    }

    // Transform points.
    for (auto& point3D : mappoints_) {
        point3D.second.XYZ() -= translation;
        point3D.second.XYZ() *= scale;
    }

    for(auto& camera: cameras_){

        if(camera.second.NumLocalCameras()>1){
            std::vector<Eigen::Vector4d> local_qvecs;
            std::vector<Eigen::Vector3d> local_tvecs;
            for(size_t i = 0; i < camera.second.NumLocalCameras(); ++i){
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                camera.second.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
                local_tvec = scale * local_tvec;
                local_qvecs.push_back(local_qvec);
                local_tvecs.push_back(local_tvec);
            }
            camera.second.SetLocalCameraExtrinsics(local_qvecs,local_tvecs);
        }
    }
}

Eigen::Matrix3x4d Reconstruction::NormalizeWoScale(const double extent, 
    const double p0, const double p1, const bool use_images) {
    CHECK_GT(extent, 0);
    CHECK_GE(p0, 0);
    CHECK_LE(p0, 1);
    CHECK_GE(p1, 0);
    CHECK_LE(p1, 1);
    CHECK_LE(p0, p1);

    Eigen::Matrix3x4d M = Eigen::Matrix3x4d::Identity();

    if ((use_images && register_image_ids_.size() < 2) || (!use_images && mappoints_.size() < 2)) {
        return M;
    }

    EIGEN_STL_UMAP(class Image*, Eigen::Vector3d)
    proj_centers;

    for (size_t i = 0; i < register_image_ids_.size(); ++i) {
        class Image& image = Image(register_image_ids_[i]);
        const Eigen::Vector3d proj_center = image.ProjectionCenter();
        proj_centers[&image] = proj_center;
    }

    // Coordinates of image centers or point locations.
    std::vector<float> coords_x;
    std::vector<float> coords_y;
    std::vector<float> coords_z;
    if (use_images) {
        coords_x.reserve(proj_centers.size());
        coords_y.reserve(proj_centers.size());
        coords_z.reserve(proj_centers.size());
        for (const auto& proj_center : proj_centers) {
            coords_x.push_back(static_cast<float>(proj_center.second(0)));
            coords_y.push_back(static_cast<float>(proj_center.second(1)));
            coords_z.push_back(static_cast<float>(proj_center.second(2)));
        }
    } else {
        coords_x.reserve(mappoints_.size());
        coords_y.reserve(mappoints_.size());
        coords_z.reserve(mappoints_.size());
        for (const auto& point3D : mappoints_) {
            coords_x.push_back(static_cast<float>(point3D.second.X()));
            coords_y.push_back(static_cast<float>(point3D.second.Y()));
            coords_z.push_back(static_cast<float>(point3D.second.Z()));
        }
    }

    // Determine robust bounding box and mean.

    std::sort(coords_x.begin(), coords_x.end());
    std::sort(coords_y.begin(), coords_y.end());
    std::sort(coords_z.begin(), coords_z.end());

    const size_t P0 = static_cast<size_t>((coords_x.size() > 3) ? p0 * (coords_x.size() - 1) : 0);
    const size_t P1 = static_cast<size_t>((coords_x.size() > 3) ? p1 * (coords_x.size() - 1) : coords_x.size() - 1);

    const Eigen::Vector3d bbox_min(coords_x[P0], coords_y[P0], coords_z[P0]);
    const Eigen::Vector3d bbox_max(coords_x[P1], coords_y[P1], coords_z[P1]);

    Eigen::Vector3d mean_coord(0, 0, 0);
    for (size_t i = P0; i <= P1; ++i) {
        mean_coord(0) += coords_x[i];
        mean_coord(1) += coords_y[i];
        mean_coord(2) += coords_z[i];
    }
    mean_coord /= (P1 - P0 + 1);

    // Calculate scale and translation, such that
    // translation is applied before scaling.
    double scale = 1.0;
    const Eigen::Vector3d translation = mean_coord;
    M.block<3, 1>(0, 3) = -translation;

    // Transform images.
    for (auto& image_proj_center : proj_centers) {
        image_proj_center.second -= translation;
        image_proj_center.second *= scale;
        const Eigen::Quaterniond quat(image_proj_center.first->Qvec(0), image_proj_center.first->Qvec(1),
                                      image_proj_center.first->Qvec(2), image_proj_center.first->Qvec(3));
        image_proj_center.first->SetTvec(quat * -image_proj_center.second);
    }

    // Transform points.
    for (auto& point3D : mappoints_) {
        point3D.second.XYZ() -= translation;
        point3D.second.XYZ() *= scale;
    }

    return M;
}

Eigen::Matrix3x4d Reconstruction::Centration(const double extent, 
                                            const double p0,
                                            const double p1) {
    CHECK_GT(extent, 0);
    CHECK_GE(p0, 0);
    CHECK_LE(p0, 1);
    CHECK_GE(p1, 0);
    CHECK_LE(p1, 1);
    CHECK_LE(p0, p1);

    Eigen::Matrix3x4d transform = Eigen::Matrix3x4d::Identity();
    if (register_image_ids_.size() < 2) {
        return transform;
    }

    EIGEN_STL_UMAP(class Image*, Eigen::Vector3d)
    proj_centers;
    for (size_t i = 0; i < register_image_ids_.size(); ++i) {
        class Image& image = Image(register_image_ids_[i]);
        const Eigen::Vector3d proj_center = image.TvecPrior();
        proj_centers[&image] = proj_center;
    }

    // Coordinates of image centers or point locations.
    std::vector<float> coords_x;
    std::vector<float> coords_y;
    std::vector<float> coords_z;
    coords_x.reserve(proj_centers.size());
    coords_y.reserve(proj_centers.size());
    coords_z.reserve(proj_centers.size());
    for (const auto& proj_center : proj_centers) {
        coords_x.push_back(static_cast<float>(proj_center.second(0)));
        coords_y.push_back(static_cast<float>(proj_center.second(1)));
        coords_z.push_back(static_cast<float>(proj_center.second(2)));
    }

    // Determine robust bounding box and mean.

    std::sort(coords_x.begin(), coords_x.end());
    std::sort(coords_y.begin(), coords_y.end());
    std::sort(coords_z.begin(), coords_z.end());

    const size_t P0 = static_cast<size_t>((coords_x.size() > 3) ? p0 * (coords_x.size() - 1) : 0);
    const size_t P1 = static_cast<size_t>((coords_x.size() > 3) ? p1 * (coords_x.size() - 1) : coords_x.size() - 1);

    const Eigen::Vector3d bbox_min(coords_x[P0], coords_y[P0], coords_z[P0]);
    const Eigen::Vector3d bbox_max(coords_x[P1], coords_y[P1], coords_z[P1]);

    Eigen::Vector3d mean_coord(0, 0, 0);
    for (size_t i = P0; i <= P1; ++i) {
        mean_coord(0) += coords_x[i];
        mean_coord(1) += coords_y[i];
        mean_coord(2) += coords_z[i];
    }
    mean_coord /= (P1 - P0 + 1);

    transform.block<3, 1>(0, 3) = mean_coord;

    // Transform images.
    for (size_t i = 0; i < register_image_ids_.size(); ++i) {
        class Image& image = Image(register_image_ids_[i]);
        Eigen::Vector3d image_proj_center = image.ProjectionCenter();
        image_proj_center -= mean_coord;
        const Eigen::Quaterniond quat(image.Qvec(0), image.Qvec(1),
                                      image.Qvec(2), image.Qvec(3));
        image.SetTvec(quat * -image_proj_center);

        Eigen::Vector3d& gps_location = image.TvecPrior();
        gps_location -= mean_coord;
    }

    // Transform points.
    for (auto& point3D : mappoints_) {
        point3D.second.XYZ() -= mean_coord;
    }

    return transform;
}

void Reconstruction::Decentration(const Eigen::Matrix3x4d& transform) {

    Eigen::Vector3d translation = transform.block<3, 1>(0, 3);

    // Transform images.
    for (size_t i = 0; i < register_image_ids_.size(); ++i) {
        class Image& image = Image(register_image_ids_[i]);
        Eigen::Vector3d proj_center = image.ProjectionCenter();
        proj_center += translation;
        const Eigen::Quaterniond quat(image.Qvec(0), image.Qvec(1),
                                      image.Qvec(2), image.Qvec(3));
        image.SetTvec(quat * -proj_center);

        Eigen::Vector3d& gps_location = image.TvecPrior();
        gps_location += translation;
    }

    // Transform points.
    for (auto& point3D : mappoints_) {
        point3D.second.XYZ() += translation;
    }
}

bool Reconstruction::Merge(const Reconstruction& reconstruction, const double max_reproj_error) {
    const double kMinInlierObservations = 0.3;

    Eigen::Matrix3x4d alignment;
    if (!ComputeAlignmentBetweenReconstructions(reconstruction, *this, kMinInlierObservations, max_reproj_error,
                                                &alignment)) {
        std::cout << "Merge Failed!" << std::endl;
        return false;
    }

    std::cout << "Similarity Transform: " << std::endl;
    std::cout << alignment.transpose() << std::endl;

    const SimilarityTransform3 tform(alignment);

    // Find common and missing images in the two reconstructions.

    std::unordered_set<image_t> common_image_ids;
    common_image_ids.reserve(reconstruction.NumRegisterImages());
    std::unordered_set<image_t> missing_image_ids;
    missing_image_ids.reserve(reconstruction.NumRegisterImages());

    for (const auto& image_id : reconstruction.RegisterImageIds()) {
        if (ExistsImage(image_id)) {
            common_image_ids.insert(image_id);
        } else {
            missing_image_ids.insert(image_id);
        }
    }

    // Register the missing images in this reconstruction.

    for (const auto image_id : missing_image_ids) {
        auto reg_image = reconstruction.Image(image_id);
        reg_image.SetRegistered(false);
        AddImage(reg_image);
        RegisterImage(image_id);
        if (!ExistsCamera(reg_image.CameraId())) {
            AddCamera(reconstruction.Camera(reg_image.CameraId()));
        }
        auto& image = Image(image_id);
        tform.TransformPose(&image.Qvec(), &image.Tvec());
    }

    // Merge the two point clouds using the following two rules:
    //    - copy points to this reconstruction with non-conflicting tracks,
    //      i.e. points that do not have an already triangulated observation
    //      in this reconstruction.
    //    - merge tracks that are unambiguous, i.e. only merge points in the two
    //      reconstructions if they have a one-to-one mapping.
    // Note that in both cases no cheirality or reprojection test is performed.

    for (const auto& mappoint : reconstruction.MapPoints()) {
        Track new_track;
        Track old_track;
        std::set<mappoint_t> old_mappoint_ids;
        for (const auto& track_el : mappoint.second.Track().Elements()) {
            if (common_image_ids.count(track_el.image_id) > 0) {
                const auto& point2D = Image(track_el.image_id).Point2D(track_el.point2D_idx);
                if (point2D.HasMapPoint()) {
                    old_track.AddElement(track_el);
                    old_mappoint_ids.insert(point2D.MapPointId());
                } else {
                    new_track.AddElement(track_el);
                }
            } else if (missing_image_ids.count(track_el.image_id) > 0) {
                Image(track_el.image_id).ResetMapPointForPoint2D(track_el.point2D_idx);
                new_track.AddElement(track_el);
            }
        }

        const bool create_new_point = new_track.Length() >= 2;
        const bool merge_new_and_old_point =
            (new_track.Length() + old_track.Length()) >= 2 && old_mappoint_ids.size() == 1;
        if (create_new_point || merge_new_and_old_point) {
            Eigen::Vector3d xyz = mappoint.second.XYZ();
            tform.TransformPoint(&xyz);
            const auto point3D_id = AddMapPoint(xyz, std::move(new_track), mappoint.second.Color());
            if (old_mappoint_ids.size() == 1) {
                MergeMapPoints(point3D_id, *old_mappoint_ids.begin());
            }
        }
    }

    FilterMapPointsWithLargeReprojectionError(max_reproj_error, MapPointIds(), true);

    return true;
}

bool Reconstruction::Merge(  // -- In most cases, it only add map point to the reconstruction
    const Reconstruction& reconstruction, const CorrespondenceGraph& scene_correspondence_graph,
    const double max_reproj_error, const double min_tri_angle) {
    const double kMinInlierObservations = 0.3;

    Eigen::Matrix3x4d alignment;

    bool merge_success = ComputeAlignmentBetweenReconstructions(reconstruction, *this, scene_correspondence_graph,
                                                                kMinInlierObservations, max_reproj_error, &alignment);
    if (!merge_success) {
        merge_success = ComputeAlignmentBetweenReconstructions(reconstruction, *this, kMinInlierObservations,
                                                               max_reproj_error, &alignment);
    }
    if (!merge_success) {
        std::cout << "Merge Failed!" << std::endl;
        return false;
    }

    std::cout << "Similarity Transform: " << std::endl;
    std::cout << alignment.transpose() << std::endl;

    const SimilarityTransform3 tform(alignment);

    // Find common and missing images in the two reconstructions.

    std::unordered_set<image_t> common_image_ids;
    common_image_ids.reserve(reconstruction.NumRegisterImages());
    std::unordered_set<image_t> missing_image_ids;
    missing_image_ids.reserve(reconstruction.NumRegisterImages());

    for (const auto& image_id : reconstruction.RegisterImageIds()) {
        if (ExistsImage(image_id)) {
            common_image_ids.insert(image_id);
        } else {
            missing_image_ids.insert(image_id);
        }
    }

    // Register the missing images in this reconstruction.

    for (const auto image_id : missing_image_ids) {
        auto reg_image = reconstruction.Image(image_id);
        reg_image.SetRegistered(false);
        AddImage(reg_image);
        RegisterImage(image_id);
        if (!ExistsCamera(reg_image.CameraId())) {
            AddCamera(reconstruction.Camera(reg_image.CameraId()));
        }
        auto& image = Image(image_id);
        tform.TransformPose(&image.Qvec(), &image.Tvec());
    }

    // Merge the two point clouds using the following two rules:
    //    - copy points to this reconstruction with non-conflicting tracks,
    //      i.e. points that do not have an already triangulated observation
    //      in this reconstruction.
    //    - merge tracks that are unambiguous, i.e. only merge points in the two
    //      reconstructions if they have a one-to-one mapping.
    // Note that in both cases no cheirality or reprojection test is performed.

    for (const auto& mappoint : reconstruction.MapPoints()) {
        Track new_track;
        Track old_track;
        std::set<mappoint_t> old_mappoint_ids;
        for (const auto& track_el : mappoint.second.Track().Elements()) {
            if (common_image_ids.count(track_el.image_id) > 0) {
                const auto& point2D = Image(track_el.image_id).Point2D(track_el.point2D_idx);
                if (point2D.HasMapPoint()) {
                    old_track.AddElement(track_el);
                    old_mappoint_ids.insert(point2D.MapPointId());
                } else {
                    new_track.AddElement(track_el);
                }
            } else if (missing_image_ids.count(track_el.image_id) > 0) {
                Image(track_el.image_id).ResetMapPointForPoint2D(track_el.point2D_idx);
                new_track.AddElement(track_el);
            }
        }

        const bool create_new_point = new_track.Length() >= 2;
        const bool merge_new_and_old_point =
            (new_track.Length() + old_track.Length()) >= 2 && old_mappoint_ids.size() == 1;
        if (create_new_point || merge_new_and_old_point) {
            Eigen::Vector3d xyz = mappoint.second.XYZ();
            tform.TransformPoint(&xyz);
            const auto point3D_id = AddMapPoint(xyz, std::move(new_track), mappoint.second.Color());
            if (old_mappoint_ids.size() == 1) {
                MergeMapPoints(point3D_id, *old_mappoint_ids.begin());
            }
        }
    }

    FilterAllMapPoints(2, max_reproj_error, min_tri_angle);

    return true;
}

bool Reconstruction::Merge(const Reconstruction& reconstruction, const Eigen::Matrix3x4d transform,
                           const double max_reproj_error) {
    if (correspondence_graph_){
        correspondence_graph_ = std::shared_ptr<CorrespondenceGraph>();
    }
    const double kMinInlierObservations = 0.3;

    Eigen::Matrix3x4d alignment = transform;  // -- Use the calculated similarity transform

    std::cout << "Similarity Transform: " << std::endl;
    std::cout << alignment.transpose() << std::endl;

    const SimilarityTransform3 tform_inverse(alignment);

    const SimilarityTransform3 tform = tform_inverse.Inverse();

    // Find common and missing images in the two reconstructions.

    std::unordered_set<image_t> common_image_ids;
    common_image_ids.reserve(reconstruction.NumRegisterImages());
    std::unordered_set<image_t> missing_image_ids;
    missing_image_ids.reserve(reconstruction.NumRegisterImages());

    for (const auto& image_id : reconstruction.RegisterImageIds()) {
        if (ExistsImage(image_id)) {
            common_image_ids.insert(image_id);
        } else {
            missing_image_ids.insert(image_id);
        }
    }

    /* // uncomment these sentence if want keep the respective cameras in both
       // reconstructions
    // Add Camera for the new reconstruction
    std::unordered_map<camera_t, camera_t> camera_convert_map;

    for (const auto& camera : reconstruction.Cameras()){
        auto camera_id = camera.first;
        auto cur_camera = camera.second;
        auto camera_merge_id = NumCameras() + 1;
        cur_camera.SetCameraId(camera_merge_id);
        cameras_.emplace(camera_merge_id, cur_camera);
        camera_convert_map[camera_id] = camera_merge_id;
    }


    for (const auto image_id : missing_image_ids) {
        auto reg_image = reconstruction.Image(image_id);
        reg_image.SetRegistered(false);
        AddImage(reg_image);
        RegisterImage(image_id);
        auto& image = Image(image_id);
        image.SetCameraId(camera_convert_map[image.CameraId()]);
        tform.TransformPose(&image.Qvec(), &image.Tvec());
    }
    */

    // Register the missing images in this reconstruction.
    for (const auto image_id : missing_image_ids) {
        auto reg_image = reconstruction.Image(image_id);
        reg_image.SetRegistered(false);
        AddImage(reg_image);
        RegisterImage(image_id);
        if (!ExistsCamera(reg_image.CameraId())) {
            AddCamera(reconstruction.Camera(reg_image.CameraId()));
        }
        auto& image = Image(image_id);
        tform.TransformPose(&image.Qvec(), &image.Tvec());
    }

    // Merge the two point clouds using the following two rules:
    //    - copy points to this reconstruction with non-conflicting tracks,
    //      i.e. points that do not have an already triangulated observation
    //      in this reconstruction.
    //    - merge tracks that are unambiguous, i.e. only merge points in the two
    //      reconstructions if they have a one-to-one mapping.
    // Note that in both cases no cheirality or reprojection test is performed.

    for (const auto& mappoint : reconstruction.MapPoints()) {
        Track new_track;
        Track old_track;
        std::set<mappoint_t> old_mappoint_ids;
        for (const auto& track_el : mappoint.second.Track().Elements()) {
            if (common_image_ids.count(track_el.image_id) > 0) {
                const auto& point2D = Image(track_el.image_id).Point2D(track_el.point2D_idx);
                if (point2D.HasMapPoint()) {
                    old_track.AddElement(track_el);
                    old_mappoint_ids.insert(point2D.MapPointId());
                } else {
                    new_track.AddElement(track_el);
                }
            } else if (missing_image_ids.count(track_el.image_id) > 0) {
                Image(track_el.image_id).ResetMapPointForPoint2D(track_el.point2D_idx);
                new_track.AddElement(track_el);
            }
        }

        const bool create_new_point = new_track.Length() >= 2;
        const bool merge_new_and_old_point =
            (new_track.Length() + old_track.Length()) >= 2 && old_mappoint_ids.size() == 1;
        if (create_new_point || merge_new_and_old_point) {
            Eigen::Vector3d xyz = mappoint.second.XYZ();
            tform.TransformPoint(&xyz);
            // Map point has been added, but the observation triangulated has not been set
            const auto point3D_id = AddMapPoint(xyz, std::move(new_track), mappoint.second.Color());
            if (old_mappoint_ids.size() == 1) {
                MergeMapPoints(point3D_id, *old_mappoint_ids.begin());
            }
        }
    }

    // FilterMapPointsWithLargeReprojectionError(
    //     max_reproj_error,
    //     MapPointIds(),
    //     true);

    return true;
}

std::unordered_map<image_t, image_t> 
Reconstruction::Append(const Reconstruction& reconstruction) {
    image_t old_max_image_id = 1;
    camera_t old_max_camera_id = 1;
    for (image_t image_id : this->RegisterImageIds()) {
        camera_t camera_id = this->Image(image_id).CameraId();
        old_max_image_id = std::max(old_max_image_id, image_id);
        old_max_camera_id = std::max(old_max_camera_id, camera_id);
    }

    Reconstruction reconstruction_tmp;
    std::unordered_map<image_t, image_t> map_ids;

    std::unordered_map<image_t, std::unordered_map<point2D_t, std::pair<image_t, point2D_t> > > image_point2d_map;
    std::unordered_map<camera_t, camera_t> camera_id_map;
    for (const image_t image_id : reconstruction.RegisterImageIds()) {
        const class Image image = reconstruction.Image(image_id);
        const class Camera camera = reconstruction.Camera(image.CameraId());
        const camera_t camera_id = camera.CameraId();

        class Image new_image;
        class Camera new_camera = camera;
        image_t new_image_id = image_id + old_max_image_id;
        camera_t new_camera_id = camera_id + old_max_camera_id;
        new_image.SetImageId(new_image_id);
        new_image.SetName(image.Name());
        new_image.SetQvec(image.Qvec());
        new_image.SetTvec(image.Tvec());
        new_image.SetLabelId(image.LabelId());
        if (camera_id_map.find(camera_id) != camera_id_map.end()) {
            new_camera = reconstruction_tmp.Camera(camera_id_map.at(camera_id));
            new_image.SetCameraId(new_camera.CameraId());
        } else {
            camera_id_map[camera_id] = new_camera_id;
            new_camera.SetCameraId(new_camera_id);
            new_image.SetCameraId(new_camera_id);
            reconstruction_tmp.AddCamera(new_camera);
        }

        std::vector<class Point2D> new_points2D;
        new_points2D.reserve(image.NumPoints2D());
        for (size_t point_id = 0; point_id < image.NumPoints2D(); ++point_id) {
            class Point2D point2D = image.Point2D(point_id);
            image_point2d_map[image_id][point_id] = std::make_pair(new_image_id, new_points2D.size());
            point2D.SetMapPointId(kInvalidMapPointId);
            new_points2D.emplace_back(point2D);
        }
        new_image.SetPoints2D(new_points2D);

        reconstruction_tmp.AddImage(new_image);
        reconstruction_tmp.RegisterImage(new_image_id);

        map_ids[image_id] = new_image.ImageId();
    }

    // Append to existed reconstruction.
    for (const auto image_id : reconstruction_tmp.RegisterImageIds()) {
        auto reg_image = reconstruction_tmp.Image(image_id);
        reg_image.SetRegistered(false);
        AddImage(reg_image);
        RegisterImage(image_id);
        if (!ExistsCamera(reg_image.CameraId())) {
            AddCamera(reconstruction_tmp.Camera(reg_image.CameraId()));
        }
    }

    for (const auto& mappoint_id : reconstruction.MapPointIds()) {
        // Get old mappoint
        const auto old_mappoint = reconstruction.MapPoint(mappoint_id);

        // Use the old mappoint position
        class MapPoint new_mappoint;
        new_mappoint.SetXYZ(old_mappoint.XYZ());
        new_mappoint.SetColor(old_mappoint.Color());
        new_mappoint.SetError(old_mappoint.Error());

        // Update the old mappoint track with new image id and point2d id
        class Track new_track;
        for (const auto& track_el : old_mappoint.Track().Elements()) {
            if (image_point2d_map[track_el.image_id].count(track_el.point2D_idx) == 0) {
                continue;
            }
            auto new_image_to_point = image_point2d_map[track_el.image_id][track_el.point2D_idx];
            
            new_track.AddElement(new_image_to_point.first, new_image_to_point.second);
        }
        // // Check track size
        // if(new_track.Length() <= 2){
        //     continue;
        // }

        new_mappoint.SetTrack(new_track);

        // Update reconstruction
        // reconstruction_tmp.AddMapPointWithError(new_mappoint.XYZ(), 
        //     new_mappoint.Track(), new_mappoint.Color(), new_mappoint.Error());
        AddMapPointWithError(new_mappoint.XYZ(), std::move(new_mappoint.Track()), new_mappoint.Color(), new_mappoint.Error());
    }
    return map_ids;
}

std::vector<image_t> Reconstruction::FindCommonRegImageIds(const Reconstruction& reconstruction) const {
    std::vector<image_t> common_reg_image_ids;
    for (const auto image_id : register_image_ids_) {
        if (reconstruction.ExistsImage(image_id) && reconstruction.IsImageRegistered(image_id)) {
            CHECK_EQ(Image(image_id).Name(), reconstruction.Image(image_id).Name());
            common_reg_image_ids.push_back(image_id);
        }
    }
    return common_reg_image_ids;
}

size_t Reconstruction::FilterMapPoints(const double max_reproj_error, const double min_tri_angle,
                                       const std::unordered_set<mappoint_t>& mappoint_ids) {
    size_t num_filtered = 0;
    num_filtered += FilterMapPointsWithLargeReprojectionError(max_reproj_error, mappoint_ids);
    num_filtered += FilterMapPointsWithSmallTriangulationAngle(min_tri_angle, mappoint_ids);
    return num_filtered;
}

size_t Reconstruction::FilterMapPointsInImages(const double max_reproj_error, const double min_tri_angle,
                                               const std::unordered_set<image_t>& image_ids) {
    std::unordered_set<mappoint_t> mappoint_ids;
    for (const image_t image_id : image_ids) {
        const class Image& image = Image(image_id);
        for (const Point2D& point2D : image.Points2D()) {
            if (point2D.HasMapPoint()) {
                mappoint_ids.insert(point2D.MapPointId());
            }
        }
    }
    return FilterMapPoints(max_reproj_error, min_tri_angle, mappoint_ids);
}

size_t Reconstruction::FilterAllMapPoints(const int min_track_length, const double max_reproj_error,
                                          const double min_tri_angle) {
    // Important: First filter observations and points with large reprojection
    // error, so that observations with large reprojection error do not make
    // a point stable through a large triangulation angle.
    const std::unordered_set<mappoint_t> mappoint_ids = MapPointIds();
    size_t num_filtered = 0;
    num_filtered += FilterMapPointsWithLargeReprojectionError(max_reproj_error, mappoint_ids);
    num_filtered += FilterMapPointsWithSmallTriangulationAngle(min_tri_angle, mappoint_ids);
    num_filtered += FilterMapPointsWithSmallTrackLength(min_track_length, mappoint_ids);
    return num_filtered;
}

size_t Reconstruction::FilterMapPoints(const int min_track_length, const double max_reproj_error,
                                          const double min_tri_angle,
                                          const std::unordered_set<mappoint_t>& addressed_points) {
    // Important: First filter observations and points with large reprojection
    // error, so that observations with large reprojection error do not make
    // a point stable through a large triangulation angle.
    const std::unordered_set<mappoint_t> mappoint_ids = addressed_points;
    size_t num_filtered = 0;
    num_filtered += FilterMapPointsWithLargeReprojectionError(max_reproj_error, mappoint_ids);
    num_filtered += FilterMapPointsWithSmallTriangulationAngle(min_tri_angle, mappoint_ids);
    num_filtered += FilterMapPointsWithSmallTrackLength(min_track_length, mappoint_ids);
    return num_filtered;
}

size_t Reconstruction::FilterObservationsWithNegativeDepth() {
    size_t num_filtered = 0;
    for (const auto image_id : register_image_ids_) {
        const class Image& image = Image(image_id);
        const class Camera& camera = Camera(image.CameraId());
        if(camera.ModelName().compare("SPHERICAL")==0){
            continue;
        }
        Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();
        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
            const Point2D& point2D = image.Point2D(point2D_idx);
            
            if(camera.NumLocalCameras()>1){
                int local_camera_id = image.LocalImageIndices()[point2D_idx];
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                camera.GetLocalCameraExtrinsic(local_camera_id,local_qvec,local_tvec);
                
                Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(),local_qvec);
                Eigen::Vector3d tvec = 
                    QuaternionToRotationMatrix(local_qvec)*image.Tvec() + local_tvec;
                proj_matrix = ComposeProjectionMatrix(qvec,tvec);
            }

            if (point2D.HasMapPoint()) {
                // if(ExistsMapPoint(point2D.MapPointId())){
                const class MapPoint& mappoint = MapPoint(point2D.MapPointId());
                if (!HasPointPositiveDepth(proj_matrix, mappoint.XYZ())) {
                    DeleteObservation(image_id, point2D_idx);
                    num_filtered += 1;
                }
                // }
            }
        }
    }
    return num_filtered;
}

std::vector<image_t> Reconstruction::FilterImages(const double min_focal_length_ratio,
                                                  const double max_focal_length_ratio, const double max_extra_param) {
    std::vector<image_t> filtered_image_ids;
    for (const image_t image_id : RegisterImageIds()) {
        const class Image& image = Image(image_id);
        const class Camera& camera = Camera(image.CameraId());
        if (image.NumMapPoints() == 0) {
            if (have_prior_pose && prior_force_keyframe &&
                prior_rotations.count(image_id) && prior_translations.count(image_id)) {
                continue;
            }
            filtered_image_ids.push_back(image_id);
        } else if (camera.ModelName().compare("SPHERICAL") != 0 &&
                   camera.ModelName().compare("UNIFIED") != 0 &&
                   camera.ModelName().compare("OPENCV_FISHEYE") != 0 &&
                   camera.HasBogusParams(min_focal_length_ratio, max_focal_length_ratio, max_extra_param)) {
            filtered_image_ids.push_back(image_id);
        }
    }

    for (const image_t image_id : filtered_image_ids) {
        DeRegisterImage(image_id);
        Image(image_id).SetKeyFrame(false);
    }

    return filtered_image_ids;
}

std::vector<image_t> Reconstruction::FilterImages(const double min_focal_length_ratio,
                                                  const double max_focal_length_ratio, const double max_extra_param,
                                                  const std::unordered_set<image_t>& addressed_images) {
    std::vector<image_t> filtered_image_ids;
    for (const image_t image_id : RegisterImageIds()) {
        if(addressed_images.count(image_id) == 0){
            continue;
        }
        const class Image& image = Image(image_id);
        const class Camera& camera = Camera(image.CameraId());
        if (image.NumMapPoints() == 0) {
            if (have_prior_pose && prior_force_keyframe &&
                prior_rotations.count(image_id) && prior_translations.count(image_id)) {
                continue;
            }
            filtered_image_ids.push_back(image_id);
        } else if (camera.ModelName().compare("SPHERICAL") != 0 &&
                   camera.HasBogusParams(min_focal_length_ratio, max_focal_length_ratio, max_extra_param)) {
            filtered_image_ids.push_back(image_id);
        }
    }

    for (const image_t image_id : filtered_image_ids) {
        DeRegisterImage(image_id);
        Image(image_id).SetKeyFrame(false);
    }

    return filtered_image_ids;
}

std::vector<image_t> Reconstruction::FilterAllFarawayImages(){

    std::unordered_map<image_t, double> median_distances;
    std::vector<double> median_distances_v;
    std::vector<image_t> filtered_image_ids;
    for(const image_t image_id: RegisterImageIds()){
        const class Image& image = Image(image_id);
        std::unordered_set<image_t> neighbor_image_ids;

        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
            
            const Point2D& point2D = image.Point2D(point2D_idx);
            if(point2D.HasMapPoint()){
                const class MapPoint& mappoint = MapPoint(point2D.MapPointId());
                for (const auto& track_el : mappoint.Track().Elements()){
                    if(neighbor_image_ids.count(track_el.image_id)==0&&
                        track_el.image_id!=image_id && 
                        IsImageRegistered(track_el.image_id)){
                        neighbor_image_ids.insert(track_el.image_id);
                    }
                }
            }
        }

        std::vector<double> neighbor_distances;

        for (auto& neighbor_image_id : neighbor_image_ids) {
            Eigen::Vector3d baseline = images_.at(image_id).ProjectionCenter() - 
                images_.at(neighbor_image_id).ProjectionCenter();

            neighbor_distances.push_back(baseline.norm());
        }
        
        if(neighbor_distances.size()>0){
            int nth1 = neighbor_distances.size() / 2;
            std::nth_element(neighbor_distances.begin(), neighbor_distances.begin() + nth1, 
						 neighbor_distances.end());
            double median_distance = neighbor_distances[nth1];
            median_distances.emplace(image_id,median_distance);
            median_distances_v.push_back(median_distance);
        }
    }
    if(median_distances_v.size() == 0){
        return filtered_image_ids;
    }
    
    CHECK(median_distances_v.size()>0);
    int nth = static_cast<int>(static_cast<double>(median_distances_v.size()) * 0.95);
    std::nth_element(median_distances_v.begin(),median_distances_v.begin() + nth,median_distances_v.end());

    double threshold = median_distances_v[nth] * 20;   
    
    std::cout<<"Farway threshold: "<<threshold<<std::endl;
    for(const auto distance:median_distances){
        if(distance.second > threshold){
            const class Image& image = Image(distance.first);

            for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
                const Point2D& point2D = image.Point2D(point2D_idx);

                if (point2D.HasMapPoint()) {
                    DeleteMapPoint(point2D.MapPointId());
                }
            }
            filtered_image_ids.push_back(distance.first);
            
            DeRegisterImage(distance.first);
            Image(distance.first).SetKeyFrame(false);    
            std::cout<<"Filter faraway image: "<<distance.first
                     <<" distance: "<<distance.second<<std::endl;
        }
    }

    return filtered_image_ids;
}

void Reconstruction::DeleteImage(const image_t image_id){
    
    CHECK(ExistsImage(image_id));

    const class Image& image = Image(image_id);
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        if (point2D.HasMapPoint()) {
            DeleteObservation(image_id,point2D_idx);
        }
    }

    if (correspondence_graph_) {
        std::unordered_map<image_t, std::unordered_set<image_t>> neighbors = correspondence_graph_->ImageNeighbors() ;
        if(neighbors.find(image_id)!=neighbors.end()){

            std::unordered_set<image_t> neighbor_images = correspondence_graph_->ImageNeighbor(image_id);

            for(const auto neighbor_image: neighbor_images){
                const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id, neighbor_image);
                if(image_pairs_.find(pair_id)!=image_pairs_.end()){
                    image_pairs_.erase(pair_id);
                }
            }
        }
    }

    images_.erase(image_id);
    for(auto iter = register_image_ids_.begin(); iter<register_image_ids_.end(); ++iter){
        if(*iter == image_id){
            register_image_ids_.erase(iter);
        }
    }
}


size_t Reconstruction::ComputeNumObservations() const {
    size_t num_observation = 0;
    for (const image_t image_id : register_image_ids_) {
        num_observation += Image(image_id).NumMapPoints();
    }
    return num_observation;
}

double Reconstruction::ComputeMeanTrackLength() const {
    if (mappoints_.empty()) {
        return 0.0;
    } else {
        return ComputeNumObservations() / static_cast<double>(mappoints_.size());
    }
}

double Reconstruction::ComputeMeanObservationsPerRegImage() const {
    if (register_image_ids_.empty()) {
        return 0.0;
    } else {
        return ComputeNumObservations() / static_cast<double>(register_image_ids_.size());
    }
}

double Reconstruction::ComputeMeanReprojectionError() const {
    double error_sum = 0.0;
    size_t num_valid_errors = 0;
    for (const auto& mappoint : mappoints_) {
        if (mappoint.second.HasError()) {
            error_sum += mappoint.second.Error();
            num_valid_errors += 1;
        }
    }

    return num_valid_errors == 0 ? 0.0 : error_sum / num_valid_errors;
}

bool Reconstruction::ExtractColorsForImage(const image_t image_id, const std::string& path) {
    const class Image& image = Image(image_id);
    const class Camera& camera= Camera(image.CameraId());

    int num_local_camera = camera.NumLocalCameras();
    if (num_local_camera > 1) {
        std::vector<Bitmap> bitmaps;
        bitmaps.resize(camera.NumLocalCameras());
        std::string camera_0_name = image.Name();
        size_t cam_pos = camera_0_name.find("cam0"); 
        for(size_t i = 0; i<bitmaps.size(); ++i){
            std::string cam_name = "cam" + std::to_string(i);
            std::string camera_i_name = camera_0_name;    
            camera_i_name.replace(cam_pos,4,cam_name);

            if (!bitmaps[i].Read(JoinPaths(path, camera_i_name))) {
                return false;
            }
        }

        const std::vector<uint32_t>& local_image_indices = image.LocalImageIndices();
        const Eigen::Vector3ub kBlackColor(0, 0, 0);
        
        for(size_t i = 0; i< image.Points2D().size(); ++i){    
            const Point2D& point2D = image.Points2D()[i];
            uint32_t local_image_id = local_image_indices[i];
            if (point2D.HasMapPoint()) {
                class MapPoint& mappoint = MapPoint(point2D.MapPointId());
                if (mappoint.Color() == kBlackColor) {
                    BitmapColor<float> color;
                    if (bitmaps[local_image_id].InterpolateBilinear(point2D.X() - 0.5, point2D.Y() - 0.5, &color)) {
                        const BitmapColor<uint8_t> color_ub = color.Cast<uint8_t>();
                        mappoint.SetColor(Eigen::Vector3ub(color_ub.r, color_ub.g, color_ub.b));
                    }
                }
            }
        }
    } else {
        Bitmap bitmap;
        if (IsFileRGBD(image.Name())) {
            ExtractRGBDData(JoinPaths(path, image.Name()), bitmap, true);
        } else if (!bitmap.Read(JoinPaths(path, image.Name()))) {
            return false;
        }

        const Eigen::Vector3ub kBlackColor(0, 0, 0);
        for (const Point2D point2D : image.Points2D()) {
            if (point2D.HasMapPoint()) {
                class MapPoint & mappoint = MapPoint(point2D.MapPointId());
                if (mappoint.Color() == kBlackColor) {
                    BitmapColor<float> color;
                    if (bitmap.InterpolateBilinear(point2D.X() - 0.5, point2D.Y() - 0.5, &color)) {
                        const BitmapColor<uint8_t> color_ub = color.Cast<uint8_t>();
                        mappoint.SetColor(Eigen::Vector3ub(color_ub.r, color_ub.g, color_ub.b));
                    }
                }
            }
        }
        bitmap.Deallocate();
    }
    return true;
}

void Reconstruction::ExtractColorsForAllImages(const std::string& path) {
    std::cout << "ExtractColorsForAllImages\n" << std::endl;

    EIGEN_STL_UMAP(mappoint_t, Eigen::Vector3d) color_sums;
    std::unordered_map<mappoint_t, size_t> color_counts;

    for (size_t i = 0; i < register_image_ids_.size(); ++i) {
        const class Image& image = Image(register_image_ids_[i]);
        const class Camera& camera = Camera(image.CameraId());

        std::vector<std::string> image_paths;
        if (camera.NumLocalCameras() > 1) {
            for (size_t local_camera_id = 0; local_camera_id < camera.NumLocalCameras(); ++local_camera_id) {
                std::string image_path = "";
                if (image.HasLocalName(local_camera_id)) {
                    image_path = JoinPaths(path, image.LocalName(local_camera_id));
                } else {
                    auto image_name = image.Name();
                    auto pos = image_name.find("cam0", 0);
                    image_name.replace(pos, 4, "cam" + std::to_string(local_camera_id));
                    image_path = JoinPaths(path, image_name);
                }
                image_paths.push_back(image_path);
            }
        } else {
            const std::string image_path = JoinPaths(path, image.Name());
            image_paths.push_back(image_path);
        }

        std::vector<Bitmap> bitmaps(1);
        if (IsFileRGBD(image.Name())) {
            ExtractRGBDData(image_paths[0], bitmaps[0], true);
        } else {
            bitmaps.resize(camera.NumLocalCameras());
            for (size_t local_camera_id = 0; local_camera_id < camera.NumLocalCameras(); ++local_camera_id) {
                if (boost::filesystem::exists(image_paths[local_camera_id])) {
                    bitmaps[local_camera_id].Read(image_paths[local_camera_id]); 
                } else {
                    std::cout << StringPrintf("Could not read image %s at path %s.", image.Name().c_str(), image_paths[local_camera_id].c_str())
                              << std::endl;                    
                }
            }
        }

        const auto & local_image_indices = image.LocalImageIndices();
        for (size_t i = 0; i < image.NumPoints2D(); ++i) {
            const auto point2D = image.Point2D(i);
            if (point2D.HasMapPoint()) {
                BitmapColor<float> color;
                uint32_t local_camera_id;
                if (camera.NumLocalCameras() > 1) {
                    local_camera_id = local_image_indices.at(i);
                } else {
                    local_camera_id = 0;
                }
                // sensemap assumes that the upper left pixel center is (0.5, 0.5).
                if (bitmaps[local_camera_id].Width() != 0 && 
                    bitmaps[local_camera_id].InterpolateBilinear(point2D.X() - 0.5, point2D.Y() - 0.5, &color)) {
                    if (color_sums.count(point2D.MapPointId())) {
                        Eigen::Vector3d& color_sum = color_sums[point2D.MapPointId()];
                        color_sum(0) += color.r;
                        color_sum(1) += color.g;
                        color_sum(2) += color.b;
                        color_counts[point2D.MapPointId()] += 1;
                    } else {
                        color_sums.emplace(point2D.MapPointId(), Eigen::Vector3d(color.r, color.g, color.b));
                        color_counts.emplace(point2D.MapPointId(), 1);
                    }
                }
            }
        }
        for (size_t local_camera_id = 0; local_camera_id < camera.NumLocalCameras(); ++local_camera_id) {
            bitmaps[local_camera_id].Deallocate();
        }
        std::cout << StringPrintf("\rLoad Image#%d", i);
    }
    std::cout << std::endl;

    const Eigen::Vector3ub kBlackColor = Eigen::Vector3ub::Zero();
    for (auto& mappoint : mappoints_) {
        if (color_sums.count(mappoint.first)) {
            Eigen::Vector3d color = color_sums[mappoint.first] / color_counts[mappoint.first];
            color.unaryExpr(std::ptr_fun<double, double>(std::round));
            mappoint.second.SetColor(color.cast<uint8_t>());
        } else {
            mappoint.second.SetColor(kBlackColor);
        }
    }
}

void Reconstruction::ColorHarmonization(const std::string& path) {
    
    std::cout << "Color Harmonization" << std::endl;

    image_t max_image_id = 0;
    std::unordered_map<mappoint_t, std::vector<Eigen::Vector3ub> > track_colors;
    auto mappoint_ids = MapPointIds();
    for (auto & mappoint_id : mappoint_ids) {
        track_colors[mappoint_id].resize(MapPoint(mappoint_id).Track().Length());
    }
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < register_image_ids_.size(); ++i) {
        const class Image& image = Image(register_image_ids_[i]);
        const std::string image_path = JoinPaths(path, image.Name());

        max_image_id = std::max(max_image_id, register_image_ids_[i]);

        Bitmap bitmap;
        if (IsFileRGBD(image.Name())) {
            ExtractRGBDData(image_path, bitmap, true);
        } else if (!bitmap.Read(image_path)) {
            std::cout << StringPrintf("Could not read image %s at path %s.", image.Name().c_str(), image_path.c_str())
                      << std::endl;
            continue;
        }

        for (const Point2D point2D : image.Points2D()) {
            if (point2D.HasMapPoint()) {
                BitmapColor<uint8_t> color;
                bitmap.GetPixel(point2D.X(), point2D.Y(), &color);

                const class MapPoint & mappoint = mappoints_.at(point2D.MapPointId());
                const class Track & track = mappoint.Track();

                for (size_t k = 0; k < track.Length(); ++k) {
                    auto track_elem = track.Element(k);
                    if (track_elem.image_id == register_image_ids_[i]) {
                        track_colors[point2D.MapPointId()][k] = Eigen::Vector3ub(color.r, color.g, color.b);
                        break;
                    }
                }
            }
        }
        bitmap.Deallocate();
    }

    std::unordered_map<image_pair_t, float> image_pairs;
    for (const auto & mappoint : mappoints_) {
        const auto & track = mappoint.second.Track();
        for (int i = 0; i < track.Length(); ++i) {
            const image_t image_id1 = track.Element(i).image_id;
            for (int j = i + 1; j < track.Length(); ++j) {
                const image_t image_id2 = track.Element(j).image_id;
                image_pair_t pair_id = utility::ImagePairToPairId(image_id1, image_id2);
                image_pairs[pair_id] += 1;
            }
        }
    }

    std::cout << "Graph Edge Compression" << std::endl;

    std::vector<std::pair<image_pair_t, float> > sorted_image_pairs;
    sorted_image_pairs.reserve(image_pairs.size());

    MinimumSpanningTree<image_t, float> mst_extractor;

    std::vector<image_pair_t> image_pair_ids;
    image_pair_ids.reserve(image_pairs.size());
    for (auto image_pair : image_pairs) {
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        image_pair_ids.push_back(image_pair.first);
        
        mst_extractor.AddEdge(image_id1, image_id2, -image_pair.second);
        sorted_image_pairs.emplace_back(image_pair.first, image_pair.second);
    }
    std::unordered_set<std::pair<image_t, image_t> > minimum_spanning_tree;
    mst_extractor.Extract(&minimum_spanning_tree);

    const int num_redundance_edges = 0.4 * (image_pairs.size() - NumRegisterImages() + 1);
    std::nth_element(sorted_image_pairs.begin(), sorted_image_pairs.begin() + num_redundance_edges + NumRegisterImages() - 1, 
                     sorted_image_pairs.end(), 
        [&](const std::pair<image_pair_t, float>& a, const std::pair<image_pair_t, float>& b) {
            return a.second > b.second;
        });

    image_pair_ids.clear();
    image_pair_ids.reserve(image_pairs.size());
    std::unordered_set<image_pair_t> tree_image_pair_ids; 
    for (auto image_pair : minimum_spanning_tree) {
        auto pair_id = utility::ImagePairToPairId(image_pair.first, image_pair.second);
        image_pair_ids.push_back(pair_id);
        tree_image_pair_ids.insert(pair_id);
    }
    int num_edge = 0;
    while(num_edge < num_redundance_edges) {
        auto pair_id = sorted_image_pairs.at(num_edge).first;
        if (tree_image_pair_ids.find(pair_id) == tree_image_pair_ids.end()) {
            image_pair_ids.push_back(pair_id);
        }
        num_edge++;
    }

    std::sort(image_pair_ids.begin(), image_pair_ids.end());

    std::cout << StringPrintf("%d edges are preserved, %d edges are pruned!\n",
        image_pair_ids.size(), image_pairs.size() - image_pair_ids.size());

    yrb_factors.resize(max_image_id + 1);
    for (auto image_id : RegisterImageIds()) {
        yrb_factors[image_id].s_Y = 1.0f;
        yrb_factors[image_id].s_Cb = 1.0f;
        yrb_factors[image_id].s_Cr = 1.0f;
        yrb_factors[image_id].o_Y = 0.0f;
        yrb_factors[image_id].o_Cb = 0.0f;
        yrb_factors[image_id].o_Cb = 0.0f;
    }

    ceres::Problem problem_Y, problem_Cb, problem_Cr;
    ceres::LossFunction *loss_function_Y = new ceres::SoftLOneLoss(1.0);
    ceres::LossFunction *loss_function_Cb = new ceres::SoftLOneLoss(1.0);
    ceres::LossFunction *loss_function_Cr = new ceres::SoftLOneLoss(1.0);
    
    int num_image_pair = 0;

#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < image_pair_ids.size(); ++k) {
        auto image_pair_id = image_pair_ids.at(k);
        image_t image_id1;
        image_t image_id2;
        utility::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);

        std::unordered_map<mappoint_t, std::pair<int, int> > track_elem_idx_map;

        const class Image& image1 = Image(image_id1);
        for (const Point2D point2D : image1.Points2D()) {
            if (point2D.HasMapPoint()) {
                const class Track & track = MapPoint(point2D.MapPointId()).Track();
                int idx1(-1), idx2(-1);
                for (size_t k = 0; k < track.Length(); ++k) {
                    if (idx1 != -1 && idx2 != -1) {
                        break;
                    }
                    auto track_elem = track.Element(k);
                    if (track_elem.image_id == image_id1) {
                        idx1 = k;
                    } else if (track_elem.image_id == image_id2) {
                        idx2 = k;
                    }
                }
                if (idx1 != -1 && idx2 != -1) {
                    track_elem_idx_map[point2D.MapPointId()] = std::make_pair(idx1, idx2);
                }
            }
        }

        if (track_elem_idx_map.size() < 50) {
            continue;
        }

        Histogram hist1_Y(0.0f, 1.0f, 50);
        Histogram hist1_Cb(0.0f, 1.0f, 50);
        Histogram hist1_Cr(0.0f, 1.0f, 50);
        Histogram hist2_Y(0.0f, 1.0f, 50);
        Histogram hist2_Cb(0.0f, 1.0f, 50);
        Histogram hist2_Cr(0.0f, 1.0f, 50);

        for (auto elem_idx : track_elem_idx_map) {
            auto & colors = track_colors.at(elem_idx.first);
            Eigen::Vector3ub & color1 = colors.at(elem_idx.second.first);
            float V1[3];
            V1[0] = color1[0] * INV_COLOR_NORM;
            V1[1] = color1[1] * INV_COLOR_NORM;
            V1[2] = color1[2] * INV_COLOR_NORM;
            ColorRGBToYCbCr(V1);

            Eigen::Vector3ub & color2 = colors.at(elem_idx.second.second);
            float V2[3];
            V2[0] = color2[0] * INV_COLOR_NORM;
            V2[1] = color2[1] * INV_COLOR_NORM;
            V2[2] = color2[2] * INV_COLOR_NORM;
            ColorRGBToYCbCr(V2);

            hist1_Y.add_value(V1[0]);
            hist1_Cb.add_value(V1[1]);
            hist1_Cr.add_value(V1[2]);
            hist2_Y.add_value(V2[0]);
            hist2_Cb.add_value(V2[1]);
            hist2_Cr.add_value(V2[2]);
        }

#pragma omp critical
        {
        float bi = 0.01f;
        // float bi = 0.002f;
        while (bi < 1.0f) {

            ceres::CostFunction *cost_function_Y = nullptr, *cost_function_Cb = nullptr, *cost_function_Cr = nullptr;

            cost_function_Y = ColorCorrectionCostFunction::Create(
                    hist1_Y.get_approx_percentile(bi),
                    hist2_Y.get_approx_percentile(bi));
            problem_Y.AddResidualBlock(cost_function_Y, loss_function_Y, &yrb_factors[image_id1].s_Y,
                                    &yrb_factors[image_id2].s_Y, &yrb_factors[image_id1].o_Y, &yrb_factors[image_id2].o_Y);

            cost_function_Cb = ColorCorrectionCostFunction::Create(
                    hist1_Cb.get_approx_percentile(bi),
                    hist2_Cb.get_approx_percentile(bi));
            problem_Cb.AddResidualBlock(cost_function_Cb, loss_function_Cb, &yrb_factors[image_id1].s_Cb,
                                        &yrb_factors[image_id2].s_Cb, &yrb_factors[image_id1].o_Cb, &yrb_factors[image_id2].o_Cb);

            cost_function_Cr = ColorCorrectionCostFunction::Create(
                    hist1_Cr.get_approx_percentile(bi),
                    hist2_Cr.get_approx_percentile(bi));
            problem_Cr.AddResidualBlock(cost_function_Cr, loss_function_Cr, &yrb_factors[image_id1].s_Cr,
                                        &yrb_factors[image_id2].s_Cr, &yrb_factors[image_id1].o_Cr, &yrb_factors[image_id2].o_Cr);

            bi += 0.02f;
            // bi += 0.004f;
        }
        num_image_pair++;
        }
        std::cout << StringPrintf("\rProcess image pair: [%6d, %6d] %d/%d", image_id1, image_id2, num_image_pair, image_pair_ids.size()) << std::flush;
    }

#pragma omp parallel for schedule(dynamic)
    for(std::size_t i = 0; i < NumRegisterImages(); ++i){
        image_t image_id = register_image_ids_.at(i);
        if(problem_Y.HasParameterBlock(&yrb_factors[image_id].s_Y)) {
            problem_Y.SetParameterLowerBound(&yrb_factors[image_id].s_Y, 0, 1 - 0.4);
            problem_Y.SetParameterUpperBound(&yrb_factors[image_id].s_Y, 0, 1 + 0.4);
            problem_Y.SetParameterLowerBound(&yrb_factors[image_id].o_Y, 0, -30.0 / 255);
            problem_Y.SetParameterUpperBound(&yrb_factors[image_id].o_Y, 0, 30.0 / 255);
        }
        if(problem_Cb.HasParameterBlock(&yrb_factors[image_id].s_Cb)) {
            problem_Cb.SetParameterLowerBound(&yrb_factors[image_id].s_Cb, 0, 1 - 0.2);
            problem_Cb.SetParameterUpperBound(&yrb_factors[image_id].s_Cb, 0, 1 + 0.2);
            problem_Cb.SetParameterLowerBound(&yrb_factors[image_id].o_Cb, 0, -5.0 / 255);
            problem_Cb.SetParameterUpperBound(&yrb_factors[image_id].o_Cb, 0, 5.0 / 255);
        }
        if(problem_Cr.HasParameterBlock(&yrb_factors[image_id].s_Cr)) {
            problem_Cr.SetParameterLowerBound(&yrb_factors[image_id].s_Cr, 0, 1 - 0.2);
            problem_Cr.SetParameterUpperBound(&yrb_factors[image_id].s_Cr, 0, 1 + 0.2);
            problem_Cr.SetParameterLowerBound(&yrb_factors[image_id].o_Cr, 0, -5.0 / 255);
            problem_Cr.SetParameterUpperBound(&yrb_factors[image_id].o_Cr, 0, 5.0 / 255);
        }
    }

    std::cout << "Start Graph Optimization" << std::endl;

    ceres::Solver::Options solver_options;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // solver_options.linear_solver_type = ceres::CGNR;
    // solver_options.preconditioner_type = ceres::JACOBI;
    solver_options.num_threads = GetEffectiveNumThreads(-1);

    ceres::Solver::Summary summary_Y;
    ceres::Solve(solver_options, &problem_Y, &summary_Y);
    printf("Solve channel Y\n");
    PrintSolverSummary(summary_Y);

    ceres::Solver::Summary summary_Cb;
    ceres::Solve(solver_options, &problem_Cb, &summary_Cb);
    printf("Solve channel Cb\n");
    PrintSolverSummary(summary_Cb);

    ceres::Solver::Summary summary_Cr;
    ceres::Solve(solver_options, &problem_Cr, &summary_Cr);
    printf("Solve channel Cr\n");
    PrintSolverSummary(summary_Cr);

    for (auto i : RegisterImageIds()) {
        std::cout << StringPrintf("Local Luminance Adjust(%d): scale (%f %f %f), offset (%f %f %f)", i, 
            yrb_factors[i].s_Y, yrb_factors[i].s_Cb, yrb_factors[i].s_Cr, 
            yrb_factors[i].o_Y, yrb_factors[i].o_Cb, yrb_factors[i].o_Cr) << std::endl;
    }
}

size_t Reconstruction::FilterMapPointsWithBoundBox (
    const Eigen::Vector3f bb_min, const Eigen::Vector3f bb_max, 
    Eigen::Matrix3f pivot) {
    size_t num_filtered = 0;

    Eigen::Vector3f points_box_min = bb_min;
    Eigen::Vector3f points_box_max = bb_max;
    for (auto image_id : RegisterImageIds()) {
        Eigen::Vector3f C = images_.at(image_id).ProjectionCenter().cast<float>();
        points_box_min[0] = std::min(points_box_min[0], C[0]);
        points_box_min[1] = std::min(points_box_min[1], C[1]);
        points_box_min[2] = std::min(points_box_min[2], C[2]);
        points_box_max[0] = std::max(points_box_max[0], C[0]);
        points_box_max[1] = std::max(points_box_max[1], C[1]);
        points_box_max[2] = std::max(points_box_max[2], C[2]);
    }

    const auto& mappoint_ids = MapPointIds();
    for (const auto mappoint_id : mappoint_ids) {
        if (!ExistsMapPoint(mappoint_id)) {
            continue;
        }

        class MapPoint& mappoint = MapPoint(mappoint_id);
        Eigen::Vector3f temp_point(mappoint.X(), mappoint.Y(), mappoint.Z());
        Eigen::Vector3f point = pivot * temp_point;
        if (point.x() < points_box_min.x() || point.x() > points_box_max.x() ||
            point.y() < points_box_min.y() || point.y() > points_box_max.y() ||
            point.z() < points_box_min.z() || point.z() > points_box_max.z()){
            DeleteMapPoint(mappoint_id);
            num_filtered += mappoint.Track().Length();
            continue;
        }
    }

    return num_filtered;
}

size_t Reconstruction::FilterMapPointsWithSpatialDistribution (
    const std::unordered_set<mappoint_t>& mappoint_ids,
    const float spacing_factor) {

    std::unordered_map<mappoint_t, int> mappoint2pointid;
    std::vector<FT> point_spacings;
    FT average_spacing, stdev_spacing;
    Eigen::Matrix3f pivot;
    {
        std::vector<point_3_t> points_3(mappoint_ids.size());
        int num_point = 0;
        for (const auto mappoint_id : mappoint_ids) {
            if (!ExistsMapPoint(mappoint_id)) {
                continue;
            }
            class MapPoint& mappoint = MapPoint(mappoint_id);
            mappoint2pointid[mappoint_id] = num_point;
            points_3[num_point] = point_3_t(mappoint.X(), mappoint.Y(), mappoint.Z());
            num_point++;
        }
        points_3.resize(num_point);

        ComputeMeanStdevDistance(point_spacings, average_spacing, stdev_spacing, points_3);
    }
   
    
    Eigen::Vector3f points_box_min, points_box_max;
    {
        std::vector<Eigen::Vector3f> temp_points(mappoint_ids.size());
        int num_point = 0;
        const FT spacing_thres = average_spacing + spacing_factor * stdev_spacing;
        for (const auto mappoint_id : mappoint_ids) {
            if (!ExistsMapPoint(mappoint_id)) {
                continue;
            }
            class MapPoint& mappoint = MapPoint(mappoint_id);
            if (point_spacings[mappoint2pointid[mappoint_id]] > spacing_thres){
                continue;
            }
            temp_points[num_point].x() = mappoint.X();
            temp_points[num_point].y() = mappoint.Y();
            temp_points[num_point].z() = mappoint.Z();
            num_point++;
        }
        temp_points.resize(num_point);
        pivot = PovitMatrix(temp_points);

        points_box_min = pivot * temp_points[0];
        points_box_max = pivot * temp_points[0];
        // std::vector<Eigen::Vector3f> transformed_points(num_point);
        for (int i = 0; i < num_point; ++i) {
            // auto &point = transformed_points[i];
            Eigen::Vector3f point = pivot * temp_points[i];

            points_box_min[0] = std::min(points_box_min[0], point[0]);
            points_box_min[1] = std::min(points_box_min[1], point[1]);
            points_box_min[2] = std::min(points_box_min[2], point[2]);
            points_box_max[0] = std::max(points_box_max[0], point[0]);
            points_box_max[1] = std::max(points_box_max[1], point[1]);
            points_box_max[2] = std::max(points_box_max[2], point[2]);
        }

        for (auto image_id : RegisterImageIds()) {
            Eigen::Vector3f C = images_.at(image_id).ProjectionCenter().cast<float>();
            points_box_min[0] = std::min(points_box_min[0], C[0]);
            points_box_min[1] = std::min(points_box_min[1], C[1]);
            points_box_min[2] = std::min(points_box_min[2], C[2]);
            points_box_max[0] = std::max(points_box_max[0], C[0]);
            points_box_max[1] = std::max(points_box_max[1], C[1]);
            points_box_max[2] = std::max(points_box_max[2], C[2]);
        }
    }

    size_t num_filtered = 0;
    for (const auto mappoint_id : mappoint_ids) {
        if (!ExistsMapPoint(mappoint_id)) {
            continue;
        }

        class MapPoint& mappoint = MapPoint(mappoint_id);
        Eigen::Vector3f temp_point(mappoint.X(), mappoint.Y(), mappoint.Z());
        Eigen::Vector3f point = pivot * temp_point;
        if (point.x() < points_box_min.x() || point.x() > points_box_max.x() ||
            point.y() < points_box_min.y() || point.y() > points_box_max.y() ||
            point.z() < points_box_min.z() || point.z() > points_box_max.z()){
            DeleteMapPoint(mappoint_id);
            num_filtered += mappoint.Track().Length();
            continue;
        }
    }
    std::cout << "  => num filtered small track length: " << num_filtered << std::endl;
    return num_filtered;
}

size_t Reconstruction::FilterMapPointsWithSmallTrackLength(const int min_track_length,
                                                           const std::unordered_set<mappoint_t>& mappoint_ids) {
    size_t num_filtered = 0;

    for (const auto mappoint_id : mappoint_ids) {
        if (!ExistsMapPoint(mappoint_id)) {
            continue;
        }

        class MapPoint& mappoint = MapPoint(mappoint_id);

        if (mappoint.Track().Length() < min_track_length) {
            DeleteMapPoint(mappoint_id);
            num_filtered += mappoint.Track().Length();
            continue;
        }
    }
    std::cout << "  => num filtered small track length: " << num_filtered << std::endl;
    return num_filtered;
}

size_t Reconstruction::FilterMapPointsWithSmallTriangulationAngle(const double min_tri_angle,
                                                                  const std::unordered_set<mappoint_t>& mappoint_ids) {
    size_t num_filtered = 0;

    // Minimum triangulation angle in radians.
    const double min_tri_angle_rad = DegToRad(min_tri_angle);
    // Cache for image projection centers.
    EIGEN_STL_UMAP(image_t, Eigen::Vector3d) proj_centers;

    for (const auto mappoint_id : mappoint_ids) {
        if (!ExistsMapPoint(mappoint_id)) {
            continue;
        }

        const class MapPoint& mappoint = MapPoint(mappoint_id);
        
        // Calculate triangulation angle for all pairwise combinations of image
        // poses in the track. Only delete point if none of the combinations
        // has a sufficient triangulation angle.
        bool keep_point = false;
        for (size_t i1 = 0; i1 < mappoint.Track().Length(); ++i1) {
            const image_t image_id1 = mappoint.Track().Element(i1).image_id;
            const point2D_t point_id1 = mappoint.Track().Element(i1).point2D_idx;
            const class Image & image1 = Image(image_id1);
            const class Point2D & point2D1 = image1.Point2D(point_id1);
            if (point2D1.InOverlap()) {
                keep_point = true;
                break;
            }
        }
        for (size_t i1 = 0; !keep_point && (i1 < mappoint.Track().Length()); ++i1) {
            const image_t image_id1 = mappoint.Track().Element(i1).image_id;
            const point2D_t point_id1 = mappoint.Track().Element(i1).point2D_idx;
            const class Image & image1 = Image(image_id1);
            const class Camera & camera1 = Camera(image1.CameraId());
            const class Point2D & point2D1 = image1.Point2D(point_id1);

            Eigen::Vector3d proj_center1;    

            if(camera1.NumLocalCameras()>1){
                int local_camera_id1 = image1.LocalImageIndices()[point_id1];
                image_t local_image_full_id1 = image_id1*camera1.NumLocalCameras()+local_camera_id1;
                if(proj_centers.count(local_image_full_id1) ==0){

                    Eigen::Vector4d local_qvec;
                    Eigen::Vector3d local_tvec;
                    camera1.GetLocalCameraExtrinsic(local_camera_id1,local_qvec,local_tvec);
                    Eigen::Vector4d qvec = ConcatenateQuaternions(image1.Qvec(),local_qvec);
                    Eigen::Vector3d tvec = 
                        QuaternionToRotationMatrix(local_qvec)*image1.Tvec() + local_tvec;

                    proj_center1 = ProjectionCenterFromPose(qvec,tvec);
                    proj_centers.emplace(local_image_full_id1,proj_center1);
                }
                else{
                    proj_center1 = proj_centers.at(local_image_full_id1);
                }
            }
            else{
                if (proj_centers.count(image_id1) == 0){
                    proj_center1 = image1.ProjectionCenter();
                    proj_centers.emplace(image_id1, proj_center1);
                }
                else{
                    proj_center1 = proj_centers.at(image_id1);
                }    
            }

            for (size_t i2 = 0; i2 < i1; ++i2){
                
                const image_t image_id2 = mappoint.Track().Element(i2).image_id;
                const point2D_t point_id2 = mappoint.Track().Element(i2).point2D_idx;
                const class Image & image2 = Image(image_id2);
                const class Camera & camera2 = Camera(image2.CameraId());
                const class Point2D & point2D2 = image2.Point2D(point_id2);

                Eigen::Vector3d proj_center2;
                
                if(camera2.NumLocalCameras() == 1){
                    CHECK(proj_centers.find(image_id2)!=proj_centers.end());
                    proj_center2 = proj_centers.at(image_id2);
                }
                else{
                    int local_camera_id2 = image2.LocalImageIndices()[point_id2];
                    image_t local_image_full_id2 = image_id2*camera2.NumLocalCameras()+local_camera_id2;
                    CHECK(proj_centers.find(local_image_full_id2)!=proj_centers.end());

                    proj_center2 = proj_centers.at(local_image_full_id2);
                }

                const double tri_angle =
                    CalculateTriangulationAngle(proj_center1,
                                                proj_center2,
                                                mappoint.XYZ());
                if (tri_angle >= min_tri_angle_rad){
                    keep_point = true;
                    break;
                }
            }
        }

        if (!keep_point) {
            num_filtered += 1;
            DeleteMapPoint(mappoint_id);
        }
    }

    std::cout << "  => num filtered small angle: " << num_filtered << " (" << min_tri_angle << "deg)" << std::endl;
    return num_filtered;
}

size_t Reconstruction::FilterMapPointsWithLargeReprojectionError(const double max_reproj_error,
                                                                 const std::unordered_set<mappoint_t>& mappoint_ids,
                                                                 bool verbose) {
    const double max_squared_reproj_error = max_reproj_error * max_reproj_error;

    std::unordered_map<int, int> statistic;
    size_t num_two_view_geometry = 0;

    size_t num_filtered = 0;
    size_t num_filtered_rgbd = 0;
    size_t num_filtered_reproj_depth = 0;
    
    for (const auto mappoint_id : mappoint_ids) {
        if (!ExistsMapPoint(mappoint_id)) {
            continue;
        }

        class MapPoint& mappoint = MapPoint(mappoint_id);

        if (mappoint.Track().Length() < 2) {
            num_two_view_geometry++;
            DeleteMapPoint(mappoint_id);
            num_filtered += mappoint.Track().Length();
            continue;
        }

        double reproj_error_sum = 0.0;
        double tmp_error = 0.0, tmp_inlier_error = 0.0;
        int tmp_count = 0, tmp_inlier_count = 0;

        std::vector<TrackElement> track_elems_to_delete;
        // std::cout<<"Mappoint "<<mappoint_id<<" squared error: ";
        for (const auto & track_el : mappoint.Track().Elements()) {
            const class Image & image = Image(track_el.image_id);
            const class Camera & camera = Camera(image.CameraId());
            const Point2D & point2D = image.Point2D(track_el.point2D_idx);

            double squared_reproj_error;

            if(camera.NumLocalCameras()>1){
                int local_camera_id = image.LocalImageIndices()[track_el.point2D_idx];
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                camera.GetLocalCameraExtrinsic(local_camera_id,local_qvec,local_tvec);
                
                Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(),local_qvec);
                Eigen::Vector3d tvec = 
                    QuaternionToRotationMatrix(local_qvec)*image.Tvec() + local_tvec;
                
                squared_reproj_error = 
                    CalculateSquaredReprojectionErrorRig(point2D.XY(), mappoint.XYZ(),
                                                      qvec, tvec, local_camera_id,
                                                      camera); 
                //std::cout<<"squared_reproj_error:"<<sqrt(squared_reproj_error)<<" ";
            }
            else{
                squared_reproj_error =
                    CalculateSquaredReprojectionError(point2D.XY(), mappoint.XYZ(),
                                                      image.Qvec(), image.Tvec(),
                                                      camera);
            }

            Eigen::Vector3d proj_point3D;
            if (rgbd_filter_depth_weight > 0 || rgbd_max_reproj_depth > 0) {
                proj_point3D =
                    QuaternionRotatePoint(image.Qvec(), mappoint.XYZ()) + image.Tvec();
            }

            double squared_depth_error = 0.0;
            if (rgbd_filter_depth_weight > 0.0) {
                const float depth = point2D.Depth();
                const float weight = point2D.DepthWeight() * rgbd_filter_depth_weight;
                if (depth > 0.0f) {
                    squared_depth_error = weight * weight * 
                                        (proj_point3D[2] - depth) * (proj_point3D[2] - depth);
                }
            }

            if (!point2D.InOverlap() && 
                squared_reproj_error > max_squared_reproj_error){
                track_elems_to_delete.push_back(track_el);
                tmp_error += squared_reproj_error;
                tmp_count++;
                // std::cout<<"large squared_reproj_error:"<<sqrt(squared_reproj_error)<<" ";
            }
            else if (!point2D.InOverlap() && 
                depth_enabled && 
                rgbd_max_reproj_depth > 0 && 
                proj_point3D[2] > rgbd_max_reproj_depth) {
                track_elems_to_delete.push_back(track_el);
                tmp_error += squared_reproj_error;
                tmp_count++;
                num_filtered_reproj_depth++;
            }
            else if (!point2D.InOverlap() && 
                depth_enabled && 
                squared_reproj_error + squared_depth_error > max_squared_reproj_error){
                track_elems_to_delete.push_back(track_el);
                tmp_error += squared_reproj_error;
                tmp_count++;
                num_filtered_rgbd++;
            }
            else{
                tmp_inlier_error += squared_reproj_error;
                tmp_inlier_count++;
                reproj_error_sum += std::sqrt(squared_reproj_error);
            }
        }


        if (track_elems_to_delete.size() >= mappoint.Track().Length() - 1) {
            statistic[(int)max_reproj_error]++;
            num_filtered += mappoint.Track().Length();
            DeleteMapPoint(mappoint_id);
        } else {
            num_filtered += track_elems_to_delete.size();
            for (const auto& track_el : track_elems_to_delete) {
                DeleteObservation(track_el.image_id, track_el.point2D_idx);
            }
            mappoint.SetError(reproj_error_sum / mappoint.Track().Length());
            statistic[mappoint.Error()]++;
        }
    }
    // std::cout<<std::endl;

    if (verbose) {
        std::cout << std::endl << "two_view_geometry: " << num_two_view_geometry << std::endl;
        for (auto& reproj_map : statistic) {
            std::cout << "reprojection error : " << reproj_map.first << " / " << reproj_map.second << std::endl;
        }
    }
    std::cout << "  => num filtered reprojection: " << num_filtered << " (" << max_reproj_error << "pixel)" << std::endl;
    if (depth_enabled && rgbd_max_reproj_depth > 0) {
        std::cout << "  => num filtered reprojection(depth): " << num_filtered_reproj_depth << std::endl;
    }
    if (depth_enabled && rgbd_filter_depth_weight > 0) {
        std::cout << "  => num filtered reprojection(rgbd): " << num_filtered_rgbd << std::endl;
    }
    return num_filtered;
}

void Reconstruction::SetObservationAsTriangulated(const image_t image_id, const point2D_t point2D_idx,
                                                  const bool is_continued_mappoint, bool verbose) {
    if (!correspondence_graph_) {
        if (verbose) {
            std::cout << "Corrspondence graph is null, return directly" << std::endl;
        }
        return;
    }
    if (verbose) {
        std::cout << "Should be not called when merging" << std::endl;
    }
    const class Image& image = Image(image_id);
    const class Camera& camera = Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(point2D_idx);
    const std::vector<CorrespondenceGraph::Correspondence>& corrs =
        correspondence_graph_->FindCorrespondences(image_id, point2D_idx);

    CHECK(image.IsRegistered());
    CHECK(point2D.HasMapPoint());

    for (const auto& corr : corrs) {
        class Image& corr_image = Image(corr.image_id);
        class Camera& corr_camera = Camera(corr_image.CameraId());
        const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
        corr_image.IncrementCorrespondenceHasMapPoint(corr.point2D_idx);
        // Update number of shared Map Points between image pairs and make sure to
        // only count the correspondences once (not twice forward and backward).
        if (point2D.MapPointId() == corr_point2D.MapPointId() && 
            camera.NumLocalCameras() <= 2 && corr_camera.NumLocalCameras() <= 2 &&
            (is_continued_mappoint || image_id < corr.image_id)) {
            const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id, corr.image_id);
            image_pairs_[pair_id].first += 1;
            if (image_pairs_[pair_id].first > image_pairs_[pair_id].second) {
                std::cout << is_continued_mappoint << std::endl;
                std::cout << image_id << ", " << corr.image_id << std::endl;
                std::cout << point2D_idx << ", " << corr.point2D_idx << std::endl;
                std::cout << image_pairs_[pair_id].first << ", " << image_pairs_[pair_id].second << std::endl;
            }
            CHECK_LE(image_pairs_[pair_id].first, image_pairs_[pair_id].second)
                << "The correspondence graph must not contain duplicate matches";
        }
    }
}

void Reconstruction::ResetTriObservations(const image_t image_id, const point2D_t point2D_idx,
                                          const bool is_deleted_mappoint) {
    if (!correspondence_graph_) {
        return;
    }

    const class Image& image = Image(image_id);
    const class Camera& camera = Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(point2D_idx);
    if (!correspondence_graph_->ExistsImage(image_id)) {
        return;
    }

    const std::vector<CorrespondenceGraph::Correspondence>& corrs =
        correspondence_graph_->FindCorrespondences(image_id, point2D_idx);

    CHECK(image.IsRegistered());
    CHECK(point2D.HasMapPoint());

    for (const auto& corr : corrs) {
	    if (ExistsImage(corr.image_id) == false) continue;
        class Image& corr_image = Image(corr.image_id);
        class Camera& corr_camera = Camera(corr_image.CameraId());
        const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
        corr_image.DecrementCorrespondenceHasMapPoint(corr.point2D_idx);
        if (point2D.MapPointId() == corr_point2D.MapPointId() && 
            camera.NumLocalCameras() <= 2 && corr_camera.NumLocalCameras() <= 2 &&
            (!is_deleted_mappoint || image_id < corr.image_id)) {
            const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(image_id, corr.image_id);
            if (image_pairs_[pair_id].first <= 0) {
                std::cout << image_id << ", " << corr.image_id << std::endl;
                std::cout << point2D_idx << ", " << corr.point2D_idx << std::endl;
                std::cout << image_pairs_[pair_id].first << ", " << image_pairs_[pair_id].second << std::endl;
            }
            image_pairs_[pair_id].first -= 1;
            CHECK_GE(image_pairs_[pair_id].first, 0) << "The scene graph graph must not contain duplicate matches";
        }
    }
}

void ExtractInlierImages(const std::vector<image_t>& images, const size_t num_inliers,
                         const std::vector<char>& inlier_mask, std::vector<image_t>& inlier_images,
                         std::vector<image_t>& outlier_images) {
    inlier_images.resize(num_inliers);
    outlier_images.resize(images.size() - num_inliers);

    size_t j = 0;
    size_t k = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_images[j] = images[i];
            j += 1;
        } else {
            outlier_images[k] = images[i];
            k += 1;
        }
    }
    CHECK_EQ(j, num_inliers);
    CHECK_EQ(k, images.size()-num_inliers);
}

void Reconstruction::ComputePrimaryPlane(const double max_distance_to_plane, const int max_plane_count) {
    ComputeBaselineDistance();

    planes_for_images.clear();
    images_for_planes.clear();

    planes_for_lidars.clear();
    lidars_for_planes.clear();

    EIGEN_STL_UMAP(image_t, Eigen::Vector3d) camera_centers;
    std::vector<image_t> image_ids;
    for (const auto image : images_) {
        if (!image.second.IsRegistered()) {
            continue;
        }
        camera_centers.emplace(image.first, image.second.ProjectionCenter());
        image_ids.push_back(image.first);
    }

    std::vector<image_t> remain_image_ids = image_ids;

    RANSACOptions ransac_options;
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.25;
    ransac_options.min_num_trials = 100;
    ransac_options.max_num_trials = 10000;
    ransac_options.max_error = baseline_distance * max_distance_to_plane;
    
    std::cout<<"max error is: "<<ransac_options.max_error<<std::endl;

    for (int iplane = 0; iplane < max_plane_count; ++iplane) {
        std::vector<Eigen::Vector3d> points;
        for (auto image_id : remain_image_ids) {
            points.push_back(camera_centers.at(image_id));
        }

        LORANSAC<PlaneEstimator, PlaneEstimator> plane_ransac(ransac_options);
        const auto plane_report = plane_ransac.Estimate(points, points);

        if (!plane_report.success || plane_report.support.num_inliers < 100) {
            break;
        }

        std::vector<image_t> inlier_images, outlier_images;
        ExtractInlierImages(remain_image_ids, plane_report.support.num_inliers, plane_report.inlier_mask, inlier_images,
                            outlier_images);

        std::cout << "Inlier plane points: " << inlier_images.size() << std::endl;

        std::vector<Eigen::Vector3d> inlier_image_centers;
        for (const auto inlier_image : inlier_images) {
            planes_for_images.emplace(inlier_image, plane_report.model);
            inlier_image_centers.push_back(camera_centers.at(inlier_image));
        }
        images_for_planes.emplace(iplane, inlier_image_centers);

        remain_image_ids = outlier_images;
    }

    {
        EIGEN_STL_UMAP(sweep_t, Eigen::Vector3d) lidar_centers;
        std::vector<sweep_t> sweep_ids;
        for (const auto sweep_id : RegisterSweepIds()) {
            class LidarSweep & lidar_sweep = LidarSweep(sweep_id);
            lidar_centers.emplace(sweep_id, lidar_sweep.ProjectionCenter());
            sweep_ids.push_back(sweep_id);
        }

        std::vector<sweep_t> remain_sweep_ids = sweep_ids;

        RANSACOptions ransac_options;
        ransac_options.confidence = 0.9999;
        ransac_options.min_inlier_ratio = 0.25;
        ransac_options.min_num_trials = 100;
        ransac_options.max_num_trials = 10000;
        ransac_options.max_error = baseline_distance * max_distance_to_plane;
        
        std::cout<<"max error is: "<<ransac_options.max_error<<std::endl;

        for (int iplane = 0; iplane < max_plane_count; ++iplane) {
            std::vector<Eigen::Vector3d> points;
            for (auto sweep_id : remain_sweep_ids) {
                points.push_back(lidar_centers.at(sweep_id));
            }

            LORANSAC<PlaneEstimator, PlaneEstimator> plane_ransac(ransac_options);
            const auto plane_report = plane_ransac.Estimate(points, points);

            if (!plane_report.success || plane_report.support.num_inliers < 100) {
                break;
            }

            std::vector<sweep_t> inlier_sweeps, outlier_sweeps;
            ExtractInlierImages(remain_sweep_ids, plane_report.support.num_inliers, plane_report.inlier_mask, inlier_sweeps,
                                outlier_sweeps);

            std::cout << "Inlier plane points: " << inlier_sweeps.size() << std::endl;

            std::vector<Eigen::Vector3d> inlier_sweep_centers;
            for (const auto inlier_sweep : inlier_sweeps) {
                planes_for_lidars.emplace(inlier_sweep, plane_report.model);
                inlier_sweep_centers.push_back(lidar_centers.at(inlier_sweep));
            }
            lidars_for_planes.emplace(iplane, inlier_sweep_centers);

            remain_sweep_ids = outlier_sweeps;
        }
    }
}

void Reconstruction::ComputeBaselineDistance() {
    CHECK(register_image_ids_.size() >= 2);

    baseline_distance = 0;
    int distance_count = 0;

    const std::vector<image_t>& registered_images = RegisterImageIds();

    std::vector<float> dists(registered_images.size());
    std::fill(dists.begin(), dists.end(), -1);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0 ; i < registered_images.size(); ++i) {
        image_t image_id1 = registered_images[i];
        CHECK(ExistsImage(image_id1));
        CHECK(IsImageRegistered(image_id1));

        double min_distance =std::numeric_limits<double>::max();
        image_t min_image_id2 = kInvalidImageId;
        for(size_t j = i + 1; j < registered_images.size(); ++j){
            image_t image_id2 = registered_images[j];
            CHECK(ExistsImage(image_id2));
            CHECK(IsImageRegistered(image_id2));
            
            if (correspondence_graph_ && !correspondence_graph_->ExistImagePair(image_id1, image_id2)) {
                continue;
            }
            Eigen::Vector3d baseline = images_.at(image_id1).ProjectionCenter() - 
                images_.at(image_id2).ProjectionCenter();

            if(min_distance > baseline.norm()){
                min_distance = baseline.norm();
                min_image_id2 = image_id2;
            }
        }
        if(min_image_id2 != kInvalidImageId){
            dists[i] = min_distance;
        }
    }

    for (size_t i = 0 ; i < registered_images.size(); ++i) {
        if (dists[i] >= 0) {
            baseline_distance += dists[i];
            distance_count++;
        }
    }
    if (distance_count > 0) {
        baseline_distance /= static_cast<double>(distance_count);
    } else {
        image_t image_id1 = register_image_ids_[0];
        image_t image_id2 = register_image_ids_[1];

        Eigen::Vector3d baseline = images_.at(image_id1).ProjectionCenter() - images_.at(image_id2).ProjectionCenter();
        baseline_distance = baseline.norm();
    }
    std::cout << "consecutive baseline distance count: " << distance_count << std::endl;
    std::cout << "base line distance: "<<baseline_distance<<std::endl;
}

bool Reconstruction::IsolatedImage(int neighbor_scope,const image_t image_id, double baseline_distance) {
    class Image& image = Image(image_id);
    if (!image.IsRegistered()) {
        return false;
    }

    bool b_isolated = true;
    for (int scope = -neighbor_scope; scope <= neighbor_scope; scope++) {
        if (scope == 0) {
            continue;
        }

        if (scope + static_cast<int>(image_id) > 0 &&
            ExistsImage(scope + static_cast<int>(image_id))) {
            class Image& image_neighbor = Image(scope + static_cast<int>(image_id));

            if (image_neighbor.IsRegistered()&& image.LabelId() == image_neighbor.LabelId()) {
                Eigen::Vector3d current_baseline = image.ProjectionCenter() - image_neighbor.ProjectionCenter();

                if (current_baseline.norm() < 3.0 * fabs(static_cast<double>(scope) * baseline_distance)) {
                    b_isolated = false;
                    break;
                }
            }
        }
    }
    return b_isolated;
}

void Reconstruction::WritePlaneForCameras(const std::string& path, const double max_distance_to_plane,
                                          const int max_plane_count) {
    std::ofstream file_planes(path);
    if (!file_planes.is_open()) {
        return;
    }

    ComputePrimaryPlane(max_distance_to_plane, max_plane_count);

    for (const auto plane : images_for_planes) {
        std::vector<Eigen::Vector3d> camera_centers = plane.second;
        int color[3];
        color[0] = rand() % 256;
        color[1] = rand() % 256;
        color[2] = rand() % 256;

        for (const auto xyz : camera_centers) {
            file_planes << "v " << xyz[0] << " " << xyz[1] << " " << xyz[2] << " " << color[0] << " " << color[1] << " "
                        << color[2] << std::endl;
        }
    }
    file_planes.close();
}

void Reconstruction::WriteReconstruction(const std::string& path, bool write_binary) const {
    if (write_binary) {
        WriteBinary(path);
    } else {
        WriteText(path);
    }
}

void Reconstruction::ReadReconstruction(const std::string& path,bool camera_rig) {
    if (ExistsFile(JoinPaths(path, "cameras.bin")) && ExistsFile(JoinPaths(path, "images.bin")) &&
        ExistsFile(JoinPaths(path, "points3D.bin"))) {
        ReadBinary(path,camera_rig);
    } else if (ExistsFile(JoinPaths(path, "cameras.txt")) && ExistsFile(JoinPaths(path, "images.txt")) &&
               ExistsFile(JoinPaths(path, "points3D.txt"))) {
        ReadText(path);
    } else {
        // LOG(FATAL) << "cameras, images, points3D files do not exist at " << path;
        std::cerr << "cameras, images, points3D files do not exist at " << path << std::endl;
        exit(StateCode::MODEL_FILE_IS_NOT_EXIST);
    }
}

void Reconstruction::ReadText(const std::string& path) {
    ReadCamerasText(JoinPaths(path, "cameras.txt"));
    ReadPoints3DText(JoinPaths(path, "points3D.txt"));
    if (ExistsFile(JoinPaths(path, "keyframes_list.txt"))) ReadKeyFramesText(JoinPaths(path, "keyframes_list.txt"));
    ReadImagesText(JoinPaths(path, "images.txt"));
    // if (ExistsFile(JoinPaths(path, "update_images.txt"))) {
    //     auto new_image_ids = ReadUpdateImagesText(JoinPaths(path, "update_images.txt"));
    //     for (auto & image_id : RegisterImageIds()) {
    //         Image(image_id).SetLabelId(0);
    //     }
    //     for (auto & new_image_id : new_image_ids) {
    //         Image(new_image_id).SetLabelId(kInvalidLabelId);
    //     }
    // }
    if (ExistsFile(JoinPaths(path, "color_correction.txt"))) {
        ReadColorCorrectionText(JoinPaths(path, "color_correction.txt"));
    }
}

void Reconstruction::ReadBinary(const std::string& path, bool camera_rig) {
    if (ExistsFile(JoinPaths(path, "local_cameras.bin"))) {
        ReadCamerasBinary(JoinPaths(path, "cameras.bin"), false);
        ReadLocalCamerasBinary(JoinPaths(path, "local_cameras.bin"));
    } else {
        ReadCamerasBinary(JoinPaths(path, "cameras.bin"), camera_rig);
    }
    ReadImagesBinary(JoinPaths(path, "images.bin"));
    if (ExistsFile(JoinPaths(path, "local_images.bin"))) {
        ReadLocalImagesBinary(JoinPaths(path, "local_images.bin"));
    }
    // FIXME:
    if (ExistsFile(JoinPaths(path, "keyframes_list.txt"))) ReadKeyFramesText(JoinPaths(path, "keyframes_list.txt"));
    ReadPoints3DBinary(JoinPaths(path, "points3D.bin"));
    // if (ExistsFile(JoinPaths(path, "update_images.txt"))) {
    //     auto new_image_ids = ReadUpdateImagesText(JoinPaths(path, "update_images.txt"));
    //     for (auto & image_id : RegisterImageIds()) {
    //         Image(image_id).SetLabelId(0);
    //     }
    //     for (auto & new_image_id : new_image_ids) {
    //         Image(new_image_id).SetLabelId(kInvalidLabelId);
    //     }
    // }
    // if (ExistsFile(JoinPaths(path, "color_correction.bin"))) {
    //     ReadColorCorrectionBinary(JoinPaths(path, "color_correction.bin"));
    // }
    if (ExistsFile(JoinPaths(path, "color_correction.txt"))) {
        ReadColorCorrectionText(JoinPaths(path, "color_correction.txt"));
    }

    if (ExistsFile(JoinPaths(path, "lidars.bin"))) {
        ReadLidarBinary(JoinPaths(path, "lidars.bin"));
        ReadLocalLidarBinary(JoinPaths(path, "local_lidars.bin"));
    }
}

void Reconstruction::WriteText(const std::string& path) const {
    WriteCamerasText(JoinPaths(path, "cameras.txt"));
    WriteImagesText(JoinPaths(path, "images.txt"));
    // WriteKeyFramesText(JoinPaths(path, "keyframes_list.txt")); //FIXME: Disable keyframe list output
    WritePoints3DText(JoinPaths(path, "points3D.txt"));
    // WritePoseText(JoinPaths(path, "pose.txt"));
    if (this->GetNewImageIds().size() != 0) {
        WriteUpdateImagesText(JoinPaths(path, "update_images.txt"));
    }
    if (!yrb_factors.empty()) {
        WriteColorCorrectionText(JoinPaths(path, "color_correction.txt"));
    }
}
void Reconstruction::WriteBinary(const std::string& path) const {
    WriteCamerasBinary(JoinPaths(path, "cameras.bin"));
    WriteLocalCamerasBinary(JoinPaths(path, "local_cameras.bin"));
    WriteImagesBinary(JoinPaths(path, "images.bin"));
    WriteLocalImagesBinary(JoinPaths(path, "local_images.bin"));
    // FIXME:
    // WriteKeyFramesText(JoinPaths(path, "keyframes_list.txt")); // FIXME: Disable keyframe list output
    WritePoints3DBinary(JoinPaths(path, "points3D.bin"));
    if (this->GetNewImageIds().size() != 0) {
        WriteUpdateImagesText(JoinPaths(path, "update_images.txt"));
    }
    // WriteColorCorrectionBinary(JoinPaths(path, "color_correction.bin"));
    if (!yrb_factors.empty()) {
        WriteColorCorrectionText(JoinPaths(path, "color_correction.txt"));
    }
    if (register_sweep_ids_.size() > 0) {
        WriteLidarBinary(JoinPaths(path, "lidars.bin"));
        WriteLocalLidarBinary(JoinPaths(path, "local_lidars.bin"));
    }
}

void Reconstruction::WriteCamerasText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# Camera list with one line of data per camera:" << std::endl;
    file << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
    file << "# Number of cameras: " << cameras_.size() << std::endl;

    for (const auto& camera : cameras_) {
        std::ostringstream line;

        line << camera.first << " ";
        line << camera.second.ModelName() << " ";
        line << camera.second.Width() << " ";
        line << camera.second.Height() << " ";

        for (const double param : camera.second.Params()) {
            line << param << " ";
        }

        std::string line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);

        file << line_string << std::endl;
    }
    file.close();
}


void Reconstruction::WriteCameraIsRigText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# Camera list with is_from_rig per camera:" << std::endl;
    file << "#   CAMERA_ID, IS_FROM_RIG" << std::endl;
    file << "# Number of cameras: " << cameras_.size() << std::endl;

    for (const auto& camera : cameras_) {
        std::ostringstream line;

        line << camera.first << " ";
        line << camera.second.IsFromRIG() << " ";

        std::string line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);

        file << line_string << std::endl;
    }
    file.close();
}

void Reconstruction::WriteImagesText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# Image list with two lines of data per image:" << std::endl;
    file << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, "
            "NAME"
         << std::endl;
    file << "#   POINTS2D[] as (X, Y, MAPPOINT_ID)" << std::endl;
    file << "# Number of images: " << register_image_ids_.size()
         << ", mean observations per image: " << ComputeMeanObservationsPerRegImage() << std::endl;

    // int image_write_counter = 0;
    for (const auto& image : images_) {
        if (!image.second.IsRegistered()) {
            continue;
        }

        std::ostringstream line;
        std::string line_string;

        line << image.first << " ";

        // QVEC (qw, qx, qy, qz)
        const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(image.second.Qvec());
        line << normalized_qvec(0) << " ";
        line << normalized_qvec(1) << " ";
        line << normalized_qvec(2) << " ";
        line << normalized_qvec(3) << " ";

        // TVEC
        line << image.second.Tvec(0) << " ";
        line << image.second.Tvec(1) << " ";
        line << image.second.Tvec(2) << " ";

        line << image.second.CameraId() << " ";

        line << image.second.Name();

        file << line.str() << std::endl;

        line.str("");
        line.clear();

        for (const Point2D& point2D : image.second.Points2D()) {
            line << point2D.X() << " ";
            line << point2D.Y() << " ";
            if (point2D.HasMapPoint()) {
                line << point2D.MapPointId() << " ";
            } else {
                line << -1 << " ";
            }
        }
        line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);
        file << line_string << std::endl;

        // std::cout << "\rWrite Image TXT [ " << image_write_counter+1 << " / " << images_.size() << " ]" << std::flush; 
        // image_write_counter++;
    }
    // std::cout << "\n";
    file.close();
}

void Reconstruction::WriteKeyFramesText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    for (const auto& image : images_) {
        if (image.second.IsKeyFrame()) {
            std::ostringstream line;
            line << image.first << " ";
            line << image.second.Name();
            file << line.str() << std::endl;
        }
    }
    file.close();
}

void Reconstruction::WritePoseText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# Image Pose output:" << std::endl;
    file << "#   IMAGE_ID, TX, TY, TZ, QX, QY, QZ, QW " << std::endl;
    file << "#   POINTS2D[] as (X, Y, MAPPOINT_ID)" << std::endl;
    file << "# Number of images: " << register_image_ids_.size()
         << ", mean observations per image: " << ComputeMeanObservationsPerRegImage() << std::endl;
    size_t image_index = 0;

    // Sort the images
    std::vector<class Image> ordered_images;
    ordered_images.reserve(images_.size());
    for (const auto image : images_) {
        ordered_images.push_back(image.second);
    }

    std::sort(ordered_images.begin(), ordered_images.end(),
              [](const class Image& image1, const class Image& image2) { return image1.Name() < image2.Name(); });

    for (const auto& image : ordered_images) {
        if (!image.IsRegistered()) {
            continue;
        }

        std::ostringstream line;
        std::string line_string;

        auto rot = image.RotationMatrix();
        auto tvec = image.Tvec();

        // Transpose
        tvec = -rot.transpose() * tvec;
        auto qvec = RotationMatrixToQuaternion(rot.transpose());

        // TUM
        line << image.Name().substr(0, image.Name().size() - 4) << " ";

        // TVEC
        line << tvec(0) << " ";
        line << tvec(1) << " ";
        line << tvec(2) << " ";

        // QVEC (qx, qy, qz, qw)
        const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
        line << normalized_qvec(1) << " ";
        line << normalized_qvec(2) << " ";
        line << normalized_qvec(3) << " ";
        line << normalized_qvec(0) << " ";

        line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);
        file << line_string << std::endl;

        image_index++;
    }
    file.close();
}

void Reconstruction::WriteLocText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# Image Pose output:" << std::endl;
    file << "#   IMAGE_NAME, TX, TY, TZ, QW, QX, QY, QZ " << std::endl;

    // Sort the images
    std::vector<class Image> ordered_images;
    ordered_images.reserve(images_.size());
    for (const auto image : images_) {
        ordered_images.push_back(image.second);
    }

    std::sort(ordered_images.begin(), ordered_images.end(),
              [](const class Image& image1, const class Image& image2) { return image1.Name() < image2.Name(); });

    for (const auto& image : ordered_images) {
        if (!image.IsRegistered() || image.LabelId() == 0) {
            continue;
        }

        std::ostringstream line;
        std::string line_string;

        auto rot = image.RotationMatrix();
        auto tvec = image.Tvec();

        // Transpose
        tvec = -rot.transpose() * tvec;
        auto qvec = RotationMatrixToQuaternion(rot.transpose());

        line << image.Name() << " ";
        // TVEC
        line << tvec(0) << " ";
        line << tvec(1) << " ";
        line << tvec(2) << " ";

        // QVEC (qw, qx, qy, qz)
        const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
        line << normalized_qvec(0) << " ";
        line << normalized_qvec(1) << " ";
        line << normalized_qvec(2) << " ";
        line << normalized_qvec(3) << " ";

        line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);
        file << line_string << std::endl;
    }
    file.close();
}

void Reconstruction::WritePoints3DText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# 3D point list with one line of data per point:" << std::endl;
    file << "#   MAPPOINT_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)"
         << std::endl;
    file << "# Number of points: " << mappoints_.size() << ", mean track length: " << ComputeMeanTrackLength()
         << std::endl;

    // int mappoint_write_counter = 0;
    for (const auto& mappoint : mappoints_) {
        file << mappoint.first << " ";
        file << mappoint.second.XYZ()(0) << " ";
        file << mappoint.second.XYZ()(1) << " ";
        file << mappoint.second.XYZ()(2) << " ";
        file << static_cast<int>(mappoint.second.Color(0)) << " ";
        file << static_cast<int>(mappoint.second.Color(1)) << " ";
        file << static_cast<int>(mappoint.second.Color(2)) << " ";
        file << mappoint.second.Error() << " ";

        std::ostringstream line;

        for (const auto& track_el : mappoint.second.Track().Elements()) {
            line << track_el.image_id << " ";
            line << track_el.point2D_idx << " ";
        }

        std::string line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);

        file << line_string << std::endl;

        // std::cout << "\rWrite Point3D TXT [ " << mappoint_write_counter+1 << " / " << mappoints_.size() << " ]" << std::flush; 
        // mappoint_write_counter++;
    }
    // std::cout << "\n";
    file.close();
}

void Reconstruction::WriteUpdateImagesText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    auto & new_image_ids = GetNewImageIds();
    // file << new_image_ids.size() << std::endl;

    for (auto image_id : new_image_ids) {
        auto & image = this->Image(image_id);
        file << image_id << " " << image.Name() << std::endl;
    }

    file.close();
}

void Reconstruction::WriteRegisteredImagesInfo(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# Cluster Images output:" << std::endl;
    file << "#   IMAGE_ID IMAGE_NAME " << std::endl;
    
    auto & image_ids = RegisterImageIds();
    // file << new_image_ids.size() << std::endl;

    for (auto image_id : image_ids) {
        auto & image = this->Image(image_id);
        file << image_id << " " << image.Name() << std::endl;
    }

    file.close();
}

void Reconstruction::WriteColorCorrectionText(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    for (auto image_id : RegisterImageIds()) {
        YCrCbFactor param = yrb_factors.at(image_id);
        file << image_id << " " << param.s_Y << " " << param.s_Cb << " " << param.s_Cr << " "
                                << param.o_Y << " " << param.o_Cb << " " << param.o_Cr << std::endl;
    }

    file.close();
}

void Reconstruction::WriteCamerasBinary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, cameras_.size());

    for (const auto& camera : cameras_) {
        WriteBinaryLittleEndian<camera_t>(&file, camera.first);
        WriteBinaryLittleEndian<int>(&file, camera.second.ModelId());
        WriteBinaryLittleEndian<uint64_t>(&file, camera.second.Width());
        WriteBinaryLittleEndian<uint64_t>(&file, camera.second.Height());
        for (const double param : camera.second.Params()) {
            WriteBinaryLittleEndian<double>(&file, param);
        }

        // // for camera-rig
		// if (camera.second.NumLocalCameras() > 1) {
		// 	WriteBinaryLittleEndian<local_camera_t>(&file,camera.second.NumLocalCameras());
		// 	// local intrinsics
		// 	WriteBinaryLittleEndian<size_t>(&file,camera.second.LocalParams().size());
		// 	for (const double local_param: camera.second.LocalParams()){
		// 		WriteBinaryLittleEndian<double>(&file, local_param);
		// 	}
		// 	// local qvecs
		// 	WriteBinaryLittleEndian<size_t>(&file,camera.second.LocalQvecs().size());
		// 	for (const double local_qvec: camera.second.LocalQvecs()){
		// 		WriteBinaryLittleEndian<double>(&file, local_qvec);
		// 	}
		// 	// local tvecs
		// 	WriteBinaryLittleEndian<size_t>(&file, camera.second.LocalTvecs().size());
		// 	for (const double local_tvec : camera.second.LocalTvecs()){
		// 		WriteBinaryLittleEndian<double>(&file, local_tvec);
		// 	}
		// }
    }
    file.close();
}

void Reconstruction::WriteCameraIsRigBinary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, cameras_.size());

    for (const auto& camera : cameras_) {
        WriteBinaryLittleEndian<camera_t>(&file, camera.first);
        WriteBinaryLittleEndian<std::size_t>(&file, camera.second.IsFromRIG());

    }
    file.close();
}

void Reconstruction::WriteLocalCamerasBinary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, cameras_.size());

    for (const auto& camera : cameras_) {
        WriteBinaryLittleEndian<camera_t>(&file, camera.first);
		WriteBinaryLittleEndian<local_camera_t>(&file,camera.second.NumLocalCameras());

        // for camera-rig
		if (camera.second.NumLocalCameras() > 1) {
			// local intrinsics
			WriteBinaryLittleEndian<size_t>(&file,camera.second.LocalParams().size());
			for (const double local_param: camera.second.LocalParams()){
				WriteBinaryLittleEndian<double>(&file, local_param);
			}
			// local qvecs
			WriteBinaryLittleEndian<size_t>(&file,camera.second.LocalQvecs().size());
			for (const double local_qvec: camera.second.LocalQvecs()){
				WriteBinaryLittleEndian<double>(&file, local_qvec);
			}
			// local tvecs
			WriteBinaryLittleEndian<size_t>(&file, camera.second.LocalTvecs().size());
			for (const double local_tvec : camera.second.LocalTvecs()){
				WriteBinaryLittleEndian<double>(&file, local_tvec);
			}
		}
    }
    file.close();
}

void Reconstruction::WriteImagesBinary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, register_image_ids_.size());

    // int write_image_counter = 0;
    for (const auto& image : images_) {
        if (!image.second.IsRegistered()) {
            continue;
        }

        WriteBinaryLittleEndian<image_t>(&file, image.first);

        const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(image.second.Qvec());
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(0));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(1));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(2));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(3));

        WriteBinaryLittleEndian<double>(&file, image.second.Tvec(0));
        WriteBinaryLittleEndian<double>(&file, image.second.Tvec(1));
        WriteBinaryLittleEndian<double>(&file, image.second.Tvec(2));

        WriteBinaryLittleEndian<camera_t>(&file, image.second.CameraId());

        const std::string name = image.second.Name() + '\0';
        file.write(name.c_str(), name.size());

        WriteBinaryLittleEndian<uint64_t>(&file, image.second.NumPoints2D());
        for (const Point2D& point2D : image.second.Points2D()) {
            WriteBinaryLittleEndian<double>(&file, point2D.X());
            WriteBinaryLittleEndian<double>(&file, point2D.Y());
            WriteBinaryLittleEndian<mappoint_t>(&file, point2D.MapPointId());
        }
        // std::cout << "\rWrite Image Binary [ " << write_image_counter << " / " << images_.size() << " ]" << std::flush; 
        // write_image_counter++;
    }

    // std::cout << "\n";
    file.close();
}

void Reconstruction::WriteLocalImagesBinary(const std::string& path) const {
	std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, register_image_ids_.size());

    for (const auto& image : images_) {
        if (!IsImageRegistered(image.first)){
            continue;
        }
        WriteBinaryLittleEndian<image_t>(&file, image.first);

        auto camera = cameras_.at(image.second.CameraId());
        local_camera_t num_local_camera = camera.NumLocalCameras();
        WriteBinaryLittleEndian<uint32_t>(&file, num_local_camera);

        if (num_local_camera <= 1) continue;

		const auto & local_tvec_priors = image.second.LocalTvecsPrior();

		WriteBinaryLittleEndian<local_camera_t>(&file, local_tvec_priors.size());
		for (size_t local_image_id = 0; local_image_id < local_tvec_priors.size(); ++local_image_id) {
			const Eigen::Vector3d& tvec = local_tvec_priors[local_image_id];
			WriteBinaryLittleEndian<double>(&file, tvec.x());
			WriteBinaryLittleEndian<double>(&file, tvec.y());
			WriteBinaryLittleEndian<double>(&file, tvec.z());
		}

		const auto & local_qvec_priors = image.second.LocalQvecsPrior();

		WriteBinaryLittleEndian<local_camera_t>(&file, local_qvec_priors.size());
		for (size_t local_image_id = 0; local_image_id < local_qvec_priors.size(); ++local_image_id) {
			const Eigen::Vector4d& qvec = local_qvec_priors[local_image_id];
			WriteBinaryLittleEndian<double>(&file, qvec.w());
			WriteBinaryLittleEndian<double>(&file, qvec.x());
			WriteBinaryLittleEndian<double>(&file, qvec.y());
			WriteBinaryLittleEndian<double>(&file, qvec.z());
		}

        WriteBinaryLittleEndian<local_camera_t>(&file, image.second.LocalNames().size());
        for (size_t local_image_id = 0; local_image_id < image.second.LocalNames().size(); ++local_image_id) {
            const std::string image_name = image.second.LocalName(local_image_id) + '\0';
            file.write(image_name.c_str(), image_name.size());
        }
    }
    file.close();
}

void Reconstruction::WritePoints3DBinary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, mappoints_.size());

    // int point3d_write_counter = 0;
    for (const auto& point3D : mappoints_) {
        WriteBinaryLittleEndian<mappoint_t>(&file, point3D.first);
        WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(0));
        WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(1));
        WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(2));
        WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(0));
        WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(1));
        WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(2));
        WriteBinaryLittleEndian<double>(&file, point3D.second.Error());


        WriteBinaryLittleEndian<uint64_t>(&file, point3D.second.Track().Length());
        for (const auto& track_el : point3D.second.Track().Elements()) {
            WriteBinaryLittleEndian<image_t>(&file, track_el.image_id);
            WriteBinaryLittleEndian<point2D_t>(&file, track_el.point2D_idx);
        }
        // std::cout << "\rWrite Point3D Binary [ " << point3d_write_counter+1 << " / " << mappoints_.size() << " ]" << std::flush; 
        // point3d_write_counter++;
    }
    // std::cout << "\n";
    file.close();
}

void Reconstruction::ExportMapPoints(const std::string& path) const {
    const auto & mappoint_ids = MapPointIds();

    std::vector<PlyPoint> points;
    points.reserve(mappoint_ids.size());
    
    for (const auto & mappoint_id : mappoint_ids) {
        const class MapPoint & mappoint = MapPoint(mappoint_id);
        PlyPoint point;
        point.x = mappoint.X();
        point.y = mappoint.Y();
        point.z = mappoint.Z();
        point.r = mappoint.Color(0);
        point.g = mappoint.Color(1);
        point.b = mappoint.Color(2);
        points.push_back(point);
    }
    WriteBinaryPlyPoints(path, points, false, true);
}

void Reconstruction::WriteColorCorrectionBinary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    WriteBinaryLittleEndian<image_t>(&file, NumRegisterImages());
    for (auto image_id : RegisterImageIds()) {
        YCrCbFactor param = yrb_factors.at(image_id);
        WriteBinaryLittleEndian<image_t>(&file, image_id);
        WriteBinaryLittleEndian<double>(&file, param.s_Y);
        WriteBinaryLittleEndian<double>(&file, param.s_Cb);
        WriteBinaryLittleEndian<double>(&file, param.s_Cr);
        WriteBinaryLittleEndian<double>(&file, param.o_Y);
        WriteBinaryLittleEndian<double>(&file, param.o_Cb);
        WriteBinaryLittleEndian<double>(&file, param.o_Cr);
    }

    file.close();
}

void Reconstruction::WriteLidarBinary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, register_sweep_ids_.size());

    for (const auto& lidar_sweep : lidarsweeps_) {
        if (!lidar_sweep.second.IsRegistered()) {
            continue;
        }

        WriteBinaryLittleEndian<sweep_t>(&file, lidar_sweep.first);

        const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(lidar_sweep.second.Qvec());
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(0));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(1));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(2));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(3));

        WriteBinaryLittleEndian<double>(&file, lidar_sweep.second.Tvec(0));
        WriteBinaryLittleEndian<double>(&file, lidar_sweep.second.Tvec(1));
        WriteBinaryLittleEndian<double>(&file, lidar_sweep.second.Tvec(2));

        const std::string name = lidar_sweep.second.Name() + '\0';
        file.write(name.c_str(), name.size());
    }
    file.close();
}

void Reconstruction::WriteLocalLidarBinary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(0, 0));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(0, 1));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(0, 2));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(0, 3));

    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(1, 0));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(1, 1));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(1, 2));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(1, 3));

    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(2, 0));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(2, 1));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(2, 2));
    WriteBinaryLittleEndian<double>(&file, lidar_to_cam_matrix(2, 3));

    file.close();
}

void Reconstruction::ReadCamerasBinary(const std::string& path,bool camera_rig) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_cameras; ++i) {
        class Camera camera;
        camera.SetCameraId(ReadBinaryLittleEndian<camera_t>(&file));
        camera.SetModelId(ReadBinaryLittleEndian<int>(&file));
        camera.SetWidth(ReadBinaryLittleEndian<uint64_t>(&file));
        camera.SetHeight(ReadBinaryLittleEndian<uint64_t>(&file));
        ReadBinaryLittleEndian<double>(&file, &camera.Params());
        CHECK(camera.VerifyParams());

        if (camera_rig) {
			local_camera_t num_local_camera = 
				ReadBinaryLittleEndian<local_camera_t>(&file);
			// for camera-rig
			camera.SetNumLocalCameras(num_local_camera);

			// Intrinsics
			size_t local_params_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalParams().resize(local_params_size);
			ReadBinaryLittleEndian<double>(&file,&camera.LocalParams());
			
			// Local qvecs
			size_t local_qvecs_size,local_tvecs_size;
			local_qvecs_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalQvecs().resize(local_qvecs_size);
			ReadBinaryLittleEndian<double>(&file, &camera.LocalQvecs());

			// Local tvecs
			local_tvecs_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalTvecs().resize(local_tvecs_size);
			ReadBinaryLittleEndian<double>(&file, &camera.LocalTvecs());

			CHECK(camera.VerifyLocalParams());
		}

        cameras_.emplace(camera.CameraId(), camera);
    }
    file.close();
}

void Reconstruction::ReadCameraIsRigBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_cameras; ++i) {
        camera_t camera_id = ReadBinaryLittleEndian<camera_t>(&file);
        cameras_[camera_id].SetFromRIG(ReadBinaryLittleEndian<int>(&file));
    }
    file.close();
}


void Reconstruction::ReadLocalCamerasBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_cameras; ++i) {
        camera_t camera_id = ReadBinaryLittleEndian<camera_t>(&file);
        local_camera_t num_local_camera = ReadBinaryLittleEndian<local_camera_t>(&file);
        class Camera& camera = cameras_.at(camera_id);

        if (num_local_camera > 1) {
			// for camera-rig
			camera.SetNumLocalCameras(num_local_camera);

			// Intrinsics
			size_t local_params_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalParams().resize(local_params_size);
			ReadBinaryLittleEndian<double>(&file,&camera.LocalParams());

			// Local qvecs
			size_t local_qvecs_size,local_tvecs_size;
			local_qvecs_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalQvecs().resize(local_qvecs_size);
			ReadBinaryLittleEndian<double>(&file, &camera.LocalQvecs());

			// Local tvecs
			local_tvecs_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalTvecs().resize(local_tvecs_size);
			ReadBinaryLittleEndian<double>(&file, &camera.LocalTvecs());

			CHECK(camera.VerifyLocalParams());
            std::cout << "camera id: " << camera_id << ", local param = " << VectorToCSV(camera.LocalParams()) << std::endl
                      << "local qvecs = "<<VectorToCSV(camera.LocalQvecs())<<std::endl
                      << "local tvecs = " << VectorToCSV(camera.LocalTvecs())<<std::endl; 
		}
    }
    file.close();
}

void Reconstruction::ReadImagesBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_reg_images; ++i) {
        class Image image;

        image.SetImageId(ReadBinaryLittleEndian<image_t>(&file));

        image.Qvec(0) = ReadBinaryLittleEndian<double>(&file);
        image.Qvec(1) = ReadBinaryLittleEndian<double>(&file);
        image.Qvec(2) = ReadBinaryLittleEndian<double>(&file);
        image.Qvec(3) = ReadBinaryLittleEndian<double>(&file);
        image.NormalizeQvec();

        image.Tvec(0) = ReadBinaryLittleEndian<double>(&file);
        image.Tvec(1) = ReadBinaryLittleEndian<double>(&file);
        image.Tvec(2) = ReadBinaryLittleEndian<double>(&file);

        image.SetCameraId(ReadBinaryLittleEndian<camera_t>(&file));

        char name_char;
        do {
            file.read(&name_char, 1);
            if (name_char != '\0') {
                image.Name() += name_char;
            }
        } while (name_char != '\0');

        const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&file);

        std::vector<Eigen::Vector2d> points2D;
        points2D.reserve(num_points2D);
        std::vector<mappoint_t> point3D_ids;
        point3D_ids.reserve(num_points2D);
        for (size_t j = 0; j < num_points2D; ++j) {
            const double x = ReadBinaryLittleEndian<double>(&file);
            const double y = ReadBinaryLittleEndian<double>(&file);
            points2D.emplace_back(x, y);
            point3D_ids.push_back(ReadBinaryLittleEndian<mappoint_t>(&file));
        }

        if (ExistsCamera(image.CameraId())){
            image.SetUp(Camera(image.CameraId()));
        }
        image.SetPoints2D(points2D);

        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
            if (point3D_ids[point2D_idx] != kInvalidMapPointId) {
                image.SetMapPointForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
            }
        }

        image.SetRegistered(true);
        image.SetPoseFlag(true);
        register_image_ids_.push_back(image.ImageId());

        images_.emplace(image.ImageId(), image);

        // std::cout << "\rLoad Image Binary [ " << i+1 << " / " << num_reg_images << " ]" << std::flush; 
    }
    // std::cout << "\n";
    file.close();
}

void Reconstruction::ReadLocalImagesBinary(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_images = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_images; ++i) {
        image_t image_id = ReadBinaryLittleEndian<image_t>(&file);

        uint32_t num_local_camera = ReadBinaryLittleEndian<uint32_t>(&file);
        if (!ExistsImage(image_id)){
            std::cout << "LocalImages !ExistsImage(image_id): " << image_id << std::endl;
            continue;
        }
        if (num_local_camera <= 1) continue;
		auto& image = images_.at(image_id);

		local_camera_t num_local_tvec_priors = ReadBinaryLittleEndian<local_camera_t>(&file);
		auto& local_tvecs_prior = image.LocalTvecsPrior();
		for (local_camera_t local_image_id = 0; local_image_id < num_local_tvec_priors; ++local_image_id) {
			Eigen::Vector3d tvec;
			tvec.x() = ReadBinaryLittleEndian<double>(&file);
			tvec.y() = ReadBinaryLittleEndian<double>(&file);
			tvec.z() = ReadBinaryLittleEndian<double>(&file);
			local_tvecs_prior.push_back(tvec);
		}

		local_camera_t num_local_qvec_priors = ReadBinaryLittleEndian<local_camera_t>(&file);
		auto& local_qvecs_prior = image.LocalQvecsPrior();
		for (local_camera_t local_image_id = 0; local_image_id < num_local_qvec_priors; ++local_image_id) {
			Eigen::Vector4d qvec;
			qvec.w() = ReadBinaryLittleEndian<double>(&file);
			qvec.x() = ReadBinaryLittleEndian<double>(&file);
			qvec.y() = ReadBinaryLittleEndian<double>(&file);
			qvec.z() = ReadBinaryLittleEndian<double>(&file);
			local_qvecs_prior.push_back(qvec);
		}

		local_camera_t num_local_image_names = ReadBinaryLittleEndian<local_camera_t>(&file);
		std::vector<std::string> local_image_names(num_local_image_names, "");
		for (local_camera_t local_image_id = 0; local_image_id < num_local_image_names; ++local_image_id) {
			char image_name_char;
			std::string image_name;
			do {
				file.read(&image_name_char, 1);
				if (image_name_char != '\0') {
					image_name += image_name_char;
				}
			} while (image_name_char != '\0');
			local_image_names[local_image_id] = image_name;
		}
		image.SetLocalNames(local_image_names);
    }
    file.close();
}

void Reconstruction::ReadPoints3DBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_points3D = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_points3D; ++i) {
        class MapPoint point3D;

        const mappoint_t point3D_id = ReadBinaryLittleEndian<mappoint_t>(&file);
        num_added_mappoints_ = std::max(num_added_mappoints_, point3D_id);

        point3D.XYZ()(0) = ReadBinaryLittleEndian<double>(&file);
        point3D.XYZ()(1) = ReadBinaryLittleEndian<double>(&file);
        point3D.XYZ()(2) = ReadBinaryLittleEndian<double>(&file);
        point3D.Color(0) = ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.Color(1) = ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.Color(2) = ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.SetError(ReadBinaryLittleEndian<double>(&file));

        const size_t track_length = ReadBinaryLittleEndian<uint64_t>(&file);
        for (size_t j = 0; j < track_length; ++j) {
            const image_t image_id = ReadBinaryLittleEndian<image_t>(&file);
            const point2D_t point2D_idx = ReadBinaryLittleEndian<point2D_t>(&file);
            point3D.Track().AddElement(image_id, point2D_idx);
        }
        point3D.Track().Compress();

        mappoints_.emplace(point3D_id, point3D);
        // std::cout << "\rLoad Point3D Binary [ " << i << " / " << num_points3D << " ]" << std::flush; 
    }
    // std::cout << "\n";
    file.close();
}

void Reconstruction::ReadCamerasText(const std::string& path) {
    cameras_.clear();

    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);

        class Camera camera;

        // ID
        std::getline(line_stream, item, ' ');
        camera.SetCameraId(std::stoul(item));

        // MODEL
        std::getline(line_stream, item, ' ');
        camera.SetModelIdFromName(item);

        // WIDTH
        std::getline(line_stream, item, ' ');
        camera.SetWidth(std::stoll(item));

        // HEIGHT
        std::getline(line_stream, item, ' ');
        camera.SetHeight(std::stoll(item));

        // PARAMS
        camera.Params().clear();
        while (!line_stream.eof()) {
            std::getline(line_stream, item, ' ');
            camera.Params().push_back(std::stold(item));
        }
        camera.SetPriorFocalLength(true);
        CHECK(camera.VerifyParams());

        cameras_.emplace(camera.CameraId(), camera);
    }
    file.close();
}


void Reconstruction::ReadCameraIsRigText(const std::string& path) {

    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream line_stream(line);

        // ID
        std::getline(line_stream, item, ' ');
        camera_t camera_id = std::stoul(item);

        // Is from rig
        std::getline(line_stream, item, ' ');
        cameras_[camera_id].SetFromRIG(std::stoul(item));
    }
    file.close();
}

void Reconstruction::ReadImagesText(const std::string& path) {
    images_.clear();
    register_image_ids_.clear();

    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream1(line);

        // ID
        std::getline(line_stream1, item, ' ');
        const image_t image_id = std::stoul(item);

        class Image image;
        image.SetImageId(image_id);

        image.SetRegistered(true);
        image.SetPoseFlag(true);
        register_image_ids_.push_back(image_id);

        // QVEC (qw, qx, qy, qz)
        std::getline(line_stream1, item, ' ');
        image.Qvec(0) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(1) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(2) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(3) = std::stold(item);

        image.NormalizeQvec();

        // TVEC
        std::getline(line_stream1, item, ' ');
        image.Tvec(0) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Tvec(1) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Tvec(2) = std::stold(item);

        // CAMERA_ID
        std::getline(line_stream1, item, ' ');
        image.SetCameraId(std::stoul(item));

        // NAME
        std::getline(line_stream1, item, ' ');
        image.SetName(item);

        // POINTS2D
        if (!std::getline(file, line)) {
            break;
        }

        StringTrim(&line);
        std::stringstream line_stream2(line);

        std::vector<Eigen::Vector2d> points2D;
        std::vector<mappoint_t> point3D_ids;

        if (!line.empty()) {
            while (!line_stream2.eof()) {
                Eigen::Vector2d point;

                std::getline(line_stream2, item, ' ');
                point.x() = std::stold(item);

                std::getline(line_stream2, item, ' ');
                point.y() = std::stold(item);

                points2D.push_back(point);

                std::getline(line_stream2, item, ' ');
                if (item == "-1") {
                    point3D_ids.push_back(kInvalidMapPointId);
                } else {
                    point3D_ids.push_back(std::stoll(item));
                }
            }
        }

        image.SetUp(Camera(image.CameraId()));
        image.SetPoints2D(points2D);

        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
            if (point3D_ids[point2D_idx] != kInvalidMapPointId) {
                image.SetMapPointForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
            }
        }

        images_.emplace(image.ImageId(), image);
    }
    file.close();
}

void Reconstruction::ReadKeyFramesText(const std::string& path) {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream1(line);

        // ID
        std::getline(line_stream1, item, ' ');
        const image_t image_id = std::stoul(item);

        if (ExistsImage(image_id)) {
            Image(image_id).SetKeyFrame(true);
        }

        // NAME
        std::getline(line_stream1, item, ' ');
    }

    file.close();
}

void Reconstruction::ReadPoints3DText(const std::string& path) {
    mappoints_.clear();
    num_added_mappoints_ = 0;
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);

        // ID
        std::getline(line_stream, item, ' ');
        const mappoint_t point3D_id = std::stoll(item);

        // Make sure, that we can add new 3D points after reading 3D points
        // without overwriting existing 3D points.
        num_added_mappoints_ = std::max(num_added_mappoints_, point3D_id);

        class MapPoint point3D;

        // XYZ
        std::getline(line_stream, item, ' ');
        point3D.XYZ(0) = std::stold(item);

        std::getline(line_stream, item, ' ');
        point3D.XYZ(1) = std::stold(item);

        std::getline(line_stream, item, ' ');
        point3D.XYZ(2) = std::stold(item);

        // Color
        std::getline(line_stream, item, ' ');
        point3D.Color(0) = static_cast<uint8_t>(std::stoi(item));

        std::getline(line_stream, item, ' ');
        point3D.Color(1) = static_cast<uint8_t>(std::stoi(item));

        std::getline(line_stream, item, ' ');
        point3D.Color(2) = static_cast<uint8_t>(std::stoi(item));

        // ERROR
        std::getline(line_stream, item, ' ');
        point3D.SetError(std::stold(item));

        // TRACK
        while (!line_stream.eof()) {
            TrackElement track_el;

            std::getline(line_stream, item, ' ');
            StringTrim(&item);
            if (item.empty()) {
                break;
            }
            track_el.image_id = std::stoul(item);

            std::getline(line_stream, item, ' ');
            track_el.point2D_idx = std::stoul(item);

            point3D.Track().AddElement(track_el);
        }

        point3D.Track().Compress();

        mappoints_.emplace(point3D_id, point3D);
    }
    file.close();
}

void Reconstruction::ReadPoseText(const std::string& path) {
    images_.clear();
    register_image_ids_.clear();

    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    image_t image_counter = 1;
    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream1(line);

        // ID
        std::getline(line_stream1, item, ' ');
        const std::string image_name = item;

        class Image image;
        image.SetImageId(image_counter);

        image.SetRegistered(true);
        register_image_ids_.push_back(image_counter);

        // TVEC
        Eigen::Vector3d tvec_wc;
        Eigen::Vector4d qvec_wc;

        std::getline(line_stream1, item, ' ');
        tvec_wc[0] = std::stold(item);
        std::getline(line_stream1, item, ' ');
        tvec_wc[1] = std::stold(item);
        std::getline(line_stream1, item, ' ');
        tvec_wc[2] = std::stold(item);
        
        // QVEC (qw, qx, qy, qz)
        std::getline(line_stream1, item, ' ');
        qvec_wc[0] = std::stold(item);
        std::getline(line_stream1, item, ' ');
        qvec_wc[1] = std::stold(item);
        std::getline(line_stream1, item, ' ');
        qvec_wc[2] = std::stold(item);
        std::getline(line_stream1, item, ' ');
        qvec_wc[3] = std::stold(item);


        qvec_wc = NormalizeQuaternion(qvec_wc);
        Eigen::Vector4d qvec_cw = Eigen::Vector4d(qvec_wc[0], -qvec_wc[1], -qvec_wc[2], -qvec_wc[3]);
        const Eigen::Quaterniond quat(qvec_cw(0), qvec_cw(1), qvec_cw(2), qvec_cw(3));
        Eigen::Vector3d tvec_cw = quat * -tvec_wc;


        image.Tvec() = tvec_cw;
        image.Qvec() = qvec_cw;

        image.NormalizeQvec();

        image.SetCameraId(1);
        image.SetName(image_name);

        images_.emplace(image.ImageId(), image);

        image_counter++;
    }
    file.close();
}

std::vector<image_t> Reconstruction::ReadUpdateImagesText(const std::string& path) {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    std::vector<image_t> image_ids;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream1(line);

        // IMAGE_ID
        std::getline(line_stream1, item, ' ');
        const image_t image_id = std::stoul(item);
        // IMAGE_NAME
        std::getline(line_stream1, item, ' ');
        const std::string image_name = item;
        image_ids.emplace_back(image_id);
    }

    file.close();

    return image_ids;
}

void Reconstruction::ReadColorCorrectionText(const std::string& path) {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    image_t max_image_id = 0;
    std::unordered_map<image_t, YCrCbFactor> yrb_factor_map;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream1(line);

        // IMAGE_ID
        std::getline(line_stream1, item, ' ');
        const image_t image_id = std::stoul(item);

        YCrCbFactor param;
        // COLOR CORRECTION PARAM
        std::getline(line_stream1, item, ' ');
        param.s_Y = std::stold(item);
        std::getline(line_stream1, item, ' ');
        param.s_Cb = std::stold(item);
        std::getline(line_stream1, item, ' ');
        param.s_Cr = std::stold(item);
        std::getline(line_stream1, item, ' ');
        param.o_Y = std::stold(item);
        std::getline(line_stream1, item, ' ');
        param.o_Cb = std::stold(item);
        std::getline(line_stream1, item, ' ');
        param.o_Cr = std::stold(item);
        yrb_factor_map[image_id] = param;
        max_image_id = std::max(max_image_id, image_id);
    }

    yrb_factors.resize(max_image_id + 1);
    yrb_factors.shrink_to_fit();
    for (auto yrb_factor : yrb_factor_map) {
        yrb_factors[yrb_factor.first] = yrb_factor.second;
    }

    file.close();
}

void Reconstruction::WriteAlignmentBinary(const std::string& path, const std::unordered_set<image_t>& image_ids) const {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    constexpr uint64_t version = 1;
    WriteBinaryLittleEndian<uint64_t>(&file, 1);

    std::unordered_set<camera_t> camera_ids;
    for (auto image_id : image_ids) {
        camera_t camera_id = Image(image_id).CameraId();
        camera_ids.insert(camera_id);
    }
    WriteBinaryLittleEndian<uint64_t>(&file, camera_ids.size());
    for (const auto& camera_id : camera_ids) {
        const auto& camera = Camera(camera_id);
        WriteBinaryLittleEndian<camera_t>(&file, camera.CameraId());

        WriteBinaryLittleEndian<int>(&file, camera.ModelId());
        WriteBinaryLittleEndian<uint64_t>(&file, camera.Width());
        WriteBinaryLittleEndian<uint64_t>(&file, camera.Height());
        for (const double param : camera.Params()) {
            WriteBinaryLittleEndian<double>(&file, param);
        }
    }

    WriteBinaryLittleEndian<uint64_t>(&file, image_ids.size());
    for (const auto& image_id : image_ids) {
        const auto& image = Image(image_id);
        WriteBinaryLittleEndian<image_t>(&file, image.ImageId());

        const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(image.Qvec());
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(0));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(1));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(2));
        WriteBinaryLittleEndian<double>(&file, normalized_qvec(3));

        WriteBinaryLittleEndian<double>(&file, image.Tvec(0));
        WriteBinaryLittleEndian<double>(&file, image.Tvec(1));
        WriteBinaryLittleEndian<double>(&file, image.Tvec(2));

        WriteBinaryLittleEndian<camera_t>(&file, image.CameraId());

        const std::string name = image.Name() + '\0';
        file.write(name.c_str(), name.size());
    }

    file.close();
}
std::unordered_set<image_t> Reconstruction::ReadAlignmentBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const uint64_t version = ReadBinaryLittleEndian<uint64_t>(&file);
    CHECK(version == 1);

    camera_t max_camera_id = 0;
    for (const auto & camera : cameras_) {
        max_camera_id = std::max(max_camera_id, camera.first);
    }

    std::unordered_map<camera_t, camera_t> camera_ids_map;
    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_cameras; ++i) {
        class Camera camera;
        camera_t src_camera_id = ReadBinaryLittleEndian<camera_t>(&file);
        camera_t dst_camera_id = ++max_camera_id;
        camera_ids_map[src_camera_id] = dst_camera_id;

        camera.SetCameraId(dst_camera_id);
        camera.SetModelId(ReadBinaryLittleEndian<int>(&file));
        camera.SetWidth(ReadBinaryLittleEndian<uint64_t>(&file));
        camera.SetHeight(ReadBinaryLittleEndian<uint64_t>(&file));
        ReadBinaryLittleEndian<double>(&file, &camera.Params());
        CHECK(camera.VerifyParams());

        cameras_.emplace(camera.CameraId(), camera);
    }

    camera_t max_image_id = 0;
    for (const auto & image : images_) {
        max_image_id = std::max(max_image_id, image.first);
    }

    std::unordered_set<image_t> image_ids;
    const size_t num_images = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_images; ++i) {
        class Image image;

        image_t src_image_id = ReadBinaryLittleEndian<camera_t>(&file);
        image_t dst_image_id = ++max_image_id;
        image_ids.insert(dst_image_id);

        image.SetImageId(dst_image_id);

        image.Qvec(0) = ReadBinaryLittleEndian<double>(&file);
        image.Qvec(1) = ReadBinaryLittleEndian<double>(&file);
        image.Qvec(2) = ReadBinaryLittleEndian<double>(&file);
        image.Qvec(3) = ReadBinaryLittleEndian<double>(&file);
        image.NormalizeQvec();

        image.Tvec(0) = ReadBinaryLittleEndian<double>(&file);
        image.Tvec(1) = ReadBinaryLittleEndian<double>(&file);
        image.Tvec(2) = ReadBinaryLittleEndian<double>(&file);

        camera_t src_camera_id = ReadBinaryLittleEndian<camera_t>(&file);
        camera_t dst_camera_id = camera_ids_map[src_camera_id];
        image.SetCameraId(dst_camera_id);

        char name_char;
        do {
            file.read(&name_char, 1);
            if (name_char != '\0') {
                image.Name() += name_char;
            }
        } while (name_char != '\0');

        image.SetRegistered(true);
        register_image_ids_.push_back(image.ImageId());

        images_.emplace(image.ImageId(), image); 
    }

    file.close();
    return image_ids;
}

void Reconstruction::ReadColorCorrectionBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    image_t max_image_id = 0;
    std::unordered_map<image_t, YCrCbFactor> yrb_factor_map;

    const size_t num_images = ReadBinaryLittleEndian<uint32_t>(&file);
    for (size_t i = 0; i < num_images; ++i) {
        image_t image_id = ReadBinaryLittleEndian<image_t>(&file);
        YCrCbFactor param;
        param.s_Y = ReadBinaryLittleEndian<double>(&file);
        param.s_Cb = ReadBinaryLittleEndian<double>(&file);
        param.s_Cr = ReadBinaryLittleEndian<double>(&file);
        param.o_Y = ReadBinaryLittleEndian<double>(&file);
        param.o_Cb = ReadBinaryLittleEndian<double>(&file);
        param.o_Cr = ReadBinaryLittleEndian<double>(&file);
        yrb_factor_map[image_id] = param;
        max_image_id = std::max(max_image_id, image_id);
    }

    yrb_factors.resize(max_image_id + 1);
    yrb_factors.shrink_to_fit();
    for (auto yrb_factor : yrb_factor_map) {
        yrb_factors[yrb_factor.first] = yrb_factor.second;
    }

    file.close();
}

void Reconstruction::ReadLidarBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    uint64_t num_register_sweep = 0;
    num_register_sweep = ReadBinaryLittleEndian<uint64_t>(&file);

    for (int i = 0; i < num_register_sweep; ++i) {
        sweep_t sweep_id = ReadBinaryLittleEndian<sweep_t>(&file);

        Eigen::Vector4d qvec;
        qvec[0] = ReadBinaryLittleEndian<double>(&file);
        qvec[1] = ReadBinaryLittleEndian<double>(&file);
        qvec[2] = ReadBinaryLittleEndian<double>(&file);
        qvec[3] = ReadBinaryLittleEndian<double>(&file);

        Eigen::Vector3d tvec;
        tvec[0] = ReadBinaryLittleEndian<double>(&file);
        tvec[1] = ReadBinaryLittleEndian<double>(&file);
        tvec[2] = ReadBinaryLittleEndian<double>(&file);
        
        std::string sweep_name;
        char name_char;
        do {
            file.read(&name_char, 1);
            if (name_char != '\0') {
                sweep_name += name_char;
                // lidarsweep.Name() += name_char;
            }
        } while (name_char != '\0');

        class LidarSweep lidarsweep(sweep_id, sweep_name);
        lidarsweep.Qvec() = qvec;
        lidarsweep.Tvec() = tvec;

        lidarsweep.SetRegistered(true);
        lidarsweeps_[sweep_id] = lidarsweep;
        register_sweep_ids_.push_back(sweep_id);
    }
    file.close();
}

void Reconstruction::ReadLocalLidarBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    lidar_to_cam_matrix(0, 0) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(0, 1) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(0, 2) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(0, 3) = ReadBinaryLittleEndian<double>(&file);

    lidar_to_cam_matrix(1, 0) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(1, 1) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(1, 2) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(1, 3) = ReadBinaryLittleEndian<double>(&file);

    lidar_to_cam_matrix(2, 0) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(2, 1) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(2, 2) = ReadBinaryLittleEndian<double>(&file);
    lidar_to_cam_matrix(2, 3) = ReadBinaryLittleEndian<double>(&file);

    file.close();
}

void Reconstruction::OutputPriorResiduals(){

    const std::vector<image_t>& registered_images = RegisterImageIds();
    std::vector<Eigen::Vector3d> prior_locations;
    std::vector<Eigen::Vector3d> estimated_locations;
    
    for (size_t i = 0 ; i < registered_images.size(); ++i) {
        image_t image_id = registered_images[i];
        class Image&  image = images_.at(image_id);
        
        if(prior_locations_gps.find(image_id) != prior_locations_gps.end()){

            CHECK(prior_locations_gps_inlier.find(image_id)!= prior_locations_gps_inlier.end());
            
            if(prior_locations_gps_inlier.at(image_id)){
    
                Eigen::Vector3d gps_prior = prior_locations_gps.at(image_id).first;
                Eigen::Vector3d camera_center = image.ProjectionCenter(); 

                std::cout<<"image id difference: "<<(gps_prior-camera_center).norm()<<std::endl;

                prior_locations.push_back(gps_prior);
                estimated_locations.push_back(camera_center);
            }
        }
    }

    RANSACOptions ransac_options;
    ransac_options.max_error = 3; //the maximum error is 3 meters
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.25;
    ransac_options.min_num_trials = 200;
    ransac_options.max_num_trials = 10000;

    LORANSAC<CameraAlignmentEstimator, CameraAlignmentEstimator> ransac(ransac_options);

    auto report = ransac.Estimate(estimated_locations,prior_locations);

    std::cout<<"Restimate transform inlier num: "<<report.support.num_inliers<<" transform the reconstruction"<<std::endl;
    std::cout<<"Restimate transform: "<<std::endl;
    std::cout<<report.model<<std::endl;
}

void Reconstruction::OutputPriorResidualsTxt(std::string save_path){

    const std::vector<image_t>& registered_images = RegisterImageIds();

    std::string rtk_error_path = JoinPaths(save_path, "rtk_residuals.txt");
    std::ofstream file(rtk_error_path, std::ios::trunc);
    CHECK(file.is_open()) << rtk_error_path;

    std::vector<Eigen::Vector3d> residuals_locations;
    std::vector<Eigen::Vector3d> residuals_rotation_axis;
    std::vector<double> residuals_rotation_angle;
    std::vector<double> residuals_tranlation_distance;

    for (size_t i = 0 ; i < registered_images.size(); ++i){
        image_t image_id = registered_images[i];
        class Image&  image = images_.at(image_id);
        if (image.HasQvecPrior() && image.HasTvecPrior()){
            Eigen::Vector4d qvec = image.Qvec();
            Eigen::Vector3d tvec = image.Tvec();
            Eigen::Vector4d qvec_prior = image.QvecPrior();
            Eigen::Vector3d tvec_prior = QuaternionToRotationMatrix(qvec_prior) * -image.TvecPrior();;
            // std::cout << tvec_prior.transpose() << " <=> " << tvec.transpose() << std::endl;

            Eigen::Vector4d qvec_delt;
            Eigen::Vector3d tvec_delt;
            ComputeRelativePose(qvec_prior, tvec_prior, qvec, tvec, &qvec_delt, &tvec_delt);

            Eigen::AngleAxisd rotation_vector;
            rotation_vector.fromRotationMatrix(QuaternionToRotationMatrix(qvec_delt));
            // std::cout << "image (" << image_id << ") difference: " << tvec_delt.transpose() 
            //             << " \t " << rotation_vector.axis().transpose() << "  " << rotation_vector.angle() * (180 / 3.141592653) << std::endl;

            std::ostringstream line;
            line << "image (" << image.Name() << ") difference: " << tvec_delt.transpose() 
                        << " \t " << rotation_vector.axis().transpose() << "  " << rotation_vector.angle() * (180 / 3.141592653) << std::endl;

            std::string line_string = line.str();
            file << line_string << std::endl;

            residuals_locations.push_back(tvec_delt);
            residuals_tranlation_distance.push_back(tvec_delt.norm());

            residuals_rotation_axis.push_back(rotation_vector.axis());
            residuals_rotation_angle.push_back(rotation_vector.angle() * 180 / 3.141592653);
        }
    }

    {
        double sum = std::accumulate(std::begin(residuals_tranlation_distance), std::end(residuals_tranlation_distance), 0.0);  
        double mean =  sum / residuals_tranlation_distance.size();
        double accum  = 0.0;  
        std::for_each (std::begin(residuals_tranlation_distance), std::end(residuals_tranlation_distance), [&](const double d) {  
            accum  += (d-mean)*(d-mean);  
        });  
        double stdev = sqrt(accum/(residuals_tranlation_distance.size()-1)); 
        file << "residuals_tranlation_distance => mean: " << mean << "\t stdev: " << stdev << std::endl;
    }

    {
        double sum = std::accumulate(std::begin(residuals_rotation_angle), std::end(residuals_rotation_angle), 0.0);  
        double mean =  sum / residuals_rotation_angle.size(); 
        double accum  = 0.0;  
        std::for_each (std::begin(residuals_rotation_angle), std::end(residuals_rotation_angle), [&](const double d) {  
            accum  += (d-mean)*(d-mean);  
        });  
        double stdev = sqrt(accum/(residuals_rotation_angle.size()-1)); 
        file << "residuals_rotation_angle => mean: " << mean << "\t stdev: " << stdev << std::endl;
    }

    file.close();
}

Eigen::Matrix3x4d Reconstruction::AlignWithPriorLocations(double max_error, double max_error_horizontal, long max_gps_time_offset){

    if(optimization_use_horizontal_gps_only){
        ComputeImageCenterProjectionHorizontal();
    }

    // GetGPSTimeOffset();

    prior_locations_gps_inlier.clear();
    for(auto prior_location_gps: prior_locations_gps){
        prior_locations_gps_inlier.emplace(prior_location_gps.first,false);
    }

    prior_horizontal_locations_gps_inlier.clear();
    for(auto prior_location_gps: prior_locations_gps){
        prior_horizontal_locations_gps_inlier.emplace(prior_location_gps.first,false);
    }

    const std::vector<image_t>& registered_images = RegisterImageIds();
    std::vector<Eigen::Vector3d> prior_locations;
    std::vector<Eigen::Vector3d> estimated_locations;
    std::vector<image_t> aligned_images;

    for (size_t i = 0 ; i < registered_images.size(); ++i) {
        image_t image_id = registered_images[i];
        class Image& image = images_.at(image_id);

        // reset the status of gps.
        image.SetPriorInlier(false);
        
        if (image.HasTvecPrior()) {
            prior_locations.push_back(image.TvecPrior());
            estimated_locations.push_back(image.ProjectionCenter()); 
            aligned_images.push_back(image_id);
        } 
        else if (prior_locations_gps.find(image_id) != prior_locations_gps.end()) {
            if (prior_locations_gps.at(image_id).second >= 3) {
                prior_locations.push_back(prior_locations_gps.at(image_id).first);

                if (optimization_use_horizontal_gps_only) {
                    CHECK(image_center_projections.find(image_id) != image_center_projections.end());
                    estimated_locations.push_back(image_center_projections.at(image_id));
                } else {
                    estimated_locations.push_back(image.ProjectionCenter());
                }

                aligned_images.push_back(image_id);
            }
        }
    }
    std::cout<<"high precision gps count for transform computation: "<<estimated_locations.size()<<std::endl;
    if(estimated_locations.size()<3){
        b_aligned = false;
        return Eigen::Matrix3x4d::Identity();
    }

    const double max_error2 = max_error * max_error;

    RANSACOptions ransac_options;
    ransac_options.max_error = max_error;
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.25;
    ransac_options.min_num_trials = 200;
    ransac_options.max_num_trials = 10000;
  
    LORANSAC<CameraAlignmentEstimator, CameraAlignmentEstimator> ransac(ransac_options);

    auto report = ransac.Estimate(estimated_locations,prior_locations);
    std::cout << "inlier: " << report.support.num_inliers << std::endl;

    if(report.success && (report.support.num_inliers > 20)||report.support.num_inliers > aligned_images.size()/3){
        
        // Extract inliers of the alignment and transform the reconstruction
        Eigen::Matrix3x4d transform = report.model;

        CameraAlignmentEstimator camera_alignment_estimator;    
        std::vector<Eigen::Vector3d> inlier_camera_locations;
        std::vector<Eigen::Vector3d> inlier_gps_locations;

        for(size_t i = 0; i<report.inlier_mask.size(); ++i){
            if(report.inlier_mask[i]){
                inlier_camera_locations.push_back(estimated_locations[i]);
                inlier_gps_locations.push_back(prior_locations[i]);
            }    
        }
        std::vector<Eigen::Matrix<double, 3, 4>> refined_transforms =
            camera_alignment_estimator.Estimate(inlier_camera_locations, inlier_gps_locations);



        std::vector<double> residuals;
        residuals.resize(estimated_locations.size());
        camera_alignment_estimator.Residuals(estimated_locations,prior_locations,refined_transforms[0],&residuals);

        size_t num_inlier = 0;
        std::vector<bool> inlier_mask(residuals.size(), false);  
        double average_residual = 0.0, average_residual_inlier = 0.0;
        // for (const auto& residual : residuals) {
        for (size_t i = 0; i < residuals.size(); ++i) {
            average_residual += residuals.at(i);
            if (residuals.at(i) < max_error2) {
                inlier_mask[i] = true;
                average_residual_inlier += residuals[i];
                num_inlier++;
            } else {
                inlier_mask[i] = false;
            }
        }
        average_residual /= residuals.size();
        std::cout<<"average residuals: "<<sqrt(average_residual)<<std::endl;
        std::cout<<"average residuals inliers: "<<sqrt(average_residual_inlier / num_inlier)<<std::endl;

        // double average_residual_inlier = 0.0;
        // for (size_t i = 0; i < residuals.size(); ++i) {
        //     if(report.inlier_mask[i]){
        //         average_residual_inlier += residuals[i];
        //     }
        // }
        // average_residual_inlier /= report.support.num_inliers;
        // std::cout<<"average residuals inliers: "<<sqrt(average_residual_inlier)<<std::endl;

        TransformReconstruction(refined_transforms[0],false);
        
        std::cout<<"Transform inlier num: "<<report.support.num_inliers<<" transform the reconstruction"<<std::endl;
        std::cout<<"Transform: "<<std::endl;
        std::cout<<refined_transforms[0]<<std::endl;
        b_aligned = true;

        // std::vector<char> inlier_mask = report.inlier_mask;
        CHECK_EQ(inlier_mask.size(),aligned_images.size());
        for(size_t i = 0; i<aligned_images.size(); ++i){
            if(inlier_mask[i]){
                if (!images_.at(aligned_images[i]).HasTvecPrior()) {
                    CHECK(prior_locations_gps_inlier.find(aligned_images[i]) != prior_locations_gps_inlier.end());
                    prior_locations_gps_inlier.at(aligned_images[i]) = true;
                } else {
                    images_.at(aligned_images[i]).SetPriorInlier(true);
                }
            }
        }

        if (!optimization_use_horizontal_gps_only) {
            // Check whether the horizontal coords of the remaining low precision gps points are consistent with the
            // cameras
            int horizontal_gps_inlier_count = 0;
            double average_horizontal_residual = 0.0;
            for (size_t i = 0; i < registered_images.size(); ++i) {
                image_t image_id = registered_images[i];
                class Image& image = images_.at(image_id);

                if (prior_locations_gps.find(image_id) != prior_locations_gps.end()) {
                    if (!prior_locations_gps_inlier.at(image_id)) {
                        Eigen::Vector3d horizontal_prior_location_gps = prior_locations_gps.at(image_id).first;
                        horizontal_prior_location_gps[2] = 0;

                        Eigen::Vector3d horizontal_estimated_location = image.ProjectionCenter();
                        horizontal_estimated_location[2] = 0;

                        double residual = (horizontal_prior_location_gps - horizontal_estimated_location).norm();
                        if (residual < max_error_horizontal) {
                            prior_horizontal_locations_gps_inlier.at(image_id) = true;
                            horizontal_gps_inlier_count++;
                            average_horizontal_residual += residual;
                        }
                    }
                }
            }
            if (horizontal_gps_inlier_count > 0) {
                average_horizontal_residual /= horizontal_gps_inlier_count;
            }
            std::cout << "horizontal gps inlier count: " << horizontal_gps_inlier_count << std::endl;
            std::cout << "average horizontal residual: " << average_horizontal_residual << std::endl;
            projection_plane << 0, 0, 1, 0;
        }

        return refined_transforms[0];
    }
    else{
        b_aligned = false;
        std::cout<<"Transform inlier num: "<<report.support.num_inliers<<std::endl;
        std::cout<<"Not enough prior inliers"<<std::endl;
        return Eigen::Matrix3x4d::Identity();
    }
}

bool Reconstruction::TryScaleAdjustmentWithDepth(double min_depth_weights) {
    double sum_weights = 0.0;
    std::vector<std::pair<double, double>> scales;
    for (auto & mappoint : mappoints_) {
        Eigen::Vector3d point3D = mappoint.second.XYZ();
        for (auto & element : mappoint.second.Track().Elements()) {
            auto & image = images_[element.image_id];
            auto & point2D = image.Points2D()[element.point2D_idx];
            
            if (point2D.Depth() > std::numeric_limits<float>::epsilon()) {
                Eigen::Vector3d proj_point3D =
                    QuaternionRotatePoint(image.Qvec(), point3D) + image.Tvec();

                if (std::abs(proj_point3D.z()) > std::numeric_limits<float>::epsilon()) {
                    sum_weights += point2D.DepthWeight();
                    scales.emplace_back(point2D.Depth() / proj_point3D.z(), point2D.DepthWeight());
                }
            }
        }
    }
    if (scales.empty()) return false;
    if (sum_weights < min_depth_weights) return false;

    std::sort(scales.begin(), scales.end(), [](std::pair<double, double> a, std::pair<double, double> b) {
        return a.first < b.first;
    });
    auto scales_begin = scales.begin();
    auto scales_end = scales.end();
    if (scales.size() >= 3) {
        scales_begin = scales.begin() + size_t(scales.size() * 0.3 + 0.5);
        scales_end   = scales.begin() + size_t(scales.size() * 0.7 + 0.5);
    }

    double average_scale = 0.0;
    double weights_scale = 0.0;
    for (auto iter = scales_begin; iter != scales_end; iter++) {
        average_scale += iter->first * iter->second;
        weights_scale += iter->second;
    }
    average_scale /= weights_scale;
    std::cout << "Depth Scale Refinement: x" << average_scale 
              << " from " << (scales_end - scales_begin) << " tracks"
              << " with total weights " << weights_scale
              << std::endl;

    Rescale(average_scale);
    for (auto & pair : Cameras()) {
        auto & camera = Camera(pair.first);
        for (auto & tvec : camera.LocalTvecs()) {
            tvec *= average_scale;
        }
    }

    return true;
}

void Reconstruction::ComputeImageCenterProjectionHorizontal(){
    
    // compute the horizontal x axis and y axis
    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d ATA_C = Eigen::Matrix3d::Zero();
    Eigen::Vector3d x_in_camera_coord_frame(1,0,0);

    double x_axis_constraint_count = 0.0;
    Eigen::Vector3d average_center = Eigen::Vector3d::Zero();
    const std::vector<image_t>& registered_images = RegisterImageIds();
    for (size_t i = 0 ; i < registered_images.size(); ++i) {
        image_t image_id = registered_images[i];
        class Image&  image = images_.at(image_id);
        Eigen::Vector4d qvec = image.Qvec();
        Eigen::Matrix3d R = QuaternionToRotationMatrix(qvec);
        Eigen::Vector3d x_in_world_coord_frame = R.transpose() * x_in_camera_coord_frame; 
        ATA += x_in_world_coord_frame * (x_in_world_coord_frame.transpose());
        x_axis_constraint_count += 1.0;
        Eigen::Vector3d center = image.ProjectionCenter();
        average_center = average_center + center; 
    }

    average_center = average_center / x_axis_constraint_count;
    ATA /= x_axis_constraint_count;
    std::cout<<"[ComputeImageCenterProjectionHorizontal] ATA Matrix: "<<std::endl;
    for(unsigned int m=0;m<3;m++){
		for(unsigned int n=0;n<3;n++){
			std::cout<<ATA(m,n)<<" ";
		}
		std::cout<<std::endl;
	}	

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(ATA);
	Eigen::Vector3d eigenValues = eigenSolver.eigenvalues();
	Eigen::Matrix3d eigenVector = eigenSolver.eigenvectors();

	std::cout<<"[ComputeImageCenterProjectionHorizontal] eigenvalues: "<<std::endl;
	double min_eigenvalue = std::numeric_limits<double>::max();
	int min_idx=-1;
	for(unsigned int i=0;i<3;i++){
		if(min_eigenvalue>eigenValues(i)){
			min_eigenvalue=eigenValues(i);
			min_idx=i;
		}
		std::cout<<eigenValues(i)<<" ";
	}
	std::cout<<std::endl;

	std::cout<<"[ComputeImageCenterProjectionHorizontal] eigenVectors"<<std::endl;
	for(unsigned int m=0;m<3;m++){
		for(unsigned int n=0;n<3;n++){
			std::cout<<eigenVector(n,m)<<" ";
		}
		std::cout<<std::endl;
	}

	Eigen::Vector3d up_vector = eigenVector.block<3,1>(0,min_idx);
	if(up_vector(1)>0){
		up_vector = -up_vector;
	}
    	    
    // project to obtain new centers
    image_center_projections.clear();
    projection_plane <<up_vector[0],up_vector[1],up_vector[2], -average_center.dot(up_vector);

    for (size_t i = 0 ; i < registered_images.size(); ++i) {
        
        image_t image_id = registered_images[i];
        class Image&  image = images_.at(image_id);
        Eigen::Vector3d center = image.ProjectionCenter(); 
        double d = center.dot(up_vector) + projection_plane[3];
        Eigen::Vector3d projection_center = center - d * up_vector;    
    
        image_center_projections.emplace(image_id,projection_center);
    }
}

void Reconstruction::GetGPSTimeOffset(long max_time_ofset){

    prior_locations_gps.clear();

    const std::vector<image_t>& registered_images = RegisterImageIds();
    std::unordered_map<label_t,std::vector<image_t> >registered_images_label_map;
        

    for (size_t i = 0 ; i < registered_images.size(); ++i) {
        image_t image_id = registered_images[i];
        class Image&  image = images_.at(image_id);
        int label_id = image.LabelId();
        registered_images_label_map[label_id].push_back(image_id);      
    }

    std::cout<<"get time offset label count "<<registered_images_label_map.size()<<std::endl;
    for(const auto registered_images_per_label:registered_images_label_map){
        
        time_offsets_label_map[registered_images_per_label.first] = 0;
        std::vector<std::string> image_names_per_label;
        std::unordered_map<std::string, Eigen::Vector3d> sfm_locations_per_label;
        std::cout<<"register image count for label "<<registered_images_per_label.first<<" "
                 <<registered_images_per_label.second.size()<<std::endl;

        for(size_t j = 0; j< registered_images_per_label.second.size(); ++j){
            image_t image_id = registered_images_per_label.second[j];
            class Image&  image = images_.at(image_id);   
            image_names_per_label.push_back(image.Name());
           
            if(optimization_use_horizontal_gps_only){
                CHECK(image_center_projections.find(image_id) != image_center_projections.end());
                sfm_locations_per_label.emplace(image.Name(), image_center_projections.at(image_id));
            }
            else{
                sfm_locations_per_label.emplace(image.Name(),image.ProjectionCenter());
            }
        }

        double min_error = std::numeric_limits<double>::max();
        long best_time_offset = 0;
        int point_count_for_offset_estimation = 0;

        for(long time_offset = -max_time_ofset; time_offset <= max_time_ofset; time_offset += 1000){
            std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> gps_locations_per_label;
            GPSLocationsToImages(original_gps_locations,image_names_per_label,gps_locations_per_label,time_offset);    

            std::vector<Eigen::Vector3d> sfm_locations_v;
            std::vector<Eigen::Vector3d> gps_locations_v;
            std::vector<double> residuals;            
            sfm_locations_v.reserve(gps_locations_per_label.size());
            gps_locations_v.reserve(gps_locations_per_label.size());

            for(const auto& gps_location: gps_locations_per_label){
                if(gps_location.second.second >= 3){
                    sfm_locations_v.push_back(sfm_locations_per_label.at(gps_location.first));
                    gps_locations_v.push_back(gps_location.second.first);
                }
            }
            residuals.resize(sfm_locations_v.size());

            if(sfm_locations_v.size()>3){
                CameraAlignmentEstimator camera_alignment_estimator;
                std::vector<CameraAlignmentEstimator::M_t> similarity_trans =
                    camera_alignment_estimator.Estimate(sfm_locations_v, gps_locations_v);
            
                camera_alignment_estimator.Residuals(sfm_locations_v,gps_locations_v,similarity_trans[0],&residuals);

                double average_residual = 0.0;

                for(const auto& residual: residuals){
                    average_residual += residual;
                }

                average_residual /= residuals.size();

                if(average_residual < min_error){
                    min_error = average_residual;
                    best_time_offset = time_offset;
                    point_count_for_offset_estimation = sfm_locations_v.size();
                }
            }
        }

        time_offsets_label_map[registered_images_per_label.first] = best_time_offset;
        
        std::cout<<"best time offset for the "<<registered_images_per_label.first<<": "<<best_time_offset<<std::endl;
        std::cout<<"best residual: "<<min_error<<std::endl;
        std::cout<<"point_count_for_offset_estimation: "<<point_count_for_offset_estimation<<std::endl;

        std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> best_image_locations;
        GPSLocationsToImages(original_gps_locations,image_names_per_label,best_image_locations,best_time_offset); 
        std::cout<<"best image locations size: "<<best_image_locations.size()<<std::endl;

        for(size_t j = 0; j< registered_images_per_label.second.size(); ++j){
            image_t image_id = registered_images_per_label.second[j];
            class Image&  image = images_.at(image_id);   
            if(best_image_locations.find(image.Name())!=best_image_locations.end()){
                prior_locations_gps.emplace(image_id, best_image_locations.at(image.Name()));        
            }
        }
        std::cout<<"prior_locations_gps size: "<<prior_locations_gps.size()<<std::endl;
    }
    
}



void Reconstruction::TransformReconstruction(const Eigen::Matrix3x4d transform, bool inverse) {
    const SimilarityTransform3 tform_inverse(transform);

    // Create a sim3 transform to transform the image and point to another reconstruction
    const SimilarityTransform3 tform = inverse ? tform_inverse.Inverse() : tform_inverse;
    // const SimilarityTransform3 tform = tform_inverse;

    // Transform Images
    std::vector<image_t> all_image_ids;
    all_image_ids.reserve(NumImages());
    const auto & all_images = Images();
    for (auto image : all_images) {
        all_image_ids.push_back(image.first);
    }
    for (const auto image_id : all_image_ids) {
        auto& image = Image(image_id);
        // if(!image.HasPose()){
        //     continue;
        // }
        tform.TransformPose(&image.Qvec(), &image.Tvec());  // Perfrom the calculated transform
    }

    // Transform Map Point
    for (auto& mappoint : mappoints_) {
        Eigen::Vector3d xyz = mappoint.second.XYZ();
        tform.TransformPoint(&xyz);
        mappoint.second.SetXYZ(xyz);
    }
}

void Reconstruction::AddPriorToResult(){
    image_t start_image_id = 1;
    for(auto image: images_){
        if(image.first > start_image_id){
            start_image_id = image.first;
        }
    }
    start_image_id ++;

    image_t new_idx = 1;

    const std::vector<image_t> registered_images = this->RegisterImageIds();
    for (image_t image_id : registered_images) {
        class Image image = images_.at(image_id);
        Eigen::Vector3d location;
        if (image.HasTvecPrior()) {
            location = image.TvecPrior();
        } else {
            if(prior_locations_gps_inlier.count(image_id) == 0 ||
               !prior_locations_gps_inlier.at(image_id)){
                continue;
            }
            location = prior_locations_gps.at(image_id).first;
        }

        class Image image_gps;

        image_gps.SetImageId(start_image_id+new_idx); 
        image_gps.SetName("prior_" + image.Name());
        image_gps.SetCameraId(image.CameraId());

        Eigen::Vector4d Qvec = image.HasQvecPrior() ? image.QvecPrior() : image.Qvec();
        image_gps.SetTvec(QuaternionToRotationMatrix(Qvec) * -location);
        image_gps.SetQvec(Qvec);

        AddImage(image_gps);
        RegisterImage(image_gps.ImageId());

        new_idx++;
    }
}

void Reconstruction::ComputeTriAngles(){

    const std::unordered_set<mappoint_t>& mappoint_ids = MapPointIds();
    
    // Cache for image projection centers.
    EIGEN_STL_UMAP(image_t, Eigen::Vector3d) proj_centers;

    for (const auto mappoint_id : mappoint_ids) {
        if (!ExistsMapPoint(mappoint_id)) {
            continue;
        }

        class MapPoint& mappoint = MapPoint(mappoint_id);

        double max_tri_angle = 0.0;

        // Calculate triangulation angle for all pairwise combinations of image
        // poses in the track. Only delete point if none of the combinations
        // has a sufficient triangulation angle.
        bool keep_point = false;
        for (size_t i1 = 0; i1 < mappoint.Track().Length(); ++i1) {
            const image_t image_id1 = mappoint.Track().Element(i1).image_id;
            const point2D_t point_id1 = mappoint.Track().Element(i1).point2D_idx;
            const class Image & image1 = Image(image_id1);
            const class Camera & camera1 = Camera(image1.CameraId());

            Eigen::Vector3d proj_center1;    

            if(camera1.NumLocalCameras()>1){
                int local_camera_id1 = image1.LocalImageIndices()[point_id1];
                image_t local_image_full_id1 = image_id1*camera1.NumLocalCameras()+local_camera_id1;
                if(proj_centers.count(local_image_full_id1) ==0){

                    Eigen::Vector4d local_qvec;
                    Eigen::Vector3d local_tvec;
                    camera1.GetLocalCameraExtrinsic(local_camera_id1,local_qvec,local_tvec);
                    Eigen::Vector4d qvec = ConcatenateQuaternions(image1.Qvec(),local_qvec);
                    Eigen::Vector3d tvec = 
                        QuaternionToRotationMatrix(local_qvec)*image1.Tvec() + local_tvec;

                    proj_center1 = ProjectionCenterFromPose(qvec,tvec);
                    proj_centers.emplace(local_image_full_id1,proj_center1);
                }
                else{
                    proj_center1 = proj_centers.at(local_image_full_id1);
                }
            }
            else{
                if (proj_centers.count(image_id1) == 0){
                    proj_center1 = image1.ProjectionCenter();
                    proj_centers.emplace(image_id1, proj_center1);
                }
                else{
                    proj_center1 = proj_centers.at(image_id1);
                }    
            }

            for (size_t i2 = 0; i2 < i1; ++i2){
                const image_t image_id2 = mappoint.Track().Element(i2).image_id;
                const point2D_t point_id2 = mappoint.Track().Element(i2).point2D_idx;
                const class Image & image2 = Image(image_id2);
                
                Eigen::Vector3d proj_center2;
                
                if(camera1.NumLocalCameras() == 1){
                    CHECK(proj_centers.find(image_id2)!=proj_centers.end());
                    proj_center2 = proj_centers.at(image_id2);
                }
                else{
                    int local_camera_id2 = image2.LocalImageIndices()[point_id2];
                    image_t local_image_full_id2 = image_id2*camera1.NumLocalCameras()+local_camera_id2;
                    CHECK(proj_centers.find(local_image_full_id2)!=proj_centers.end());

                    proj_center2 = proj_centers.at(local_image_full_id2);
                }
                
                const double tri_angle =
                    CalculateTriangulationAngle(proj_center1,
                                                proj_center2,
                                                mappoint.XYZ());
                

                if(tri_angle > max_tri_angle){
                    max_tri_angle = tri_angle;
                }
            }           
        }
        mappoint.SetTriAngle(max_tri_angle);

    }
}

std::unordered_set<image_t> Reconstruction::FindImageForMapPoints(const std::unordered_set<mappoint_t> mappoints){
    
    std::unordered_set<image_t> visible_images;
    for(auto mappoint_id: mappoints){
        if(!ExistsMapPoint(mappoint_id)){
            continue;
        }
    
        class MapPoint& mappoint = mappoints_[mappoint_id];
        const Track& track = mappoint.Track();
    
        for (const auto& track_el : track.Elements()) {
            if(visible_images.count(track_el.image_id) == 0){
                visible_images.insert(track_el.image_id);   
            } 
        }
    }
    return visible_images;
}


std::unordered_map<std::string, image_t> Reconstruction::GetImageNames() const {
    std::unordered_map<std::string, image_t> image_names;
    for (const auto& image_data : images_) {
        if (!IsImageRegistered(image_data.first)) {
            continue;
        }
        image_names[image_data.second.Name()] = image_data.first;
    }
    return image_names;
}

const std::vector<image_t> Reconstruction::GetNewImageIds() const {
    std::vector<image_t> new_image_ids;
    for (auto & image : Images()) {
        // updated images.
        if (image.second.IsRegistered() && image.second.LabelId() != 0) {
            new_image_ids.push_back(image.first);
        }
    }
    std::sort(new_image_ids.begin(), new_image_ids.end());
    return new_image_ids;
}

const std::vector<image_t> Reconstruction::RegisterImageSortIds() const {
    std::vector<image_t> new_image_ids;
    new_image_ids = register_image_ids_;
    std::sort(new_image_ids.begin(), new_image_ids.end());
    return new_image_ids;
}

const std::vector<sweep_t> Reconstruction::RegisterSweepSortIds() const {
    std::vector<sweep_t> new_sweep_ids;
    if (!register_sweep_ids_.empty()){
        new_sweep_ids = register_sweep_ids_;
        std::sort(new_sweep_ids.begin(), new_sweep_ids.end());
    }
    return new_sweep_ids;
}

void Reconstruction::TranscribeImageIdsToDatabase(const std::shared_ptr<FeatureDataContainer> feature_data_container) {
    std::unordered_map<image_t, image_t> old_to_new_image_ids;
    old_to_new_image_ids.reserve(NumImages());

    EIGEN_STL_UMAP(image_t, class Image) new_images;
    new_images.reserve(NumImages());

    for (auto& image : images_) {
        if (!feature_data_container->ExistImage(image.second.Name())) {
            LOG(FATAL) << "Image with name " << image.second.Name() << " does not exist in database";
            continue;
        }

        const auto database_image = feature_data_container->GetImage(image.second.Name());
        old_to_new_image_ids.emplace(image.second.ImageId(), database_image.ImageId());
        image.second.SetImageId(database_image.ImageId());
        new_images.emplace(database_image.ImageId(), image.second);
    }

    images_ = std::move(new_images);

    for (auto& image_id : register_image_ids_) {
        image_id = old_to_new_image_ids.at(image_id);
    }

    for (auto& point3D : mappoints_) {
        for (auto& track_el : point3D.second.Track().Elements()) {
            track_el.image_id = old_to_new_image_ids.at(track_el.image_id);
        }
    }
}

// Remove all the point2d which do not has mappoint
void Reconstruction::FilterUselessPoint2D(std::shared_ptr<Reconstruction> reconstruction_out,
    std::unordered_set<image_t> const_id_set) {
    std::unordered_map<image_t, std::unordered_map<point2D_t, point2D_t>> update_point2d_map;
    
    reconstruction_out->yrb_factors.resize(yrb_factors.size());

    // std::cout << "1. Set camera bin ... " << std::endl;
    for (const auto& cur_camera : Cameras()) {
        reconstruction_out->AddCamera(cur_camera.second);
    }

    // std::cout << "2. Update Point2d ... " << std::endl;
    const auto &image_ids = RegisterImageIds();
    for (const auto &image_id : image_ids) {
        const auto cur_image = Image(image_id);
        // Get all the old point2ds
        const auto old_point2ds = cur_image.Points2D();
        std::vector<class Point2D> new_point2Ds;  // Store new point2d

        point2D_t new_point_id = 0;
        for (point2D_t point_id = 0; point_id < old_point2ds.size(); point_id++) {
            if (!old_point2ds[point_id].HasMapPoint() && 
                (const_id_set.count(image_id) == 0)) {
                continue;
            }

            class Point2D new_point2D;
            new_point2D.SetXY(old_point2ds[point_id].XY());
            new_point2Ds.emplace_back(std::move(new_point2D));

            update_point2d_map[image_id][point_id] = new_point_id;
            new_point_id++;
        }
    
        

        // if (new_point2Ds.empty()) {
        //     continue;
        // }

        class Image new_image;
        // Update image id
        new_image.SetImageId(cur_image.ImageId());
        new_image.SetCameraId(cur_image.CameraId());

        // Update the camera rotation
        new_image.SetQvec(cur_image.Qvec());
        new_image.SetTvec(cur_image.Tvec());
        if (cur_image.HasQvecPrior()) {
            new_image.SetQvecPrior(cur_image.QvecPrior());
        }
        if (cur_image.HasTvecPrior()) {
            new_image.SetTvecPrior(cur_image.TvecPrior());
        }

        new_image.SetName(cur_image.Name());
        new_image.SetPoints2D(new_point2Ds);

        new_image.SetLabelId(cur_image.LabelId());

        // Update reconstruction
        reconstruction_out->AddImage(new_image);
        reconstruction_out->RegisterImage(new_image.ImageId());
        if (!yrb_factors.empty()) {
            reconstruction_out->yrb_factors[new_image.ImageId()] = yrb_factors.at(cur_image.ImageId());
        }
    }

    const auto sort_sweep_ids = RegisterSweepSortIds();
    for (int idx = 0; idx < sort_sweep_ids.size(); idx++){
        const auto cur_sweep = LidarSweep(sort_sweep_ids.at(idx));

        class LidarSweep new_sweep(cur_sweep.SweepID(), cur_sweep.Name());
        new_sweep.SetQvec(cur_sweep.Qvec());
        new_sweep.SetTvec(cur_sweep.Tvec());
        new_sweep.SetRegistered(cur_sweep.IsRegistered());
        reconstruction_out->AddLidarSweep(new_sweep);
        reconstruction_out->RegisterLidarSweep(cur_sweep.SweepID());
    }
    reconstruction_out->lidar_to_cam_matrix = lidar_to_cam_matrix;

    // std::cout << "3. Update 3d point track id ... " << std::endl;
    const auto &mappoint_ids = MapPointIds();
    for (const auto &mappoint_id : mappoint_ids) {
        class MapPoint new_mappoint;
        // Get old mappoint
        const auto old_mappoint = MapPoint(mappoint_id);

        // Use the old mappoint position
        new_mappoint.SetXYZ(old_mappoint.XYZ());
        new_mappoint.SetColor(old_mappoint.Color());
        new_mappoint.SetError(old_mappoint.Error());

        // Update the old mappoint track with new image id and point2d id
        class Track new_track;
        for (const auto &track_el : old_mappoint.Track().Elements()) {
            if (!update_point2d_map.count(track_el.image_id)) {
                continue;
            }

            if (!update_point2d_map[track_el.image_id].count(track_el.point2D_idx)) {
                continue;
            }
            auto new_point2d_id = update_point2d_map[track_el.image_id][track_el.point2D_idx];

            new_track.AddElement(track_el.image_id, new_point2d_id);
        }
        new_mappoint.SetTrack(new_track);

        reconstruction_out->AddMapPointWithError(new_mappoint.XYZ(), std::move(new_mappoint.Track()), new_mappoint.Color(),
                                                 new_mappoint.Error());
    }
    std::cout << "Filter Useless Point2D" << std::endl;
}

void Reconstruction::ResetPointStatus() {
    const auto& mappoint_ids = MapPointIds();
    for (const auto mappoint_id : mappoint_ids) {
        if (!ExistsMapPoint(mappoint_id)) {
            continue;
        }

        const class MapPoint& mappoint = MapPoint(mappoint_id);
        for (size_t i = 0; i < mappoint.Track().Length(); ++i) {
            TrackElement track_elem = mappoint.Track().Element(i);
            class Image & image = Image(track_elem.image_id);
            class Point2D & point2D = image.Point2D(track_elem.point2D_idx);
            if (point2D.InOverlap()) {
                point2D.SetOverlap(false);
            }
        }
    }
}

void Reconstruction::Rescale(const double scale) {
    const auto& register_image_ids = RegisterImageIds();
    for (const auto& image_id : register_image_ids) {
        class Image& image = Image(image_id);
        image.Tvec() *= scale;
    }
    const auto& mappoint_ids = MapPointIds();
    for (const auto& mappoint_id : mappoint_ids) {
        class MapPoint& mappoint = MapPoint(mappoint_id);
        mappoint.SetXYZ(mappoint.XYZ() * scale);
    }
}

void Reconstruction::RescaleAll(const double scale) {
    Rescale(scale);
    for (auto & pair : cameras_) {
        auto & camera = cameras_.at(pair.first);
        for (auto & tvec : camera.LocalTvecs()) {
            tvec *= scale;
        }
    }
}

// Lidar
void Reconstruction::AddLidarSweep(const class LidarSweep& sweep){
    const auto sweep_id = sweep.SweepID();
    CHECK(!ExistsLidarSweep(sweep_id));
    lidarsweeps_[sweep_id] = sweep;
    if (sweep.IsRegistered()){
        register_sweep_ids_.push_back(sweep_id);
    }
    return;
}

sweep_t Reconstruction::AddLidarData(const std::string& lidar_path, const std::string& lidar_name_t) {
    std::string lidar_name = lidar_name_t;
    // if (ExistsPath(JoinPaths(lidar_path, lidar_name))){
    // class LidarSweep lidarsweep;
    sweep_t sweep_id = added_sweep_ids_.size();
    auto pc = ReadPCD(JoinPaths(lidar_path, lidar_name));
    if (pc.info.height == 1 ){
        RebuildLivoxMid360(pc, 4);
    }

    class LidarSweep lidarsweep(sweep_id, lidar_name, pc);
    lidarsweeps_[sweep_id] = lidarsweep;
    added_sweep_ids_.push_back(sweep_id);

    std::cout << "add sweep size: " << added_sweep_ids_.size() << std::endl;

    return sweep_id;
    
}

void Reconstruction::LoadLidar(const std::vector<RigNames>& rig_list, const std::string& lidar_path) {
    const auto& image_names = GetImageNames();
    size_t num_add_sweep = 0;
    for (size_t i = 0; i < rig_list.size(); i++){
        std::string image_name = rig_list[i].img;
        if (image_names.count(image_name) < 1){
            std::cout << "Warning: image_names.count(" << image_name << ") == 0" << std::endl;
            continue;
        }
        if (ExistsPath(JoinPaths(lidar_path, rig_list[i].pcd))){
            AddLidarData(lidar_path, rig_list[i].pcd);

            if (!IsNaN(rig_list[i].q.sum()) && !IsNaN(rig_list[i].t.sum())) {
                lidarsweeps_[num_add_sweep].SetQvecPrior(InvertQuaternion(rig_list[i].q));
                lidarsweeps_[num_add_sweep].SetTvecPrior(rig_list[i].t);
                std::cout << "lidar#" << num_add_sweep << " " << rig_list[i].q.transpose() << " " << rig_list[i].t.transpose() << std::endl;
            }

            num_add_sweep++;
            // if (num_add_sweep >= 50){
            //     break;
            // }
        }else {
            std::cout << JoinPaths(lidar_path, rig_list[i].pcd) << "input file empty" << std::endl;
        }
    }
    std::cout << "Loaded " << num_add_sweep << " lidar sweeps" << std::endl;
    return;
}

void Reconstruction::LoadLidar(const std::string& lidar_path) {
    std::vector<std::string> sweep_list = GetRecursiveFileList(lidar_path);
    sweep_t num_add_sweep = 0;
    added_sweep_ids_.clear();
    sweep_name_to_id.clear();

    for (size_t i = 0; i < sweep_list.size(); i++){
        auto lidar_name = sweep_list[i];
        lidar_name = lidar_name.substr(lidar_path.size(), lidar_name.size() - lidar_path.size());
        if (ExistsPath(JoinPaths(lidar_path, lidar_name))){
            class LidarSweep lidarsweep(num_add_sweep, lidar_name);
            
            auto lidar_name = GetPathBaseName(lidarsweep.Name());
            lidar_name = lidar_name.substr(0, lidar_name.rfind("."));
            long long lidar_timestamp = std::atof(lidar_name.c_str()) * 1e9;
            // long long lidar_timestamp = std::atof(lidar_name.c_str());
            lidarsweep.timestamp_ = lidar_timestamp;
            
            lidarsweeps_[num_add_sweep] = lidarsweep;
            added_sweep_ids_.push_back(num_add_sweep);

            // std::cout << "lidar_name: " << lidar_name << " " << num_add_sweep << "," << lidar_timestamp << std::endl;
            sweep_name_to_id[lidar_name] = num_add_sweep;
            num_add_sweep++;
        }
    }
    std::cout << "Loaded " << num_add_sweep << " lidar sweeps" << std::endl;
}

void Reconstruction::UpdateNeighborsRelatPose(
    std::unordered_set<image_t> & image_ids){
    for (auto image_id1 : image_ids){
        if (!images_[image_id1].IsKeyFrame()){
            continue;
        }
        Eigen::Matrix4d image_pose1 = Eigen::Matrix4d::Identity();
        image_pose1.block<3,3>(0,0) = images_[image_id1].RotationMatrix();
        image_pose1.block<3,1>(0,3) = images_[image_id1].Tvec();

        // for(auto image_id2 : map_sweep_neighbors_[image_id1]){
        //     if (image_ids.find(image_id2) == image_ids.end() 
        //         || image_id1 == image_id2){
        //         continue;
        //     }
        for(auto image_id2 : image_ids){
            if (image_id1 == image_id2 || !images_[image_id2].IsKeyFrame()){
                continue;
            }
            Eigen::Matrix4d image_pose2 = Eigen::Matrix4d::Identity();
            image_pose2.block<3,3>(0,0) = images_[image_id2].RotationMatrix();
            image_pose2.block<3,1>(0,3) = images_[image_id2].Tvec();

            Eigen::Matrix4d relative_pose = image_pose2 * image_pose1.inverse();
            // if (!lidarsweeps_[image_id1].ExistsNeighbor(image_id2)){
            if (1){
                lidarsweeps_[image_id1].UpdateNeighbor(image_id2, relative_pose);
            }
        }
    }
    return;
}

bool Reconstruction::OutputLidarPointCloud2World(
    const std::string& lidar_path, 
    bool output_frame){
    LidarPointCloud lidar_sweeps_cloud, laser_cloud_temp;

    for (auto sweep_id: register_sweep_ids_){

        class LidarSweep sweep_ref = lidarsweeps_.at(sweep_id);

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topRows(3) = sweep_ref.InverseProjectionMatrix();

        LidarPointCloud ref_less_features, ref_less_features_t; 
        LidarPointCloud ref_less_surfs = sweep_ref.GetSurfPointsLessFlat();
        LidarPointCloud ref_less_corners = sweep_ref.GetCornerPointsLessSharp();
        ref_less_features = ref_less_surfs;
        ref_less_features += ref_less_corners;
        LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T);

        lidar_sweeps_cloud += ref_less_features_t;

        if (output_frame){
            std::string ref_les_name = lidar_path + "/" + sweep_ref.Name() + ".ply";
            std::string parent_path = GetParentDir(ref_les_name);
            if (!ExistsPath(parent_path)) {
                boost::filesystem::create_directories(parent_path);
            }
            WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t.Convert2Ply(),
                                false, true);
        }
    }

    // // TODO downsample
    // PlyPointCloud laserCloudStack = lidar_sweeps_cloud;
    // std::string cloud_path = lidar_path + "/lidar_cloud.ply";
    std::string cloud_path = lidar_path + "/lidar.ply";
    if (!ExistsPath(GetParentDir(cloud_path))) {
        boost::filesystem::create_directories(GetParentDir(cloud_path));
    }
    WriteBinaryPlyPoints(cloud_path.c_str(), lidar_sweeps_cloud.Convert2Ply(), false, true);

    std::vector<LidarPoint>().swap(lidar_sweeps_cloud.points);
    // std::cout << "===============================================================" << std::endl
    //           << "                    " << "save " << i << " iterBA" << std::endl 
    //           << "===============================================================" << std::endl;
    return true;
};

// bool Reconstruction::OutputLocalLidarPointCloud2World(
//     std::vector<image_t> image_ids, 
//     int i, const std::string& lidar_path,
//     const Eigen::Matrix3x4d Tr,
//     bool downsample_flag){
//     LidarPointCloud lidar_sweeps_cloud, laser_cloud_temp;
//     std::string path = lidar_path + images_.at(image_ids.at(0)).Name() + "-" 
//                        + images_.at(image_ids.at(1)).Name()+ "/";
//     if (!ExistsPath(path)) {
//         boost::filesystem::create_directories(path);
//     }
//     std::string frame_path = lidar_path + "localBA/" + std::to_string(i) + "/";
//     if (!ExistsPath(frame_path)) {
//         boost::filesystem::create_directories(frame_path);
//     }
//     for (auto image_id: image_ids){
//         class Image image = images_.at(image_id);
//         Eigen::Matrix3x4d Tr_ref_inv = image.InverseProjectionMatrix();
//         Eigen::Matrix4d Tr_ref_inv_4d = Eigen::Matrix4d::Identity();
//         Tr_ref_inv_4d.topRows(3) = Tr_ref_inv;

//         class LidarSweep sweep_ref = lidarsweeps_.at(sweep_t(image_id));

//         auto& camera_ref = Camera(sweep_ref.CameraId());
//         Eigen::Matrix3x4d Tr_lidar2camera = camera_ref.GetLidar2Camera();
//         Eigen::Matrix4d Tr_lidar2camera_4d = Eigen::Matrix4d::Identity();
//         Tr_lidar2camera_4d.topRows(3) = Tr_lidar2camera;
//         Eigen::Matrix4d T = Tr_ref_inv_4d * Tr_lidar2camera_4d;

//         LidarPointCloud ref_less_features, ref_less_features_t; 
//         LidarPointCloud ref_less_surfs = sweep_ref.GetSurfPointsLessFlat();
//         LidarPointCloud ref_less_corners = sweep_ref.GetCornerPointsLessSharp();
//         ref_less_features = ref_less_surfs;
//         ref_less_features += ref_less_corners;
//         LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T);

//         std::string ref_les_name = frame_path + image.Name() + "_ref.ply";
//         WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t.Convert2Ply(),
//                              false, false);
//         lidar_sweeps_cloud += ref_less_features_t;
//     }

//     // TODO downsample
//     LidarPointCloud laserCloudStack = lidar_sweeps_cloud;
    
//     std::string cloud_path = path + std::to_string(i) +"_lidar_cloud.ply";
//     WriteBinaryPlyPoints(cloud_path.c_str(), laserCloudStack.Convert2Ply(), false, false);

//     std::cout << "::: " << "save " << i << " local iterBA" << std::endl;
//     return true;
// };

// bool Reconstruction::GlobalPointCloud(){
//     LidarPointCloud global_corner_points, global_surf_points;
   
//     for (auto image_id: added_sweep_ids_){
//         class Image image = images_.at(image_id);
//         Eigen::Matrix3x4d Tr_ref_inv = image.InverseProjectionMatrix();
//         Eigen::Matrix4d Tr_ref_inv_4d = Eigen::Matrix4d::Identity();
//         Tr_ref_inv_4d.topRows(3) = Tr_ref_inv;

//         class LidarSweep sweep_ref = lidarsweeps_.at(sweep_t(image_id));

//         auto& camera_ref = Camera(sweep_ref.CameraId());
//         Eigen::Matrix3x4d Tr_lidar2camera = camera_ref.GetLidar2Camera();
//         Eigen::Matrix4d Tr_lidar2camera_4d = Eigen::Matrix4d::Identity();
//         Tr_lidar2camera_4d.topRows(3) = Tr_lidar2camera;
//         Eigen::Matrix4d T = Tr_ref_inv_4d * Tr_lidar2camera_4d;

//         LidarPointCloud ref_less_cor_t, ref_less_surf_t; 
//         LidarPointCloud ref_less_surfs = sweep_ref.GetSurfPointsLessFlat();
//         LidarPointCloud ref_less_cors = sweep_ref.GetCornerPointsLessSharp();
//         LidarPointCloud::TransfromPlyPointCloud (ref_less_surfs, ref_less_surf_t, T);
//         LidarPointCloud::TransfromPlyPointCloud (ref_less_cors, ref_less_cor_t, T);

//         global_corner_points += ref_less_cor_t;
//         global_surf_points += ref_less_surf_t;
//     }

//     // TODO downsample
//     global_corner_points_.points.clear();
//     global_surf_points_.points.clear();
//     // WlopSimplifyPlyPointCloud(global_corner_points, global_corner_points_, 5, 0.4);
//     // WlopSimplifyPlyPointCloud(global_surf_points, global_surf_points_, 1, 1.0);
//     LidarPointCloud::GridSimplifyPointCloud(global_corner_points, global_corner_points_, 0.4);
//     LidarPointCloud::GridSimplifyPointCloud(global_surf_points, global_surf_points_, 0.8);

//     return true;
// };

// bool Reconstruction::LocalPointCloud(const std::unordered_set<image_t>& local_images){
//     LidarPointCloud global_corner_points, global_surf_points;
//     for (auto image_id: local_images){
//         class Image image = images_.at(image_id);
//         Eigen::Matrix3x4d Tr_ref_inv = image.InverseProjectionMatrix();
//         Eigen::Matrix4d Tr_ref_inv_4d = Eigen::Matrix4d::Identity();
//         Tr_ref_inv_4d.topRows(3) = Tr_ref_inv;

//         class LidarSweep sweep_ref = lidarsweeps_.at(sweep_t(image_id));

//         auto& camera_ref = Camera(sweep_ref.CameraId());
//         Eigen::Matrix3x4d Tr_lidar2camera = camera_ref.GetLidar2Camera();
//         Eigen::Matrix4d Tr_lidar2camera_4d = Eigen::Matrix4d::Identity();
//         Tr_lidar2camera_4d.topRows(3) = Tr_lidar2camera;
//         Eigen::Matrix4d T = Tr_ref_inv_4d * Tr_lidar2camera_4d;

//         LidarPointCloud ref_less_cor_t, ref_less_surf_t; 
//         LidarPointCloud ref_less_surfs = sweep_ref.GetSurfPointsLessFlat();
//         LidarPointCloud ref_less_cors = sweep_ref.GetCornerPointsLessSharp();
//         LidarPointCloud::TransfromPlyPointCloud (ref_less_surfs, ref_less_surf_t, T);
//         LidarPointCloud::TransfromPlyPointCloud (ref_less_cors, ref_less_cor_t, T);

//         global_corner_points += ref_less_cor_t;
//         global_surf_points += ref_less_surf_t;
//     }

//     // TODO downsample
//     global_corner_points_.points.clear();
//     global_surf_points_.points.clear();
//     // WlopSimplifyPlyPointCloud(global_corner_points, global_corner_points_, 5, 0.4);
//     // WlopSimplifyPlyPointCloud(global_surf_points, global_surf_points_, 1, 1.0);
//     LidarPointCloud::GridSimplifyPointCloud(global_corner_points, global_corner_points_, 0.4);
//     LidarPointCloud::GridSimplifyPointCloud(global_surf_points, global_surf_points_, 0.8);

//     return true;
// };

bool Reconstruction::Copy(
    const std::unordered_set<image_t> &target_images, 
    const std::unordered_set<mappoint_t> &target_mappoints,
    std::shared_ptr<Reconstruction> res)
{
    res->baseline_distance = baseline_distance;
    res->primary_plane = primary_plane;
    res->planes_for_images = planes_for_images;
    res->images_for_planes = images_for_planes;
    res->have_prior_pose  = have_prior_pose;
    res->prior_rotations = prior_rotations;
    res->prior_translations = prior_translations;
    res->has_gps_prior = has_gps_prior;
    res->prior_locations_gps = prior_locations_gps;
    res->prior_locations_gps_inlier = prior_locations_gps_inlier;
    res->prior_horizontal_locations_gps_inlier = prior_horizontal_locations_gps_inlier;
    res->b_aligned = b_aligned;
    
    res->correspondence_graph_ = correspondence_graph_;
    res->cameras_ = cameras_;
    res->register_image_ids_.clear();
    for (const auto &img : target_images) {
        res->images_[img] = images_[img];
        res->register_image_ids_.emplace_back(img);
    }
    for (const auto &mpt : target_mappoints) {
        res->mappoints_[mpt] = mappoints_[mpt];
    }

    res->image_pairs_ = image_pairs_;

    // update all track and mappoint
    {
        for (auto &mappt : res->mappoints_) {
            auto &track = mappt.second.Track();
            auto &track_eles = track.Elements();

            std::vector<size_t> delete_tracks;
            for (int i = 0; i < track_eles.size(); ++ i) {
                image_t img = track_eles[i].image_id;

                if (target_images.count(img) == 0) {
                    delete_tracks.emplace_back(i);
                }
            }

            for (int i = 0; i < delete_tracks.size(); ++ i) {
                track.DeleteElement(delete_tracks[i] - i);
            }
        }

        // remove mappoints
        std::vector<mappoint_t> deleted_mappts;
        for (auto &mappt : res->mappoints_) {
            if (mappt.second.Track().Length() < 2) {
                deleted_mappts.emplace_back(mappt.first);
                continue;
            }

            const auto &track_eles = mappt.second.Track().Elements();
            std::unordered_set<image_t> visible_images;
            for (int i = 0; i < track_eles.size(); ++ i) {
                visible_images.insert(track_eles[i].image_id);

                if (visible_images.size() > 1) break;
            }

            if (visible_images.size() < 2) 
            {
                deleted_mappts.emplace_back(mappt.first);
            }
        }
        
        for (auto pt : deleted_mappts) {
            res->DeleteMapPoint(pt);
        }
    }

    res->num_added_mappoints_ = res->mappoints_.size();

    return true;
}
}  // namespace sensemap
