//Copyright (c) 2021, SenseTime Group.
//All rights reserved.
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "../system_io.h"

// CGAL: depth-map initialization
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Projection_traits_xy_3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <CGAL/tags.h>

#include <CGAL/Epick_d.h>
#include <CGAL/point_generators_d.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Fuzzy_iso_box.h>
#include <CGAL/Search_traits_d.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

#include "base/common.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/point2d.h"
#include "base/pose.h"
#include "base/camera_models.h"
#include "base/undistortion.h"
#include "base/reconstruction_manager.h"
#include "base/projection.h"

#include "graph/correspondence_graph.h"
#include "controllers/incremental_mapper_controller.h"

#include "lidar/pcd.h"
#include "lidar/lidar_sweep.h"
#include "mvs/workspace.h"
#include "mvs/delaunay/delaunay_triangulation.h"

#include "util/ply.h"
#include "util/proc.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/exception_handler.h"

#include "base/version.h"

std::string configuration_file_path;
const float max_value = std::log1p(1000);
float lidar_factor = 1.0;

using namespace sensemap;
using namespace mvs;

void ReadImageMapText(const std::string& path,
    std::unordered_map<image_t, image_t>& id2rig,
    std::unordered_map<image_t, std::vector<image_t>>& rig2ids){
    id2rig.clear();
    rig2ids.clear();

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

        std::getline(line_stream, item, ' ');
        image_t image_id = std::stoul(item);

        std::getline(line_stream, item, ' ');
        image_t rig_id = std::stoul(item);

        id2rig[image_id] = rig_id;
        rig2ids[rig_id].push_back(image_id);
    }
    std::cout << "ReadImageMapText Done, " 
        << id2rig.size() << "-" << rig2ids.size() << std::endl;

    return;
}

void SaveLidarImagePair(const std::string& path, 
                   const std::unordered_map<image_t, std::vector<image_t>>& rig2ids,
                   const std::unordered_map<sweep_t, image_t>& lidar2rig_map,
                   std::unordered_map<image_t, sweep_t>& image2lidar_map){
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# ID list with one line of data per Image:" << std::endl;
    file << "#   Sparse_Image_ID, Sparse_LIDAR_ID" << std::endl;
    file << "# Number of Images: " << lidar2rig_map.size() * rig2ids.begin()->second.size() << std::endl;

    for(const auto& l2r : lidar2rig_map){
        for (const auto id : rig2ids.at(l2r.second)){
            std::ostringstream line;
            line << id << " " << l2r.first << " ";

            std::string line_string = line.str();
            line_string = line_string.substr(0, line_string.size() - 1);

            file << line_string << std::endl;
            image2lidar_map[id] = l2r.first;
        }
    }

    file.close();
    std::cout << "has " << image2lidar_map.size() << " correspondence, save in " << path<< std::endl;
}

void FindLidarImagePair(const std::shared_ptr<Reconstruction> reconstruction_,
                        const std::unordered_map<image_t, image_t>& id2rig,
                        const std::unordered_map<image_t, std::vector<image_t>>& rig2ids,
                        std::unordered_map<sweep_t, image_t>& lidar2rig_map){
    lidar2rig_map.clear();

    std::vector<std::pair<image_t, long long> > rig_timestamps;
    for (const auto rig : rig2ids) {
        auto & rec_image = reconstruction_->Image(rig.second.at(0));
        auto image_name = GetPathBaseName(rec_image.Name());
        image_name = image_name.substr(0, image_name.rfind("."));
        long long image_timestamp = std::stoll(image_name);
        rig_timestamps.emplace_back(rig.first, image_timestamp);

        for (int i = 0; i < rig.second.size(); i++){
            auto & rec_image = reconstruction_->Image(rig.second.at(i));
            rec_image.timestamp_ = image_timestamp;
        }
    }
    std::sort(rig_timestamps.begin(), rig_timestamps.end(), 
        [&](const std::pair<image_t, long long> & a, const std::pair<image_t, long long> & b) {
            return a.second < b.second;
        });

    std::vector<std::pair<sweep_t, long long> > sweep_timestamps;
    for (auto sweep : reconstruction_->LidarSweeps()){
        auto lidar_name = GetPathBaseName(sweep.second.Name());
        lidar_name = lidar_name.substr(0, lidar_name.rfind("."));
        long long lidar_timestamp = std::atof(lidar_name.c_str()) * 1e9;
        sweep.second.timestamp_ = lidar_timestamp;

        sweep_timestamps.emplace_back(sweep.first, lidar_timestamp);
    }
    std::sort(sweep_timestamps.begin(), sweep_timestamps.end(), 
        [&](const std::pair<sweep_t, long long> & a, const std::pair<sweep_t, long long> & b) {
            return a.second < b.second;
        });
    
    for (long long rig_idx = 0, sweep_idx = 0; sweep_idx < sweep_timestamps.size(); sweep_idx++){
        // std::cout << "rig_idx: " << rig_idx << std::endl;
        for (; rig_idx < rig_timestamps.size() - 1; rig_idx++){
            long long prev_dtime = rig_timestamps.at(rig_idx).second - sweep_timestamps.at(sweep_idx).second;
            long long next_dtime = rig_timestamps.at(rig_idx + 1).second - sweep_timestamps.at(sweep_idx).second;

            if (prev_dtime > 0 && next_dtime > 0){
                lidar2rig_map[sweep_timestamps.at(sweep_idx).first] = rig_timestamps.at(rig_idx).first;
                // std::cout << " =>1 : " 
                //     << float(sweep_timestamps.at(sweep_idx).second - rig_timestamps[lidar2rig_map[sweep_timestamps.at(sweep_idx).first]].second) / 1e9
                //     << " / "  << sweep_timestamps.at(sweep_idx).first <<"(" << sweep_timestamps.at(sweep_idx).second << ") - " 
                //     << lidar2rig_map[sweep_timestamps.at(sweep_idx).first] << "-" << rig_timestamps.at(rig_idx).first 
                //     << "(" << rig_timestamps.at(rig_idx).second << ")" << std::endl;
                break;
            }


            if (prev_dtime < 0 && next_dtime > 0){
                if (-prev_dtime < next_dtime){
                    lidar2rig_map[sweep_timestamps.at(sweep_idx).first] = rig_timestamps.at(rig_idx).first;
                } else {
                    lidar2rig_map[sweep_timestamps.at(sweep_idx).first] = rig_timestamps.at(rig_idx + 1).first;
                }
                // std::cout << " =>2 : " << sweep_timestamps.at(sweep_idx).second - rig_timestamps[lidar2rig_map[sweep_timestamps.at(sweep_idx).first]].second
                //     << " / " << sweep_timestamps.at(sweep_idx).first <<"(" << sweep_timestamps.at(sweep_idx).second << ") - " 
                //     << lidar2rig_map[sweep_timestamps.at(sweep_idx).first] << "(" 
                //     << rig_timestamps[lidar2rig_map[sweep_timestamps.at(sweep_idx).first]].second << ")" << std::endl;
                break;
            }
        }

        if (rig_idx == rig_timestamps.size() - 1){
            // std::cout << "rig_idx: " << rig_idx << std::endl;
            long long prev_dtime = rig_timestamps.at(rig_idx).second - sweep_timestamps.at(sweep_idx).second;
            if (prev_dtime < 0){
                lidar2rig_map[sweep_timestamps.at(sweep_idx).first] = rig_timestamps.at(rig_idx).first;
                // std::cout << " =>3 : " << (sweep_timestamps.at(sweep_idx).second - rig_timestamps[rig_idx].second) / 1e9
                //     << " / " << sweep_timestamps.at(sweep_idx).first <<"(" << sweep_timestamps.at(sweep_idx).second << ") - " 
                //     << lidar2rig_map[sweep_timestamps.at(sweep_idx).first] << "-" << rig_timestamps.at(rig_idx).first << "(" 
                //     << rig_timestamps[rig_idx].second << ")" << std::endl;
            }
        }
    }
    std::cout << "lidar2rig_map size: " << lidar2rig_map.size() << std::endl;
    
    return;
};

void Translation(const std::shared_ptr<Reconstruction> reconstruction_,
              const sweep_t sweep_id,
              std::vector<sensemap::PlyPoint>& pc){
    
    const auto& lidar_sweep = reconstruction_->LidarSweep(sweep_id);
    const auto T_lidar = lidar_sweep.InverseProjectionMatrix();
    for (size_t idx = 0; idx < pc.size(); idx++){
        Eigen::Vector3d point3d_l;
        point3d_l << pc[idx].x, pc[idx].y, pc[idx].z;
        Eigen::Vector3d point3d_w = T_lidar.block<3,3>(0,0)*point3d_l+T_lidar.block<3,1>(0,3);
        pc.at(idx).x = point3d_w.x();
        pc.at(idx).y = point3d_w.y();
        pc.at(idx).z = point3d_w.z();
    }
}

void GetColor(const std::string images_path,
              const std::shared_ptr<Reconstruction> reconstruction_,
              const sweep_t sweep_id,
              const std::vector<image_t>& image_ids,
              std::vector<sensemap::PlyPoint>& pc,
              std::vector<float>& pnt_intensity){

    std::vector<Bitmap> bitmaps;
    bitmaps.resize(image_ids.size());
    for(int i = 0; i < image_ids.size(); i++){
        const auto& image = reconstruction_->Image(image_ids.at(i));
        auto image_name = image.Name();
        image_name = image_name.append(".jpg");
        std::string image_path = JoinPaths(images_path, image_name);
        bitmaps.at(i).Read(image_path);
    }

    std::set<size_t> remove_ids;
    for (size_t idx = 0; idx < pc.size(); idx++){
        Eigen::Vector3d point3d_w;
        point3d_w << pc[idx].x, pc[idx].y, pc[idx].z;
        
        bool is_ok = false;
        for (int i = 0; i < image_ids.size(); i++){
            const auto& image = reconstruction_->Image(image_ids.at(i));

            const Eigen::Vector3d proj_point3D =
                QuaternionRotatePoint(image.Qvec(), point3d_w) + image.Tvec();
            
            if (proj_point3D.z() < std::numeric_limits<double>::epsilon()){
                continue; 
            }

            const auto& camera = reconstruction_->Camera(image.CameraId());
            const Eigen::Vector2d proj_point2D =
                camera.WorldToImage(proj_point3D.hnormalized());
            if (proj_point2D.x() < 0 || proj_point2D.x() > camera.Width() ||
                proj_point2D.y() < 0 || proj_point2D.y() > camera.Height()){
                continue;
            }

            BitmapColor<uint8_t> color;
            bitmaps.at(i).GetPixel(proj_point2D.x(), proj_point2D.y(), &color);
            pc.at(idx).r = color.r;
            pc.at(idx).g = color.g;
            pc.at(idx).b = color.b;
            is_ok = true;
            break;
        }

        if(!is_ok){
            remove_ids.insert(idx);
        }
    }

    size_t i = 0, j = 0;
    for ( ; i < pc.size(); i++){
        if (remove_ids.find(i) != remove_ids.end()){
            continue;
        }
        pc.at(j) = pc.at(i);
        pnt_intensity.at(j) = pnt_intensity.at(i);
        j++;
    }
    pc.resize(j);
    pnt_intensity.resize(j);
    return;
}

void FusionLidarPoints(const std::string images_path,
    const std::string lidar_path,
    const std::shared_ptr<Reconstruction> reconstruction_,
    const std::unordered_map<sweep_t, image_t>& lidar2rig_map,
    const std::unordered_map<image_t, std::vector<image_t>>& rig2img,
    std::vector<sensemap::PlyPoint>& points,
    std::vector<uint32_t>& points_vis,
    std::vector<float>& points_intensity){
    std::cout << "Fusion Lidar Points Begin" << std::endl;
    points.clear();
    points_vis.clear();
    points_intensity.clear();

    size_t indices_print_step = lidar2rig_map.size() / 20 + 1;
    size_t progress = 0;
    for (const auto& l2r : lidar2rig_map){
        const auto& lidar_sweep = reconstruction_->LidarSweep(l2r.first);
        // std::cout << "Lidar pose: " << lidar_sweep.SweepID() << "-" 
        //     << lidar_sweep.Name() << "\t" 
        //     << lidar_sweep.Qvec().transpose() << ", "
        //     << lidar_sweep.Tvec().transpose() << std::endl;
        auto pc = ReadPCD(JoinPaths(lidar_path, lidar_sweep.Name()));

        if (points.empty()){
            points.reserve(pc.info.num_points * lidar2rig_map.size());
            points_vis.reserve(pc.info.num_points * lidar2rig_map.size());
            points_intensity.reserve(pc.info.num_points * lidar2rig_map.size());
        }

        std::vector<sensemap::PlyPoint> pnts;
        std::vector<float> intensity;
        ConvertPcd2Ply(pc, pnts, intensity);

        Translation(reconstruction_, l2r.first, pnts);

        const auto image_ids = rig2img.at(l2r.second);
        GetColor(images_path, reconstruction_, l2r.first, rig2img.at(l2r.second), pnts, intensity);
        
        points.insert(points.end(), pnts.begin(), pnts.end());
        const auto vis = std::vector<uint32_t>(pnts.size(), l2r.first);
        points_vis.insert(points_vis.end(), vis.begin(), vis.end());
        points_intensity.insert(points_intensity.end(), intensity.begin(), intensity.end());

        if (progress % indices_print_step == 0 || progress == lidar2rig_map.size() - 1) {
            std::cout<<"\r";
            std::cout<<"Points inserted ["<<progress<<" / "<<lidar2rig_map.size()<<"]"<<std::flush;
        }
        progress++;
    }
    points.shrink_to_fit();
    points_vis.shrink_to_fit();
    points_intensity.shrink_to_fit();

    if (points.size() != points_vis.size() || points.size() != points_intensity.size()){
        std::cout << "Error: num points is different, " << points.size() << ", " << points_vis.size() 
                  << ", " << points_intensity.size() << std::endl;
    }

    return;
}


void ComputeAverageDistance(const std::vector<PlyPoint>& fused_points,
  std::vector<float> &point_spacings, float* average_spacing,
  const int nb_neighbors) {
  Timer timer;
  timer.Start();

    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef Kernel::Point_3                                     Point_3;
    typedef boost::tuple<Point_3,int>                           Point_and_int;
    typedef CGAL::Search_traits_3<Kernel>                       Traits_base;
    typedef CGAL::Search_traits_adapter<Point_and_int,
    CGAL::Nth_of_tuple_property_map<0, Point_and_int>,
    Traits_base>                                              Traits;
    typedef CGAL::Orthogonal_k_neighbor_search<Traits>          K_neighbor_search;
    typedef K_neighbor_search::Tree                             Tree;
    typedef K_neighbor_search::Distance                         Distance;
    typedef K_neighbor_search::Point_with_transformed_distance  Point_with_distance;

  const size_t num_point = fused_points.size();
  std::vector<Point_3> points(num_point);
  for (std::size_t i = 0; i < num_point; i++) {
    const auto &fused_point = fused_points[i];
    points[i] = Point_3(fused_point.x, fused_point.y, fused_point.z);
  }
  std::vector<int> indices(num_point);
  std::iota(indices.begin(), indices.end(), 0);

  // Instantiate a KD-tree search.
  std::cout << "Instantiating a search tree for point cloud spacing ..." << std::endl;
  Tree tree(
    boost::make_zip_iterator(boost::make_tuple(points.begin(),indices.begin())),
    boost::make_zip_iterator(boost::make_tuple(points.end(),indices.end()))
  );
#ifdef CGAL_LINKED_WITH_TBB
  tree.build<CGAL::Parallel_tag>();
#endif

  *average_spacing = 0.0f;
  point_spacings.resize(num_point);

#ifndef CGAL_LINKED_WITH_TBB
  const int num_eff_threads = GetEffectiveNumThreads(-1);
  std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
  std::unique_ptr<ThreadPool> thread_pool;
  thread_pool.reset(new ThreadPool(num_eff_threads));
  {
    std::cout << "Starting average spacing computation ..." << std::endl;
    auto ComputePointSpace = [&](std::size_t i) {
      const auto &query = points[i];
      const auto &fused_point = fused_points[i];
      Eigen::Vector3f point(&fused_point.x);
      // performs k + 1 queries (if unique the query point is
      // output first). search may be aborted when k is greater
      // than number of input points
      Distance tr_dist;
      K_neighbor_search search(tree, query, nb_neighbors + 1);
      auto &point_spacing = point_spacings[i];
      point_spacing = 0.0f;
      std::size_t k = 0;
      for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
      {
        point_spacing += tr_dist.inverse_of_transformed_distance(it->second);
      }
      // output point spacing
      if (k > 1) {
        point_spacing /= (k - 1);
      }
    };

    for (std::size_t i = 0; i < num_point; ++i) {
      thread_pool->AddTask(ComputePointSpace, i);
    }
    thread_pool->Wait();

    for (auto & point_spacing : point_spacings) {
      *average_spacing += point_spacing;
    }
  }
#else
  {
    std::cout << "Starting average spacing computation(Parallel) ..." << std::endl;
    tbb::parallel_for (tbb::blocked_range<std::size_t> (0, num_point),
                     [&](const tbb::blocked_range<std::size_t>& r) {
                       for (std::size_t s = r.begin(); s != r.end(); ++ s) {
                        // Neighbor search can be instantiated from
                        // several threads at the same time
                        const auto &query = points[s];
                        const Eigen::Vector3f point(&fused_points.at(s).x);
                        K_neighbor_search search(tree, query, nb_neighbors + 1);

                        auto &point_spacing = point_spacings[s];
                        point_spacing = 0.0f;
                        std::size_t k = 0;
                        Distance tr_dist;
                        for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
                        {
                          point_spacing += tr_dist.inverse_of_transformed_distance(it->second);
                        }
                        // output point spacing
                        if (k > 1) {
                          point_spacing /= (k - 1);
                        }
                       }
                     });
    for (auto & point_spacing : point_spacings) {
      *average_spacing += point_spacing;
    }
  }
  #endif

  *average_spacing /= num_point;
  std::cout << "Average spacing: " << *average_spacing << std::endl;
  timer.PrintMinutes();
}

void Filtered(const float min_intensity, 
              const int nb_neighbors,
              const float remove_factor,
              std::vector<sensemap::PlyPoint>& points,
              std::vector<uint32_t>& points_vis,
              std::vector<float>& points_intensity){
    Timer timer;
    timer.Start();

    size_t num_pnts = points.size();

    size_t num_outlier(0);
    std::vector<float> dists;
    float average_dist;
    ComputeAverageDistance(points, dists, &average_dist, nb_neighbors);
    if (average_dist > 1e-6){
        double accum  = 0.0;
        for (size_t i = 0; i < num_pnts; ++i){
            accum += (dists[i] - average_dist) * (dists[i] - average_dist);
        }
        // float stdev_dist = sqrt(accum / (fused_points_score->size() - 1));
        const float stdev_dist = sqrt(accum / num_pnts);
        const float dist_thres = average_dist + remove_factor * stdev_dist;
        std::cout << "remove_factor, average_dist , stdev_dist: " << remove_factor << ", " 
            << average_dist << ", " << stdev_dist << std::endl;
        
        size_t i = 0, j = 0;
        for (; i < num_pnts; i++){
            if (points_intensity.at(i) < min_intensity || dists[i] > dist_thres){
                num_outlier++;
                continue;
            }
            points.at(j) = points.at(i);
            points_vis.at(j) = points_vis.at(i);
            points_intensity.at(j) = points_intensity.at(i);
            j++;
        }
        points.resize(j);
        points_vis.resize(j);
        points_intensity.resize(j);
    } else {
        size_t i = 0, j = 0;
        for (; i < num_pnts; i++){
            if (points_intensity.at(i) < min_intensity){
                num_outlier++;
                continue;
            }
            points.at(j) = points.at(i);
            points_vis.at(j) = points_vis.at(i);
            points_intensity.at(j) = points_intensity.at(i);
            j++;
        }
        points.resize(j);
        points_vis.resize(j);
        points_intensity.resize(j);
    }
    
    std::cout << StringPrintf("Remove %d outliers in %.3fs\n", 
                              num_outlier, timer.ElapsedSeconds());
    return;
}

// void DelaunaySample(std::string dense_reconstruction_path, 
//                     std::vector<PlyPoint>& points,
//                     std::vector<std::vector<uint32_t> >& points_vis,
//                     std::vector<std::vector<float> >& points_weights,
//                     const float distInsert,
//                     const float diffDepth){
//     std::cout << "DelaunaySample: " << std::endl;
//     std::unique_ptr<Workspace> workspace_;
//     Workspace::Options workspace_options;
//     workspace_options.max_image_size = -1;
//     workspace_options.image_as_rgb = true;
//     workspace_options.image_path = JoinPaths(dense_reconstruction_path, IMAGES_DIR);
//     workspace_options.workspace_path = dense_reconstruction_path;
//     workspace_options.workspace_format = "perspective";
//     workspace_.reset(new Workspace(workspace_options));
//     const Model& model = workspace_->GetModel();
//     std::cout << "model" << std::endl;
//     std::vector<float> points_sco;
//     PointsSample(points, points_sco, points_vis, points_weights, model, 5.0f, 0.01);
//     return;
// }

void Save(std::string dense_path,
          std::vector<sensemap::PlyPoint>& points,
          const std::vector<uint32_t>& points_vis,
          const std::vector<float>& points_intensity){
    size_t num_pnts = points.size();
    std::vector<std::vector<uint32_t> > points_visibility(num_pnts);
    std::vector<std::vector<float> > points_weight(num_pnts);
    
    for (size_t i = 0; i < num_pnts; i++){
        points_visibility[i].push_back(points_vis.at(i));

        float value = lidar_factor * points_intensity.at(i);
        points_weight[i].push_back(value);
    }
    // DelaunaySample(dense_path, points, points_visibility, points_weight, 5.0, 0.01);

    std::string save_path = JoinPaths(dense_path, LIDAR_NAME);
    WriteBinaryPlyPoints(save_path, points, true, true);
    WritePointsVisibility(save_path + ".vis", points_visibility);
    WritePointsWeight(save_path + ".wgt", points_weight);
    std::cout << "Save points to " << save_path << std::endl;
    return;
}

void LidarFusion(const std::string& lidar_path,
                 const std::string& workspace_path,
                 const int reconstrction_idx,
                 std::shared_ptr<Reconstruction> reconstruction,
                 std::unordered_map<image_t, sweep_t>& image2lidar_map,
                 std::vector<sensemap::PlyPoint>& points,
                 std::vector<uint32_t>& points_vis,
                 std::vector<float>& points_intensity){
    points.clear();
    points_vis.clear();
    points_intensity.clear();

    std::string sparse_path = JoinPaths(workspace_path, 
        std::to_string(reconstrction_idx), DENSE_DIR, SPARSE_DIR);

    reconstruction->ReadReconstruction(sparse_path);
    std::cout << "reconstruction: " << reconstruction->NumImages() << std::endl;

    std::unordered_map<image_t, image_t> id2rig;
    std::unordered_map<image_t, std::vector<image_t>> rig2ids;
    ReadImageMapText(sparse_path + "/imageId_map.txt", id2rig, rig2ids);

    std::unordered_map<sweep_t, image_t> lidar2rig_map;
    FindLidarImagePair(reconstruction, id2rig, rig2ids, lidar2rig_map);

    SaveLidarImagePair(sparse_path + "/image2lidar.txt", rig2ids, lidar2rig_map, image2lidar_map);

    std::string image_path = JoinPaths(workspace_path, 
        std::to_string(reconstrction_idx), DENSE_DIR, IMAGES_DIR);
    FusionLidarPoints(image_path, lidar_path, reconstruction, lidar2rig_map, rig2ids,
                      points, points_vis, points_intensity);

    Filtered(10, 6, 3, points, points_vis, points_intensity);
    std::cout << "reconstruction2: " << reconstruction->NumImages() << std::endl;

    std::string dense_path = JoinPaths(workspace_path, 
        std::to_string(reconstrction_idx), DENSE_DIR);
    Save(dense_path, points, points_vis, points_intensity);
};

namespace CGAL {
typedef CGAL::Simple_cartesian<double> kernel_t;
typedef CGAL::Projection_traits_xy_3<kernel_t> Geometry;
typedef CGAL::Delaunay_triangulation_2<Geometry> Delaunay;
typedef CGAL::Delaunay::Face_circulator FaceCirculator;
typedef CGAL::Delaunay::Face_handle FaceHandle;
typedef CGAL::Delaunay::Vertex_circulator VertexCirculator;
typedef CGAL::Delaunay::Vertex_handle VertexHandle;
typedef kernel_t::Point_3 Point;
}

struct RasterDepthDataPlaneData {
    Eigen::Matrix3f Kinv;
    DepthMap& depthMap;
    // NormalMap& normalMap;
    Eigen::Vector3f normal;
    Eigen::Vector3f normalPlane;
    inline void operator()(const int r, const int c) {
        if (r < 0 || r >= depthMap.GetHeight() || 
            c < 0 || c >= depthMap.GetWidth()) {
            return;
        }
        Eigen::Vector3f xc = Kinv * Eigen::Vector3f(c, r, 1.0f);
        const float z = 1.0f / normalPlane.dot(xc);
        // ASSERT(z > 0);
        depthMap.Set(r, c, z);
        // normalMap.SetSlice(r, c, normal.data());
    }
};
void GeneratePrior(const std::string& workspace_path,
                   const int reconstrction_idx,
                   const std::shared_ptr<Reconstruction> reconstruction,
                   const std::unordered_map<image_t, sweep_t>& image2lidar_map,
                   const std::vector<sensemap::PlyPoint>& points,
                   const std::vector<uint32_t>& points_vis,
                   const std::vector<float>& points_intensity,
                   const bool is_delaunay = false,
                   const bool has_plane = false){
    std::cout << "Generate Prior Maps Begin ..." << std::endl;

    std::string dense_path = JoinPaths(workspace_path, 
        std::to_string(reconstrction_idx), DENSE_DIR);
    std::string pri_depth_path = JoinPaths(dense_path, DEPTHS_DIR);
    std::string pri_intensity_path = JoinPaths(dense_path, "intensity_maps");
    
    size_t progress = 0;
    size_t num_images = reconstruction->Images().size();
    size_t indices_print_step = num_images / 20 + 1;

    auto DelaunayPrior = [&](const image_t image_id,
                             std::vector<Eigen::Vector4f>& uvd, 
                             DepthMap& prior_depth_map,
                             DepthMap& prior_wgt_map) {
        const auto image = reconstruction->Image(image_id);
        const auto& camera = reconstruction->Camera(image.CameraId());
        Eigen::Matrix3f K = camera.CalibrationMatrix().cast<float>();
        Eigen::Matrix3f Kinv = K.inverse();
        const int width = prior_depth_map.GetWidth();
        const int height = prior_depth_map.GetHeight();
        
        // Generate wgt map
        CGAL::Delaunay delaunay;
        for (const auto& point : uvd) {
            delaunay.insert(CGAL::Point(point.x(), point.y(), point.z()));
        }

        float sum_edge = 0;
        int num_edge = 0;
        std::vector<float> edges;

        for (CGAL::Delaunay::Face_iterator it = delaunay.faces_begin();
            it != delaunay.faces_end(); ++it) {
            const CGAL::Delaunay::Face& face = *it;
            const CGAL::Point i0(face.vertex(0)->point());
            const CGAL::Point i1(face.vertex(1)->point());
            const CGAL::Point i2(face.vertex(2)->point());

            Eigen::Vector3f c0(i0[0] * i0[2], i0[1] * i0[2], i0[2]);
            Eigen::Vector3f c1(i1[0] * i1[2], i1[1] * i1[2], i1[2]);
            Eigen::Vector3f c2(i2[0] * i2[2], i2[1] * i2[2], i2[2]);
            c0 = Kinv * c0;
            c1 = Kinv * c1;
            c2 = Kinv * c2;

            Eigen::Vector3f edge1(c1 - c0);
            Eigen::Vector3f edge2(c2 - c0);
            Eigen::Vector3f edge3(c2 - c1);
            
            float edge_t = edge1.norm() + edge2.norm() + edge3.norm();
            edges.push_back(edge_t);
            sum_edge += edge_t;
            num_edge++;
        }
        float mean_edge = sum_edge / num_edge;
        float accum  = 0.0;
        for (int i = 0; i < edges.size(); i++){
            accum += (mean_edge - edges.at(i)) * (mean_edge - edges.at(i));
        }
        float stdev = sqrt(accum/(edges.size()-1)); 
        RasterDepthDataPlaneData data = { Kinv, prior_depth_map};

        for (CGAL::Delaunay::Face_iterator it = delaunay.faces_begin();
            it != delaunay.faces_end(); ++it) {
            const CGAL::Delaunay::Face& face = *it;
            const CGAL::Point i0(face.vertex(0)->point());
            const CGAL::Point i1(face.vertex(1)->point());
            const CGAL::Point i2(face.vertex(2)->point());

            int u_min = std::min(i0[0], std::min(i1[0], i2[0]));
            int u_max = std::max(i0[0], std::max(i1[0], i2[0]));
            int v_min = std::min(i0[1], std::min(i1[1], i2[1]));
            int v_max = std::max(i0[1], std::max(i1[1], i2[1]));
            u_min = std::max(0, u_min);
            v_min = std::max(0, v_min);
            u_max = std::min(width - 1, u_max);
            v_max = std::min(height - 1, v_max);

            // compute the plane defined by the 3 points
            Eigen::Vector3f c0(i0[0] * i0[2], i0[1] * i0[2], i0[2]);
            Eigen::Vector3f c1(i1[0] * i1[2], i1[1] * i1[2], i1[2]);
            Eigen::Vector3f c2(i2[0] * i2[2], i2[1] * i2[2], i2[2]);
            c0 = Kinv * c0;
            c1 = Kinv * c1;
            c2 = Kinv * c2;

            Eigen::Vector3f edge1(c1 - c0);
            Eigen::Vector3f edge2(c2 - c0);
            Eigen::Vector3f edge3(c2 - c1);
            float edge_line = edge1.norm() + edge2.norm() + edge3.norm();
            float max_edge = std::max(edge3.norm(), std::max(edge1.norm(), edge2.norm()));

            data.normal = edge2.cross(edge1).normalized();
            data.normalPlane = data.normal * (1.0f / data.normal.dot(c0));

            const auto view = (c0 + c1 + c2).normalized();
            if (std::abs(data.normal.dot(view)) < 0.1 || 2 * max_edge > mean_edge + stdev){
            // if (std::abs(data.normal.dot(view)) < 0.05){
            // if (edge_line > mean_edge + 2 * stdev){
                continue;
            }

            Eigen::Vector2d v1(i1.x() - i0.x(), i1.y() - i0.y());
            Eigen::Vector2d v2(i2.x() - i1.x(), i2.y() - i1.y());
            Eigen::Vector2d v3(i0.x() - i2.x(), i0.y() - i2.y());

            for (int v = v_min; v <= v_max; ++v) {
                for (int u = u_min; u <= u_max; ++u) {
                    double m1 = v1.x() * (v - i0.y()) - (u - i0.x()) * v1.y();
                    double m2 = v2.x() * (v - i1.y()) - (u - i1.x()) * v2.y();
                    double m3 = v3.x() * (v - i2.y()) - (u - i2.x()) * v3.y();
                    if (m1 >= 0 && m2 >= 0 && m3 >= 0) {
                        data(v, u);
                    }
                }
            }
        }

        // Generate wgt map
        CGAL::Delaunay wgt_delaunay;
        for (const auto& point : uvd) {
            wgt_delaunay.insert(CGAL::Point(point.x(), point.y(), point.w()));
        }
        RasterDepthDataPlaneData wgt_data = { Kinv, prior_wgt_map};
        for (CGAL::Delaunay::Face_iterator it = wgt_delaunay.faces_begin();
            it != wgt_delaunay.faces_end(); ++it) {
            const CGAL::Delaunay::Face& face = *it;
            const CGAL::Point i0(face.vertex(0)->point());
            const CGAL::Point i1(face.vertex(1)->point());
            const CGAL::Point i2(face.vertex(2)->point());

            int u_min = std::min(i0[0], std::min(i1[0], i2[0]));
            int u_max = std::max(i0[0], std::max(i1[0], i2[0]));
            int v_min = std::min(i0[1], std::min(i1[1], i2[1]));
            int v_max = std::max(i0[1], std::max(i1[1], i2[1]));
            u_min = std::max(0, u_min);
            v_min = std::max(0, v_min);
            u_max = std::min(width - 1, u_max);
            v_max = std::min(height - 1, v_max);

            // compute the plane defined by the 3 points
            Eigen::Vector3f c0(i0[0] * i0[2], i0[1] * i0[2], i0[2]);
            Eigen::Vector3f c1(i1[0] * i1[2], i1[1] * i1[2], i1[2]);
            Eigen::Vector3f c2(i2[0] * i2[2], i2[1] * i2[2], i2[2]);
            c0 = Kinv * c0;
            c1 = Kinv * c1;
            c2 = Kinv * c2;

            Eigen::Vector3f edge1(c1 - c0);
            Eigen::Vector3f edge2(c2 - c0);
            Eigen::Vector3f edge3(c2 - c1);
            float edge_line = edge1.norm() + edge2.norm() + edge3.norm();
            
            wgt_data.normal = edge2.cross(edge1).normalized();
            wgt_data.normalPlane = wgt_data.normal * (1.0f / wgt_data.normal.dot(c0));

            Eigen::Vector2d v1(i1.x() - i0.x(), i1.y() - i0.y());
            Eigen::Vector2d v2(i2.x() - i1.x(), i2.y() - i1.y());
            Eigen::Vector2d v3(i0.x() - i2.x(), i0.y() - i2.y());

            for (int v = v_min; v <= v_max; ++v) {
                for (int u = u_min; u <= u_max; ++u) {
                    const auto depth = prior_depth_map.Get(v, u);
                    if (depth < 1e-6){
                        continue;
                    }
                    double m1 = v1.x() * (v - i0.y()) - (u - i0.x()) * v1.y();
                    double m2 = v2.x() * (v - i1.y()) - (u - i1.x()) * v2.y();
                    double m3 = v3.x() * (v - i2.y()) - (u - i2.x()) * v3.y();
                    if (m1 >= 0 && m2 >= 0 && m3 >= 0) {
                        wgt_data(v, u);
                    }
                }
            }
        }
    };

    auto SparsePrior = [&](const image_t image_id,
                           std::vector<Eigen::Vector4f>& uvdw, 
                           DepthMap& prior_depth_map,
                           DepthMap& prior_wgt_map) {
        if (!prior_depth_map.IsValid() || !prior_wgt_map.IsValid()){
            return;
        }
        const auto image = reconstruction->Image(image_id);
        const auto& camera = reconstruction->Camera(image.CameraId());
        Eigen::Matrix3f K = camera.CalibrationMatrix().cast<float>();
        Eigen::Matrix3f Kinv = K.inverse();
        const int width = prior_depth_map.GetWidth();
        const int height = prior_depth_map.GetHeight();

        const int pixel_area = 2 * width / 1000 + 1;
        // const int pixel_area = 1;

        for (const auto & point : uvdw) {
            float z = point.z();
            int u = point.x();
            int v = point.y();
            float wgt = std::log1p(point.w()) / max_value;
            wgt = std::min(wgt, 1.0f);
            int u_min = std::max(u - pixel_area, 0);
            int u_max = std::min(u + pixel_area, width - 1);
            int v_min = std::max(v - pixel_area, 0);
            int v_max = std::min(v + pixel_area, height - 1);
            for (int y = v_min; y <= v_max; ++y) {
                for (int x = u_min; x <= u_max; ++x) {
                    float d = prior_depth_map.Get(y, x);
                    if (d < 1e-6 || d > z) {
                        prior_depth_map.Set(y, x, z);
                        prior_wgt_map.Set(y, x, wgt);
                    }
                }
            }
        }    
    };

    auto Generate = [&](image_t image_id) {
        const auto image = reconstruction->Image(image_id);
        const auto& camera = reconstruction->Camera(image.CameraId());

        auto sweep_id = -1;
        if (image2lidar_map.find(image_id) != image2lidar_map.end()){
            sweep_id = image2lidar_map.at(image_id);
        }
    
        std::vector<Eigen::Vector4f> uvd;
        uvd.reserve(camera.Width()*camera.Height());
        float min_depth = std::numeric_limits<float>::max();
        float max_depth = std::numeric_limits<float>::min();
        for (size_t pnt_idx = 0; pnt_idx < points.size(); pnt_idx++){
            if (points_vis.at(pnt_idx) != sweep_id){
                continue;
            }
            Eigen::Vector3d point3d_w;
            point3d_w << points[pnt_idx].x, points[pnt_idx].y, points[pnt_idx].z;

            const Eigen::Vector3d proj_point3D =
                QuaternionRotatePoint(image.Qvec(), point3d_w) + image.Tvec();
            
            if (proj_point3D.z() < std::numeric_limits<double>::epsilon()){
                continue; 
            }
            const Eigen::Vector2d proj_point2D =
                camera.WorldToImage(proj_point3D.hnormalized());
            if (proj_point2D.x() < 0 || proj_point2D.x() > camera.Width() ||
                proj_point2D.y() < 0 || proj_point2D.y() > camera.Height()){
                continue;
            }
            uvd.emplace_back(proj_point2D.x(), proj_point2D.y(), proj_point3D.z(), points_intensity.at(pnt_idx));
            min_depth = std::min((float)proj_point3D.z(), min_depth);
            max_depth = std::max((float)proj_point3D.z(), max_depth);
        }
        uvd.shrink_to_fit();

        // std::string lidar_name = reconstruction->LidarSweep(sweep_id).Name();
        // std::string lidar_plane_path = JoinPaths(workspace_path, "planes", lidar_name + ".obj");
        // if(has_plane && ExistsFile(lidar_plane_path)){

        // }

        DepthMap prior_depth_map(camera.Width(), camera.Height(), min_depth, max_depth);
        prior_depth_map.Fill(0);
        DepthMap prior_wgt_map(camera.Width(), camera.Height(), 0.0f, 1.0f);
        prior_wgt_map.Fill(0);
        if (!uvd.empty() && uvd.size() > 1){
            if (is_delaunay && uvd.size() > 3){
                DelaunayPrior(image_id, uvd, prior_depth_map, prior_wgt_map);
            } else {
                SparsePrior(image_id, uvd, prior_depth_map, prior_wgt_map);
            }
        }

        auto image_name = image.Name();
        image_name = image_name.append(".jpg");
        auto prior_name = StringPrintf("%s.%s", image_name.c_str(), DEPTH_EXT);
        std::string depth_map_path = JoinPaths(pri_depth_path, prior_name);
        std::string parent_depth_map_path = GetParentDir(depth_map_path); 
        if (!ExistsPath(parent_depth_map_path)){
            boost::filesystem::create_directories(parent_depth_map_path);
        }
        prior_depth_map.Write(depth_map_path);
        prior_depth_map.ToBitmap().Write(depth_map_path + ".init.jpg");

        std::string intensity_map_path = JoinPaths(pri_intensity_path, prior_name);
        std::string parent_intensity_map_path = GetParentDir(intensity_map_path); 
        if (!ExistsPath(parent_intensity_map_path)){
            boost::filesystem::create_directories(parent_intensity_map_path);
        }
        prior_wgt_map.Write(intensity_map_path);
        prior_wgt_map.ToBitmap().Write(intensity_map_path + ".init.jpg");

        if (progress % indices_print_step == 0 || progress == num_images - 1) {
            std::cout<<"\r";
            std::cout<<"Generate Prior ["<< progress + 1 <<" / "<< num_images <<"]"<<std::flush;
        }
        progress++;
    };


    int num_eff_threads = GetEffectiveNumThreads(-1);
    int num_threads = std::min((size_t)num_eff_threads, reconstruction->NumRegisterImages());
    std::cout << "num_eff_threads: " << num_threads << std::endl;

    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_threads));
    // std::cout << reconstruction->Images().size() << std::endl;
    for (auto image : reconstruction->Images()){
        thread_pool->AddTask(Generate, image.first);
    }
    thread_pool->Wait();
    std::cout << "\nGenerate Prior Maps Done" << std::endl;
};

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
	Timer fusion_timer;
	fusion_timer.Start();

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());
    
	int reconstrction_idx = -1;
	if (argc > 2) {
        reconstrction_idx = atoi(argv[2]);
    }

    std::string lidar_path = param.GetArgument("lidar_path", "");
    CHECK(!lidar_path.empty()) << "lidar_path empty";

	std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    bool is_lidar_prior_delaunay = param.GetArgument("is_lidar_prior_delaunay", 0);
    lidar_factor = static_cast<float>(param.GetArgument("lidar_factor", 0.5f));

    size_t num_reconstruction = 0;
    for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
        const auto& reconstruction_path = 
            JoinPaths(workspace_path, std::to_string(reconstruction_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        num_reconstruction++;
    }

    size_t reconstruction_begin = reconstrction_idx < 0 ? 0 : reconstrction_idx;
    num_reconstruction = reconstrction_idx < 0 ? num_reconstruction : reconstrction_idx+1;
    for (size_t reconstruction_idx = reconstruction_begin; reconstruction_idx < num_reconstruction; 
        reconstruction_idx++) {
        
        float begin_memroy, end_memory;
		GetAvailableMemory(begin_memroy);

        std::shared_ptr<Reconstruction> reconstruction =  std::make_shared<Reconstruction>();
        std::unordered_map<sweep_t, image_t> image2lidar_map;
        std::vector<sensemap::PlyPoint> points;
        std::vector<uint32_t> points_vis;
        std::vector<float> points_intensity;
        LidarFusion(lidar_path, workspace_path, reconstruction_idx, reconstruction, 
                    image2lidar_map, points, points_vis, points_intensity);

        GeneratePrior(workspace_path, reconstruction_idx, reconstruction, image2lidar_map, 
                      points, points_vis, points_intensity, is_lidar_prior_delaunay);

		GetAvailableMemory(end_memory);
		std::cout << StringPrintf("Lidar Fusion Reconstruction %d Elapsed time: %.3f [minutes],\t Memory: %3f (%3f - %3f) [G]", 
								reconstruction_idx, fusion_timer.ElapsedMinutes(), 
								(begin_memroy - end_memory), begin_memroy, end_memory).c_str()
				<< std::endl;
    }

    return 0;
}
