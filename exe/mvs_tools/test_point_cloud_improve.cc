//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <utility>

#include "util/misc.h"
#include "util/string.h"
#include "util/timer.h"
#include "util/math.h"
#include "base/common.h"
#include "base/reconstruction_manager.h"
#include "base/undistortion.h"
#include "controllers/patch_match_controller.h"
#include "mvs/workspace.h"
#include "../Configurator_yaml.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <CGAL/tags.h>
#include <boost/iterator/zip_iterator.hpp>

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

#include "base/version.h"

using namespace sensemap;

std::string configuration_file_path;
std::unique_ptr<mvs::Workspace> workspace_;
std::string ply_path;

void ColorMap(const float intensity, uint8_t &r, uint8_t &g, uint8_t &b){
    if(intensity <= 0.25) {
        r = 0;
        g = 0;
        b = (intensity) * 4 * 255;
        return;
    }

    if(intensity <= 0.5) {
        r = 0;
        g = (intensity - 0.25) * 4 * 255;
        b = 255;
        return;
    }

    if(intensity <= 0.75) {
        r = 0;
        g = 255;
        b = (0.75 - intensity) * 4 * 255;
        return;
    }

    if(intensity <= 1.0) {
        r = (intensity - 0.75) * 4 * 255;
        g = (1.0 - intensity) * 4 * 255;
        b = 0;
        return;
    }
};

void ComputeAverageDistance(const std::vector<PlyPoint>& fused_points,
  std::vector<float> &point_spacings, float* average_spacing,
  const int nb_neighbors) {

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

    Timer timer;
    timer.Start();

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
            point_spacing = tr_dist.inverse_of_transformed_distance(it->second);
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
        std::cout << "Starting average spacing computation(tbb) ..." << std::endl;
        tbb::parallel_for (tbb::blocked_range<std::size_t> (0, num_point),
                        [&](const tbb::blocked_range<std::size_t>& r) {
                        for (std::size_t s = r.begin(); s != r.end(); ++ s) {
                            // Neighbor search can be instantiated from
                            // several threads at the same time
                            const auto &query = points[s];
                            K_neighbor_search search(tree, query, nb_neighbors + 1);

                            auto &point_spacing = point_spacings[s];
                            point_spacing = 0.0f;
                            std::size_t k = 0;
                            Distance tr_dist;
                            for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
                            {
                            float edist = tr_dist.inverse_of_transformed_distance(it->second);
                            point_spacing += edist;
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


void ComputePlaneScore(std::vector<PlyPoint>& pnts, 
						std::vector<std::vector<uint32_t> >& vis_pnts,
						std::vector<std::vector<float> >&  wgt_pnts,
						std::vector<float>& scores,
						std::vector<float>& dists,
						std::vector<float>& grads,
						std::vector<float>& neig_num,
						const float fused_dist_insert = 5.0,
						float radius_factor = 75 ){
	const int voxel_step = 15;
	const int num_neig_pnt = int(1.5 * 3.14 * voxel_step * voxel_step / 9.0f);
	const int num_min_pnt = 7;
	bool voxel_simp = true;
	// bool voxel_simp = false;
	bool has_wgt = false;

	std::unique_ptr<ThreadPool> samp_thread_pool;
	int num_eff_threads = GetEffectiveNumThreads(-1);
	
	if (pnts.size() ==wgt_pnts.size() ){
		has_wgt = true;
	}
	std::cout << "has_wgt: " << has_wgt << std::endl;

	std::vector<float> spacings;
	float average_spacing;
	ComputeAverageDistance(pnts, spacings, &average_spacing, 6);

	// int nb_neighbors = 1000;
	// const float nb_radius = 0.5;

    radius_factor /= fused_dist_insert;

    const auto& model = workspace_->GetModel();
    const float mean_dist = model.ComputeNthDistance(0.75);
    const float mean_angular = model.ComputeMeanAngularResolution();
    float angular_threshold = std::min(0.0215f, mean_angular * radius_factor * fused_dist_insert);

	float nb_radius = 1.0 * (average_spacing * radius_factor 
                                              + angular_threshold * mean_dist);
	
    std::cout << "\t=> nb_radius, average_dist, radius_factor, angular_threshold, mean_dist: " 
                      << nb_radius << ", " << average_spacing << ", " << radius_factor << ", " 
                      << angular_threshold << ", " << mean_dist << std::endl;

	// Voxel 
	std::vector<PlyPoint> simp_pnts;
	std::vector<float> simp_pnt_weights;
	std::unordered_map<uint64_t, uint64_t > index_map_voxels;
	std::vector<uint64_t> l0_voxels_keys;
	std::vector<std::unordered_map<uint64_t, std::vector<size_t> >> v_m_voxels_map;
	std::vector<std::unordered_map<uint64_t, float >> v_m_voxels_weight;

	const size_t num_point = pnts.size();
	Eigen::Vector3f lt, rb;
	size_t lenx = 0, leny = 0, lenz = 0, slide = 0;
	float voxel_length = 0;
	std::cout << "Compute Bounding Box ( " <<  num_point << " points )"<< std::endl;
	if (voxel_simp){
		lt.setConstant(std::numeric_limits<float>::max());
		rb.setConstant(std::numeric_limits<float>::lowest());
		for (size_t i = 0; i < num_point; ++i) {
			const PlyPoint& point = pnts[i];
			lt.x() = std::min(lt.x(), point.x);
			lt.y() = std::min(lt.y(), point.y);
			lt.z() = std::min(lt.z(), point.z);
			rb.x() = std::max(rb.x(), point.x);
			rb.y() = std::max(rb.y(), point.y);
			rb.z() = std::max(rb.z(), point.z);
		}
		std::cout << "LT: " << lt.transpose() << std::endl;
		std::cout << "RB: " << rb.transpose() << std::endl;

		int num_pyramid = 5;
		v_m_voxels_map.resize(num_pyramid);
		v_m_voxels_weight.resize(num_pyramid);

		std::vector<size_t> v_lenx(num_pyramid, 0), v_leny(num_pyramid, 0), v_lenz(num_pyramid, 0);
		std::vector<size_t> v_slide(num_pyramid, 0);
		std::vector<float> v_voxel_length(num_pyramid, 1.0f);

		auto LevelVox = [&](int level){
			v_voxel_length.at(level) = nb_radius / voxel_step * std::pow(2.0f, level);

			v_lenx.at(level) = (rb.x() - lt.x()) / v_voxel_length.at(level) + 1;
			v_leny.at(level) = (rb.y() - lt.y()) / v_voxel_length.at(level) + 1;
			v_lenz.at(level) = (rb.z() - lt.z()) / v_voxel_length.at(level) + 1;
			v_slide.at(level) = v_lenx.at(level)  * v_leny.at(level);

			// std::cout << "\t=> "<< level << "-level v_voxel_length, lenx, leny, lenz, slide: " 
			// 	<< v_voxel_length.at(level) << ", " << v_lenx.at(level) << ", " << v_leny.at(level) 
			// 	<< ", " << v_lenz.at(level) << ", " << v_slide.at(level) << std::endl;

			for (size_t i = 0; i < num_point; ++i) {
				const PlyPoint& point = pnts.at(i);
				if (point.x < lt.x() || point.y < lt.y() || point.z < lt.z() ||
					point.x > rb.x() || point.y > rb.y() || point.z > rb.z()) {
					continue;
				}
				uint64_t ix = (point.x - lt.x()) / v_voxel_length.at(level);
				uint64_t iy = (point.y - lt.y()) / v_voxel_length.at(level);
				uint64_t iz = (point.z - lt.z()) / v_voxel_length.at(level);

				uint64_t key = iz * v_slide.at(level) + iy * v_lenx.at(level) + ix;
				if (level ==0 && v_m_voxels_map.at(level)[key].empty()) {
					l0_voxels_keys.push_back(key);
				}
				v_m_voxels_map.at(level)[key].push_back(i);

				float wgt = 0.0;
				if (has_wgt){
					for (const auto wgt_pnt : wgt_pnts.at(i) ){
						wgt += wgt_pnt;
					}
				} else {
					wgt = (float)vis_pnts.at(i).size();
				}
				if (v_m_voxels_weight.at(level).find(key) ==  v_m_voxels_weight.at(level).end()){
					v_m_voxels_weight.at(level)[key] = wgt;
				} else {
					v_m_voxels_weight.at(level)[key] += wgt;
				}
			}
		};
		
		int num_l_avail_threads = std::min(num_eff_threads, num_pyramid);
		samp_thread_pool.reset(new ThreadPool(num_l_avail_threads));
		for (int level = 0; level < num_pyramid; level++){
			samp_thread_pool->AddTask(LevelVox, level);
		}
		samp_thread_pool->Wait();

		lenx = v_lenx.at(0);
		leny = v_leny.at(0);
		lenz = v_lenz.at(0);
		slide = v_slide.at(0);
		voxel_length = v_voxel_length.at(0);

		std::cout << "\t=> "<< 0 << "-level v_voxel_length, lenx, leny, lenz, slide: " 
			<< voxel_length << ", " << lenx << ", " << leny 
			<< ", " << lenz << ", " << slide << std::endl;

		const int num_neighbors = 26;
		const int neighbor_offs[num_neighbors][3] = 
			{ { -1, -1, -1 }, { 0, -1, -1 }, { 1, -1, -1}, 
			{ -1, 0, -1 }, { 0, 0, -1 }, { 1, 0, -1}, 
			{ -1, 1, -1 }, { 0, 1, -1 }, { 1, 1, -1}, 
			{ -1, -1, 0 }, { 0, -1, 0 }, { 1, -1, 0 }, 
			{ -1, 0, 0 }, { 1, 0, 0}, 
			{ -1, 1, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, 
			{ -1, -1, 1 }, { 0, 1, 1 }, { 1, -1, 1}, 
			{ -1, 0, 1 }, { 0, 0, 1 }, { 1, 0, 1}, 
			{ -1, 1, 1 }, { 0, 1, 1 }, { 1, 1, 1}};

		std::vector<uint64_t> voxel_map_index;
		std::vector<bool > l0_voxel_filter(l0_voxels_keys.size(), true);

		auto VoxFilter = [&](size_t begin_id, size_t end_id) {
			for (size_t id = begin_id; id < end_id; id++){
				const auto key = l0_voxels_keys.at(id);
				uint64_t iz = key / slide;
				uint64_t iy = (key % slide) / lenx;
				uint64_t ix = (key % slide) % lenx;
				
				bool is_ok = true;
				std::vector<bool> voxel_continuous(num_neighbors, true);
				for (int level = 0; level < num_pyramid; level++){
					uint64_t iz_level = iz / std::pow(2, level);
					uint64_t iy_level = iy / std::pow(2, level);
					uint64_t ix_level = ix / std::pow(2, level);

					uint64_t level_key = iz_level * v_slide.at(level) + iy_level * v_lenx.at(level) + ix_level;

					int num_eff_vox = 0;
					int num_neig_vox = 0;
					float sum_weight = 0.0f;
					for (int i = 0; i < num_neighbors; i++){
						uint64_t neighbor_key = (iz_level + neighbor_offs[i][2]) * v_slide.at(level) +
												(iy_level + neighbor_offs[i][1]) * v_lenx.at(level) + 
												(ix_level + neighbor_offs[i][0]);
						if (!voxel_continuous.at(i) || v_m_voxels_map.at(level).find(neighbor_key) == v_m_voxels_map.at(level).end()){
							voxel_continuous.at(i) = level < 3;
							continue;
						} 
						sum_weight += v_m_voxels_weight.at(level)[neighbor_key];
						num_neig_vox++;
						if (v_m_voxels_weight.at(level)[neighbor_key] > 5 * v_m_voxels_weight.at(level)[level_key]){
							num_eff_vox++;
						}
					}
					float mean_weight = sum_weight / num_neig_vox;
					if ( (num_neig_vox < 2 && level == 0) || (num_eff_vox > 8 && mean_weight >  2 * v_m_voxels_weight.at(level)[level_key])) {
						is_ok = false;
						break;
					}
				}
				l0_voxel_filter.at(id) = is_ok;
			}
		};

		num_eff_threads = GetEffectiveNumThreads(-1);
		int num_vox_per_threads = std::ceil(l0_voxels_keys.size() / num_eff_threads);
		int num_v_avail_threads = std::ceil(l0_voxels_keys.size() / num_vox_per_threads);
		samp_thread_pool.reset(new ThreadPool(num_v_avail_threads));
		for (size_t i = 0; i < num_v_avail_threads; i++){
			size_t begin_idx = i * num_vox_per_threads;
			size_t end_idx = std::min((i+ 1) * num_vox_per_threads, l0_voxels_keys.size());
			if (begin_idx >= end_idx){
				continue;
			}
			samp_thread_pool->AddTask(VoxFilter, begin_idx, end_idx);
		}
		samp_thread_pool->Wait();

		size_t num_index = 0;
		for (size_t i = 0; i < l0_voxels_keys.size(); i++){
			if (l0_voxel_filter.at(i)){
					size_t key = l0_voxels_keys.at(i);
					voxel_map_index.push_back(key);
					index_map_voxels[key] = num_index;
					num_index++;
			}
		}

		std::cout << "voxel size: " << l0_voxels_keys.size() << " -> " << voxel_map_index.size() << std::endl;

		simp_pnts.resize(voxel_map_index.size());
		simp_pnt_weights.resize(voxel_map_index.size());
		// auto SamplePoint = [&](uint64_t voxel_key, PlyPoint *samp_point, float * samp_point_weight) {
		auto SamplePoint = [&](size_t begin_idx, size_t end_idx) {
			for (size_t idx = begin_idx; idx < end_idx; idx++ ){
				uint64_t voxel_key = voxel_map_index.at(idx);
				PlyPoint& samp_point = simp_pnts[idx];
				float& samp_point_weight = simp_pnt_weights[idx];
				auto voxel_map = v_m_voxels_map.at(0).at(voxel_key);

				Eigen::Vector3f X(0, 0, 0);
				float sum_w(0.0);

				for (int k = 0; k < voxel_map.size(); ++k) {
					size_t point_idx = voxel_map.at(k);
					CHECK_LT(point_idx, num_point);
					const PlyPoint& point = pnts.at(point_idx);

					float wgt = 0.0;
					if (has_wgt){
						for (const auto wgt_pnt : wgt_pnts.at(point_idx) ){
							wgt += wgt_pnt;
						}
					} else {
						wgt = (float)vis_pnts.at(point_idx).size();
					}
					// wgt = 1.0f;

					X += wgt * Eigen::Vector3f(&point.x);
					sum_w += wgt;
				}
				X /= sum_w;
		
				// PlyPoint point;
				samp_point_weight = sum_w;
				samp_point.x = X[0];
				samp_point.y = X[1];
				samp_point.z = X[2];
				// float intensity = std::min(sum_w / 500.0f, 1.0f);
				// ColorMap(intensity, samp_point.r, samp_point.g, samp_point.b);
			}
		};

		num_eff_threads = GetEffectiveNumThreads(-1);
		int num_pnts_per_threads = std::ceil(voxel_map_index.size() / num_eff_threads);
		int num_avail_threads = std::ceil(voxel_map_index.size() / num_pnts_per_threads);
		samp_thread_pool.reset(new ThreadPool(num_avail_threads));
		for (size_t i = 0; i < num_avail_threads; ++i) {
			size_t begin_idx = i * num_pnts_per_threads;
			size_t end_idx = std::min((i+ 1) * num_pnts_per_threads, voxel_map_index.size());
			if (begin_idx >= end_idx){
				continue;
			}
			samp_thread_pool->AddTask(SamplePoint, begin_idx, end_idx);
		}
		samp_thread_pool->Wait();
	} else {
		simp_pnts = pnts;
		std::vector<float> temp_weights(pnts.size(), 1.0f);
		simp_pnt_weights.swap(temp_weights);
	}
	// WriteBinaryPlyPoints(ply_path + "-simp.ply", simp_pnts, 1, 1);
    {
        float mean_weight = std::accumulate(simp_pnt_weights.begin(), simp_pnt_weights.end(), 0.0f);
        mean_weight /= simp_pnt_weights.size();
        double accum  = 0.0;
        std::for_each (simp_pnt_weights.begin(), simp_pnt_weights.end(), 
            [&](const double d) {
            accum  += ( d - mean_weight ) * ( d - mean_weight);
        });
        float stdev = sqrt(accum/(simp_pnt_weights.size()-1));
        
        float max_weight = mean_weight + stdev;
        for (size_t i = 0; i < simp_pnts.size(); i++){
            float intensity = std::min(simp_pnt_weights.at(i) / max_weight, 1.0f);
            ColorMap(intensity, simp_pnts.at(i).r, simp_pnts.at(i).g, simp_pnts.at(i).b);
        }
		WriteBinaryPlyPoints(ply_path + "-simp.ply", simp_pnts, 1, 1);
        std::cout << "\t => save simp points ( " << simp_pnts.size() 
            << " points. mean_weight, stdev: " << mean_weight << ", " << stdev  << ")" << std::endl;
    }

	scores.clear();
	dists.clear();
	neig_num.clear();
	grads.clear();

	const int D = 3;
	typedef CGAL::Epick_d<CGAL::Dimension_tag<D> > K;
	typedef K::Point_d Point_3;
	typedef CGAL::Search_traits_d<K,CGAL::Dimension_tag<D> >  Traits;
	typedef CGAL::Random_points_in_cube_d<Point_3>       Random_points_iterator;
	typedef CGAL::Counting_iterator<Random_points_iterator> N_Random_points_iterator;
	typedef CGAL::Kd_tree<Traits> Tree;
	typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;
	typedef CGAL::Fuzzy_iso_box<Traits> Fuzzy_iso_box;

	const size_t num_simp = simp_pnts.size();
	scores.resize(num_point);
	dists.resize(num_point);
	neig_num.resize(num_point);
    grads.resize(num_point);

	std::vector<Point_3> points(num_simp);
	for (std::size_t i = 0; i < num_simp; i++) {
		const auto &fused_point = simp_pnts[i];
		points[i] = Point_3(fused_point.x, fused_point.y, fused_point.z);
	}

	// Instantiate a KD-tree search.
	std::cout << "Instantiating a search tree for point cloud spacing ..." << std::endl;
	Tree tree(points.begin(), points.end());

#ifdef CGAL_LINKED_WITH_TBB
  	tree.build<CGAL::Parallel_tag>();
#endif

	struct PlaneData{
		float score = -1.f;
		float dist = -1.f;
		Eigen::Vector3f normal = Eigen::Vector3f::Zero();
		uint16_t num_neighbor = 0;
		bool is_valid(){
			// return bool(score > -0.5 && dist > -0.5 && normal.norm() > 0.5 && num_neighbor > 25 );
			return bool(score > -0.5 && dist > -0.5 && normal.norm() > 0.5);
			// return bool(score > -0.5 && dist > -0.5);
		};
	};

	const float normal_threld = std::cos(DegToRad(15.0f));
	auto ComputeScore = [&](std::size_t s, float radius){
		// Neighbor search can be instantiated from
		// several threads at the same time
		const auto &query = Point_3(pnts[s].x, pnts[s].y, pnts[s].z);
		Fuzzy_sphere fs(query, radius);

		std::vector<Point_3> cgal_near_pnts;
		std::back_insert_iterator<std::vector<Point_3>> its(cgal_near_pnts);
		tree.search(its, fs);

		std::vector<Eigen::Vector3f> neighbors;
		std::vector<float> neighbors_weight;
		neighbors.reserve(cgal_near_pnts.size() + 1);
		neighbors_weight.reserve(cgal_near_pnts.size() + 1);

		neighbors.push_back(Eigen::Vector3f ((float)query.at(0), (float)query.at(1), (float)query.at(2)));
		neighbors_weight.push_back(1.0f);
		for (const auto pnt : cgal_near_pnts){
			Eigen::Vector3f point(pnt.at(0), pnt.at(1), pnt.at(2));
			float wgt(1.0f);
			// if (voxel_simp && 
			// 	!(point.x() < lt.x() || point.y() < lt.y() || point.z() < lt.z() ||
			// 	point.x() > rb.x() || point.y() > rb.y() || point.z() > rb.z())){
			 if (voxel_simp){
				uint64_t ix = (point.x() - lt.x()) / voxel_length;
				uint64_t iy = (point.y() - lt.y()) / voxel_length;
				uint64_t iz = (point.z() - lt.z()) / voxel_length;

				uint64_t key = iz * slide + iy * lenx + ix;
				wgt = simp_pnt_weights.at(index_map_voxels[key]);
			}
			neighbors_weight.push_back(wgt);
			neighbors.push_back(point);
		}
		if (neighbors.size() <= num_min_pnt) {
			struct PlaneData data;
			data.score = -1.0f;
			data.dist = -1.0f;
			// data.dist = dist;
			data.normal = Eigen::Vector3f::Zero();
			data.num_neighbor = neighbors.size();

			return data;
		}

		Eigen::Vector3f svd_singular_values;
		Eigen::Vector3f svd_singular_vector;
		Eigen::Vector3f centroid(Eigen::Vector3f::Zero());
		float sum_wgt(0.0f);
		for (int idx = 1; idx < neighbors.size(); idx++) {
			const auto &point = neighbors.at(idx);
			const auto &wgt = neighbors_weight.at(idx);
			centroid += wgt * point;
			sum_wgt += wgt;
		}
		centroid /= sum_wgt;
		// float mean_wgt = sum_wgt / float(neighbors.size() - 1);
		// float accum = 0.0f;
		// for (int idx = 1; idx < neighbors.size(); idx++){
		// 	accum += ( neighbors_weight.at(idx) - mean_wgt ) 
		// 		  * ( neighbors_weight.at(idx) - mean_wgt);
		// }
		// float stdev = sqrt(accum / float(neighbors.size() -2));
		// float wgt_threld = mean_wgt - stdev;
		// int num_eff_neighbors = 0.0f;
		// for (int idx = 1; idx < neighbors.size(); idx++){
		// 	if (neighbors_weight.at(idx) > wgt_threld){
		// 		num_eff_neighbors++;
		// 	}
		// }
		
		// bool select_wgt = num_eff_neighbors > 50 && wgt_threld > 0.0f;
		Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
		for (int idx = 1; idx < neighbors.size(); idx++) {
			// if (select_wgt && neighbors_weight.at(idx) < wgt_threld){
			// 	continue;
			// }
			const auto &point = neighbors.at(idx);
			float wgt = neighbors_weight.at(idx);
			Eigen::Vector3f V = wgt * (point - centroid);
			// Eigen::Vector3f V = point - centroid;
			M += V * V.transpose();
		}

		Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
		svd_singular_values = svd.singularValues();
		svd_singular_vector = svd.matrixV().col(2);
		svd_singular_vector.normalize();

		float mean_dist = 0;
		for (int idx = 1; idx < neighbors.size(); idx++) {
			const auto &point = neighbors.at(idx);
			float wgt = neighbors_weight.at(idx);
			mean_dist += wgt * std::abs((point - centroid).dot(svd_singular_vector));
		}
		mean_dist /=  sum_wgt;

		// float score = exp(- 0.5 * svd_singular_values[2]/svd_singular_values[1]);
		float svd_score = exp(- 0.8 * svd_singular_values[2]/svd_singular_values[1]) 
							* exp(  0.4 * std::sqrt(svd_singular_values[1]/svd_singular_values[0])  - 0.4);
		// float score = exp(- std::sqrt(svd_singular_values[2]/svd_singular_values[1])) 
		// 					* exp(  0.2 * svd_singular_values[1]/svd_singular_values[0] - 0.2);
		// float score = exp(- svd_singular_values[2]/svd_singular_values[1]) 
		// 					* exp(  0.5 * (svd_singular_values[1]/svd_singular_values[0]) *  (svd_singular_values[1]/svd_singular_values[0])  - 0.5);

		// float normal_cos = std::abs(svd_singular_vector.dot(
		// 	Eigen::Vector3f(pnts[s].nx, pnts[s].ny, pnts[s].nz)));
		// float normal_score = std::min(normal_cos, normal_threld) + 1 - normal_threld;
		// float normal_score = 1.0f;
        // if (normal_cos < normal_threld){
        //     normal_score = 0.5 + 0.5 * normal_cos;
    	// }

		float dist = std::abs((neighbors.at(0) - centroid).dot(svd_singular_vector));

		struct PlaneData data;
		// data.score = svd_score * normal_score;
		data.score = svd_score;
		data.dist = std::min( 0.1f * dist / mean_dist, 1.0f);
		data.normal = svd_singular_vector;
		data.num_neighbor = neighbors.size();

		return data;
	};
	
    auto& images = model.images;
	std::vector<Bitmap> bitmaps_;
	bitmaps_.resize(images.size());
	{
		std::cout << "Read Images..." << std::endl;
		// for (size_t image_idx = 0; image_idx < images.size(); image_idx++){
		auto Read = [&](int image_idx){
			bitmaps_.at(image_idx).Read(workspace_->GetBitmapPath(image_idx),
				workspace_->GetOptions().image_as_rgb);
		};
		std::unique_ptr<ThreadPool> read_thread_pool;
		const int num_eff_threads = std::min(GetEffectiveNumThreads(-1), (int)images.size());
		read_thread_pool.reset(new ThreadPool(num_eff_threads));
		for (size_t image_idx = 0; image_idx < images.size(); image_idx++){
			read_thread_pool->AddTask(Read, image_idx);
		}
		read_thread_pool->Wait();
		std::cout << "Read Images Done " << std::endl;
	}

    const float filter_kernel_x[3][3] = {{-0.25, 0, 0.25}, {-0.5, 0, 0.5}, {-0.25, 0 ,0.25}};
    const float filter_kernel_y[3][3] = {{-0.25, -0.5, -0.25}, {0, 0, 0}, {0.25, 0.5, 0.25}};
    auto ComputeGradKernel = [&](size_t image_id, int col, int row, int delt){
        float conv_x = 0, conv_y = 0;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                BitmapColor<uint8_t> color;
                bitmaps_.at(image_id).GetPixel(col + dc * delt, row + dr * delt, &color);
                float grey = 0.5 * color.r + 0.25 * color.g + 0.25 * color.b;
                conv_x += grey * filter_kernel_x[dr + 1][dc + 1];
                conv_y += grey * filter_kernel_y[dr + 1][dc + 1];
            }
        }
        float mag = sqrt(conv_x * conv_x + conv_y * conv_y) / 255.f;
        return mag;
    };

    auto ComputeGrad = [&](size_t pnt_id){
        const auto pnt = pnts.at(pnt_id);
        const auto pnt_vis = vis_pnts.at(pnt_id);
        const Eigen::Vector3f xyz(pnt.x,pnt.y,pnt.z);
        std::vector<float> pnt_conf;
        float m_conf(0.0f);
        for (const auto vis : pnt_vis){
            Eigen::Map<const Eigen::RowMatrix3x4f> P(images[vis].GetP());
            const Eigen::Vector3f next_proj = P * xyz.homogeneous();
            int vis_col = static_cast<int>(std::round(next_proj(0) / next_proj(2)));
            int vis_row = static_cast<int>(std::round(next_proj(1) / next_proj(2)));

            const int vis_width = images.at(vis).GetWidth();
            const int vis_height = images.at(vis).GetHeight();
            int delt = 1;
			float max_grad = 0;
			for (int t = 0; t < 5; t++){
				int step = std::pow(2, t) * delt;
				if (vis_col > step && vis_col < vis_width - step && 
                	vis_row > step && vis_row < vis_height - step){
					float grad = ComputeGradKernel(vis, vis_col, vis_row, step);
                	// pnt_conf.push_back(grad);
					if (max_grad < grad){
						max_grad = grad;
					}
				}
			}
            pnt_conf.push_back(max_grad);
        }
        
        if (pnt_conf.size() >= 1){
			for (size_t i = 0; i < pnt_conf.size(); ++i) {
				m_conf += pnt_conf[i];
			}
            m_conf = m_conf / pnt_conf.size();
        } else {
            m_conf = 0.0f;
        }
		// if (pnt_conf.size() >= 1){
		// 	// size_t nth = pnt_conf.size() / 2;
		// 	size_t nth = pnt_conf.size() - 1;
        //     std::nth_element(pnt_conf.begin(), pnt_conf.begin() + nth, pnt_conf.end());
		// 	m_conf = pnt_conf.at(nth);
        // } else {
        //     m_conf = 0.0f;
        // }
        // return (1.0f - m_conf) * (1.0f - m_conf);
		m_conf = std::min(m_conf, 0.4f);
		float grad_score = (std::cos( 2 * 3.141593 * m_conf) + 1) / 2;
        return grad_score * grad_score;
    };

	std::size_t progress = 0;
	std::size_t indices_print_step = num_point / 20;

#ifndef CGAL_LINKED_WITH_TBB
	std::cout << "Starting muti-threads nearst neighbor computation ..." << std::endl;
	// const int num_eff_threads = GetEffectiveNumThreads(-1);
	std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
	std::unique_ptr<ThreadPool> thread_pool;
	thread_pool.reset(new ThreadPool(num_eff_threads));
	{
		auto ComputePointSpace = [&](std::size_t s) {

			if (progress % indices_print_step == 0) {
				std::cout<<"\r";
				std::cout<<"Points computed ["<<progress<<" / "<< num_point <<"]"<<std::flush;
			}
			++progress;

			PlaneData data_1;
			float radius_step = 3.0f;
			float radius_factor = 1.0f / radius_step;
			while (!data_1.is_valid()) {
				radius_factor *= 1.2;
				data_1 = ComputeScore(s, nb_radius * radius_factor);
				if (radius_factor > 1.8) {
					data_1.score = 0.1;
					data_1.dist = 0.1f;
					break;
				}
			}
			radius_factor *= radius_step;

			PlaneData data_2= ComputeScore(s, nb_radius * radius_factor);

			// float norm_score = std::abs(data_1.normal.dot(data_2.normal))
			// 	* std::sqrt(std::abs(data_2.normal.dot(Eigen::Vector3f(pnts[s].nx, pnts[s].ny, pnts[s].nz))));
			float norm_score = std::abs(data_1.normal.dot(data_2.normal));

			// float normal_diff = std::abs(data_2.normal.dot(
			// 	Eigen::Vector3f(pnts[s].nx, pnts[s].ny, pnts[s].nz)));
			// float normal_error = normal_diff > std::cos(DegToRad(15.0f)) ? 1.0f : normal_diff;
			// float norm_score = std::abs(data_1.normal.dot(data_2.normal)) * normal_error;

			float score = data_1.score * data_2.score * norm_score;

			// float dist = std::sqrt(score * data_1.dist * data_2.dist);
			// float dist = std::sqrt(score) * data_2.dist;
			float dist = score * score * data_1.dist;
			dists.at(s) = dist;

			// float normal_diff = std::abs(data_2.normal.dot(
			// 	Eigen::Vector3f(pnts[s].nx, pnts[s].ny, pnts[s].nz)));
			// float normal_error =  0.5 + 0.5 * std::min(normal_diff + 1.f - std::cos(DegToRad(15.0f)), 1.0f);
			// scores.at(s) = score * normal_error;
			scores.at(s) = score;

			neig_num.at(s) =  (float)data_1.num_neighbor / num_neig_pnt;

            grads.at(s) = ComputeGrad(s);
			return;
		};

		for (std::size_t i = 0; i < num_point; ++i) {
			thread_pool->AddTask(ComputePointSpace, i);
		}
		thread_pool->Wait();
	}
#else
	std::cout << "Starting parallel plane score computation ..." << std::endl;
	tbb::parallel_for (tbb::blocked_range<std::size_t> (0, num_point),
					[&](const tbb::blocked_range<std::size_t>& r) {
			for (std::size_t s = r.begin(); s != r.end(); ++ s) {

			if (progress % indices_print_step == 0) {
				std::cout<<"\r";
				std::cout<<"Points computed ["<<progress<<" / "<< num_point <<"]"<<std::flush;
			}
			++progress;

			PlaneData data_1;
			float radius_step = 3.0f;
			float radius_factor = 1.0f / radius_step;
			while (!data_1.is_valid()) {
				radius_factor *= 1.2;
				data_1 = ComputeScore(s, nb_radius * radius_factor);
				if (radius_factor > 1.8) {
					data_1.score = 0.1;
					data_1.dist = 0.1f;
					break;
				}
			}
			radius_factor *= radius_step;

			PlaneData data_2= ComputeScore(s, nb_radius * radius_factor);

			// float normal_diff = std::abs(data_2.normal.dot(
			// 	Eigen::Vector3f(pnts[s].nx, pnts[s].ny, pnts[s].nz)));
			// float normal_error = normal_diff > std::cos(DegToRad(30.0f)) ? 1.0f : normal_diff;
			// float norm_score = std::abs(data_1.normal.dot(data_2.normal)) * normal_error;
			// float norm_score = std::abs(data_1.normal.dot(data_2.normal))
			// 	* std::sqrt(normal_error);
			// float norm_score =  std::sqrt(std::abs(data_1.normal.dot(data_2.normal)) * normal_error);
			float norm_score = std::abs(data_1.normal.dot(data_2.normal));

			float score = data_1.score * data_2.score * norm_score;
			// float normal_diff = std::abs(data_2.normal.dot(
			// 	Eigen::Vector3f(pnts[s].nx, pnts[s].ny, pnts[s].nz)));
			// float normal_error = 0.5 + 0.5 * std::min(normal_diff + 1.f - std::cos(DegToRad(15.0f)), 1.0f);
			// scores.at(s) = score * normal_error;
			scores.at(s) = score;

			// float dist = std::sqrt(score * data_1.dist * data_2.dist);
			// float dist = std::sqrt(score) * data_1.dist;
			// float dist = score * data_2.dist;
			float dist = score * score * data_1.dist;
			dists.at(s) = dist;

			neig_num.at(s) = (float)data_1.num_neighbor / num_neig_pnt;

            grads.at(s) = ComputeGrad(s);
		}
	});
#endif

	float max_score = *std::max_element(scores.begin(),scores.end());
	float min_score = *std::min_element(scores.begin(),scores.end());
	std::cout << "\nmax, min: " << *std::max_element(scores.begin(),scores.end()) << ", "
		<< *std::min_element(scores.begin(),scores.end()) << std::endl;

	// for (int i = 0; i < num_point; i ++){
	// 	scores.at(i) = (scores.at(i) - min_score) / (max_score - min_score);
	// }

	// std::cout << "max, min: " << *std::max_element(scores.begin(),scores.end()) << ", "
	// 	<< *std::min_element(scores.begin(),scores.end()) << std::endl;

}

int main(int argc, char *argv[]) {
    using namespace sensemap;
	using namespace mvs;

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);


	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
    bool fused_delaunay_sample = param.GetArgument("fused_delaunay_sample", true);
	float fused_dist_insert = param.GetArgument("dist_insert", 5.0f);
	float plane_raidus_factor = param.GetArgument("plane_raidus_factor", 100.0f);
	float plane_dist_threld = param.GetArgument("plane_dist_threld", 0.85f);
	float plane_score_thred = param.GetArgument("plane_score_thred", 0.9f);

	if (!fused_delaunay_sample){
		fused_dist_insert = 1.0f;
	}

	mvs::Workspace::Options workspace_options;
	workspace_options.workspace_path = JoinPaths(workspace_path, std::to_string(0), DENSE_DIR);
	workspace_options.image_path = JoinPaths(workspace_path, std::to_string(0), DENSE_DIR, IMAGES_DIR);
	workspace_options.image_as_rgb = false;
	workspace_options.cache_size = 1e-6;
	workspace_options.workspace_format = "perspective";
    workspace_options.input_type = PHOTOMETRIC_TYPE;
	workspace_.reset(new mvs::Workspace(workspace_options));
	
	ply_path = JoinPaths(workspace_path, std::to_string(0), DENSE_DIR, FUSION_NAME);
	auto input_vis_path = ply_path + ".vis";
    auto input_wgt_path = ply_path + ".wgt";

    std::vector<PlyPoint> ply_points;
	std::vector<std::vector<uint32_t> > vis_points;
    std::vector<std::vector<float> >  weight_points;

	if (!ExistsFile(ply_path)){
		std::cout << "!ExistsFile(" << ply_path << ")" << std::endl;
		return -1;
	}

	// Pwn_vector points;
	ply_points = ReadPly(ply_path);
	ReadPointsVisibility(input_vis_path, vis_points);
	if(ExistsFile(input_wgt_path)){
		ReadPointsWeight(input_wgt_path, weight_points);
	} else {
		weight_points.resize(vis_points.size());
		for (size_t i = 0; i < vis_points.size(); i++){
			std::vector<float> temp_weight(vis_points[i].size(), 1.0f);
			weight_points[i].swap(temp_weight);
		}
	}

    Timer timer;
    timer.Start();

	std::vector<float> plane_scores;
	std::vector<float> dist_socres;
	std::vector<float> grads_scores;
	std::vector<float> neig_num;
	ComputePlaneScore(ply_points, vis_points, weight_points, plane_scores, dist_socres, 
					  grads_scores, neig_num, fused_dist_insert, plane_raidus_factor);

#if 1
	std::vector<PlyPoint> dist_pnts;
	std::vector<PlyPoint> color_pnts;
	std::vector<PlyPoint> color_plane_pnts;
	std::vector<PlyPoint> grad_pnts;
	std::vector<PlyPoint> score_grad_pnts;
	std::vector<PlyPoint> neig_pnts;
	for (int i = 0; i < ply_points.size(); i++){
		PlyPoint pnt = ply_points.at(i);
		float dist = dist_socres.at(i);
		dist = std::min(dist, 1.0f);
		ColorMap(dist, pnt.r, pnt.g, pnt.b);
		dist_pnts.push_back(pnt);

		float neig_number = neig_num.at(i);
		neig_number = std::min(neig_number, 1.0f);
		ColorMap(neig_number, pnt.r, pnt.g, pnt.b);
		neig_pnts.push_back(pnt);

		float grad = grads_scores.at(i);
		grad = std::min(grad, 1.0f);
		ColorMap(grad, pnt.r, pnt.g, pnt.b);
		grad_pnts.push_back(pnt);

		float score = plane_scores.at(i);
		score = std::min(score, 1.0f);
		ColorMap(score, pnt.r, pnt.g, pnt.b);
		color_pnts.push_back(pnt);

		score *= grad;
		ColorMap(score, pnt.r, pnt.g, pnt.b);
		score_grad_pnts.push_back(pnt);
		if (score > plane_score_thred){
			color_plane_pnts.push_back(pnt);
		}
	}
	WriteBinaryPlyPoints(ply_path + "-score.ply", color_pnts, 1, 1);
	WriteBinaryPlyPoints(ply_path + "-plane-color.ply", color_plane_pnts, 1, 1);
	WriteBinaryPlyPoints(ply_path + "-dist-color.ply", dist_pnts, 1, 1);
	WriteBinaryPlyPoints(ply_path + "-grad.ply", grad_pnts, 1, 1);
	WriteBinaryPlyPoints(ply_path + "-color.ply", score_grad_pnts, 1, 1);
	WriteBinaryPlyPoints(ply_path + "-neig.ply", neig_pnts, 1, 1);
#endif

	int new_num = 0;
	int ori_num = ply_points.size();
	std::vector<PlyPoint> remove_points;
	for (int i = 0; i < ori_num; i++){
		// float score = plane_scores.at(i);
		float score = dist_socres.at(i);
		score = std::min(score, 1.0f);
		if (score > plane_dist_threld){
			remove_points.push_back(ply_points.at(i));
			continue;
		}
		ply_points.at(new_num) = ply_points.at(i);
		vis_points.at(new_num) = vis_points.at(i);
		weight_points.at(new_num) = weight_points.at(i);
		plane_scores.at(new_num) = plane_scores.at(i) * grads_scores.at(i);
		new_num++;	
		// tem_ply_points.push_back(ply_points.at(i));
		// tem_vis_points.push_back(vis_points.at(i));
		// tem_weight_points.push_back(weight_points.at(i));
		// tem_score_points.push_back(plane_scores.at(i));
	}
	ply_points.resize(new_num);
	vis_points.resize(new_num);
	weight_points.resize(new_num);
	plane_scores.resize(new_num);

	std::cout << "size: " << ori_num << " -> "<< new_num << std::endl;
	
    timer.PrintMinutes();

	std::cout << "Save Data..." << std::endl;
	std::string output_path = ply_path + "-samp.ply";
	WriteBinaryPlyPoints(output_path, ply_points, true, true);
	WritePointsVisibility(output_path + ".vis", vis_points);
	WritePointsWeight(output_path + ".wgt", weight_points);
	WritePointsScore(output_path + ".sco", plane_scores);
	std::string output_path_r = ply_path + "-remove.ply";
	WriteBinaryPlyPoints(output_path_r, remove_points, true, true);

	std::vector<PlyPoint> color_map;
	for (int i = 0; i <= 100; i++){
		PlyPoint pnt;
		pnt.x = i * 0.01f;
		pnt.y = 0.f;
		pnt.z = 0.f;
		ColorMap(i * 0.01f, pnt.r, pnt.g, pnt.b);
		color_map.push_back(pnt);
		if (i % 10 == 0) {
			pnt.y = 0.01f;
			pnt.z = 0.f;
			color_map.push_back(pnt);
			pnt.y = 0.f;
			pnt.z = 0.01f;
			color_map.push_back(pnt);
			pnt.y = 0.f;
			pnt.z = -0.01f;
			color_map.push_back(pnt);
			pnt.y = -0.01f;
			pnt.z = 0.f;
			color_map.push_back(pnt);
		}
	}
	WriteBinaryPlyPoints(ply_path + "-map.ply", color_map, false, true);


    timer.PrintMinutes();

	return 0;
}
