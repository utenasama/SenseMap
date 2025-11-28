#include "util/common.h"
#include "util/timer.h"
#include "util/threading.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

// Types
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

namespace sensemap
{

Eigen::Matrix3f ComputePovitMatrix(
    const std::vector<Eigen::Vector3f> &points){
    Eigen::Matrix3f pivot;
    Eigen::Vector3f centroid(Eigen::Vector3f::Zero());
    for (const auto &point : points) {
        centroid += point;
    }
    std::size_t point_num = points.size();
    centroid /= point_num;

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (const auto &point : points) {
        Eigen::Vector3f V = point - centroid;
        M += V * V.transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    pivot = svd.matrixU().transpose();
    return pivot;
}

float ComputeAvergeSapcing(
    std::vector<float>& point_spacings, 
    const std::vector<Eigen::Vector3f> &points_sparse,
    const unsigned int nb_neighbors){
	Timer timer;
  	timer.Start();

		
	const size_t num_point = points_sparse.size();
	std::vector<Point_3> points(num_point);
	for (std::size_t i = 0; i < num_point; i++) {
		const auto &point = points_sparse[i];
		points[i] = Point_3(point[0], point[1], point[2]);
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

	float average_spacing = 0.0f;
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
		// const auto &fused_point = points_sparse[i];
		Eigen::Vector3f point(points_sparse[i]);
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
			average_spacing += point_spacing;
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
							// const Eigen::Vector3f point(&fused_points.at(s).x);
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
		average_spacing += point_spacing;
		}
	}
#endif

	average_spacing /= num_point;
	std::cout << "Average spacing: " << average_spacing << std::endl;
	timer.PrintMinutes();

    return average_spacing;
}

}
