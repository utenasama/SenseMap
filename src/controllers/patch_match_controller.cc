//Copyrigt (c) 2019, SenseTime Group.
//All rights reserved.

#include <stdio.h>
#include<algorithm>
#include<vector>
#include <malloc.h>
#include "yaml-cpp/yaml.h"

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

#include "util/exception_handler.h"
#include "base/triangulation.h"
#include "base/reconstruction.h"
#include "util/types.h"
#include "util/math.h"
#include "util/threading.h"
#include "util/misc.h"
#include "util/cuda.h"
#include "util/panorama.h"
#include "util/hash.h"
#include "base/common.h"
#include "util/kmeans.h"
#include "util/proc.h"

#include "mvs/utils.h"
#include "mvs/patch_match_cuda.h"
#include "mvs/depth_map.h"
#include "mvs/normal_map.h"
#include "mvs/delaunay/delaunay_triangulation.h"

#include "controllers/patch_match_controller.h"

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

// #define SAVE_CONF_MAP

namespace sensemap {
namespace mvs {

using namespace utility;

const int num_perspective_per_image = 6;
const double rolls[num_perspective_per_image] = {0, 0, 0, 0, 0, 0};
const double pitches[num_perspective_per_image]= {0, 60, 120, 180, 240, 300};
const double yaws[num_perspective_per_image] = {0, 0, 0, 0, 0, 0};

const double fov_w = 60.0;
const double fov_h = 90.0;

namespace {

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

struct RasterDepthDataPlaneData {
    Eigen::Matrix3f Kinv;
    DepthMap& depthMap;
    NormalMap& normalMap;
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
        normalMap.SetSlice(r, c, normal.data());
    }
};
// Raster the given triangle and output the position of each pixel of the triangle;
// based on "Advanced Rasterization" by Nick (Nicolas Capens)
// http://devmaster.net/forums/topic/1145-advanced-rasterization
void RasterizeTriangle(const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, 
                       const Eigen::Vector3f &v3, RasterDepthDataPlaneData & parser) {
	// std::cout << "RasterizeTriangle" << std::endl
    //           << "v1: " << v1.transpose() << std::endl
    //           << "v2: " << v2.transpose() << std::endl
    //           << "v3: " << v3.transpose() << std::endl;
    
    // 28.4 fixed-point coordinates
	const int64_t Y1 = floor(float(16) * v1(1) + 0.5);
	const int64_t Y2 = floor(float(16) * v2(1) + 0.5);
	const int64_t Y3 = floor(float(16) * v3(1) + 0.5);

	const int64_t X1 = floor(float(16) * v1(0) + 0.5);
	const int64_t X2 = floor(float(16) * v2(0) + 0.5);
	const int64_t X3 = floor(float(16) * v3(0) + 0.5);

	// Deltas
	const int64_t DX12 = X1 - X2;
	const int64_t DX23 = X2 - X3;
	const int64_t DX31 = X3 - X1;

	const int64_t DY12 = Y1 - Y2;
	const int64_t DY23 = Y2 - Y3;
	const int64_t DY31 = Y3 - Y1;

	// Fixed-point deltas
	const int64_t FDX12 = DX12 << 4;
	const int64_t FDX23 = DX23 << 4;
	const int64_t FDX31 = DX31 << 4;

	const int64_t FDY12 = DY12 << 4;
	const int64_t FDY23 = DY23 << 4;
	const int64_t FDY31 = DY31 << 4;

	// Bounding rectangle
	int minx = (int)((std::min(std::min(X1, X2), X3) + 0xF) >> 4);
	int maxx = (int)((std::max(std::max(X1, X2), X3) + 0xF) >> 4);
	int miny = (int)((std::min(std::min(Y1, Y2), Y3) + 0xF) >> 4);
	int maxy = (int)((std::max(std::max(Y1, Y2), Y3) + 0xF) >> 4);

	// Block size, standard 8x8 (must be power of two)
	const int q = 8;

	// Start in corner of 8x8 block
	minx &= ~(q - 1);
	miny &= ~(q - 1);

	// Half-edge constants
	int64_t C1 = DY12 * X1 - DX12 * Y1;
	int64_t C2 = DY23 * X2 - DX23 * Y2;
	int64_t C3 = DY31 * X3 - DX31 * Y3;

	// Correct for fill convention
	if (DY12 < 0 || (DY12 == 0 && DX12 > 0)) C1++;
	if (DY23 < 0 || (DY23 == 0 && DX23 > 0)) C2++;
	if (DY31 < 0 || (DY31 == 0 && DX31 > 0)) C3++;

	// Loop through blocks
	int pixy = miny;
	for (int y = miny; y < maxy; y += q)
	{
		for (int x = minx; x < maxx; x += q)
		{
			// Corners of block
			const int64_t x0 = int64_t(x) << 4;
			const int64_t x1 = int64_t(x + q - 1) << 4;
			const int64_t y0 = int64_t(y) << 4;
			const int64_t y1 = int64_t(y + q - 1) << 4;

			// Evaluate half-space functions
			const bool a00 = C1 + DX12 * y0 - DY12 * x0 > 0;
			const bool a10 = C1 + DX12 * y0 - DY12 * x1 > 0;
			const bool a01 = C1 + DX12 * y1 - DY12 * x0 > 0;
			const bool a11 = C1 + DX12 * y1 - DY12 * x1 > 0;
			const int a = (a00 << 0) | (a10 << 1) | (a01 << 2) | (a11 << 3);

			const bool b00 = C2 + DX23 * y0 - DY23 * x0 > 0;
			const bool b10 = C2 + DX23 * y0 - DY23 * x1 > 0;
			const bool b01 = C2 + DX23 * y1 - DY23 * x0 > 0;
			const bool b11 = C2 + DX23 * y1 - DY23 * x1 > 0;
			const int b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3);

			const bool c00 = C3 + DX31 * y0 - DY31 * x0 > 0;
			const bool c10 = C3 + DX31 * y0 - DY31 * x1 > 0;
			const bool c01 = C3 + DX31 * y1 - DY31 * x0 > 0;
			const bool c11 = C3 + DX31 * y1 - DY31 * x1 > 0;
			const int c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3);

			// Skip block when outside an edge
			if (a == 0x0 || b == 0x0 || c == 0x0) continue;

			int nowpixy = pixy;

			// Accept whole block when totally covered
			if (a == 0xF && b == 0xF && c == 0xF)
			{
				for (int iy = 0; iy < q; iy++)
				{
					for (int ix = x; ix < x + q; ix++){
                        // Eigen::Vector3i ixy;
                        // ixy << ix, nowpixy, 1;
						// parser(ixy);
                        parser(nowpixy, ix);
                    }

					++nowpixy;
				}
			}
			else // Partially covered block
			{
				int64_t CY1 = C1 + DX12 * y0 - DY12 * x0;
				int64_t CY2 = C2 + DX23 * y0 - DY23 * x0;
				int64_t CY3 = C3 + DX31 * y0 - DY31 * x0;

				for (int iy = y; iy < y + q; iy++)
				{
					int64_t CX1 = CY1;
					int64_t CX2 = CY2;
					int64_t CX3 = CY3;

					for (int ix = x; ix < x + q; ix++)
					{
						if (CX1 > 0 && CX2 > 0 && CX3 > 0){
                            // Eigen::Vector3i ixy;
                            // ixy << ix, nowpixy, 1;
                            // parser(ixy);
                            parser(nowpixy, ix);
                        }

						CX1 -= FDY12;
						CX2 -= FDY23;
						CX3 -= FDY31;
					}

					CY1 += FDX12;
					CY2 += FDX23;
					CY3 += FDX31;

					++nowpixy;
				}
			}
		}

		pixy += q;
	}
}

void InitDepthMapFromDelaunay(Problem& problem, 
                              std::vector<Eigen::Vector3f>& points,
                              const Image& image) {
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K(image.GetK());
    Eigen::Matrix3f Kinv = K.inverse();

    DepthMap& depth_map = problem.depth_maps->at(problem.ref_image_idx);
    NormalMap& normal_map = problem.normal_maps->at(problem.ref_image_idx);

    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();

    CGAL::Delaunay delaunay;

    for (const auto& point : points) {
        delaunay.insert(CGAL::Point(point.x(), point.y(), point.z()));
    }

    std::cout << "RasterDepthDataPlaneData" << std::endl;
    RasterDepthDataPlaneData data = { Kinv, depth_map, normal_map };

    std::cout << "Init Depth from triangulation" << std::endl;
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

        data.normal = edge2.cross(edge1).normalized();
		data.normalPlane = data.normal * (1.0f / data.normal.dot(c0));

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

        // // RasterizeTriangle(Eigen::Vector3f(i2[0], i2[1], i2[2]), 
        // //                   Eigen::Vector3f(i1[0], i1[1], i1[2]), 
        // //                   Eigen::Vector3f(i0[0], i0[1], i0[2]), data);
        // for (int v = v_min; v <= v_max; ++v) {
        //     for (int u = u_min; u <= u_max; ++u) {
        //         int t1 = (u - i0[0]) * (v - i1[1]) - (u - i1[0]) * (v - i0[1]);
        //         int t2 = (u - i1[0]) * (v - i2[1]) - (u - i2[0]) * (v - i1[1]);
        //         int t3 = (u - i2[0]) * (v - i0[1]) - (u - i0[0]) * (v - i2[1]);
        //         if ((t1 >= 0 && t2 >= 0 && t3 >= 0) ||
        //             (t1 < 0 && t2 < 0 && t3 < 0)) {
        //             data(v, u);
        //         }
        //     }
        // }
    }

}

void InitDepthMapFromSparsePoint(Problem& problem, 
                                 std::vector<Eigen::Vector3f>& points,
                                 const Image& image) {
    std::cout << "InitDepthMapFromSparsePoint" << std::endl;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K(image.GetK());
    Eigen::Matrix3f Kinv = K.inverse();

    DepthMap& depth_map = problem.depth_maps->at(problem.ref_image_idx);
    NormalMap& normal_map = problem.normal_maps->at(problem.ref_image_idx);

    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();
    const int pixel_area = 3;

    for (const auto & point : points) {
        float z = point.z();
        int u = point.x();
        int v = point.y();
        int u_min = std::max(u - pixel_area, 0);
        int u_max = std::min(u + pixel_area, width - 1);
        int v_min = std::max(v - pixel_area, 0);
        int v_max = std::min(v + pixel_area, height - 1);
        for (int y = v_min; y <= v_max; ++y) {
            for (int x = u_min; x <= u_max; ++x) {
                float d = depth_map.Get(y, x);
                if (d == 0 || d > z) {
                    depth_map.Set(y, x, z);
                    normal_map.Set(y, x, 0, 0);
                    normal_map.Set(y, x, 1, 0);
                    normal_map.Set(y, x, 2, 0);
                }
            }
        }
    }    
}

std::pair<float, float> 
Model2Depth(const TriangleMesh& model, const Image& image, Problem& problem) {
    std::chrono::high_resolution_clock::time_point start_time = 
        std::chrono::high_resolution_clock::now();

    DepthMap& depth_map = problem.depth_maps->at(problem.ref_image_idx);
    NormalMap& normal_map = problem.normal_maps->at(problem.ref_image_idx);
    
    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();

    const Eigen::RowMatrix3d K = Eigen::RowMatrix3f(image.GetK()).cast<double>();
    const Eigen::RowMatrix3d R = Eigen::RowMatrix3f(image.GetR()).cast<double>();
    const Eigen::Vector3d T = Eigen::Vector3f(image.GetT()).cast<double>();

    // const double fovx = std::atan(width * 0.5 / K(0, 0));
    // const double fovy = std::atan(height * 0.5 / K(1, 1));
    // double thresh_fov = std::min(std::cos(fovx), std::cos(fovy)) * 0.8;
    // thresh_fov = std::max(thresh_fov, 0.0);
    const double thresh_fov = std::cos(90.0);

    const Eigen::Vector3d C = -R.transpose() * T;
    const Eigen::Vector3d ray = R.row(2);
    const Eigen::RowMatrix3d Kinv = K.inverse();

    std::pair<float, float> depth_range;
    depth_range.first = std::numeric_limits<float>::max();
    depth_range.second = std::numeric_limits<float>::epsilon();

    for (size_t i = 0; i < model.faces_.size(); ++i) {
        const Eigen::Vector3i& facet = model.faces_[i];
        const Eigen::Vector3d p0 = model.vertices_[facet[0]];
        const Eigen::Vector3d p1 = model.vertices_[facet[1]];
        const Eigen::Vector3d p2 = model.vertices_[facet[2]];
        const Eigen::Vector3d centroid = (p0 + p1 + p2) / 3;
        
        if ((centroid - C).dot(model.face_normals_[i]) > 0) {
            continue;
        }

        const double dev_angle = ray.dot((centroid - C).normalized());
        // Outside of FOV.
        if (dev_angle < thresh_fov) {
            continue;
        }

        Eigen::Vector3d uv[3];
        Eigen::Vector3d Xc[3];

        Xc[0] = R * p0 + T;
        uv[0] = K * Xc[0]; uv[0] /= uv[0][2];
        Xc[1] = R * p1 + T;
        uv[1] = K * Xc[1]; uv[1] /= uv[1][2];
        Xc[2] = R * p2 + T;
        uv[2] = K * Xc[2]; uv[2] /= uv[2][2];
        
        int u_min = std::min(uv[0][0], std::min(uv[1][0], uv[2][0]));
        int u_max = std::max(uv[0][0], std::max(uv[1][0], uv[2][0]));
        int v_min = std::min(uv[0][1], std::min(uv[1][1], uv[2][1]));
        int v_max = std::max(uv[0][1], std::max(uv[1][1], uv[2][1]));
        u_min = std::max(0, u_min);
        v_min = std::max(0, v_min);
        u_max = std::min(width - 1, u_max);
        v_max = std::min(height - 1, v_max);
        if (Xc[0][2] <= 0 || Xc[1][2] <= 0 || Xc[2][2] <= 0 ||
            u_min > u_max || v_min > v_max) {
            continue;
        }

        Eigen::Vector3d edge1(Xc[1] - Xc[0]);
        Eigen::Vector3d edge2(Xc[2] - Xc[0]);

        const Eigen::Vector3d normal = edge1.cross(edge2).normalized();
		const Eigen::Vector3d normalPlane = normal * (1.0 / normal.dot(Xc[0]));

        Eigen::Vector2d v1 = (uv[1] - uv[0]).head<2>();
        Eigen::Vector2d v2 = (uv[2] - uv[1]).head<2>();
        Eigen::Vector2d v3 = (uv[0] - uv[2]).head<2>();

        for (int v = v_min; v <= v_max; ++v) {
            for (int u = u_min; u <= u_max; ++u) {
                double m1 = v1.x() * (v - uv[0][1]) - (u - uv[0][0]) * v1.y();
                double m2 = v2.x() * (v - uv[1][1]) - (u - uv[1][0]) * v2.y();
                double m3 = v3.x() * (v - uv[2][1]) - (u - uv[2][0]) * v3.y();
                if (m1 < 0 && m2 < 0 && m3 < 0) {
                    Eigen::Vector3d xc = Kinv * Eigen::Vector3d(u, v, 1.0);
                    const double z = 1.0 / normalPlane.dot(xc);
                    const float nd = depth_map.Get(v, u);
                    if (nd == 0 || z < nd) {
                        depth_map.Set(v, u, z);
                        normal_map.Set(v, u, 0, normal[0]);
                        normal_map.Set(v, u, 1, normal[1]);
                        normal_map.Set(v, u, 2, normal[2]);

                        depth_range.first = 
                            std::min(depth_range.first, (float)z);
                        depth_range.second = 
                            std::max(depth_range.second, (float)z);
                    }
                }
            }
        }
    }
    std::chrono::high_resolution_clock::time_point end_time = 
        std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                    end_time - start_time).count() / 1e6;
    std::cout << StringPrintf("Model2Depth: %lf [seconds]", duration) 
                << std::endl;
    
    if (depth_range.first == std::numeric_limits<float>::max()) {
        depth_range.first = 0.0f;
        depth_range.second = 100.0f;
    }
    depth_range.first *= 0.9f;
    depth_range.second *= 1.1f;

    // depth_range.first = std::max(depth_range.first, 0.000001f);
    depth_range.first = 0.000001f;

    return depth_range;
}

std::pair<float, float> 
Ply2Depth(const std::vector<PlyPoint>& ply_points, const Image& image, Problem& problem) {
    std::chrono::high_resolution_clock::time_point start_time = 
        std::chrono::high_resolution_clock::now();

    DepthMap& depth_map = problem.depth_maps->at(problem.ref_image_idx);
    NormalMap& normal_map = problem.normal_maps->at(problem.ref_image_idx);
    
    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();

    const Eigen::RowMatrix3d K = Eigen::RowMatrix3f(image.GetK()).cast<double>();
    const Eigen::RowMatrix3d R = Eigen::RowMatrix3f(image.GetR()).cast<double>();
    const Eigen::Vector3d T = Eigen::Vector3f(image.GetT()).cast<double>();

    // const double fovx = std::atan(width * 0.5 / K(0, 0));
    // const double fovy = std::atan(height * 0.5 / K(1, 1));
    // double thresh_fov = std::min(std::cos(fovx), std::cos(fovy)) * 0.8;
    // thresh_fov = std::max(thresh_fov, 0.0);
    const double thresh_fov = std::cos(90.0);

    const Eigen::Vector3d C = -R.transpose() * T;
    const Eigen::Vector3d ray = R.row(2);
    const Eigen::RowMatrix3d Kinv = K.inverse();

    std::pair<float, float> depth_range;
    depth_range.first = std::numeric_limits<float>::max();
    depth_range.second = std::numeric_limits<float>::epsilon();

    for (size_t i = 0; i < ply_points.size(); ++i) {
        const auto pnt = ply_points[i];
        const Eigen::Vector3d centroid(pnt.x, pnt.y, pnt.z);
        const Eigen::Vector3d normal(pnt.nx, pnt.ny, pnt.nz);
        const Eigen::Vector3d image_normal(R * normal);
        
        if ((centroid - C).dot(normal) > 0) {
            continue;
        }

        // const double dev_angle = ray.dot((centroid - C).normalized());
        // // Outside of FOV.
        // if (dev_angle < thresh_fov) {
        //     continue;
        // }

        Eigen::Vector3d uv;
        Eigen::Vector3d Xc;

        Xc = R * centroid + T;
        uv = K * Xc; 
        uv[0] /= uv[2];
        uv[1] /= uv[2];

		const Eigen::Vector3d normalPlane = normal * (1.0 / normal.dot(Xc));
        
        if (Xc[2] <= 0 || uv[0] < 0 || uv[1] < 0 ||
            uv[0] > width - 1 || uv[1] > height - 1) {
            continue;
        }

        int windows_size = std::pow(2, PLY_INIT_DEPTH_LEVEL);
        for (int i = std::floor(uv[0]) - windows_size; i <= std::ceil(uv[0]) + windows_size; i++){
            for (int j = std::floor(uv[1]) - windows_size; j <= std::ceil(uv[1]) + windows_size; j++ ){
                if (i < 0 || i >  width - 1 || j < 0 || j > height - 1){
                    continue;
                }
                if (i % 2 == 0 || j % 2 == 0){
                    continue;
                }

                Eigen::Vector3d xc = Kinv * Eigen::Vector3d(i, j, 1.0);
                // const double z = 1.0 / normalPlane.dot(xc);
                const double z = Xc[2];
                const float nd = depth_map.Get(j, i);
                if (nd == 0 || z < nd) {
                    depth_map.Set(j, i, z);
                    normal_map.Set(j, i, 0, image_normal[0]);
                    normal_map.Set(j, i, 1, image_normal[1]);
                    normal_map.Set(j, i, 2, image_normal[2]);

                    depth_range.first = 
                        std::min(depth_range.first, (float)z);
                    depth_range.second = 
                        std::max(depth_range.second, (float)z);
                }
            }
        }
    }
    std::chrono::high_resolution_clock::time_point end_time = 
        std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                    end_time - start_time).count() / 1e6;
    std::cout << StringPrintf("Ply2Depth: %lf [seconds]", duration) 
                << std::endl;
    
    if (depth_range.first == std::numeric_limits<float>::max()) {
        depth_range.first = 0.0f;
        depth_range.second = 100.0f;
    }
    depth_range.first *= 0.9f;
    depth_range.second *= 1.1f;

    // depth_range.first = std::max(depth_range.first, 0.000001f);
    depth_range.first = 0.000001f;

    return depth_range;
}

void FillDepthMap(const PatchMatchOptions options,
                  const std::string& workspace_path,
                  const Problem& problem,
                  const mvs::Image& image,
                  const std::string& image_name,
                  const int level,
                  const bool fill_flag,
                  std::string input_type){
    const auto& depth_maps_path = JoinPaths(workspace_path, DEPTHS_DIR);
    const auto& normal_maps_path = JoinPaths(workspace_path, NORMALS_DIR);
    const auto& conf_maps_path = JoinPaths(workspace_path, CONFS_DIR);

    std::string rescale_input_type = (input_type == GEOMETRIC_TYPE ? 
        PHOTOMETRIC_TYPE : GEOMETRIC_TYPE);
    const std::string file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), rescale_input_type.c_str(), DEPTH_EXT);
    // std::cout << "fill depth name: " << file_name << std::endl;
    const std::string depth_map_path = JoinPaths(depth_maps_path, file_name);
    const std::string normal_map_path = JoinPaths(normal_maps_path, file_name);
    const std::string conf_map_path = JoinPaths(conf_maps_path, file_name);

    DepthMap depth_map;
    NormalMap normal_map;
#ifdef SAVE_CONF_MAP
    Mat<float> conf_map;
#endif
    depth_map.Read(depth_map_path);
    normal_map.Read(normal_map_path);
#ifdef SAVE_CONF_MAP
    conf_map.Read(conf_map_path); 
#endif

    // pyramid factor
    float factor_level = 1 / std::pow(2, level);
    float max_size = std::max(image.GetHeight(), image.GetWidth());
    float min_image_size = std::min(320.f, max_size);
    factor_level = std::max(factor_level, min_image_size / max_size);

    // rescale
    // image.Rescale(factor_level);
    int out_height = std::round(image.GetHeight() * factor_level);
    int out_width = std::round(image.GetWidth() * factor_level);
    const float in_height = depth_map.GetHeight();
    const float in_width = depth_map.GetWidth();
    const float scale_height = out_height / in_height;
    const float scale_width = out_width / in_width;
    if (scale_height != 1){
        depth_map.Rescale(scale_width, scale_height);
        normal_map.Rescale(scale_width, scale_height);
#ifdef SAVE_CONF_MAP
        DepthMap temp_conf_map(conf_map, -1.0f, -1.0f);
        temp_conf_map.Rescale(scale_width, scale_height);
        conf_map = Mat<float>(scale_width, scale_height, 1);
        conf_map.Set(temp_conf_map.GetData());
#endif
    } 

    const std::string out_file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), input_type.c_str(), DEPTH_EXT);
    // std::cout << "out file name: " << out_file_name << std::endl;
    const std::string out_depth_map_path = 
        JoinPaths(depth_maps_path, out_file_name);
    const std::string out_normal_map_path = 
        JoinPaths(normal_maps_path, out_file_name);
    const std::string out_conf_map_path = 
        JoinPaths(conf_maps_path, out_file_name);
    // read ref depth map and normal map 
    if (fill_flag) {
        DepthMap ref_depth_map;
        ref_depth_map.Read(out_depth_map_path);
        NormalMap ref_normal_map;
        ref_normal_map.Read(out_normal_map_path);
        
        const int width = depth_map.GetWidth();
        const int height = depth_map.GetHeight();
        const int ref_width = ref_depth_map.GetWidth();
        const int ref_height = ref_depth_map.GetHeight();

        float scale_height = (float)ref_height / height;
        float scale_width = (float)ref_width / width;

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int u = col * scale_width + 0.5;
                int v = row * scale_height + 0.5;
                if (u < 0 || u >= width || v < 0 || v >= height) {
                    continue;
                }
                float depth_ref = ref_depth_map.Get(v, u);
                if (depth_ref < 0.00001) {
                    continue;
                }

                depth_map.Set(row, col, depth_ref);
                float normal_ref[3];
                ref_normal_map.GetSlice(v, u, normal_ref);
                normal_map.SetSlice(row, col, normal_ref);
            }
        }  
    }

    depth_map.Write(out_depth_map_path);
    normal_map.Write(out_normal_map_path);
#ifdef SAVE_CONF_MAP
    conf_map.Write(out_conf_map_path);
#endif
    if (options.verbose) {
        depth_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", (out_depth_map_path + "-up").c_str(), level));
        normal_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", (out_normal_map_path + "-up").c_str(), level));
    }
}

void MutiThreadSaveCrossPointCloud(
    const std::string ply_path, const bool has_sem,
    const std::vector<PlyPoint>& points,
    const  std::vector<std::vector<uint32_t> >& points_visibility,
    const std::vector<std::vector<float> >& points_vis_weight = std::vector<std::vector<float>>(),
    const std::vector<float>& points_score = std::vector<float>()){

    int num_eff_threads = std::min(GetEffectiveNumThreads(-1), 6);
    std::unique_ptr<ThreadPool> wirte_thread_pool;
    wirte_thread_pool.reset(new ThreadPool(num_eff_threads));
    wirte_thread_pool->AddTask([&](){
        WriteBinaryPlyPoints(ply_path, points, true, true);
    });
    wirte_thread_pool->AddTask([&](){
        WritePointsVisibility(ply_path + ".vis", points_visibility);
    });
    if (has_sem) {
        wirte_thread_pool->AddTask([&](){
            WritePointsSemantic(ply_path + ".sem", points, false);
        });
        wirte_thread_pool->AddTask([&](){
            WritePointsSemanticColor(
                JoinPaths(GetParentDir(ply_path), FUSION_SEM_NAME), 
                points);
        });
    }
    if (points_score.size() == points.size() ){
        wirte_thread_pool->AddTask([&](){
            WritePointsScore(ply_path + ".sco", points_score );
        });
    }
    if (points_vis_weight.size() == points.size()){
        wirte_thread_pool->AddTask([&](){
            WritePointsWeight(ply_path + ".wgt", points_vis_weight);
        });
    }
    wirte_thread_pool->Wait();

    std::cout << "MutiThreadSaveCrossPointCloud(" << points.size() 
                    << ", " << has_sem << ", " << ply_path  << " ) ... Done" << std::endl;
    return;
}

void MutiThreadAppendCrossPointCloud(
    const std::string ply_path, const bool has_sem,
    const std::vector<PlyPoint>& points,
    const  std::vector<std::vector<uint32_t> >& points_visibility,
    const std::vector<std::vector<float> >& points_vis_weight = std::vector<std::vector<float>>(),
    const std::vector<float> points_score = std::vector<float>()){
    int num_eff_threads = std::min(GetEffectiveNumThreads(-1), 6);

    std::unique_ptr<ThreadPool> append_thread_pool;
    append_thread_pool.reset(new ThreadPool(num_eff_threads));
    append_thread_pool->AddTask([&](){
        AppendWriteBinaryPlyPoints(ply_path, points);
    });
    append_thread_pool->AddTask([&](){
        AppendWritePointsVisibility(ply_path + ".vis", points_visibility);
    });
    if (has_sem) {
        append_thread_pool->AddTask([&](){
            AppendWritePointsSemantic(ply_path + ".sem", points, false);
        });
        append_thread_pool->AddTask([&](){
            AppendWritePointsSemanticColor(
                JoinPaths(GetParentDir(ply_path), FUSION_SEM_NAME), 
                points);
        });
    }
    if (points_score.size() == points.size() ){
        append_thread_pool->AddTask([&](){
            AppendWWritePointsScore(ply_path + ".sco", points_score );
        });
    }
    if (points_vis_weight.size() == points.size()){
        append_thread_pool->AddTask([&](){
            AppendWritePointsWeight(ply_path + ".wgt", points_vis_weight);
        });
    }
    append_thread_pool->Wait();

    std::cout << "MutiThreadAppendCrossPointCloud (" << points.size() 
                    << ", " << has_sem << ", " << ply_path  << " ) ... Done" << std::endl;
    return;
}
}

PatchMatchController::PatchMatchController(const PatchMatchOptions& options,
			                               const std::string& workspace_path,
                                           const std::string& lidar_path,
                                           const int reconstrction_idx,
                                           const int cluster_idx)
    : options_(options),
      workspace_path_(workspace_path),
      lidar_path_(lidar_path),
      num_cluster_(0),
      select_reconstruction_idx_(reconstrction_idx),
      select_cluster_idx_(cluster_idx){
	CHECK(options_.Check());
}

void PatchMatchController::ReadGpuIndices() {
    // gpu_indices_ = CSVToVector<int>(options_.gpu_index);
    // if (gpu_indices_.size() == 1 && gpu_indices_[0] == -1) {
    //     const int num_cuda_devices = GetNumCudaDevices();
    //     CHECK_GT(num_cuda_devices, 0);
    //     gpu_indices_.resize(num_cuda_devices * THREADS_PER_GPU);
    //     for(int i = 0; i < num_cuda_devices * THREADS_PER_GPU; ++i) {
    //         gpu_indices_[i] = i / THREADS_PER_GPU;
    //     }
    // } else {
    //     const int num_cuda_devices = gpu_indices_.size();
    //     std::vector<int> gpu_indices_tmp(num_cuda_devices * THREADS_PER_GPU);
    //     for (int i = 0; i < num_cuda_devices; ++i) {
    //         for (int j = 0; j < THREADS_PER_GPU; ++j) {
    //             gpu_indices_tmp[i * THREADS_PER_GPU + j] = gpu_indices_[i];
    //         }
    //     }
    //     gpu_indices_ = gpu_indices_tmp;
    // }

    size_t num_threads = 0;
    for (int i = 0; i < threads_per_gpu_.size(); ++i) {
        num_threads += threads_per_gpu_[i];
    }
    gpu_indices_.resize(num_threads);
    int thread_id_map = 0;
    for (int i = 0; i < threads_per_gpu_.size(); ++i) {
        for (int j = 0; j < threads_per_gpu_[i]; ++j) {
            gpu_indices_[thread_id_map++] = i;
        }
    }
}

void PatchMatchController::ReadCrossFilterGpuIndices() {
    cross_filter_gpu_indices_ = CSVToVector<int>(options_.gpu_index);
    if (cross_filter_gpu_indices_.size() == 1 && cross_filter_gpu_indices_[0] == -1) {
        const int num_cuda_devices = GetNumCudaDevices();
        CHECK_GT(num_cuda_devices, 0);
        cross_filter_gpu_indices_.resize(num_cuda_devices);
        for(int i = 0; i < num_cuda_devices; ++i) {
            cross_filter_gpu_indices_[i] = i;
        }
    }
}

void PatchMatchController::GetGpuProp() {
    auto gpu_indices = CSVToVector<int>(options_.gpu_index);
    if (gpu_indices.size() == 1 && gpu_indices[0] == -1) { 
        const int num_cuda_devices = GetNumCudaDevices();
        CHECK_GT(num_cuda_devices, 0);
        gpu_indices.resize(num_cuda_devices);
        for(int i = 0; i < num_cuda_devices; ++i) {
            gpu_indices[i] = i;
        }
    }
    const float G_byte = 1024 * 1024 * 1024;

    int max_gpu_index = 0;
    for (int i = 0; i < gpu_indices.size(); ++i) {
        max_gpu_index = std::max(max_gpu_index, gpu_indices.at(i));
    }

    max_gpu_memory_array_.resize(max_gpu_index + 1, 0);
    max_gpu_cudacore_array_.resize(max_gpu_index + 1, 0);
    max_gpu_texture_layered_0_.resize(max_gpu_index + 1, 0);
    max_gpu_texture_layered_1_.resize(max_gpu_index + 1, 0);
    max_gpu_texture_layered_2_.resize(max_gpu_index + 1, 0);
    for (int i = 0; i < gpu_indices.size(); i++){
        int idx = gpu_indices[i];
        cudaDeviceProp device;
        GetDeviceProp(idx, device);

        size_t gpu_free_size, gpu_total_size;
        CUDA_SAFE_CALL(cudaMemGetInfo(&gpu_free_size, &gpu_total_size));
        printf("Avail/Total Gpu Memory %f/%f\n", gpu_free_size / G_byte, gpu_total_size / G_byte);
        
        float available_gpu_memory = gpu_free_size / G_byte;
        if (options_.max_gpu_memory < 0 || options_.max_gpu_memory > available_gpu_memory) {
            max_gpu_memory_array_[idx] = available_gpu_memory;
        } else {
            max_gpu_memory_array_[idx] = options_.max_gpu_memory;
        }
        max_gpu_cudacore_array_[idx] = ConvertSMVer2Cores(device.major, device.minor) * device.multiProcessorCount;
        max_gpu_texture_layered_0_[idx] = device.maxTexture2DLayered[0];
        max_gpu_texture_layered_1_[idx] = device.maxTexture2DLayered[1];
        max_gpu_texture_layered_2_[idx] = device.maxTexture2DLayered[2];
        
        if (options_.cuda_maxTexture1DLayered_0 > device.maxTexture2DLayered[0]){
            options_.cuda_maxTexture1DLayered_0 = device.maxTexture2DLayered[0];
        }
        if (options_.cuda_maxTexture1DLayered_1 > device.maxTexture2DLayered[1]){
            options_.cuda_maxTexture1DLayered_1 = device.maxTexture2DLayered[1];
        }
        if (options_.cuda_maxTexture1DLayered_2 > device.maxTexture2DLayered[2]){
            options_.cuda_maxTexture1DLayered_2 = device.maxTexture2DLayered[2];
        }

        printf("Option: GPU#%d\n", idx);
        printf("        Global Memory = %f(G)\n", max_gpu_memory_array_[idx]);
        printf("        Max Layered Texture 2D Size =(%d,%d) x %d\n",
            max_gpu_texture_layered_0_[idx], max_gpu_texture_layered_1_[idx], max_gpu_texture_layered_2_[idx]);
        printf("        Stream Processor Size = %d\n", device.multiProcessorCount);
        printf("        CUDA Cores = %d\n", max_gpu_cudacore_array_[idx]);
    }
}

void PatchMatchController::EstimateThreadsPerGPU() {
    std::cout << "EstimateThreadsPerGPU " << std::endl;
    const float G_byte = 1024 * 1024 * 1024;

    const auto& dense_reconstruction_path = JoinPaths(workspace_path_, std::to_string(select_reconstruction_idx_), 
                                                      DENSE_DIR, SPARSE_DIR);
    std::cout << dense_reconstruction_path << std::endl;
    Reconstruction reconstruction;
    reconstruction.ReadBinary(dense_reconstruction_path);

    size_t max_width = 0, max_height = 0;
    for (auto image_id : reconstruction.RegisterImageIds()) {
        class sensemap::Image image = reconstruction.Image(image_id);
        class sensemap::Camera camera = reconstruction.Camera(image.CameraId());
        max_width = std::max(max_width, camera.Width());
        max_height = std::max(max_height, camera.Height());
    }

    float total_byte = (72 + 5 * options_.max_num_src_images);
    if (options_.geom_consistency) {
        total_byte += 4 * options_.max_num_src_images;
    }
    if (options_.refine_with_semantic) {
        total_byte += (1 + options_.max_num_src_images);
    }
    if (options_.has_prior_depth) {
        total_byte += 16;
    }
    if (options_.median_filter) {
        total_byte += 4;
    }
    if (options_.geom_consistency && options_.est_curvature) {
        total_byte += 4;
    }
    float used_gpu_memory = 1.4 * (max_width * max_height / G_byte) * total_byte;

    printf("Used GPU Memory: %f\n", used_gpu_memory);
    
    int num_tasks = 0;
    auto gpu_indices = CSVToVector<int>(options_.gpu_index);
    std::cout << "gpu_index: " << options_.gpu_index << " " << gpu_indices.size() << " " << gpu_indices[0] << std::endl;
    if (gpu_indices.size() == 1 && gpu_indices[0] == -1) {
        const int num_cuda_devices = GetNumCudaDevices();
        CHECK_GT(num_cuda_devices, 0);
        threads_per_gpu_.resize(num_cuda_devices, 0);
        for (int gpu_idx = 0; gpu_idx < num_cuda_devices; ++gpu_idx) {
            threads_per_gpu_[gpu_idx] = std::min(int(max_gpu_memory_array_[gpu_idx] / used_gpu_memory), MAX_THREADS_PER_GPU);
            num_tasks += threads_per_gpu_[gpu_idx];
            printf("GPU#%d: %d threads for patch match\n", gpu_idx, threads_per_gpu_[gpu_idx]);
        }
    } else {
        int max_gpu_index = 0;
        for (auto gpu_idx : gpu_indices) {
            max_gpu_index = std::max(gpu_idx, max_gpu_index);
        }
        threads_per_gpu_.resize(max_gpu_index + 1, 0);
        for (auto gpu_idx : gpu_indices) {
            threads_per_gpu_[gpu_idx] = std::min(int(max_gpu_memory_array_[gpu_idx] / used_gpu_memory), MAX_THREADS_PER_GPU);
            num_tasks += threads_per_gpu_[gpu_idx];
            printf("GPU#%d: %d threads for patch match\n", gpu_idx, threads_per_gpu_[gpu_idx]);
        }
    }
    if (num_tasks == 0) {
        ExceptionHandler(LIMITED_GPU_MEMORY, 
            JoinPaths(GetParentDir(workspace_path_), "errors/dense"), "DensePatchMatch").Dump();
        exit(LIMITED_GPU_MEMORY);
    }
}

int PatchMatchController::ReadClusterBox() {
    num_box_ = -1;
    
    const auto reconstruction_path = 
        JoinPaths(workspace_path_, std::to_string(select_reconstruction_idx_));
    const auto box_path = 
        JoinPaths(reconstruction_path, DENSE_DIR, BOX_YAML);
    
    if (ExistsFile(box_path)) {
        YAML::Node box_node = YAML::LoadFile(box_path);
        num_box_ = box_node["num_clusters"].as<int>();

        int vec_id = 0;
        for (YAML::const_iterator it= box_node["transformation"].begin(); 
            it != box_node["transformation"].end();++it){
            box_rot_(vec_id / 3, vec_id %3) = it->as<float>();
            vec_id++;
        }
        if (vec_id != 9){
            std::cout << "Yaml transformation is bad!" << std::endl;
        }
        // std::cout << box_rot_ << std::endl;

        roi_box_.x_min = box_node["box"]["x_min"].as<float>();
        roi_box_.y_min = box_node["box"]["y_min"].as<float>();
        roi_box_.z_min = box_node["box"]["z_min"].as<float>();
        roi_box_.x_max = box_node["box"]["x_max"].as<float>();
        roi_box_.y_max = box_node["box"]["y_max"].as<float>();
        roi_box_.z_max = box_node["box"]["z_max"].as<float>();
        roi_box_.SetBoundary(options_.roi_box_width * 1.2, options_.roi_box_factor * 1.2);
        roi_box_.z_box_min = -FLT_MAX;
        roi_box_.z_box_max = FLT_MAX;
        roi_box_.rot = box_rot_;
        std::cout << StringPrintf("ROI(box.yaml): [%f %f %f] -> [%f %f %f]\n", 
                    roi_box_.x_box_min, roi_box_.y_box_min, roi_box_.z_box_min,
                    roi_box_.x_box_max, roi_box_.y_box_max, roi_box_.z_box_max);

        for (int idx = 0; idx < num_box_; idx++){
            Box box;
            box.x_min = box_node[std::to_string(idx)]["x_min"].as<float>();
            box.y_min = box_node[std::to_string(idx)]["y_min"].as<float>();
            box.z_min = box_node[std::to_string(idx)]["z_min"].as<float>();
            // box.z_min = -FLT_MAX;
            box.x_max = box_node[std::to_string(idx)]["x_max"].as<float>();
            box.y_max = box_node[std::to_string(idx)]["y_max"].as<float>();
            box.z_max = box_node[std::to_string(idx)]["z_max"].as<float>();
            // box.z_max = FLT_MAX;
            if (options_.roi_box_width < 0 && options_.roi_box_factor < 0){
                options_.roi_box_factor = 0.01;
            }
            box.SetBoundary(options_.roi_box_width * 2, options_.roi_box_factor * 2);
            box.z_box_min = -FLT_MAX;
            box.z_box_max = FLT_MAX;
            box.rot = box_rot_;
            roi_child_boxs_.push_back(box);
            std::cout << StringPrintf("ROI - %d(box.yaml): [%f %f %f] -> [%f %f %f]\n", 
                    idx, box.x_box_min, box.y_box_min, box.z_box_min,
                    box.x_box_max, box.y_box_max, box.z_box_max);
        }
    }

    std::cout << "Reading Box Yaml (" << num_box_ << " cluster)..." << std::endl;
    return num_box_;
}

std::pair<float, float> 
PatchMatchController::InitDepthMap(const PatchMatchOptions& options,
                                   const size_t problem_idx,
                                   const Image& image,
                                   const std::string& dense_path,
                                   const std::string& workspace_path) {
    const int width = image.GetWidth();
    const int height = image.GetHeight();

    const auto& model = workspace_->GetModel();
    auto& problem = problems_.at(problem_idx);

    const std::string output_type =
        options.geom_consistency ? GEOMETRIC_TYPE : PHOTOMETRIC_TYPE;
    const std::string image_name = model.GetImageName(problem.ref_image_idx);
    const std::string file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), output_type.c_str(), DEPTH_EXT);
    const std::string depth_map_path =
        JoinPaths(workspace_path, DEPTHS_DIR, file_name);
    const std::string normal_map_path =
        JoinPaths(workspace_path, NORMALS_DIR, file_name);

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K(image.GetK());
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R(image.GetR());
    Eigen::Vector3f T(image.GetT());

    std::pair<float, float> depth_range;
    depth_range.first = std::numeric_limits<float>::max();
    depth_range.second = std::numeric_limits<float>::epsilon();

    std::vector<unsigned char> mask(width * height, 0);
    std::vector<Eigen::Vector3f> uvd;
    bool has_lidar = ExistsDir(lidar_path_);

    for (const auto& point : model.points) {
        for (const auto& image_idx : point.track) {
            if (image_idx == problem.ref_image_idx) {
                Eigen::Vector3f X(point.x, point.y, point.z);
                const Eigen::Vector3f Xc = K * (R * X + T);
                if (Xc[2] <= 0) {
                    continue;
                }
                int fu = Xc[0] / Xc[2];
                int fv = Xc[1] / Xc[2];
                if (fu < 0 || fu >= width || fv < 0 || fv >= height) {
                    continue;
                }
                uvd.emplace_back(fu, fv, Xc[2]);
                mask[fv * width + fu] = 1;
                if (!has_lidar) {
                    depth_range.first = std::min(depth_range.first, Xc[2]);
                    depth_range.second = std::max(depth_range.second, Xc[2]);
                }
            }
        }
    }

    if (options_.init_from_rgbd) {
        const std::string rgbd_path =
        JoinPaths(dense_path, DEPTHS_DIR, image_name + "." + DEPTH_EXT);

        if (ExistsFile(rgbd_path)) {
            DepthMap depthmap;
            depthmap.Read(rgbd_path);
            double height_scale = 1.0 * height / depthmap.GetHeight();
            double width_scale = 1.0 * width / depthmap.GetWidth();

            std::random_device rd; 
            std::mt19937 e(rd());
            std::uniform_real_distribution<double> pertub_val(-1, 1); // random [-1, 1]
            for (int y = 0; y < depthmap.GetHeight(); y++) {
                for (int x = 0; x < depthmap.GetWidth(); x++) {
                    float depth = depthmap.Get(y, x);
                    if (depth > 0.0f) {
                        depth *= 1.0 + pertub_val(e) * 0.05;  // pertub depth
                        uvd.emplace_back(x * width_scale, y * height_scale, depth);
                        depth_range.first = std::min(depth_range.first, depth);
                        depth_range.second = std::max(depth_range.second, depth);
                    }
                }
            }
        }
    }

    std::cout << "See " << uvd.size() << " Map Points" << std::endl;

    if (options.init_from_delaunay && !options_.init_from_rgbd) {
        InitDepthMapFromDelaunay(problem, uvd, image);
    } else {
        InitDepthMapFromSparsePoint(problem, uvd, image);
    }

    if (depth_range.first == std::numeric_limits<float>::max()) {
        depth_range.first = 0.0f;
        depth_range.second = 100.0f;
    }
    depth_range.first *= 0.9f;
    depth_range.second *= 1.1f;

    // depth_range.first = std::max(depth_range.first, 0.000001f);
    depth_range.first = 0.000001f;

    return depth_range;
}


void PatchMatchController::PyramidRefineDepthNormal(
        const std::string &workspace_path, 
        const int level, 
        bool fill_flag,
        std::string input_type,
        PatchMatchOptions options) {
  
    PrintHeading1(StringPrintf("Refine depth map (level =  %d)", level));
    Timer reading_timer;
    reading_timer.Start();

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> refine_map_thread_pool_;
    int num_threads = std::min(num_eff_threads, (int)problems_.size());
    refine_map_thread_pool_.reset(new ThreadPool(num_eff_threads));
    std::cout << "Number of problems:  " << problems_.size() << std::endl;

    const auto& model = workspace_->GetModel();
    for (int problem_idx = 0; problem_idx < problems_.size(); ++problem_idx) {
        Problem& problem = problems_.at(problem_idx);
        refine_map_thread_pool_->AddTask(FillDepthMap,
            options, workspace_path, problem,
            model.images.at(problem.ref_image_idx),
            model.GetImageName(problem.ref_image_idx),
            level, fill_flag, input_type);
    }
    refine_map_thread_pool_->Wait();

    reading_timer.PrintSeconds();
}

void PatchMatchController::RefineDepthAndNormalMap(
                  const PatchMatchOptions options,
                  const std::string& workspace_path,
                  const int image_idx,
                  DepthMap& ref_depth_map,
                  NormalMap& ref_normal_map,
                  const int level){
    
    std::cout << "RefineDepthAndNormalMap" << std::endl;
    
    if (options.verbose) {
        const auto& model = workspace_->GetModel();
        const auto& depth_maps_path = JoinPaths(workspace_path, DEPTHS_DIR);
        const auto& normal_maps_path = JoinPaths(workspace_path, NORMALS_DIR);
        const std::string out_file_name = StringPrintf("%s.%s.%s", 
            model.GetImageName(image_idx).c_str(), options.output_type.c_str(), DEPTH_EXT);
        // std::cout << "out file name: " << out_file_name << std::endl;
        const std::string out_depth_map_path = 
            JoinPaths(depth_maps_path, out_file_name);
        const std::string out_normal_map_path = 
            JoinPaths(normal_maps_path, out_file_name);
        std::string parent_path = GetParentDir(out_depth_map_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        ref_depth_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", (out_depth_map_path + "-ref").c_str(), level));
        parent_path = GetParentDir(out_normal_map_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        ref_normal_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", (out_normal_map_path + "-ref").c_str(), level));
    }

    const DepthMap& depth_map = depth_maps_.at(image_idx);
    const NormalMap& normal_map = normal_maps_.at(image_idx);

    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();
    const int ref_width = ref_depth_map.GetWidth();
    const int ref_height = ref_depth_map.GetHeight();

    float scale_height = (float) height / ref_height;
    float scale_width = (float) width / ref_width;

    for (int row = 0; row < ref_height; row++) {
        for (int col = 0; col < ref_width; col++) {
            if (ref_depth_map.Get(row, col) > 1.0e-6){
                continue;
            }
            int u = col * scale_width + 0.5;
            int v = row * scale_height + 0.5;
            if (u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }
            float depth_ori = depth_map.Get(v, u);

            ref_depth_map.Set(row, col, depth_ori);
            float normal_ref[3];
            normal_map.GetSlice(v, u, normal_ref);
            ref_normal_map.SetSlice(row, col, normal_ref);
        }
    }  

    if (options.verbose) {
        const auto& model = workspace_->GetModel();
        const auto& depth_maps_path = JoinPaths(workspace_path, DEPTHS_DIR);
        const auto& normal_maps_path = JoinPaths(workspace_path, NORMALS_DIR);
        const std::string out_file_name = StringPrintf("%s.%s.%s", 
            model.GetImageName(image_idx).c_str(), options.output_type.c_str(), DEPTH_EXT);
        // std::cout << "out file name: " << out_file_name << std::endl;
        const std::string out_depth_map_path = 
            JoinPaths(depth_maps_path, out_file_name);
        const std::string out_normal_map_path = 
            JoinPaths(normal_maps_path, out_file_name);
        std::string parent_path = GetParentDir(out_depth_map_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        ref_depth_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", (out_depth_map_path + "-up").c_str(), level));
        parent_path = GetParentDir(out_normal_map_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        ref_normal_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", (out_normal_map_path + "-up").c_str(), level));
    }
}

float PatchMatchController::ScoreRemoveOutlier(const int cluster_id = -1) {
    std::vector<PlyPoint>* fused_points; 
    std::vector<float>* fused_points_score; 
    std::vector<std::vector<uint32_t> >* fused_points_visibility;
    std::vector<std::vector<float> >* fused_points_vis_weight;
    bool has_weight = false;
    if (cluster_id < 0){
        fused_points = &cross_fused_points_;
        fused_points_score = &cross_fused_points_score_;
        fused_points_visibility = &cross_fused_points_visibility_;
        if (!cross_fused_points_vis_weight_.empty()){
            has_weight = true;
            fused_points_vis_weight = &cross_fused_points_vis_weight_;
        }
    } else {
        fused_points = &cluster_points_[cluster_id];
        fused_points_score = &cluster_points_score_[cluster_id];
        fused_points_visibility = &cluster_points_visibility_[cluster_id];
    }
    if (fused_points_score->size() != fused_points->size()){
        std::cout << "fused_points_score_ is filled by 0" << std::endl;
        std::vector<float> temp_scores(fused_points->size(), 0.0f);
        fused_points_score->swap(temp_scores);
    }
    Timer timer;
    timer.Start();

    const size_t num_fused_point = fused_points->size();

    if (0){
        size_t i = 0, j = 0;
        for (i = 0; i < fused_points->size(); ++i){
            const auto fused_point = fused_points->at(i);
            if (std::isnan(fused_point.x) || std::isinf(fused_point.x) ||
                std::isnan(fused_point.y) || std::isinf(fused_point.y) ||
                std::isnan(fused_point.z) || std::isinf(fused_point.z)) {
                continue;
            } 
            // std::cout << "i - j : " << i << " - " << j << std::endl;
            fused_points->at(j) = fused_points->at(i);
            fused_points_visibility->at(j) = fused_points_visibility->at(i);
            fused_points_score->at(j) = fused_points_score->at(i);
            if (has_weight){
                fused_points_vis_weight->at(j) = fused_points_vis_weight->at(i);
            }
            j = j + 1;
        }
        fused_points->resize(j);
        fused_points_visibility->resize(j);
        fused_points_score->resize(j);
        fused_points->shrink_to_fit();
        fused_points_visibility->shrink_to_fit();
        fused_points_score->shrink_to_fit();
        if (has_weight){
            fused_points_vis_weight->resize(j);
            fused_points_vis_weight->shrink_to_fit();
        }
        std::cout << "Number of nan or inf points: " << i - j << std::endl; 
    }

    std::vector<float> dists;
    float average_dist;
    ComputeAverageDistance(*fused_points, dists, &average_dist, options_.nb_neighbors);
    int num_weak_vis = 0;

    std::vector<float> score_color, score_dist;
    if (options_.verbose){
        {
            std::vector<float> temp_color(fused_points_score->size(), 0.0f);
            score_color.swap(temp_color);
        }
        {
            std::vector<float> temp_dist(fused_points_score->size(), 0.0f);
            score_dist.swap(temp_dist);
        }
        for (size_t i = 0; i < fused_points_score->size(); ++i) {
            if (fused_points_score->at(i) < 1e-6){
                continue;
            }
            score_color.at(i) = fused_points_score->at(i);
            score_dist.at(i) = std::min(float(dists[i] / (options_.max_spacing_factor * average_dist)), 1.0f);
        }
    }
    if (average_dist > 1e-6){
        const float inv_factor = 1.0f / (options_.max_spacing_factor * average_dist);
        for (size_t i = 0; i < num_fused_point; ++i) {
            auto& score = fused_points_score->at(i);
            float factor = std::min(dists[i] * inv_factor, 1.0f);
            // score = score * 0.6 + factor * 0.4;
            score = score * factor;
            if (score > 1e-6){
                num_weak_vis++;
            }
        }
    } else {
        std::cout << "average_dist <= 1e-6\tfactor = 1.0f" << std::endl;
    }

    const float percent_weak_vis = std::max(float(num_weak_vis / num_fused_point), 0.015f);
    const float robust_min = Percentile(*fused_points_score, 100.0f - 100 * percent_weak_vis);
    const float robust_max = Percentile(*fused_points_score, 100);
    const float robust_range = robust_max - robust_min;
    const float robust_inv_range = 1.0f / robust_range;

    std::cout << "num_weak_vis: " << num_weak_vis 
              << " (" << percent_weak_vis
              << ")" << std::endl;
    std::cout << "robust min: " << robust_min << std::endl;
    std::cout << "robust max: " << robust_max << std::endl;

    std::vector<float> ratios(256, 0.0f);
    std::vector<float> outlier_ratios(256, 0.0f);
    int num_pnts = 0;
    std::vector<int > vals(num_fused_point, 255);
    for (size_t i = 0; i < num_fused_point; ++i) {
        const float gray = 
            std::min(std::max((fused_points_score->at(i) - robust_min), 0.0f), 
            robust_range) * robust_inv_range;
        vals[i] = gray * 255;
        if (vals[i] < 0 || vals[i] > 255){
            std::cout << "ratios vals[" << i << "] is bad!!!\t" << vals[i] << ", " << fused_points_score->at(i) << ", " << gray << std::endl;
            vals[i] = 255;
        }
        ratios[vals[i]]++;
        num_pnts++;
    }

    if (0){
        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(0));
        const auto& dense_reconstruction_path = 
            JoinPaths(reconstruction_path, DENSE_DIR);

        std::string ratios_path = JoinPaths(dense_reconstruction_path, "score.csv");
        std::ofstream ratios_file(ratios_path, std::ios::trunc);
        CHECK(ratios_file.is_open()) << ratios_path;
        uint64_t acc_num_temp = 0;
        for (size_t i = 1; i < 256; ++i) {
            acc_num_temp += ratios[i];
            ratios_file << i << ", " << ratios[i] << ", " << acc_num_temp << std::endl;
        }
        ratios_file.close();
        std::cout << "Save score.csv" << std::endl;

        const float robust_color_min = Percentile(score_color, 100.0f - 100 * percent_weak_vis);
        const float robust_color_max = Percentile(score_color, 100);
        const float robust_color_range = robust_color_max - robust_color_min;

        const float robust_dist_min = Percentile(score_dist, 100.0f - 100 * percent_weak_vis);
        const float robust_dist_max = Percentile(score_dist, 100);
        const float robust_dist_range = robust_dist_max - robust_dist_min;

        std::vector<float> color_ratios(256, 0.0f);
        std::vector<float> dist_ratios(256, 0.0f);
        std::vector<int > dist_vals(fused_points_score->size(), 255);
        std::vector<int > color_vals(fused_points_score->size(), 255);

        for (size_t i = 0; i < score_dist.size(); ++i) {
            const float dist_gray = 
                std::min(std::max((score_dist.at(i) - robust_dist_min), 0.0f), 
                robust_dist_range) / robust_dist_range;
            dist_vals[i] = dist_gray * 255;
            if (dist_vals[i] < 0 || dist_vals[i] > 255){
                dist_vals[i] = 255;
            }
            dist_ratios[dist_vals[i]]++;

            const float color_gray = 
                std::min(std::max((score_color.at(i) - robust_color_min), 0.0f), 
                robust_color_range) / robust_color_range;
            color_vals[i] = color_gray * 255;
            if (color_vals[i] < 0 || color_vals[i] > 255){
                color_vals[i] = 255;
            }
            color_ratios[color_vals[i]]++;
        }

        std::string dist_path = JoinPaths(dense_reconstruction_path, "score_dist.csv");
        std::ofstream dist_file(dist_path, std::ios::trunc);
        CHECK(dist_file.is_open()) << dist_path;
        uint64_t acc_dist_num_temp = 0;
        for (size_t i = 1; i < 256; ++i) {
            acc_dist_num_temp += dist_ratios[i];
            dist_file << i << ", " << dist_ratios[i] << ", " << acc_dist_num_temp << std::endl;
        }
        dist_file.close();
        std::cout << "Dist score: "<< robust_dist_min << "-" << robust_dist_max 
                  << "\nSaved dist score distrub" << std::endl;

        std::string color_path = JoinPaths(dense_reconstruction_path, "score_color.csv");
        std::ofstream color_file(color_path, std::ios::trunc);
        CHECK(color_file.is_open()) << color_path;
        uint64_t acc_color_num_temp = 0;
        for (size_t i = 1; i < 256; ++i) {
            acc_color_num_temp += color_ratios[i];
            color_file << i << ", " << color_ratios[i] << ", " << acc_color_num_temp << std::endl;
        }
        color_file.close();
        std::cout << "Color score: "<< robust_color_min << "-" << robust_color_max 
                  << "\nSaved color score distrub" << std::endl;
    }

    for (size_t i = 0; i < 256; ++i) {
        outlier_ratios[i] = ratios[i] / num_weak_vis;
        ratios[i] /= num_fused_point;
    }
    float aum_ratio = 0.0f;
    std::vector<float> acc_ratios(256, 0.0f);

    float outlier_aum_ratio = 0.0f;
    std::vector<float> outlier_acc_ratios(256, 0.0f);
    outlier_ratios[0] = 0;

    for (size_t i = 0; i < 256; ++i) {
        aum_ratio += ratios[i];
        acc_ratios[i] = aum_ratio;

        outlier_aum_ratio += outlier_ratios[i];
        outlier_acc_ratios[i] = outlier_aum_ratio;
    }
    size_t num_outlier(0);
    // const float score_thres = mval + options.outlier_deviation_factor * stdev;

    size_t i, j;
    for (i = 0, j = 0; i < num_fused_point; ++i) {
        // if ((ratios[val] < options_.outlier_max_density && acc_ratios[val] >= options_.outlier_percent) ||
        if ((outlier_ratios[vals[i]] < options_.outlier_max_density && outlier_acc_ratios[vals[i]] >= options_.outlier_percent) ||
            vals[i] == 255) {
            num_outlier++;
        } else {
            fused_points->at(j) = fused_points->at(i);
            fused_points_visibility->at(j) = fused_points_visibility->at(i);
            fused_points_score->at(j) = fused_points_score->at(i);
            if (has_weight){
                fused_points_vis_weight->at(j) = fused_points_vis_weight->at(i);
            }
            j = j + 1;
        }
    }
    fused_points->resize(j);
    fused_points_visibility->resize(j);
    fused_points_score->resize(j);
    fused_points->shrink_to_fit();
    fused_points_visibility->shrink_to_fit();
    fused_points_score->shrink_to_fit();
    if (has_weight){
        fused_points_vis_weight->resize(j);
        fused_points_vis_weight->shrink_to_fit();
    }
    std::cout << StringPrintf("Score Remove %d outliers in %.3fs\n", 
                                num_outlier, timer.ElapsedSeconds());
    return average_dist;
}

float PatchMatchController::DistRemoveOutlier(float remove_factor, const int cluster_id = -1) {
    std::vector<PlyPoint>* fused_points; 
    std::vector<float>* fused_points_score; 
    std::vector<std::vector<uint32_t> >* fused_points_visibility;
    std::vector<std::vector<float> >* fused_points_vis_weight;
    bool has_weight = false;
    if (cluster_id < 0){
        fused_points = &cross_fused_points_;
        fused_points_score = &cross_fused_points_score_;
        fused_points_visibility = &cross_fused_points_visibility_;
        if (!cross_fused_points_vis_weight_.empty()){
            has_weight = true;
            fused_points_vis_weight = &cross_fused_points_vis_weight_;
        }
    } else {
        fused_points = &cluster_points_[cluster_id];
        fused_points_score = &cluster_points_score_[cluster_id];
        fused_points_visibility = &cluster_points_visibility_[cluster_id];
    }
    if (fused_points_score->size() != fused_points->size()){
        std::cout << "fused_points_score_ is filled by 0" << std::endl;
        std::vector<float> temp_scores(fused_points->size(), 0.0f);
        fused_points_score->swap(temp_scores);
    }
    Timer timer;
    timer.Start();

    std::vector<float> dists;
    float average_dist;
    ComputeAverageDistance(*fused_points, dists, &average_dist, options_.nb_neighbors);

    size_t num_outlier(0);
    if (average_dist > 1e-6){
        double accum  = 0.0;
        double accum_weig  = 0.0;
        for (size_t i = 0; i < fused_points_score->size(); ++i){
            double pnt_weig = 0;
            if (has_weight){
                for (auto weig : fused_points_vis_weight->at(i)){
                    pnt_weig += weig;
                }
            } else {
                pnt_weig = fused_points_visibility->at(i).size();
            }
            accum += pnt_weig * (dists[i] - average_dist) * (dists[i] - average_dist);
            accum_weig += pnt_weig;
        }
        // float stdev_dist = sqrt(accum / (fused_points_score->size() - 1));
        const float stdev_dist = sqrt(accum / accum_weig);
        const float dist_thres = average_dist + remove_factor * stdev_dist;
        std::cout << "remove_factor, average_dist , stdev_dist: " << remove_factor << ", " 
            << average_dist << ", " << stdev_dist << std::endl;

        size_t i, j;
        for (i = 0, j = 0; i < fused_points_score->size(); ++i) {
            if (dists[i] < dist_thres){
                fused_points->at(j) = fused_points->at(i);
                fused_points_visibility->at(j) = fused_points_visibility->at(i);
                fused_points_score->at(j) = fused_points_score->at(i);
                if (has_weight){
                    fused_points_vis_weight->at(j) = fused_points_vis_weight->at(i);
                }
                j = j + 1;
            } else {
                num_outlier++;
            }
        }
        fused_points->resize(j);
        fused_points_visibility->resize(j);
        fused_points_score->resize(j);
        fused_points->shrink_to_fit();
        fused_points_visibility->shrink_to_fit();
        fused_points_score->shrink_to_fit();
        if (has_weight){
            fused_points_vis_weight->resize(j);
            fused_points_vis_weight->shrink_to_fit();
        }
    } else {
        std::cout << "average_dist <= 1e-6\tfactor = 1.0f" << std::endl;
    }

    std::cout << StringPrintf("Dist Remove %d outliers in %.3fs\n", 
                                num_outlier, timer.ElapsedSeconds());
    return average_dist;
}

void PatchMatchController::VoxelRemoveOutlier(const double average_dist,
                                              float factor,
                                              const int cluster_id = -1) {
    // float factor = options_.voxel_factor;
    const double filter_x = average_dist * factor;
    const double filter_y = average_dist * factor; 
    const double filter_z = average_dist * factor;
    std::vector<PlyPoint>* fused_points; 
    std::vector<float>* fused_points_score; 
    std::vector<std::vector<uint32_t> >* fused_points_visibility;
    std::vector<std::vector<float> >* fused_points_vis_weight;
    bool has_weight = false;
    if (cluster_id < 0){
        fused_points = &cross_fused_points_;
        fused_points_score = &cross_fused_points_score_;
        fused_points_visibility = &cross_fused_points_visibility_;
        if (!cross_fused_points_vis_weight_.empty()){
            has_weight = true;
            fused_points_vis_weight = &cross_fused_points_vis_weight_;
        }
    } else {
        fused_points = &cluster_points_[cluster_id];
        fused_points_score = &cluster_points_score_[cluster_id];
        fused_points_visibility = &cluster_points_visibility_[cluster_id];
    }

    Timer timer;
    timer.Start();

    size_t i, j;
    size_t num_point = fused_points->size();
    std::cout << "Compute Bounding Box(size: " << filter_x << " x " 
        << filter_y << " x " << filter_z << ")" << std::endl;
    Eigen::Vector3f lt, rb;
    lt.setConstant(std::numeric_limits<float>::max());
    rb.setConstant(std::numeric_limits<float>::lowest());
    for (i = 0; i < num_point; ++i) {
        const PlyPoint& point = fused_points->at(i);
        lt.x() = std::min(lt.x(), point.x);
        lt.y() = std::min(lt.y(), point.y);
        lt.z() = std::min(lt.z(), point.z);
        rb.x() = std::max(rb.x(), point.x);
        rb.y() = std::max(rb.y(), point.y);
        rb.z() = std::max(rb.z(), point.z);
    }
    std::cout << "LT: " << lt.transpose() << std::endl;
    std::cout << "RB: " << rb.transpose() << std::endl;

    size_t lenx = (rb.x() - lt.x()) / filter_x + 1;
    size_t leny = (rb.y() - lt.y()) / filter_y + 1;
    size_t lenz = (rb.z() - lt.z()) / filter_z + 1;
    uint64_t slide = lenx * leny;
    const int num_neighbors = 26;
    int neighbor_offs[num_neighbors][3] = 
    { { -1, -1, -1 }, { 0, -1, -1 }, { 1, -1, -1}, 
      { -1, 0, -1 }, { 0, 0, -1 }, { 1, 0, -1}, 
      { -1, 1, -1 }, { 0, 1, -1 }, { 1, 1, -1}, 
      { -1, -1, 0 }, { 0, -1, 0 }, { 1, -1, 0 }, 
      { -1, 0, 0 }, { 1, 0, 0}, 
      { -1, 1, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, 
      { -1, -1, 1 }, { 0, 1, 1 }, { 1, -1, 1}, 
      { -1, 0, 1 }, { 0, 0, 1 }, { 1, 0, 1}, 
      { -1, 1, 1 }, { 0, 1, 1 }, { 1, 1, 1}};

    std::unordered_map<uint64_t, std::vector<size_t> > m_voxels_map;
    for (i = 0; i < num_point; ++i) {
        const PlyPoint& point = fused_points->at(i);
        if (point.x < lt.x() || point.y < lt.y() || point.z < lt.z() ||
            point.x > rb.x() || point.y > rb.y() || point.z > rb.z()) {
            continue;
        }
        uint64_t ix = (point.x - lt.x()) / filter_x;
        uint64_t iy = (point.y - lt.y()) / filter_y;
        uint64_t iz = (point.z - lt.z()) / filter_z;

        uint64_t key = iz * slide + iy * lenx + ix;
        m_voxels_map[key].push_back(i);
    }

    std::vector<std::pair<uint64_t, std::vector<size_t> > > m_voxels_vec(m_voxels_map.size());
    i = 0;
    for (auto voxel_map : m_voxels_map) {
        m_voxels_vec[i].first = voxel_map.first;
        m_voxels_vec[i].second = voxel_map.second;
        i++;
    }

    std::vector<uint64_t > vec_remove_pnts;
#pragma omp parallel for
    for (j = 0; j < m_voxels_vec.size(); ++j) {
        int num_neighbor = 0;

        num_neighbor++;
        uint64_t key = m_voxels_vec[j].first;
        uint64_t iz = key / slide;
        uint64_t iy = (key % slide) / lenx;
        uint64_t ix = (key % slide) % lenx;

        for (int i = 0; i < num_neighbors; i++){
            uint64_t neighbor_key = (iz + neighbor_offs[i][2]) * slide +
                                    (iy + neighbor_offs[i][1]) * lenx + 
                                    (ix + neighbor_offs[i][0]);
            if (m_voxels_map.find(neighbor_key) == m_voxels_map.end()){
                continue;
            } 
            uint64_t next_neighbor_key = (iz + neighbor_offs[i][2] * 2) * slide +
                                         (iy + neighbor_offs[i][1] * 2) * lenx + 
                                         (ix + neighbor_offs[i][0] * 2);
            if (m_voxels_map[neighbor_key].size() > factor / 2 + 0.5){
                num_neighbor++;
               if (m_voxels_map.find(next_neighbor_key) == m_voxels_map.end()){
                    continue;
                }
                if (m_voxels_map[next_neighbor_key].size() > factor / 2 + 0.5){
                    num_neighbor++;
                }
            } else if (m_voxels_map[neighbor_key].size() > 0.5){
                if (m_voxels_map.find(next_neighbor_key) == m_voxels_map.end()){
                    continue;
                }
                if (m_voxels_map[neighbor_key].size() + m_voxels_map[neighbor_key].size() > factor){
                    num_neighbor++;
                }
            }
            
            if (num_neighbor > 3){
                break;
            }
        }

        #pragma omp critical
        if (num_neighbor < 3) {
            for (auto pnt_id : m_voxels_vec[j].second){
                vec_remove_pnts.push_back(pnt_id);
            }
        }
    }

    std::cout << "Number of points, average_dist, factor, voxels_num: " 
              << num_point << ", " << average_dist << ", " << factor 
              << ", " << m_voxels_map.size() << std::endl;
#if 0
    {
        auto SamplePoint = [&](uint64_t voxel_key, PlyPoint *samp_point, 
                                PlyPoint *neigh_point) {
            auto voxel_map = m_voxels_map[voxel_key];

            Eigen::Vector3f X(0, 0, 0);
            Eigen::Vector3f Xn(0, 0, 0);
            BitmapColor<uint8_t> color;
            float sum_w(0.0);
            int max_samples(0);
            uint8_t best_label(-1), samps_per_label[256];
            memset(samps_per_label, 0, sizeof(uint8_t) * 256);

            for (int k = 0; k < voxel_map.size(); ++k) {
                size_t point_idx = voxel_map.at(k);
                CHECK_LT(point_idx, num_point);
                const PlyPoint& point = fused_points->at(point_idx);
                float w = 1.0f;
                X += w * Eigen::Vector3f(&point.x);
                Xn += w * Eigen::Vector3f(&point.nx);
                sum_w += w;

                uint8_t sid = (point.s_id + 256) % 256;
                samps_per_label[sid]++;
                if (samps_per_label[sid] > max_samples) {
                max_samples = samps_per_label[sid];
                best_label = point.s_id;
                }
            }
            X /= sum_w;
            Xn = (Xn / sum_w).normalized();
            // color /= sum_w;
            {
                float intensity = (float)std::min((int)voxel_map.size(), 5) / 5.0f;
                ColorMap(intensity, color.r, color.g, color.b);
            }

            // PlyPoint point;
            (*samp_point).x = X[0];
            (*samp_point).y = X[1];
            (*samp_point).z = X[2];
            (*samp_point).nx = Xn[0];
            (*samp_point).ny = Xn[1];
            (*samp_point).nz = Xn[2];
            (*samp_point).r = color.r;
            (*samp_point).g = color.g;
            (*samp_point).b = color.b;
            (*samp_point).s_id = best_label;

            int num_neighbor = 0;
            num_neighbor++;
            uint64_t key = voxel_key;
            uint64_t iz = key / slide;
            uint64_t iy = (key % slide) / lenx;
            uint64_t ix = (key % slide) % lenx;
            for (int i = 0; i < num_neighbors; i++){
                uint64_t neighbor_key = (iz + neighbor_offs[i][2]) * slide +
                                        (iy + neighbor_offs[i][1]) * lenx + 
                                        (ix + neighbor_offs[i][0]);
                if (m_voxels_map[neighbor_key].size() > 2){
                    num_neighbor++;
                }
            }

            {
                float intensity = (float)std::min(num_neighbor, 5) / 5.0f;
                ColorMap(intensity, color.r, color.g, color.b);
            }
            (*neigh_point).x = X[0];
            (*neigh_point).y = X[1];
            (*neigh_point).z = X[2];
            (*neigh_point).nx = Xn[0];
            (*neigh_point).ny = Xn[1];
            (*neigh_point).nz = Xn[2];
            (*neigh_point).r = color.r;
            (*neigh_point).g = color.g;
            (*neigh_point).b = color.b;
            (*neigh_point).s_id = best_label;
        };

        std::vector<uint64_t> voxel_map_index;
        for (auto voxel_map : m_voxels_map) {
            // if (voxel_map.second.size() > 1) {
            voxel_map_index.push_back(voxel_map.first);
            // }
        }
        std::vector<PlyPoint> simplified_points;
        std::vector<PlyPoint> neighbor_points;
        simplified_points.resize(voxel_map_index.size());
        neighbor_points.resize(voxel_map_index.size());
        for (size_t i = 0; i < voxel_map_index.size(); ++i) {
            SamplePoint(voxel_map_index.at(i), &simplified_points[i], &neighbor_points[i]);
        }
        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(0));
        const auto& dense_reconstruction_path = 
            JoinPaths(reconstruction_path, DENSE_DIR);
        WriteBinaryPlyPoints(dense_reconstruction_path + "/" + to_string(factor) + "voxel_num_points.ply", 
                            simplified_points, true, true);
        WriteBinaryPlyPoints(dense_reconstruction_path + "/" + to_string(factor) + "voxel_neigh_points.ply", 
                            neighbor_points, true, true);
        WriteBinaryPlyPoints(dense_reconstruction_path + "/" + to_string(factor) + "ori_points.ply", 
                            *fused_points, true, true);
    }
#endif
    
    int num_outlier = 0;
    if (!vec_remove_pnts.empty()){
        std::sort(vec_remove_pnts.begin(), vec_remove_pnts.end());
        size_t i, j, k;
        for (i = 0, j = 0, k = 0; i < fused_points->size(); ++i){
            if (i == vec_remove_pnts[k]){
                k++;
                num_outlier++;
            }else {
                fused_points->at(j) = fused_points->at(i);
                fused_points_score->at(j) = fused_points_score->at(i);
                fused_points_visibility->at(j) = fused_points_visibility->at(i);
                if (has_weight){
                    fused_points_vis_weight->at(j) = fused_points_vis_weight->at(i);
                }
                j = j + 1;
            }
        }
        fused_points->resize(j);
        fused_points_score->resize(j);
        fused_points_visibility->resize(j);
        fused_points->shrink_to_fit();
        fused_points_score->shrink_to_fit();
        fused_points_visibility->shrink_to_fit();
        if (has_weight){
            fused_points_vis_weight->resize(j);
            fused_points_vis_weight->shrink_to_fit();
        }
    }
    std::cout << StringPrintf("Voxel Remove %d outliers in %.3fs\n", 
                                num_outlier, timer.ElapsedSeconds());
}

void PatchMatchController::PlaneScoreCompute(const double average_dist, 
    const float factor = 1, const int cluster_id = -1) {
    
    std::cout << "Plane Score Compute begin ...." << std::endl;
    const int voxel_step = 15;
	// const int num_min_pnt = int(0.5 * 3.14 * voxel_step * voxel_step / 9.0f);
	const int num_min_pnt = 9;
    const float max_normal_cos = 1.0f - std::cos(DegToRad(options_.max_normal_error));

    std::vector<PlyPoint>* fused_points; 
    std::vector<float>* fused_points_score; 
    std::vector<std::vector<uint32_t> >* fused_points_visibility;
    std::vector<std::vector<float> >* fused_points_vis_weight;
    bool has_weight = false;
    if (cluster_id < 0){
        fused_points = &cross_fused_points_;
        fused_points_score = &cross_fused_points_score_;
        fused_points_visibility = &cross_fused_points_visibility_;
        if (!cross_fused_points_vis_weight_.empty()){
            has_weight = true;
            fused_points_vis_weight = &cross_fused_points_vis_weight_;
        }
    } else {
        fused_points = &cluster_points_[cluster_id];
        fused_points_score = &cluster_points_score_[cluster_id];
        fused_points_visibility = &cluster_points_visibility_[cluster_id];
    }

    Timer timer;
    timer.Start();

	std::unique_ptr<ThreadPool> samp_thread_pool;
	int num_eff_threads = GetEffectiveNumThreads(-1);

    bool voxel_simp = true;

    float radius_factor = options_.plane_raidus_factor;
    if (options_.fused_delaunay_sample){
        radius_factor /= options_.fused_dist_insert;
    }

    const auto& model = workspace_->GetModel();
    const std::vector<Image>& images = model.images;
    const float mean_dist = model.ComputeNthDistance( 0.75);
    const float mean_angular = model.ComputeMeanAngularResolution();
    float angular_threshold = std::min(0.0215f, mean_angular * options_.plane_raidus_factor);

    const float nb_radius = factor * (average_dist * radius_factor 
                                              + angular_threshold * mean_dist);
    // const float nb_radius = 1.0;

    std::cout << "\t=> nb_radius, average_dist, radius_factor, angular_threshold, mean_dist: " 
                      << nb_radius << ", " << average_dist << ", " << radius_factor << ", " 
                      << angular_threshold << ", " << mean_dist << std::endl;

    // voxel simp
	std::vector<PlyPoint> voxel_pnts;
    std::unordered_map<uint64_t, size_t> voxel_key2id;
	std::vector<PlyPoint> simp_pnts;
	std::vector<float> simp_pnt_weights;
	std::unordered_map<uint64_t, uint64_t > index_map_voxels;
	std::vector<uint64_t> l0_voxels_keys;
	std::vector<std::unordered_map<uint64_t, std::vector<size_t> >> v_m_voxels_map;
	std::vector<std::unordered_map<uint64_t, float >> v_m_voxels_weight;

	const size_t num_point = fused_points->size();
	Eigen::Vector3f lt, rb;
	size_t lenx = 0, leny = 0, lenz = 0, slide = 0;
    float voxel_length = 0;
	// if (voxel_simp){
	{
        Timer voxel_timer;
        voxel_timer.Start();

		lt.setConstant(std::numeric_limits<float>::max());
		rb.setConstant(std::numeric_limits<float>::lowest());
		for (size_t i = 0; i < num_point; ++i) {
			const PlyPoint& point = fused_points->at(i);
			lt.x() = std::min(lt.x(), point.x);
			lt.y() = std::min(lt.y(), point.y);
			lt.z() = std::min(lt.z(), point.z);
			rb.x() = std::max(rb.x(), point.x);
			rb.y() = std::max(rb.y(), point.y);
			rb.z() = std::max(rb.z(), point.z);
		}
	    // std::cout << "\t=> Compute Bounding Box ( " <<  num_point << " points ): "
        //                   <<  rb.transpose() << " ->" << lt.transpose() << std::endl;

		int num_pyramid = 5;
		v_m_voxels_map.resize(num_pyramid);
		v_m_voxels_weight.resize(num_pyramid);
		std::vector<size_t> v_lenx(num_pyramid, 0), v_leny(num_pyramid, 0), v_lenz(num_pyramid, 0);
		std::vector<size_t> v_slide(num_pyramid, 0);
		std::vector<float> v_voxel_length(num_pyramid, 1.0f);

		auto LevelVox = [&](int level){
			v_voxel_length.at(level) = nb_radius / voxel_step * std::pow(2.0f, level);
            const float inv_length = 1.0f / v_voxel_length.at(level);

			v_lenx.at(level) = (rb.x() - lt.x()) * inv_length + 1;
			v_leny.at(level) = (rb.y() - lt.y()) * inv_length + 1;
			v_lenz.at(level) = (rb.z() - lt.z()) * inv_length + 1;
			v_slide.at(level) = v_lenx.at(level)  * v_leny.at(level);

			for (size_t i = 0; i < num_point; ++i) {
				const PlyPoint& point = fused_points->at(i);
				if (point.x < lt.x() || point.y < lt.y() || point.z < lt.z() ||
					point.x > rb.x() || point.y > rb.y() || point.z > rb.z()) {
					continue;
				}
				uint64_t ix = (point.x - lt.x()) * inv_length;
				uint64_t iy = (point.y - lt.y()) * inv_length;
				uint64_t iz = (point.z - lt.z()) * inv_length;

				uint64_t key = iz * v_slide.at(level) + iy * v_lenx.at(level) + ix;
				if (level ==0 && v_m_voxels_map.at(level)[key].empty()) {
					l0_voxels_keys.push_back(key);
				}
				v_m_voxels_map.at(level)[key].push_back(i);

				float wgt = 0.0;
				if (has_weight){
					for (const auto wgt_pnt : fused_points_vis_weight->at(i) ){
						wgt += wgt_pnt;
					}
				} else {
					wgt = (float)fused_points_visibility->at(i).size();
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

		std::cout << "\t => "<< 0 << "-level v_voxel_length, lenx, leny, lenz, slide: " 
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
                    const float factor = 1.0f / std::pow(2, level);
					uint64_t iz_level = iz * factor;
					uint64_t iy_level = iy * factor;
					uint64_t ix_level = ix * factor;

					uint64_t level_key = iz_level * v_slide.at(level) + iy_level * v_lenx.at(level) + ix_level;

					int num_eff_vox = 0;
					int num_neig_vox = 0;
					float sum_weight = 0.0f;
					for (int i = 0; i < num_neighbors; i++){
						uint64_t neighbor_key = (iz_level + neighbor_offs[i][2]) * v_slide.at(level) +
												(iy_level + neighbor_offs[i][1]) * v_lenx.at(level) + 
												(ix_level + neighbor_offs[i][0]);
						if (v_m_voxels_map.at(level).find(neighbor_key) == v_m_voxels_map.at(level).end()){
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
					if ( (num_neig_vox < 2 && level < 2) || (num_eff_vox > 8 && mean_weight >  2 * v_m_voxels_weight.at(level)[level_key])) {
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

		std::cout << "\t =>voxel filter: " << l0_voxels_keys.size() << " -> " << voxel_map_index.size() << std::endl;

		simp_pnts.resize(voxel_map_index.size());
		simp_pnt_weights.resize(voxel_map_index.size());
		auto SamplePoint = [&](size_t begin_idx, size_t end_idx) {
			for (size_t idx = begin_idx; idx < end_idx; idx++ ){
				uint64_t voxel_key = voxel_map_index.at(idx);
				PlyPoint& samp_point = simp_pnts[idx];
				float& samp_point_weight = simp_pnt_weights[idx];
				auto voxel_map = v_m_voxels_map.at(0)[voxel_key];

				Eigen::Vector3f X(0, 0, 0);
				float sum_w(0.0);

				for (int k = 0; k < voxel_map.size(); ++k) {
					size_t point_idx = voxel_map.at(k);
					CHECK_LT(point_idx, num_point);
					const PlyPoint& point = fused_points->at(point_idx);

					float wgt = 0.0;
					if (has_weight){
						for (const auto wgt_pnt : fused_points_vis_weight->at(point_idx) ){
							wgt += wgt_pnt;
						}
					} else {
						wgt = (float)fused_points_visibility->at(point_idx).size();
					}

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
		num_eff_threads = std::ceil(voxel_map_index.size() / num_pnts_per_threads);
		std::unique_ptr<ThreadPool> samp_thread_pool;
		samp_thread_pool.reset(new ThreadPool(num_eff_threads));
		for (size_t i = 0; i < num_eff_threads; ++i) {
			size_t begin_idx = i * num_pnts_per_threads;
			size_t end_idx = std::min((i+ 1) * num_pnts_per_threads, voxel_map_index.size());
			if (begin_idx >= end_idx){
				continue;
			}
			samp_thread_pool->AddTask(SamplePoint, begin_idx, end_idx);
		}
		samp_thread_pool->Wait();

        // voxel pnt
        size_t num_voxel_pnts = l0_voxels_keys.size();
        voxel_pnts.resize(num_voxel_pnts);
        for (size_t idx = 0; idx < num_voxel_pnts; idx++){
            const auto key = l0_voxels_keys.at(idx);
            uint64_t iz = key / slide;
            uint64_t iy = (key % slide) / lenx;
            uint64_t ix = (key % slide) % lenx;
            PlyPoint& voxel_point = voxel_pnts[idx];
            voxel_point.x = (ix + 0.5) * voxel_length + lt.x();
            voxel_point.y = (iy + 0.5) * voxel_length + lt.y();
            voxel_point.z = (iz + 0.5) * voxel_length + lt.z();

            voxel_key2id[key] = idx;
        }

        std::cout << "\t=> Voxel Simp:" <<  num_point << " -> (voxel samp: "<< voxel_map_index.size() 
            << " / voxel pnt: " << voxel_pnts.size() << ") in " 
            << voxel_timer.ElapsedSeconds() << "[s]" << std::endl;
        std::cout << "\t=> LT: " << lt.transpose() <<  " / RB: " << rb.transpose() << std::endl;
	}

    if (options_.verbose && !simp_pnts.empty()  && simp_pnts.size() == simp_pnt_weights.size()) {
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

        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(select_reconstruction_idx_));
        const auto& dense_reconstruction_path = 
            JoinPaths(reconstruction_path, DENSE_DIR);
        if ( !ExistsFile(dense_reconstruction_path + "/"  + "fused-simp.ply")){
            WriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-simp.ply", simp_pnts, 1, 1);
        } else {
            AppendWriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-simp.ply", simp_pnts);
        }
        std::cout << "\t => save simp points ( " << simp_pnts.size() 
            << " points. mean_weight, stdev: " << mean_weight << ", " << stdev  << ")" << std::endl;
    }

    std::vector<float> scores;
    std::vector<float> dists;

	const int D = 3;
	typedef CGAL::Epick_d<CGAL::Dimension_tag<D> > K;
	typedef K::Point_d Point_3;
	typedef CGAL::Search_traits_d<K,CGAL::Dimension_tag<D> >  Traits;
	typedef CGAL::Random_points_in_cube_d<Point_3>       Random_points_iterator;
	typedef CGAL::Counting_iterator<Random_points_iterator> N_Random_points_iterator;
	typedef CGAL::Kd_tree<Traits> Tree;
    typedef CGAL::Sliding_midpoint<Traits> Sliding_midpoint;
	typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;
	typedef CGAL::Fuzzy_iso_box<Traits> Fuzzy_iso_box;

	// const size_t num_point = fused_points->size();
	scores.resize(num_point);
	dists.resize(num_point);

    const size_t num_simp = simp_pnts.size();
	std::vector<Point_3> points(num_simp);
	for (std::size_t i = 0; i < num_simp; i++) {
		const auto &fused_point = simp_pnts[i];
		points[i] = Point_3(fused_point.x, fused_point.y, fused_point.z);
	}

	// Instantiate a KD-tree search.
	std::cout << "\t=> Instantiating a search tree for point cloud spacing ..." << std::endl;
	Sliding_midpoint sliding(100);
	Tree tree(points.begin(), points.end(), sliding);
#ifdef CGAL_LINKED_WITH_TBB
  	tree.build<CGAL::Parallel_tag>();
#endif

	struct PlaneData{
		float score = -1.f;
		// float dist = -1.f;
        float mean_dist = -1.f;
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
		Eigen::Vector3f normal = Eigen::Vector3f::Zero();
		uint16_t num_neighbor = 0;

		bool is_valid(){
			return bool(score > -0.1f && mean_dist > -0.1f && normal.norm() > 0.5);
		};
	};

	const float normal_threld = std::cos(DegToRad(options_.max_normal_error));
    const float inv_voxel_length = 1.0f / voxel_length;
	auto ComputeVoxelScore = [&](std::size_t s, float radius){
		// Neighbor search can be instantiated from
		// several threads at the same time
		const auto &query = Point_3(voxel_pnts.at(s).x, voxel_pnts.at(s).y, voxel_pnts.at(s).z);
		Fuzzy_sphere fs(query, radius);

		std::vector<Point_3> cgal_near_pnts;
		std::back_insert_iterator<std::vector<Point_3>> its(cgal_near_pnts);
		tree.search(its, fs);

		std::vector<Eigen::Vector3f> neighbors;
		std::vector<float> neighbors_weight;
		neighbors.resize(cgal_near_pnts.size() + 1);
		neighbors_weight.resize(cgal_near_pnts.size() + 1);

		neighbors[0] = Eigen::Vector3f ((float)query.at(0), (float)query.at(1), (float)query.at(2));
		neighbors_weight[0] = 1.0f;
        for (int k = 0; k < cgal_near_pnts.size(); ++k){
            auto & pnt = cgal_near_pnts.at(k);
        // for (const auto pnt : cgal_near_pnts){
			Eigen::Vector3f point(pnt[0], pnt[1], pnt[2]);
            float wgt(1.0f);
            if (voxel_simp){
				uint64_t ix = (point.x() - lt.x()) * inv_voxel_length;
				uint64_t iy = (point.y() - lt.y()) * inv_voxel_length;
				uint64_t iz = (point.z() - lt.z()) * inv_voxel_length;

				uint64_t key = iz * slide + iy * lenx + ix;
				wgt = simp_pnt_weights.at(index_map_voxels[key]);
			}
            neighbors_weight[k + 1] = wgt;
            neighbors[k + 1] = std::move(point);
			// neighbors_weight.push_back(wgt);
			// neighbors.push_back(point);
		}

		if (neighbors.size() < num_min_pnt) {
			struct PlaneData data;
			data.score = -1.0f;
			// data.dist = -1.0f;
			data.mean_dist = -1.0f;
			data.center = Eigen::Vector3f::Zero();
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

		Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
		for (int idx = 1; idx < neighbors.size(); idx++) {
			const auto &point = neighbors.at(idx);
			float wgt = neighbors_weight.at(idx);
			Eigen::Vector3f V = wgt * (point - centroid);
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

        float svd_score = exp(- 0.8 * svd_singular_values[2]/svd_singular_values[1]) 
							* exp(  0.4 * std::sqrt(svd_singular_values[1]/svd_singular_values[0])  - 0.4);
		// float score = exp(- svd_singular_values[2]/svd_singular_values[1]);
		// float score = exp(- svd_singular_values[2]/svd_singular_values[1]) 
		// 					* exp(  0.5 * svd_singular_values[1]/svd_singular_values[0] - 0.5);
		// float score = exp(- svd_singular_values[1]/svd_singular_values[0]) 
		// 					  * exp(1 - svd_singular_values[1]/svd_singular_values[2]);

		// float normal_cos = std::abs(svd_singular_vector.dot(
		// 	Eigen::Vector3f(fused_points->at(s).nx, fused_points->at(s).ny, fused_points->at(s).nz)));
		// float normal_score = std::min(normal_cos, normal_threld) + 1 - normal_threld;
        // float normal_score = 1.0f;
        // if (normal_cos < normal_threld){
        //     normal_score = 0.5 + 0.5 * normal_cos;
    	// }

		// float dist = std::abs((neighbors.at(0) - centroid).dot(svd_singular_vector));

		struct PlaneData data;
		// data.score = svd_score * normal_score;
		data.score = svd_score;
		// data.dist = std::min(0.1f *dist / mean_dist, 1.0f);
        data.mean_dist = mean_dist;
		data.center = centroid;
		data.normal = svd_singular_vector;
		data.num_neighbor = neighbors.size();

		return data;
	};

    // compute voxel_pnts score

	struct DistData{
        float mean_dist = -1.f;
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
		Eigen::Vector3f normal = Eigen::Vector3f::Zero();
	};

    size_t num_voxel_pnts = voxel_pnts.size();
    std::vector<std::pair<float, struct DistData>> voxel_scores;
    voxel_scores.resize(num_voxel_pnts);
	std::size_t progress = 0;
	std::size_t indices_print_step = num_voxel_pnts / 20;
#ifndef CGAL_LINKED_WITH_TBB
	num_eff_threads = GetEffectiveNumThreads(-1);
	std::cout << "\t=> Starting muti-threads nearst neighbor computation ...(" 
        << num_eff_threads << " threads)" << std::endl;
	std::unique_ptr<ThreadPool> thread_pool;
	thread_pool.reset(new ThreadPool(num_eff_threads));
	{
		auto ComputePointSpace = [&](std::size_t s) {

			if (progress % indices_print_step == 0) {
				std::cout<<"\r";
				std::cout<<"\tPoints score computed ["<< progress <<" / "<< num_voxel_pnts <<"]"<<std::flush;
			}
			++progress;

            DistData dist;
			PlaneData data_1;
			float radius_step = 3.0f;
			float radius_factor = 1.0f / radius_step;
			while (!data_1.is_valid()) {
				radius_factor *= 1.2;
				data_1 = ComputeVoxelScore(s, nb_radius * radius_factor);
				if (radius_factor > 1.8) {
                    voxel_scores.at(s) = std::pair<float, DistData>(0.1f, dist);
					return;
				}
			}
			radius_factor *= radius_step;

			PlaneData data_2= ComputeVoxelScore(s, nb_radius * radius_factor);
			float norm_score = std::abs(data_1.normal.dot(data_2.normal));
			float score = data_1.score * data_2.score * norm_score;

			// float dist = score * score * data_1.dist;
            dist.mean_dist = data_1.mean_dist;
            dist.center = data_1.center;
            dist.normal = data_1.normal;
            voxel_scores.at(s) = std::pair<float, DistData>(score, dist);
		};

		for (std::size_t i = 0; i < num_voxel_pnts; ++i) {
			thread_pool->AddTask(ComputePointSpace, i);
		}
		thread_pool->Wait();
	}
#else
	std::cout << "\t=> Starting parallel plane score computation (tbb)..." << std::endl;
	tbb::parallel_for (tbb::blocked_range<std::size_t> (0, num_voxel_pnts),
					[&](const tbb::blocked_range<std::size_t>& r) {
			for (std::size_t s = r.begin(); s != r.end(); ++ s) {

			if (progress % indices_print_step == 0) {
				std::cout<<"\r";
				std::cout<<"Points score computed ["<<progress<<" / "<< num_voxel_pnts <<"]"<<std::flush;
			}
			++progress;

            DistData dist;
			PlaneData data_1;
			float radius_step = 3.0f;
			float radius_factor = 1.0f / radius_step;
			while (!data_1.is_valid()) {
				radius_factor *= 1.2;
				data_1 = ComputeVoxelScore(s, nb_radius * radius_factor);
				if (radius_factor > 1.8) {
                    voxel_scores.at(s) = std::pair<float, DistData>(0.1f, dist);
					return;
				}
			}
			radius_factor *= radius_step;

			PlaneData data_2= ComputeVoxelScore(s, nb_radius * radius_factor);

			float norm_score = std::abs(data_1.normal.dot(data_2.normal));
			float score = data_1.score * data_2.score * norm_score;

            dist.mean_dist = data_1.mean_dist;
            dist.center = data_1.center;
            dist.normal = data_1.normal;
            voxel_scores.at(s) = std::pair<float, DistData>(score, dist);
		}
	});
#endif
    std::cout << std::endl;

#if 1
    auto ComputeGradKernel = [&](size_t image_id, int col, int row, int delt){
        float conv_x = 0, conv_y = 0;
        BitmapColor<uint8_t> color;
        bitmaps_.at(image_id).GetPixel(col, row - delt, &color);
        float grey1 = 0.5 * color.r + 0.25 * color.g + 0.25 * color.b;
        bitmaps_.at(image_id).GetPixel(col, row + delt, &color);
        float grey2 = 0.5 * color.r + 0.25 * color.g + 0.25 * color.b;
        conv_x = 0.8f * (grey2 - grey1);

        bitmaps_.at(image_id).GetPixel(col - delt, row, &color);
        float grey3 = 0.5 * color.r + 0.25 * color.g + 0.25 * color.b;
        bitmaps_.at(image_id).GetPixel(col + delt, row, &color);
        float grey4 = 0.5 * color.r + 0.25 * color.g + 0.25 * color.b;
        conv_y = 0.8f * (grey4 - grey3);

        float mag = sqrt(conv_x * conv_x + conv_y * conv_y) / 255.f;
        return mag;
    };
#else
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
#endif
    
    const float grad_factor = 2.0f * M_PI;
    auto ComputeGrad = [&](size_t pnt_id){
        const auto pnt = fused_points->at(pnt_id);
        const auto pnt_vis = fused_points_visibility->at(pnt_id);
        const Eigen::Vector3f xyz(pnt.x,pnt.y,pnt.z);
        std::vector<float> pnt_conf;
        float m_conf(0.0f);

        auto VisGrad = [&](size_t vis){
            Eigen::Map<const Eigen::RowMatrix3x4f> P(images[vis].GetP());
            const Eigen::Vector3f next_proj = P * xyz.homogeneous();
            const float inv_z = 1.0f / next_proj(2);
            int vis_col = static_cast<int>(std::round(next_proj(0) * inv_z));
            int vis_row = static_cast<int>(std::round(next_proj(1) * inv_z));

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
        };

        if (pnt_vis.size() > 1){
            auto select_vis = std::minmax_element(pnt_vis.begin(), pnt_vis.end());
            VisGrad(*select_vis.first);
            VisGrad(*select_vis.second);
        } else {
            VisGrad(pnt_vis.at(0));
        }
        // for (const auto vis : pnt_vis){
        //     VisGrad(vis);
        // }

        if (pnt_conf.size() >= 1){
            for (size_t i = 0; i < pnt_conf.size(); ++i) {
                m_conf += pnt_conf[i];
            }
            m_conf = m_conf / pnt_conf.size();
        } else {
            m_conf = 0.0f;
        }
        m_conf = std::min(m_conf, 0.4f);
		float grad_score = (std::cos(grad_factor * m_conf) + 1) / 2;
        return grad_score * grad_score;
    };

    auto ComputeScore = [&](std::size_t s){
        const auto& pnt = fused_points->at(s);
        int voxel_ix = (pnt.x - lt.x()) * inv_voxel_length;
        int voxel_iy = (pnt.y - lt.y()) * inv_voxel_length;
        int voxel_iz = (pnt.z - lt.z()) * inv_voxel_length;
        uint64_t key =  voxel_iz * slide + voxel_iy * lenx + voxel_ix;

        float score = voxel_scores.at(voxel_key2id[key]).first;
        float grad = ComputeGrad(s);
        scores.at(s) = grad * score;

        DistData dist = voxel_scores.at(voxel_key2id[key]).second;
        Eigen::Vector3f pnt_v(pnt.x, pnt.y, pnt.z);
        float pnt_dist = std::abs((pnt_v - dist.center).dot(dist.normal));
        pnt_dist = std::min(0.1 * pnt_dist / dist.mean_dist, 1.0);
        dists.at(s) = score * score* pnt_dist;

        // geoms.at(s) = score;
        // grads.at(s) = grad;
        return;
    };

#pragma omp parallel for //schedule(dynamic)
    for (size_t i = 0; i < num_point; i++){
        ComputeScore(i);
    }
	// float max_score = *std::max_element(scores.begin(),scores.end());
	// float min_score = *std::min_element(scores.begin(),scores.end());
	// std::cout << "\t=>max, min: " << max_score << ", " <<min_score << std::endl;
    fused_points_score->swap(scores);
    std::cout << "\t=> omp parallel Compute Grad-Score ... Done" << std::endl;

    if (options_.verbose){
        std::vector<PlyPoint> dist_pnts;
        std::vector<PlyPoint> color_pnts;
        std::vector<PlyPoint> color_plane_pnts;
        for (int i = 0; i < fused_points->size(); i++){
            PlyPoint pnt = fused_points->at(i);
            float dist = dists.at(i);
            dist = std::min(dist, 1.0f);
            ColorMap(dist, pnt.r, pnt.g, pnt.b);
            dist_pnts.push_back(pnt);

            float score = fused_points_score->at(i);
            score = std::min(score, 1.0f);
            ColorMap(score, pnt.r, pnt.g, pnt.b);
            color_pnts.push_back(pnt);
            if (score < 0.8){
                continue;
            }
            color_plane_pnts.push_back(pnt);
        }

        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(select_reconstruction_idx_));
        const auto& dense_reconstruction_path = 
            JoinPaths(reconstruction_path, DENSE_DIR);
        if ( !ExistsFile(dense_reconstruction_path + "/"  + "fused-color.ply")){
            WriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-color.ply", color_pnts, 1, 1);
        } else {
            AppendWriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-color.ply", color_pnts);
        }

        if ( !ExistsFile(dense_reconstruction_path + "/"  + "fused-plane-color.ply")){
            WriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-plane-color.ply", color_plane_pnts, 1, 1);
        } else {
            AppendWriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-plane-color.ply", color_plane_pnts);
        }

        if ( !ExistsFile(dense_reconstruction_path + "/"  + "fused-dist-color.ply")){
            WriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-dist-color.ply", dist_pnts, 1, 1);
        } else {
            AppendWriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-dist-color.ply", dist_pnts);
        }

        std::vector<PlyPoint> color_voxel_pnts;
        for (int i = 0; i < num_voxel_pnts; i++){
            PlyPoint pnt = voxel_pnts.at(i);
            float score = voxel_scores.at(i).first;
            ColorMap(score, pnt.r, pnt.g, pnt.b);
            color_voxel_pnts.push_back(pnt);
        }
        if ( !ExistsFile(dense_reconstruction_path + "/"  + "fused-voxel-color.ply")){
            WriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-voxel-color.ply", color_voxel_pnts, 1, 1);
        } else {
            AppendWriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-voxel-color.ply", color_voxel_pnts);
        }

        std::cout << "\t=> save score color point clouds " << std::endl;
    }

    if (options_.outlier_removal){
        std::vector<PlyPoint> remove_points;
        int new_num = 0;
        for (int i = 0; i < num_point; i++){
            // float score = plane_scores.at(i);
            float score = dists.at(i);
            score = std::min(score, 1.0f);
            // if (score > 0.85 && i % 2 ==0){
            if (score > options_.plane_dist_threld){
                remove_points.push_back(fused_points->at(i));
                continue;
            }
            fused_points->at(new_num) = fused_points->at(i);
            fused_points_visibility->at(new_num) = fused_points_visibility->at(i);
            if (has_weight){
                fused_points_vis_weight->at(new_num) = fused_points_vis_weight->at(i);
            }
            
            fused_points_score->at(new_num) = fused_points_score->at(i);
            new_num++;
        }
        fused_points->resize(new_num);
        fused_points_visibility->resize(new_num);
        if (has_weight){
            fused_points_vis_weight->resize(new_num);
        }
        fused_points_score->resize(new_num);
        std::cout << "\t=> outlier " << num_point -  new_num 
            << " points removal with dist score ( pnt: " 
            << num_point << " -> " << new_num << ")" << std::endl;

        if (options_.verbose && !remove_points.empty() ) {
            const auto& reconstruction_path = 
                JoinPaths(workspace_path_, std::to_string(select_reconstruction_idx_));
            const auto& dense_reconstruction_path = 
                JoinPaths(reconstruction_path, DENSE_DIR);
            if ( !ExistsFile(dense_reconstruction_path + "/"  + "fused-dist-remove.ply")){
                WriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-dist-remove.ply", remove_points, 1, 1);
            } else {
                AppendWriteBinaryPlyPoints(dense_reconstruction_path + "/"  + "fused-dist-remove.ply", remove_points);
            }
        }
    }

    std::cout << StringPrintf("\tPlane Score  %d points in %.3fs", 
        fused_points->size() , timer.ElapsedSeconds()) << std::endl;
}

void PatchMatchController::AddBlackList(const std::vector<uint8_t>& black_list) {
  semantic_label_black_list_ = black_list;
}

void PatchMatchController::ComputeSemanticLabel(
    const std::vector<int> &cluster_images,
    const std::string semantic_path) {

    Timer timer;
    timer.Start();

    std::cout << "ComputeSemanticLabel ..." << std::endl;
    const auto& model = workspace_->GetModel();
    const std::vector<Image>& images = model.images;
    std::vector<std::unique_ptr<Bitmap> > semantic_maps; 
    semantic_maps.resize(images.size());

    auto Read = [&](std::vector<int> image_idxs){
        for (int image_idx : image_idxs){
            const std::string image_name = model.GetImageName(image_idx);
            const auto semantic_name = JoinPaths(semantic_path, image_name);
            const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
            if (ExistsFile(semantic_base_name + "png")) {
                semantic_maps.at(image_idx).reset(new Bitmap);
                semantic_maps.at(image_idx)->Read(semantic_base_name + "png", false);
            } else if (ExistsFile(semantic_base_name + "jpg")) {
                semantic_maps.at(image_idx).reset(new Bitmap);
                semantic_maps.at(image_idx)->Read(semantic_base_name + "jpg", false);
            } else if (ExistsFile(semantic_base_name + "JPG")) {
                semantic_maps.at(image_idx).reset(new Bitmap);
                semantic_maps.at(image_idx)->Read(semantic_base_name + "JPG", false);
            }
        }
    };

    std::unique_ptr<ThreadPool> semantic_thread_pool;
    const int num_eff_threads = std::min(GetEffectiveNumThreads(-1),
            (int)cluster_images.size());
    semantic_thread_pool.reset(new ThreadPool(num_eff_threads));

    int num_images = std::ceil((float)cluster_images.size() / num_eff_threads);
    int num_cluster = std::ceil((float)cluster_images.size() / num_images);
    std::cout << "\tnum_eff_threads, cluster number, cluster images number: " << num_eff_threads 
              << ", " << num_cluster << ", " << cluster_images.size() << std::endl;
    for (int i = 0; i < num_cluster; i++){
        std::vector<int > image_idxs;
        image_idxs.clear();
        if ((i+1) * num_images < cluster_images.size()){
            image_idxs.insert(image_idxs.end(), 
                cluster_images.begin() + i * num_images, 
                cluster_images.begin() + (i+1) * num_images);
        } else {
            image_idxs.insert(image_idxs.end(), 
                cluster_images.begin() + i * num_images, 
                cluster_images.end());
        }
        // std::cout << "image_idxs: " << image_idxs.size() ;
        // for(int j = 0; j < image_idxs.size(); j++){
        //     std::cout << " " << image_idxs[j];
        // };
        // std::cout << std::endl;
        semantic_thread_pool->AddTask(Read, image_idxs);
    }
    semantic_thread_pool->Wait();

    size_t num_points = cross_fused_points_.size();
    size_t num_cluster_pnts = std::ceil((float)num_points / num_eff_threads);
    int num_sem_cluster = std::ceil((float)num_points / num_cluster_pnts);

    std::vector<std::vector<size_t >> remove_pnts_id;
    remove_pnts_id.resize(num_sem_cluster);
    auto Semantic = [&](size_t begin_id, size_t end_id, int cluster_id){
        for (size_t i = begin_id; i < end_id; i++){
            const auto & pnt = cross_fused_points_.at(i);
            const auto & visibilitys = cross_fused_points_visibility_.at(i);
            std::vector<uint8_t> fused_point_seg_id;
            for (const auto vis : visibilitys){
                if (semantic_maps.at(vis)) {
                    Eigen::Vector3f xyz(pnt.x, pnt.y, pnt.z);
                    Eigen::Map<const Eigen::RowMatrix3x4f> P(images[vis].GetP());
                    const Eigen::Vector3f next_proj = P * xyz.homogeneous();
                    int col = static_cast<int>(std::round(next_proj(0) / next_proj(2)));
                    int row = static_cast<int>(std::round(next_proj(1) / next_proj(2)));

                    const int width = images.at(vis).GetWidth();
                    const int height = images.at(vis).GetHeight();
                    if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
                        BitmapColor<uint8_t> semantic;
                        semantic_maps.at(vis)->GetPixel(col, row, &semantic);
                        fused_point_seg_id.push_back(semantic.r);
                    }

                }/* else {
                    std::cout << "semantic_maps " << vis << " is empty!" << std::endl;
                }*/
            }

            if (!fused_point_seg_id.empty()) {
                int max_samples = 0;
                int best_label = -1;
                int samps_per_label[256];
                memset(samps_per_label, 0, sizeof(int) * 256);
                for (auto id : fused_point_seg_id) {
                    uint8_t sid = (id + 256) % 256;
                    samps_per_label[sid]++;
                    if (samps_per_label[sid] > max_samples) {
                        max_samples = samps_per_label[sid];
                        best_label = id;
                    }
                }
                size_t num_fused_point = fused_point_seg_id.size();
                for (auto id : semantic_label_black_list_) {
                    float ratio = samps_per_label[id] / (float)num_fused_point;
                    if (ratio > options_.num_consistent_semantic_ratio || id == LABEL_SKY) {
                        remove_pnts_id[cluster_id].push_back(i);
                        // continue;
                    }
                }
                cross_fused_points_.at(i).s_id = best_label;
            }/* else {
                std::cout << "fused_point_seg_id.empty() " << std::endl;
            }*/
        }
    };

    for(int i = 0; i < num_sem_cluster; i++){
        semantic_thread_pool->AddTask(Semantic, i * num_cluster_pnts, 
            std::min((i+1) * num_cluster_pnts, num_points), i);
    }
    semantic_thread_pool->Wait();

    std::vector<size_t > vec_remove_pnts;
    for (int cluster_id = 0; cluster_id < num_sem_cluster; cluster_id++){
        if (!remove_pnts_id[cluster_id].empty()){
            vec_remove_pnts.insert(vec_remove_pnts.end(), 
                remove_pnts_id[cluster_id].begin(),
                remove_pnts_id[cluster_id].end());
        }
    }
    int num_outlier = 0;
    if (!vec_remove_pnts.empty()){
        std::sort(vec_remove_pnts.begin(), vec_remove_pnts.end());
        size_t i, j, k;
        for (i = 0, j = 0, k = 0; i < cross_fused_points_.size(); ++i){
            if (i == vec_remove_pnts[k]){
                k++;
                num_outlier++;
            }else {
                cross_fused_points_.at(j) = cross_fused_points_.at(i);
                cross_fused_points_visibility_.at(j) = cross_fused_points_visibility_.at(i);
                if (!cross_fused_points_vis_weight_.empty()){
                    cross_fused_points_vis_weight_.at(j) = cross_fused_points_vis_weight_.at(i);
                }
                j = j + 1;
            }
        }
        cross_fused_points_.resize(j);
        cross_fused_points_visibility_.resize(j);
        cross_fused_points_.shrink_to_fit();
        cross_fused_points_visibility_.shrink_to_fit();
        if (!cross_fused_points_vis_weight_.empty()){
            cross_fused_points_vis_weight_.resize(j);
            cross_fused_points_vis_weight_.shrink_to_fit();
        }
    }
    std::cout << StringPrintf("Semantic Remove %d outliers in %.3fs\n", 
                              num_outlier, timer.ElapsedSeconds());
}

void PatchMatchController::ConvertVisibility2Dense(
    const std::unordered_map<size_t, image_t> & cluster_image_idx_to_id,
    std::vector<std::vector<uint32_t> >& fused_vis) {
    
    const auto dense_image_id_to_idx = workspace_->GetDenseImageId2Idx();

    for (size_t i = 0; i < fused_vis.size(); i++){
        for (size_t j = 0; j < fused_vis[i].size(); j++){
            size_t image_idx = fused_vis[i][j];
            image_t image_id = cluster_image_idx_to_id.at(image_idx);
            fused_vis[i][j] = dense_image_id_to_idx.at(image_id);
        }
    }
}

void PatchMatchController::Run() {
    PrintHeading1("Patch Match");

    // ReadWorkspace();
    GetGpuProp();
    EstimateThreadsPerGPU();
    ReadGpuIndices();
    ReadCrossFilterGpuIndices();

    thread_pool_.reset(new ThreadPool(gpu_indices_.size()));

    Reconstruct();
}

void PatchMatchController::Reconstruct() {
    const float min_triangulation_angle_rad = 
        DegToRad(options_.min_triangulation_angle);
    const float kTriangulationAnglePercentile = 75;

    size_t reconstruction_idx = select_reconstruction_idx_;
    PrintHeading1(StringPrintf("Reconstructing# %d", reconstruction_idx));

    const auto& reconstruction_path = 
        JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
    const auto& dense_reconstruction_path = 
        JoinPaths(reconstruction_path, DENSE_DIR);
    if (!ExistsDir(dense_reconstruction_path)) {
        return;
    }

    const auto& undistort_image_path = 
        JoinPaths(dense_reconstruction_path, IMAGES_DIR);
    if (!ExistsDir(undistort_image_path)) {
        return;
    }

    const auto& undistort_sparse_path = 
        JoinPaths(dense_reconstruction_path, SPARSE_DIR);
    if (!ExistsDir(undistort_sparse_path)) {
        return;
    }

    auto semantic_maps_path = JoinPaths(dense_reconstruction_path, SEMANTICS_DIR);
    auto mask_maps_path = JoinPaths(dense_reconstruction_path, MASKS_DIR);
    bool has_mask = boost::filesystem::exists(mask_maps_path);
    
    std::cout << "Reading workspace..." << std::endl;
    Workspace::Options workspace_options;

    workspace_options.max_image_size = options_.max_image_size;
    workspace_options.image_as_rgb = false;
    // workspace_options.cache_size = options_.cache_size;
    workspace_options.cache_size = 1e-6;
    workspace_options.image_path = undistort_image_path;
    workspace_options.workspace_path = dense_reconstruction_path;
    workspace_options.workspace_format = options_.image_type;
    workspace_options.input_type = PHOTOMETRIC_TYPE;
    workspace_.reset(new Workspace(workspace_options));

    // size_t cluster_begin = select_cluster_idx_ < 0 ? 0 : select_cluster_idx_;
    // size_t cluster_end = select_cluster_idx_ < 0 ? num_cluster_ : select_cluster_idx_ + 1;
    // for (size_t sfm_cluster_idx = cluster_begin; sfm_cluster_idx < cluster_end; sfm_cluster_idx++) {
    // size_t sfm_cluster_idx = 0;
    if (1){
        // std::string cluster_rect_path = 
        //     JoinPaths(reconstruction_path, std::to_string(sfm_cluster_idx));
        // if (!ExistsDir(cluster_rect_path) && sfm_cluster_idx != 0){
        //     return;
        // }
        // if (!ExistsDir(cluster_rect_path)){
        //     boost::filesystem::create_directories(cluster_rect_path);
        // }
        const Model model = workspace_->GetModel();

        float begin_memroy;
        GetAvailableMemory(begin_memroy);
        std::cout << "Reconstructing begin free memory: " << begin_memroy << " G" << std::endl;

        // It's better to initialize depth map with delaunay method.
        BuildPriorModel(reconstruction_path);

        std::cout << "ComputeDepthRanges" << std::endl;
        std::vector<std::pair<float, float> > depth_ranges_temp;
        depth_ranges_temp = model.ComputeDepthRanges();

        std::cout << "Load update_images" << std::endl;
        std::string update_path = JoinPaths(undistort_sparse_path, "update_images.txt");
        std::unordered_set<int> update_image_idxs;
        if (options_.map_update && ExistsFile(update_path)){
            model.GetUpdateImageidxs(update_path, update_image_idxs);
        } else {
            for (int i = 0; i < model.images.size(); i++){
                update_image_idxs.insert(i);
            }
        }

        std::cout << "ComputeOverlappingImages" << std::endl;
        const size_t kCheckNum = options_.max_num_src_images * 2;
        const size_t kCheckNum_g = options_.max_num_src_images * 5;
        const double kMinTriangulationAngle = options_.min_triangulation_angle;
        std::vector<std::vector<int> > overlapping_cluster_images = 
        model.GetMaxOverlappingImages(kCheckNum_g, kMinTriangulationAngle, update_image_idxs);

        // std::vector<std::vector<int> > overlapping_cluster_images = overlapping_images;
        for (size_t i = 0; i < overlapping_cluster_images.size(); ++i) {
            const size_t eff_num_images = 
            std::min(overlapping_cluster_images[i].size(), kCheckNum);
            overlapping_cluster_images[i].resize(eff_num_images);
            overlapping_cluster_images[i].shrink_to_fit();
        }

        if (ReadClusterBox() < 0){
            auto cluster_roi_path = reconstruction_path;
            if (!ExistsFile(JoinPaths(cluster_roi_path, ROI_BOX_NAME)) && 
                ExistsFile(JoinPaths(cluster_roi_path, "..", ROI_BOX_NAME))){
                cluster_roi_path = JoinPaths(cluster_roi_path, "..");
            }
             auto ori_box_path = JoinPaths(cluster_roi_path, ROI_BOX_NAME);

            if (ExistsFile(ori_box_path)){
                ReadBoundBoxText(ori_box_path, roi_box_);
                roi_box_.SetBoundary(options_.roi_box_width * 1.2, options_.roi_box_factor * 1.2);
                roi_box_.z_box_min = -FLT_MAX;
                roi_box_.z_box_max = FLT_MAX;
                std::cout << StringPrintf("ROI(box.txt): [%f %f %f] -> [%f %f %f]\n", 
                    roi_box_.x_box_min, roi_box_.y_box_min, roi_box_.z_box_min,
                    roi_box_.x_box_max, roi_box_.y_box_max, roi_box_.z_box_max);
            } else {
                std::vector<Eigen::Vector3f> points;
                for (const Model::Point & point : model.points) {
                    Eigen::Vector3f p(&point.x);
                    points.emplace_back(p);
                }
                for (const mvs::Image & image : model.images) {
                    Eigen::Vector3f C(image.GetC());
                    points.emplace_back(C);
                }
                roi_box_.x_min = roi_box_.y_min = roi_box_.z_min = FLT_MAX;
                roi_box_.x_max = roi_box_.y_max = roi_box_.z_max = -FLT_MAX;
                for (auto point : points) {
                    roi_box_.x_min = std::min(roi_box_.x_min, point.x());
                    roi_box_.y_min = std::min(roi_box_.y_min, point.y());
                    roi_box_.z_min = std::min(roi_box_.z_min, point.z());
                    roi_box_.x_max = std::max(roi_box_.x_max, point.x());
                    roi_box_.y_max = std::max(roi_box_.y_max, point.y());
                    roi_box_.z_max = std::max(roi_box_.z_max, point.z());
                }
                float x_offset = (roi_box_.x_max - roi_box_.x_min) * 0.05;
                float y_offset = (roi_box_.y_max - roi_box_.y_min) * 0.05;
                float z_offset = (roi_box_.z_max - roi_box_.z_min) * 0.05;
                roi_box_.x_box_min = roi_box_.x_min - x_offset;
                roi_box_.x_box_max = roi_box_.x_max + x_offset;
                roi_box_.y_box_min = roi_box_.y_min - y_offset;
                roi_box_.y_box_max = roi_box_.y_max + y_offset;
                roi_box_.z_box_min = roi_box_.z_min - z_offset;
                roi_box_.z_box_max = roi_box_.z_max + z_offset;
                roi_box_.rot = Eigen::Matrix3f::Identity();
                std::cout << StringPrintf("ROI(sparse): [%f %f %f] -> [%f %f %f]\n", 
                    roi_box_.x_box_min, roi_box_.y_box_min, roi_box_.z_box_min,
                    roi_box_.x_box_max, roi_box_.y_box_max, roi_box_.z_box_max);
            }
            num_box_ = 1;
        }

        std::vector<std::vector<int>> cluster_image_map;
        uint64_t max_images_num;

        const std::string cluster_yaml_path = JoinPaths(dense_reconstruction_path, PATCH_MATCH_CLUATER_YAML);
        if (ExistsFile(cluster_yaml_path)){
            mvs::MVSCluster mvs_cluster;
            mvs_cluster.ReadImageClusterYaml(cluster_yaml_path, cluster_image_map, cluster_step_);
            num_cluster_ = cluster_image_map.size();
        } else {
            mvs::MVSCluster mvs_cluster;
            mvs_cluster.PatchMatchImageCluster(options_, cluster_image_map, 
                    max_images_num, gpu_indices_.size(), model.images, 
                    overlapping_cluster_images, update_image_idxs, has_mask);
            num_cluster_ = cluster_image_map.size();
        }

        int cluster_begin = select_cluster_idx_ < 0 ? 0 : select_cluster_idx_ * cluster_step_;
        int cluster_end = select_cluster_idx_ < 0 ? num_cluster_ : select_cluster_idx_ * cluster_step_ + cluster_step_;
        cluster_end = std::min(num_cluster_, cluster_end);
        if (cluster_begin >= num_cluster_){
            std::cout << "crash bug: cluster_begin(" << select_cluster_idx_ << " * " <<  cluster_step_ 
                << ") >= num_cluster_(" <<  num_cluster_ << ")" << std::endl;
                ExceptionHandler(INVALID_INPUT_PARAM, 
                    JoinPaths(GetParentDir(workspace_path_), "errors/dense"), "DensePatchMatch").Dump();
                exit(INVALID_INPUT_PARAM);
            return;
        }

        {
            num_all_images_ = 0;
            num_processed_images_ = 0;
            int iter_num_per_image = (options_.pyramid_max_level / options_.pyramid_delta_level + 1.0);
            if (options_.geom_consistency){
                iter_num_per_image = iter_num_per_image * 2 + options_.num_iter_geom_consistency - 1;
            }
            if (options_.init_from_dense_points){
                iter_num_per_image += 1;
            }
            for (int idx = cluster_begin; idx < cluster_end; idx++) {  
                num_all_images_ += cluster_image_map[idx].size();
            }
            num_all_images_ = num_all_images_ * iter_num_per_image;
        }

        std::cout << "num_cluster, cluster_begin, cluster_end, num_all_images: " 
            << num_cluster_ << ", " << cluster_begin << ", " << cluster_end << ", " << num_all_images_ << std::endl;

        int cluster_id;
        for (cluster_id = cluster_begin; cluster_id < cluster_end; cluster_id++) {  

            PrintHeading1(StringPrintf("Cluster Reconstructing# %d - %d", 
                                        reconstruction_idx, cluster_id));
            float begin_memroy;
            GetAvailableMemory(begin_memroy);
            std::cout << cluster_id << "-Cluster begin free memory: " << begin_memroy << " G" << std::endl;
            const Model model = workspace_->GetModel();

            std::unordered_set<int> unique_image_idxs;
            unique_image_idxs.insert(cluster_image_map[cluster_id].begin(), cluster_image_map[cluster_id].end());
            std::vector<std::vector<int> > overlapping_images = 
                model.GetMaxOverlappingImages(kCheckNum_g, kMinTriangulationAngle, unique_image_idxs);
            for (int cluster_image_id = 0; cluster_image_id < cluster_image_map[cluster_id].size(); ++cluster_image_id) {
                int i = cluster_image_map[cluster_id].at(cluster_image_id);
                std::cout << "\r" << "Process Frame#" << i;
                Problem problem;
                problem.ref_image_idx = i;

                if (options_.max_num_src_images <= 0) {
                    problem.src_image_idxs = cluster_image_map[cluster_id];
                    problem.src_image_idxs.erase(problem.src_image_idxs.begin() + i);
                } else {
                    std::vector<uint32_t> src_image_idxs;
                    src_image_idxs.reserve(overlapping_images[i].size());
                    for (size_t j = 0; j < overlapping_images[i].size(); ++j) {
                        if (unique_image_idxs.find(overlapping_images[i][j]) == unique_image_idxs.end()) {
                            continue;
                        }
                        src_image_idxs.push_back(overlapping_images[i][j]);
                    }

                    const size_t eff_max_num_src_images = std::min(
                        src_image_idxs.size(), options_.max_num_src_images);
                    problem.src_image_idxs.reserve(eff_max_num_src_images);
                    for (size_t j = 0; j < eff_max_num_src_images; ++j) {
                        problem.src_image_idxs.push_back(src_image_idxs[j]);
                    }
                }
                if (problem.src_image_idxs.empty()) {
                    std::cout << StringPrintf(
                            "WARNING: Ignoring reference image %s, because it has no "
                            "source images.",
                            model.GetImageName(problem.ref_image_idx).c_str())
                        << std::endl;
                    continue;
                } else {
                    auto depth_range = depth_ranges_temp.at(problem.ref_image_idx);
                    if (depth_range.first < 1e-6 && depth_range.second < 1e-6) {
                        std::cout << StringPrintf(
                            "WARNING: Ignoring reference image %s, because it's scale callapsed.",
                            model.GetImageName(problem.ref_image_idx).c_str())
                        << std::endl;
                        continue;
                    }
                    problem.src_image_scales.resize(problem.src_image_idxs.size(), 
                                                    1.0f);
                    problems_.push_back(problem);
                    used_images_.insert(problem.ref_image_idx);
                    depth_ranges_.push_back(depth_range);
                }
                for (const auto& src_image_id : problem.src_image_idxs){
                    ref_2_srcs_map_[problem.ref_image_idx].push_back(src_image_id);
                    src_2_refs_map_[src_image_id].push_back(problem.ref_image_idx);
                    ref_2_problem_map_[problem.ref_image_idx] = problems_.size() -1;
                }
            }
            std::cout << std::endl;

            flag_depth_maps_.resize(model.images.size(), 0);
#ifdef CACHED_DEPTH_MAP
            {
                bitmaps_.resize(model.images.size());
                semantic_maps_.resize(model.images.size());
                depth_maps_.resize(model.images.size());
                normal_maps_.resize(model.images.size());
                mask_maps_.resize(model.images.size());
                depth_maps_temp_.resize(model.images.size());
                normal_maps_temp_.resize(model.images.size());

                prior_wgt_maps_.resize(model.images.size());

                std::cout << "Reading BitMaps..." << std::endl;
                std::unique_ptr<ThreadPool> read_thread_pool;
                const int num_eff_threads = std::min(GetEffectiveNumThreads(-1),
                     (int)problems_.size());
                std::cout << "Read num_eff_threads: " << num_eff_threads << std::endl;
                read_thread_pool.reset(new ThreadPool(num_eff_threads));
                auto Read = [&](int image_idx){
                    bitmaps_.at(image_idx).Read(workspace_->GetBitmapPath(image_idx),
                        workspace_->GetOptions().image_as_rgb);
                    if (options_.refine_with_semantic) {
                        const std::string image_name = model.GetImageName(image_idx);
                        const auto semantic_name = JoinPaths(semantic_maps_path, image_name);
                        const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
                        if (ExistsFile(semantic_base_name + "png")) {
                            semantic_maps_.at(image_idx).Read(semantic_base_name + "png", false);
                        } else if (ExistsFile(semantic_base_name + "jpg")) {
                            semantic_maps_.at(image_idx).Read(semantic_base_name + "jpg", false);
                        } else if (ExistsFile(semantic_base_name + "JPG")) {
                            semantic_maps_.at(image_idx).Read(semantic_base_name + "JPG", false);
                        }
                    }
                    // filter depth map with mask
                    if (has_mask) {
                        const std::string image_name = model.GetImageName(image_idx);
                        const auto & name_parts = StringSplit(image_name, ".");
                        std::string mask_path = JoinPaths(mask_maps_path, image_name.substr(0, image_name.size() - name_parts.back().size() - 1) + ".png");
                        if (boost::filesystem::exists(mask_path)) {
                            mask_maps_.at(image_idx).Read(mask_path, false);
                        }
                    }
                };
                for (const auto problem : problems_){
                    const auto image_idx = problem.ref_image_idx;
                    read_thread_pool->AddTask(Read, image_idx);
                }
                read_thread_pool->Wait();
            }
#endif

            if (options_.init_from_dense_points){
                auto fused_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
                if (ExistsFile(fused_path)){
                    prior_points_ = ReadPly(fused_path);
                } else {
                    ClusterReconstruction(cluster_id, 
                                  cluster_image_map, 
                                  overlapping_images,
                                  dense_reconstruction_path, 
                                  options_, true);
                    std::cout << "Build Dense PointCloud..." << std::endl;
                    prior_points_.swap(cross_fused_points_);
                }
            }
            std::fill(flag_depth_maps_.begin(), flag_depth_maps_.end(), 0);
            ClusterReconstruction(cluster_id, 
                                  cluster_image_map, 
                                  overlapping_images,
                                  dense_reconstruction_path, 
                                //   cluster_rect_path,
                                  options_, false);
        }
        num_save_fused_points_ = 0;

        // if (cluster_id == num_cluster_ && options_.patch_match_fusion  && options_.filter && options_.geo_filter){
        //     MergeFusedPly(reconstruction_path);
        // }
    }

    std::cout << StringPrintf("Depth Recovery Cost Time: %.3fmin\n", depth_elapsed_time_);
    std::cout << StringPrintf("Cross Filter Cost Time: %.3fmin\n", cross_elapsed_time_);
    std::cout << StringPrintf("Depth Fusion Cost Time: %.3fmin\n", fusion_elapsed_time_);
}

void PatchMatchController::ClusterReconstruction(
        const int cluster_id, 
        const std::vector<std::vector<int>>& cluster_image_map, 
        const std::vector<std::vector<int> >& overlapping_images,
        const std::string dense_reconstruction_path,
        // const std::string cluster_reconstruction_path,
        PatchMatchOptions options,
        bool prior_ply = false){
    const auto& undistort_image_path = 
        JoinPaths(dense_reconstruction_path, IMAGES_DIR);
    const auto semantic_maps_path = 
        JoinPaths(dense_reconstruction_path, SEMANTICS_DIR);

    const auto stereo_reconstruction_path = 
        JoinPaths(dense_reconstruction_path, STEREO_DIR);
    if (!ExistsDir(stereo_reconstruction_path)){
        boost::filesystem::create_directories(stereo_reconstruction_path);
    }
    
    // parameters setting
    bool para_ref = true;
    int para_max_level = options.pyramid_max_level;
    int para_delta_level = options.pyramid_delta_level;
    
    struct PyramidParamters
    {
        std::string input_type;
        std::string output_type;

        bool init_from_dense_points;
        bool init_from_model;
        bool init_from_visible_map;
        bool init_from_delaunay;
        bool init_from_global_map;
        bool geom_consistency;

        bool filter;
        bool conf_filter;

        bool ref_flag;
        bool fill_flag;
        int level;
        int max_level; 
        bool plane_regularizer;
        bool save_flag;

        bool rgbd = false;
    };
    
    std::vector<PyramidParamters> pyramid_parameters;

    if (prior_ply){
        options.geo_filter = true;
        options.filter_min_num_consistent = 1;

        options.fused_delaunay_sample = false;
        options.outlier_removal = true;
        options.save_depth_map = false;
        options.patch_match_fusion = true;

        PyramidParamters pyramid_parameter_photo = 
            { PHOTOMETRIC_TYPE, PHOTOMETRIC_TYPE, 
              options.init_from_dense_points,
              options.init_from_model, options.init_from_visible_map, 
              options.init_from_delaunay, options.init_from_global_map, false, // patchmatch params
              true, false, //filter params
              false, false, PLY_INIT_DEPTH_LEVEL, PLY_INIT_DEPTH_LEVEL, false, false}; // prior_fused_points
        pyramid_parameters.push_back(pyramid_parameter_photo);
    } else {
        int iter_level = para_max_level;
        do {
            if (iter_level == para_max_level) {
                PyramidParamters pyramid_parameter_photo =
                    { PHOTOMETRIC_TYPE, PHOTOMETRIC_TYPE, 
                    options.init_from_dense_points,
                    options.init_from_model, options.init_from_visible_map, 
                    options.init_from_delaunay, options.init_from_global_map, false, // patchmatch params
                    options.filter && (!options.geom_consistency), options.conf_filter, //filter params
                    false, false, iter_level, para_max_level, false, 
                    (iter_level == 0) && (!options.geom_consistency),
                    options.init_from_rgbd}; // multi-scale
                pyramid_parameters.push_back(pyramid_parameter_photo);
            } else if (para_ref) {
                PyramidParamters pyramid_parameter_photo = 
                    { PHOTOMETRIC_TYPE, PHOTOMETRIC_TYPE, false, 
                    false, false, false, false, false, // patchmatch params
                    true, true,                      //filter params
                    true, true, iter_level, iter_level, false, false};// multi-scale
                pyramid_parameters.push_back(pyramid_parameter_photo);
            }

            if (options.geom_consistency) {
                const bool max_level_flag = (iter_level == para_max_level);
                const int num_level_iter_geom_consistency = max_level_flag ? 
                    options.num_iter_geom_consistency : 1;
                std::cout << "max_level_flag, num_level_iter_geom_consistency: " 
                    << max_level_flag << ", " 
                    << num_level_iter_geom_consistency << std::endl;
                for (int j = 0; j < num_level_iter_geom_consistency; j++){                    
                    PyramidParamters pyramid_parameter_geom = 
                        { PHOTOMETRIC_TYPE, GEOMETRIC_TYPE,
                        options.init_from_dense_points,
                        options.init_from_model, options.init_from_visible_map,
                        options.init_from_delaunay, options.init_from_global_map, 
                        options.geom_consistency, // patchmatch params
                        options.filter && (iter_level == 0) && 
                        (j == num_level_iter_geom_consistency - 1), 
                        options.conf_filter,  //filter params
                        false, false, iter_level, para_max_level, 
                        options.plane_regularizer, (iter_level == 0) && 
                        (j == num_level_iter_geom_consistency - 1), 
                        options.init_from_rgbd}; // multi-scale
                    pyramid_parameters.push_back(pyramid_parameter_geom);
                }
            }
            iter_level -= para_delta_level;
        } while(iter_level >= 0);
    }
    float factor = 1.0f;
    bool has_prior_depth_map = false;
    double depth_elapsed_time = 0.0;
    double cross_elapsed_time = 0.0;
    double fusion_elapsed_time = 0.0;

    for (int i = 0; i < pyramid_parameters.size(); i++){
        auto iter = pyramid_parameters.at(i);

        auto iter_options = options;
        iter_options.output_type = iter.output_type;
        iter_options.init_from_dense_points = iter.init_from_dense_points;
        iter_options.init_from_model = iter.init_from_model;
        iter_options.init_from_visible_map = iter.init_from_visible_map;
        iter_options.init_from_delaunay = iter.init_from_delaunay;
        iter_options.init_from_global_map = iter.init_from_global_map;
        iter_options.geom_consistency = iter.geom_consistency;
        iter_options.init_depth_random = !(iter.geom_consistency ||
                                        iter.init_from_visible_map ||
                                        iter.init_from_global_map ||
                                        iter.init_from_model ||
                                        iter.init_from_dense_points ||
                                        iter.level != iter.max_level);

        iter_options.filter = iter.filter;
        iter_options.conf_filter = iter.conf_filter;
        if (iter.rgbd ){
            if (!has_prior_depth_map){
                iter_options.geo_filter = true;
            } else {
                iter_options.has_prior_depth = true;
            }
        }

        iter_options.plane_regularizer = iter.plane_regularizer;

        // reduce random disturbance 
        float factor_delta_level = 1;
        iter_options.random_angle1_range = 
            factor_delta_level * options.random_angle1_range;
        iter_options.random_angle2_range = 
            factor_delta_level * options.random_angle2_range;
        iter_options.random_depth_ratio = 
            factor_delta_level * options.random_depth_ratio;
        if (iter.level != iter.max_level && iter.geom_consistency){
            iter_options.num_iterations = 3;
        }

        // if (iter.level != iter.max_level && 
        //     ((para_ref && !iter.ref_flag) || 
        //     (!para_ref && iter.geom_consistency))) {
        //     PyramidRefineDepthNormal(stereo_reconstruction_path, iter.level,
        //                     iter.fill_flag, iter.input_type, iter_options);
        // }
        workspace_->ResetInputType(iter.input_type);

        if (i > 0 || iter_options.geom_consistency) {
            iter_options.propagate_depth = false;
        }

        iter_options.Print();

        Timer depth_timer;
        depth_timer.Start();
        
        for (int problem_idx = 0; problem_idx < problems_.size(); 
            problem_idx++) {
            thread_pool_->AddTask(&PatchMatchController::ProcessProblem,
                                this,
                                iter_options,
                                problem_idx,
                                dense_reconstruction_path,
                                stereo_reconstruction_path,
                                iter.level,
                                iter.fill_flag,
                                iter.save_flag);
        }
        thread_pool_->Wait();
        depth_elapsed_time += depth_timer.ElapsedMinutes();
        std::cout << StringPrintf("Cluster#%d Depth Recovery Cost Time: %.3fmin\n", cluster_id, depth_elapsed_time);

#ifdef CACHED_DEPTH_MAP
        if (iter.geom_consistency){
            depth_maps_.swap(depth_maps_temp_);
            normal_maps_.swap(normal_maps_temp_);
        }
        {
            std::vector<DepthMap> depth_maps_temp;
            depth_maps_temp.resize(depth_maps_.size());
            std::vector<NormalMap> normal_maps_temp;
            normal_maps_temp.resize(normal_maps_.size());
            depth_maps_temp_.swap(depth_maps_temp);
            normal_maps_temp_.swap(normal_maps_temp);
        }
        
#endif

        if ((iter.filter && iter_options.geo_filter && !iter.ref_flag) || 
            (iter.rgbd && !has_prior_depth_map)){
            workspace_->ResetInputType(iter.output_type);

            Timer cross_timer;
            cross_timer.Start();
#if 1

            Timer filter_timer;
            filter_timer.Start();

            if (prior_ply){
                float factors = std::pow(2, PLY_INIT_DEPTH_LEVEL);
                options.gpu_memory_factor *= (factors * factors);
            }
            int num_cuda_devices = cross_filter_gpu_indices_.size();
            std::vector<std::unordered_set<int >> cluster_problem_ids;
            std::vector<std::unordered_set<int >> cluster_whole_ids;
            std::unordered_map<image_t, bool> images_exclusive;
            std::vector<int > cluster_gpu_idx;
            CrossFilterImageCluster(cluster_problem_ids, cluster_whole_ids, 
                images_exclusive, cluster_gpu_idx, options.gpu_memory_factor);

            std::cout << "Cross Filter Number of problems:  " << cluster_whole_ids.size() << std::endl;

            depth_maps_info_.resize(depth_maps_.size());
            for (int i = 0; i < depth_maps_.size(); i++){
                depth_maps_info_.at(i).width = depth_maps_.at(i).GetWidth();
                depth_maps_info_.at(i).height = depth_maps_.at(i).GetHeight();
                depth_maps_info_.at(i).depth_min = depth_maps_.at(i).GetDepthMin();
                depth_maps_info_.at(i).depth_max = depth_maps_.at(i).GetDepthMax();
            }

            const int num_eff_threads = num_cuda_devices;
            // int num_threads = std::min(num_eff_threads, (int)cluster_whole_ids.size());
            std::cout << "Cross Filter num_threads: " << num_cuda_devices << std::endl;
            cross_filter_thread_pool_.reset(new ThreadPool(num_cuda_devices));
// #ifdef CACHED_DEPTH_MAP
//                     normal_maps_.clear();
//                     normal_maps_.shrink_to_fit();
//                     std::cout << "Release NormalMaps: " << normal_maps_.capacity() << std::endl;
// #endif
            cluster_points_.clear();
            cluster_points_score_.clear();
            cluster_points_visibility_.clear();
            cluster_points_.resize(cluster_whole_ids.size());
            cluster_points_score_.resize(cluster_whole_ids.size());
            cluster_points_visibility_.resize(cluster_whole_ids.size());

            cross_fused_points_.clear();
            cross_fused_points_score_.clear();
            cross_fused_points_visibility_.clear();
            cross_fused_points_vis_weight_.clear();
            cross_fused_points_.shrink_to_fit();
            cross_fused_points_score_.shrink_to_fit();
            cross_fused_points_visibility_.shrink_to_fit();
            cross_fused_points_vis_weight_.shrink_to_fit();

            auto CrossFilterPerCuda = [&](
                // const std::vector<std::unordered_set<int >> &cluster_ref_ids,
                // const std::vector<std::unordered_set<int >> &cluster_whole_ids,
                const std::vector<int > &cluster_gpu_idx,
                const std::vector<std::vector<int> >& overlapping_images,
                PatchMatchOptions options, const std::string &workspace_path,
                const int cuda_idx, const int level, const bool prior_ply){
                for (int cross_cluster_id = 0; cross_cluster_id < cluster_whole_ids.size(); cross_cluster_id++){
                    if (cluster_gpu_idx.at(cross_cluster_id) == cuda_idx){
                        ProcessCrossFilter(cluster_problem_ids, 
                                            cluster_whole_ids,
                                            images_exclusive,
                                            cluster_gpu_idx,
                                            overlapping_images,
                                            iter_options,
                                            stereo_reconstruction_path,
                                            cross_cluster_id, 
                                            iter.level, 
                                            prior_ply, 
                                            iter.rgbd && !has_prior_depth_map);
                    }
                }
                
            };
            for (auto cuda_id : cross_filter_gpu_indices_){
                cross_filter_thread_pool_->AddTask( CrossFilterPerCuda, 
                                                    // cluster_problem_ids, 
                                                    // cluster_whole_ids,
                                                    cluster_gpu_idx,
                                                    overlapping_images,
                                                    iter_options,
                                                    stereo_reconstruction_path,
                                                    cuda_id, iter.level, prior_ply);
            }
            cross_filter_thread_pool_->Wait();
            if (iter.rgbd && !has_prior_depth_map){
                depth_maps_.swap(depth_maps_temp_);
                normal_maps_.swap(normal_maps_temp_);
                has_prior_depth_map = true;
            }

            {
                std::vector<DepthMap> depth_maps_temp;
                depth_maps_temp.resize(depth_maps_.size());
                std::vector<NormalMap> normal_maps_temp;
                normal_maps_temp.resize(normal_maps_.size());
                depth_maps_temp_.swap(depth_maps_temp);
                normal_maps_temp_.swap(normal_maps_temp);
            }

            if (options.patch_match_fusion){
                std::cout << "patch match fusion merge cluster points ..." << std::endl;
                for (int i = 0; i < cluster_whole_ids.size(); i++){
                    if (cluster_points_.at(i).empty()){
                        std::cout << "cluster_points " << i << " is empty!" << std::endl;
                        continue;
                    }  
                    cross_fused_points_.insert(
                        cross_fused_points_.end(), 
                        cluster_points_.at(i).begin(),
                        cluster_points_.at(i).end());
                    std::vector<PlyPoint> ().swap(cluster_points_.at(i));

                    cross_fused_points_score_.insert(
                        cross_fused_points_score_.end(), 
                        cluster_points_score_.at(i).begin(),
                        cluster_points_score_.at(i).end());
                    std::vector<float> ().swap(cluster_points_score_.at(i));

                    cross_fused_points_visibility_.insert(
                        cross_fused_points_visibility_.end(), 
                        cluster_points_visibility_.at(i).begin(),
                        cluster_points_visibility_.at(i).end());
                    std::vector<std::vector<uint32_t> >().swap(cluster_points_visibility_.at(i));
                }
                cluster_points_.clear();
                cluster_points_.shrink_to_fit();
                cluster_points_score_.clear();
                cluster_points_score_.shrink_to_fit();
                cluster_points_visibility_.clear();
                cluster_points_visibility_.shrink_to_fit();
                {
                    std::vector<std::vector<PlyPoint> >().swap(cluster_points_);
                    std::vector<std::vector<float> >().swap(cluster_points_score_);
                    std::vector<std::vector<std::vector<uint32_t> >>().swap(cluster_points_visibility_);
                }
            }

            std::cout << "Patch_Match CrossFilterPerCuda Elapsed time:" << filter_timer.ElapsedMinutes() << "[minutes]\n" << std::endl;

#else
            for (int problem_idx = 0; problem_idx < problems_.size(); 
                ++problem_idx) {
                const auto &depth_range = depth_ranges_[problem_idx];
                iter_options.depth_min = depth_range.first;
                iter_options.depth_max = depth_range.second;
                iter_options.geom_consistency = 1;
                iter_options.init_depth_random = 0;
                thread_pool_->AddTask(
                    &PatchMatchController::ProcessFilterProblem,
                    this,
                    iter_options,
                    problem_idx,
                    stereo_reconstruction_path,
                    iter.level, false);
            }
            thread_pool_->Wait();
#endif
            cross_elapsed_time += cross_timer.ElapsedMinutes();
        }
    }

    std::cout << StringPrintf("Cluster#%d Cross Filter Cost Time: %.3fmin\n", cluster_id, cross_elapsed_time);

    if (!prior_ply){
        float begin_memroy, end_memory;
        GetAvailableMemory(begin_memroy);

        problems_.clear();
        problems_.shrink_to_fit();
        // bitmaps_.clear();
        // bitmaps_.shrink_to_fit();
        semantic_maps_.clear();
        semantic_maps_.shrink_to_fit();
        depth_ranges_.clear();
        depth_ranges_.shrink_to_fit();
        depth_maps_.clear();
        depth_maps_.shrink_to_fit();
        normal_maps_.clear();
        normal_maps_.shrink_to_fit();
        mask_maps_.clear();
        mask_maps_.shrink_to_fit();
        depth_maps_temp_.clear();
        depth_maps_temp_.shrink_to_fit();
        normal_maps_temp_.clear();
        normal_maps_temp_.shrink_to_fit();
        used_images_.clear();
        src_2_refs_map_.clear();
        ref_2_srcs_map_.clear();
        ref_2_problem_map_.clear();

        malloc_trim(0);
        GetAvailableMemory(end_memory);
        std::cout << "Clear maps, begin / end freem-memory :" 
                << begin_memroy << " / " << end_memory 
                << "\t max-memory: " 
                << end_memory - begin_memroy << std::endl;
    } else {
        int num_maps = depth_maps_.size();
        depth_maps_.clear();
        normal_maps_.clear();

        depth_maps_.resize(num_maps);
        normal_maps_.resize(num_maps);
    }


    Timer fusion_timer;
    fusion_timer.Start();

    if (options.patch_match_fusion && options.filter && options.geo_filter){
        std::cout << "Cross-ori points size: " << cross_fused_points_.size() << std::endl;

        Timer fusion_timer;
        fusion_timer.Start();
        
        const Model& model_fusion = workspace_->GetModel();

        if (options.verbose && cross_fused_points_.size() > 0){
            const std::string ply_path = JoinPaths(dense_reconstruction_path, 
                "ori-" +std::to_string(cluster_id) + "-" + FUSION_NAME);
            WriteBinaryPlyPoints(ply_path, cross_fused_points_, true, true);
            WritePointsVisibility(ply_path+".vis", cross_fused_points_visibility_);
            WritePointsScore(ply_path+".sco", cross_fused_points_score_);
        }
        
        if (options.fused_delaunay_sample && !prior_ply && cross_fused_points_.size() > 0){
            float begin_memroy, end_memory;
            GetAvailableMemory(begin_memroy);

            PointsSample(cross_fused_points_, 
                        cross_fused_points_score_,
                        cross_fused_points_visibility_, 
                        cross_fused_points_vis_weight_, 
                        model_fusion, 
                        options_.fused_dist_insert, 
                        options_.fused_diff_depth);

            malloc_trim(0);
            GetAvailableMemory(end_memory);
            std::cout << cluster_id << "-Cluster " 
                << " PointsSample,  begin / end freem-memory :" 
                << begin_memroy << " / " << end_memory 
                << "\t max-memory: " 
                << end_memory - begin_memroy << std::endl;
        }

        if (options.outlier_removal && cross_fused_points_.size() > 0){
            DistRemoveOutlier(3);
        }

        std::vector<float> dists;
        float average_dist = -1.0f;

        if (options.outlier_removal && cross_fused_points_.size() > 0){
            for (int i = 0; i < cross_fused_points_.size(); i++){
                int num_view = cross_fused_points_visibility_.at(i).size();
                float &cross_score = cross_fused_points_score_.at(i);
                if (num_view > 2){
                    cross_score = 0.0f;
                }
            }

            std::vector<float> sum_weights(cross_fused_points_score_);
            std::sort(sum_weights.begin(), sum_weights.end());
            float nth2 = sum_weights.at(sum_weights.size() * 0.15);
            float nth3 = sum_weights.at(sum_weights.size() * 0.99);
            std::cout << "Remove outlier(score min_score: " << nth2 
                << " , max_score: " << nth3 << ") ..." << std::endl;

            if (nth3 - nth2 > 1e-6){
                ComputeAverageDistance(cross_fused_points_, dists, &average_dist, options_.nb_neighbors);

                float voxel_factor = options_.voxel_factor;
                if (options.fused_delaunay_sample && !prior_ply){
                    voxel_factor /= options.fused_dist_insert;
                }
                VoxelRemoveOutlier(average_dist, voxel_factor);

                voxel_factor *= 3;
                VoxelRemoveOutlier(average_dist, voxel_factor);

                for (int i = 0; i < cross_fused_points_.size(); i++){
                    int num_view = cross_fused_points_visibility_.at(i).size();
                    float &cross_score = cross_fused_points_score_.at(i);
                    cross_score = (float)std::min(std::max(cross_score - nth2, 0.0f), nth3 - nth2) / (float)(nth3 - nth2);
                }

                ScoreRemoveOutlier();
                cross_fused_points_score_.clear();
                cross_fused_points_score_.shrink_to_fit();

                DistRemoveOutlier(5);
            }
        }

        if (options.plane_optimization && cross_fused_points_.size() > 0){
            if (average_dist < 0){
                ComputeAverageDistance(cross_fused_points_, dists, &average_dist, options_.nb_neighbors);
            }
            PlaneScoreCompute(average_dist);
        }

        if (ExistsDir(semantic_maps_path) && !prior_ply && cross_fused_points_.size() > 0){
            ComputeSemanticLabel(cluster_image_map.at(cluster_id), semantic_maps_path);
        }

        int num_cross_fused_point = cross_fused_points_.size();
        if (!prior_ply){
            std::cout << "Write Points ..." << std::endl;
            const std::string ply_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
            bool has_sem = ExistsDir(semantic_maps_path);
            if (cluster_id == 0 && !options_.map_update){
                MutiThreadSaveCrossPointCloud(ply_path, has_sem, cross_fused_points_, 
                    cross_fused_points_visibility_,  cross_fused_points_vis_weight_, 
                    cross_fused_points_score_);
            } else {
                MutiThreadAppendCrossPointCloud(ply_path, has_sem, cross_fused_points_, 
                    cross_fused_points_visibility_,  cross_fused_points_vis_weight_, 
                    cross_fused_points_score_);
            }

            if (num_box_ > 1) {
                std::size_t num_points = cross_fused_points_.size();
                std::vector<std::set<int>> pnts_2_cluster(num_points);
                for (std::size_t i = 0; i < num_points; i++){
                    const auto& pnt = cross_fused_points_.at(i);
                    const Eigen::Vector3f xyz(pnt.x, pnt.y, pnt.z);
                    const Eigen::Vector3f rot_xyz = box_rot_ * xyz;
                    for (int j = 0; j < num_box_; j++){
                        if (rot_xyz(0) < roi_child_boxs_.at(j).x_box_min || 
                            rot_xyz(0) > roi_child_boxs_.at(j).x_box_max ||
                            rot_xyz(1) < roi_child_boxs_.at(j).y_box_min || 
                            rot_xyz(1) > roi_child_boxs_.at(j).y_box_max ||
                            rot_xyz(2) < roi_child_boxs_.at(j).z_box_min || 
                            rot_xyz(2) > roi_child_boxs_.at(j).z_box_max){
                            continue;
                        }
                        pnts_2_cluster.at(i).insert(j);
                        // num_points_per_cluster.at(j)++;
                    }
                }

                std::vector<std::size_t> num_points_per_cluster(num_box_, 0);
                std::vector<std::vector<std::size_t>> cluster_2_pnts(num_box_);
                for (std::size_t i = 0; i < num_points; i++){
                    for (const auto& box_id : pnts_2_cluster.at(i)){
                        cluster_2_pnts.at(box_id).push_back(i);
                        num_points_per_cluster.at(box_id)++;
                    }
                }

                 for (int i = 0; i < num_box_; i++){
                    std::cout << "num_points_per_cluster.at(i): " << num_points_per_cluster.at(i) << std::endl;
                 }
                bool has_weight = (cross_fused_points_.size() == cross_fused_points_vis_weight_.size());
                bool has_score = (cross_fused_points_.size() == cross_fused_points_score_.size());
                std::cout << "has_weight, has_score: " << has_weight << ", " << has_score << std::endl;

                std::vector<std::vector<PlyPoint>> cluster_fused_points(num_box_);
                std::vector<std::vector<std::vector<uint32_t> >> cluster_points_visibility(num_box_);
                std::vector<std::vector<std::vector<float> >> cluster_points_vis_weight(num_box_);
                std::vector<std::vector<float>> cluster_points_score(num_box_);
                for (int i = 0; i < num_box_; i++){
                    const std::string cluster_path = JoinPaths(
                            dense_reconstruction_path, "..", std::to_string(i), FUSION_NAME);
                    const std::string cluster_box_path = JoinPaths(
                            dense_reconstruction_path, "..", std::to_string(i), ROI_BOX_NAME);
                    CreateDirIfNotExists(GetParentDir(cluster_path));
                    if (!ExistsFile(cluster_box_path)){
                        WriteBoundBoxText(cluster_box_path, roi_child_boxs_.at(i));
                    }
                    // if (num_points_per_cluster.at(i) < 1){
                    //     continue;
                    // }
                    cluster_fused_points.at(i).reserve(num_points_per_cluster.at(i));
                    cluster_points_visibility.at(i).reserve(num_points_per_cluster.at(i));
                    if (has_weight){
                        cluster_points_vis_weight.at(i).reserve(num_points_per_cluster.at(i));
                    }
                    if (has_score){
                        cluster_points_score.at(i).reserve(num_points_per_cluster.at(i));
                    }

                    for (std::size_t j = 0; j < num_points_per_cluster.at(i); j++){
                        std::size_t pnt_id = cluster_2_pnts.at(i).at(j);
                        cluster_fused_points.at(i).push_back(std::move(cross_fused_points_.at(pnt_id)));
                        cluster_points_visibility.at(i).push_back(std::move(cross_fused_points_visibility_.at(pnt_id)));
                        if (has_weight){
                            cluster_points_vis_weight.at(i).push_back(std::move(cross_fused_points_vis_weight_.at(pnt_id)));
                        }
                        if (has_score){
                            cluster_points_score.at(i).push_back(std::move(cross_fused_points_score_.at(pnt_id)));
                        }
                    }

                    if (cluster_id == 0){
                        MutiThreadSaveCrossPointCloud(cluster_path, has_sem, cluster_fused_points.at(i), 
                            cluster_points_visibility.at(i),  cluster_points_vis_weight.at(i), 
                            cluster_points_score.at(i));
                    } else {
                        MutiThreadAppendCrossPointCloud(cluster_path, has_sem, cluster_fused_points.at(i), 
                            cluster_points_visibility.at(i),  cluster_points_vis_weight.at(i), 
                            cluster_points_score.at(i));
                    }
                }

            }

            num_save_fused_points_ += num_cross_fused_point;
            float begin_memroy, end_memory;
            GetAvailableMemory(begin_memroy);
            {
                 std::vector<PlyPoint>().swap(cross_fused_points_);
                 std::vector<std::vector<uint32_t> > ().swap(cross_fused_points_visibility_);
                 std::vector<std::vector<float> >().swap(cross_fused_points_vis_weight_);
                 std::vector<float >().swap(cross_fused_points_score_);
            }
            GetAvailableMemory(end_memory);
            std::cout << cluster_id << "-Cluster "<< "saved "   
                << num_cross_fused_point << "( "  
                << num_save_fused_points_ << " )"
                << " points,  begin / end freem-memory :" 
                << begin_memroy << " / " << end_memory 
                << "\t max-memory: " 
                << end_memory - begin_memroy << std::endl;
        }

        fusion_elapsed_time += fusion_timer.ElapsedMinutes();

        std::cout << StringPrintf("Cluster#%d Depth Fusion Cost Time: %.3fmin\n", cluster_id, fusion_elapsed_time);
    }

    depth_elapsed_time_ += depth_elapsed_time;
    cross_elapsed_time_ += cross_elapsed_time;
    fusion_elapsed_time_ += fusion_elapsed_time;

    bitmaps_.clear();
    bitmaps_.shrink_to_fit();

    malloc_trim(0);
}

void PatchMatchController::BuildPriorModel(const std::string& workspace_path) {
    if (options_.init_from_model) {
        ReadTriangleMeshObj(JoinPaths(workspace_path, DENSE_DIR, "model.obj"), 
                            prior_model_, false);
    } else if (options_.init_from_global_map) {
        std::string model_path = 
            JoinPaths(workspace_path, DENSE_DIR, "prior_model.obj");
        if (ExistsFile(model_path)) {
            ReadTriangleMeshObj(model_path, prior_model_, false);
            return;
        }

        mvs::PointCloud pointcloud;

        const Model& model = workspace_->GetModel();
        const auto& points = model.points;
        pointcloud.pointViews.Resize(points.size());
        pointcloud.pointWeights.Resize(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            const Eigen::Vector3f pt(points[i].x, points[i].y, points[i].z);
            const std::vector<int>& vis_pt = points[i].track;
            pointcloud.points.Insert(pt);
            pointcloud.colors.Insert(Eigen::Vector3ub(0, 0, 0));
            for (const auto& view : vis_pt) {                 
                pointcloud.pointViews[i].Insert(view);
                pointcloud.pointWeights[i].Insert(1.0f);
            }
        }

        auto options = options_.delaunay_options;

        mvs::DelaunayMeshing(prior_model_, pointcloud, model.images, 
                             options.sampInsert, options.dist_insert, options.diff_depth, 2.0f, -1.0f,
                             false, 4, 1, 1, 4, 3, 0.1, 1000, 400);

        // clean the mesh
        // prior_model_.Clean(1.0, 2.0, 1, 30, 10, false);
        prior_model_.Clean(options.decimate_mesh, options.remove_spurious, 
                           options.remove_spikes, options.close_holes, 
                           options.smooth_mesh, false);
        // extra cleaning trying to close more holes
        // prior_model_.Clean(1.f, 0.f, 1, 30, 0, false);
        prior_model_.Clean(1.f, 0.f, options.remove_spikes, options.close_holes, 
                           0, false);
        // extra cleaning to remove non-manifold problems created by closing holes.
        prior_model_.Clean(1.f, 0.f, false, 0, 0, true);

        prior_model_.RemoveIsolatedPieces(options.num_isolated_pieces);
        prior_model_.ComputeNormals();

        WriteTriangleMeshObj(
            JoinPaths(workspace_path, DENSE_DIR, "prior_model.obj"), 
            prior_model_, false);
    }
}

void ReadLidarPoints(const std::string lidar_path,
                     std::vector<Eigen::Vector3f> &lidar_points) {
    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    int32_t num = 1000000;
    // float *data = (float*)std::malloc(num*sizeof(float));
    std::vector<float> data(num);
    float* ptr = data.data();

    // pointers
    float *px = ptr + 0;
    float *py = ptr + 1;
    float *pz = ptr + 2;
    float *pr = ptr + 3;

    // load point cloud
    FILE *stream;
    stream = std::fopen(lidar_path.c_str(), "rb");
    num = std::fread(ptr, sizeof(float), num, stream) / 4;
    lidar_points.resize(num);
    for (int32_t i = 0; i < num; i++){
        Eigen::Vector4d point_v;
        lidar_points[i].x() = *px;
        lidar_points[i].y() = *py;
        lidar_points[i].z() = *pz;

        px+=4; py+=4; pz+=4; pr+=4;
    }
    fclose(stream);
}

void PatchMatchController::BuildLidarModel(const std::string& lidar_path) {
    if (!ExistsDir(lidar_path)) {
        return;
    }
    const Model& model = workspace_->GetModel();
    Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();
    {
        CHECK(ExistsFile(options_.lidar2cam_calibfile));
        Eigen::Matrix3x4d dTr;
        std::ifstream calib_file(options_.lidar2cam_calibfile, std::ios::in);
        calib_file >> dTr(0, 0) >> dTr(0, 1) >> dTr(0, 2) >> dTr(0, 3);
        calib_file >> dTr(1, 0) >> dTr(1, 1) >> dTr(1, 2) >> dTr(1, 3);
        calib_file >> dTr(2, 0) >> dTr(2, 1) >> dTr(2, 2) >> dTr(2, 3);
        calib_file.close();
        Tr.block<3, 4>(0, 0) = dTr.cast<float>();
    }

    const double voxel_size = options_.sample_radius_for_lidar;
    const int num_camera_param = 7;
    const std::string camera_rig_params = options_.camera_rig_params;
    std::vector<std::string> params = StringSplit(camera_rig_params, ",");
    int num_local_cameras = params.size() / num_camera_param;
    if (num_local_cameras == 0) num_local_cameras = 1;

    std::vector<Eigen::Matrix4f> lidar2cam(num_local_cameras);
    std::fill(lidar2cam.begin(), lidar2cam.end(), Tr);

    bool cam_rig = (num_local_cameras > 1);
    if (cam_rig) {
        for (int i = 0; i < num_local_cameras; ++i) {
            Eigen::Quaternionf qvec;
            qvec.w() = std::atof(params[i * num_camera_param].c_str());
            qvec.x() = std::atof(params[i * num_camera_param + 1].c_str());
            qvec.y() = std::atof(params[i * num_camera_param + 2].c_str());
            qvec.z() = std::atof(params[i * num_camera_param + 3].c_str());
            Eigen::Vector3f tvec;
            tvec[0] = std::atof(params[i * num_camera_param + 4].c_str());
            tvec[1] = std::atof(params[i * num_camera_param + 5].c_str());
            tvec[2] = std::atof(params[i * num_camera_param + 6].c_str());

            Eigen::Matrix4f local_extrinsic = Eigen::Matrix4f::Identity();
            local_extrinsic.block<3, 3>(0, 0) = qvec.toRotationMatrix();
            local_extrinsic.block<3, 1>(0, 3) = tvec;
            lidar2cam[i] = local_extrinsic * Tr;
        }
    }

    std::unordered_map<std::string, int> m_voxel_map_idx;
    std::vector<VertInfo> m_voxel_list;
    std::vector<std::shared_ptr<Bitmap> > bitmaps(model.images.size(), nullptr);

    int local_camera_idx = 0;
    for (size_t i = 0; i < model.images.size(); ++i) {
        const mvs::Image& image = model.images.at(i);
        const std::string image_name = model.GetImageName(i);
        size_t pos = image_name.find_last_of('.');
        std::string lidar_name = image_name.substr(0, pos) + ".bin";
        std::vector<std::string> vnames = StringSplit(lidar_name, "/");
        if (cam_rig) {
            std::string cam_idx = vnames[vnames.size() - 2];
            if (cam_idx.compare("cam0") == 0) {
                local_camera_idx = 0;
            } else if (cam_idx.compare("cam1") == 0) {
                local_camera_idx = 1;
            } else if (cam_idx.compare("cam2") == 0) {
                local_camera_idx = 2;
            } else if (cam_idx.compare("cam3") == 0) {
                local_camera_idx = 3;
            }
        }
        lidar_name = vnames.back();
        lidar_name = JoinPaths(lidar_path, lidar_name);
        if (!ExistsFile(lidar_name)) {
            continue;
        }
        std::cout << lidar_name << std::endl;

        bitmaps[i] = std::make_shared<Bitmap>();
        bitmaps[i]->Read(image.GetPath());

        const int width = image.GetWidth();
        const int height = image.GetHeight();

        const Eigen::RowMatrix3f K(image.GetK());
        const Eigen::RowMatrix3f R(image.GetR());
        const Eigen::Vector3f T(image.GetT());

        std::vector<Eigen::Vector3f> lidar_points_one;
        ReadLidarPoints(lidar_name, lidar_points_one);
        for (const auto& point : lidar_points_one) {
            if (point.norm() < 2.0) {
                continue;
            }
            Eigen::Vector3f TP = (lidar2cam[local_camera_idx] * point.homogeneous()).head<3>();
            Eigen::Vector3f proj = K * TP;
            int u = proj[0] / proj[2];
            int v = proj[1] / proj[2];
            if (proj[2] <= 0 || u < 0 || u >= width || v < 0 || v >= height) {
                continue;
            }
            TP.noalias() = R.transpose() * (TP - T);

            BitmapColor<uint8_t> color;
            bitmaps[i]->GetPixel(u, v, &color);

            int ix = TP.x() * voxel_size;
            int iy = TP.y() * voxel_size;
            int iz = TP.z() * voxel_size;
            std::string key("");
            key.append(std::to_string(ix));
            key.append(std::to_string(iy));
            key.append(std::to_string(iz));
            auto it = m_voxel_map_idx.find(key);
            if (it == m_voxel_map_idx.end()) {
                m_voxel_map_idx[key] = m_voxel_list.size();
                VertInfo vert_info;
                vert_info.X = TP;
                vert_info.color = Eigen::Vector3f(color.r, color.g, color.b);
                vert_info.view_ids.push_back(i);
                m_voxel_list.emplace_back(vert_info);
            } else {
                VertInfo& vert_info = m_voxel_list.at(it->second);
                float w = vert_info.view_ids.size();
                vert_info.X = (vert_info.X * w + TP) / (w + 1);
                vert_info.color = (vert_info.color * w +
                    Eigen::Vector3f(color.r, color.g, color.b)) / (w + 1);
                vert_info.view_ids.push_back(i);
            }
        }
    }
    lidar_samps_.clear();
    for (auto voxel : m_voxel_list) {
        if (voxel.view_ids.size() >= 3) {
            lidar_samps_.emplace_back(voxel);
        }
    }
    // lidar_samps_ = m_voxel_list;
    m_voxel_list.clear();
    m_voxel_map_idx.clear();
}

void PatchMatchController::ProcessProblem(PatchMatchOptions options,
                                          const int problem_idx,
                                          const std::string &dense_path,
                                          const std::string &workspace_path,
                                          const int level,
                                          const bool fill_flag,
                                          bool save_flag = true) {
    if (IsStopped()) {
        return;
    }

    const auto& model = workspace_->GetModel();
    Problem& problem = problems_.at(problem_idx);

    auto image_name = model.GetImageName(problem.ref_image_idx);

    PrintHeading1(StringPrintf("Reconstructing view %d / %d (%s, level = %d, progress: %.1f%)",
                               problem_idx + 1, problems_.size(), image_name.c_str(), level,
                               (float) 100 * (++num_processed_images_) / num_all_images_));

    // if (options.depth_min < 0 || options.depth_max < 0) {
    //     CHECK(options.depth_min > 0 &&
    //           options.depth_max > 0)
    //         << " - You must manually set the minimum and maximum depth, since no "
    //            "sparse model is provided in the workspace.";
    // }

    const auto& semantic_maps_path = JoinPaths(dense_path, SEMANTICS_DIR);
    const auto& depth_maps_path = JoinPaths(workspace_path, DEPTHS_DIR);
    const auto& normal_maps_path = JoinPaths(workspace_path, NORMALS_DIR);
    const auto& conf_maps_path = JoinPaths(workspace_path, CONFS_DIR);
    const auto& curvature_maps_path = JoinPaths(workspace_path, CURVATURES_DIR);
    const auto& consistency_maps_path = 
        JoinPaths(workspace_path, CONSISTENCY_DIR);

    const auto& wgt_maps_path = JoinPaths(dense_path, "intensity_maps");

    bool has_semantic = ExistsPath(semantic_maps_path);

    auto file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), options.output_type.c_str(), DEPTH_EXT);
    auto depth_map_path = JoinPaths(depth_maps_path, file_name);
    auto normal_map_path = JoinPaths(normal_maps_path, file_name);
    auto conf_map_path = JoinPaths(conf_maps_path, file_name);
    auto curvature_map_path = JoinPaths(curvature_maps_path, file_name);
    auto consistency_graph_path = JoinPaths(consistency_maps_path, file_name);
    if (!options.init_from_model && 
        ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
        (!options.write_consistency_graph ||
        ExistsFile(consistency_graph_path)) && options.pyramid_max_level == 0) {
        return;
    }

    std::vector<Image> images = model.images;

    problem.images = &images;
#ifndef CACHED_DEPTH_MAP
    std::vector<Bitmap> bitmaps_;
    std::vector<Bitmap> semantic_maps_;
    std::vector<DepthMap> depth_maps_;
    std::vector<NormalMap> normal_maps_;
    std::vector<Bitmap> mask_maps_;
    bitmaps_.resize(images.size());
    semantic_maps_.resize(images.size());
    depth_maps_.resize(images.size());
    normal_maps_.resize(images.size());
    mask_maps_.resize(images.size());
#endif
    problem.semantic_maps = &semantic_maps_;
    problem.mask_maps = &mask_maps_;
    problem.depth_maps = &depth_maps_;
    problem.normal_maps = &normal_maps_;
    problem.flag_depth_maps = &flag_depth_maps_;

    problem.prior_wgt_maps = &prior_wgt_maps_;
    Image image = model.images.at(problem.ref_image_idx);

    float factor_level = 1 / std::pow(2,level);
    float max_size = std::max(image.GetHeight(), image.GetWidth());
    float min_image_size = std::min(320.f, max_size);
    factor_level = std::max(factor_level, min_image_size/max_size);

    if (!options.geom_consistency) {
        image.Rescale(factor_level);
        if (!depth_maps_.at(problem.ref_image_idx).IsValid() ||
            !normal_maps_.at(problem.ref_image_idx).IsValid()){
            depth_maps_.at(problem.ref_image_idx) = DepthMap(image.GetWidth(), 
                image.GetHeight(), options.depth_min, options.depth_max);
            normal_maps_.at(problem.ref_image_idx) = NormalMap(image.GetWidth(),
                                                            image.GetHeight());
        }
        if (level != options.pyramid_max_level) {
            if (!depth_maps_.at(problem.ref_image_idx).IsValid() ||
                !normal_maps_.at(problem.ref_image_idx).IsValid()){
                // const std::string ref_depth_map_path = 
                //     workspace_->GetDepthMapPath(problem.ref_image_idx);
                // const std::string ref_normal_map_path = 
                //     workspace_->GetNormalMapPath(problem.ref_image_idx);
                auto src_file_name = workspace_->GetFileName(problem.ref_image_idx);
                const std::string ref_depth_map_path = JoinPaths(depth_maps_path, src_file_name);
                const std::string ref_normal_map_path = JoinPaths(normal_maps_path, src_file_name); 
                if (!ExistsFile(ref_depth_map_path) || 
                    !ExistsFile(ref_normal_map_path)) {
                    return;
                }
                depth_maps_.at(problem.ref_image_idx).Read(ref_depth_map_path);
                normal_maps_.at(problem.ref_image_idx).Read(ref_normal_map_path);
                // std::cout << "read depth: " << problem.ref_image_idx << std::endl;
            }
            if (!depth_maps_.at(problem.ref_image_idx).IsValid() ||
                !normal_maps_.at(problem.ref_image_idx).IsValid()) {
                return;
            }
        }else if (options.init_from_visible_map || options.init_from_rgbd) {
            std::cout << "InitDepthMap" << std::endl;
            std::pair<float, float> depth_range = InitDepthMap(
                options, problem_idx, image, dense_path, workspace_path);
            // depth_ranges_.at(problem_idx) = depth_range;
        } else if (options.init_from_global_map || options.init_from_model) {
            std::cout << "InitDepthMap From model" << std::endl;
            std::pair<float, float> depth_range = Model2Depth(
                prior_model_, image, problem);
            // depth_ranges_.at(problem_idx) = depth_range;
        } else if (options.init_from_dense_points){
            std::cout << "InitDepthMap From Dense Points" << std::endl;
            std::pair<float, float> depth_range = Ply2Depth(
                prior_points_, image, problem);
        }

#ifdef DEBUG_MODEL_PM
        std::string parent_path = GetParentDir(depth_map_path); 
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        depth_maps_.at(problem.ref_image_idx).ToBitmap().Write(depth_map_path + ".init.jpg");

        parent_path = GetParentDir(normal_map_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        normal_maps_.at(problem.ref_image_idx).ToBitmap().Write(normal_map_path + ".init.jpg");
#endif
    }

    //Rescale the depthmap
    // std::string image_name = model.GetImageName(problem.ref_image_idx);
    
    if (options.depth_min < 0 || options.depth_max < 0) {
        options.depth_min = depth_ranges_.at(problem_idx).first;
        options.depth_max = depth_ranges_.at(problem_idx).second;
        std::cout << options.depth_min << ", " 
                  << options.depth_max << std::endl;
        CHECK(options.depth_min >= 0 && options.depth_max > 0)
            << " - You must manually set the minimum and maximum depth, "
               "since no sparse model is provided in the workspace.";
    }

    {
        // Collect all used images in current problem.
        std::unordered_set<int> used_image_idxs(problem.src_image_idxs.begin(),
                                                problem.src_image_idxs.end());
        used_image_idxs.insert(problem.ref_image_idx);

        options.filter_min_num_consistent =
            std::min(static_cast<int>(used_image_idxs.size()) - 1,
                    options.filter_min_num_consistent);

        int used_image_num = 0;
        for (const auto image_idx : used_image_idxs) {
            if (used_images_.find(image_idx) == used_images_.end()) {
                continue;
            }              
            used_image_num ++;
        }
        if (used_image_num < 2){
            std::cout << StringPrintf(
                "WARNING: Ignoring image %s, because source images do not "
                "exist.", image_name.c_str())
            << std::endl;
            return;
        }
        std::unordered_set<int> invalid_image_idxs;
        std::cout << "Reading inputs..." << std::endl;  
        // set factor parameter
        for (const auto image_idx : used_image_idxs) {
            if (used_images_.find(image_idx) == used_images_.end()) {
                invalid_image_idxs.insert(image_idx);
                std::cout << "image_idx is not in used_images_" << std::endl;
                continue;
            }
#ifndef CACHED_DEPTH_MAP
            if (bitmaps_.at(image_idx).Width() != images.at(image_idx).GetWidth() ||
                bitmaps_.at(image_idx).Height() != images.at(image_idx).GetHeight()){
                Bitmap bitmap;
                if (!bitmap.Read(workspace_->GetBitmapPath(image_idx),
                    workspace_->GetOptions().image_as_rgb)) {
                    invalid_image_idxs.insert(image_idx);
                    continue;
                }
                bitmaps_.at(image_idx) = std::move(bitmap);
                
                if (options.refine_with_semantic && has_semantic) {
                    const std::string image_name = model.GetImageName(image_idx);
                    const auto semantic_name = JoinPaths(semantic_maps_path, image_name);
                    const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
                    if (ExistsFile(semantic_base_name + "png")) {
                        semantic_maps_.at(image_idx).Read(semantic_base_name + "png", false);
                    } else if (ExistsFile(semantic_base_name + "jpg")) {
                        semantic_maps_.at(image_idx).Read(semantic_base_name + "jpg", false);
                    } else if (ExistsFile(semantic_base_name + "JPG")) {
                        semantic_maps_.at(image_idx).Read(semantic_base_name + "JPG", false);
                    } else {
                        invalid_image_idxs.insert(image_idx);
                        continue;
                    }
                }
                if (boost::filesystem::exists(mask_maps_path) && mask_maps_[image_idx].Width() == 0) {
                    const std::string image_name = model.GetImageName(image_idx);
                    const auto & name_parts = StringSplit(image_name, ".");
                    std::string mask_path = JoinPaths(mask_maps_path, image_name.substr(0, image_name.size() - name_parts.back().size() - 1), ".png");
                    std::cout << "mask_path: " << mask_path << std::endl;
                    mask_maps_.at(image_idx).Read(mask_path);
                }
            }
#endif  
            if (bitmaps_.at(image_idx).Width() != images.at(image_idx).GetWidth() ||
                bitmaps_.at(image_idx).Height() != images.at(image_idx).GetHeight()){
                std::cout << "crash bug: " << image_idx << " " << model.GetImageName(image_idx) << std::endl;
                ExceptionHandler(COLLAPSED_IMAGE_DIMENSION, 
                    JoinPaths(GetParentDir(workspace_path_), "errors/dense"), "DensePatchMatch").Dump();
                exit(COLLAPSED_IMAGE_DIMENSION);
                // continue;
            }
            images.at(image_idx).SetBitmap(bitmaps_.at(image_idx));
            // Bitmap bitmap;
            // if (!bitmap.Read(workspace_->GetBitmapPath(image_idx),workspace_->GetOptions().image_as_rgb)) {
            //     invalid_image_idxs.insert(image_idx);
            //     continue;
            // }
            // images.at(image_idx).SetBitmap(bitmap);

            if (options.geom_consistency) {
                if (!depth_maps_.at(image_idx).IsValid() ||
                    !normal_maps_.at(image_idx).IsValid()){
                    std::cout << "depth_maps_, normal_maps_ is valid!" << std::endl;
                    // const std::string src_depth_map_path = 
                    //     workspace_->GetDepthMapPath(image_idx);
                    // const std::string src_normal_map_path = 
                    //     workspace_->GetNormalMapPath(image_idx);
                    auto src_file_name = workspace_->GetFileName(image_idx);
                    const std::string src_depth_map_path = JoinPaths(depth_maps_path, src_file_name);
                    const std::string src_normal_map_path = JoinPaths(normal_maps_path, src_file_name); 
                    if (!ExistsFile(src_depth_map_path) || 
                        !ExistsFile(src_normal_map_path)) {
                        invalid_image_idxs.insert(image_idx);
                        continue;
                    }
                    depth_maps_.at(image_idx).Read(src_depth_map_path);
                    normal_maps_.at(image_idx).Read(src_normal_map_path);
                    // std::cout << "read depth: " << image_idx << std::endl;
                    if (!depth_maps_.at(image_idx).IsValid() ||
                        !normal_maps_.at(image_idx).IsValid()) {
                        invalid_image_idxs.insert(image_idx);
                        continue;
                    }                    
                }
            } else if (options.propagate_depth && problem.ref_image_idx != image_idx) {
                auto src_file_name = workspace_->GetFileName(image_idx);
                const std::string src_depth_map_path = JoinPaths(depth_maps_path, src_file_name);
                const std::string src_normal_map_path = JoinPaths(normal_maps_path, src_file_name);
                if (ExistsFile(src_depth_map_path)) {
                    depth_maps_.at(image_idx).Read(src_depth_map_path);
                    flag_depth_maps_.at(image_idx) = true;
                }
                // if (ExistsFile(src_normal_map_path)) {
                //     normal_maps_.at(image_idx).Read(src_normal_map_path);
                // }
                // flag_depth_maps_.at(image_idx) = true;
            }

            images.at(image_idx).Rescale(factor_level);
        }
        if (invalid_image_idxs.find(problem.ref_image_idx) != 
            invalid_image_idxs.end()) {
            std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because refer image is do not "
                    "be used.", image_name.c_str())
                      << std::endl;
            return;
        }

        // Remove invalid image idx.
        int i, j;
        for (i = 0, j = 0; i < problem.src_image_idxs.size(); ++i) {
            const int image_idx = problem.src_image_idxs.at(i);
            if (invalid_image_idxs.find(image_idx) == 
                invalid_image_idxs.end()) {
                problem.src_image_idxs[j] = problem.src_image_idxs[i];
                problem.src_image_scales[j] = problem.src_image_scales[i];
                j = j + 1;
            }
        }
        problem.src_image_idxs.resize(j);
        problem.src_image_scales.resize(j);
        if (problem.src_image_idxs.size() == 0) {
            std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because source images do not "
                    "exist.", image_name.c_str())
                      << std::endl;
            return;
        }
    }

//    problem.Print();
//    options.Print();
    if (prior_wgt_maps_.at(problem.ref_image_idx).IsValid()){
        std::cout << "Problem prior_wgt_maps size: " << prior_wgt_maps_.at(problem.ref_image_idx).GetHeight() 
            << " * " << prior_wgt_maps_.at(problem.ref_image_idx).GetWidth() << std::endl;
    }

#ifndef CUDA_ENABLED
    options.use_gpu = false;
#endif

    DepthMap ref_depth_map;
    NormalMap ref_normal_map;
    Mat<unsigned short> ref_curvature_map;
#ifdef SAVE_CONF_MAP
    Mat<float> ref_conf_map;
#endif

    const int gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());
    CHECK_GE(gpu_index, -1);
    int thread_idx = 0;
    for (int i = 0; i < threads_per_gpu_.size(); ++i) {
        if (i >= gpu_index) {
            break;
        }
        thread_idx += threads_per_gpu_.at(i);
    }
    options.gpu_index = std::to_string(gpu_index);
    options.thread_index = thread_pool_->GetThreadIndex() - thread_idx;

    // options.Print();
    problem.Print();

    PatchMatchACMMCuda patch_match_cuda(options, problem);
    patch_match_cuda.Run();

    ref_depth_map = patch_match_cuda.GetDepthMap();
    ref_normal_map = patch_match_cuda.GetNormalMap();
    if (options.est_curvature && options.geom_consistency) {
        ref_curvature_map = patch_match_cuda.GetCurvatureMap();
    }
#ifdef SAVE_CONF_MAP
    ref_conf_map = patch_match_cuda.GetConfMap();
#endif

    if (!ref_depth_map.IsValid() || !ref_normal_map.IsValid()) {
        return;
    }

    if (fill_flag){
        RefineDepthAndNormalMap(options, workspace_path, problem.ref_image_idx,
            ref_depth_map, ref_normal_map, level);
    }

#ifndef CACHED_DEPTH_MAP
    save_flag = true;
#endif
    if (save_flag){
        std::string parent_path = GetParentDir(depth_map_path);
        if (!(options.geo_filter && options.filter)){
            if (!ExistsPath(parent_path)) {
                boost::filesystem::create_directories(parent_path);
            }
            ref_depth_map.Write(depth_map_path);
            if (options.verbose) {
                ref_depth_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", 
                                                            depth_map_path.c_str(), 
                                                            level));
            }
        }

        parent_path = GetParentDir(normal_map_path);
        if (!(options.geo_filter && options.filter)){       
            if (!ExistsPath(parent_path)) {
                boost::filesystem::create_directories(parent_path);
            }
            ref_normal_map.Write(normal_map_path);
            if (options.verbose) {
                ref_normal_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", 
                                                            normal_map_path.c_str(), 
                                                            level));
            }
        }

        if (options.write_consistency_graph) {
            // patch_match.GetConsistencyGraph().Write(consistency_graph_path);
        }

        if (options.est_curvature && options.geom_consistency) {
            parent_path = GetParentDir(curvature_map_path);
            if (!ExistsPath(parent_path)) {
                boost::filesystem::create_directories(parent_path);
            }
            ref_curvature_map.Write(curvature_map_path);
            if (options.verbose) {
                Mat<float> cmap = Mat<float>(ref_curvature_map.GetWidth(), ref_curvature_map.GetHeight(), 1);
                Mat<float> gmap = Mat<float>(ref_curvature_map.GetWidth(), ref_curvature_map.GetHeight(), 1);
                for (int r = 0; r < cmap.GetHeight(); ++r) {
                    for (int c = 0; c < cmap.GetWidth(); ++c) {
                        float curv = ref_curvature_map.Get(r, c, 1) / 10000.f;
                        float grad = ref_curvature_map.Get(r, c, 0) / 10000.f;
                        cmap.Set(r, c, curv);
                        gmap.Set(r, c, grad);
                    }
                }
                DepthMap(cmap, -1, -1).ToBitmap().Write(
                    StringPrintf("%s-%d-curv.jpg", curvature_map_path.c_str(), level));
                DepthMap(gmap, -1, -1).ToBitmap().Write(
                    StringPrintf("%s-%d-grad.jpg", curvature_map_path.c_str(), level));
            }
        }

    #ifdef SAVE_CONF_MAP
        parent_path = GetParentDir(conf_map_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        ref_conf_map.Write(conf_map_path);

        // FILE *fp = fopen(JoinPaths(conf_map_path + ".txt").c_str(), "w");
        // for (int i = 0; i < ref_conf_map.GetHeight(); ++i) {
        //     for (int j = 0; j < ref_conf_map.GetWidth(); ++j) {
        //         fprintf(fp, "%f ", ref_conf_map.Get(i, j));
        //     }
        //     fprintf(fp, "\n");
        // }
        // fclose(fp);

        if (options.verbose) {
            DepthMap(ref_conf_map, -1, -1).ToBitmap().Write(
                StringPrintf("%s-%d.jpg", conf_map_path.c_str(), level));
        }
    #endif

    }
#ifdef DEBUG_MODEL_PM
    std::string parent_path = GetParentDir(depth_map_path); 
    if (!ExistsPath(parent_path)) {
        boost::filesystem::create_directories(parent_path);
    }
    ref_depth_map.ToBitmap().Write(depth_map_path + ".ori.jpg");
    
    parent_path = GetParentDir(normal_map_path);
    if (!ExistsPath(parent_path)) {
        boost::filesystem::create_directories(parent_path);
    }
    ref_normal_map.ToBitmap().Write(normal_map_path + ".ori.jpg");
#endif

#ifdef CACHED_DEPTH_MAP
    if (options.geom_consistency){
        if (!save_flag){
            depth_maps_temp_.at(problem.ref_image_idx) = std::move(ref_depth_map);
            normal_maps_temp_.at(problem.ref_image_idx) = std::move(ref_normal_map);
        } else if (options.geo_filter && options.filter){
            depth_maps_temp_.at(problem.ref_image_idx) = std::move(ref_depth_map);
            normal_maps_temp_.at(problem.ref_image_idx) = std::move(ref_normal_map);
        }
    } else {
        depth_maps_.at(problem.ref_image_idx) = std::move(ref_depth_map);
        normal_maps_.at(problem.ref_image_idx) = std::move(ref_normal_map);
        flag_depth_maps_.at(problem.ref_image_idx) = true;
    }
#endif
}

void PatchMatchController::ProcessFilterProblem (
    PatchMatchOptions options,
    const int problem_idx,
    const std::string &workspace_path,
    const int level, 
    const bool deduplication_flag) {
    if (IsStopped()) {
        return;
    }

    PrintHeading1(StringPrintf("Reconstructing view %d / %d (level = %d)",
                               problem_idx + 1, problems_.size(), level));

    // if (options.depth_min < 0 || options.depth_max < 0) {
    //     CHECK(options.depth_min > 0 &&
    //           options.depth_max > 0)
    //         << " - You must manually set the minimum and maximum depth, since no "
    //            "sparse model is provided in the workspace.";
    // }

    const auto& model = workspace_->GetModel();

    Problem& problem = problems_.at(problem_idx);

    const auto& depth_maps_path = JoinPaths(workspace_path, DEPTHS_DIR);
    // const auto& normal_maps_path = JoinPaths(workspace_path, NORMALS_DIR);
    // const auto& conf_maps_path = JoinPaths(workspace_path, CONFS_DIR);
    // const auto& consistency_maps_path = 
    //     JoinPaths(workspace_path, CONSISTENCY_DIR);

    const auto &image_name = model.GetImageName(problem.ref_image_idx);
    const std::string file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), options.output_type.c_str(), DEPTH_EXT);
    const std::string depth_map_path = JoinPaths(depth_maps_path, file_name);
    // const std::string normal_map_path = JoinPaths(normal_maps_path, file_name);
    // const std::string conf_map_path = JoinPaths(conf_maps_path, file_name);
    // const std::string consistency_graph_path = 
    //     JoinPaths(consistency_maps_path, file_name);
    // if (!options_.init_from_model && 
    //     ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
    //     (!options.write_consistency_graph ||
    //     ExistsFile(consistency_graph_path))) {
    //     return;
    // }

    std::vector<Image> images = model.images;
    problem.images = &images;
#ifndef CACHED_DEPTH_MAP
    std::vector<DepthMap> depth_maps_;
    // std::vector<NormalMap> normal_maps_;
    // std::vector<Mat<float> > conf_maps_;
    depth_maps_.resize(images.size());
    // normal_maps_.resize(images.size());
    // conf_maps_.resize(images.size());
#endif
    problem.depth_maps = &depth_maps_;
    // problem.normal_maps = &normal_maps_;
    // problem.conf_maps = &conf_maps_;

    float factor_level = 1 / std::pow(2,level);
    float max_size = std::max(images.at(problem.ref_image_idx).GetHeight(),
                                    images.at(problem.ref_image_idx).GetWidth());
    float min_image_size = std::min(320.f, max_size);
    factor_level = std::max(factor_level, min_image_size/max_size);
    
    if (deduplication_flag){
        {
            std::unique_lock<std::mutex> lock(deduplication_mutex_);
            locked_images_.insert(problem.ref_image_idx);
        }
        int i, j;
        for (int i = 0, j = 0;  i < problem.src_image_idxs.size(); ++i){
            const int image_idx = problem.src_image_idxs.at(i);
            if (locked_images_.find(image_idx) == locked_images_.end()){
                problem.src_image_idxs[j] = problem.src_image_idxs[i];
                problem.src_image_scales[j] = problem.src_image_scales[i];
                j = j + 1;
            } else {
                std::cout << "locked_images_: " << image_idx << " / " << problem.ref_image_idx << std::endl;
            }
        }
    }

    {
        // Collect all used images in current problem.
        std::unordered_set<int> used_image_idxs(problem.src_image_idxs.begin(),
                                                problem.src_image_idxs.end());
        used_image_idxs.insert(problem.ref_image_idx);

        options.filter_min_num_consistent =
            std::min(static_cast<int>(used_image_idxs.size()) - 1,
                    options.filter_min_num_consistent);

        int used_image_num = 0;
        for (const auto image_idx : used_image_idxs) {
            if (used_images_.find(image_idx) == used_images_.end()) {
                continue;
            }              
            used_image_num ++;
        }
        if (used_image_num < 2){
            std::cout << StringPrintf(
                "WARNING: Ignoring image %s, because source images do not "
                "exist.", image_name.c_str())
            << std::endl;
            return;
        }
#ifdef CACHED_DEPTH_MAP
        // Only access workspace from one thread at a time and only spawn 
        // resample threads from one master thread at a time.
        std::unique_lock<std::mutex> lock(workspace_mutex_);
#endif
        std::unordered_set<int> invalid_image_idxs;
        std::cout << "Reading inputs..." << std::endl;  

        // set factor parameter
        // const float factor_level = 1 / std::pow(2,level);
        for (const auto image_idx : used_image_idxs) {
            if (used_images_.find(image_idx) == used_images_.end()) {
                invalid_image_idxs.insert(image_idx);
                continue;
            }
// #ifdef CACHED_DEPTH_MAP
//             images.at(image_idx).SetBitmap(workspace_->GetBitmap(image_idx));
// #else
//             Bitmap bitmap;
//             if (!bitmap.Read(workspace_->GetBitmapPath(image_idx),workspace_->GetOptions().image_as_rgb)) {
//                 invalid_image_idxs.insert(image_idx);
//                 continue;
//             }
//             images.at(image_idx).SetBitmap(bitmap);
// #endif
            // const std::string src_depth_map_path = 
            //     workspace_->GetDepthMapPath(image_idx);
            auto src_file_name = workspace_->GetFileName(image_idx);
            const std::string src_depth_map_path = JoinPaths(depth_maps_path, src_file_name);
            if (!ExistsFile(src_depth_map_path)) {
                invalid_image_idxs.insert(image_idx);
                continue;
            }
            
            depth_maps_.at(image_idx).Read(src_depth_map_path);
            // normal_maps_.at(image_idx).Read(src_normal_map_path);
            if (!depth_maps_.at(image_idx).IsValid()
            //  ||
            //     !normal_maps_.at(image_idx).IsValid()
                ) {
                invalid_image_idxs.insert(image_idx);
                continue;
            }

            images.at(image_idx).Rescale(factor_level);
        }

        if (invalid_image_idxs.find(problem.ref_image_idx) != 
            invalid_image_idxs.end()) {
            std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because refer image is do not "
                    "be used.", image_name.c_str())
                      << std::endl;
            return;
        }

        // Remove invalid image idx.
        int i, j;
        for (i = 0, j = 0; i < problem.src_image_idxs.size(); ++i) {
            const int image_idx = problem.src_image_idxs.at(i);
            if (invalid_image_idxs.find(image_idx) == 
                invalid_image_idxs.end()) {
                problem.src_image_idxs[j] = problem.src_image_idxs[i];
                problem.src_image_scales[j] = problem.src_image_scales[i];
                j = j + 1;
            }
        }
        problem.src_image_idxs.resize(j);
        problem.src_image_scales.resize(j);
        if (problem.src_image_idxs.size() == 0) {
            std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because source images do not "
                    "exist.", image_name.c_str())
                      << std::endl;
            return;
        }
    }

//    problem.Print();
//    patch_match_options.Print();

#ifndef CUDA_ENABLED
    options.use_gpu = false;
#endif

    DepthMap ref_depth_map;
    // NormalMap ref_normal_map;
    // Mat<float> ref_conf_map;

    // if (options.use_gpu) {
        int gpu_index;        
        
        gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());
        CHECK_GE(gpu_index, -1);
        // options.gpu_index = std::to_string(gpu_index);
        // options.thread_index = thread_pool_->GetThreadIndex() % THREADS_PER_GPU;
        int thread_idx = 0;
        for (int i = 0; i < threads_per_gpu_.size(); ++i) {
            if (i >= gpu_index) {
                break;
            }
            thread_idx += threads_per_gpu_.at(i);
        }
        options.gpu_index = std::to_string(gpu_index);
        options.thread_index = thread_pool_->GetThreadIndex() - thread_idx;

		// options.Print();
        problem.Print();
        
        PatchMatchACMMCuda patch_match_cuda(options, problem);
        patch_match_cuda.RunFilter(deduplication_flag);

        ref_depth_map = patch_match_cuda.GetDepthMap();
        // ref_normal_map = patch_match_cuda.GetNormalMap();
        // ref_conf_map = patch_match_cuda.GetConfMap();
    // }
    // else {
	// 	options.Print();
    //     problem.Print();

    //     std::cout << "PatchMatchACMM" << std::endl;

    //     PatchMatchACMM patch_match(options, problem);
    //     patch_match.RunFilter();

    //     ref_depth_map = patch_match.GetDepthMap();
    //     ref_normal_map = patch_match.GetNormalMap();
    //     ref_conf_map = patch_match.GetConfMap();
    // }

    if (!ref_depth_map.IsValid()/* || !ref_normal_map.IsValid()*/) {
        return;
    }
    std::string parent_path = GetParentDir(depth_map_path);
    if (!ExistsPath(parent_path)) {
        boost::filesystem::create_directories(parent_path);
    }
    ref_depth_map.Write(depth_map_path);

#ifdef DEBUG_MODEL_PM 
        ref_depth_map.ToBitmap().Write(StringPrintf("%s-%d-filter.jpg", 
                                                    depth_map_path.c_str(), 
                                                    level));
#endif

    // parent_path = GetParentDir(normal_map_path);
    // if (!ExistsPath(parent_path)) {
    //     boost::filesystem::create_directories(parent_path);
    // }
    // ref_normal_map.Write(normal_map_path);
    // if (options.verbose) {
    //     ref_normal_map.ToBitmap().Write(StringPrintf("%s-%d-filter.jpg", 
    //                                                  normal_map_path.c_str(), 
    //                                                  level));
    // }
    // if (options.write_consistency_graph) {
    //     // patch_match.GetConsistencyGraph().Write(consistency_graph_path);
    // }

    //     parent_path = GetParentDir(conf_map_path);
    //     if (!ExistsPath(parent_path)) {
    //         boost::filesystem::create_directories(parent_path);
    //     }
    //     ref_conf_map.Write(conf_map_path);
    //     if (options.verbose) {
    //         DepthMap(ref_conf_map, -1, -1).ToBitmap().Write(
    //             StringPrintf("%s-%d-filter.jpg", conf_map_path.c_str(), level));
    //     }
    if (deduplication_flag){
        std::unique_lock<std::mutex> lock(deduplication_mutex_);
        locked_images_.erase(problem.ref_image_idx);
    }
}

int PatchMatchController::CrossFilterImageCluster(
    std::vector<std::unordered_set<int >> &cluster_problem_ids,
    std::vector<std::unordered_set<int >> &cluster_whole_ids,
    std::unordered_map<image_t, bool>& images_exclusive,
    std::vector<int >& cluster_gpu_idx,
    float gpu_memory_factor){
    PrintHeading1("Cross Filter Image Cluster...");
    int max_height = -1, max_width = -1;
    const Model& model = workspace_->GetModel();
    Eigen::Vector3f image_center(0, 0, 0);
    std::cout << " => problems_.size(): " << problems_.size() << std::endl;
    for (size_t i = 0; i < problems_.size(); ++i){
        const mvs::Image& image = model.images.at(problems_[i].ref_image_idx);
        if (max_height < (int)image.GetHeight()){
            max_height = (int)image.GetHeight();
        }
        if (max_width < (int)image.GetWidth()){
            max_width = (int)image.GetWidth();
        }
        const float *C = image.GetC();
        const Eigen::Vector3f image_c(C[0], C[1], C[2]);
        image_center += image_c;
    }
    image_center = image_center / problems_.size();

    int image_memory = max_height * max_width * (sizeof(float) * 2 + sizeof(unsigned int));
    const uint64_t G_byte = 1.0e9;

    std::vector<uint64_t > num_images_per_gpu(max_gpu_memory_array_.size(), 0);
    std::vector<uint64_t> vec_max_image_num(max_gpu_memory_array_.size(), 0);
    std::cout << " => gpu max number images: ";
    for (const auto gpu_idx : cross_filter_gpu_indices_) {
        vec_max_image_num[gpu_idx] = std::min(
                (uint64_t)(G_byte * (max_gpu_memory_array_[gpu_idx] * gpu_memory_factor / image_memory)), 
                 (uint64_t)max_gpu_texture_layered_2_[gpu_idx]);
        std::cout << gpu_idx << " - " << vec_max_image_num[gpu_idx] << ",  ";
    }
    std::cout << std::endl;
    std::unordered_map<int, float> src_scores;
    
    int init_problem_id = -1;
    int init_image_id = -1;
    float max_distance = -1;
    for (size_t i = 0; i < problems_.size(); ++i){
        src_scores[i] = 0;
        const float *C = model.images[problems_[i].ref_image_idx].GetC();
        const Eigen::Vector3f image_C(C[0], C[1], C[2]);
        float distance = (image_C - image_center).norm();
        
        if (distance > max_distance){
            init_problem_id = i;
            init_image_id = problems_[i].ref_image_idx;
            max_distance = (Eigen::Vector3f(model.images[problems_[i].ref_image_idx].GetC()) - image_center).norm();
        }
    }

    std::unordered_set<int > problem_ids;
    std::unordered_set<int > whole_ids;
    int next_prob_id = init_problem_id;
    int next_ref_id = init_image_id;

    std::cout << "cross Filter Cluster: (eg: cluster id =>  gpu_idx, image_size, nex_gpu_idx)" << std::endl;
    int next_gpu_idx = cross_filter_gpu_indices_.at(0);
    float max_num = 0;
    for (auto gup_id : cross_filter_gpu_indices_){
        if (max_num < vec_max_image_num[gup_id]){
            max_num = vec_max_image_num[gup_id];
            next_gpu_idx = gup_id;
        }
    }

    while (next_ref_id >= 0){
        problem_ids.insert(next_prob_id);
        whole_ids.insert(next_ref_id);
        src_scores.erase(next_prob_id);

        for (const auto src_id : problems_[next_prob_id].src_image_idxs){
            if (whole_ids.find(src_id) == whole_ids.end()){
                whole_ids.insert(src_id);
            }
            for (const auto connect_ref_id : src_2_refs_map_[src_id]){
                if (src_scores.find(ref_2_problem_map_[connect_ref_id]) != src_scores.end()){
                    src_scores[ref_2_problem_map_[connect_ref_id]] ++;
                }
            }
        }

        float max_score = -1;
        next_ref_id = -1;
        for (auto src_score : src_scores){
            if (src_score.second > max_score){
                max_score = src_score.second;
                next_prob_id = src_score.first;
                next_ref_id = problems_[next_prob_id].ref_image_idx;
            }
        }

        int next_whole_size = whole_ids.size();        
        {
            if (whole_ids.find(next_ref_id) == whole_ids.end()){
                    next_whole_size++;
            }
            for (const auto src_id : problems_[next_prob_id].src_image_idxs){
                if (whole_ids.find(src_id) == whole_ids.end()){
                    next_whole_size++;
                }
            }
        }
        if (next_whole_size >= vec_max_image_num.at(next_gpu_idx) || src_scores.empty()){
            cluster_problem_ids.push_back(problem_ids);
            cluster_whole_ids.push_back(whole_ids);
            cluster_gpu_idx.push_back(next_gpu_idx);
            num_images_per_gpu[next_gpu_idx] += cluster_problem_ids.size();
            problem_ids.clear();
            whole_ids.clear();

            // compute next gpu
            float min_load_score = std::numeric_limits<float>::max();
            for (const auto gpu_idx : cross_filter_gpu_indices_){
                if (vec_max_image_num[gpu_idx] < options_.max_num_src_images + 1){
                    continue;
                }
                float load_score = (float ) num_images_per_gpu[gpu_idx] / (float) max_gpu_cudacore_array_[gpu_idx];
                if (load_score < min_load_score){
                    min_load_score = load_score;
                    next_gpu_idx = gpu_idx;
                }
            }

            std::cout  << "\t" << cluster_problem_ids.size() << "-cluster  => " << cluster_gpu_idx.back() << ", "
                << cluster_whole_ids.back().size() << ", " << next_gpu_idx << "(" << min_load_score << ") " << std::endl;
        }
    }

#if 0
    int num_cluster_ply = cluster_whole_ids.size();
    cv::RNG rng(12345);
    std::vector<PlyPoint> locations;
    std::vector<PlyPoint> common;

    vector<Eigen::Vector3i> color_k(num_cluster_ply);
    for (size_t i = 0; i < num_cluster_ply; i++) {
        int r_color = rng.uniform(0, 255);
        int g_color = rng.uniform(0, 255);
        int b_color = rng.uniform(0, 255);

        color_k[i][0] = r_color;
        color_k[i][1] = g_color;
        color_k[i][2] = b_color;

        for (const auto & image_idx : cluster_whole_ids.at(i)) {
            PlyPoint point;
            point.x = float(model.images[image_idx].GetC()[0]);
            point.y = float(model.images[image_idx].GetC()[1]);
            point.z = float(model.images[image_idx].GetC()[2]);
            point.r = r_color;
            point.g = g_color;
            point.b = b_color;
            Eigen::Vector3f cam_ray(model.images[image_idx].GetViewingDirection());
            point.nx = cam_ray.x();
            point.ny = cam_ray.y();
            point.nz = cam_ray.z();
            locations.emplace_back(point);
        }

        for (const auto & image_idx : cluster_ref_ids.at(i)) {
            PlyPoint point;
            point.x = float(model.images[image_idx].GetC()[0]);
            point.y = float(model.images[image_idx].GetC()[1]);
            point.z = float(model.images[image_idx].GetC()[2]);
            point.r = r_color;
            point.g = g_color;
            point.b = b_color;
            Eigen::Vector3f cam_ray(model.images[image_idx].GetViewingDirection());
            point.nx = cam_ray.x();
            point.ny = cam_ray.y();
            point.nz = cam_ray.z();
            common.emplace_back(point);
        }
    }

    std::string SavePlyPath = "./locations.ply";
    std::string SavePlyPath_common = "./locations_common.ply";
    sensemap::WriteBinaryPlyPoints(SavePlyPath, locations, false, true);
    sensemap::WriteBinaryPlyPoints(SavePlyPath_common, common, false, true);
    std::cout << "WriteBinaryPlyPoints :" << SavePlyPath << std::endl;
#endif
    std::cout << "......" << std::endl;
    size_t num_exclusive = 0;
    for (size_t i = 0; i < problems_.size(); ++i){
        auto image_id = problems_[i].ref_image_idx;
        int num_cluster_map_image = 0;
        for (int j = 0; j < cluster_whole_ids.size(); j++){
            if (cluster_whole_ids.at(j).find(image_id) != cluster_whole_ids.at(j).end()){
                num_cluster_map_image++;
            }
        }
        bool is_exclusive = num_cluster_map_image > 1 ? 0 : 1;
        images_exclusive[image_id] = is_exclusive;
        if (is_exclusive){
            num_exclusive++;
        }
    }
    std::cout << "num_exclusive: " << num_exclusive << "/" << problems_.size() << std::endl;
    // exit(-1);
    return cluster_whole_ids.size();
};

void PatchMatchController::ProcessCrossFilter(
    const std::vector<std::unordered_set<int >> &cluster_problem_ids,
    const std::vector<std::unordered_set<int >> &cluster_whole_ids,
    const std::unordered_map<image_t, bool>& images_exclusive,
    const std::vector<int > &cluster_gpu_idx,
    const std::vector<std::vector<int> >& overlapping_images,
    PatchMatchOptions options, const std::string &workspace_path,
    const int cluster_id, const int level, const bool prior_ply = false, 
    const bool process_rgbd_prior = false){
    if (IsStopped()) {
        return;
    }
    PrintHeading1(StringPrintf("CrossFilter cluster %d / %d (level = %d), cuda_id: %d",
                               cluster_id + 1, cluster_whole_ids.size(), level, (int)cluster_gpu_idx.at(cluster_id)));

    std::vector<int > whole_ids, problem_ids;
    std::vector<bool > ref_flags;
    {
        problem_ids.insert(problem_ids.end(), cluster_problem_ids.at(cluster_id).begin(), 
                        cluster_problem_ids.at(cluster_id).end());
        std::sort(problem_ids.begin(), problem_ids.end());

        whole_ids.insert(whole_ids.end(), cluster_whole_ids.at(cluster_id).begin(), 
                        cluster_whole_ids.at(cluster_id).end());
        std::sort(whole_ids.begin(), whole_ids.end());

        std::vector<int > ref_ids;
        for (const auto& problem_id : cluster_problem_ids.at(cluster_id)){
            ref_ids.push_back(problems_[problem_id].ref_image_idx);
        }
        std::sort(ref_ids.begin(), ref_ids.end());
        ref_flags.resize(whole_ids.size());
        int k = 0;
        // std::cout << "ref_falg: " ;
        for (int i = 0; i < whole_ids.size(); i++){
            if (ref_ids[k] == whole_ids[i]){
                ref_flags[i] = 1;
                k++;
            } else {
                ref_flags[i] = 0;
            }
        }
        // std::cout << std::endl;
    }

    const auto& model = workspace_->GetModel();
    const double kMinTriangulationAngle = options_.min_triangulation_angle;
    // std::vector<std::vector<int> > overlapping_images = 
    //     model.GetMaxOverlappingImages(INT_MAX, kMinTriangulationAngle);
    for (auto problem_id : problem_ids){
        auto& problem = problems_[problem_id];
        const int rid = problem.ref_image_idx;
        auto& src_image_extend_idxs = problem.src_image_extend_idxs;
        src_image_extend_idxs.clear();
        src_image_extend_idxs.shrink_to_fit();
        // int num_extend_src = 0;
        std::unordered_set<size_t > set_src_ids;
        for (size_t i = 0; i < problem.src_image_idxs.size(); i++){
            image_t src_id = problem.src_image_idxs.at(i);
            if (cluster_whole_ids.at(cluster_id).find(src_id) == 
                cluster_whole_ids.at(cluster_id).end()){
                continue;
            }
            src_image_extend_idxs.push_back(src_id);
            set_src_ids.insert(src_id);
        }
        for (size_t i = 0, k = src_image_extend_idxs.size(); i < overlapping_images[rid].size() && k < MAX_NUM_CROSS_SRC; ++i){
            if (cluster_whole_ids.at(cluster_id).find(overlapping_images[rid][i]) == 
                cluster_whole_ids.at(cluster_id).end()){
                continue;
            }
            if (set_src_ids.find(overlapping_images[rid][i]) != set_src_ids.end()){
                continue;
            }
            src_image_extend_idxs.push_back(overlapping_images[rid][i]);
            k++;
        }
    }

    std::string ref_images_str, problems_str;
    for (const auto& problem_id : problem_ids){
        ref_images_str += "\t";
        ref_images_str += std::to_string(problem_id);
        ref_images_str += "-";
        ref_images_str += std::to_string(problems_[problem_id].ref_image_idx);
        ref_images_str += "/ (" + std::to_string(problems_[problem_id].src_image_extend_idxs.size()) + ") ";
        for (const auto& src_id : problems_[problem_id].src_image_extend_idxs) {
            ref_images_str += std::to_string(src_id);
            ref_images_str += " ";
        }
        ref_images_str += "\n";
    }
    std::cout << "\nCluster-" << cluster_id << ": problems ids(" << problem_ids.size() <<") \n" << ref_images_str << std::endl;

    std::string whole_images_str;
    for (const auto& whole_id : whole_ids){
        whole_images_str += std::to_string(whole_id);
        whole_images_str += " ";
    }
    std::cout << "Cluster-" << cluster_id << ": whole image ids(" << whole_ids.size() <<") " << whole_images_str << std::endl;

    std::vector<Image> images = model.images;
#ifndef CACHED_DEPTH_MAP
        std::vector<DepthMap> depth_maps_;
        depth_maps_.resize(images.size());
#endif

    const auto& depth_maps_path = JoinPaths(workspace_path, DEPTHS_DIR);
    const auto& normal_maps_path = JoinPaths(workspace_path, NORMALS_DIR);
    const auto& conf_maps_path = JoinPaths(workspace_path, CONFS_DIR);
    // const auto& point_cloud_path = JoinPaths(workspace_path, "ply");
    const auto& consistency_maps_path = 
        JoinPaths(workspace_path, CONSISTENCY_DIR);
        
        std::vector<Problem> cluster_problems;
    for (const auto &problem_idx : problem_ids){
        Problem& problem = problems_.at(problem_idx);

        const auto &image_name = model.GetImageName(problem.ref_image_idx);
        const std::string file_name = StringPrintf("%s.%s.%s", 
            image_name.c_str(), options.output_type.c_str(), DEPTH_EXT);
        const std::string depth_map_path = JoinPaths(depth_maps_path, file_name);
        const std::string normal_map_path = JoinPaths(normal_maps_path, file_name);
        const std::string conf_map_path = JoinPaths(conf_maps_path, file_name);
        const std::string consistency_graph_path = 
            JoinPaths(consistency_maps_path, file_name);

        // float factor_level = 1 / std::pow(2,level);
        // float max_size = std::max(images.at(problem.ref_image_idx).GetHeight(),
        //                                 images.at(problem.ref_image_idx).GetWidth());
        // float min_image_size = std::min(320.f, max_size);
        // factor_level = std::max(factor_level, min_image_size/max_size);

        problem.images = &images;
        problem.depth_maps = &depth_maps_;
        problem.normal_maps = &normal_maps_;

        {
            // Collect all used images in current problem.
            std::unordered_set<int> used_image_idxs(problem.src_image_extend_idxs.begin(),
                                                    problem.src_image_extend_idxs.end());
            used_image_idxs.insert(problem.ref_image_idx);

            options.filter_min_num_consistent =
                std::min(static_cast<int>(used_image_idxs.size()) - 1,
                        options.filter_min_num_consistent);

            int used_image_num = 0;
            for (const auto image_idx : used_image_idxs) {
                if (used_images_.find(image_idx) == used_images_.end()) {
                    continue;
                }              
                used_image_num ++;
            }
            if (used_image_num < 2){
                std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because source images do not "
                    "exist.", image_name.c_str())
                << std::endl;
                continue;
            }

            std::unordered_set<int> invalid_image_idxs;
            // std::cout << "Reading inputs..." << std::endl;  

            for (const auto image_idx : used_image_idxs) {
                if (used_images_.find(image_idx) == used_images_.end()) {
                    invalid_image_idxs.insert(image_idx);
                    continue;
                }
                // const std::string src_depth_map_path = 
                //     workspace_->GetDepthMapPath(image_idx);
                // const std::string src_normal_map_path = 
                //     workspace_->GetNormalMapPath(image_idx);
                auto src_file_name = workspace_->GetFileName(image_idx);
                const std::string src_depth_map_path = JoinPaths(depth_maps_path, src_file_name);
                const std::string src_normal_map_path = JoinPaths(normal_maps_path, src_file_name); 
                
                if (depth_maps_.at(image_idx).IsValid() && 
                    normal_maps_.at(image_idx).IsValid()){
                    continue;
                }
                if (!ExistsFile(src_depth_map_path) || 
                    !ExistsFile(src_normal_map_path)) {
                    invalid_image_idxs.insert(image_idx);
                    continue;
                }
                depth_maps_.at(image_idx).Read(src_depth_map_path);
                normal_maps_.at(image_idx).Read(src_normal_map_path);
                if (!depth_maps_.at(image_idx).IsValid() ||
                    !normal_maps_.at(image_idx).IsValid()) {
                    invalid_image_idxs.insert(image_idx);
                    continue;
                }
                depth_maps_info_.at(image_idx).width = depth_maps_.at(image_idx).GetWidth();
                depth_maps_info_.at(image_idx).height = depth_maps_.at(image_idx).GetHeight();
                depth_maps_info_.at(image_idx).depth_min = depth_maps_.at(image_idx).GetDepthMin();
                depth_maps_info_.at(image_idx).depth_max = depth_maps_.at(image_idx).GetDepthMax();
            }

            if (invalid_image_idxs.find(problem.ref_image_idx) != 
                invalid_image_idxs.end()) {
                std::cout << StringPrintf(
                        "WARNING: Ignoring image %s, because refer image is do not "
                        "be used.", image_name.c_str())
                        << std::endl;
                continue;
            }

            for (const auto image_idx : used_image_idxs) {
                if (invalid_image_idxs.find(image_idx) != 
                    invalid_image_idxs.end()){
                    continue;
                }
                if (depth_maps_.at(image_idx).GetWidth() != images.at(image_idx).GetWidth() ||
                    depth_maps_.at(image_idx).GetHeight() != images.at(image_idx).GetHeight()){
                    float factor_x = (float)depth_maps_.at(image_idx).GetWidth() / images.at(image_idx).GetWidth();
                    float factor_y = (float)depth_maps_.at(image_idx).GetHeight() / images.at(image_idx).GetHeight();
                    images.at(image_idx).Rescale(factor_x, factor_y);
                    // std::cout << "image " << image_idx << "(" 
                    //     << images.at(image_idx).GetWidth() << "x"
                    //     << images.at(image_idx).GetHeight() << ")   " 
                    //     << depth_maps_.at(image_idx).GetWidth() << "x"
                    //     << depth_maps_.at(image_idx).GetHeight() << std::endl;
                }
            }

            // Remove invalid image idx.
            int i, j;
            for (i = 0, j = 0; i < problem.src_image_extend_idxs.size(); ++i) {
                const int image_idx = problem.src_image_extend_idxs.at(i);
                if (invalid_image_idxs.find(image_idx) == 
                    invalid_image_idxs.end()) {
                    problem.src_image_extend_idxs[j] = problem.src_image_extend_idxs[i];
                    j = j + 1;
                }
            }
            problem.src_image_extend_idxs.resize(j);
            if (problem.src_image_extend_idxs.size() == 0) {
                std::cout << StringPrintf(
                        "WARNING: Ignoring image %d(%s), because source images do not "
                        "exist.", problem.ref_image_idx, image_name.c_str())
                        << std::endl;
                continue;
            }
        }
        cluster_problems.push_back(problem);
    }

    std::unordered_set<image_t> all_image_ids;
    for (const auto& temp_problem : cluster_problems){
        all_image_ids.insert(temp_problem.ref_image_idx);
        for (const auto& src_id : temp_problem.src_image_extend_idxs){
            all_image_ids.insert(src_id);
        }
    }
    int i = 0, j = 0;
    for (i = 0; i < whole_ids.size(); i++){
        if (all_image_ids.find(whole_ids[i]) == all_image_ids.end()){
            std::cout << StringPrintf("Remove image %d(%s), from Cluster %d ",
                        whole_ids[i], model.GetImageName(whole_ids[i]).c_str(), 
                        cluster_id)
                        << std::endl;
            continue;
        }
        whole_ids.at(j) = whole_ids.at(i);
        ref_flags.at(j) = ref_flags.at(i);
        j++;
    }
    whole_ids.resize(j);
    ref_flags.resize(j);

    // int gpu_index = cross_filter_gpu_indices_.at(cross_filter_thread_pool_->GetThreadIndex());
    int gpu_index = cluster_gpu_idx.at(cluster_id);
    options.gpu_index = std::to_string(gpu_index);
    options.thread_index = 0;

    CrossFilterCuda cross_filter_cuda(options, cluster_problems, whole_ids, ref_flags);
    cross_filter_cuda.Run();
    int clear_num = 0;
    for (int i = 0; i < whole_ids.size(); i++){
        if (images_exclusive.at(whole_ids.at(i))){
            depth_maps_.at(whole_ids.at(i)) = std::move(DepthMap());
            clear_num++;
        }
    }

    const int num_valid_thread = 
        std::min(cross_filter_gpu_indices_.size(), cluster_whole_ids.size());
    std::unique_ptr<ThreadPool> save_thread_pool;
    const int num_all_threads = GetEffectiveNumThreads(-1);
    const int num_empty_thread = (num_all_threads - num_valid_thread) / num_valid_thread;
    int num_eff_threads = std::max(num_empty_thread, 1);
    num_eff_threads = std::min(num_eff_threads, (int)problem_ids.size());
    std::cout << "Cluster-" << cluster_id 
              << " valid num_eff_threads: " << num_eff_threads << std::endl;
    save_thread_pool.reset(new ThreadPool(num_eff_threads));

    if (process_rgbd_prior){
        auto ProcessPriorMap = [&](int problem_id, 
                        DepthMap& ref_depth_map,
                        NormalMap& ref_normal_map){
            const int image_idx = problems_[problem_id].ref_image_idx;
            const auto &image_name = model.GetImageName(image_idx);
            const std::string file_name = StringPrintf("%s.%s.%s", 
                image_name.c_str(), options.output_type.c_str(), DEPTH_EXT);
            // const std::string depth_map_path = JoinPaths(depth_maps_path, file_name);
            const std::string rgbd_path = JoinPaths(workspace_path, "..", DEPTHS_DIR, image_name + "." + DEPTH_EXT);
            const std::string rgbd_norm_path = JoinPaths(workspace_path, "..", NORMALS_DIR, image_name + "." + DEPTH_EXT);
            if (!boost::filesystem::exists(GetParentDir(rgbd_norm_path)) && options.verbose) {
                boost::filesystem::create_directories(GetParentDir(rgbd_norm_path));
            }
            if (!ExistsFile(rgbd_path)){
                depth_maps_temp_.at(image_idx) = ref_depth_map;
                normal_maps_temp_.at(image_idx) = ref_normal_map;
                return;
            }
            {
                DepthMap prior_depth_map;
                prior_depth_map.Read(rgbd_path);

                if (prior_depth_map.GetWidth() != ref_depth_map.GetWidth() ||
                    prior_depth_map.GetHeight() != ref_depth_map.GetHeight()){
                    std::cout << "Cross " << image_name << "  prior_depth_map != ref_depth_map" << std::endl;
                }

                Mat<bool > filter_mask(ref_depth_map.GetWidth(), ref_depth_map.GetHeight(), 1);
                filter_mask.Fill(0);
                const int window_radius = ref_depth_map.GetWidth() * 0.001;
                for (int row = 0; row < ref_depth_map.GetHeight(); row++) {
                    for (int col = 0; col < ref_depth_map.GetWidth(); col++) {
                        const float depth_ref = ref_depth_map.Get(row, col);
                        if (depth_ref > 1.0e-6){
                            filter_mask.Set(row, col, 1);
                            for (int i = -window_radius; i <= window_radius; i++){
                                for (int j = -window_radius; j <= window_radius; j++){
                                    if (row + i < 0 || row + i >= ref_depth_map.GetHeight() ||
                                        col + j < 0 || col + j >= ref_depth_map.GetWidth() ||
                                        (i == 0 && j == 0)){
                                            continue;
                                    }
                                    float depth_tem = ref_depth_map.Get(row + i, col + j);
                                    float depth_prior = prior_depth_map.Get(row + i, col + j);
                                    // if (depth_tem < 1.0e-6 && depth_prior < 1.0e-6){
                                    if (depth_tem < 1.0e-6 && (depth_prior > depth_ref || depth_prior < 1.0e-6)){
                                    // if ((depth_tem < 1.0e-6 || depth_tem > depth_ref) && 
                                    //     (depth_prior > depth_ref || depth_prior < 1.0e-6)){
                                        prior_depth_map.Set(row + i, col + j, depth_ref);
                                        filter_mask.Set(row + i, col + j, 1);
                                    }
                                }
                            }
                        }
                    }
                }
                {
                    DepthMap filter_depth_map;
                    filter_depth_map = ref_depth_map;
                    for (int row = 0; row < ref_depth_map.GetHeight(); row++) {
                        for (int col = 0; col < ref_depth_map.GetWidth(); col++) {
                            if (!filter_mask.Get(row, col)){
                                continue;
                            }
                            const float depth_ref = filter_depth_map.Get(row, col);
                            if (depth_ref < 1.0e-6){
                                float depth_prior = prior_depth_map.Get(row, col);
                                filter_depth_map.Set(row, col, depth_prior);
                            }
                        }
                    }
                    if (options.verbose) {
                        filter_depth_map.ToBitmap().Write(StringPrintf("%s-depth-filter.jpg", 
                                                                    rgbd_path.c_str()));
                    }
                }

                {
                    NormalMap filter_normal_map;
                    filter_normal_map = ref_normal_map;
                    for (int row = 0; row < ref_normal_map.GetHeight(); row++) {
                        for (int col = 0; col < ref_normal_map.GetWidth(); col++) {
                        if (ref_depth_map.Get(row, col) >  1.0e-6){
                                continue;
                            } 
                            filter_normal_map.Set(row, col, 0, 0.0f);
                            filter_normal_map.Set(row, col, 1, 0.0f);
                            filter_normal_map.Set(row, col, 2, 0.0f);
                            continue;
                        }
                    }
                    if (options.verbose) {
                        filter_normal_map.ToBitmap().Write(StringPrintf("%s-normal-filter.jpg", 
                                                                    rgbd_path.c_str()));
                    }
                }

                Eigen::Map<const Eigen::RowMatrix3f> K(images[image_idx].GetK());
                const Eigen::RowMatrix3f inv_K = K.inverse();
                const int delt_dist = 2;
                for (int row = 0; row < ref_normal_map.GetHeight(); row++) {
                    for (int col = 0; col < ref_normal_map.GetWidth(); col++) {
                       if (filter_mask.Get(row, col)){
                            continue;
                        }
                        if (row < delt_dist || row >= ref_normal_map.GetHeight() - delt_dist ||
                            col < delt_dist || col >= ref_normal_map.GetWidth() - delt_dist){
                            ref_normal_map.Set(row, col, 0, 0.0f);
                            ref_normal_map.Set(row, col, 1, 0.0f);
                            ref_normal_map.Set(row, col, 2, 0.0f);
                            continue;
                        }
                        //      0
                        //   2     3
                        //      1
                        const int row0 = row - delt_dist;
                        const int row1 = row + delt_dist;
                        const int row2 = row;
                        const int row3 = row;
                        const int col0 = col;
                        const int col1 = col;
                        const int col2 = col - delt_dist;
                        const int col3 = col + delt_dist;
                        float depth0 = prior_depth_map.Get(row0, col0);
                        float depth1 = prior_depth_map.Get(row1, col1);
                        float depth2 = prior_depth_map.Get(row2, col2);
                        float depth3 = prior_depth_map.Get(row3, col3);
                        if (depth0 < 1e-6 || depth1 < 1e-6 || depth2 < 1e-6 || depth3 < 1e-6){
                            ref_normal_map.Set(row, col, 0, 0.0f);
                            ref_normal_map.Set(row, col, 1, 0.0f);
                            ref_normal_map.Set(row, col, 2, 0.0f);
                            continue;
                        }
                        Eigen::Vector3f xyz0 = inv_K * Eigen::Vector3f(col0 * depth0, row0 * depth0, depth0);
                        Eigen::Vector3f xyz1 = inv_K * Eigen::Vector3f(col1 * depth1, row1 * depth1, depth1);
                        Eigen::Vector3f xyz2 = inv_K * Eigen::Vector3f(col2 * depth2, row2 * depth2, depth2);
                        Eigen::Vector3f xyz3 = inv_K * Eigen::Vector3f(col3 * depth3, row3 * depth3, depth3);

                        Eigen::Vector3f normal = ((xyz1 - xyz0).cross(xyz3 - xyz2)).normalized();
                        ref_normal_map.Set(row, col, 0, normal.x());
                        ref_normal_map.Set(row, col, 1, normal.y());
                        ref_normal_map.Set(row, col, 2, normal.z());
                    }
                }
                {
                    NormalMap filter_normal_map(ref_depth_map.GetWidth(), ref_depth_map.GetHeight());
                    filter_normal_map.Fill(0.0);
                    const int delt_dist = 2;
                    for (int row = 0; row < ref_normal_map.GetHeight(); row++) {
                        for (int col = 0; col < ref_normal_map.GetWidth(); col++) {
                            if (row < delt_dist || row >= ref_normal_map.GetHeight() - delt_dist ||
                                col < delt_dist || col >= ref_normal_map.GetWidth() - delt_dist){
                                filter_normal_map.Set(row, col, 0, 0.0f);
                                filter_normal_map.Set(row, col, 1, 0.0f);
                                filter_normal_map.Set(row, col, 2, 0.0f);
                                continue;
                            }
                            const int row0 = row - delt_dist;
                            const int row1 = row + delt_dist;
                            const int row2 = row;
                            const int row3 = row;
                            const int col0 = col;
                            const int col1 = col;
                            const int col2 = col - delt_dist;
                            const int col3 = col + delt_dist;
                            float depth0 = prior_depth_map.Get(row0, col0);
                            float depth1 = prior_depth_map.Get(row1, col1);
                            float depth2 = prior_depth_map.Get(row2, col2);
                            float depth3 = prior_depth_map.Get(row3, col3);
                            if (depth0 < 1e-6 || depth1 < 1e-6 || depth2 < 1e-6 || depth3 < 1e-6){
                                filter_normal_map.Set(row, col, 0, 0.0f);
                                filter_normal_map.Set(row, col, 1, 0.0f);
                                filter_normal_map.Set(row, col, 2, 0.0f);
                                continue;
                            }
                            Eigen::Vector3f xyz0 = inv_K * Eigen::Vector3f(col0 * depth0, row0 * depth0, depth0);
                            Eigen::Vector3f xyz1 = inv_K * Eigen::Vector3f(col1 * depth1, row1 * depth1, depth1);
                            Eigen::Vector3f xyz2 = inv_K * Eigen::Vector3f(col2 * depth2, row2 * depth2, depth2);
                            Eigen::Vector3f xyz3 = inv_K * Eigen::Vector3f(col3 * depth3, row3 * depth3, depth3);

                            Eigen::Vector3f normal = ((xyz1 - xyz0).cross(xyz3 - xyz2)).normalized();
                            filter_normal_map.Set(row, col, 0, normal.x());
                            filter_normal_map.Set(row, col, 1, normal.y());
                            filter_normal_map.Set(row, col, 2, normal.z());
                        }
                    }
                }

                for (int row = 0; row < ref_depth_map.GetHeight(); row++) {
                    for (int col = 0; col < ref_depth_map.GetWidth(); col++) {
                        float depth_ref = ref_depth_map.Get(row, col);
                        // if (ref_depth_map.Get(row, col) > 1.0e-6){
                        //     continue;
                        // }
                        float depth_ori = prior_depth_map.Get(row, col);
                        if (depth_ori > 1.0e-6 && (depth_ori < depth_ref || depth_ref < 1.0e-6)){
                            ref_depth_map.Set(row, col, depth_ori);
                        }
                    }
                }
                if (options.verbose) {
                    ref_depth_map.ToBitmap().Write(StringPrintf("%s-depth-prior.jpg", 
                                                                rgbd_path.c_str()));
                    ref_normal_map.ToBitmap().Write(StringPrintf("%s-normal-prior.jpg", 
                                                                rgbd_path.c_str()));
                }
                depth_maps_temp_.at(image_idx) = ref_depth_map;
                normal_maps_temp_.at(image_idx) = ref_normal_map;

                // CreateDirIfNotExists(GetParentDir(rgbd_path));

                // ref_depth_map.Write(rgbd_path);
                // if (options.verbose) {
                //     ref_depth_map.ToBitmap().Write(StringPrintf("%s-refine.jpg", 
                //                                                 rgbd_path.c_str()));
                // } 

                // ref_normal_map.Write(rgbd_norm_path);
                // if (options.verbose) {
                //     ref_normal_map.ToBitmap().Write(StringPrintf("%s-refine.jpg", 
                //                                                 rgbd_norm_path.c_str()));
                // }
            }

            const std::string prior_wgt_path = JoinPaths(workspace_path, "..", "intensity_maps", image_name + "." + DEPTH_EXT);
            if (ExistsFile(prior_wgt_path)){
                prior_wgt_maps_.at(image_idx).Read(prior_wgt_path);
            }
            // std::cout << "Read " << image_name << " intensity map, " << prior_wgt_maps_.at(image_idx).GetHeight() << " * " 
            //     << prior_wgt_maps_.at(image_idx).GetWidth() << std::endl;
        };
        std::string save_images_str;
        for (auto problem_id : problem_ids){
            int image_idx = problems_[problem_id].ref_image_idx;
            DepthMap ref_depth_map = cross_filter_cuda.GetRefineDepthMap(image_idx, depth_maps_info_.at(image_idx));
            if (!ref_depth_map.IsValid()){
                continue;
            }
            save_thread_pool->AddTask(ProcessPriorMap, problem_id, ref_depth_map, normal_maps_.at(image_idx));
            save_images_str += std::to_string(image_idx);
            save_images_str += " ";
        }
        save_thread_pool->Wait();
        std::cout << "Prior-" << cluster_id << " process image ids(" 
            << problem_ids.size() <<"): " << save_images_str << std::endl;

        return;
    }

    if (options.save_depth_map || !options.patch_match_fusion){
        // std::mutex save_map_mutex_;
        auto SaveMap = [&](int problem_id){
            const int image_idx = problems_[problem_id].ref_image_idx;
            DepthMap ref_depth_map;
            {
                ref_depth_map = cross_filter_cuda.GetRefineDepthMap(image_idx, depth_maps_info_.at(image_idx));
                if (!ref_depth_map.IsValid()){
                    return;
                } 
            }
            
            const auto &image_name = model.GetImageName(image_idx);
            const std::string file_name = StringPrintf("%s.%s.%s", 
                image_name.c_str(), options.output_type.c_str(), DEPTH_EXT);
            const std::string depth_map_path = JoinPaths(depth_maps_path, file_name);
            const std::string normal_map_path = JoinPaths(normal_maps_path, file_name);
            {
                std::string parent_path = GetParentDir(depth_map_path); 
                if (!ExistsPath(parent_path)) {
                    boost::filesystem::create_directories(parent_path);
                }
                ref_depth_map.Write(depth_map_path);
                if (options.verbose) {
                    ref_depth_map.ToBitmap().Write(StringPrintf("%s-cross.jpg", 
                                                                depth_map_path.c_str()));
                }
#if 0
                {
                    Bitmap bitmap;
                    bitmap.Read(workspace_->GetBitmapPath(image_idx));
                    images[image_idx].SetBitmap(bitmap);

                    std::vector<PlyPoint> image_points;
                    Eigen::Map<const Eigen::RowMatrix3x4f> inv_P(images[image_idx].GetInvP());
                    Eigen::Map<const Eigen::RowMatrix3f> R(images[image_idx].GetR());

                    int step = options.step_size;
                    for (int row = 3; row < ref_depth_map.GetHeight()-3; row += step){
                        for (int col = 3; col < ref_depth_map.GetWidth()-3; col += step){
                            float depth = ref_depth_map.Get(row, col);
                            if (depth < 1e-6){
                                continue;
                            }
                            Eigen::Vector3f xyz =
                                inv_P * Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

                            if (options_.roi_fuse){
                                const Eigen::Vector3f rot_xyz = roi_box_.rot * xyz;
                                if (rot_xyz(0) < roi_box_.x_box_min || rot_xyz(0) > roi_box_.x_box_max ||
                                    rot_xyz(1) < roi_box_.y_box_min || rot_xyz(1) > roi_box_.y_box_max ||
                                    rot_xyz(2) < roi_box_.z_box_min || rot_xyz(2) > roi_box_.z_box_max){
                                    continue;
                                }
                            }

                            if (std::isnan(xyz.x()) || std::isinf(xyz.x()) ||
                                std::isnan(xyz.y()) || std::isinf(xyz.y()) ||
                                std::isnan(xyz.z()) || std::isinf(xyz.z())) {
                                continue;
                            }
                            
                            Eigen::Vector3f normal;
                            normal = R.inverse() * Eigen::Vector3f(
                                ref_normal_map.Get(row, col, 0), ref_normal_map.Get(row, col, 1),
                                ref_normal_map.Get(row, col, 2));
                            
                            PlyPoint pnt;
                            pnt.x = xyz(0);
                            pnt.y = xyz(1);
                            pnt.z = xyz(2);
                            pnt.nx = normal(0);
                            pnt.ny = normal(1);
                            pnt.nz = normal(2);
                            BitmapColor<uint8_t> color;
                            images[image_idx].GetBitmap().GetPixel(col, row, &color);
                            pnt.r = color.r;
                            pnt.g = color.g;
                            pnt.b = color.b;
                            image_points.push_back(pnt);
                        }
                    }
                    WriteBinaryPlyPoints(StringPrintf("%s-cross.ply", 
                        depth_map_path.c_str()), image_points, true, true);
                }
#endif
            }

            NormalMap& ref_normal_map = normal_maps_.at(image_idx);
            {
                std::string parent_path = GetParentDir(normal_map_path);
                if (!ExistsPath(parent_path)) {
                    boost::filesystem::create_directories(parent_path);
                }
                ref_normal_map.Write(normal_map_path);
                if (options.verbose) {
                    ref_normal_map.ToBitmap().Write(StringPrintf("%s-cross.jpg", 
                                                                normal_map_path.c_str()));
                } 
            }
            if (!options.patch_match_fusion && images_exclusive.at(image_idx)){
                ref_normal_map = std::move(NormalMap());
            }
        };
        std::string save_images_str;
        for (auto problem_id : problem_ids){
            save_thread_pool->AddTask(SaveMap, problem_id);
            save_images_str += std::to_string(problems_[problem_id].ref_image_idx);
            save_images_str += " ";
        }
        save_thread_pool->Wait();
        std::cout << "Cluster-" << cluster_id << " save image ids(" 
            << problem_ids.size() <<"): " << save_images_str << std::endl;
    }

    if (options.patch_match_fusion){
        cross_filter_cuda.DedupRun();

        std::cout << "Fusion cluster-" << cluster_id << " begin ..." << std::endl;
 
        Timer save_timer;
        save_timer.Start();
        
        std::vector<std::vector<PlyPoint> > problem_points(problem_ids.size());
        std::vector<std::vector<float> > problem_points_score(problem_ids.size());
        std::vector<std::vector<std::vector<uint32_t> >> problem_points_visibility(problem_ids.size());

        const float inv_color_factor = 1.0f / 255.0f;

        auto ComputeScore = [&](int problem_id,
                                const Eigen::Vector3f& xyz,
                                const float delt_depth,
                                const std::vector<uint32_t>& pnt_vis,
                                const int row, const int col){
            const int image_idx = problems_[problem_id].ref_image_idx;
            const int width = images.at(image_idx).GetWidth();
            const int height = images.at(image_idx).GetHeight();
            float score = 0;
            std::vector<float> pnt_conf;

            float m_conf(0.0f);
            if (pnt_vis.size() < 3){
                if (bitmaps_.at(image_idx).Data()){
                    if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
                        BitmapColor<uint8_t> lcolor, rcolor, tcolor, bcolor;
                        bitmaps_.at(image_idx).GetPixel(col - 1, row, &lcolor);
                        bitmaps_.at(image_idx).GetPixel(col, row - 1, &tcolor);
                        bitmaps_.at(image_idx).GetPixel(col + 1, row, &rcolor);
                        bitmaps_.at(image_idx).GetPixel(col, row + 1, &bcolor);
                        float gx = (lcolor.r - rcolor.r) * inv_color_factor;
                        float gy = (tcolor.r - bcolor.r) * inv_color_factor;
                        float grad = sqrt(gx * gx + gy * gy);
                        pnt_conf.push_back(grad);
                    } else {
                        pnt_conf.push_back(0);
                    }
                }

                for (const auto vis : pnt_vis){
                    Eigen::Map<const Eigen::RowMatrix3x4f> P(images[vis].GetP());
                    const Eigen::Vector3f next_proj = P * xyz.homogeneous();
                    const float inv_z = 1.0f / next_proj(2);
                    int vis_col = static_cast<int>(std::round(next_proj(0) * inv_z));
                    int vis_row = static_cast<int>(std::round(next_proj(1) * inv_z));

                    const int vis_width = images.at(vis).GetWidth();
                    const int vis_height = images.at(vis).GetHeight();
                    if (vis_col > 1 && vis_col < vis_width - 1 && vis_row > 1 && vis_row < vis_height - 1) {
                        BitmapColor<uint8_t> lcolor, rcolor, tcolor, bcolor;
                        bitmaps_.at(vis).GetPixel(vis_col - 1, vis_row, &lcolor);
                        bitmaps_.at(vis).GetPixel(vis_col, vis_row - 1, &tcolor);
                        bitmaps_.at(vis).GetPixel(vis_col + 1, vis_row, &rcolor);
                        bitmaps_.at(vis).GetPixel(vis_col, vis_row + 1, &bcolor);
                        float gx = (lcolor.r - rcolor.r) * inv_color_factor;
                        float gy = (tcolor.r - bcolor.r) * inv_color_factor;
                        float grad = sqrt(gx * gx + gy * gy);
                        pnt_conf.push_back(grad);
                    }else {
                        pnt_conf.push_back(0);
                    }
                }
                for (size_t i = 0; i < pnt_conf.size(); ++i) {
                    m_conf += pnt_conf[i];
                }
                if (pnt_conf.size() >= 1){
                    m_conf = m_conf / pnt_conf.size();
                } else {
                    m_conf = 0.0f;
                }
            } else {
                m_conf = 1.0f;
            }
            m_conf = 1.0f - m_conf;
            // std::cout << "delt_depth, m_conf: " << delt_depth << ", " << m_conf << std::endl;

            score = std::min(0.6f * delt_depth + 0.4f * m_conf, 1.0f);

            return score;
        };

        auto ConvertVisibility = [&](int problem_id,
                                    const Mat<uint32_t>& vis_map,
                                    const int row, const int col){
            std::vector<uint32_t> pnt_vis;
            const auto &vis_problem = problems_[problem_id];
            int image_idx = vis_problem.ref_image_idx;
            const auto& src_idxs = vis_problem.src_image_extend_idxs;
            CHECK_LE(src_idxs.size(), MAX_NUM_CROSS_SRC);

            uint32_t vis_value = vis_map.Get(row, col);
            pnt_vis.push_back(image_idx);
            for (int i = 0; i < src_idxs.size(); i++){
                uint32_t mask = (1 << (MAX_NUM_CROSS_SRC - 1 - i));
                if (vis_value & mask){
                    pnt_vis.push_back(src_idxs[i]);
                }
            }

            return pnt_vis;
        };
        
        auto SavePoints = [&](int points_id, int problem_id, 
                            const DepthMap& ref_depth_map, 
                            const Mat<uint32_t>& vis_map){
            const int image_idx = problems_[problem_id].ref_image_idx;
            if (all_image_ids.find(image_idx) == all_image_ids.end()){
                return;
            }
            DepthMap delt_depth_map = cross_filter_cuda.GetDeltDepthMap(image_idx);
            NormalMap& ref_normal_map = normal_maps_.at(image_idx);
            if (!prior_ply){
                Bitmap bitmap;
                bitmap.Read(workspace_->GetBitmapPath(image_idx));
                images[image_idx].SetBitmap(bitmap);
            }

            const float thd_depth = 0.975 * model.ComputeNthDepthImage(image_idx, 0.1);
            std::vector<PlyPoint> image_points;
            std::vector<float > image_points_score;
            std::vector<std::vector<uint32_t> > image_points_visibility;
            Eigen::Map<const Eigen::RowMatrix3x4f> inv_P(images[image_idx].GetInvP());
            Eigen::Map<const Eigen::RowMatrix3f> R(images[image_idx].GetR());

#if 0
            int step = options.step_size;
            for (int row = 3; row < ref_depth_map.GetHeight()-3; row += step){
                for (int col = 3; col < ref_depth_map.GetWidth()-3; col += step){
                    float depth = ref_depth_map.Get(row, col);
                    if (depth < 1e-6){
                        continue;
                    }
                    Eigen::Vector3f xyz =
                        inv_P * Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

                    if (options_.roi_fuse){
                        const Eigen::Vector3f rot_xyz = roi_box_.rot * xyz;
                        if (rot_xyz(0) < roi_box_.x_box_min || rot_xyz(0) > roi_box_.x_box_max ||
                            rot_xyz(1) < roi_box_.y_box_min || rot_xyz(1) > roi_box_.y_box_max ||
                            rot_xyz(2) < roi_box_.z_box_min || rot_xyz(2) > roi_box_.z_box_max){
                            continue;
                        }
                    }

                    if (std::isnan(xyz.x()) || std::isinf(xyz.x()) ||
                        std::isnan(xyz.y()) || std::isinf(xyz.y()) ||
                        std::isnan(xyz.z()) || std::isinf(xyz.z())) {
                        continue;
                    }
                    
                    Eigen::Vector3f normal;
                    normal = R.inverse() * Eigen::Vector3f(
                        ref_normal_map.Get(row, col, 0), ref_normal_map.Get(row, col, 1),
                        ref_normal_map.Get(row, col, 2));
                    
                    PlyPoint pnt;
                    pnt.x = xyz(0);
                    pnt.y = xyz(1);
                    pnt.z = xyz(2);
                    pnt.nx = normal(0);
                    pnt.ny = normal(1);
                    pnt.nz = normal(2);
                    if (!prior_ply){
                        BitmapColor<uint8_t> color;
                        images[image_idx].GetBitmap().GetPixel(col, row, &color);
                        pnt.r = color.r;
                        pnt.g = color.g;
                        pnt.b = color.b;
                    }
                    image_points.push_back(pnt);

                    auto vis = ConvertVisibility(problem_id, vis_map, row, col);
                    image_points_visibility.push_back(vis);

                    float delt_depth = delt_depth_map.Get(row, col);
                    if (delt_depth < 1e-7 || delt_depth > 1.0f){
                        delt_depth = 1.0f;
                    }
                    float score = ComputeScore(problem_id, xyz, delt_depth, vis, row, col);
                    image_points_score.push_back(score);
                }

            }
#else
            int step = options.step_size;
            int win_step = 6;
            int num_samp = std::ceil(float(win_step) / step);

            int num_samp_row = (ref_depth_map.GetHeight()-6) / win_step + 1;
            int num_samp_col = (ref_depth_map.GetWidth()-6) / win_step + 1;
            std::vector<std::vector<PlyPoint>> win_pnts(num_samp_row * num_samp_col);
            std::vector<std::vector<float> > win_pnts_sco(num_samp_row * num_samp_col);
            std::vector<std::vector<std::vector<uint32_t> > > win_pnts_vis(num_samp_row * num_samp_col);

#pragma omp parallel for schedule(dynamic)
            for (int row = 3; row < ref_depth_map.GetHeight()-3; row += win_step){
#pragma omp parallel for schedule(dynamic)
                for (int col = 3; col < ref_depth_map.GetWidth()-3; col += win_step){
                    // windows size: 2*5+1
                    std::vector<std::pair<float, int>> depths(num_samp * num_samp);
                    int num_pnts = 0;
                    int num_dr = 0, num_dc = 0;
                    float max_depth = 0.0;
                    for(int dr = 0; dr < win_step; dr += step){
                        if (row + dr >= ref_depth_map.GetHeight()-3){
                            continue;
                        }
                        num_dc = 0;
                        for(int dc = 0; dc < win_step; dc += step){
                            if (col + dc >= ref_depth_map.GetWidth()-3){
                                continue;
                            }
                            float depth = ref_depth_map.Get(row + dr, col + dc);
                            if (depth < 1e-6){
                                continue;
                            }
                            depths.at(num_pnts) = std::make_pair(depth, num_dr * num_samp + num_dc);
                            num_pnts++;
                            num_dc++;
                        }
                        num_dr++;
                    }
                    if (num_pnts == 0){
                        continue;
                    }
                    depths.resize(num_pnts);
                    depths.shrink_to_fit();
                    // float temp = depths.at(0).first;
                    std::sort(depths.begin(),depths.end(), 
                          [](const std::pair<float, int> & depth1,
                          const std::pair<float, int> & depth2) {
                          return depth1.first > depth2.first;});
                    // std::cout << "depths.size(): " << depths.size() << "(" << row << "*" << col << "): " 
                    //     << depths.at(0).first << "/" << temp << std::endl;

                    std::pair<float, int> ref_depth;
                    Eigen::Vector3f ref_xyz(Eigen::Vector3f::Zero()), ref_normal(Eigen::Vector3f::Zero());
                    std::set<uint32_t> ref_pnt_vis;
                    float ref_score;
                    BitmapColor<uint8_t> ref_color;
                    int valid_id = -1;
                    for (int i = 0; i < depths.size(); i++){
                        ref_depth = depths.at(i);
                        int ref_col = col + ref_depth.second % num_samp;
                        int ref_row = row + ref_depth.second / num_samp;
                        ref_xyz = inv_P * Eigen::Vector4f(ref_col* ref_depth.first,
                            ref_row * ref_depth.first, ref_depth.first, 1.0f);
                        // std::cout << "ref_xyz: " << ref_xyz.transpose() << std::endl;
                        if (options_.roi_fuse){
                            const Eigen::Vector3f rot_xyz = roi_box_.rot * ref_xyz;
                            if (rot_xyz(0) < roi_box_.x_box_min || rot_xyz(0) > roi_box_.x_box_max ||
                                rot_xyz(1) < roi_box_.y_box_min || rot_xyz(1) > roi_box_.y_box_max ||
                                rot_xyz(2) < roi_box_.z_box_min || rot_xyz(2) > roi_box_.z_box_max){
                                continue;
                            }
                        }

                        if (std::isnan(ref_xyz.x()) || std::isinf(ref_xyz.x()) ||
                            std::isnan(ref_xyz.y()) || std::isinf(ref_xyz.y()) ||
                            std::isnan(ref_xyz.z()) || std::isinf(ref_xyz.z())) {
                            continue;
                        }
                        ref_normal = R.inverse() * Eigen::Vector3f(ref_normal_map.Get(ref_row, ref_col, 0), 
                            ref_normal_map.Get(ref_row, ref_col, 1), ref_normal_map.Get(ref_row, ref_col, 2));
                        
                        auto vis = ConvertVisibility(problem_id, vis_map, ref_row, ref_col);
                        ref_pnt_vis.insert(vis.begin(), vis.end());

                        float delt_depth = delt_depth_map.Get(ref_row, ref_col);
                        if (delt_depth < 1e-7 || delt_depth > 1.0f){
                            delt_depth = 1.0f;
                        }
                        ref_score = ComputeScore(problem_id, ref_xyz, delt_depth, vis, ref_row, ref_col);
                        if (!prior_ply){
                            images[image_idx].GetBitmap().GetPixel(ref_col, ref_row, &ref_color);
                        }
                        valid_id = i;
                        break;
                    }
                    if (valid_id < 0){
                        continue;
                    }
                    int win_id = (col - 3) / win_step + (row - 3) / win_step * num_samp_col;

                    std::vector<Eigen::Vector3f> v_xyz;
                    std::vector<std::set<uint32_t>> v_vis;
                    std::vector<float> v_sco;

                    if (valid_id == depths.size() - 1){
                        PlyPoint pnt;
                        pnt.x = ref_xyz(0);
                        pnt.y = ref_xyz(1);
                        pnt.z = ref_xyz(2);
                        pnt.nx = ref_normal(0);
                        pnt.ny = ref_normal(1);
                        pnt.nz = ref_normal(2);
                        if (!prior_ply){
                            pnt.r = ref_color.r;
                            pnt.g = ref_color.g;
                            pnt.b = ref_color.b;
                        }
                        // std::cout << "add1 in image_points" << std::endl;
                        std::vector<uint32_t> vis(ref_pnt_vis.begin(), ref_pnt_vis.end());
                        win_pnts.at(win_id).push_back(pnt);
                        win_pnts_vis.at(win_id).push_back(vis);
                        win_pnts_sco.at(win_id).push_back(ref_score);
                        continue;
                    }
                    v_xyz.push_back(ref_xyz);
                    v_vis.push_back(ref_pnt_vis);
                    v_sco.push_back(ref_score);

                    for (int i = valid_id + 1; i < depths.size(); i++){
                        if ((ref_depth.first - depths.at(i).first) < ref_depth.first * 0.01 &&
                            (ref_depth.first - depths.at(i).first) > -ref_depth.first * 0.01 && 
                            depths.at(i).first > thd_depth){
                            std::pair<float, int> src_depth = depths.at(i);
                            int src_col = col + src_depth.second % num_samp;
                            int src_row = row + src_depth.second / num_samp;

                            auto vis = ConvertVisibility(problem_id, vis_map, src_row, src_col);
                            v_vis.push_back(std::set<uint32_t>(vis.begin(), vis.end()));

                            Eigen::Vector3f src_xyz = inv_P * Eigen::Vector4f(src_col* src_depth.first,
                                                                            src_row * src_depth.first, src_depth.first, 1.0f);
                            v_xyz.push_back(src_xyz);

                            float delt_depth = delt_depth_map.Get(src_row, src_col);
                            if (delt_depth < 1e-7 || delt_depth > 1.0f){
                                delt_depth = 1.0f;
                            }
                            float src_score = ComputeScore(problem_id, src_xyz, delt_depth, vis, src_row, src_col);
                            v_sco.push_back(src_score);
                        } else {
                            // insert ref_pnt
                            Eigen::Vector3f temp_xyz = Eigen::Vector3f::Zero();
                            std::set<uint32_t> temp_vis;
                            float temp_sco = 0.f;
                            float wgt = 0;
                            for (int i = 0; i < v_xyz.size(); i++){
                                temp_xyz += v_vis.at(i).size() * v_xyz.at(i);
                                wgt += v_vis.at(i).size();

                                temp_vis.insert(v_vis.at(i).begin(), v_vis.at(i).end());

                                temp_sco +=  v_vis.at(i).size() * v_sco.at(i);
                            }
                            temp_xyz /= wgt;
                            temp_sco /= wgt;

                            PlyPoint pnt;
                            pnt.x = temp_xyz(0);
                            pnt.y = temp_xyz(1);
                            pnt.z = temp_xyz(2);
                            pnt.nx = ref_normal(0);
                            pnt.ny = ref_normal(1);
                            pnt.nz = ref_normal(2);
                            if (!prior_ply){
                                pnt.r = ref_color.r;
                                pnt.g = ref_color.g;
                                pnt.b = ref_color.b;
                            }
                            std::vector<uint32_t> vis(temp_vis.begin(), temp_vis.end());

                            win_pnts.at(win_id).push_back(pnt);
                            win_pnts_vis.at(win_id).push_back(vis);
                            win_pnts_sco.at(win_id).push_back(temp_sco);

                            // new ref_depth
                            ref_depth = depths.at(i);
                            int ref_col = col + ref_depth.second % num_samp;
                            int ref_row = row + ref_depth.second / num_samp;
                            ref_xyz = inv_P * Eigen::Vector4f(ref_col* ref_depth.first,
                                ref_row * ref_depth.first, ref_depth.first, 1.0f);
                                
                            if (options_.roi_fuse){
                                const Eigen::Vector3f rot_xyz = roi_box_.rot * ref_xyz;
                                if (rot_xyz(0) < roi_box_.x_box_min || rot_xyz(0) > roi_box_.x_box_max ||
                                    rot_xyz(1) < roi_box_.y_box_min || rot_xyz(1) > roi_box_.y_box_max ||
                                    rot_xyz(2) < roi_box_.z_box_min || rot_xyz(2) > roi_box_.z_box_max){
                                    continue;
                                }
                            }

                            if (std::isnan(ref_xyz.x()) || std::isinf(ref_xyz.x()) ||
                                std::isnan(ref_xyz.y()) || std::isinf(ref_xyz.y()) ||
                                std::isnan(ref_xyz.z()) || std::isinf(ref_xyz.z())) {
                                continue;
                            }
                            ref_normal = R.inverse() * Eigen::Vector3f(ref_normal_map.Get(ref_row, ref_col, 0), 
                                ref_normal_map.Get(ref_row, ref_col, 1), ref_normal_map.Get(ref_row, ref_col, 2));
                            
                            ref_pnt_vis.clear();
                            auto ref_vis = ConvertVisibility(problem_id, vis_map, ref_row, ref_col);
                            ref_pnt_vis.insert(ref_vis.begin(), ref_vis.end());

                            float delt_depth = delt_depth_map.Get(ref_row, ref_col);
                            if (delt_depth < 1e-7 || delt_depth > 1.0f){
                                delt_depth = 1.0f;
                            }
                            ref_score = ComputeScore(problem_id, ref_xyz, delt_depth, vis, ref_row, ref_col);
                            if (!prior_ply){
                                images[image_idx].GetBitmap().GetPixel(ref_col, ref_row, &ref_color);
                            }

                            v_xyz.clear();
                            v_vis.clear();
                            v_sco.clear();
                            v_xyz.push_back(ref_xyz);
                            v_vis.push_back(ref_pnt_vis);
                            v_sco.push_back(ref_score);
                        }
                    }
                    {
                        Eigen::Vector3f temp_xyz = Eigen::Vector3f::Zero();
                        std::set<uint32_t> temp_vis;
                        float temp_sco = 0;
                        float wgt = 0;
                        for (int i = 0; i < v_xyz.size(); i++){
                            temp_xyz += v_vis.at(i).size() * v_xyz.at(i);
                            wgt += v_vis.at(i).size();

                            temp_vis.insert(v_vis.at(i).begin(), v_vis.at(i).end());

                            temp_sco +=  v_vis.at(i).size() * v_sco.at(i);
                        }
                        temp_xyz /= wgt;
                        temp_sco /= wgt;

                        PlyPoint pnt;
                        pnt.x = temp_xyz(0);
                        pnt.y = temp_xyz(1);
                        pnt.z = temp_xyz(2);
                        pnt.nx = ref_normal(0);
                        pnt.ny = ref_normal(1);
                        pnt.nz = ref_normal(2);
                        if (!prior_ply){
                            pnt.r = ref_color.r;
                            pnt.g = ref_color.g;
                            pnt.b = ref_color.b;
                        }
                        std::vector<uint32_t> vis(temp_vis.begin(), temp_vis.end());

                        win_pnts.at(win_id).push_back(pnt);
                        win_pnts_vis.at(win_id).push_back(vis);
                        win_pnts_sco.at(win_id).push_back(temp_sco);
                    }
                }
            }
            int num_win_pnts = 0;
            for (int win_id = 0; win_id < num_samp_row * num_samp_col; win_id++){
                num_win_pnts += win_pnts.at(win_id).size();
            }
            image_points.reserve(num_win_pnts);
            image_points_visibility.reserve(num_win_pnts);
            image_points_score.reserve(num_win_pnts);
            for (int win_id = 0; win_id < num_samp_row * num_samp_col; win_id++){
                for (int i = 0; i < win_pnts.at(win_id).size(); i++){
                    image_points.push_back(win_pnts.at(win_id).at(i));
                    image_points_visibility.push_back(win_pnts_vis.at(win_id).at(i));
                    image_points_score.push_back(win_pnts_sco.at(win_id).at(i));
                }
            }
#endif
            // if (options.verbose) {
            //     const auto &image_name = model.GetImageName(image_idx);
            //     const std::string file_name = StringPrintf("%s.%s.%s", 
            //         image_name.c_str(), options.output_type.c_str(), DEPTH_EXT);
            //     const std::string depth_map_path = JoinPaths(depth_maps_path, file_name);
            //     delt_depth_map.ToBitmap().Write(StringPrintf("%s-deplc.jpg", 
            //                                                 depth_map_path.c_str()));
            // } 
            if (images_exclusive.at(image_idx)){
                ref_normal_map = std::move(NormalMap());
            }
            problem_points[points_id].swap(image_points);
            problem_points_score[points_id].swap(image_points_score);
            problem_points_visibility[points_id].swap(image_points_visibility);
        };

        int points_id = 0;
        for (auto problem_id : problem_ids){
            int image_idx = problems_[problem_id].ref_image_idx;

            DepthMap ref_depth_map = cross_filter_cuda.GetDeduplicDepthMap(
                                    image_idx, depth_maps_info_.at(image_idx));
            Mat<uint32_t> vis_map = cross_filter_cuda.GetVisibilyMap(image_idx);
            if (!ref_depth_map.IsValid() || !vis_map.IsValid()){
                continue;
            }
            save_thread_pool->AddTask(SavePoints, 
                                    points_id, 
                                    problem_id, 
                                    ref_depth_map,
                                    vis_map);
            points_id++;
            // break;
        }
        save_thread_pool->Wait();
        std::cout << StringPrintf("Cluster#%d SavePoints Cost Time: %.3fmin\n", cluster_id, save_timer.ElapsedMinutes());

        cross_filter_cuda.DestroyDepthMapTexture();

        for (int i = 0; i < points_id; i++){
            if (problem_points.at(i).empty()){
                continue;
            }
            cluster_points_.at(cluster_id).insert(cluster_points_.at(cluster_id).end(), 
                problem_points.at(i).begin(),
                problem_points.at(i).end());
            cluster_points_score_.at(cluster_id).insert(
                cluster_points_score_.at(cluster_id).end(), 
                problem_points_score.at(i).begin(),
                problem_points_score.at(i).end());
            cluster_points_visibility_.at(cluster_id).insert(
                cluster_points_visibility_.at(cluster_id).end(), 
                problem_points_visibility.at(i).begin(),
                problem_points_visibility.at(i).end());
        }
        std::cout << "Fusion ... cluster - " << cluster_id << " point size =  " 
            << cluster_points_.at(cluster_id).size() << std::endl;
        // WriteBinaryPlyPoints("./test_pnts.ply", cluster_points_.at(cluster_id), true, true);
        // exit(-1);
    }
}

Eigen::RowMatrix3f EulerToRotationMatrix(double roll, double yaw, double pitch){
    Eigen::AngleAxisf rollAngle = 
        Eigen::AngleAxisf(roll / 180 * M_PI, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle = 
        Eigen::AngleAxisf(pitch / 180 * M_PI, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle = 
        Eigen::AngleAxisf(yaw / 180 * M_PI, Eigen::Vector3f::UnitZ());
    
    Eigen::Quaternionf q = pitchAngle * rollAngle * yawAngle;

    return q.matrix();
}

#define THREADS_PER_GPU 3

PanoramaPatchMatchController::PanoramaPatchMatchController(
    const PatchMatchOptions& options,
    const std::string& image_path,
    const std::string& workspace_path,
    const int reconstrction_idx)
    : options_(options),
      image_path_(image_path),
      workspace_path_(workspace_path),
      num_reconstruction_(0),
      select_reconstruction_idx_(reconstrction_idx) {}

void PanoramaPatchMatchController::ReadGpuIndices() {
    gpu_indices_ = CSVToVector<int>(options_.gpu_index);
    if (gpu_indices_.size() == 1 && gpu_indices_[0] == -1) {
        const int num_cuda_devices = GetNumCudaDevices();
        CHECK_GT(num_cuda_devices, 0);
        gpu_indices_.resize(num_cuda_devices * THREADS_PER_GPU);
        for(int i = 0; i < num_cuda_devices * THREADS_PER_GPU; ++i) {
            gpu_indices_[i] = i / THREADS_PER_GPU;
        }
    } else {
        const int num_cuda_devices = gpu_indices_.size();
        std::vector<int> gpu_indices_tmp(num_cuda_devices * THREADS_PER_GPU);
        for (int i = 0; i < num_cuda_devices; ++i) {
            for (int j = 0; j < THREADS_PER_GPU; ++j) {
                gpu_indices_tmp[i * THREADS_PER_GPU + j] = gpu_indices_[i];
            }
        }
        gpu_indices_ = gpu_indices_tmp;
    }
}

void PanoramaPatchMatchController::ReadWorkspace() {
    std::cout << "Reading workspace..." << std::endl;
    for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }

        num_reconstruction_++;
    }
    std::cout << StringPrintf("Read %d reconstructions", num_reconstruction_) 
              << std::endl;
}


void PanoramaPatchMatchController::PyramidRefineDepthNormal(
    const PatchMatchOptions& options,
    const int level, 
    bool fill_flag) {
  
    PrintHeading1(StringPrintf("Refine depth map (level =  %d)", level));
    Timer reading_timer;
    reading_timer.Start();
    
    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> refine_map_thread_pool_;
    int num_threads = std::min(num_eff_threads, (int)problems_.size());
    refine_map_thread_pool_.reset(new ThreadPool(num_eff_threads));
    std::cout << "Number of problems:  " << problems_.size() << std::endl;

    for (int problem_idx = 0; problem_idx < problems_.size(); ++problem_idx) {
        Problem& problem = problems_.at(problem_idx);
        refine_map_thread_pool_->AddTask(FillDepthMap,
            options, JoinPaths(component_path_, STEREO_DIR), problem,
            perspective_images_.at(problem.ref_image_idx),
            perspective_image_names_.at(problem.ref_image_idx),
            level, fill_flag, PHOTOMETRIC_TYPE);
    }
    refine_map_thread_pool_->Wait();

    reading_timer.PrintSeconds();
}

void PanoramaPatchMatchController::Run() {
    ReadWorkspace();
    ReadGpuIndices();

    thread_pool_.reset(new ThreadPool(gpu_indices_.size()));

    // for (size_t reconstruction_idx = 0; reconstruction_idx < num_reconstruction_;
    //      reconstruction_idx++) {
    
    size_t reconstruction_begin = select_reconstruction_idx_ < 0 ? 0 : select_reconstruction_idx_;
    num_reconstruction_ = select_reconstruction_idx_ < 0 ? num_reconstruction_ : select_reconstruction_idx_+1;
    for (size_t reconstruction_idx = reconstruction_begin; reconstruction_idx < num_reconstruction_; 
         reconstruction_idx++) {

        PrintHeading1(StringPrintf("Reconstructing# %d", reconstruction_idx));

        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(reconstruction_idx));

        Workspace::Options workspace_options;
        workspace_options.max_image_size = options_.max_image_size;
        workspace_options.image_as_rgb = false;
        workspace_options.cache_size = options_.cache_size;
        workspace_options.image_path = image_path_;
        workspace_options.workspace_path = reconstruction_path;
        workspace_options.workspace_format = "panorama";
        workspace_options.input_type = 
            options_.geom_consistency ? PHOTOMETRIC_TYPE : "";

        workspace_.reset(new Workspace(workspace_options));

        perspective_image_names_.clear();
        perspective_images_.clear();
        perspective_src_images_idx_.clear();
#ifdef CACHED_DEPTH_MAP
        depth_maps_.clear();
        normal_maps_.clear();
        conf_maps_.clear();
#endif
        problems_.clear();
        depth_ranges_.clear();

        component_path_ = JoinPaths(reconstruction_path, DENSE_DIR);

        PrepareData();
        ReadProblems();

        BuildPriorModel(reconstruction_path);

        // parameters setting
        bool para_ref = true;
        int para_max_level = options_.pyramid_max_level;
        int para_delta_level = options_.pyramid_delta_level;

        struct PyramidParamters
        {
            std::string input_type;
            std::string output_type;
            bool init_from_model;
            bool init_from_visible_map;
            bool init_from_delaunay;
            bool init_from_global_map;
            bool geom_consistency;

            bool filter;
            bool conf_filter;

            bool ref_flag;
            bool fill_flag;
            int level;
            int max_level; 
            bool plane_regularizer;
        };
        
        std::vector<PyramidParamters> pyramid_parameters;

        int iter_level = para_max_level;
        do {
            if (iter_level == para_max_level) {
                PyramidParamters pyramid_parameter_photo = 
                    { iter_level == para_max_level ? PHOTOMETRIC_TYPE : GEOMETRIC_TYPE, PHOTOMETRIC_TYPE,
                    options_.init_from_model, options_.init_from_visible_map, 
                    options_.init_from_delaunay, options_.init_from_global_map,
                    false, options_.filter && (iter_level == 0), options_.conf_filter, 
                    false, false, iter_level, 
                    para_max_level, false};
                pyramid_parameters.push_back(pyramid_parameter_photo);
            } else if (para_ref) {
                PyramidParamters pyramid_parameter_photo = 
                    { PHOTOMETRIC_TYPE, PHOTOMETRIC_TYPE, false, false, false, false, false, true, true, true, false, 
                    iter_level, iter_level, false};
                pyramid_parameters.push_back(pyramid_parameter_photo);
            }

            if (options_.geom_consistency) {
                PyramidParamters pyramid_parameter_geom = 
                    { PHOTOMETRIC_TYPE, GEOMETRIC_TYPE,
                    options_.init_from_model, options_.init_from_visible_map,
                    options_.init_from_delaunay, options_.init_from_global_map, 
                    options_.geom_consistency, options_.filter && (iter_level == 0), 
                    options_.conf_filter, false, para_ref, 
                    iter_level, para_max_level, options_.plane_regularizer };
                pyramid_parameters.push_back(pyramid_parameter_geom);
            }
            iter_level -= para_delta_level;
        } while(iter_level >= 0);

        float factor = 1.0f;
#ifdef CACHED_DEPTH_MAP
        {
            depth_maps_.resize(perspective_images_.size());
            normal_maps_.resize(perspective_images_.size());
            conf_maps_.resize(perspective_images_.size());
        }
#endif

        for (auto iter : pyramid_parameters){
            auto iter_options = options_;
            iter_options.output_type = iter.output_type;
            iter_options.init_from_model = iter.init_from_model;
            iter_options.init_from_visible_map = iter.init_from_visible_map;
            iter_options.init_from_delaunay = iter.init_from_delaunay;
            iter_options.init_from_global_map = iter.init_from_global_map;
            iter_options.geom_consistency = iter.geom_consistency;
            iter_options.init_depth_random = !(iter.geom_consistency ||
                                               iter.init_from_visible_map ||
                                               iter.init_from_global_map ||
                                               iter.init_from_model ||
                                               iter.level != iter.max_level);

            iter_options.filter = iter.filter;
            iter_options.conf_filter = iter.conf_filter;

            iter_options.plane_regularizer = iter.plane_regularizer;

            // reduce random disturbance 
            float factor_delta_level = 1/std::pow(2,iter.max_level-iter.level);
            if (iter.input_type == REF_TYPE){
                factor_delta_level = 4;
                iter_options.random_angle1_range = 
                    factor_delta_level * options_.random_angle1_range;
                iter_options.random_angle2_range = 
                    factor_delta_level * options_.random_angle2_range;
                iter_options.random_depth_ratio = 
                    factor_delta_level * options_.random_depth_ratio;
            }
            if (iter.level != iter.max_level && 
                ((para_ref && !iter.ref_flag) || 
                 (!para_ref && iter.geom_consistency))) {
                PyramidRefineDepthNormal(iter_options, iter.level, iter.fill_flag);
            }

#ifdef CACHED_DEPTH_MAP
            if (iter.init_from_model || (iter.level == iter.max_level && 
                (!iter.geom_consistency || iter.init_from_model))) {
                depth_maps_.clear();
                normal_maps_.clear();
                conf_maps_.clear();
                depth_maps_.resize(perspective_images_.size());
                normal_maps_.resize(perspective_images_.size());
                conf_maps_.resize(perspective_images_.size());
            }
#endif
            for (int problem_idx = 0; problem_idx < problems_.size(); 
                ++problem_idx) {
                thread_pool_->AddTask(&PanoramaPatchMatchController::ProcessProblem,
                                    this,
                                    iter_options,
                                    problem_idx,
                                    iter.level, 
                                    iter.ref_flag,
                                    iter.input_type);
            }
            thread_pool_->Wait();

            if (iter_options.filter && iter_options.geo_filter) {
                workspace_options.input_type = iter.output_type;
                workspace_.reset(new Workspace(workspace_options));
                iter_options.geom_consistency = 1;
                iter_options.init_depth_random = 0;
                for (int problem_idx = 0; problem_idx < problems_.size(); 
                    ++problem_idx) {
                    thread_pool_->AddTask(
                        &PanoramaPatchMatchController::ProcessFilterProblem,
                        this,
                        iter_options,
                        problem_idx,
                        iter.level, 
                        iter.ref_flag,
                        iter.input_type);
                }
                thread_pool_->Wait();
            }
        }

        GetTimer().PrintMinutes();
    }
}

void PanoramaPatchMatchController::PrepareData() {
    std::cout << "Create Perspective workspace..." << std::endl;

    const float min_triangulation_angle_rad =
        DegToRad(options_.min_triangulation_angle);
    const int perspective_image_size = options_.max_image_size;
    const Model& model = workspace_->GetModel();
    const int num_image = model.images.size();

    std::string workspace_sparse_path = JoinPaths(component_path_, SPARSE_DIR);

    CreateDirIfNotExists(component_path_);
    CreateDirIfNotExists(workspace_sparse_path);

    std::string workspace_stereo_path = JoinPaths(component_path_, STEREO_DIR);
    CreateDirIfNotExists(workspace_stereo_path);

    CreateDirIfNotExists(JoinPaths(workspace_stereo_path, DEPTHS_DIR));
    CreateDirIfNotExists(JoinPaths(workspace_stereo_path, NORMALS_DIR));
    CreateDirIfNotExists(JoinPaths(workspace_stereo_path, CONSISTENCY_DIR));

    std::string workspace_image_path = JoinPaths(component_path_, IMAGES_DIR);
    std::string sparse_file = JoinPaths(workspace_sparse_path, "cameras.bin");
    if (ExistsFile(sparse_file) && ExistsDir(workspace_image_path)) {
        ImportPanoramaWorkspace(component_path_, perspective_image_names_, 
            perspective_images_, perspective_image_ids_,
            perspective_src_images_idx_, depth_ranges_, true);
        
        Panorama panorama;
        const mvs::Image& image = model.images.at(0);
        const int width = image.GetWidth();
        const int height = image.GetHeight();
        panorama.PerspectiveParamsProcess(perspective_image_size, fov_w, fov_h, num_perspective_per_image,
                                                  width, height);
        rmap_idx_ = panorama.GetPanoramaRmapId();

        return;
    }

    CreateDirIfNotExists(workspace_image_path);

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    Eigen::RowMatrix3f Rs[num_perspective_per_image];
    for (int i = 0; i < num_perspective_per_image; ++i) {
        Rs[i] = EulerToRotationMatrix(rolls[i], yaws[i], pitches[i]);
        Rs[i].transposeInPlace();
    }
    {
        std::cout << "Generate Perspective images..." << std::endl;

        std::vector<image_t> image_ids(num_image);
        for (size_t image_idx = 0; image_idx < num_image; ++image_idx) {
            image_ids.at(image_idx) = model.GetImageId(image_idx);
        }

        Panorama panorama;
        const mvs::Image& image = model.images.at(0);
        Bitmap initial_bitmap;
        if (!initial_bitmap.Read(image.GetPath())) {
            std::cerr << "Error! Panorama Initial file path " << image.GetPath() 
                        << " does not exist" << std::endl;
            exit(-1);
        }
        panorama.PerspectiveParamsProcess(perspective_image_size, fov_w, fov_h, num_perspective_per_image,
                                                  initial_bitmap.Width(), initial_bitmap.Height());
        rmap_idx_ = panorama.GetPanoramaRmapId();

        float focal_length;
        int width, height;
        focal_length = panorama.GetPerspectiveFocalLength(0);
        width = panorama.GetPerspectiveWidth(0);
        height = panorama.GetPerspectiveHeight(0);

        float K[9];
        K[0] = focal_length; K[1] = 0.0f;         K[2] = width * 0.5f;
        K[3] = 0.0f;         K[4] = focal_length; K[5] = height * 0.5f;
        K[6] = 0.0f;         K[7] = 0.0f;         K[8] = 1.0f;

        std::cout << "camera params: " << focal_length << " " << width * 0.5 << " " << height * 0.5 << std::endl;
        
        perspective_image_names_.resize(num_image * num_perspective_per_image);
        perspective_images_.resize(num_image * num_perspective_per_image);
        perspective_image_ids_.resize(num_image * num_perspective_per_image, -1);

        auto ConvertPanorama = [&](int ref_image_idx) {
            std::cout << "Convert Panorama#" << ref_image_idx << std::endl;
            const mvs::Image& image = model.images.at(ref_image_idx);
            Bitmap bitmap;
            if (!bitmap.Read(image.GetPath())) {
                std::cerr << "Error! File " << image.GetPath() 
                          << " does not exist" << std::endl;
                return;
            }
           
            Eigen::Vector3f ref_T_tmp(image.GetT());
            Eigen::RowMatrix3f ref_R_tmp(image.GetR());                                   

            std::vector<Bitmap> perspective_images(num_perspective_per_image);
            for (int i = 0; i < num_perspective_per_image; ++i) {
                panorama.PanoramaToPerspectives(&bitmap, perspective_images);
            }

            std::string image_name = model.GetImageName(ref_image_idx);
            const auto pos = image_name.find_last_of('.', image_name.length());
            std::string image_name_base = image_name.substr(0, pos);
            for (int i = 0; i < perspective_images.size(); ++i) {
                const int idx = ref_image_idx * num_perspective_per_image + i;
                std::string iimage_name = StringPrintf("%s_%d.jpg", 
                    image_name_base.c_str(), i);
                std::string iimage_path = 
                    JoinPaths(workspace_image_path, iimage_name);
                const std::string parent_path = GetParentDir(iimage_path);
                if (!ExistsPath(parent_path)) {
                    boost::filesystem::create_directories(parent_path);
                }
                perspective_image_names_[idx] = iimage_name;
                perspective_images[i].Write(iimage_path);

                Eigen::RowMatrix3f R = Rs[i] * ref_R_tmp;
                Eigen::Vector3f T = Rs[i] * ref_T_tmp;
                mvs::Image p_image(iimage_path, width, height, K, 
                                   R.data(), T.data(), false);
                // Bitmap bitmap;
                // bitmap.Read(iimage_path, false);
                // p_image.SetBitmap(bitmap);
                p_image.SetBitmap(perspective_images[i]);
                p_image.GetBitmap().ConvertToGray();
                perspective_images_[idx] = p_image;
                perspective_image_ids_[idx] = image_ids.at(ref_image_idx);
            }
        };

        for (int ref_image_idx = 0; ref_image_idx < num_image; 
             ++ref_image_idx) {
            thread_pool->AddTask(ConvertPanorama, ref_image_idx);
        }
        thread_pool->Wait();
    }

    std::cout << "Compute Adjoin Perspective images..." << std::endl;

    std::vector<std::unordered_map<int, int>> shared_num_points;
    std::vector<std::unordered_map<int, float>> triangulation_angles;
    if (shared_num_points.empty()) {
        shared_num_points = model.ComputeSharedPoints();
    }
    if (triangulation_angles.empty()) {
        const float kTriangulationAnglePercentile = 75;
        triangulation_angles =
            model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
    }

    std::vector<std::vector<int> > src_image_idxs(num_image);
    auto ComputeOverlapping = [&](int ref_image_idx) {
        const mvs::Image& image = model.images.at(ref_image_idx);

        const auto& overlapping_images = shared_num_points.at(ref_image_idx);
        const auto& overlapping_triangulation_angles = 
            triangulation_angles.at(ref_image_idx);

        std::vector<std::pair<int, int> > src_images;
        src_images.reserve(overlapping_images.size());
        for (const auto& image : overlapping_images) {
            if (overlapping_triangulation_angles.at(image.first) >=
                min_triangulation_angle_rad) {
                src_images.emplace_back(image.first, image.second);
            }
        }

        const size_t eff_max_num_src_images = 
            std::min(src_images.size(), options_.max_num_src_images);

        std::partial_sort(src_images.begin(),
                          src_images.begin() + eff_max_num_src_images,
                          src_images.end(),
                          [](const std::pair<int, int>& image1,
                            const std::pair<int, int>& image2) {
                            return image1.second > image2.second;
                          });

        src_image_idxs[ref_image_idx].reserve(eff_max_num_src_images);
        for (size_t j = 0; j < eff_max_num_src_images; ++j) {
            src_image_idxs[ref_image_idx].push_back(src_images[j].first);
        }
    };

    for (int ref_image_idx = 0; ref_image_idx < num_image; ++ref_image_idx) {
        thread_pool->AddTask(ComputeOverlapping, ref_image_idx);
    }
    thread_pool->Wait();

    const int strip = num_perspective_per_image;
    perspective_src_images_idx_.resize(num_image * num_perspective_per_image);
    for (int i = 0; i < num_image; ++i) {
        for (const auto& image_idx : src_image_idxs[i]) {
            for (int j = 0; j < num_perspective_per_image; ++j) {
                perspective_src_images_idx_[i * strip + j].push_back(image_idx * strip + j);
            }
        }
    }

    std::cout << "Compute Depth Range..." << std::endl;
    // compute depth ranges for each perspective image.
    std::vector<std::vector<int> > mappoints_per_image(num_image);
    std::vector<Model::Point> points = model.points;
    for (int i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        for (const auto& image_idx : point.track) {
            mappoints_per_image[image_idx].push_back(i);
        }
    }

    std::vector<std::vector<float>> depths(perspective_images_.size());
    auto ComputeDepthRange = [&](int image_idx) {
        const auto& mappoint_ids = mappoints_per_image[image_idx];
        for (int i = 0; i < num_perspective_per_image; ++i) {
            const int perspective_idx = 
                image_idx * num_perspective_per_image + i;
            const mvs::Image& image = perspective_images_[perspective_idx];

            const int width = image.GetWidth();
            const int height = image.GetHeight();
            float focal = image.GetK()[0];
            float principal_x = image.GetK()[2];
            float principal_y = image.GetK()[5];

            Eigen::Map<const Eigen::Vector3f> R0(&image.GetR()[0]);
            Eigen::Map<const Eigen::Vector3f> R1(&image.GetR()[3]);
            Eigen::Map<const Eigen::Vector3f> R2(&image.GetR()[6]);
            const float T0 = image.GetT()[0];
            const float T1 = image.GetT()[1];
            const float T2 = image.GetT()[2];

            int num_visible_mappoint = 0;
            for (const auto& mappoint_id : mappoint_ids) {
                const Model::Point& point = points[mappoint_id];
                Eigen::Vector3f X(point.x, point.y, point.z);
                const float depth = R2.dot(X) + T2;
                const float x = R0.dot(X) + T0;
                const float y = R1.dot(X) + T1;
                int u = focal * x / depth + principal_x;
                int v = focal * y / depth + principal_y;
                if (u >= 0 && u < width && v >= 0 && v < height && depth > 0) {
                    depths[perspective_idx].push_back(depth);
                    num_visible_mappoint++;
                }
            }
            if (num_visible_mappoint == 0) {
                depths[perspective_idx].push_back(1e-5f);
                depths[perspective_idx].push_back(100.0f);
            }
        }
    };

    for (int image_idx = 0; image_idx < num_image; ++image_idx) {
        thread_pool->AddTask(ComputeDepthRange, image_idx);
    }
    thread_pool->Wait();

    depth_ranges_.resize(depths.size());
    for (size_t image_idx = 0; image_idx < depth_ranges_.size(); ++image_idx) {
        auto& depth_range = depth_ranges_[image_idx];

        auto& image_depths = depths[image_idx];

        if (image_depths.empty()) {
            depth_range.first = -1.0f;
            depth_range.second = -1.0f;
            continue;
        }

        std::sort(image_depths.begin(), image_depths.end());

        const float kMinPercentile = 0.01f;
        const float kMaxPercentile = 0.99f;
        depth_range.first = image_depths[image_depths.size() * kMinPercentile];
        depth_range.second = image_depths[image_depths.size() * kMaxPercentile];

        const float kStretchRatio = 0.25f;
        depth_range.first *= (1.0f - kStretchRatio);
        depth_range.second *= (1.0f + kStretchRatio);
        std::cout << "depth range#" << image_idx << ": " 
                  << depth_ranges_[image_idx].first << ", " 
                  << depth_ranges_[image_idx].second 
                  << std::endl;
    }

    // for visualization with colmap gui.
    std::ofstream ofs(workspace_stereo_path + "/patch-match.cfg", 
                      std::ofstream::out);
    for (const auto& image_name : perspective_image_names_) {
        ofs << image_name << std::endl;
        ofs << "__auto__, " << options_.max_num_src_images << std::endl;
    }
    ofs.close();

    ExportPanoramaWorkspace(component_path_, perspective_image_names_, 
        perspective_images_, perspective_image_ids_,
        perspective_src_images_idx_, depth_ranges_);
}

void PanoramaPatchMatchController::ReadProblems() {
    std::cout << "Reading configuration..." << std::endl;

    problems_.clear();

    for (int ref_image_idx = 0; ref_image_idx < perspective_images_.size();
         ++ref_image_idx) {
        Problem problem;
        problem.ref_image_idx = ref_image_idx;
        problem.src_image_idxs = perspective_src_images_idx_[ref_image_idx];
        problem.src_image_scales.resize(problem.src_image_idxs.size(), 1.0f);

        if (problem.src_image_idxs.empty()) {
            std::cout << StringPrintf(
                      "WARNING: Ignoring reference image %s, because it has no "
                      "source images.",
                      perspective_image_names_[ref_image_idx].c_str())
                      << std::endl;
        } else {
            problems_.push_back(problem);
            used_images_.insert(problem.ref_image_idx);
        }
    }
    std::cout << StringPrintf("Configuration has %d problems...",
                              problems_.size())
              << std::endl;
}

void PanoramaPatchMatchController::BuildPriorModel(const std::string& workspace_path) {
    if (options_.init_from_model) {
        ReadTriangleMeshObj(JoinPaths(workspace_path, DENSE_DIR, "model.obj"), 
                            prior_model_, false);
    } else if (options_.init_from_global_map) {
        std::string model_path = 
            JoinPaths(workspace_path, DENSE_DIR, "prior_model.obj");
        if (ExistsFile(model_path)) {
            ReadTriangleMeshObj(model_path, prior_model_, false);
            return;
        }
        mvs::PointCloud pointcloud;

        const Model& model = workspace_->GetModel();
        const auto& points = model.points;
        pointcloud.pointViews.Resize(points.size());
        pointcloud.pointWeights.Resize(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            const Eigen::Vector3f pt(points[i].x, points[i].y, points[i].z);
            const std::vector<int>& vis_pt = points[i].track;
            pointcloud.points.Insert(pt);
            pointcloud.colors.Insert(Eigen::Vector3ub(0, 0, 0));
            for (const auto& view : vis_pt) {                 
                pointcloud.pointViews[i].Insert(view);
                pointcloud.pointWeights[i].Insert(1.0f);
            }
        }

        auto options = options_.delaunay_options;

        mvs::DelaunayMeshing(prior_model_, pointcloud, model.images, 
                             options.sampInsert, options.dist_insert, options.diff_depth, 2.0f, -1.0f,
                             false, 4, 1, 1, 4, 3, 0.1, 1000, 400);

        // clean the mesh
        // prior_model_.Clean(1.0, 2.0, 1, 30, 10, false);
        prior_model_.Clean(options.decimate_mesh, options.remove_spurious, 
                           options.remove_spikes, options.close_holes, 
                           options.smooth_mesh, false);
        // extra cleaning trying to close more holes
        // prior_model_.Clean(1.f, 0.f, 1, 30, 0, false);
        prior_model_.Clean(1.f, 0.f, options.remove_spikes, options.close_holes, 
                           0, false);
        // extra cleaning to remove non-manifold problems created by closing holes.
        prior_model_.Clean(1.f, 0.f, false, 0, 0, true);

        prior_model_.RemoveIsolatedPieces(options.num_isolated_pieces);
        prior_model_.ComputeNormals();

        WriteTriangleMeshObj(
            JoinPaths(workspace_path, DENSE_DIR, "prior_model.obj"), 
            prior_model_, false);
    }
}

void PanoramaPatchMatchController::ProcessProblem(
    const PatchMatchOptions& options,
    const size_t problem_idx,
    const int level,
    const bool ref_flag,
    std::string input_type) {
    if (IsStopped()) {
        return;
    }

    auto& problem = problems_.at(problem_idx);
    const int gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());
    CHECK_GE(gpu_index, -1);

    auto image_name = perspective_image_names_[problem.ref_image_idx];
    auto file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), options.output_type.c_str(), DEPTH_EXT);
    auto stereo_path = JoinPaths(component_path_, STEREO_DIR);
    auto depth_map_path = JoinPaths(stereo_path, DEPTHS_DIR, file_name);
    auto normal_map_path = JoinPaths(stereo_path, NORMALS_DIR, file_name);
    auto conf_map_path = JoinPaths(stereo_path, CONFS_DIR, file_name);
    auto curvature_map_path = JoinPaths(stereo_path, CURVATURES_DIR, file_name);
    const std::string consistency_graph_path = JoinPaths(
        component_path_, STEREO_DIR, CONSISTENCY_DIR, file_name);
    const std::string workspace_image_path = 
        JoinPaths(component_path_, IMAGES_DIR);

    if (!options.init_from_model && 
        ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
        (!options.write_consistency_graph ||
        ExistsFile(consistency_graph_path)) && options.pyramid_max_level == 0) {
        return;
    }

    PrintHeading1(StringPrintf("Reconstructing view %d / %d  (level = %d)", 
                               problem_idx + 1, problems_.size(), level));

    auto patch_match_options = options;
    patch_match_options.gpu_index = std::to_string(gpu_index);
    patch_match_options.thread_index = thread_pool_->GetThreadIndex() % THREADS_PER_GPU;

#ifndef CACHED_DEPTH_MAP
    std::vector<DepthMap> depth_maps_;
    std::vector<NormalMap> normal_maps_;
    depth_maps_.resize(perspective_images_.size());
    normal_maps_.resize(perspective_images_.size());
#endif
    problem.depth_maps = &depth_maps_;
    problem.normal_maps = &normal_maps_;

    std::vector<mvs::Image> perspective_images_process;
    if (options.pyramid_max_level == 0) {
        problem.images = &perspective_images_;
    } else {
        perspective_images_process = perspective_images_;
        problem.images = &perspective_images_process;
    }
    Image image = perspective_images_.at(problem.ref_image_idx);

    float factor_level = 1 / std::pow(2,level);
    float max_size = std::max(image.GetHeight(), image.GetWidth());
    float min_image_size = std::min(480.f, max_size);
    factor_level = std::max(factor_level, min_image_size / max_size);

    if (!patch_match_options.geom_consistency) {
        image.Rescale(factor_level);
        depth_maps_.at(problem.ref_image_idx) = DepthMap(
            image.GetWidth(), image.GetHeight(), 
            patch_match_options.depth_min, patch_match_options.depth_max);
        normal_maps_.at(problem.ref_image_idx) = NormalMap(image.GetWidth(),
                                                           image.GetHeight());
        if (level != patch_match_options.pyramid_max_level) {
            const std::string file_name = StringPrintf("%s.%s.%s", 
                image_name.c_str(), input_type.c_str(), DEPTH_EXT);
            const std::string ref_depth_map_path = JoinPaths(
                component_path_, STEREO_DIR, DEPTHS_DIR, file_name);
            const std::string ref_normal_map_path = JoinPaths(
                component_path_, STEREO_DIR, NORMALS_DIR, file_name);
            if (!ExistsFile(ref_depth_map_path) || 
                !ExistsFile(ref_normal_map_path)) {
                return;
            }
            depth_maps_.at(problem.ref_image_idx).Read(ref_depth_map_path);
            normal_maps_.at(problem.ref_image_idx).Read(ref_normal_map_path);
            if (!depth_maps_.at(problem.ref_image_idx).IsValid() ||
                !normal_maps_.at(problem.ref_image_idx).IsValid()) {
                return;
            }
        } else if (patch_match_options.init_from_visible_map || patch_match_options.init_from_rgbd) {
            std::cout << "InitDepthMap" << std::endl;
            std::pair<float, float> depth_range = InitDepthMap(
                patch_match_options, problem_idx, image,
                JoinPaths(component_path_, STEREO_DIR));
            depth_ranges_.at(problem_idx) = depth_range;
        } else if (patch_match_options.init_from_global_map || 
                   patch_match_options.init_from_model) {
            std::pair<float, float> depth_range = Model2Depth(
                prior_model_, image, problem);
            depth_ranges_.at(problem_idx) = depth_range;
        }
        // if (patch_match_options.verbose && 
        //     level == patch_match_options.pyramid_max_level) {
        //     depth_maps_.at(problem.ref_image_idx).ToBitmap().Write(depth_map_path + ".init.jpg");
        //     normal_maps_.at(problem.ref_image_idx).ToBitmap().Write(normal_map_path + ".init.jpg");
        // }
    }

    if (patch_match_options.depth_min < 0 || 
        patch_match_options.depth_max < 0) {
        patch_match_options.depth_min = depth_ranges_.at(problem_idx).first;
        patch_match_options.depth_max = depth_ranges_.at(problem_idx).second;
        std::cout << patch_match_options.depth_min << ", " 
                  << patch_match_options.depth_max << std::endl;
        CHECK(patch_match_options.depth_min >= 0 &&
              patch_match_options.depth_max > 0)
            << " - You must manually set the minimum and maximum depth, "
               "since no sparse model is provided in the workspace.";
    }

    {
      // Collect all used images in current problem.
      std::unordered_set<int> used_image_idxs(problem.src_image_idxs.begin(),
                                              problem.src_image_idxs.end());
      used_image_idxs.insert(problem.ref_image_idx);

      patch_match_options.filter_min_num_consistent =
          std::min(static_cast<int>(used_image_idxs.size()) - 1,
                  patch_match_options.filter_min_num_consistent);

    
      int used_image_num = 0;
      for (const auto image_idx : used_image_idxs) {
        if (used_images_.find(image_idx) == used_images_.end()) {
            continue;
        }              
        used_image_num ++;
      }
      if (used_image_num < 2){
        std::cout << StringPrintf(
                  "WARNING: Ignoring image %s, because source images do not "
                  "exist.", image_name.c_str())
                    << std::endl;
        return;
      }
#ifdef CACHED_DEPTH_MAP
      // Only access workspace from one thread at a time and only spawn resample
      // threads from one master thread at a time.
      std::unique_lock<std::mutex> lock(workspace_mutex_);
#endif

      std::unordered_set<int> invalid_image_idxs;
      std::cout << "Reading inputs..." << std::endl;
      for (const auto image_idx : used_image_idxs) {
        if (used_images_.find(image_idx) == used_images_.end()) {
            invalid_image_idxs.insert(image_idx);
            continue;
        }
        mvs::Image* image;
        if (options.pyramid_max_level == 0) {
            image = &perspective_images_.at(image_idx);
        } else {
            image = &perspective_images_process.at(image_idx);
        }
        if (image->GetBitmap().Width() == 0) {
            Bitmap bitmap;
            if (!bitmap.Read(image->GetPath(), workspace_->GetOptions().image_as_rgb)) {
                invalid_image_idxs.insert(image_idx);
                continue;
            }
            if (options.pyramid_max_level > 0) { // Multi-scale
                image->SetBitmap(bitmap);
            }
            perspective_images_.at(image_idx).SetBitmap(bitmap);
        }
        // if (patch_match_options.geom_consistency/* && 
        //     !patch_match_options.init_depth_delaunay*/) {
        if (patch_match_options.geom_consistency) {
            if (!depth_maps_.at(image_idx).IsValid()) {
                std::string image_name = perspective_image_names_[image_idx];
                const std::string file_name = StringPrintf("%s.%s.%s", 
                    image_name.c_str(), input_type.c_str(), DEPTH_EXT);
                const std::string src_depth_map_path = JoinPaths(
                    component_path_, STEREO_DIR, DEPTHS_DIR, file_name);
                const std::string src_normal_map_path = JoinPaths(
                    component_path_, STEREO_DIR, NORMALS_DIR, file_name);
                if (!ExistsFile(src_depth_map_path) || 
                    !ExistsFile(src_normal_map_path)) {
                    invalid_image_idxs.insert(image_idx);
                    continue;
                }
                depth_maps_.at(image_idx).Read(src_depth_map_path);
                normal_maps_.at(image_idx).Read(src_normal_map_path);
                if (!depth_maps_.at(image_idx).IsValid() ||
                    !normal_maps_.at(image_idx).IsValid()) {
                    invalid_image_idxs.insert(image_idx);
                    continue;
                }
            }
        }

        if (ref_flag || options.pyramid_max_level == 0){
            continue;
        }
           
        image->Rescale(factor_level);
      }

      if (invalid_image_idxs.find(problem.ref_image_idx) != 
          invalid_image_idxs.end()) {
          std::cout << StringPrintf(
                  "WARNING: Ignoring image %s, because refer image is do not "
                  "be used.", image_name.c_str())
                      << std::endl;
          return;
      }

      // Remove invalid image idx.
      int i, j;
      for (i = 0, j = 0; i < problem.src_image_idxs.size(); ++i) {
          const int image_idx = problem.src_image_idxs.at(i);
          if (invalid_image_idxs.find(image_idx) == invalid_image_idxs.end()) {
              problem.src_image_idxs[j] = problem.src_image_idxs[i];
              problem.src_image_scales[j] = problem.src_image_scales[i];
              j = j + 1;
          }
      }
      problem.src_image_idxs.resize(j);
      problem.src_image_scales.resize(j);
      if (problem.src_image_idxs.size() == 0) {
            std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because source images do not "
                    "exist.", image_name.c_str())
                      << std::endl;
            return;
      }
    }

    problem.Print();
    // patch_match_options.Print();

    DepthMap ref_depth_map;
    NormalMap ref_normal_map;
    Mat<unsigned short> ref_curvature_map;
#ifdef SAVE_CONF_MAP
    Mat<float> ref_conf_map;
#endif

    PatchMatchACMMCuda patch_match_cuda(patch_match_options, problem);
    patch_match_cuda.Run();
    
    ref_depth_map = patch_match_cuda.GetDepthMap();
    ref_normal_map = patch_match_cuda.GetNormalMap();
    if (patch_match_options.est_curvature && 
        patch_match_options.geom_consistency) {
        ref_curvature_map = patch_match_cuda.GetCurvatureMap();
    }
#ifdef SAVE_CONF_MAP
    if (patch_match_options.geom_consistency) {
        ref_conf_map = patch_match_cuda.GetConfMap();
    }
#endif

    if (!ref_depth_map.IsValid() || !ref_normal_map.IsValid()) {
        return;
    }

    {
        std::chrono::high_resolution_clock::time_point start_time = 
            std::chrono::high_resolution_clock::now();
        std::string parent_path = GetParentDir(depth_map_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        ref_depth_map.Write(depth_map_path);
        if (options.verbose) {
            ref_depth_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", 
                                                        depth_map_path.c_str(), 
                                                        level));
        }

        parent_path = GetParentDir(normal_map_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        ref_normal_map.Write(normal_map_path);
        if (options.verbose) {
            ref_normal_map.ToBitmap().Write(StringPrintf("%s-%d.jpg", 
                                                         normal_map_path.c_str(), 
                                                         level));
        }

        if (patch_match_options.write_consistency_graph) {
            // parent_path = GetParentDir(consistency_graph_path);
            // if (!ExistsPath(parent_path)) {
            //     boost::filesystem::create_directories(parent_path);
            // }
            // patch_match.GetConsistencyGraph().Write(consistency_graph_path);
        }

        if (options.est_curvature && options.geom_consistency) {
            parent_path = GetParentDir(curvature_map_path);
            if (!ExistsPath(parent_path)) {
                boost::filesystem::create_directories(parent_path);
            }
            ref_curvature_map.Write(curvature_map_path);
        }

#ifdef SAVE_CONF_MAP
        if (patch_match_options.geom_consistency) {
            parent_path = GetParentDir(conf_map_path);
            if (!ExistsPath(parent_path)) {
                boost::filesystem::create_directories(parent_path);
            }
            ref_conf_map.Write(conf_map_path);
            if (options.verbose) {
                DepthMap(ref_conf_map, -1, -1).ToBitmap().Write(
                    StringPrintf("%s-%d.jpg", conf_map_path.c_str(), level));
            }
        }
#endif

        std::chrono::high_resolution_clock::time_point end_time = 
            std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                        end_time - start_time).count() / 1e6;
        std::cout << StringPrintf("Serialization: %lf [seconds]", duration) 
                  << std::endl;

#ifdef CACHED_DEPTH_MAP
        // TODO:
        depth_maps_.at(problem.ref_image_idx) = std::move(ref_depth_map);
        normal_maps_.at(problem.ref_image_idx) = std::move(ref_normal_map);
#ifdef SAVE_CONF_MAP
        if (patch_match_options.geom_consistency) {
            conf_maps_.at(problem.ref_image_idx) = std::move(ref_conf_map);
        }
#endif
#endif
    }
}

void PanoramaPatchMatchController::ProcessFilterProblem(
    const PatchMatchOptions& options,
    const size_t problem_idx,
    const int level,
    const bool ref_flag,
    std::string input_type) {
    if (IsStopped()) {
        return;
    }

    auto& problem = problems_.at(problem_idx);
    const int gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());
    CHECK_GE(gpu_index, -1);
    
    const std::string image_name = 
        perspective_image_names_[problem.ref_image_idx];
    const std::string file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), options.output_type.c_str(), DEPTH_EXT);
    const std::string depth_map_path =
        JoinPaths(component_path_, STEREO_DIR, DEPTHS_DIR, file_name);
    // const std::string normal_map_path =
    //     JoinPaths(component_path_, STEREO_DIR, NORMALS_DIR, file_name);
    // const std::string conf_map_path = 
    //     JoinPaths(component_path_, STEREO_DIR, CONFS_DIR, file_name);
    // const std::string consistency_graph_path = JoinPaths(
    //     component_path_, STEREO_DIR, CONSISTENCY_DIR, file_name);
    // const std::string workspace_image_path = 
    //     JoinPaths(component_path_, IMAGES_DIR);

    // if (ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
    //     (!options.write_consistency_graph ||
    //     ExistsFile(consistency_graph_path))) {
    //     return;
    // }

    PrintHeading1(StringPrintf("Reconstructing view %d / %d  (level = %d)", 
                               problem_idx + 1, problems_.size(), level));

    auto patch_match_options = options;
    if (patch_match_options.depth_min < 0 || 
        patch_match_options.depth_max < 0) {
        patch_match_options.depth_min =
            depth_ranges_.at(problem.ref_image_idx).first;
        patch_match_options.depth_max =
            depth_ranges_.at(problem.ref_image_idx).second;
        std::cout << patch_match_options.depth_min << ", " 
                  << patch_match_options.depth_max << std::endl;
        CHECK(patch_match_options.depth_min > 0 &&
              patch_match_options.depth_max > 0)
            << " - You must manually set the minimum and maximum depth, "
               "since no sparse model is provided in the workspace.";
    }

    patch_match_options.gpu_index = std::to_string(gpu_index);
    patch_match_options.thread_index = thread_pool_->GetThreadIndex()
                                                    % THREADS_PER_GPU;

    std::vector<mvs::Image> perspective_images_process;
    if (options.pyramid_max_level == 0) {
        problem.images = &perspective_images_;
    } else {
        perspective_images_process = perspective_images_;
        problem.images = &perspective_images_process;
    }

    float factor_level = 1 / std::pow(2,level);
    float max_size = std::max(
                      perspective_images_.at(problem.ref_image_idx).GetHeight(),
                      perspective_images_.at(problem.ref_image_idx).GetWidth());
    float min_image_size = std::min(480.f, max_size);
    factor_level = std::max(factor_level, min_image_size/max_size);

#ifndef CACHED_DEPTH_MAP
    std::vector<DepthMap> depth_maps_;
    // std::vector<NormalMap> normal_maps_;
    // std::vector<Mat<float> > conf_maps_;
    depth_maps_.resize(perspective_images_.size());
    // normal_maps_.resize(perspective_images_.size());
    // conf_maps_.resize(perspective_images_.size());
#endif

    problem.depth_maps = &depth_maps_;
    // problem.normal_maps = &normal_maps_;
    // problem.conf_maps = &conf_maps_;

    {
      // Collect all used images in current problem.
      std::unordered_set<int> used_image_idxs(problem.src_image_idxs.begin(),
                                              problem.src_image_idxs.end());
      used_image_idxs.insert(problem.ref_image_idx);

      patch_match_options.filter_min_num_consistent =
          std::min(static_cast<int>(used_image_idxs.size()) - 1,
                  patch_match_options.filter_min_num_consistent);

    
      int used_image_num = 0;
      for (const auto image_idx : used_image_idxs) {
        if (used_images_.find(image_idx) == used_images_.end()) {
            continue;
        }              
        used_image_num ++;
      }
      if (used_image_num < 2){
        return;
      }
#ifdef CACHED_DEPTH_MAP
      // Only access workspace from one thread at a time and only spawn resample
      // threads from one master thread at a time.
      std::unique_lock<std::mutex> lock(workspace_mutex_);
#endif

      std::cout << "Reading inputs..." << std::endl;
      for (const auto image_idx : used_image_idxs) {
        if (used_images_.find(image_idx) == used_images_.end()) {
            continue;
        }
        mvs::Image* image;
        if (options.pyramid_max_level == 0) {
            image = &perspective_images_.at(image_idx);
        } else {
            image = &perspective_images_process.at(image_idx);
        }
        if (!depth_maps_.at(image_idx).IsValid()) {
            std::string image_name = perspective_image_names_[image_idx];
            const std::string file_name =
                StringPrintf("%s.%s.%s", image_name.c_str(), 
                                input_type.c_str(), DEPTH_EXT);
            const std::string src_depth_map_path =
                JoinPaths(component_path_, STEREO_DIR, DEPTHS_DIR,
                            file_name);
            depth_maps_.at(image_idx).Read(src_depth_map_path);

            // const std::string src_normal_map_path =
            //     JoinPaths(component_path_, STEREO_DIR, NORMALS_DIR, 
            //                 file_name);
            // normal_maps_.at(image_idx).Read(src_normal_map_path);

            // const std::string src_conf_map_path =
            //     JoinPaths(component_path_, STEREO_DIR, CONFS_DIR, 
            //                 file_name);
            // conf_maps_.at(image_idx).Read(src_conf_map_path);
        }
        

        if (ref_flag || options.pyramid_max_level == 0){
            continue;
        }

        image->Rescale(factor_level);
      }
    }

    problem.Print();
    // patch_match_options.Print();

    DepthMap ref_depth_map;
    // NormalMap ref_normal_map;
    // Mat<float> ref_conf_map;

    PatchMatchACMMCuda patch_match_cuda(patch_match_options, problem);
    patch_match_cuda.RunFilter();
    
    ref_depth_map = patch_match_cuda.GetDepthMap();
    // ref_normal_map = patch_match_cuda.GetNormalMap();
    // ref_conf_map = patch_match_cuda.GetConfMap();
    
    if (!ref_depth_map.IsValid()/* || !ref_normal_map.IsValid()*/) {
        return;
    }

    std::string parent_path = GetParentDir(depth_map_path);
    if (!ExistsPath(parent_path)) {
        boost::filesystem::create_directories(parent_path);
    }
    ref_depth_map.Write(depth_map_path);
    if (options.verbose) {
        ref_depth_map.ToBitmap().Write(depth_map_path + "-filter.jpg");
    }

    // parent_path = GetParentDir(normal_map_path);
    // if (!ExistsPath(parent_path)) {
    //     boost::filesystem::create_directories(parent_path);
    // }
    // ref_normal_map.Write(normal_map_path);
    // if (options.verbose) {
    //     ref_normal_map.ToBitmap().Write(normal_map_path + "-filter.jpg");
    // }

    // if (patch_match_options.write_consistency_graph) {
    //     // parent_path = GetParentDir(consistency_graph_path);
    //     // if (!ExistsPath(parent_path)) {
    //     //     boost::filesystem::create_directories(parent_path);
    //     // }
    //     // patch_match.GetConsistencyGraph().Write(consistency_graph_path);
    // }

    //     parent_path = GetParentDir(conf_map_path);
    //     if (!ExistsPath(parent_path)) {
    //         boost::filesystem::create_directories(parent_path);
    //     }
    //     ref_conf_map.Write(conf_map_path);
    //     if (options.verbose) {
    //         DepthMap(ref_conf_map, -1, -1).ToBitmap().Write(conf_map_path + "-filter.jpg");
    //     }
}

std::pair<float, float> PanoramaPatchMatchController::InitDepthMap(
    const PatchMatchOptions& options,
    const size_t problem_idx,
    const Image& image,
    const std::string& workspace_path) {

    const auto& model = workspace_->GetModel();
    auto& problem = problems_.at(problem_idx);

    const std::string output_type =
        options.geom_consistency ? GEOMETRIC_TYPE : PHOTOMETRIC_TYPE;
    const std::string image_name = 
        perspective_image_names_.at(problem.ref_image_idx);
    const std::string file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), output_type.c_str(), DEPTH_EXT);
    const std::string depth_map_path =
        JoinPaths(workspace_path, DEPTHS_DIR, file_name);
    const std::string normal_map_path =
        JoinPaths(workspace_path, NORMALS_DIR, file_name);

    const int ori_ref_image_idx = problem_idx / num_perspective_per_image;
    const int ori_ref_image_idx_off = 
        problem_idx - ori_ref_image_idx * num_perspective_per_image;

    const size_t width = image.GetWidth();
    const size_t height = image.GetHeight();

    // const mvs::Image& image = perspective_images_.at(problem.ref_image_idx);
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K(image.GetK());
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R(image.GetR());
    Eigen::Vector3f T(image.GetT());

    std::pair<float, float> depth_range;
    depth_range.first = std::numeric_limits<float>::max();
    depth_range.second = std::numeric_limits<float>::epsilon();

    std::vector<Eigen::Vector3f> uvd;
    for (const auto& point : model.points) {
        for (int elem_id = 0; elem_id < point.track.size(); ++elem_id) {
            if (point.track.at(elem_id) == ori_ref_image_idx) {
                const Eigen::Vector2d& point2d = point.points2d.at(elem_id);
                const int x = point2d.x();
                const int y = point2d.y();
                if (x < 0 || x >= width || y < 0 || y >= height) {
                    continue;
                }
                int f_idx = rmap_idx_.at(y * width + x);
                if (ori_ref_image_idx_off == f_idx) {
                    const Eigen::Vector3f X(point.x, point.y, point.z);
                    const Eigen::Vector3f Xc = K * (R * X + T);
                    if (Xc[2] <= 0) {
                        continue;
                    }
                    int u = Xc[0] / Xc[2] + 0.5f;
                    int v = Xc[1] / Xc[2] + 0.5f;
                    uvd.emplace_back(u, v, Xc[2]);

                    depth_range.first = std::min(depth_range.first, Xc[2]);
                    depth_range.second = std::max(depth_range.second, Xc[2]); 
                }
            }
        }
    }

    if (options.init_from_delaunay) {
        InitDepthMapFromDelaunay(problem, uvd, image);
    } else {
        InitDepthMapFromSparsePoint(problem, uvd, image);
    }

    if (depth_range.first == std::numeric_limits<float>::max()) {
        depth_range.first = 0.0f;
        depth_range.second = 100.0f;
    }
    depth_range.first *= 0.5f;
    depth_range.second *= 2.1f;

    depth_range.first = std::max(depth_range.first, 0.000001f);
    return depth_range;
}

} // namespace mvs
} // namespace 