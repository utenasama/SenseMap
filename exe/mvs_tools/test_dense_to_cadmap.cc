//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <exception>
#include <Eigen/Dense>
#include <boost/filesystem/path.hpp>
#include <boost/tuple/tuple.hpp>
#include <opencv2/opencv.hpp>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/license/Point_set_processing_3.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>


#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "util/bitmap.h"
#include "util/semantic_table.h"
#include "util/timer.h"
#include "util/threading.h"
#include "base/common.h"
#include "estimators/plane_estimator.h"
#include "optim/ransac/loransac.h"
#include "mvs/workspace.h"
#include "mvs/delaunay/delaunay_triangulation.h"

#include "base/version.h"
#include "../Configurator_yaml.h"

typedef CGAL::Simple_cartesian<double>                       Kernel;
typedef CGAL::Polyhedron_3<Kernel>                           Polyhedron;
typedef Polyhedron::HalfedgeDS                               HalfedgeDS;
typedef Kernel::Point_3                                      Point;
typedef Kernel::Vector_3                                     Vector;
typedef Kernel::Segment_3                                    Segment;
typedef Kernel::Ray_3                                        Ray;
typedef Kernel::Triangle_3                                   Triangle;
typedef std::list<Triangle>::iterator                        Iterator;
typedef CGAL::AABB_triangle_primitive<Kernel, Iterator>      Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive>                 Traits;
typedef CGAL::AABB_tree<Traits>                              Tree;
typedef Tree::Primitive_id                                   Primitive_id;
typedef Kernel::FT FT;
typedef Kernel::Point_3 point_3_t;
typedef CGAL::Search_traits_3<Kernel> tree_traits_3_t;
typedef CGAL::Orthogonal_k_neighbor_search<tree_traits_3_t> neighbor_search_3_t;
typedef neighbor_search_3_t::iterator search_3_iterator_t;
typedef neighbor_search_3_t::Tree tree_3_t;
typedef boost::tuple<int, point_3_t> indexed_point_3_tuple_t;

std::string configuration_file_path;

using namespace sensemap;

struct CadMapOptions {
	std::string workspace_path = "";
	std::string image_type = "perspective";
    std::string cadmap_type = "cadmap"; // "cadmap", "tdom"
    int max_grid_size = 1000;
    bool output_by_gravity = 0;
    bool output_ground = 0;
    bool output_height_range = 0;
    bool output_point_normal = 0;
    bool output_point_density = 0;
    bool output_meshmap = 0;
    bool output_mesh_normal = 0;
    bool output_camera_track = 0;
    bool output_roofmap = 0;
    bool output_sideview = 0;
    bool verbose = 0;

    int camera_track_radius = 10;
    int point_size = 1;
    size_t target_num_point = -1;

    int min_consistent_facet = 1000;
    float dist_point_to_line = 0.0;
    float angle_diff_thres = 20.0;
    float dist_ratio_point_to_plane1 = 10.0;
    float dist_ratio_point_to_plane2 = 20.0;
    float ratio_singlevalue_xz = 1500.0;
    float ratio_singlevalue_yz = 400.0;
    float dist_insert = 5.0;
    float diff_depth = 0.01;
    float filter_low_ratio = 0.01;
    float filter_high_ratio = 0.9;
    float inlier_ratio_plane_points = 0.999;
    int nb_neighbors = 6;    
    float max_spacing_factor = 6.0;

    void Print() {
        PrintHeading2("CadMapOptions");

        PrintOption(workspace_path);
        PrintOption(image_type);
        PrintOption(cadmap_type);
        PrintOption(max_grid_size);
        PrintOption(output_by_gravity);
        PrintOption(output_sideview);
        PrintOption(output_ground);
        PrintOption(output_height_range);
        PrintOption(output_point_normal);
        PrintOption(output_point_density);
        PrintOption(output_meshmap);
        PrintOption(output_mesh_normal);
        PrintOption(output_camera_track);
        PrintOption(output_roofmap);
        PrintOption(verbose);
        PrintOption(camera_track_radius);
        PrintOption(point_size);
        PrintOption(target_num_point);
        PrintOption(min_consistent_facet);
        PrintOption(dist_point_to_line);
        PrintOption(angle_diff_thres);
        PrintOption(dist_ratio_point_to_plane1);
        PrintOption(dist_ratio_point_to_plane2);
        PrintOption(ratio_singlevalue_xz);
        PrintOption(ratio_singlevalue_yz);
        PrintOption(dist_insert);
        PrintOption(diff_depth);
        PrintOption(filter_low_ratio);
        PrintOption(filter_high_ratio);
        PrintOption(inlier_ratio_plane_points);
        PrintOption(nb_neighbors);
        PrintOption(max_spacing_factor);
    }
};

struct PlanePoint {
    double dist = 0.0;
    Eigen::Vector3d X;
    Eigen::Vector3d normal;
    Eigen::Vector3ub rgb;
    std::vector<uint32_t> viss;
    uint8_t s_id;
};

int X_AXIS = 1;
int Y_AXIS = 2;
int Z_AXIS = 4;

// in case the input points are without normal
bool CheckPointsNormal(const std::vector<std::vector<PlanePoint> >& nonplane_points)
{
    int count = 0;
    int valid_count = 0;
    for(const auto& points : nonplane_points)
    {
        for(const auto& point : points)
        {
            auto normal = point.normal;
            //uninit normal norm should around 0, 
            //valid norm should be around 1, check no-zero value would be enough 
            if(normal.norm() > 0.1)
            {   
                valid_count++;
            }
        }
        count += points.size();
    }
    std::cout << "points with valid normal count : " << valid_count << std::endl;
    std::cout << "all points count : " << count << std::endl;
    
    float valid_normal_ratio = float(valid_count) / float(count);
    std::cout << "points with valid normal ratio : " << valid_normal_ratio << std::endl;
    if(valid_normal_ratio < 0.5) 
    {
        std::cout << "Warning: No enough input points with normals!" << std::endl;
        return false;
    }
    return true;
}

bool DetectPlaneWithRansac(const std::vector<Eigen::Vector3d>& wall_points, 
                            const std::vector<Eigen::Vector3d>& wall_normals,
                            const Eigen::Vector3d& plane_orientation,
                            std::vector<Eigen::Vector3d>& plane_normals)
{
    typedef std::pair<Kernel::Point_3, Kernel::Vector_3>         Point_with_normal;
    typedef std::vector<Point_with_normal>                       Pwn_vector;
    typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
    typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;
    typedef CGAL::Shape_detection::Efficient_RANSAC_traits
    <Kernel, Pwn_vector, Point_map, Normal_map>             Traits;
    typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> Efficient_ransac;
    typedef CGAL::Shape_detection::Plane<Traits>            Plane;

    // Points with normals.
    Pwn_vector points;
    for(int i = 0; i < wall_points.size(); ++i)
    {
        Point p(wall_points[i][0], wall_points[i][1], wall_points[i][2]);
        Vector n(wall_normals[i][0], wall_normals[i][1], wall_normals[i][2]);
        Point_with_normal pn(p, n);
        points.emplace_back(pn);
    }

    // Instantiate shape detection engine.
    Efficient_ransac ransac;
    // Provide input data.
    ransac.set_input(points);
    // Register planar shapes via template method.
    ransac.add_shape_factory<Plane>();
    // Detect registered shapes with default parameters.

    // Set parameters for shape detection.
    Efficient_ransac::Parameters parameters;
    // Set probability to miss the largest primitive at each iteration.
    parameters.probability = 0.05;
    // Detect shapes with at least 200 points.
    parameters.min_points = 600;
    // // Set maximum Euclidean distance between a point and a shape.
    // parameters.epsilon = 0.05;
    // // Set maximum Euclidean distance between points to be clustered.
    // parameters.cluster_epsilon = 1.2;
    // Set maximum normal deviation.
    // 0.9 < dot(surface_normal, point_normal);
    parameters.normal_threshold = 0.8;
    // Detect shapes.
    ransac.detect(parameters);
    // Print number of detected shapes and unassigned points.
    std::cout << ransac.shapes().end() - ransac.shapes().begin()
    << " detected shapes, "
    << ransac.number_of_unassigned_points()
    << " unassigned points." << std::endl;
    float unassigned_points_ratio = ransac.number_of_unassigned_points() / float(wall_points.size());
    float wall_points_ratio_thred = 0.5f;
    std::cout << "wall points num : " << wall_points.size() << std::endl;
    std::cout << "unassigned points ratio : " << unassigned_points_ratio << std::endl;
    if(unassigned_points_ratio > wall_points_ratio_thred)
    {
        std::cout << " The scene doesn't have enough valid walls!"<< std::endl;
        return false;
    }

    // Efficient_ransac::shapes() provides
    // an iterator range to the detected shapes.
    Efficient_ransac::Shape_range shapes = ransac.shapes();
    Efficient_ransac::Shape_range::iterator it = shapes.begin();
    auto& point_map = ransac.point_map();
    
    std::vector<Eigen::Vector3d> plane_grp_normals;
    std::vector<int> plane_grp_pt_nums;
    std::vector<std::vector<Plane*>> plane_groups; 
    // set 10 degree angle thred for plane grouping
    float angle_thred = 35 / 180.0 * M_PI;
    float cos_value_thred = std::cos(angle_thred);
    float ground_cos_value_thred = std::cos(70 / 180.0 * M_PI);
    std::cout << "normal group angle cos value thred : " << cos_value_thred << std::endl; 
    while (it != shapes.end()) {
        // Get specific parameters depending on the detected shape.
        if (Plane* plane = dynamic_cast<Plane*>(it->get())) {
            auto& p_ids = plane->indices_of_assigned_points();
            // ransac.point_map()
            Kernel::Vector_3 normal = plane->plane_normal();
            Eigen::Vector3d n(normal[0], normal[1], normal[2]);
            bool find_group = false;
           
            for(int i = 0; i < plane_grp_normals.size(); ++i)
            {
                int plane_size = plane_groups[i].size();
                auto g_normal = plane_grp_normals[i] / float(plane_grp_pt_nums[i]);
                auto group_normal = g_normal.normalized(); 
                if(group_normal.dot(n) > cos_value_thred)
                {
                    plane_groups[i].emplace_back(plane);
                    plane_grp_normals[i] += n * p_ids.size();
                    plane_grp_pt_nums[i] += p_ids.size();
                    find_group = true;
                }
            }
            if(!find_group)
            {
                plane_grp_normals.emplace_back(n * p_ids.size());
                plane_grp_pt_nums.push_back(p_ids.size());
                std::vector<Plane*> plane_group;
                plane_group.emplace_back(plane);
                plane_groups.emplace_back(plane_group);
            }
        } 
        // Proceed with the next detected shape.
        it++;
    }
    if(plane_groups.empty()) return false;

    std::cout << "plane groups: " << plane_groups.size() << std::endl;

    int max_group_point_size = 0, second_group_point_size = 0;
    std::vector<Plane*> max_group, second_group;
    Eigen::Vector3d max_normal(0.0, 0.0, 0.0), second_normal(0.0, 0.0, 0.0);
    for(int i = 0; i < plane_groups.size(); ++i)
    {
        auto& group =  plane_groups[i];
        int size_sum = 0;
        Eigen::Vector3d group_normal(0.0, 0.0, 0.0);
        for(auto& plane : group)
        {
            int p_size = plane->indices_of_assigned_points().size();
            auto plane_normal = plane->plane_normal();
            Eigen::Vector3d new_n(plane_normal[0], plane_normal[1], plane_normal[2]);
            group_normal += new_n * p_size;
            size_sum += p_size;
        }
        group_normal = group_normal / float(size_sum);
        group_normal.normalize();
        if (std::abs(max_normal.dot(group_normal)) < cos_value_thred &&
            std::abs(plane_orientation.dot(group_normal)) < ground_cos_value_thred) {
            if(size_sum > max_group_point_size) {
                second_group_point_size = max_group_point_size;
                second_group = max_group;
                second_normal = max_normal;
                max_group_point_size = size_sum;
                max_group = group;
                max_normal = group_normal;
            } else if (size_sum > second_group_point_size) {
                second_group_point_size = size_sum;
                second_group = group;
                second_normal = group_normal;
            }
        }
    }

    plane_normals.clear();
    if (max_group_point_size > 0) {
        max_normal.normalize();
        std::cout << "max_normal: " << max_normal.transpose() << std::endl;
        plane_normals.emplace_back(max_normal);
    }
    if (second_group_point_size > 0) {
        second_normal.normalize();
        std::cout << "second_normal: " << second_normal.transpose() << std::endl;
        plane_normals.emplace_back(second_normal);
    }
    return !plane_normals.empty();
}

bool ParseAlignFile(const std::string &align_file_path, 
                    std::vector<Eigen::Vector4d> &layered_planes, 
                    int &pivot_idx, Eigen::Matrix3x4d &T, 
                    Eigen::Matrix3x4d &M, Eigen::Matrix3x4d& SM){

    auto ParseTrivialCharacters = [&](std::fstream& fin, std::string& line) {
        while(std::getline(fin, line)) {
            if (!line.empty() && line[0] != '#') {
                break;
            }
        }
    };
    
    try{
        std::fstream fin(align_file_path);
        if (!fin){
            return false;
        }

        // number of floor.
        std::string line, item;
        ParseTrivialCharacters(fin, line);

        std::stringstream line_stream(line);
        std::getline(line_stream, item, ' ');
        int num_floor = std::stoi(item.c_str());
        std::cout << "num_floor: " << num_floor << std::endl;

        // plane equation.
        ParseTrivialCharacters(fin, line);
        
        std::cout << "plane equation: ";
        Eigen::Vector4d layered_plane;
        for(int i = 0; i < num_floor; i++){
            line_stream.clear();
            line_stream = std::stringstream(line);
            std::getline(line_stream, item, ' ');
            layered_plane[0] = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            layered_plane[1] = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            layered_plane[2] = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            layered_plane[3] = std::atof(item.c_str());
            std::getline(fin, line);
            layered_planes.push_back(layered_plane);
            std::cout << layered_plane.transpose() << std::endl;
        }

        // pivot plane index.
        ParseTrivialCharacters(fin, line);

        line_stream.clear();
        line_stream = std::stringstream(line);
        std::getline(line_stream, item, ' ');
        int pivot_idx = std::atoi(item.c_str());
        std::cout << "pivot_idx: " << pivot_idx << std::endl;

        // plane transformation matrix.
        ParseTrivialCharacters(fin, line);
        for (int i = 0; i < 3; ++i) {
            line_stream.clear();
            line_stream = std::stringstream(line);
            std::getline(line_stream, item, ' ');
            T(i, 0) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            T(i, 1) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            T(i, 2) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            T(i, 3) = std::atof(item.c_str());
            std::getline(fin, line);
        }
        std::cout << "Transformation Matrix: " << std::endl << T << std::endl;

        // projection matrix.
        ParseTrivialCharacters(fin, line);
        for (int i = 0; i < 3; ++i) {
            line_stream.clear();
            line_stream = std::stringstream(line);
            std::getline(line_stream, item, ' ');
            M(i, 0) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            M(i, 1) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            M(i, 2) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            M(i, 3) = std::atof(item.c_str());
            std::getline(fin, line);
        }
        std::cout << "Projection Matrix: " << std::endl << M << std::endl;

        // sideview projection matrix.
        ParseTrivialCharacters(fin, line);
        for (int i = 0; i < 3; ++i) {
            line_stream.clear();
            line_stream = std::stringstream(line);
            std::getline(line_stream, item, ' ');
            SM(i, 0) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            SM(i, 1) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            SM(i, 2) = std::atof(item.c_str());
            std::getline(line_stream, item, ' ');
            SM(i, 3) = std::atof(item.c_str());
            std::getline(fin, line);
        }
        std::cout << "Projection Matrix(sideview): " << std::endl << SM << std::endl;
    } catch(std::exception &e){
        std::cout << e.what() << std::endl;
        return false;
    }

    return true;
}


void DownsamplePoints(std::vector<PlyPoint>& points, 
    std::vector<std::vector<uint32_t> >& vis_points,
    const size_t target_num_point) {
    
    int step = points.size() / target_num_point;
    step = std::max(1, step);

    std::cout << "point sample step: " << step << std::endl;

    size_t i, j;
    for (i = 0, j = 0; i < points.size(); i += step) {
        points.at(j) = points.at(i);
        vis_points.at(j) = vis_points.at(i);
        j = j + 1;
    }
    points.resize(j);
    vis_points.resize(j);
    std::cout << "Sampled points: " << j << std::endl;
}

void BubbleSort(std::vector<Eigen::Vector4d>& planes,
                std::vector<std::vector<Eigen::Vector3d> >& points, 
                std::vector<double>& inlier_errors,
                bool ascend = true) {
    int i, j;
    for (i = 0; i < planes.size() - 1; ++i) {
        for (j = 0; j < planes.size() - 1 - i; ++j) {
            auto plane1 = planes.at(j);
            auto plane2 = planes.at(j + 1);
            if (ascend) {
                if (plane1.w() > plane2.w()) {
                    std::swap(planes.at(j), planes.at(j + 1));
                    std::swap(points.at(j), points.at(j + 1));
                    std::swap(inlier_errors.at(j), inlier_errors.at(j + 1));
                }
            } else {
                if (plane1.w() < plane2.w()) {
                    std::swap(planes.at(j), planes.at(j + 1));
                    std::swap(points.at(j), points.at(j + 1));
                    std::swap(inlier_errors.at(j), inlier_errors.at(j + 1));
                }
            }
        }
    }
}

double ComputeTriangleMeanArea(const std::vector<Eigen::Vector3d>& mesh_vertices,
                               const std::vector<Eigen::Vector3i>& mesh_facets, 
                               const std::vector<int> facet_list = std::vector<int>()) {
    std::vector<int> facet_list_tmp = facet_list;
    if (facet_list_tmp.empty()) {
        facet_list_tmp.resize(mesh_facets.size());
        std::iota(facet_list_tmp.begin(), facet_list_tmp.end(), 0);
    }
    if (facet_list_tmp.empty()) {
        return 0;
    }

    double area = 0;
    for (auto i : facet_list_tmp) {
        auto facet = mesh_facets.at(i);
        auto v0 = mesh_vertices.at(facet[0]);
        auto v1 = mesh_vertices.at(facet[1]);
        auto v2 = mesh_vertices.at(facet[2]);
        double a = (v0 - v1).norm();
        double b = (v0 - v2).norm();
        double c = (v1 - v2).norm();
        double s = (a + b + c) * 0.5;
        area += std::sqrt(s * (s - a) * (s - b) * (s - c));
    }
    area /= facet_list_tmp.size();
    return area;
}

void EstimateConsistentMultiPlanes(const CadMapOptions &cad_options,
                                   const std::vector<Eigen::Vector3d>& points,
                                   const std::vector<Eigen::Vector3i>& facets,
                                   const std::vector<Eigen::Vector3d>& wall_plane_normals,
                                   std::vector<Eigen::Vector4d>& planes,
                                   std::vector<double>& inlier_errors) {

    const double radian_diff_thres = std::cos(cad_options.angle_diff_thres / 180 * M_PI);
    const double cos_thres = std::cos(25.0f / 180.0f * M_PI);
    size_t i, j;

    std::vector<std::vector<int> > adj_facets_per_vertex(points.size());
    std::vector<std::vector<int> > adj_facets_per_facet(facets.size());
    for (i = 0; i < facets.size(); ++i) {
        auto facet = facets.at(i);
        adj_facets_per_vertex.at(facet[0]).push_back(i);
        adj_facets_per_vertex.at(facet[1]).push_back(i);
        adj_facets_per_vertex.at(facet[2]).push_back(i);
    }
    for (i = 0; i < facets.size(); ++i) {
        auto facet = facets.at(i);
        std::unordered_set<int> adj_facets;
        for (j = 0; j < 3; ++j) {
            for (auto facet_id : adj_facets_per_vertex.at(facet[j])) {
                if (facet_id != i) {
                    adj_facets.insert(facet_id);
                }
            }
        }
        if (adj_facets.size() == 0) {
            continue;
        }
        std::copy(adj_facets.begin(), adj_facets.end(), std::back_inserter(adj_facets_per_facet.at(i)));
    }

    // Compute face normals.
    std::vector<Eigen::Vector3d> face_normals(facets.size());
    for (i = 0; i < facets.size(); ++i) {
        auto facet = facets.at(i);
        auto& vtx0 = points.at(facet[0]);
        auto& vtx1 = points.at(facet[1]);
        auto& vtx2 = points.at(facet[2]);
        face_normals.at(i) = (vtx1 - vtx0).cross(vtx2 - vtx0).normalized();
    }

    double m_area = ComputeTriangleMeanArea(points, facets);
    std::cout << "Mean triangle area: " << m_area << std::endl;

    struct PlaneInfo {
        Eigen::Vector4d plane;
        double equivalent;
        double inlier_error;
        int inlier_number;
    };
    std::vector<PlaneInfo> plane_infos;

    std::vector<std::vector<int> > consistent_facets;
    std::vector<char> assigned(facets.size(), 0);

    planes.clear();
    inlier_errors.clear();

    for (i = 0; i < facets.size(); ++i) {
        if (assigned.at(i)) {
            continue;
        }

        std::queue<int> Q;
        Q.push(i);
        assigned.at(i) = 1;

        auto m_normal = face_normals.at(i);
        auto f = facets.at(i);
        Eigen::Vector3d m_C = (points.at(f[0]) + points.at(f[1]) + points.at(f[2])) / 3;
        double m_dist = 0.0;

        std::vector<int> consistent_facets_;
        consistent_facets_.push_back(i);

        while(!Q.empty()) {
            auto facet_id = Q.front();
            Q.pop();

            for (auto adj_facet : adj_facets_per_facet.at(facet_id)) {
                if (assigned.at(adj_facet)) {
                    continue;
                }
                auto m_nNormal = (m_normal / consistent_facets_.size()).normalized();
                double angle = m_nNormal.dot(face_normals.at(adj_facet));
                if (angle < radian_diff_thres) {
                    continue;
                }
                auto f = facets.at(adj_facet);
                Eigen::Vector3d C = (points.at(f[0]) + points.at(f[1]) + points.at(f[2])) / 3;
                Eigen::Vector3d mm_C = m_C / consistent_facets_.size();
                double mm_dist = m_dist / consistent_facets_.size();
                double dist = std::fabs(m_nNormal.dot(C - mm_C));
                if (m_dist != 0 && 
                    dist > cad_options.dist_ratio_point_to_plane1 * mm_dist) {
                    continue;
                }
                m_dist += dist;

                Q.push(adj_facet);

                assigned.at(adj_facet) = 1;
                consistent_facets_.push_back(adj_facet);
                m_normal += face_normals.at(adj_facet);
                m_C += C;
            }
        }
        
        if (consistent_facets_.size() == 0) {
            continue;
        }

        m_dist /= consistent_facets_.size();
        m_dist = m_dist == 0 ? 1e-3 : m_dist;

        std::vector<Eigen::Vector3d> facet_centaries;
        facet_centaries.reserve(consistent_facets_.size());
        for (auto facet_id : consistent_facets_) {
            auto f = facets.at(facet_id);
            auto C = (points.at(f[0]) + points.at(f[1]) + points.at(f[2])) / 3;
            facet_centaries.push_back(C);
        }

        RANSACOptions options;
        options.max_error = m_dist * cad_options.dist_ratio_point_to_plane1;
        LORANSAC<PlaneEstimator, PlaneLocalEstimator> estimator(options);
        auto report = estimator.Estimate(facet_centaries);
        if (!report.success) {
            continue;
        }
        auto model = report.model;
  
        j = 0;
        double mean_error = 0.0;
        Eigen::Vector3d mean_C(0, 0, 0);
        for (int k = 0; k < consistent_facets_.size(); ++k) {
            mean_error += std::fabs(model.dot(facet_centaries.at(k).homogeneous()));
            if (report.inlier_mask.at(k)) {
                assigned.at(consistent_facets_[k]) = 1;
                consistent_facets_.at(j) = consistent_facets_.at(k);
                facet_centaries.at(j) = facet_centaries.at(k);
                mean_C += facet_centaries.at(j);
                j = j + 1;
            } else {
                assigned.at(consistent_facets_[k]) = 0;
            }
        }
        mean_error /= consistent_facets_.size();
        mean_C /= j;
        consistent_facets_.resize(j);
        facet_centaries.resize(j);

        double area = 0;
        for (auto i : consistent_facets_) {
            auto facet = facets.at(i);
            auto v0 = points.at(facet[0]);
            auto v1 = points.at(facet[1]);
            auto v2 = points.at(facet[2]);
            double a = (v0 - v1).norm();
            double b = (v0 - v2).norm();
            double c = (v1 - v2).norm();
            double s = (a + b + c) * 0.5;
            area += std::sqrt(s * (s - a) * (s - b) * (s - c));
        }

        double equivalent = area / m_area * consistent_facets_.size();
        if (equivalent < cad_options.min_consistent_facet) {
            continue;
        }

        consistent_facets.emplace_back(consistent_facets_);

        m_normal.normalize();
        if (model.head<3>().dot(m_normal) < 0) {
            model = -model;
        }

	    std::cout << StringPrintf("consistent facets: %d, equivalent: %f, normal: %f %f %f\n", 
                                  consistent_facets_.size(), equivalent, model[0], model[1], model[2]);

        if (wall_plane_normals.size() > 1) {
            Eigen::Vector3d prior_plane_normal = wall_plane_normals[0].cross(wall_plane_normals[1]).normalized();
            float rad = prior_plane_normal.dot(model.head<3>());
            if (std::fabs(rad) < cos_thres) {
                std::cout << "The estimated plane normal deviate from the prior by " << std::acos(rad) * 180.0f / M_PI << std::endl;
                continue;
            }
        }

        PlaneInfo plane_info;
        plane_info.plane = model;
        plane_info.equivalent = equivalent;
        plane_info.inlier_error = mean_error;
        plane_info.inlier_number = consistent_facets_.size();

        plane_infos.emplace_back(plane_info);
    }

    int num_plane = plane_infos.size();
    std::cout << "Detect " << num_plane << " planes" << std::endl;

#if 0
    std::vector<float> nx, ny, nz;
    for (i = 0, j = 0; i < plane_infos.size(); ++i) {
        auto plane = plane_infos.at(i).plane;
        nx.push_back(plane.x());
        ny.push_back(plane.y());
        nz.push_back(plane.z());
    }
    int nth = nx.size() / 2;
    std::nth_element(nx.begin(), nx.begin() + nth, nx.end());
    std::nth_element(ny.begin(), ny.begin() + nth, ny.end());
    std::nth_element(nz.begin(), nz.begin() + nth, nz.end());

    Eigen::Vector3d m_normal(nx.at(nth), ny.at(nth), nz.at(nth));
    m_normal.normalize();
#else
    Eigen::Vector3d m_normal = Eigen::Vector3d::Zero();
    double max_equivalent = 0;
    for (auto & plane_info : plane_infos) {
        if (max_equivalent < plane_info.equivalent) {
            max_equivalent = plane_info.equivalent;
            m_normal = plane_info.plane.head<3>();
        }
    }
    std::cout << "Main plane normal: " << m_normal.transpose() << std::endl;
#endif
    const float angle_thres = std::cos(M_PI * 10.0 / 180);
    for (i = 0, j = 0; i < plane_infos.size(); ++i) {
        auto plane = plane_infos.at(i).plane;
        float cos_angle = plane.head<3>().dot(m_normal);
        if (cos_angle > angle_thres) {
            plane_infos.at(j) = plane_infos.at(i);
            j = j + 1;
        } else {
            std::cout << "  => Remove plane#" << i << ": " << plane.transpose() 
                      << ", error: " << std::acos(cos_angle) * 180 / M_PI 
                      << std::endl;
        }
    }
    plane_infos.resize(j);

    std::cout << "Remove " << num_plane - plane_infos.size() << " non-horizontal planes" << std::endl;

    std::sort(plane_infos.begin(), plane_infos.end(), 
        [&](const PlaneInfo& plane1, const PlaneInfo& plane2) {
        return plane1.inlier_number > plane2.inlier_number;
    });

    for (auto & plane_info : plane_infos) {
        planes.emplace_back(plane_info.plane);
        inlier_errors.emplace_back(plane_info.inlier_error);
    }
}

bool FitGround(const CadMapOptions& cad_options,
               const std::vector<mvs::Image>& images,
               const std::vector<PlyPoint>& fused_points,
               const std::vector<std::vector<uint32_t> >& fused_visibility,
               const TriangleMesh &mesh,
               const std::vector<Eigen::Vector3d> wall_plane_normals,
               std::vector<Eigen::Vector4d>& out_planes,
               std::vector<std::vector<Eigen::Vector3d> >& inlier_points) {
    
    size_t i, j;

    out_planes.clear();
    // Collect plane candidates.
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3ub> colors;
    std::vector<std::vector<uint32_t> > visibilities;
    points.reserve(fused_points.size());
    visibilities.reserve(fused_points.size());
    for (i = 0; i < fused_points.size(); ++i) {
        auto point = fused_points.at(i);
        if (point.s_id == LABLE_GROUND) {
            points.emplace_back(point.x, point.y, point.z);
            colors.emplace_back(point.r, point.g, point.b);
            visibilities.emplace_back(fused_visibility.at(i));
        }
    }
    if (points.size() <= 3) {
        std::cout << "Warning! No labeled ground points" << std::endl;
        return false;
    }

    std::vector<Eigen::Vector4d> planes;
    std::vector<double> inlier_errors;
#if 0
    mvs::PointCloud pointcloud;
    size_t num_point = points.size();
    pointcloud.pointViews.Resize(num_point);
    pointcloud.pointWeights.Resize(num_point);
    pointcloud.pointTypes.Resize(num_point);
    for (i = 0; i < num_point; ++i) {
        if (i % 1000 == 0) {
            std::cout << "\r" << i << " / " << num_point;
        }
        pointcloud.points.Insert(points.at(i).cast<float>());
        pointcloud.colors.Insert(colors.at(i));
        pointcloud.pointTypes.Insert(0);
        for (const auto& view : visibilities.at(i)) {                 
            pointcloud.pointViews[i].Insert(view);
            pointcloud.pointWeights[i].Insert(1.0f);
        }
    }

    TriangleMesh plane_mesh;
    mvs::DelaunayMeshing(plane_mesh, pointcloud, images, true, 10.0, 0.025, 2.0f, -1.0f,
        false, 4, 1, 1, 4, 3, 0.1, 1000, 400);
    plane_mesh.Clean(1.f, 5.f, false, 0, 50, true);

    EstimateConsistentMultiPlanes(cad_options, plane_mesh.vertices_, plane_mesh.faces_, wall_plane_normals, 
                                  planes, inlier_errors);
#else
    std::vector<Eigen::Vector3i> plane_facets;
    for (size_t i = 0; i < mesh.faces_.size(); ++i) {
        auto facet = mesh.faces_.at(i);
        auto label0 = mesh.vertex_labels_.at(facet[0]);
        auto label1 = mesh.vertex_labels_.at(facet[1]);
        auto label2 = mesh.vertex_labels_.at(facet[2]);
        if (label0 == LABLE_GROUND && label1 == LABLE_GROUND && label2 == LABLE_GROUND) {
            plane_facets.emplace_back(mesh.faces_.at(i));
        }
    }

    EstimateConsistentMultiPlanes(cad_options, mesh.vertices_, plane_facets, wall_plane_normals, 
                                  planes, inlier_errors);
#endif

    bool detected_plane = (planes.size() != 0);
    // if (planes.size() == 0) {
    //     std::cerr << "ERROR! no plane detected!" << std::endl;
    //     return false;
    // }
    if (detected_plane) {
        std::vector<std::vector<Eigen::Vector3d> > layered_points;
        std::vector<std::unordered_set<uint32_t> > layered_visibilities;
        layered_points.resize(planes.size());
        layered_visibilities.resize(planes.size());
#pragma omp parallel for schedule(dynamic)
        for (size_t i1 = 0; i1 < points.size(); ++i1) {
            const auto & point = points.at(i1);
            const auto & viss = visibilities.at(i1);
            int best_plane_idx = -1;
            double min_dist_to_plane = FLT_MAX;
            for (size_t j1 = 0; j1 < planes.size(); ++j1) {
                const auto & plane = planes.at(j1);
                double dist = std::abs(plane.dot(point.homogeneous()));
                if (min_dist_to_plane > dist) {
                    min_dist_to_plane = dist;
                    best_plane_idx = j1;
                }
            }
            if (best_plane_idx != -1 && 
                min_dist_to_plane < cad_options.dist_ratio_point_to_plane2 * inlier_errors.at(best_plane_idx)) {
#pragma omp critical
                {
                    layered_points.at(best_plane_idx).emplace_back(point);
                    layered_visibilities.at(best_plane_idx).insert(viss.begin(), viss.end());
                }
            }
        }

        std::cout << "Detect multi planes on different layers" << std::endl;
        size_t max_layered_point = 0;
        int layer_id = 0;
        for (i = 0; i < layered_points.size(); ++i) {
            std::cout << StringPrintf("  => Layer#%02d: %d points\n", i, layered_points.at(i).size());
            if (max_layered_point < layered_points.at(i).size()) {
                layer_id = i;
                max_layered_point = layered_points.at(i).size();
            }
        }

        if (max_layered_point <= 3) {
            std::cout << "Warning! No enough ground points!" << std::endl;
            return false;
        }

        std::cout << "Estimate main plane parameter" << std::endl;
        for (i = 0; i < layered_points.size(); ++i) {
            points = layered_points.at(i);
            std::cout << StringPrintf("Collect %d ground points on Layer#%d\n",
                points.size(), i);
            if (points.size() <= 3) {
                continue;
            }

            //Estimate planar equation.
            // PlaneLocalEstimator local_estimator;
            // auto H = local_estimator.Estimate(points).at(0);
            RANSACOptions ransac_option;
            ransac_option.max_error = inlier_errors.at(i) * 2;
            LORANSAC<PlaneEstimator, PlaneLocalEstimator> estimator(ransac_option);
            auto report = estimator.Estimate(points);
            auto H = report.model;

            // Correct Plane Normal.
            float sum_d = 0.0f;
            for (auto vis : layered_visibilities.at(i)) {
                const mvs::Image image = images.at(vis);
                Eigen::Vector3d C = Eigen::Vector3f(image.GetC()).cast<double>();
                sum_d += H.dot(C.homogeneous());
            }
            if (sum_d < 0) {
                H = -H;
            }

            out_planes.emplace_back(H);
            inlier_points.emplace_back(points);
            std::cout << "Plane parameter: " << H.transpose() << std::endl;
        }

        size_t num_plane = out_planes.size();
        // Merge planes.
        const float cos_thres = std::cos(2.0 * M_PI / 180.0);
        std::vector<int> plane_idx(out_planes.size());
        std::iota(plane_idx.begin(), plane_idx.end(), 0);

        std::vector<bool> removed(out_planes.size(), false);
        while(out_planes.size() > 1) {
            bool merge_success = false;
            for (i = 0; i < out_planes.size(); ++i) {
                if (removed[i]) continue;
                Eigen::Vector4d plane1 = out_planes.at(i);
                float inlier_error = inlier_errors.at(i) * cad_options.dist_ratio_point_to_plane2;
                for (j = 0; j < i; ++j) {
                    if (removed[j]) continue;
                    Eigen::Vector4d plane2 = out_planes.at(j);
                    float cos = plane1.head<3>().dot(plane2.head<3>());
                    float dist = std::abs(plane1.w() - plane2.w());
                    if (cos > cos_thres && dist < inlier_error) {
                        std::cout << StringPrintf("Merge Plane#%d => #%d\n", plane_idx.at(i), plane_idx.at(j));

                        inlier_points.at(j).insert(inlier_points.at(j).end(), inlier_points.at(i).begin(), inlier_points.at(i).end());
                        inlier_errors.at(j) = (inlier_errors.at(j) + inlier_errors.at(i)) * 0.5;

                        // update plane equation.
                        PlaneLocalEstimator plane_estimator;
                        auto H = plane_estimator.Estimate(inlier_points.at(j)).at(0);
                        if (H.head<3>().dot(out_planes.at(j).head<3>()) < 0) {
                            out_planes.at(j) = -H;
                        } else {
                            out_planes.at(j) = H;
                        }

                        removed[i] = true;
                        merge_success = true;
                        break;
                    }
                }
                if (merge_success) {
                    break;
                }
            }
            if (!merge_success) {
                break;
            }
        }

        for (i = 0, j = 0; i < out_planes.size(); ++i) {
            if (!removed[i]) {
                out_planes[j] = out_planes[i];
                inlier_points[j] = inlier_points[i];
                inlier_errors[j] = inlier_errors[i];
                plane_idx[j] = plane_idx[i];
                j++;
            }
        }
        out_planes.resize(j);
        inlier_points.resize(j);
        inlier_errors.resize(j);
        plane_idx.resize(j);

        std::cout << "Merge " << num_plane - out_planes.size() << " planes" << std::endl;

        // Removal small isolated island.
        num_plane = out_planes.size();
        size_t max_num_inlier = 0;
        std::for_each(inlier_points.begin(), inlier_points.end(), 
            [&](const std::vector<Eigen::Vector3d>& layer_points) {
                max_num_inlier = std::max(max_num_inlier, layer_points.size());
            });
        for (i = 0, j = 0; i < out_planes.size(); ++i) {
            float ratio = inlier_points.at(i).size() * 1.0 / max_num_inlier;
            std::cout << "Plane inlier points ratio: " << ratio << std::endl;
            if (ratio >= cad_options.inlier_ratio_plane_points) {
                out_planes.at(j) = out_planes.at(i);
                inlier_points.at(j) = inlier_points.at(i);
                inlier_errors.at(j) = inlier_errors.at(i);
                j = j + 1;
            }
        }
        out_planes.resize(j);
        inlier_points.resize(j);
        inlier_errors.resize(j);

        std::cout << "Removal " << num_plane - out_planes.size() << " small planes" << std::endl;
        
        for (i = 0; i < out_planes.size(); ++i) {
            std::cout << StringPrintf("Refine Plane#%d (%d points)\n", i, inlier_points.at(i).size());
            Timer timer;
            timer.Start();
            RANSACOptions ransac_option;
            ransac_option.max_num_trials = 10000;
            ransac_option.max_error = inlier_errors.at(i);
            LORANSAC<PlaneEstimator, PlaneLocalEstimator> estimator(ransac_option);
            auto report = estimator.Estimate(inlier_points.at(i));
            auto H = report.model;
            if (H.head<3>().dot(out_planes.at(i).head<3>()) < 0) {
                out_planes.at(i) = -H;
            } else {
                out_planes.at(i) = H;
            }
            timer.PrintSeconds();
        }

        BubbleSort(out_planes, inlier_points, inlier_errors, false);
        std::cout << "Sorted plane parameters" << std::endl;
    } else if (wall_plane_normals.size() == 2) {
        std::cout << "Warning! Failed to find consistent planes, try to fit plane assisted with wall normals!" << std::endl;

        std::vector<int> plane_facets;
        for (size_t i = 0; i < mesh.faces_.size(); ++i) {
            auto facet = mesh.faces_.at(i);
            auto label0 = mesh.vertex_labels_.at(facet[0]);
            auto label1 = mesh.vertex_labels_.at(facet[1]);
            auto label2 = mesh.vertex_labels_.at(facet[2]);
            if (label0 == LABLE_GROUND && label1 == LABLE_GROUND && label2 == LABLE_GROUND) {
                plane_facets.push_back(i);
            }
        }
        
        double total_area = 0.0;
        Eigen::Vector3d m_center(0.0, 0.0, 0.0);
        for (auto i : plane_facets) {
            auto facet = mesh.faces_.at(i);
            auto v0 = mesh.vertices_.at(facet[0]);
            auto v1 = mesh.vertices_.at(facet[1]);
            auto v2 = mesh.vertices_.at(facet[2]);
            double a = (v0 - v1).norm();
            double b = (v0 - v2).norm();
            double c = (v1 - v2).norm();
            double s = (a + b + c) * 0.5;
            double area = std::sqrt(s * (s - a) * (s - b) * (s - c));
            total_area += area;
            m_center += area * (v0 + v1 + v2) / 3;
        }
        m_center /= total_area;

        Eigen::Vector3d m_normal = (wall_plane_normals.at(0).cross(wall_plane_normals.at(1)));
        Eigen::Vector4d plane;
        plane.head<3>() = m_normal;
        plane[3] = -m_normal.dot(m_center);

        // Correct Plane Normal.
        size_t num_facet = 0;
        Eigen::Vector3d m_facet_normal(0, 0, 0);
        for (auto i : plane_facets) {
            auto facet = mesh.faces_.at(i);
            auto v0 = mesh.vertices_.at(facet[0]);
            auto v1 = mesh.vertices_.at(facet[1]);
            auto v2 = mesh.vertices_.at(facet[2]);
            auto normal = (v1 - v0).cross(v2 - v0).normalized();
            m_facet_normal += normal;
            num_facet++;
        }
        m_facet_normal = (m_facet_normal / num_facet).normalized();
        if (m_facet_normal.dot(plane.head<3>()) < 0) {
            plane = -plane;
        }

        inlier_points.resize(1);
        inlier_points[0] = points;
        out_planes.emplace_back(plane);
    } else {
        // Estimate planar equation.

        std::cout << "Warning! Failed to find consistent planes, try to fit plane directly!" << std::endl;

        PlaneLocalEstimator local_estimator;
        auto H = local_estimator.Estimate(points).at(0);

        std::vector<Eigen::Vector3d>& refine_points = points;
        double mean_dist = 0.0;
        for (i = 0; i < refine_points.size(); ++i) {
            mean_dist += std::abs(H.dot(refine_points[i].homogeneous()));
        }
        mean_dist /= refine_points.size();

        std::cout << "Refine plane parameter" << std::endl;
        std::cout << "Collect " << refine_points.size() << " ground points" << std::endl;

        // Estimate planar equation.
        RANSACOptions ransac_options;
        ransac_options.max_error = 2.0 * mean_dist;
        LORANSAC<PlaneEstimator, PlaneLocalEstimator> estimator(ransac_options);
        const auto refine_report = estimator.Estimate(refine_points);
        if (refine_report.success) {
            Eigen::Vector4d refine_H = refine_report.model;
            inlier_points.resize(1);
            // Correct Plane Normal.
            inlier_points[0].reserve(refine_report.inlier_mask.size());
            for (i = 0; i < refine_report.inlier_mask.size(); ++i) {
                if (refine_report.inlier_mask.at(i)) {
                    inlier_points[0].emplace_back(refine_points.at(i));
                }
            }

            // Correct Plane Normal.
            std::unordered_set<size_t> visib_cams;
            for (i = 0; i < points.size(); ++i) {
                for (auto vis : visibilities.at(i)) {
                    visib_cams.insert(vis);
                }
            }
            float sum_d = 0.0f;
            for (auto vis : visib_cams) {
                const mvs::Image image = images.at(vis);
                Eigen::Vector3d C = Eigen::Vector3f(image.GetC()).cast<double>();
                sum_d += refine_H.dot(C.homogeneous());
            }
            out_planes.emplace_back(refine_H);
        } else {
            std::cout << "Fitting plane with least square method" << std::endl;
            PlaneLocalEstimator estimator;
            Eigen::Vector4d refine_H = estimator.Estimate(refine_points).at(0);
            inlier_points.resize(1);
            out_planes.emplace_back(refine_H);
        }
    }
    for (i = 0; i < out_planes.size(); ++i) {
        auto plane = out_planes.at(i);
        std::cout << "plane#" << i << ": " << plane.transpose() << std::endl;
    }
    return !out_planes.empty();
}

bool EstimateMultiPlanesByGravity(const CadMapOptions& options, 
    const std::vector<PlyPoint>& fused_points,
    std::vector<Eigen::Vector4d>& planes,
    std::vector<std::vector<Eigen::Vector3d> >& inlier_points) {
    std::cout << "Estimate Planes from gravity!" << std::endl;
    std::vector<Eigen::Vector3d> plane_points;
    for (auto & point : fused_points) {
        if (point.s_id == LABLE_GROUND) {
            Eigen::Vector3d ppoint = Eigen::Vector3f(&point.x).cast<double>();
            plane_points.push_back(ppoint);
        }
    }

    // Estimate planar equation.
    RANSACOptions ransac_options;
    ransac_options.max_error = 0.5;
    LORANSAC<PlaneEstimator, PlaneLocalEstimator> estimator(ransac_options);

    const int num_plane_point = plane_points.size();
    float inlier_ratio = 0.0;
    planes.clear();
    inlier_points.clear();
    do {
        std::cout << "Estimate plane from " << plane_points.size() << " points" << std::endl;
        const auto report = estimator.Estimate(plane_points);
        if (!report.success) {
            break;
        }
        Eigen::Vector4d plane = report.model;

        Eigen::Vector3d C(0.0, 0.0, 0.0);
        std::vector<Eigen::Vector3d> layer_inlier_points, outlier_points;
        layer_inlier_points.reserve(plane_points.size());
        outlier_points.reserve(plane_points.size());
        for (size_t i = 0; i < report.inlier_mask.size(); ++i) {
            if (report.inlier_mask.at(i)) {
                layer_inlier_points.emplace_back(plane_points.at(i));
                C += plane_points.at(i);
            } else {
                outlier_points.emplace_back(plane_points.at(i));
            }
        }
        if (layer_inlier_points.empty()) {
            break;
        }
        inlier_ratio = layer_inlier_points.size() * 1.0f / num_plane_point;
        std::cout << "planarity: " << inlier_ratio << std::endl;
        if (inlier_ratio < options.inlier_ratio_plane_points) {
            break;
        }
        inlier_points.emplace_back(layer_inlier_points);
        // if (plane.z() > 0) {
        //     plane = -plane;
        // }
        C /= layer_inlier_points.size();
        
        // Compute plane's projection of C.
        double a = plane[0], b = plane[1], c = plane[2], d = plane[3];
        Eigen::Vector3d xp;
        xp[0] = (b * b + c * c) * C[0] - a * (b * C[1] + c * C[2] + d);
        xp[1] = (a * a + c * c) * C[1] - b * (a * C[0] + c * C[2] + d);
        xp[2] = (a * a + b * b) * C[2] - c * (a * C[0] + b * C[1] + d);
        
        // Update plane equation according to gravity.
        plane[0] = plane[1] = 0.0;
        plane[2] = -1;
        plane[3] = xp[2];
        planes.push_back(plane);

        plane_points = outlier_points;
    } while(inlier_ratio >= options.inlier_ratio_plane_points);

    if (planes.empty()) {
        Eigen::Vector3d C(0.0, 0.0, 0.0);
        if (!plane_points.empty()) {
            for (auto & point : plane_points) {
                C += point;
            }
            C /= plane_points.size();
        } else {
            for (auto & point : fused_points) {
                C[0] += point.x;
                C[1] += point.y;
                C[2] += point.z;
            }
            C /= fused_points.size();
        }

        // Update plane equation according to gravity.
        Eigen::Vector4d plane;
        plane[0] = plane[1] = 0.0;
        plane[2] = -1;
        plane[3] = C[2];
        planes.push_back(plane);
    } else {
        size_t best_plane_idx = 0;
        size_t max_num_inlier = 0;
        for (size_t i = 0; i < planes.size(); ++i) {
            if (max_num_inlier < inlier_points[i].size()) {
                max_num_inlier = inlier_points[i].size();
                best_plane_idx = i;
            }
        }
        Eigen::Vector4d best_plane = planes[best_plane_idx];
        planes.clear();
        planes.shrink_to_fit();
        planes.push_back(best_plane);
    }
    for (size_t i = 0; i < planes.size(); ++i) {
        std::cout << "Plane#" << i << ": " << planes[i].transpose() << std::endl;
    }
    return !planes.empty();
}

Eigen::Matrix3d CalcTransformation(
    const std::vector<std::vector<PlanePoint> >& noplane_points,
    const std::vector<Eigen::Vector3d >& wall_plane_normals,
    const Eigen::Vector4d& plane) {
    
    Eigen::Vector3d zaxis(plane.data());
    if(!CheckPointsNormal(noplane_points) || wall_plane_normals.empty())
    {
        Eigen::Vector3d tmpv; 
        if(plane.x() != 0 || plane.y() != 0)
        {
            tmpv[0] = plane.y();
            tmpv[1] = -plane.x();
            tmpv[2] = 0;
        } else{
            tmpv[0] = 0;
            tmpv[1] = plane.z();
            tmpv[2] = -plane.y();
        }
        
        Eigen::Vector3d yaxis = zaxis.cross(tmpv).normalized();//(init_y_dir - init_y_dir.dot(zaxis) * zaxis).normalized();
        Eigen::Vector3d xaxis = yaxis.cross(zaxis);
        Eigen::Matrix3d transformation;

        transformation.row(0) = yaxis;
        transformation.row(1) = xaxis;
        transformation.row(2) = -zaxis;

        return transformation;
    }

    Eigen::Vector3d final_normal_refine = wall_plane_normals[0].normalized();
    Eigen::Vector3d yaxis = (final_normal_refine - final_normal_refine.dot(zaxis) * zaxis).normalized();
    Eigen::Vector3d xaxis = yaxis.cross(zaxis);
    Eigen::Matrix3d transformation;

    transformation.row(0) = yaxis;
    transformation.row(1) = xaxis;
    transformation.row(2) = -zaxis;

    return transformation;
}

Eigen::Matrix3x4d ComputePlaneAxis(
    const std::vector<Eigen::Vector4d>& planes,
    const std::vector<Eigen::Vector3d>& wall_plane_normals,
    const std::vector<std::vector<PlanePoint> >& noplane_points,
    const std::vector<std::vector<PlanePoint> >& plane_points,
    const std::vector<std::vector<Eigen::Vector3d> >& layered_cams,
    int& pivot_layer_id) {

    int layer_idx = 0, num_max_cam = 0;
    for (size_t i = 0; i < layered_cams.size(); ++i) {
        auto cams = layered_cams.at(i);
        if (cams.size() > num_max_cam && !noplane_points.empty()) {
            num_max_cam = cams.size();
            layer_idx = i;
        }
    }
    pivot_layer_id = layer_idx;

    auto plane = planes.at(layer_idx);

    Eigen::Matrix3d transformation = CalcTransformation(noplane_points, wall_plane_normals, plane);
    
    const Eigen::Vector3d m_vNormal = plane.head<3>();
    Eigen::Vector3d plane_centroid(0, 0, 0);
    for (auto & point : plane_points.at(layer_idx)) {
        Eigen::Vector3d X = transformation * (point.X - m_vNormal * point.dist);
        plane_centroid += X;
    }
    if (plane_points.at(layer_idx).size() != 0) {
        plane_centroid /= plane_points.at(layer_idx).size();
    }

    Eigen::Matrix3x4d C = Eigen::Matrix3x4d::Identity();
    C(2, 3) = -plane_centroid[2];

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = transformation;
    
    Eigen::Matrix3x4d MC = C * T;

    return MC;
}

Eigen::Matrix3x4d ComputePlaneAxisByGravity(
    const std::vector<Eigen::Vector4d>& planes,
    const std::vector<std::vector<PlanePoint> >& noplane_points,
    const std::vector<std::vector<PlanePoint> >& plane_points,
    int& pivot_layer_id) {
    std::vector<std::pair<int, Eigen::Vector3d> > barycenters;
    for (std::size_t i = 0; i < plane_points.size(); ++i) {
        Eigen::Vector3d C(0, 0, 0);
        for (const auto & point : plane_points.at(i)) {
            C += point.X;
        }
        barycenters.emplace_back(i, C / plane_points.at(i).size());
    }
    Eigen::Vector3d plane_centroid(0, 0, 0);
    if (!barycenters.empty()) {
        std::sort(barycenters.begin(), barycenters.end(), 
        [&](const auto& X, const auto& Y) {
            return X.second.z() < Y.second.z();
        });
        pivot_layer_id = barycenters.at(0).first;
        plane_centroid = barycenters.at(0).second;
    }

    Eigen::Matrix3x4d C = Eigen::Matrix3x4d::Identity();
    C(2, 3) = -plane_centroid[2];

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    
    Eigen::Matrix3x4d MC = C * T;

    return MC;
}

void ConvertToPlaneAxis(const Eigen::Matrix3x4d& MC,
    std::vector<PlanePoint>& noplane_points,
    std::vector<PlanePoint>& plane_points,
    std::vector<PlanePoint>& roof_points,
    std::vector<Eigen::Vector3d>& layered_cams) {
    for (auto & point : noplane_points) {
        point.X = MC * point.X.homogeneous();
        point.normal = MC * point.normal.homogeneous();
    }
    for (auto & point : plane_points) {
        point.X = MC * point.X.homogeneous();
        point.normal = MC * point.normal.homogeneous();
    }
    for (auto & point : roof_points) {
        point.X = MC * point.X.homogeneous();
        point.normal = MC * point.normal.homogeneous();
    }
    for (auto & cam : layered_cams) {
        cam = MC * cam.homogeneous();
    }
}

bool IsFinite(const Eigen::Vector3d& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

void FilterPoints(std::vector<std::vector<PlanePoint> >& nonplane_points,
                  std::vector<std::vector<PlanePoint> >& plane_points,
                  std::vector<std::vector<PlanePoint> >& roof_points, 
                  std::vector<Eigen::Vector4d>& planes,
                  const float low_ratio = 0.02f, const float high_ratio = 0.98f){
    auto Filter = [&](std::vector<PlanePoint>& points) {
        if (points.empty()) {
            return;
        }

        std::sort(points.begin(), points.end(), 
            [&](const PlanePoint& pp1, const PlanePoint& pp2) {
                return std::fabs(pp1.dist) < std::fabs(pp2.dist);
            });

        size_t num_point = points.size();
        size_t start_point = (num_point - 1) * low_ratio;
        size_t end_point = std::min(size_t((num_point - 1) * high_ratio), size_t(num_point - 1));
        double low_thres = std::fabs(points.at(start_point).dist);
        double high_thres = std::fabs(points.at(end_point).dist);

        size_t i, j;
        for (i = start_point, j = 0; i <= end_point; ++i) {
            double fdist = std::fabs(points.at(i).dist);
            if (fdist < low_thres || fdist > high_thres) {
                continue;
            }
            points.at(j) = points.at(i);
            j = j + 1;
        }
        points.resize(j);
    };

    size_t i, j;
    for (i = 0, j = 0; i < nonplane_points.size(); ++i) {
        auto& layer_nonplane_points = nonplane_points.at(i);
        Filter(layer_nonplane_points);

        std::cout << StringPrintf("Collect %d nonplane points of Layer#%d\n",
            layer_nonplane_points.size(), i);
        std::cout << StringPrintf("Collect %d plane points of Layer#%d\n",
            plane_points.at(i).size(), i);
        if (nonplane_points.at(i).empty()) {
            continue;
        }
        
        // std::sort(layer_nonplane_points.begin(), layer_nonplane_points.end(),
        //     [&](const PlanePoint& p1, const PlanePoint& p2) {
        //         return p1.dist < p2.dist;
        //     });
        
        auto& layer_roof_points = roof_points.at(i);
        std::sort(layer_roof_points.begin(), layer_roof_points.end(),
            [&](const PlanePoint& p1, const PlanePoint& p2) {
                return p1.dist > p2.dist;
            });

        nonplane_points.at(j) = layer_nonplane_points;
        roof_points.at(j) = layer_roof_points;
        plane_points.at(j) = plane_points.at(i);
        planes.at(j) = planes.at(i);
        j = j + 1;
    }
    nonplane_points.resize(j);
    roof_points.resize(j);
    plane_points.resize(j);
    planes.resize(j);
}

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

FT ComputeAvergeSapcing(
    std::vector<FT>& point_spacings, 
    const std::vector<point_3_t> &points,
    const unsigned int nb_neighbors,
    tree_3_t& tree){

    FT average_spacing = (FT)0.0;
    std::size_t point_num = points.size();
    point_spacings.resize(point_num);
    
    // iterate over input points, compute and output point spacings
    for (std::size_t i = 0; i < point_num; i++) {
        const auto &query = points[i];
        // performs k + 1 queries (if unique the query point is
        // output first). search may be aborted when k is greater
        // than number of input points
        neighbor_search_3_t search(tree, query, nb_neighbors + 1);
        auto &point_spacing = point_spacings[i];
        point_spacing = (FT)0.0;
        std::size_t k = 0;
        for (search_3_iterator_t search_iterator = search.begin(); 
            search_iterator != search.end() && k <= nb_neighbors; 
            search_iterator++, k++)
        {
            point_spacing += std::sqrt(search_iterator->second);
        }
        // output point spacing
        if (k > 1) {
            point_spacing /= (FT)(k - 1);
        }

        average_spacing += point_spacing;
    }
    std::cout << std::endl;
    average_spacing /= (FT)point_num;
    // std::cout << " =>Average spacing: " << average_spacing << std::endl;
    return average_spacing;
}

void UpdateModelFaces(const std::vector<int>& vertex_ids,
                            std::vector<Eigen::Vector3i>& faces)
{
    std::unordered_map<int, int> vertex_ids_swap_map;
    //arrange face vertex ids for new mesh
    int v_id = 0;
    for(auto id : vertex_ids)
    {
        vertex_ids_swap_map[id] = v_id;
        v_id++;
    }
    for(auto& face : faces)
    {
        face[0] = vertex_ids_swap_map[face[0]];
        face[1] = vertex_ids_swap_map[face[1]];
        face[2] = vertex_ids_swap_map[face[2]];
    }
}

void ExtractWallPoints(const TriangleMesh & mesh, std::vector<Eigen::Vector3d> & wall_points, 
                       std::vector<Eigen::Vector3d> & wall_normals) {
    wall_points.reserve(mesh.vertices_.size());
    wall_normals.reserve(mesh.vertices_.size());
    for (size_t i = 0; i < mesh.vertices_.size(); ++i) {
        int8_t label = mesh.vertex_labels_.at(i);
        if (label == LABEL_WALL/* || label == LABEL_BUILDING*/) {
            wall_points.emplace_back(mesh.vertices_.at(i));
            wall_normals.emplace_back(mesh.vertex_normals_.at(i));
        }
    }
    wall_points.shrink_to_fit();
    wall_normals.shrink_to_fit();
}

Eigen::Vector3d ExtractGroundOrientation(const TriangleMesh & mesh) {
    Eigen::Vector3d m_normal(0, 0, 0);
    for (size_t i = 0; i < mesh.vertices_.size(); ++i) {
        int8_t label = mesh.vertex_labels_.at(i);
        if (label == LABLE_GROUND) {
            m_normal += mesh.vertex_normals_.at(i);
        }
    }
    m_normal.normalize();
    std::cout << "ground normal: " << m_normal.transpose() << std::endl;
    return m_normal;
}

void DetermineWallClusters(const std::vector<Eigen::Vector3d> & wall_points, 
                           const std::vector<Eigen::Vector3d> & wall_normals,
                           const Eigen::Vector3d & plane_orientation,
                           std::vector<Eigen::Vector3d> & plane_normals) {
	std::cout << "wall_points: " << wall_points.size() << std::endl;
    const int min_wall_points_num = 10000;
	
    if (wall_points.size() < min_wall_points_num) {
        return;
    }
        
    plane_normals.clear();
    bool get_plane_normal = false;
    get_plane_normal = DetectPlaneWithRansac(wall_points, wall_normals, plane_orientation, plane_normals);

    int class_num = 4; 
    if(!get_plane_normal) {
        std::cout << " Start to reinit normals ... " ;
        std::vector<cv::Point3f> normals;
        normals.reserve(wall_normals.size());
        for (const auto & normal : wall_normals) {
            normals.push_back(cv::Point3f(normal.x(), normal.y(), normal.z()));
        }
        cv::Mat labels;
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1);
        std::cout << " Start to cal normals labels... " ;
        cv::kmeans(normals, class_num, labels, criteria, 3, cv::KMEANS_RANDOM_CENTERS);
        std::cout << " finished!" << std::endl;

        std::cout << " Start to cluster normals ... " ;
        std::vector<std::vector<cv::Point3f> > normal_clusters(class_num);
        for(int i = 0; i < normals.size(); i++){
            int normal_class =  labels.at<int>(i); 
            if(normal_class < 0 || normal_class >= class_num) continue;
            cv::Point3f normal = normals.at(i);
            normal_clusters[normal_class].push_back(normal);
        }
        std::cout << " finished!" << std::endl;

        float angle_thred = 35 / 180.0 * M_PI;
        float cos_value_thred = std::cos(angle_thred);
        float ground_cos_value_thred = std::cos(10 / 180.0 * M_PI);

        int max_cluster_idx = 0, second_cluster_idx = 0;
        int max_cluster_num = 0, second_cluster_num = 0;
        Eigen::Vector3d max_normal(0, 0, 0), second_normal(0, 0, 0);
        for(int i = 0; i < class_num; i++){
            Eigen::Vector3d m_normal(0.0, 0.0, 0.0);
            for (int j = 0; j < normal_clusters[i].size(); ++j) {
                const cv::Point3f &p = normal_clusters[i][j];
                if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) continue;
                if(std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z)) continue;
                m_normal += Eigen::Vector3d(p.x, p.y, p.z);
            }
            m_normal = (m_normal / normal_clusters[i].size()).normalized();
            if (std::abs(max_normal.dot(m_normal)) < cos_value_thred &&
                std::abs(plane_orientation.dot(m_normal)) < ground_cos_value_thred) {
                if(normal_clusters[i].size() > max_cluster_num){
                    second_cluster_num = max_cluster_num;
                    second_cluster_idx = max_cluster_idx;
                    second_normal = max_normal;
                    max_cluster_num = normal_clusters[i].size();
                    max_cluster_idx = i;
                    max_normal = m_normal;
                } else if (normal_clusters[i].size() > second_cluster_num) {
                    second_cluster_num = normal_clusters[i].size();
                    second_cluster_idx = i;
                    second_normal = m_normal;
                }
            }
        }

        plane_normals.clear();

        std::cout << " Max normal_clusters size : " <<   normal_clusters[max_cluster_idx].size() << std::endl;
        std::cout << "max normal : " << max_normal.transpose() << std::endl;

        cv::Point3f f_normal(max_normal[0], max_normal[1], max_normal[2]);
        Eigen::Vector3d max_normal_refine(0, 0, 0);
        int refine_normal_count = 0;
        for(int i = 0; i < normal_clusters[max_cluster_idx].size(); i++){
            const cv::Point3f &p = normal_clusters[max_cluster_idx][i];
            if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) continue;
            if(std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z)) continue;
            if(p.dot(f_normal) < 0.9) continue;
            max_normal_refine += Eigen::Vector3d(p.x, p.y, p.z);
            refine_normal_count ++;
        }
        std::cout << " refine normal cluster size : " << refine_normal_count << std::endl;
        max_normal_refine = max_normal_refine.normalized();
        plane_normals.emplace_back(max_normal_refine);

        std::cout << " Second normal_clusters size : " <<   normal_clusters[second_cluster_idx].size() << std::endl;
        std::cout << "second normal : " << second_normal.transpose() << std::endl;

        f_normal = cv::Point3f(second_normal[0], second_normal[1], second_normal[2]);
        Eigen::Vector3d second_normal_refine(0, 0, 0);
        refine_normal_count = 0;
        for(int i = 0; i < normal_clusters[second_cluster_idx].size(); i++){
            const cv::Point3f &p = normal_clusters[second_cluster_idx][i];
            if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) continue;
            if(std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z)) continue;
            if(p.dot(f_normal) < 0.9) continue;
            second_normal_refine += Eigen::Vector3d(p.x, p.y, p.z);
            refine_normal_count ++;
        }
        std::cout << " refine normal cluster size : " << refine_normal_count << std::endl;
        second_normal_refine = second_normal_refine.normalized();
        plane_normals.emplace_back(second_normal_refine);
    }
}

void PointCloudPartation(const CadMapOptions& options,
                         const std::vector<PlyPoint>& fused_points,
                         const std::vector<std::vector<uint32_t> >& vis_points,
                         const TriangleMesh& mesh,
                         std::vector<Eigen::Vector4d>& planes,
                         std::vector<std::vector<PlanePoint> >& nonplane_points,
                         std::vector<std::vector<PlanePoint> >& plane_points,
                         std::vector<std::vector<PlanePoint> >& roof_points) {
    size_t i, j;
    size_t num_point = fused_points.size();
    float average_spacing, max_point_spacing, median_dist = 0.0f;
    std::vector<float> point_spacings, dists;
    const bool has_prior = mesh.faces_.size() != 0;

    if (has_prior) {
        std::list<Triangle> triangles;
        for (auto facet : mesh.faces_) {
            auto a = mesh.vertices_.at(facet[0]);
            auto b = mesh.vertices_.at(facet[1]);
            auto c = mesh.vertices_.at(facet[2]);
            if (!IsFinite(a) || !IsFinite(b) || !IsFinite(c)) {
                continue;
            }
            if ((a - b).norm() < 1e-6 || (b - c).norm() < 1e-6 || (a - c).norm() < 1e-6) {
                continue;
            }
            Triangle tri(Point(a[0], a[1], a[2]), Point(b[0], b[1], b[2]), 
                        Point(c[0], c[1], c[2]));
            triangles.emplace_back(tri);
        }
        std::cout << "Construct AABB Tree" << std::endl;
        // Contruct AABB tree.
        Tree tree(triangles.begin(), triangles.end());

        tree.accelerate_distance_queries();
        
        std::cout << "Compute Point to Model distance" << std::endl;
        dists.resize(num_point, 0);

        size_t progress = 0;
		uint64_t indices_print_step = num_point / 100 + 1;
        auto ComputePointDistance = [&](size_t start, size_t end) {
            for (size_t k = start; k < end; ++k) {
                auto point = fused_points.at(k);
                if (point.s_id == LABLE_CEILING || point.s_id == LABEL_LAMP) {
                    continue;
                }

                Point query(point.x, point.y, point.z);
                double dist = tree.squared_distance(query);
                dists[k] = dist;
                if (progress % indices_print_step == 0) {
                    std::cout << StringPrintf("\rCompute Point#%d / %d", progress, num_point);
                }
                progress++;
            }
        };

        const int num_eff_threads = GetEffectiveNumThreads(-1);
        std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
        std::unique_ptr<ThreadPool> thread_pool;
        thread_pool.reset(new ThreadPool(num_eff_threads));

        const size_t num_slice = (num_point + num_eff_threads - 1) / num_eff_threads;
        float *ptr = dists.data();
        for (int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
            int start = std::min(thread_idx * num_slice, num_point);
            int end = std::min(num_slice * (thread_idx + 1), num_point);
            thread_pool->AddTask(ComputePointDistance, start, end);
        }
        thread_pool->Wait();

        std::cout << "Find optimal median distance" << std::endl;
        // Time-consuming for extreme large points
        std::vector<float> dists_sort = dists;
        float nth_factor = 0.85;
        while(median_dist < 1e-8 && nth_factor < 1.0) {
            size_t nth = dists_sort.size() * nth_factor;
            std::nth_element(dists_sort.begin(), dists_sort.begin() + nth,
                dists_sort.end());
            median_dist = dists_sort[nth];
            nth_factor += 0.02;
        }
        std::cout << "Median distance of point to mesh: " << median_dist 
                  << std::endl;
    } else {
        ComputeAverageDistance(fused_points, point_spacings, &average_spacing, options.nb_neighbors);
        max_point_spacing = options.max_spacing_factor * average_spacing;
    }

    // nonplane_points.reserve(fused_points.size());
    // plane_points.reserve(fused_points.size());
    nonplane_points.resize(planes.size());
    roof_points.resize(planes.size());
    plane_points.resize(planes.size());
    for (i = 0; i < planes.size(); ++i) {
        nonplane_points.at(i).reserve(num_point);
        roof_points.at(i).reserve(num_point);
        plane_points.at(i).reserve(num_point);
    }

    float max_dist_to_ground = FLT_MAX;
    if (planes.size() == 1){
        max_dist_to_ground = FLT_MAX;
    }else{
        for(i = 0; i < planes.size() - 1; i++){
            for (j = i + 1; j < planes.size(); j++){
                float dist = fabs(planes[i][3] - planes[j][3]);
                max_dist_to_ground = std::min(max_dist_to_ground, dist);
            }
        }
    }
    printf("max_dist_to_ground: %f\n", max_dist_to_ground);

    for (i = 0; i < num_point; ++i) {
        if (has_prior) {
            if (dists[i] > 1.1 * median_dist) continue;
        } else if (point_spacings[i] > max_point_spacing) {
            continue;
        }
        auto & point = fused_points.at(i);
        Eigen::Vector3d X = Eigen::Vector3f((float*)&point).cast<double>();
        Eigen::Vector3d N = Eigen::Vector3f((float*)&point + 3).cast<double>();

        int layer_idx = -1;
        if (planes.size() == 1) {
            layer_idx = 0;
        } else if (point.s_id == LABLE_GROUND) {
            int best_layer_id = -1;
            double min_dist_to_plane = FLT_MAX;
            for (j = 0; j < planes.size(); ++j) {
                const auto & plane = planes.at(j);
                double dist = std::abs(plane.dot(X.homogeneous()));
                if (min_dist_to_plane > dist) {
                    min_dist_to_plane = dist;
                    best_layer_id = j;
                }
            }
            if (best_layer_id != -1 && 
                min_dist_to_plane < max_dist_to_ground) {
                layer_idx = best_layer_id;
            }
        } else {
            for (j = 0; j < planes.size(); ++j) {
                if (j + 1 < planes.size()) {
                    auto plane0 = planes.at(j);
                    auto plane1 = planes.at(j + 1);
                    double dist0 = plane0.dot(X.homogeneous());
                    double dist1 = plane1.dot(X.homogeneous());
                    if (dist0 >= 0 && dist1 < 0) {
                        layer_idx = j;
                        break;
                    }
                } else {
                    auto plane = planes.at(j);
                    double dist = plane.dot(X.homogeneous());
                    if (dist >= 0) {
                        layer_idx = j;
                        break;
                    }
                }
            }
        }
        if (layer_idx != -1) {
            double dist = planes.at(layer_idx).dot(X.homogeneous());
            PlanePoint pp;
            pp.dist = dist;
            pp.X = X;
            pp.normal = N;
            pp.rgb = Eigen::Vector3ub(&point.r);
            pp.viss = vis_points.at(i);
            pp.s_id = point.s_id;
            
            try {
                if (point.s_id == LABLE_GROUND) {
                    plane_points.at(layer_idx).emplace_back(pp);
                } else if (point.s_id == LABLE_CEILING) {
                    roof_points.at(layer_idx).emplace_back(pp);
                } else {
                    nonplane_points.at(layer_idx).emplace_back(pp);
                }
            } catch(std::bad_alloc &ba) {
                std::cout << "plane_points: " << plane_points.at(layer_idx).size() << std::endl;
                std::cout << "roof_points: " << roof_points.at(layer_idx).size() << std::endl;
                std::cout << "nonplane_points: " << nonplane_points.at(layer_idx).size() << std::endl;
                std::cout << ba.what() << std::endl;
            }
        }
    }
}

void ExtractSubMesh(const TriangleMesh& mesh, 
                    const std::vector<int>& vtx_map,
                    TriangleMesh& sub_mesh) {
    if (vtx_map.empty()) {
        return ;
    }
    std::unordered_set<int> m_vtx_map;
    for (auto & vtx : vtx_map) {
        m_vtx_map.insert(vtx);
    }

    std::unordered_map<int, int> new_vtx_map;
    std::vector<Eigen::Vector3d> new_vertices;
    new_vertices.reserve(mesh.vertices_.size());
    std::vector<Eigen::Vector3i> new_facets;
    new_facets.reserve(mesh.faces_.size());
    for (auto & facet : mesh.faces_) {
        if (m_vtx_map.find(facet[0]) == m_vtx_map.end() &&
            m_vtx_map.find(facet[1]) == m_vtx_map.end() &&
            m_vtx_map.find(facet[2]) == m_vtx_map.end()) {
            continue;
        }

        Eigen::Vector3i new_facet;
        if (new_vtx_map.find(facet[0]) != new_vtx_map.end()) {
            new_facet[0] = new_vtx_map.at(facet[0]);
        } else {
            new_vtx_map[facet[0]] = new_vertices.size();
            new_facet[0] = new_vertices.size();
            new_vertices.emplace_back(mesh.vertices_.at(facet[0]));
        }
        
        if (new_vtx_map.find(facet[1]) != new_vtx_map.end()) {
            new_facet[1] = new_vtx_map.at(facet[1]);
        } else {
            new_vtx_map[facet[1]] = new_vertices.size();
            new_facet[1] = new_vertices.size();
            new_vertices.emplace_back(mesh.vertices_.at(facet[1]));
        }

        if (new_vtx_map.find(facet[2]) != new_vtx_map.end()) {
            new_facet[2] = new_vtx_map.at(facet[2]);
        } else {
            new_vtx_map[facet[2]] = new_vertices.size();
            new_facet[2] = new_vertices.size();
            new_vertices.emplace_back(mesh.vertices_.at(facet[2]));
        }
        new_facets.emplace_back(new_facet);
    }

    std::swap(sub_mesh.vertices_, new_vertices);
    std::swap(sub_mesh.faces_, new_facets);
}

void MeshPartation(const CadMapOptions& options,
                   const std::vector<Eigen::Vector4d>& planes,
                   const TriangleMesh& mesh,
                   std::vector<TriangleMesh>& nonplane_meshes,
                   std::vector<TriangleMesh>& plane_meshes,
                   std::vector<TriangleMesh>& roof_meshes) {
    size_t i, j;
    size_t num_vert = mesh.vertices_.size();

    float max_dist_to_ground = FLT_MAX;
    if (planes.size() == 1){
        max_dist_to_ground = FLT_MAX;
    }else{
        for(i = 0; i < planes.size() - 1; i++){
            for (j = i + 1; j < planes.size(); j++){
                float dist = fabs(planes[i][3] - planes[j][3]);
                max_dist_to_ground = std::min(max_dist_to_ground, dist);
            }
        }
    }
    printf("max_dist_to_ground: %f\n", max_dist_to_ground);

    std::vector<std::vector<int> > plane_vtxs, nonplane_vtxs, roof_vtxs;
    plane_vtxs.resize(planes.size());
    nonplane_vtxs.resize(planes.size());
    roof_vtxs.resize(planes.size());
    for (i = 0; i < planes.size(); ++i) {
        plane_vtxs.at(i).reserve(num_vert);
        nonplane_vtxs.at(i).reserve(num_vert);
        roof_vtxs.at(i).reserve(num_vert);
    }

    for (i = 0; i < num_vert; ++i) {
        const Eigen::Vector3d & X = mesh.vertices_.at(i);
        const uint8_t & label = mesh.vertex_labels_.at(i);

        int layer_idx = -1;
        if (planes.size() == 1) {
            layer_idx = 0;
        } else if (label == LABLE_GROUND) {
            int best_layer_id = -1;
            double min_dist_to_plane = FLT_MAX;
            for (j = 0; j < planes.size(); ++j) {
                const auto & plane = planes.at(j);
                double dist = std::abs(plane.dot(X.homogeneous()));
                if (min_dist_to_plane > dist) {
                    min_dist_to_plane = dist;
                    best_layer_id = j;
                }
            }
            if (best_layer_id != -1 && 
                min_dist_to_plane < max_dist_to_ground) {
                layer_idx = best_layer_id;
            }
        } else {
            for (j = 0; j < planes.size(); ++j) {
                if (j + 1 < planes.size()) {
                    auto plane0 = planes.at(j);
                    auto plane1 = planes.at(j + 1);
                    double dist0 = plane0.dot(X.homogeneous());
                    double dist1 = plane1.dot(X.homogeneous());
                    if (dist0 >= 0 && dist1 < 0) {
                        layer_idx = j;
                        break;
                    }
                } else {
                    auto plane = planes.at(j);
                    double dist = plane.dot(X.homogeneous());
                    if (dist >= 0) {
                        layer_idx = j;
                        break;
                    }
                }
            }
        }
        if (layer_idx != -1) {
            if (label == LABLE_GROUND) {
                plane_vtxs.at(layer_idx).emplace_back(i);
            } else if (label == LABLE_CEILING) {
                roof_vtxs.at(layer_idx).emplace_back(i);
            } else {
                nonplane_vtxs.at(layer_idx).emplace_back(i);
            }
        }
    }

    plane_meshes.resize(planes.size());
    nonplane_meshes.resize(planes.size());
    roof_meshes.resize(planes.size());
    for (i = 0; i < planes.size(); ++i) {
        ExtractSubMesh(mesh, plane_vtxs.at(i), plane_meshes.at(i));
        ExtractSubMesh(mesh, nonplane_vtxs.at(i), nonplane_meshes.at(i));
        ExtractSubMesh(mesh, roof_vtxs.at(i), roof_meshes.at(i));
    }
}

Eigen::Matrix3x4d ComputeProjectionMatrix(
    std::vector<PlanePoint>& nonplane_points,
    std::vector<PlanePoint>& plane_points,
    const int max_grid_res,
    const int proj_dir = (X_AXIS | Y_AXIS)) {
    if (nonplane_points.size() == 0 || max_grid_res <= 0) {
        return Eigen::Matrix3x4d::Identity();
    }

    Eigen::Vector4d bbox;
    if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
        bbox[0] = bbox[2] = nonplane_points.at(0).X.x();
        bbox[1] = bbox[3] = nonplane_points.at(0).X.y();
    } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
        bbox[0] = bbox[2] = nonplane_points.at(0).X.x();
        bbox[1] = bbox[3] = nonplane_points.at(0).X.z();
    } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
        bbox[0] = bbox[2] = nonplane_points.at(0).X.y();
        bbox[1] = bbox[3] = nonplane_points.at(0).X.z();
    }
    for (auto point : nonplane_points) {
        if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
            bbox[0] = std::min(point.X.x(), bbox[0]);
            bbox[1] = std::min(point.X.y(), bbox[1]);
            bbox[2] = std::max(point.X.x(), bbox[2]);
            bbox[3] = std::max(point.X.y(), bbox[3]);
        } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
            bbox[0] = std::min(point.X.x(), bbox[0]);
            bbox[1] = std::min(point.X.z(), bbox[1]);
            bbox[2] = std::max(point.X.x(), bbox[2]);
            bbox[3] = std::max(point.X.z(), bbox[3]);
        } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
            bbox[0] = std::min(point.X.y(), bbox[0]);
            bbox[1] = std::min(point.X.z(), bbox[1]);
            bbox[2] = std::max(point.X.y(), bbox[2]);
            bbox[3] = std::max(point.X.z(), bbox[3]);
        }
    }
    for (auto point : plane_points) {
        if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
            bbox[0] = std::min(point.X.x(), bbox[0]);
            bbox[1] = std::min(point.X.y(), bbox[1]);
            bbox[2] = std::max(point.X.x(), bbox[2]);
            bbox[3] = std::max(point.X.y(), bbox[3]);
        } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
            bbox[0] = std::min(point.X.x(), bbox[0]);
            bbox[1] = std::min(point.X.z(), bbox[1]);
            bbox[2] = std::max(point.X.x(), bbox[2]);
            bbox[3] = std::max(point.X.z(), bbox[3]);
        } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
            bbox[0] = std::min(point.X.y(), bbox[0]);
            bbox[1] = std::min(point.X.z(), bbox[1]);
            bbox[2] = std::max(point.X.y(), bbox[2]);
            bbox[3] = std::max(point.X.z(), bbox[3]);
        }
    }

    int grid_res_x, grid_res_y;
    const double scale = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]);
    if (scale > 1) {
        grid_res_y = (max_grid_res - 1);
        grid_res_x = (max_grid_res - 1) / scale;
    } else {
        grid_res_y = (max_grid_res - 1) * scale;
        grid_res_x = (max_grid_res - 1);
    }

    int xpadding = 0, ypadding = 0;
    if (grid_res_x > grid_res_y) {
        ypadding = (grid_res_x - grid_res_y) / 2;
    } else {
        xpadding = (grid_res_y - grid_res_x) / 2;
    }

    const double gridX_size = (bbox[2] - bbox[0]) / grid_res_x;
    const double gridY_size = (bbox[3] - bbox[1]) / grid_res_y;
    
    const double gxy = 1.0 / std::max(gridX_size, gridY_size);

    Eigen::Matrix3x4d M;
    if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
        M(0, 0) = gxy; M(0, 1) = M(0, 2) = 0.0; M(0, 3) = -bbox[0] * gxy + xpadding;
        M(1, 1) = gxy; M(1, 0) = M(1, 2) = 0.0; M(1, 3) = -bbox[1] * gxy + ypadding;
        M(2, 2) = gxy; M(2, 0) = M(2, 1) = 0.0; M(2, 3) = 0.0;
    } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
        M(0, 0) = gxy; M(0, 1) = M(0, 2) = 0.0; M(0, 3) = -bbox[0] * gxy + xpadding;
        M(1, 1) = gxy; M(1, 0) = M(1, 2) = 0.0; M(1, 3) = 0.0;
        M(2, 2) = gxy; M(2, 0) = M(2, 1) = 0.0; M(2, 3) = -bbox[1] * gxy + ypadding;
    } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
        M(0, 0) = gxy; M(0, 1) = M(0, 2) = 0.0; M(0, 3) = 0.0;
        M(1, 1) = gxy; M(1, 0) = M(1, 2) = 0.0; M(1, 3) = -bbox[0] * gxy + xpadding;
        M(2, 2) = gxy; M(2, 0) = M(2, 1) = 0.0; M(2, 3) = -bbox[1] * gxy + ypadding;
    }
    return M;
}

void GenerateMeshMap(const CadMapOptions& options,
                     const std::string dense_reconstruction_path,
                     const TriangleMesh mesh,
                     const Eigen::Matrix3x4d& T,
                     const Eigen::Matrix3x4d& M,
                     const Eigen::Vector4d plane,
                     const int layer_id){

    const int max_grid_res = options.max_grid_size;
    
    std::string mesh_map_path = StringPrintf("%s/mesh_map%04d.jpg",
        dense_reconstruction_path.c_str(), layer_id);
    std::string mesh_normal_path = StringPrintf("%s/mesh_normal%04d.xml",
        dense_reconstruction_path.c_str(), layer_id);
    
    cv::Mat mesh_map, mesh_normal;

    cv::Size grid_size(max_grid_res, max_grid_res);
    mesh_map = cv::Mat::zeros(grid_size, CV_8UC1);
    if(options.output_mesh_normal){
        mesh_normal = cv::Mat::zeros(grid_size, CV_32FC3);
    }

    Eigen::Vector3d p_normal = plane.head<3>();

    for(size_t i = 0; i < mesh.faces_.size(); ++i){
        const Eigen::Vector3i& facet = mesh.faces_[i];
        const Eigen::Vector3d p0 = mesh.vertices_[facet[0]];
        const Eigen::Vector3d p1 = mesh.vertices_[facet[1]];
        const Eigen::Vector3d p2 = mesh.vertices_[facet[2]];

        Eigen::Vector3d normal = mesh.face_normals_[i].normalized();

        if(!mesh.vertex_labels_.empty() && 
        (mesh.vertex_labels_[facet[0]] == LABLE_GROUND || mesh.vertex_labels_[facet[0]] == LABLE_CEILING || 
        mesh.vertex_labels_[facet[1]] == LABLE_GROUND || mesh.vertex_labels_[facet[1]] == LABLE_CEILING || 
        mesh.vertex_labels_[facet[2]] == LABLE_GROUND || mesh.vertex_labels_[facet[2]] == LABLE_CEILING)){
            continue;
        };

        if(fabs(p_normal.normalized().dot(normal.normalized())) > 0.5)
            continue;

        normal = T * normal.homogeneous();
        
        Eigen::Vector3d uv[3];
        uv[0] = M * (T * p0.homogeneous()).homogeneous();
        uv[1] = M * (T * p1.homogeneous()).homogeneous();
        uv[2] = M * (T * p2.homogeneous()).homogeneous();

        int u_min = std::min(uv[0][0], std::min(uv[1][0], uv[2][0]));
        int u_max = std::max(uv[0][0], std::max(uv[1][0], uv[2][0]));
        int v_min = std::min(uv[0][1], std::min(uv[1][1], uv[2][1]));
        int v_max = std::max(uv[0][1], std::max(uv[1][1], uv[2][1]));

        u_min = std::max(0, u_min);
        v_min = std::max(0, v_min);
        u_max = std::min(max_grid_res - 1, u_max);
        v_max = std::min(max_grid_res - 1, v_max);

        for (int v = v_min; v <= v_max; ++v) {
            for (int u = u_min; u <= u_max; ++u) {
                int t1 = (u - uv[0][0]) * (v - uv[1][1]) - (u - uv[1][0]) * (v - uv[0][1]);
                int t2 = (u - uv[1][0]) * (v - uv[2][1]) - (u - uv[2][0]) * (v - uv[1][1]);
                int t3 = (u - uv[2][0]) * (v - uv[0][1]) - (u - uv[0][0]) * (v - uv[2][1]);
                if ((t1 >= 0 && t2 >= 0 && t3 >= 0) ||
                    (t1 <= 0 && t2 <= 0 && t3 <= 0)) {

                    mesh_map.at<uint8_t>(v, u) ++;
                    if(options.output_mesh_normal){
                        for(int b = 0; b < 3; b++){
                            mesh_normal.at<cv::Vec3f>(v, u)[b] += normal[b];
                        }
                    }
                }
            }
        }
    }

    if(options.output_mesh_normal){
        for(int r = 0; r < max_grid_res; r++){
            for(int c = 0; c < max_grid_res; c++){
                if(mesh_map.at<uint8_t>(r, c) != 0){
                    mesh_normal.at<cv::Vec3f>(r, c) /= mesh_map.at<uint8_t>(r, c);
                }
            }
        }

        cv::FileStorage fs(mesh_normal_path, cv::FileStorage::WRITE);
        fs << "mesh_normal" << mesh_normal;
        fs.release();
    }

    cv::imwrite(mesh_map_path, mesh_map);
    std::cout << "Generate Mesh Map Done!" << std::endl;
}

void GenerateRoofMap(const CadMapOptions &options,
                    const std::string dense_reconstruction_path,
                    const std::vector<PlanePoint>& roof_points,
                    const Eigen::Matrix3x4d& M,
                    const int layer_id) {

    const int max_grid_res = options.max_grid_size;
    const int point_radius = options.point_size / 2;

    if (roof_points.size() == 0 || max_grid_res <= 0) {
        return;
    }

    auto roofmap_path = StringPrintf("%s/roofmap%04d.png",
        dense_reconstruction_path.c_str(), layer_id);

    cv::Mat roofmap = cv::Mat::zeros(max_grid_res, max_grid_res, CV_8UC4);

    for (auto &point : roof_points) {

        if (point.s_id != LABLE_CEILING)
            continue;

        Eigen::Vector3ub rgb = point.rgb;
        BitmapColor<uint8_t> color;
        color.r = rgb[0];
        color.g = rgb[1];
        color.b = rgb[2];

        Eigen::Vector3d proj = M * point.X.homogeneous();
        int c = proj[0];
        int r = proj[1];
        float h = point.X[2];

        if(c < 0 || c >= max_grid_res || r < 0 || r >= max_grid_res){
            continue;
        }

        int min_r = std::max(r - point_radius, 0);
        int min_c = std::max(c - point_radius, 0);
        int max_r = std::min(r + point_radius, max_grid_res - 1);
        int max_c = std::min(c + point_radius, max_grid_res - 1);
        for (int v = min_r; v <= max_r; ++v) {
            for (int u = min_c; u <= max_c; ++u) {
                roofmap.at<cv::Vec4b>(v, u)[0] = rgb[2];
                roofmap.at<cv::Vec4b>(v, u)[1] = rgb[1];
                roofmap.at<cv::Vec4b>(v, u)[2] = rgb[0];
                roofmap.at<cv::Vec4b>(v, u)[3] = 255;
            }
        }

    }

    cv::imwrite(roofmap_path, roofmap);

}

void GenerateCADMap(const CadMapOptions &options,
                    const std::string dense_reconstruction_path,
                    const std::vector<PlanePoint>& noplane_points,
                    const std::vector<PlanePoint>& plane_points,
                    const Eigen::Matrix3x4d& M,
                    const std::vector<Eigen::Vector3d>& camera_tracks,
                    const int layer_id,
                    const int proj_dir = (X_AXIS | Y_AXIS)) {

    const int max_grid_res = options.max_grid_size;
    const bool output_ground = options.output_ground;
    const bool output_height_range = options.output_height_range;
    const bool output_point_density = options.output_point_density;
    const bool output_point_normal = options.output_point_normal;
    const bool output_camera_track = options.output_camera_track;
    const int camera_track_radius = options.camera_track_radius;
    const int point_radius = options.point_size / 2;

    if (noplane_points.size() == 0 || max_grid_res <= 0) {
        return;
    }

    std::string suffix = StringPrintf("%04d", layer_id);
    if (proj_dir != 3) {
        suffix = StringPrintf("_sideview%04d", layer_id);
    }

    auto cadmap_without_ground_path = StringPrintf("%s/cadmap%s.png",
        dense_reconstruction_path.c_str(), suffix.c_str());
    auto cadmap_with_ground_path = StringPrintf("%s/cadmap_with_ground%s.png", dense_reconstruction_path.c_str(), suffix.c_str());
    auto point_normal_path = StringPrintf("%s/point_normal%s.xml",
        dense_reconstruction_path.c_str(), suffix.c_str());
    auto density_map_path = StringPrintf("%s/point_density%s.jpg",
        dense_reconstruction_path, suffix.c_str());
    auto cadmap_with_camera_track_path = StringPrintf("%s/camera_track%s.png",
        dense_reconstruction_path.c_str(), suffix.c_str());

    cv::Mat point_normal, density_map, height_range;

    if(output_point_normal){
        point_normal = cv::Mat::zeros(max_grid_res, max_grid_res, CV_32FC3);
        density_map = cv::Mat::zeros(max_grid_res, max_grid_res, CV_16UC1);
    }else if (output_point_density){
        density_map = cv::Mat::zeros(max_grid_res, max_grid_res, CV_16UC1);
    }
    if (output_height_range) {
        height_range = cv::Mat(max_grid_res, max_grid_res, CV_32FC2, cv::Scalar(FLT_MAX, FLT_MAX));
    }

    u_int16_t max_density = 0;

    cv::Mat cadmap = cv::Mat::zeros(max_grid_res, max_grid_res, CV_8UC4);
    std::vector<float> depth_buffer(max_grid_res * max_grid_res, FLT_MAX);
    for (auto point : noplane_points) {
        Eigen::Vector3d proj = M * point.X.homogeneous();
        int r, c;
        float h;
        if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
            c = proj[0];
            r = proj[1];
            h = point.X[2];
        } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
            c = proj[0];
            r = proj[2];
            h = point.X[1];
        } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
            c = proj[1];
            r = proj[2];
            h = point.X[0];
        }

        if(c < 0 || c >= max_grid_res || r < 0 || r >= max_grid_res){
            continue;
        }

        int min_r = std::max(r - point_radius, 0);
        int min_c = std::max(c - point_radius, 0);
        int max_r = std::min(r + point_radius, max_grid_res - 1);
        int max_c = std::min(c + point_radius, max_grid_res - 1);
        for (int v = min_r; v <= max_r; ++v) {
            for (int u = min_c; u <= max_c; ++u) {
                if (depth_buffer.at(v * max_grid_res + u) > h) {
                    cadmap.at<cv::Vec4b>(v, u)[0] = point.rgb[2];
                    cadmap.at<cv::Vec4b>(v, u)[1] = point.rgb[1];
                    cadmap.at<cv::Vec4b>(v, u)[2] = point.rgb[0];
                    cadmap.at<cv::Vec4b>(v, u)[3] = 255;
                    depth_buffer.at(v * max_grid_res + u) = h;
                }
            }
        }

        if(output_height_range){
            height_range.at<cv::Vec2f>(r, c)[0] = std::min(height_range.at<cv::Vec2f>(r, c)[0], h);
            height_range.at<cv::Vec2f>(r, c)[1] = std::max(height_range.at<cv::Vec2f>(r, c)[1], h);
        }

        if(output_point_normal){
            Eigen::Vector3d normal = point.normal.normalized();
            for(int b = 0; b < 3; b++){
                point_normal.at<cv::Vec3f>(r, c)[b] += normal[b];
            }
        }
        
        if(output_point_normal || output_point_density){
            density_map.at<u_int16_t>(r, c) ++;
            max_density = std::max(density_map.at<u_int16_t>(r, c), max_density);
        }
    }

    cv::imwrite(cadmap_without_ground_path, cadmap);

    if(output_height_range){
        cv::FileStorage fs1(StringPrintf("%s/height_range%s.xml",
            dense_reconstruction_path.c_str(), suffix.c_str()), cv::FileStorage::WRITE);
        fs1 << "height_range" << height_range;
        fs1.release();
    }

    if(output_ground || output_point_density || output_point_normal || output_camera_track){
        for (auto point : plane_points) {
            Eigen::Vector3d proj = M * point.X.homogeneous();
            int r, c;
            float h;
            if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
                c = proj[0];
                r = proj[1];
                h = point.X[2];
            } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
                c = proj[0];
                r = proj[2];
                h = point.X[1];
            } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
                c = proj[1];
                r = proj[2];
                h = point.X[0];
            }

            if(c < 0 || c >= max_grid_res || r < 0 || r >= max_grid_res){
                continue;
            }

            if(output_ground || output_camera_track){
                int min_r = std::max(r - point_radius, 0);
                int min_c = std::max(c - point_radius, 0);
                int max_r = std::min(r + point_radius, max_grid_res - 1);
                int max_c = std::min(c + point_radius, max_grid_res - 1);
                for (int v = min_r; v <= max_r; ++v) {
                    for (int u = min_c; u <= max_c; ++u) {
                        if (depth_buffer.at(v * max_grid_res + u) > h) {
                            cadmap.at<cv::Vec4b>(v, u)[0] = point.rgb[2];
                            cadmap.at<cv::Vec4b>(v, u)[1] = point.rgb[1];
                            cadmap.at<cv::Vec4b>(v, u)[2] = point.rgb[0];
                            cadmap.at<cv::Vec4b>(v, u)[3] = 255;
                            depth_buffer.at(v * max_grid_res + u) = h;
                        }
                    }
                }
            }

            if(output_point_normal){
                Eigen::Vector3d normal = point.normal.normalized();
                for(int b = 0; b < 3; b++){
                    point_normal.at<cv::Vec3f>(r, c)[b] += normal[b];
                }
            }
            
            if(output_point_normal || output_point_density){
                density_map.at<u_int16_t>(r, c) ++;
                max_density = std::max(density_map.at<u_int16_t>(r, c), max_density);
            }
        }
    }
    if(output_point_normal){
        for(int r = 0; r < max_grid_res; r++){
            for(int c = 0; c < max_grid_res; c++){
                if(density_map.at<u_int16_t>(r, c) != 0){
                    point_normal.at<cv::Vec3f>(r, c) /= density_map.at<u_int16_t>(r, c);
                }
            }
        }
    }

    if(output_point_density){
        density_map.convertTo(density_map, CV_8UC1, 255.f / max_density);
        cv::imwrite(density_map_path, density_map);
    }

    if(output_point_normal){
        cv::FileStorage fs(point_normal_path, cv::FileStorage::WRITE);
        fs << "point_normal" << point_normal;
        fs.release();
    }

    if(output_ground){
        cv::imwrite(cadmap_with_ground_path, cadmap);
    }

    if(output_camera_track){
        cv::Mat camera_image = cv::Mat::zeros(max_grid_res, max_grid_res, CV_8UC1);
        for(auto &track : camera_tracks){
            Eigen::Vector3d proj = M * track.homogeneous();
            int r, c;
            if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
                c = proj[0];
                r = proj[1];
            } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
                c = proj[0];
                r = proj[2];
            } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
                c = proj[1];
                r = proj[2];
            }
            cv::circle(cadmap, cv::Point(c, r), camera_track_radius, cv::Scalar(0, 0, 255), -1);
        }
        cv::imwrite(cadmap_with_camera_track_path, cadmap);
    }
}

void RenderFromModel(MatXf& zBuffer, const TriangleMesh& nonplane_mesh,
                     const TriangleMesh& plane_mesh, const Eigen::Matrix3x4d& M,
                     const int proj_dir = (X_AXIS | Y_AXIS)) {
    const int width = zBuffer.GetWidth();
    const int height = zBuffer.GetHeight();

    std::vector<Eigen::Vector3i> facets;
    facets.reserve(nonplane_mesh.faces_.size() + plane_mesh.faces_.size());
    for (auto & facet : nonplane_mesh.faces_) {
        facets.emplace_back(facet);
    }
    for (auto & facet : plane_mesh.faces_) {
        facets.emplace_back(facet);
    }

    int num_facet = 0;
    for (auto facet : facets) {
        const TriangleMesh& mesh = (num_facet < nonplane_mesh.faces_.size()) ? nonplane_mesh : plane_mesh;
        num_facet++;
        const Eigen::Vector3d& vtx0 = mesh.vertices_.at(facet[2]);
        const Eigen::Vector3d& vtx1 = mesh.vertices_.at(facet[1]);
        const Eigen::Vector3d& vtx2 = mesh.vertices_.at(facet[0]);

        Eigen::Vector3f i0 = (M * vtx0.homogeneous()).cast<float>();
        Eigen::Vector3f i1 = (M * vtx1.homogeneous()).cast<float>();
        Eigen::Vector3f i2 = (M * vtx2.homogeneous()).cast<float>();

        Eigen::Vector3f normal = (i0 - i2).cross(i1 - i2).normalized();
        if (normal[2] < 0) {
            continue;
        }

        float x0, y0, z0, x1, y1, z1, x2, y2, z2;
        int u_min, u_max, v_min, v_max;
        if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
            x0 = i0.x(); y0 = i0.y(); z0 = i0.z();
            x1 = i1.x(); y1 = i1.y(); z1 = i1.z();
            x2 = i2.x(); y2 = i2.y(); z2 = i2.z();
            u_min = std::min(i0[0], std::min(i1[0], i2[0]));
            u_max = std::max(i0[0], std::max(i1[0], i2[0]));
            v_min = std::min(i0[1], std::min(i1[1], i2[1]));
            v_max = std::max(i0[1], std::max(i1[1], i2[1]));
        } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
            x0 = i0.x(); y0 = i0.z(); z0 = i0.y();
            x1 = i1.x(); y1 = i1.z(); z1 = i1.y();
            x2 = i2.x(); y2 = i2.z(); z2 = i2.y();
            u_min = std::min(i0[0], std::min(i1[0], i2[0]));
            u_max = std::max(i0[0], std::max(i1[0], i2[0]));
            v_min = std::min(i0[2], std::min(i1[2], i2[2]));
            v_max = std::max(i0[2], std::max(i1[2], i2[2]));
        } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
            x0 = i0.y(); y0 = i0.z(); z0 = i0.x();
            x1 = i1.y(); y1 = i1.z(); z1 = i1.x();
            x2 = i2.y(); y2 = i2.z(); z2 = i2.x();
            u_min = std::min(i0[1], std::min(i1[1], i2[1]));
            u_max = std::max(i0[1], std::max(i1[1], i2[1]));
            v_min = std::min(i0[2], std::min(i1[2], i2[2]));
            v_max = std::max(i0[2], std::max(i1[2], i2[2]));
        }
        u_min = std::max(0, u_min);
        v_min = std::max(0, v_min);
        u_max = std::min(width - 1, u_max);
        v_max = std::min(height - 1, v_max);

        float norm1 = -(x0 - x1) * (y2 - y1) + (y0 - y1) * (x2 - x1);
        float norm2 = -(x1 - x2) * (y0 - y2) + (y1 - y2) * (x0 - x2);

        Eigen::Vector2d v1(x1 - x0, y1 - y0);
        Eigen::Vector2d v2(x2 - x1, y2 - y1);
        Eigen::Vector2d v3(x0 - x2, y0 - y2);

        for (int v = v_min; v <= v_max; ++v) {
            for (int u = u_min; u <= u_max; ++u) {
                double m1 = v1.x() * (v - y0) - (u - x0) * v1.y();
                double m2 = v2.x() * (v - y1) - (u - x1) * v2.y();
                double m3 = v3.x() * (v - y2) - (u - x2) * v3.y();
                if (m1 >= 0 && m2 >= 0 && m3 >= 0) {
                    float a = (-(u - x1) * (y2 - y1) + (v - y1) * (x2 - x1)) / norm1;
                    float b = (-(u - x2) * (y0 - y2) + (v - y2) * (x0 - x2)) / norm2;

                    float z = a * z0 + b * z1 + (1 - a - b) * z2;
                    if (zBuffer.Get(v, u) > z) {
                        zBuffer.Set(v, u, z);
                    }
                }
            }
        }
    }
}

void GenerateTrueCADMap(const CadMapOptions &options,
                        const std::string dense_reconstruction_path,
                        const std::vector<mvs::Image>& images,
                        const TriangleMesh& nonplane_mesh,
                        const TriangleMesh& plane_mesh,
                        const TriangleMesh& roof_mesh,
                        const std::vector<PlanePoint>& nonplane_points,
                        const std::vector<PlanePoint>& plane_points,
                        const std::vector<PlanePoint>& roof_points,
                        const Eigen::Matrix3x4d& T,
                        const Eigen::Matrix3x4d& M,
                        const std::vector<Eigen::Vector3d>& camera_tracks,
                        const int layer_id,
                        const int proj_dir = (X_AXIS | Y_AXIS)) {
    std::string suffix = StringPrintf("%04d", layer_id);
    if (proj_dir != 3) {
        suffix = StringPrintf("_sideview%04d", layer_id);
    }
    // auto dsm_path = StringPrintf("%s/dsm%s.jpg",
    //     dense_reconstruction_path.c_str(), suffix.c_str());
    auto dom_path = StringPrintf("%s/cadmap%s.png",
        dense_reconstruction_path.c_str(), suffix.c_str());
    auto dom_with_ground_path = StringPrintf("%s/cadmap_with_ground%s.png", 
        dense_reconstruction_path.c_str(), suffix.c_str());
    auto cadmap_with_camera_track_path = StringPrintf("%s/camera_track%s.png",
        dense_reconstruction_path.c_str(), suffix.c_str());

    auto image_paths = JoinPaths(dense_reconstruction_path, IMAGES_DIR);
    auto semantic_paths = JoinPaths(dense_reconstruction_path, SEMANTICS_DIR);

    const int max_grid_res = options.max_grid_size;
    const bool output_ground = options.output_ground;
    const bool output_camera_track = options.output_camera_track;
    const int camera_track_radius = options.camera_track_radius;

    Eigen::Matrix4f hT = Eigen::Matrix4f::Identity();
    hT.block<3, 4>(0, 0) = T.cast<float>();
    Eigen::Matrix4f inv_hT = hT.inverse();

    Eigen::Matrix3x4f MT = M.cast<float>() * hT;
    Eigen::Matrix4f hMT = Eigen::Matrix4f::Identity();
    hMT.block<3, 4>(0, 0) = MT;
    Eigen::Matrix4f inv_hMT = hMT.inverse();

    MatXf zBuffer(max_grid_res, max_grid_res, 1);
    zBuffer.Fill(FLT_MAX);

    std::cout << "Render from Model" << std::endl;
    RenderFromModel(zBuffer, nonplane_mesh, plane_mesh, MT.cast<double>(), proj_dir);

    std::vector<PlanePoint> points;
    points.insert(points.end(), nonplane_points.begin(), nonplane_points.end());
    points.insert(points.end(), plane_points.begin(), plane_points.end());

    std::vector<std::vector<size_t> > images_points(images.size());
    for (size_t k = 0; k < points.size(); ++k) {
        PlanePoint& pp = points.at(k);
        Eigen::Vector4f hX = inv_hT * pp.X.cast<float>().homogeneous();
        for (auto vis : pp.viss) {
            images_points.at(vis).push_back(k);
        }
    }

    const int max_capacity = 50;

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    cv::Mat dom;
    std::vector<std::vector<uint8_t> > rs, gs, bs;
    rs.resize(max_grid_res * max_grid_res);
    gs.resize(max_grid_res * max_grid_res);
    bs.resize(max_grid_res * max_grid_res);

    std::vector<Eigen::Vector4i> bboxs(images_points.size());
    auto GenerateBBoxs = [&](int image_idx, const std::vector<size_t>& image_points) {
        const mvs::Image& image = images.at(image_idx);
        Eigen::RowMatrix3x4f P(image.GetP());
        const int image_width = image.GetWidth();
        const int image_height = image.GetHeight();

        bool inside = false;
        int ix_min, iy_min, ix_max, iy_max;
        ix_min = iy_min = std::numeric_limits<int>::max();
        ix_max = iy_max = std::numeric_limits<int>::lowest();

        for (auto point_idx : image_points) {
            PlanePoint& point = points.at(point_idx);
            Eigen::Vector3d proj = M * point.X.homogeneous();
            int x, y;
            if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
                x = proj[0];
                y = proj[1];
            } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
                x = proj[0];
                y = proj[2];
            } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
                x = proj[1];
                y = proj[2];
            }

            if (x >= 0 && x < max_grid_res && y >= 0 && y < max_grid_res) {
                inside = true;
                ix_min = std::min(ix_min, x);
                iy_min = std::min(iy_min, y);
                ix_max = std::max(ix_max, x);
                iy_max = std::max(iy_max, y);
            }
        }
        if (inside) {
            bboxs[image_idx] = Eigen::Vector4i(ix_min, iy_min, ix_max, iy_max);
        } else {
            bboxs[image_idx] = Eigen::Vector4i(0, 0, 0, 0);
        }
    };

    for (size_t k = 0; k < images_points.size(); ++k) {
        thread_pool->AddTask(GenerateBBoxs, k, images_points.at(k));
    }
    thread_pool->Wait();

    auto GenerateDOMSlice = [&](int tid, int width, int height, int num_eff_threads) {
        const int height_slice = (height + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (tid + 1), height);
        for (int y = height_slice * tid; y < height_limit; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (rs[idx].empty()) {
                    dom.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0);
                    continue;
                }
                size_t nth = rs[idx].size() / 2;
                std::vector<uint8_t> & r = rs[idx];
                std::nth_element(r.begin(), r.begin() + nth, r.end());
                std::vector<uint8_t> & g = gs[idx];
                std::nth_element(g.begin(), g.begin() + nth, g.end());
                std::vector<uint8_t> & b = bs[idx];
                std::nth_element(b.begin(), b.begin() + nth, b.end());
                dom.at<cv::Vec4b>(y, x) = cv::Vec4b(b[nth], g[nth], r[nth], 255);
            }
        }
    };

    auto GenerateMapIdx = [&](int tid, int ix_min, int iy_min, int ix_max, int iy_max, MatXf* map_idx, Eigen::Matrix3x4f MT2I) {
        const int height = iy_max - iy_min + 1;
        const int width = ix_max - ix_min + 1;
        const uint64_t slice = height * width;
        const int height_slice = (height + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (tid + 1), height);
        for (int y = height_slice * tid + iy_min; y < height_limit + iy_min; ++y) {
            uint64_t pitch = (y - iy_min) * width;
            for (int x = ix_min; x <= ix_max; ++x) {
                if (rs[y * max_grid_res + x].size() > max_capacity) {
                    continue;
                }
                float d = zBuffer.Get(y, x);
                if (d >= FLT_MAX) {
                    continue;
                }
                Eigen::Vector4f uvw;
                if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
                    uvw = Eigen::Vector4f(x, y, d, 1.0);
                } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
                    uvw = Eigen::Vector4f(x, d, y, 1.0);
                } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
                    uvw = Eigen::Vector4f(d, x, y, 1.0);
                }
                Eigen::Vector3f X = MT2I * uvw;

                float invz = 1.0f / X.z();
                int u = X.x() * invz;
                int v = X.y() * invz;
                const uint64_t idx = pitch + x - ix_min;
                map_idx->Set(idx, u);
                map_idx->Set(slice + idx, v);
                map_idx->Set(slice * 2 + idx, X.z());
            }
        }
    };

    auto PixelColoring = [&](int tid, int image_width, int image_height, 
        int num_eff_threads, Bitmap* image, Bitmap* semantic_map, int* remap_idx) {
        const int height_slice = (image_height + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (tid + 1), image_height);
        const uint64_t image_slice = image_height * image_width;
        for (int y = height_slice * tid; y < height_limit; ++y) {
            uint64_t pitch = y * image_width;
            for (int x = 0; x < image_width; ++x) {
                int u = remap_idx[pitch + x];
                int v = remap_idx[image_slice + pitch + x];
                int idx = v * max_grid_res + u;
                if (u == -1 || v == -1 || rs[idx].size() > max_capacity) {
                    continue;
                }

                if (semantic_map && 
                    semantic_map->GetPixel(x, y).r == LABEL_PEDESTRIAN) {
                    continue;
                }

                BitmapColor<uint8_t> color;
                image->GetPixel(x, y, &color);

                if (rs[idx].empty()) {
                    rs[idx].reserve(max_capacity);
                    gs[idx].reserve(max_capacity);
                    bs[idx].reserve(max_capacity);
                }

                rs[idx].emplace_back(color.r);
                gs[idx].emplace_back(color.g);
                bs[idx].emplace_back(color.b);
            }
        }
    };

    auto SingleFrameColoring = [&](int tid, int ix_min, int iy_min, int ix_max, int iy_max, Bitmap* image, Bitmap* semantic_map, Eigen::Matrix3x4f MT2I) {
        const int height = iy_max - iy_min + 1;
        const int width = ix_max - ix_min + 1;
        const int height_slice = (height + num_eff_threads - 1) / num_eff_threads;
        int height_limit = std::min(height_slice * (tid + 1), height);
        const int image_width = image->Width();
        const int image_height = image->Height();
        for (int y = height_slice * tid + iy_min; y < height_limit + iy_min; ++y) {
            uint64_t pitch = (y - iy_min) * width;
            for (int x = ix_min; x <= ix_max; ++x) {
                uint32_t idx = y * max_grid_res + x;
                if (rs[idx].size() > max_capacity) {
                    continue;
                }
                float d = zBuffer.Get(y, x);
                if (d >= FLT_MAX) {
                    continue;
                }
                Eigen::Vector4f uvw;
                if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
                    uvw = Eigen::Vector4f(x, y, d, 1.0);
                } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
                    uvw = Eigen::Vector4f(x, d, y, 1.0);
                } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
                    uvw = Eigen::Vector4f(d, x, y, 1.0);
                }
                Eigen::Vector3f X = MT2I * uvw;

                float invz = 1.0f / X.z();
                int u = X.x() * invz;
                int v = X.y() * invz;
                if (u < 0 || u >= image_width || v < 0 || v >= image_height) {
                    continue;
                }
                if (semantic_map && 
                    semantic_map->GetPixel(u, v).r == LABEL_PEDESTRIAN) {
                    continue;
                }

                BitmapColor<uint8_t> color;
                image->GetPixel(u, v, &color);

                if (rs[idx].empty()) {
                    rs[idx].reserve(max_capacity);
                    gs[idx].reserve(max_capacity);
                    bs[idx].reserve(max_capacity);
                }

                rs[idx].emplace_back(color.r);
                gs[idx].emplace_back(color.g);
                bs[idx].emplace_back(color.b);
            }
        }
    };

    uint64_t indices_print_step = images_points.size() / 100 + 1;
    for (size_t k = 0; k < images_points.size(); k += 2) {
        if (images_points.at(k).empty() || bboxs[k][0] >= bboxs[k][2]) {
            continue;
        }

        const mvs::Image& image = images.at(k);
        Eigen::RowMatrix3x4f P(image.GetP());
        const int image_width = image.GetWidth();
        const int image_height = image.GetHeight();
        const uint64_t image_slice = image_height * image_width;

        std::string image_path = image.GetPath();

        Bitmap bitmap;
        bitmap.Read(image_path);

        std::string image_name = image_path.substr(image_paths.length() + 1, image_path.length() - image_paths.length() - 1);
        std::string semantic_name = image_name.substr(0, image_name.length() - 3) + "png";
        std::string semantic_path = JoinPaths(semantic_paths, semantic_name);
        std::shared_ptr<Bitmap> semantic_map;
        if (ExistsFile(semantic_path)) {
            semantic_map = std::shared_ptr<Bitmap>(new Bitmap);
            semantic_map->Read(semantic_path, false);
        }

        Eigen::Matrix3x4f MT2I = P * inv_hMT;

        int ix_min = bboxs.at(k)[0];
        int iy_min = bboxs.at(k)[1];
        int ix_max = bboxs.at(k)[2];
        int iy_max = bboxs.at(k)[3];
        if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {

            const int sub_width = ix_max - ix_min + 1;
            const int sub_height = iy_max - iy_min + 1;
            const uint64_t sub_slice = sub_height * sub_width;

            MatXf map_idx(sub_width, sub_height, 3);
            map_idx.Fill(-1);
            for(int tid = 0; tid < num_eff_threads; ++tid) {
                thread_pool->AddTask(GenerateMapIdx, tid, ix_min, iy_min, ix_max, iy_max, &map_idx, MT2I);
            };
            thread_pool->Wait();

            std::vector<float> depth_buffer(image_height * image_width);
            std::fill(depth_buffer.begin(), depth_buffer.end(), FLT_MAX);
            std::vector<int> remap_idx(image_height * image_width * 2);
            std::fill(remap_idx.begin(), remap_idx.end(), -1);
            for (int y = 0; y < sub_height; ++y) {
                uint64_t sub_pitch = y * sub_width;
                for (int x = 0; x < sub_width; ++x) {
                    int idx = sub_pitch + x;
                    int u = map_idx.Get(idx);
                    int v = map_idx.Get(sub_slice + idx);
                    if (u < 0 || u >= image_width || v < 0 || v >= image_height) {
                        continue;
                    }
                    float z = map_idx.Get(sub_slice * 2 + idx);
                    uint64_t pixel_idx = v * image_width + u;
                    if (z > 0 && depth_buffer[pixel_idx] > z) {
                        depth_buffer[pixel_idx] = z;
                        remap_idx[pixel_idx] = x + ix_min;
                        remap_idx[image_slice + pixel_idx] = y + iy_min;
                    }
                }
            }

            for(int tid = 0; tid < num_eff_threads; ++tid) {
                thread_pool->AddTask(PixelColoring, tid, image_width, 
                                    image_height, num_eff_threads, &bitmap, 
                                    semantic_map.get(), remap_idx.data());
            };
            thread_pool->Wait();
        } else {
            for(int tid = 0; tid < num_eff_threads; ++tid) {
                thread_pool->AddTask(SingleFrameColoring, tid, ix_min, iy_min, ix_max, iy_max, &bitmap, semantic_map.get(), MT2I);
            };
            thread_pool->Wait();
        }

        bitmap.Deallocate();

        if (k % indices_print_step == 0) {
            std::cout << std::flush << StringPrintf("\rProcess Image#%d", k);
        }
    }
    std::cout << std::endl;
    
    dom = cv::Mat::zeros(max_grid_res, max_grid_res, CV_8UC4);
    for(int tid = 0; tid < num_eff_threads; ++tid) {
        thread_pool->AddTask(GenerateDOMSlice, tid, max_grid_res, 
                            max_grid_res, num_eff_threads);
    };
    thread_pool->Wait();
    cv::imwrite(dom_path, dom);
    if (output_ground) {
        cv::imwrite(dom_with_ground_path, dom);
    }

    if(output_camera_track){
        for(auto &track : camera_tracks){
            Eigen::Vector3d proj = M * track.homogeneous();
            int r, c;
            if ((proj_dir & X_AXIS) && (proj_dir & Y_AXIS)) {
                c = proj[0];
                r = proj[1];
            } else if ((proj_dir & X_AXIS) && (proj_dir & Z_AXIS)) {
                c = proj[0];
                r = proj[2];
            } else if ((proj_dir & Y_AXIS) && (proj_dir & Z_AXIS)) {
                c = proj[1];
                r = proj[2];
            }
            cv::circle(dom, cv::Point(c, r), camera_track_radius, cv::Scalar(0, 0, 255), -1);
        }
        cv::imwrite(cadmap_with_camera_track_path, dom);
    }
    dom.release();
}

typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef std::pair<Point, Vector> PointVectorPair;
typedef CGAL::Parallel_if_available_tag Concurrency_tag;


bool CalcPointsNormal(const std::vector<Eigen::Vector3d>& in_points, std::vector<Eigen::Vector3d>& normals)
{
    std::vector<PointVectorPair> points;
    int nb_neighbors = 12;
    if(in_points.size() <= nb_neighbors) return false;
    for(auto& p : in_points)
    {
        Point new_p(p[0], p[1], p[2]);
        Vector n(0.0, 0.0, 0.0);
        PointVectorPair pair(new_p, n);
        points.emplace_back(pair);
    }
    
    CGAL::pca_estimate_normals<Concurrency_tag> (points, nb_neighbors,
    CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
        normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));

    for(auto& pair : points)
    {
        auto normal = pair.second;
        Eigen::Vector3d n(normal.x(), normal.y(), normal.z());
        normals.emplace_back(n);
    }
    return true;

}

void ReduceModel(TriangleMesh& mesh) {
    std::vector<Eigen::Vector3d> new_vertices;
    new_vertices.reserve(mesh.vertices_.size());
    std::vector<int8_t> new_vertex_labels;
    new_vertex_labels.reserve(mesh.vertices_.size());
    std::vector<Eigen::Vector3d> new_vertex_colors;
    new_vertex_colors.reserve(mesh.vertices_.size());
    std::vector<Eigen::Vector3i> new_faces;
    new_faces.reserve(mesh.faces_.size());

    std::unordered_map<size_t, size_t> vtx_map_idx;
    std::vector<Eigen::Vector3d>& vertices = mesh.vertices_;
    std::vector<int8_t>& vertex_labels = mesh.vertex_labels_;
    std::vector<Eigen::Vector3d>& vertex_colors = mesh.vertex_colors_;

    size_t num_vertices = 0;
    for (auto facet : mesh.faces_) {
        if (vertex_labels[facet[0]] == LABEL_SKY ||
            vertex_labels[facet[1]] == LABEL_SKY ||
            vertex_labels[facet[2]] == LABEL_SKY) {
            continue;
        }

        if (vtx_map_idx.find(facet[0]) != vtx_map_idx.end()) {
            facet[0] = vtx_map_idx[facet[0]];
        } else {
            new_vertices.emplace_back(vertices.at(facet[0]));
            new_vertex_labels.emplace_back(vertex_labels.at(facet[0]));
            new_vertex_colors.emplace_back(vertex_colors.at(facet[0]));
            vtx_map_idx[facet[0]] = num_vertices++;
            facet[0] = num_vertices - 1;
        }
        if (vtx_map_idx.find(facet[1]) != vtx_map_idx.end()) {
            facet[1] = vtx_map_idx[facet[1]];
        } else {
            new_vertices.emplace_back(vertices.at(facet[1]));
            new_vertex_labels.emplace_back(vertex_labels.at(facet[1]));
            new_vertex_colors.emplace_back(vertex_colors.at(facet[1]));
            vtx_map_idx[facet[1]] = num_vertices++;
            facet[1] = num_vertices - 1;
        }
        if (vtx_map_idx.find(facet[2]) != vtx_map_idx.end()) {
            facet[2] = vtx_map_idx[facet[2]];
        } else {
            new_vertices.emplace_back(vertices.at(facet[2]));
            new_vertex_labels.emplace_back(vertex_labels.at(facet[2]));
            new_vertex_colors.emplace_back(vertex_colors.at(facet[2]));
            vtx_map_idx[facet[2]] = num_vertices++;
            facet[2] = num_vertices - 1;
        }

        new_faces.emplace_back(facet);
    }

    std::cout << StringPrintf("Reducing vertices: %d -> %d\n", vertices.size(), num_vertices);

    mesh.vertex_normals_.clear();
    mesh.vertex_visibilities_.clear();
    mesh.vertex_status_.clear();
    std::swap(mesh.vertices_, new_vertices);
    std::swap(mesh.vertex_labels_, new_vertex_labels);
    std::swap(mesh.vertex_colors_, new_vertex_colors);
    std::swap(mesh.faces_, new_faces);
}

int main(int argc, char *argv[]) {

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    CadMapOptions options;

	options.workspace_path = param.GetArgument("workspace_path", "");
	options.image_type = param.GetArgument("image_type", "perspective");
    options.cadmap_type = param.GetArgument("cadmap_type", "cadmap");
    options.max_grid_size = param.GetArgument("max_cad_grid", 1000);
    options.output_by_gravity = param.GetArgument("cad_output_by_gravity", 0);
    options.output_ground = param.GetArgument("cad_output_ground", 0);
    options.output_height_range = param.GetArgument("cad_output_height_range", 0);
    options.output_point_normal = param.GetArgument("cad_output_point_normal", 0);
    options.output_point_density = param.GetArgument("cad_output_point_density", 0);
    options.output_meshmap = param.GetArgument("cad_output_meshmap", 0);
    options.output_mesh_normal = param.GetArgument("cad_output_mesh_normal", 0);
    options.output_camera_track = param.GetArgument("cad_output_camera_track", 0);
    options.output_roofmap = param.GetArgument("cad_output_roofmap", 0);
    options.output_sideview = param.GetArgument("cad_output_sideview", 0);

    options.camera_track_radius = param.GetArgument("cad_camera_track_radius", 10);
    options.point_size = param.GetArgument("cad_point_size", 1);

    options.target_num_point = param.GetArgument("max_num_point_to_cadmap", -1);

    options.verbose = param.GetArgument("verbose", 0);

    options.min_consistent_facet = param.GetArgument("min_consistent_facet", 1000);
    options.dist_point_to_line = param.GetArgument("dist_point_to_line", 0.f);
    options.angle_diff_thres = param.GetArgument("angle_diff", 20.0f);
    options.dist_ratio_point_to_plane1 = param.GetArgument("dist_ratio_point_to_plane1", 10.0f);
    options.dist_ratio_point_to_plane2 = param.GetArgument("dist_ratio_point_to_plane2", 20.0f);
    options.ratio_singlevalue_xz = param.GetArgument("ratio_singlevalue_xz", 1500.0f);
    options.ratio_singlevalue_yz = param.GetArgument("ratio_singlevalue_yz", 400.0f);
    options.filter_low_ratio = param.GetArgument("cad_filter_low_ratio", 0.01f);
    options.filter_high_ratio = param.GetArgument("cad_filter_high_ratio", 0.9f);
    options.inlier_ratio_plane_points = param.GetArgument("inlier_ratio_plane_points", 0.999f);
    options.nb_neighbors = param.GetArgument("nb_neighbors", 6);
    options.max_spacing_factor = param.GetArgument("max_spacing_factor", 6.0f);

    options.dist_insert = param.GetArgument("dist_insert", 5.0f);
    options.diff_depth = param.GetArgument("diff_depth", 0.01f);

    options.Print();

    std::string semantic_table_path = param.GetArgument("semantic_table_path", "");
    if (ExistsFile(semantic_table_path)) {
        LoadSemanticColorTable(semantic_table_path.c_str());
    }

    int num_reconstruction = 0;
    for (size_t reconstruction_idx = 0; ; reconstruction_idx++) {

        Timer timer;
        timer.Start();
        auto reconstruction_path = JoinPaths(options.workspace_path, std::to_string(reconstruction_idx));
        
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        auto dense_reconstruction_path =
            JoinPaths(reconstruction_path, DENSE_DIR);
        if (!ExistsDir(dense_reconstruction_path)) {
            break;
        }

        PrintHeading2(StringPrintf("Processing Reconstruction#%d", reconstruction_idx));

        auto undistort_image_path = 
            JoinPaths(dense_reconstruction_path, IMAGES_DIR);
        // std::cout << " undistort_image_path : " << undistort_image_path << std::endl;
        std::vector<mvs::Image> images;
        if (options.image_type.compare("perspective") == 0
         || options.image_type.compare("rgbd") == 0) {
            mvs::Workspace::Options workspace_options;
            workspace_options.max_image_size = -1;
            workspace_options.image_as_rgb = false;
            workspace_options.image_path = undistort_image_path;
            workspace_options.workspace_path = dense_reconstruction_path;
            workspace_options.workspace_format = options.image_type;

            mvs::Workspace workspace(workspace_options);
            const mvs::Model& model = workspace.GetModel();
            images = model.images;
        } else if (options.image_type.compare("panorama") == 0) {
            std::vector<image_t> image_ids;
            std::vector<std::string> perspective_image_names;
            std::vector<std::vector<int> > overlapping_images;
            std::vector<std::pair<float, float> > depth_ranges;
            // std::cout << "start to import workspace ..... " << std::endl;
            if (!ImportPanoramaWorkspace(dense_reconstruction_path, 
                perspective_image_names, images, image_ids, overlapping_images, 
                depth_ranges, false)) {
                break;
            }
        }

        // std::cout << "images.size: " << images.size() << std::endl;

        const auto input_path = 
            JoinPaths(dense_reconstruction_path, FUSION_NAME);
        const auto input_vis_path = 
            JoinPaths(dense_reconstruction_path, FUSION_NAME) + ".vis";
        const auto input_sem_path =
            JoinPaths(dense_reconstruction_path, FUSION_NAME) + ".sem";
        auto in_model_path = JoinPaths(dense_reconstruction_path, SEM_MODEL_NAME);

        CHECK(ExistsFile(input_path));
        CHECK(ExistsFile(input_vis_path));
        CHECK(ExistsFile(input_sem_path));
        CHECK(ExistsFile(in_model_path));

        TriangleMesh mesh;
        ReadTriangleMeshObj(in_model_path, mesh, true, true);

        ReduceModel(mesh);

        mesh.ComputeNormals();

        std::vector<Eigen::Vector3d> wall_points, wall_normals;
        ExtractWallPoints(mesh, wall_points, wall_normals);
        
        Eigen::Vector3d plane_orientation = ExtractGroundOrientation(mesh);

        std::vector<Eigen::Vector3d> wall_plane_normals;
        DetermineWallClusters(wall_points, wall_normals, plane_orientation, wall_plane_normals);

        // WriteTriangleMeshObj("./reduce_model.obj", mesh);

        std::cout << "Read points" << std::endl;
        std::vector<PlyPoint> ply_points = ReadPly(input_path);
        std::vector<std::vector<uint32_t> > vis_points;
        std::cout << "Read vis_points" << std::endl;
        ReadPointsVisibility(input_vis_path, vis_points);
        std::cout << "Read sem_points" << std::endl;
        ReadPointsSemantic(input_sem_path, ply_points);

        {
        size_t i, j;
        for (i = 0, j = 0; i < ply_points.size(); ++i) {
            auto & pt = ply_points.at(i);
            if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) ||
                std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z) ||
                std::isnan(pt.nx) || std::isnan(pt.ny) || std::isnan(pt.nz) ||
                std::isinf(pt.nx) || std::isinf(pt.ny) || std::isinf(pt.nz)) {
                continue;
            }
            ply_points.at(j) = ply_points.at(i);
            vis_points.at(j) = vis_points.at(i);
            j = j + 1;
        }
        ply_points.resize(j);
        vis_points.resize(j);
        }

        if (options.target_num_point > 0 && ply_points.size() > options.target_num_point) {
            DownsamplePoints(ply_points, vis_points, options.target_num_point);
            // WriteBinaryPlyPoints(dense_reconstruction_path + "/random-sampled-fused.ply", ply_points);
        }

        std::vector<Eigen::Vector3d> cams;
        for (auto image : images) {
            Eigen::Vector3d C = Eigen::Vector3f(image.GetC()).cast<double>();
            cams.push_back(C);
        }
        std::vector<Eigen::Vector4d> layered_planes;
        Eigen::Matrix3x4d T, M, SM;
        int pivot_idx;

        bool align_file_found = false;
        const auto align_file_path = JoinPaths( dense_reconstruction_path, "align_param.txt");
        if(ExistsFile(align_file_path)){
            if(ParseAlignFile(align_file_path, layered_planes, pivot_idx, T, M, SM)){
                printf("Loaded align file: %s\n", align_file_path.c_str());
                align_file_found = true;
            }
        }


        std::cout << "Finish loading data." << std::endl;

        std::vector<std::vector<Eigen::Vector3d> > inlier_points;
        if(!align_file_found){
            if (options.output_by_gravity) {
                EstimateMultiPlanesByGravity(options, ply_points, 
                                            layered_planes, inlier_points);
            } else if (!FitGround(options, images, ply_points, vis_points, mesh, 
                                  wall_plane_normals, layered_planes, inlier_points)) {
                std::cout << "Find plane from camera trajectoryy" << std::endl;
                double d = 0.0;
                for (auto C : cams) {
                    d = std::max(C.y(), d);
                }

                // Update plane equation according to gravity.
                Eigen::Vector4d plane;
                plane[0] = plane[2] = 0.0;
                plane[1] = -1;
                plane[3] = d;
                layered_planes.push_back(plane);
                inlier_points.resize(layered_planes.size());
                std::cout << "plane parameter: " << plane.transpose() << std::endl;
            }
            std::cout << "Finish fitting plane." << std::endl;
            if (options.verbose) {
                for (size_t k = 0; k < layered_planes.size(); ++k) {
                    FILE *fp = fopen(StringPrintf("%s/plane_points%04d.obj", 
                        dense_reconstruction_path.c_str(), k).c_str(), "w");
                    FILE *fp1 = fopen(StringPrintf("%s/plane_points%04d-1.obj", 
                        dense_reconstruction_path.c_str(), k).c_str(), "w");
                    auto plane = layered_planes.at(k);
                    auto layered_points = inlier_points.at(k); 
                    for (size_t i = 0; i < layered_points.size(); ++i) {
                        auto point = layered_points.at(i);
                        fprintf(fp1, "v %f %f %f\n", point.x(), point.y(), point.z());
                        double dist = plane.dot(point.homogeneous());
                        point = point - plane.head<3>() * dist;
                        fprintf(fp, "v %f %f %f\n", point.x(), point.y(), point.z());
                    }
                    fclose(fp);
                    fclose(fp1);
                }
            }
        }
        
        std::vector<std::vector<PlanePoint> > nonplane_points;
        std::vector<std::vector<PlanePoint> > plane_points;
        std::vector<std::vector<PlanePoint> > roof_points;
        PointCloudPartation(options, ply_points, vis_points, mesh, 
                            layered_planes, nonplane_points, plane_points, roof_points);
        std::cout << "Finish point cloud partition." << std::endl;

        std::vector<TriangleMesh> nonplane_meshes;
        std::vector<TriangleMesh> plane_meshes;
        std::vector<TriangleMesh> roof_meshes;
        MeshPartation(options, layered_planes, mesh, nonplane_meshes, 
                      plane_meshes, roof_meshes);
        std::cout << "Finish mesh partition." << std::endl;

        std::cout << "before nonplane_points num : " << nonplane_points.size() << std::endl;
        FilterPoints(nonplane_points, plane_points, roof_points, layered_planes,
                     options.filter_low_ratio, options.filter_high_ratio);

        // std::vector<Eigen::Vector3d> wall_normals;
        // CalcPointsNormal( wall_mesh.vertices_, wall_normals);

        const int num_planes = layered_planes.size();

        if (options.verbose) {
            for (size_t k = 0; k < num_planes; ++k) {
                FILE *fp = nullptr;
                fp = fopen(StringPrintf("%s/points_layer%04d.obj", 
                    dense_reconstruction_path.c_str(), k).c_str(), "w");
                for (auto & point : plane_points.at(k)) {
                    fprintf(fp, "v %f %f %f %d %d %d\n", point.X[0], point.X[1], point.X[2], (int)point.rgb[0], (int)point.rgb[1], (int)point.rgb[2]);
                }
                for (auto & point : nonplane_points.at(k)) {
                    fprintf(fp, "v %f %f %f %d %d %d\n", point.X[0], point.X[1], point.X[2], (int)point.rgb[0], (int)point.rgb[1], (int)point.rgb[2]);
                }
                fclose(fp);
            }
        }

        // Camera Partition.
        std::vector<std::vector<Eigen::Vector3d> > layered_cams;
        layered_cams.resize(num_planes);
        for (size_t i = 0; i < cams.size(); ++i) {
            auto C = cams.at(i).homogeneous();
            int layer_idx = (num_planes == 1) ? 0 : -1;
            for (size_t j = 0; (j < num_planes && num_planes > 1); ++j) {
                if (j + 1 < num_planes) {
                    auto plane0 = layered_planes.at(j);
                    auto plane1 = layered_planes.at(j + 1);
                    double dist0 = plane0.dot(C);
                    double dist1 = plane1.dot(C);
                    if (dist0 >= 0 && dist1 < 0) {
                        layer_idx = j;
                        break;
                    }
                } else {
                    auto plane = layered_planes.at(j);
                    double dist = plane.dot(C);
                    if (dist >= 0) {
                        layer_idx = j;
                        break;
                    }
                }
            }
            if (layer_idx != -1) {
                layered_cams.at(layer_idx).push_back(cams.at(i));
            }
        }
        
        if(!align_file_found){
            // if (options.output_by_gravity) {
            //     T = ComputePlaneAxisByGravity(layered_planes,
            //                                   nonplane_points, plane_points,
            //                                   pivot_idx);
            // }  else {
                T = ComputePlaneAxis(layered_planes, wall_plane_normals, nonplane_points,
                                     plane_points, layered_cams, pivot_idx);
            // }
        }

        for (int i = 0; i < plane_points.size(); ++i) {
            ConvertToPlaneAxis(T, nonplane_points.at(i), plane_points.at(i),
                                roof_points.at(i), layered_cams.at(i));
        }

        if(!align_file_found){
            std::cout << "Computing projection matrix." << std::endl;
            M = ComputeProjectionMatrix(
                nonplane_points.at(pivot_idx), plane_points.at(pivot_idx), options.max_grid_size, (X_AXIS | Y_AXIS));
            if (options.output_sideview) {
                std::cout << "Computing projection matrix(sideview)." << std::endl;
                SM = ComputeProjectionMatrix(
                    nonplane_points.at(pivot_idx), plane_points.at(pivot_idx), options.max_grid_size, (X_AXIS | Z_AXIS));
            }
        }

        std::cout << "Finish computing projection matrix." << std::endl;

        for (int i = 0; i < num_planes; ++i) {
            if(options.output_roofmap){
                GenerateRoofMap(options, dense_reconstruction_path, 
                        roof_points.at(i), M, i);
                std::cout << "Finish generating roof map." << std::endl;
            }
            if (options.cadmap_type.compare("tdom") == 0) {
                std::cout << "Generate TDOM map." << std::endl;
                GenerateTrueCADMap(options, dense_reconstruction_path, images,
                    nonplane_meshes.at(i), plane_meshes.at(i), roof_meshes.at(i), 
                    nonplane_points.at(i), plane_points.at(i), roof_points.at(i),
                    T, M, layered_cams.at(i), i, (X_AXIS | Y_AXIS));
            } else if (options.cadmap_type.compare("cadmap") == 0) {
                std::cout << "Generate CAD map." << std::endl;
                GenerateCADMap(options, dense_reconstruction_path, 
                            nonplane_points.at(i), plane_points.at(i),
                            M, layered_cams.at(i), i, (X_AXIS | Y_AXIS));
            }
            // if (options.output_sideview) {
            //     std::cout << "Generate TDOM(sideview)." << std::endl;
            //     GenerateTrueCADMap(options, dense_reconstruction_path, images,
            //     nonplane_meshes.at(i), plane_meshes.at(i), roof_meshes.at(i), 
            //     nonplane_points.at(i), plane_points.at(i), roof_points.at(i),
            //     T, SM, layered_cams.at(i), i, (X_AXIS | Z_AXIS));
            // }
            if (options.output_sideview) {
                std::cout << "Generate CAD map(sideview)." << std::endl;
                GenerateCADMap(options, dense_reconstruction_path, 
                            nonplane_points.at(i), plane_points.at(i),
                            SM, layered_cams.at(i), i, (X_AXIS | Z_AXIS));
            }

            if(options.output_meshmap || options.output_mesh_normal) {
                GenerateMeshMap(options, dense_reconstruction_path, mesh, T, M, 
                                layered_planes.at(i), i);
                std::cout << "Finish generating mesh map." << std::endl;
            }            
        }
        // GenerateOccupyGrid(JoinPaths(dense_reconstruction_path, 
        //                         "occmap.jpg"), 
        //                     nonplane_points, cams, M, max_grid_size);

        if (!align_file_found){
            std::cout << "Dumping Alignment File!" << std::endl;
            std::ofstream out_param_file;
            out_param_file.open(JoinPaths(
                    dense_reconstruction_path, "align_param.txt"), std::ios::out);

            out_param_file << StringPrintf("# Number of floor\n");
            out_param_file << num_planes << "\n" << std::endl;

            out_param_file << StringPrintf("# Plane Equation: ax + by + cz + d = 0\n");
            for (int i = 0; i < num_planes; ++i) {
                auto plane = layered_planes.at(i);
                out_param_file << StringPrintf("%f %f %f %f\n", 
                                plane[0], plane[1], plane[2], plane[3]);
            }
            out_param_file << std::endl;

            out_param_file << StringPrintf("# Pivot Plane Index\n");
            out_param_file << pivot_idx << "\n" << std::endl;

            out_param_file << StringPrintf("# Transformation Matrix to plane or gravity coordinate system\n");
            out_param_file << StringPrintf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
                                            T(0, 0), T(0, 1), T(0, 2), T(0, 3),
                                            T(1, 0), T(1, 1), T(1, 2), T(1, 3),
                                            T(2, 0), T(2, 1), T(2, 2), T(2, 3)) 
                        << std::endl;

            out_param_file << StringPrintf("# Projection Matrix: M\n");
            out_param_file << StringPrintf("# [u v w] = M * X\n");
            out_param_file << StringPrintf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
                                            M(0, 0), M(0, 1), M(0, 2), M(0, 3), 
                                            M(1, 0), M(1, 1), M(1, 2), M(1, 3),
                                            M(2, 0), M(2, 1), M(2, 2), M(2, 3)) 
                        << std::endl;
            if (options.output_sideview) {
            out_param_file << StringPrintf("# Projection Matrix(side): SM\n");
            out_param_file << StringPrintf("# [u v w] = SM * X\n");
            out_param_file << StringPrintf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
                                        SM(0, 0), SM(0, 1), SM(0, 2), SM(0, 3), 
                                        SM(1, 0), SM(1, 1), SM(1, 2), SM(1, 3),
                                        SM(2, 0), SM(2, 1), SM(2, 2), SM(2, 3)) 
                        << std::endl;
            }
            out_param_file.close();
        }
        timer.PrintMinutes();
    }

    return 0;
}

