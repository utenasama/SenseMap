// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_MVS_RECONSTRUCTION_CLUSTER_H_
#define SENSEMAP_MVS_RECONSTRUCTION_CLUSTER_H_

#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <boost/filesystem/path.hpp>
#include <boost/tuple/tuple.hpp>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "base/common.h"
#include "util/misc.h"

#include "util/threading.h"
#include "util/roi_box.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#define EPSILON std::numeric_limits<float>::epsilon()
#define MAX_INT std::numeric_limits<int>::max()

// #define OLD_FILE_SYATEM

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

namespace sensemap {
namespace mvs {

struct ImageVisibility{
    image_t image_id;
    std::uint64_t num_visib_mappoint;
    std::uint64_t num_all_mappoint;
};

class ReconstructionCluster : public Thread {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct Options {
        int cluster_num = -1;
        float max_ram = -1.0f;       // max ram, X GB(1.0e9 bytes)
        float ram_eff_factor = 0.7;
        int min_pts_per_cluster = 20000;
        float min_cell_size = -1.0f;
        float max_cell_size = -1.0f;
        int max_num_images = -1;
        float max_num_images_factor = 7.0;
        float min_num_images_factor = 1.5;

        float valid_spacing_factor = 2.0;
        float outlier_spacing_factor = 3.0;

        std::string format = "";
        int max_image_size = -1;
        
        // filter redundant images
        float min_common_view = 0.25;
        float max_filter_percent = 0.3;
        float dist_threshold = -1.0f;

        bool SetCellSize(const std::vector<Eigen::Vector3f>& poses);

        // Check the options for validity.
        bool Check() const;

        // Print the options to stdout.
        void Print() const;
    };

    ReconstructionCluster(Options& options, std::string out_workspace_path);

protected:
    ReconstructionCluster::Options options_;
    void Run();

    void InitWrokspace();
    int InitParam(int rect_id);
    std::size_t ClusterRun();
    int SaveClusteredRect();

    void ReadPriorBox(size_t rec_idx);
    void ComputePriorBox(size_t rec_idx);

    std::size_t GridCluster(
        std::vector<int> &cell_cluster_map,
        std::vector<std::size_t> &cell_point_count,
        std::vector<std::unordered_map<image_t, ImageVisibility>> & cell_image_visib,
        std::vector<Box> &cell_bound_box,
        std::vector<std::size_t>& point_cell_map);

    std::size_t CommonViewCluster(
        std::vector<int> &cell_cluster_map,
        const std::vector<std::unordered_map<image_t, ImageVisibility>> cell_image_visib,
        const std::vector<Box> &cell_bound_box,
        const std::vector<std::size_t> &cell_point_count,
        const std::size_t grid_size_x,
        const std::size_t grid_size_y,
        const float valid_spacing);

    int ImageVisibilityInsert(
        std::unordered_map<image_t, ImageVisibility>& imgvis_um1,
        const std::unordered_map<image_t, ImageVisibility>& imgvis_um2);

    int ImageVisibilityFilter(
        const std::unordered_map<image_t, ImageVisibility>& imgvis_um,
        const Options options);

    bool WhetherMerge(const std::vector<Box> &cell_bound_box,
                    const int cell_idx, const int merge_cluster_idx);

    void FilterImages(const std::shared_ptr<Reconstruction> clustered_reconstruction,
                    int& num_fiter_image, const Options options);

    void AddMappoints(const std::shared_ptr<Reconstruction> clustered_reconstruction,
                    int& num_add_mappoint);

    void OutputColorRecons();

    private:
        std::string ori_workspace_path_ = "";
        std::string out_workspace_path_ = "";
        std::string out_rect_path_ = "";
        std::size_t num_points_;
        
        std::vector<int> point_cluster_map_;
        std::vector<Box> cluster_bound_box_;
        std::vector<Eigen::Vector3f> points_;
        std::vector<mappoint_t> point_idx_;
        std::vector<std::unordered_set<image_t>> point_visibility_;
        std::vector<Eigen::Vector3f> poses_;
        std::unordered_map<image_t, uint64_t> images_numpoints_;

        Eigen::Matrix3f pivot_;
        int cluster_num_ = 0;
        int num_out_reconstructions_;
        int num_reconstruction_;

        bool has_box_prior_ = false;
        Box prior_box_;

        std::shared_ptr<Reconstruction> reconstruction_;
};

}  // namespace mvs
}  // namespace sensemap
#endif //SENSEMAP_MVS_RECONSTRUCTION_CLUSTER_H_
