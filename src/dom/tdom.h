//Copyright (c) 2022, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_DOM_TDOM_H_
#define SENSEMAP_DOM_TDOM_H_

#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include <CGAL/Simple_cartesian.h>
// #include <CGAL/Polyhedron_incremental_builder_3.h>
// #include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_triangle_primitive.h>

// #include <geotiffio.h>
// #include <xtiffio.h>
// #include <tiffio.h>
#include <gdal.h>
#include <gdal_priv.h>

#include "mvs/workspace.h"
#include "util/obj.h"
#include "util/ply.h"
#include "util/mat.h"
#include "uni_graph.h"
#include "sparse_table.h"

namespace sensemap {
namespace tdom {
typedef CGAL::Simple_cartesian<double>                       Kernel;
// typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef Kernel::Point_3                                      Point;
typedef Kernel::Vector_3                                     Vector;
typedef Kernel::Segment_3                                    Segment;
typedef Kernel::Ray_3                                        Ray;
typedef Kernel::Triangle_3                                   Triangle;
// typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef std::list<Triangle>::iterator                        Iterator;
typedef CGAL::AABB_triangle_primitive<Kernel, Iterator>      Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive>                 Traits;
typedef CGAL::AABB_tree<Traits>                              Tree;
// typedef boost::optional< Tree::Intersection_and_primitive_id<Segment>::Type > Segment_intersection;
typedef Tree::Primitive_id                                   Primitive_id;

typedef SparseTable<std::uint32_t, std::uint32_t, float> DataCosts;

enum DOMOptimizer{
    DIRECTED = 0,
    MAPMAP   = 1
    // GCO      = 2,
};
struct TDOMOptions {
    DOMOptimizer optimizer = DIRECTED;
    bool color_harmonization = false;
    // int max_resolution_per_block = 25000;
    float resample_factor = 4.0f;
    float gsd = 0.1f;
    float max_oblique_angle = 45.0f;
};

class TDOM {
public:
struct CoordinateConverter {
    float inv_gsd, x_min, y_min, z_min;
    int left, top, sub_width, sub_height;
    inline Eigen::Vector3f operator()(const float x, const float y, const float z) {
        Eigen::Vector3f X;
        X[0] = (x - x_min) * inv_gsd - left;
        X[1] = (y - y_min) * inv_gsd - top;
        X[2] = z - z_min;
        return X;
    }
};

public:
    TDOM(const TDOMOptions &options, const std::string workspace_path);

    void Run();

private:
    void ReadWorkspace();

    void SetUp(int rec_idx);

    void ComputeTransformationToUTM();

    void EstimateGSD();
    void ComputeTDOMDimension();

    void EstimateBlocks(std::vector<Eigen::Vector4i> & blocks, int & num_image_per_block, 
                        const double available_memory);

    void EstimateMemoryConsume(float & est_memory, const std::vector<Eigen::Vector4i> &blocks, 
                               const int num_image_per_block, const bool color_harmonization);

    void RefineDSM(MatXf & zBuffer, const MatXf & dist_maps);
    void RenderFromModel(MatXf & zBuffer, CoordinateConverter& coordinate_converter);
    void ComputeTransformDistance(const MatXf & zBuffer, MatXf & dist_maps);
    void ConvertToPseudoColor(const cv::Mat & fimage, cv::Mat & out_image, const cv::Mat & mask);

    // Compatible with old release version.
    void WriteGeoText(const cv::Mat & dsm, const cv::Mat & mask, std::ofstream &file);

    GDALDataset* CreateGDALDataset(const std::string &filename, const int image_type, const int channel);
    void DumpImageMetaData(GDALDataset *pdataset, const cv::Mat & image);

    void DetectHoles(const MatXf & zBuffer, MatXi & holes, std::unordered_map<int, int> & hole_id_map);
    void FillHoles(const MatXf & zBuffer, MatXi & holes, std::unordered_map<int, int> & hole_id_map);

    void DetectOcclusionForAllImages(const MatXf & zBuffer, const Tree * tree);

    void ColorMapping(const MatXf & zBuffer, const MatXf & upsample_zBuffer, 
                      const std::vector<cv::Mat> &bitmaps, cv::Mat & mosaic);

    void CollectImagePairs(const MatXf & zBuffer, 
                           const std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels,
                           const std::vector<std::vector<uint32_t> > & super_visibilities,
                           std::unordered_map<image_pair_t, float> & image_pairs,
                           std::vector<std::vector<int> > & super_ids_per_image);

    void ColorHarmonization(const MatXf & zBuffer, const std::vector<cv::Mat> & bitmaps,
                            const std::vector<image_t> & unique_image_ids,
                            const std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels,
                            const std::unordered_map<image_pair_t, float> & image_pairs,
                            const std::vector<std::vector<int> > & super_ids_per_image,
                            std::vector<YCrCbFactor> & yrb_factors);

    void BuildSuperPixels(const MatXf & zBuffer, std::vector<uint32_t>& pixel_to_super_idx,
                          std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels,
                          std::vector<std::unordered_set<uint32_t> >& neighbors_per_super,
                          std::vector<std::vector<uint32_t> >& super_visibilities);
    void InitGraphNodes(UniGraph & graph, const std::vector<uint32_t>& pixel_to_super_idx,
                        const std::vector<std::unordered_set<uint32_t> >& neighbors_per_super);
    void CalculateDataCosts(DataCosts & data_costs, const std::vector<float> & image_qualities,
                            const std::vector<cv::Mat> & bitmaps, const MatXf & zBuffer, 
                            const std::vector<std::unordered_set<uint32_t> >& neighbors_per_super,
                            const std::vector<std::vector<uint32_t> >& super_visibilities/*,
                            std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels*/);
    void Optimization(const DataCosts & data_costs, UniGraph & graph, std::unordered_map<uint32_t, int> & cell_labels);

#ifdef WITH_GCO_OPTIMIZER
    void GraphOptimization(const std::vector<float> & image_qualities,
                            const std::vector<cv::Mat> & bitmaps, const MatXf & zBuffer, 
                            const std::unordered_map<image_pair_t, float> & image_pairs,
                            const std::vector<std::vector<int> > & super_ids_per_image,
                            const std::vector<std::unordered_set<uint32_t> >& neighbors_per_super,
                            const std::vector<std::vector<uint32_t> >& super_visibilities,
                            std::unordered_map<uint32_t, std::vector<uint32_t> > & super_idx_to_pixels,
                            std::unordered_map<uint32_t, int> & cell_labels);
#endif
private:
    TDOMOptions options_;
    std::string workspace_path_;

    int num_reconstruction_;
    int select_reconstruction_idx_;

    Eigen::RowMatrix3x4d trans_;
    Eigen::RowMatrix3x4d trans_to_utm_;
    Eigen::RowMatrix4d inv_trans_to_utm_;

    mvs::Model model_;
    std::vector<uint8_t> valid_images_;

    TriangleMesh mesh_;

    std::vector<std::vector<uint32_t> > cell_visibilities_;

    bool sourth_or_north_;
    int zone_no_;
    // pcstype_t utm_zone_;

    int m_track_length_;
    float est_gsd_;
    double x_min_, y_min_, x_max_, y_max_, z_min_, z_max_;
    double utm_x_min_, utm_y_min_, utm_z_min_;
    int width_, height_;
    int upsample_width_, upsample_height_;
    int left_, top_;
    int upsample_left_, upsample_top_;
};
}
}

#endif