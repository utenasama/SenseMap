#ifndef SENSEMAP_MVS_TRIANGULATION_H_
#define SENSEMAP_MVS_TRIANGULATION_H_

#include "Types.h"
#include "AABB.h"
#include "Plane.h"
#include "Ray.h"
#include "Sphere.h"
#include "List.h"
#include "PointCloud.h"

#include "mvs/image.h"
#include "util/obj.h"
#include "util/ply.h"
#include "util/types.h"
#include "util/threading.h"
#include "mvs/workspace.h"
#include "util/roi_box.h"

#include <unordered_map>
#include <unordered_set>

// fix non-manifold vertices
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>


// Delaunay: mesh reconstruction
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#define DELAUNAY_WEAKSURF

#define DELAUNAY_MAXFLOW_IBFS

// #define DELAUNAY_SAVE_POINTS

namespace sensemap {
namespace mvs {

namespace DELAUNAY {
typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_t;
typedef kernel_t::Point_3 point_t;
typedef kernel_t::Vector_3 vector_t;
typedef kernel_t::Direction_3 direction_t;
typedef kernel_t::Segment_3 segment_t;
typedef kernel_t::Plane_3 plane_t;
typedef kernel_t::Triangle_3 triangle_t;
typedef kernel_t::Ray_3 ray_t;

typedef uint32_t vert_size_t;
typedef uint32_t cell_size_t;

typedef float edge_cap_t;

#ifdef DELAUNAY_WEAKSURF
struct view_info_t;
#endif
struct vert_info_t {
    typedef edge_cap_t Type;
    struct view_t {
        PointCloud::View idxView; // view index
        Type weight; // point's weight
        inline view_t() {}
        inline view_t(PointCloud::View _idxView, Type _weight) : idxView(_idxView), weight(_weight) {}
        inline bool operator <(const view_t& v) const { return idxView < v.idxView; }
        inline operator PointCloud::View() const { return idxView; }
    };
    typedef mvs::cList<view_t,const view_t&,0,4,uint32_t> view_vec_t;
    view_vec_t views; // faces' weight from the cell outwards
    #ifdef DELAUNAY_WEAKSURF
    view_info_t* viewsInfo; // each view caches the two faces from the point towards the camera and the end (used only by the weakly supported surfaces)
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    Eigen::Vector3f color;
    char point_type = -1; // 1: visual, 0: lidar
    std::vector<float > scores;

    inline vert_info_t() : viewsInfo(NULL) {}
    ~vert_info_t();
    void AllocateInfo();
    #else
    inline vert_info_t() {}
    #endif
    void InsertViews(const PointCloud& pc, PointCloud::Index idxPoint, const float wgt_factor);
};

struct cell_info_t {
    typedef edge_cap_t Type;
    Type f[4]; // faces' weight from the cell outwards
    Type s; // cell's weight towards s-source
    Type t; // cell's weight towards t-sink
    inline const Type* ptr() const { return f; }
    inline Type* ptr() { return f; }
};

typedef CGAL::Triangulation_vertex_base_with_info_3<vert_info_t, kernel_t> vertex_base_t;
typedef CGAL::Triangulation_cell_base_with_info_3<cell_size_t, kernel_t> cell_base_t;
typedef CGAL::Triangulation_data_structure_3<vertex_base_t, cell_base_t> triangulation_data_structure_t;
typedef CGAL::Delaunay_triangulation_3<kernel_t, triangulation_data_structure_t, CGAL::Compact_location> delaunay_t;
typedef delaunay_t::Vertex_handle vertex_handle_t;
typedef delaunay_t::Cell_handle cell_handle_t;
typedef delaunay_t::Facet facet_t;
typedef delaunay_t::Edge edge_t;

#ifdef DELAUNAY_WEAKSURF
struct view_info_t {
    cell_handle_t cell2Cam;
    cell_handle_t cell2End;
};
#endif

struct camera_cell_t {
    cell_handle_t cell; // cell containing the camera
    std::vector<facet_t> facets; // all facets on the convex-hull in view of the camera (ordered by importance)
};

struct adjacent_vertex_back_inserter_t {
    const delaunay_t& delaunay;
    const point_t& p;
    vertex_handle_t& v;
    inline adjacent_vertex_back_inserter_t(const delaunay_t& _delaunay, const point_t& _p, vertex_handle_t& _v) : delaunay(_delaunay), p(_p), v(_v) {}
    inline adjacent_vertex_back_inserter_t& operator*() { return *this; }
    inline adjacent_vertex_back_inserter_t& operator++(int) { return *this; }
    inline void operator=(const vertex_handle_t& w) {
        ASSERT(!delaunay.is_infinite(v));
        if (!delaunay.is_infinite(w) && delaunay.geom_traits().compare_distance_3_object()(p, w->point(), v->point()) == CGAL::SMALLER)
            v = w;
    }
};

template <typename TYPE>
inline Eigen::Matrix<TYPE, 3, 1> CGAL2EIGEN(const point_t& p) {
    return Eigen::Matrix<TYPE, 3, 1>((TYPE)p.x(), (TYPE)p.y(), (TYPE)p.z());
}
template <typename TYPE>
inline point_t EIGEN2CGAL(const Eigen::Matrix<TYPE, 3, 1>& p) {
    return point_t((kernel_t::RT)p.x(), (kernel_t::RT)p.y(), (kernel_t::RT)p.z());
}

// Given a facet, compute the plane containing it
Plane getFacetPlane(const facet_t& facet);

// Check if a point (p) is coplanar with a triangle (a, b, c);
// return orientation type
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("no-fma")
#endif
static int orientation(const point_t& a, const point_t& b, const point_t& c, const point_t& p);

#ifdef __GNUC__
#pragma GCC pop_options
#endif

// Given a cell and a camera inside it, if the cell is infinite,
// find all facets on the convex-hull and inside the camera frustum,
// else return all four cell's facets
template <int FacetOrientation>
void fetchCellFacets(const delaunay_t& Tr, const std::vector<facet_t>& hullFacets, const cell_handle_t& cell, const mvs::Image& imageData, std::vector<facet_t>& facets);

// information about an intersection between a segment and a facet
struct intersection_t {
    enum Type {FACET, EDGE, VERTEX};
    cell_handle_t ncell; // cell neighbor to the last intersected facet
    vertex_handle_t v1; // vertex for vertex intersection, 1st edge vertex for edge intersection
    vertex_handle_t v2; // 2nd edge vertex for edge intersection
    facet_t facet; // intersected facet
    Type type; // type of intersection (inside facet, on edge, or vertex)
    REAL dist; // distance from starting point (camera) to this facet
    bool bigger; // are we advancing away or towards the starting point?
    const Ray3 ray; // the ray from starting point into the direction of the end point (point -> camera/end-point)
    inline intersection_t() {}
    inline intersection_t(const Eigen::Vector3f& pt, const Eigen::Vector3f& dir) : dist(-FLT_MAX), bigger(true), ray(pt, dir) {}
};

// Check if a segment (p, q) is coplanar with edges of a triangle (a, b, c):
//  coplanar [in,out] : pointer to the 3 int array of indices of the edges coplanar with pq
// return number of entries in coplanar
int checkEdges(const point_t& a, const point_t& b, const point_t& c, const point_t& p, const point_t& q, int coplanar[3]);

// Check intersection between a facet (f) and a segment (s)
// (derived from CGAL::do_intersect in CGAL/Triangle_3_Segment_3_do_intersect.h)
//  coplanar [out] : pointer to the 3 int array of indices of the edges coplanar with (s)
// return -1 if there is no intersection or
// the number of edges coplanar with the segment (0 = intersection inside the triangle)
int intersect(const triangle_t& t, const segment_t& s, int coplanar[3]);

// Find which facet is intersected by the segment (seg) and return next facets to check:
//  in_facets [in] : vector of facets to check
//  out_facets [out] : vector of facets to check at next step (can be in_facets)
//  out_inter [out] : kind of intersection
// return false if no intersection found and the end of the segment was not reached
bool intersect(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter);

// same as above, but simplified only to find face intersection (otherwise terminate);
// terminate if cell containing the segment endpoint is found or if an infinite cell is encountered
bool intersectFace(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter);

// same as above, but starts from a known vertex and incident cell
bool intersectFace(const delaunay_t& Tr, const segment_t& seg, const vertex_handle_t& v, const cell_handle_t& cell, std::vector<facet_t>& out_facets, intersection_t& inter);

// Given a cell, compute the free-space support for it
edge_cap_t freeSpaceSupport(const delaunay_t& Tr, const std::vector<cell_info_t>& infoCells, const cell_handle_t& cell);

// Fetch the triangle formed by the facet vertices,
// making sure the facet orientation is kept (as in CGAL::Triangulation_3::triangle())
// return the vertex handles of the triangle
struct triangle_vhandles_t {
    vertex_handle_t verts[3];
    triangle_vhandles_t() {}
    triangle_vhandles_t(vertex_handle_t _v0, vertex_handle_t _v1, vertex_handle_t _v2)
        #ifdef _SUPPORT_CPP11
        : verts{_v0,_v1,_v2} {}
        #else
        { verts[0] = _v0; verts[1] = _v1; verts[2] = _v2; }
        #endif
};
triangle_vhandles_t getTriangle(cell_handle_t cell, int i);

// Compute the angle between the plane containing the given facet and the cell's circumscribed sphere
// return cosines of the angle
float computePlaneSphereAngle(const delaunay_t& Tr, const facet_t& facet);

} // namespace DELAUNAY

bool DelaunayInsert(PointCloud& pointcloud, 
                    DELAUNAY::delaunay_t& delaunay,
                    const std::vector<mvs::Image>& images,
                    const bool sampInsert,
                    const float distInsert,
                    const float diffDepth,
                    const float plane_insert_factor,
                    const float plane_score_thred,
                    const float b_resave);

bool DelaunayGraphCut(TriangleMesh& mesh,
                      DELAUNAY::delaunay_t& delaunay,
                      const std::vector<mvs::Image>& images,
                      const bool bUseFreeSpaceSupport=true, 
                      const unsigned nItersFixNonManifold=4,
                      const float kSigma=2.f, const float kQual=1.f, 
                      const float kb=4.f, const float kf=3.f, 
                      const float kRel=0.1f/*max 0.3*/, 
                      const float kAbs=1000.f/*min 500*/, 
                      const float kOutl=400.f/*max 700.f*/,
                      const float kInf=(float)(INT_MAX/8));

bool DelaunayMeshing(TriangleMesh& mesh,
                     PointCloud& pointcloud, 
                     const std::vector<mvs::Image>& images,
                    const bool sampInsert = true,
                     const float distInsert=5, 
                     const float diffDepth=0.01,
                    const float plane_insert_factor = 2.0f,
                     const float plane_score_thred = -1.0,
                     const bool bUseFreeSpaceSupport=true, 
                     const unsigned nItersFixNonManifold=4,
                     const float kSigma=2.f, const float kQual=1.f, 
                     const float kb=4.f, const float kf=3.f, 
                     const float kRel=0.1f/*max 0.3*/, 
                     const float kAbs=1000.f/*min 500*/, 
                     const float kOutl=400.f/*max 700.f*/,
                     const float kInf=(float)(INT_MAX/8));

int PointsCluster(std::vector<std::vector<std::size_t> > &point_cluster_map,
                  std::vector<struct Box> &cluster_bound_box,
                  const std::vector<PlyPoint> &ply_points,
                  const Model& model, const int max_num_points,
                  const float overlap_factor, const MeshBox& roi_meshbox);

bool DelaunaySample(std::vector<PlyPoint>& points,
                    std::vector<float>& points_score,
                    std::vector<std::vector<uint32_t> >& points_vis,
                    std::vector<std::vector<float> >& points_weights,
                    const std::vector<mvs::Image>& images,
                    const float distInsert,
                    const float diffDepth);

void PointsSample(std::vector<PlyPoint>& points,
                    std::vector<float>& points_score,
                    std::vector<std::vector<uint32_t> >& points_vis,
                    std::vector<std::vector<float> >& points_weights,
                    const Model& model,
                    const float distInsert,
                    const float diffDepth);

} // namespace mvs
} // namespace sensemap

#endif