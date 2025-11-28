#include "delaunay_triangulation.h"

#include "util/proc.h"
#include "util/common.h"

#ifdef DELAUNAY_MAXFLOW_IBFS
#include "IBFS/IBFS.h"
template <typename NType, typename VType>
class MaxFlow
{
public:
    // Type-Definitions
    typedef NType node_type;
    typedef VType value_type;
    typedef IBFS::IBFSGraph graph_type;

public:
    MaxFlow(size_t numNodes) {
        graph.initSize((int)numNodes, (int)numNodes*2);
    }

    inline void AddNode(node_type n, value_type source, value_type sink) {
        ASSERT(ISFINITE(source) && source >= 0 && ISFINITE(sink) && sink >= 0);
        graph.addNode((int)n, source, sink);
    }

    inline void AddEdge(node_type n1, node_type n2, value_type capacity, value_type reverseCapacity) {
        ASSERT(ISFINITE(capacity) && capacity >= 0 && ISFINITE(reverseCapacity) && reverseCapacity >= 0);
        graph.addEdge((int)n1, (int)n2, capacity, reverseCapacity);
    }

    value_type ComputeMaxFlow() {
        graph.initGraph();
        return graph.computeMaxFlow();
    }

    inline bool IsNodeOnSrcSide(node_type n) const {
        return graph.isNodeOnSrcSide((int)n);
    }

protected:
    graph_type graph;
};
#else
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
template <typename NType, typename VType>
class MaxFlow
{
public:
    // Type-Definitions
    typedef NType node_type;
    typedef VType value_type;
    typedef boost::vecS out_edge_list_t;
    typedef boost::vecS vertex_list_t;
    typedef boost::adjacency_list_traits<out_edge_list_t, vertex_list_t, boost::directedS> graph_traits;
    typedef typename graph_traits::edge_descriptor edge_descriptor;
    typedef typename graph_traits::vertex_descriptor vertex_descriptor;
    typedef typename graph_traits::vertices_size_type vertex_size_type;
    struct Edge {
        value_type capacity;
        value_type residual;
        edge_descriptor reverse;
    };
    typedef boost::adjacency_list<out_edge_list_t, vertex_list_t, boost::directedS, size_t, Edge> graph_type;
    typedef typename boost::graph_traits<graph_type>::edge_iterator edge_iterator;
    typedef typename boost::graph_traits<graph_type>::out_edge_iterator out_edge_iterator;

public:
    MaxFlow(size_t numNodes) : graph(numNodes+2), S(node_type(numNodes)), T(node_type(numNodes+1)) {}

    void AddNode(node_type n, value_type source, value_type sink) {
        ASSERT(ISFINITE(source) && source >= 0 && ISFINITE(sink) && sink >= 0);
        if (source > 0) {
            edge_descriptor e(boost::add_edge(S, n, graph).first);
            edge_descriptor er(boost::add_edge(n, S, graph).first);
            graph[e].capacity = source;
            graph[e].reverse = er;
            graph[er].reverse = e;
        }
        if (sink > 0) {
            edge_descriptor e(boost::add_edge(n, T, graph).first);
            edge_descriptor er(boost::add_edge(T, n, graph).first);
            graph[e].capacity = sink;
            graph[e].reverse = er;
            graph[er].reverse = e;
        }
    }

    void AddEdge(node_type n1, node_type n2, value_type capacity, value_type reverseCapacity) {
        ASSERT(ISFINITE(capacity) && capacity >= 0 && ISFINITE(reverseCapacity) && reverseCapacity >= 0);
        edge_descriptor e(boost::add_edge(n1, n2, graph).first);
        edge_descriptor er(boost::add_edge(n2, n1, graph).first);
        graph[e].capacity = capacity;
        graph[er].capacity = reverseCapacity;
        graph[e].reverse = er;
        graph[er].reverse = e;
    }

    value_type ComputeMaxFlow() {
        vertex_size_type n_verts(boost::num_vertices(graph));
        color.resize(n_verts);
        std::vector<edge_descriptor> pred(n_verts);
        std::vector<vertex_size_type> dist(n_verts);
        return boost::boykov_kolmogorov_max_flow(graph,
            boost::get(&Edge::capacity, graph),
            boost::get(&Edge::residual, graph),
            boost::get(&Edge::reverse, graph),
            &pred[0],
            &color[0],
            &dist[0],
            boost::get(boost::vertex_index, graph),
            S, T
        );
    }

    inline bool IsNodeOnSrcSide(node_type n) const {
        return (color[n] != boost::white_color);
    }

protected:
    graph_type graph;
    std::vector<boost::default_color_type> color;
    const node_type S;
    const node_type T;
};
#endif
/*----------------------------------------------------------------*/

namespace sensemap {
namespace mvs {

void DELAUNAY::vert_info_t::InsertViews(const PointCloud& pc, PointCloud::Index idxPoint, const float wgt_factor = 1.0f) {

    int num_ori_views = views.GetSize();
    const PointCloud::ViewArr& _views = pc.pointViews[idxPoint];
    ASSERT(!_views.IsEmpty());
    const PointCloud::WeightArr* pweights(pc.pointWeights.IsEmpty() ? NULL : pc.pointWeights.Begin()+idxPoint);
    ASSERT(pweights == NULL || _views.GetSize() == pweights->GetSize());
    FOREACH(i, _views) {
        const PointCloud::View viewID(_views[i]);
        const PointCloud::Weight weight(pweights ? (*pweights)[i] * wgt_factor : PointCloud::Weight(wgt_factor));
        // insert viewID in increasing order
        const uint32_t idx(views.FindFirstEqlGreater(viewID));
        if (idx < views.GetSize() && views[idx] == viewID) {
            // the new view is already in the array
            ASSERT(views.FindFirst(viewID) == idx);
            // update point's weight
            views[idx].weight += weight;
        } else {
            // the new view is not in the array,
            // insert it
            views.InsertAt(idx, view_t(viewID, weight));
            ASSERT(views.IsSorted());
        }
    }

    int num_views = _views.GetSize() + num_ori_views;
    if (!pc.normals.IsEmpty() && normal.isZero()){
        normal = pc.normals[idxPoint];
    }
    // color = pc.colors[idxPoint].cast<float>() * (float)_views.GetSize() / num_views 
    //       + color * (float)num_ori_views / num_views;
    color = pc.colors[idxPoint].cast<float>();
    if (pc.pointTypes.GetSize() > 0) {
        if ( (uint8_t)point_type == 255 ){
            point_type = pc.pointTypes[idxPoint];
        }
    }

    if (!pc.scores.IsEmpty()){
        // score += pc.scores[idxPoint] * _views.GetSize();
        scores.push_back(pc.scores[idxPoint]);
    }
}
#ifdef DELAUNAY_WEAKSURF
DELAUNAY::vert_info_t::~vert_info_t() {
    delete[] viewsInfo;
}
void DELAUNAY::vert_info_t::AllocateInfo() {
    ASSERT(!views.IsEmpty());
    viewsInfo = new view_info_t[views.GetSize()];
    #ifndef _RELEASE
    memset(viewsInfo, 0, sizeof(view_info_t)*views.GetSize());
    #endif
}
#endif
inline Plane DELAUNAY::getFacetPlane(const facet_t& facet)
{
    const point_t& v0(facet.first->vertex((facet.second+1)%4)->point());
    const point_t& v1(facet.first->vertex((facet.second+2)%4)->point());
    const point_t& v2(facet.first->vertex((facet.second+3)%4)->point());
    return Plane(CGAL2EIGEN<REAL>(v0), CGAL2EIGEN<REAL>(v1), CGAL2EIGEN<REAL>(v2));
}
inline int DELAUNAY::orientation(const point_t& a, const point_t& b, const point_t& c, const point_t& p)
{
    #if 0
    return CGAL::orientation(a, b, c, p);
    #else
    // inexact_orientation
    const double& px = a.x(); const double& py = a.y(); const double& pz = a.z();
    const double& qx = b.x(); const double& qy = b.y(); const double& qz = b.z();
    const double& rx = c.x(); const double& ry = c.y(); const double& rz = c.z();
    const double& sx = p.x(); const double& sy = p.y(); const double& sz = p.z();
    #if 1
    const double det(CGAL::determinant(
        qx - px, qy - py, qz - pz,
        rx - px, ry - py, rz - pz,
        sx - px, sy - py, sz - pz));
    const double eps(1e-12);
    #else
    const double pqx(qx - px);
    const double pqy(qy - py);
    const double pqz(qz - pz);
    const double prx(rx - px);
    const double pry(ry - py);
    const double prz(rz - pz);
    const double psx(sx - px);
    const double psy(sy - py);
    const double psz(sz - pz);
    const double det(CGAL::determinant(
        pqx, pqy, pqz,
        prx, pry, prz,
        psx, psy, psz));
    const double max0(MAXF3(ABS(pqx), ABS(pqy), ABS(pqz)));
    const double max1(MAXF3(ABS(prx), ABS(pry), ABS(prz)));
    const double eps(5.1107127829973299e-15 * MAXF(max0, max1));
    #endif
    if (det >  eps) return CGAL::POSITIVE;
    if (det < -eps) return CGAL::NEGATIVE;
    return CGAL::COPLANAR;
    #endif
}
template <int FacetOrientation>
void DELAUNAY::fetchCellFacets(const delaunay_t& Tr, const std::vector<facet_t>& hullFacets, const cell_handle_t& cell, const mvs::Image& imageData, std::vector<facet_t>& facets)
{
    if (!Tr.is_infinite(cell)) {
        // store all 4 facets of the cell
        for (int i=0; i<4; ++i) {
            const facet_t f(cell, i);
            ASSERT(!Tr.is_infinite(f));
            facets.push_back(f);
        }
        return;
    }
    // find all facets on the convex-hull in camera's view
    // create the 4 frustum planes
    ASSERT(facets.empty());
    typedef TFrustum<REAL,4> Frustum;
    Frustum frustum(Eigen::RowMatrix3x4f(imageData.GetP()), imageData.GetWidth(), imageData.GetWidth(), 0, 1);
    // loop over all cells
    const Eigen::Vector3f C(imageData.GetC());
    const point_t ptOrigin(EIGEN2CGAL(C));
    for (const facet_t& face: hullFacets) {
        // add face if visible
        const triangle_t verts(Tr.triangle(face));
        if (orientation(verts[0], verts[1], verts[2], ptOrigin) != FacetOrientation)
            continue;
        AABB3 ab(CGAL2EIGEN<REAL>(verts[0]));
        for (int i=1; i<3; ++i)
            ab.Insert(CGAL2EIGEN<REAL>(verts[i]));
        if (frustum.Classify(ab) == CULLED)
            continue;
        facets.push_back(face);
    }
}
inline int DELAUNAY::checkEdges(const point_t& a, const point_t& b, const point_t& c, const point_t& p, const point_t& q, int coplanar[3])
{
    int nCoplanar(0);
    switch (orientation(p,q,a,b)) {
    case CGAL::POSITIVE: return -1;
    case CGAL::COPLANAR: coplanar[nCoplanar++] = 0;
    }
    switch (orientation(p,q,b,c)) {
    case CGAL::POSITIVE: return -1;
    case CGAL::COPLANAR: coplanar[nCoplanar++] = 1;
    }
    switch (orientation(p,q,c,a)) {
    case CGAL::POSITIVE: return -1;
    case CGAL::COPLANAR: coplanar[nCoplanar++] = 2;
    }
    return nCoplanar;
}
int DELAUNAY::intersect(const triangle_t& t, const segment_t& s, int coplanar[3])
{
    const point_t& a = t.vertex(0);
    const point_t& b = t.vertex(1);
    const point_t& c = t.vertex(2);
    const point_t& p = s.source();
    const point_t& q = s.target();

    switch (orientation(a,b,c,p)) {
    case CGAL::POSITIVE:
        switch (orientation(a,b,c,q)) {
        case CGAL::POSITIVE:
            // the segment lies in the positive open halfspaces defined by the
            // triangle's supporting plane
            return -1;
        case CGAL::COPLANAR:
            // q belongs to the triangle's supporting plane
            // p sees the triangle in counterclockwise order
            return checkEdges(a,b,c,p,q,coplanar);
        case CGAL::NEGATIVE:
            // p sees the triangle in counterclockwise order
            return checkEdges(a,b,c,p,q,coplanar);
        default:
            break;
        }
    case CGAL::NEGATIVE:
        switch (orientation(a,b,c,q)) {
        case CGAL::POSITIVE:
            // q sees the triangle in counterclockwise order
            return checkEdges(a,b,c,q,p,coplanar);
        case CGAL::COPLANAR:
            // q belongs to the triangle's supporting plane
            // p sees the triangle in clockwise order
            return checkEdges(a,b,c,q,p,coplanar);
        case CGAL::NEGATIVE:
            // the segment lies in the negative open halfspaces defined by the
            // triangle's supporting plane
            return -1;
        default:
            break;
        }
    case CGAL::COPLANAR: // p belongs to the triangle's supporting plane
        switch (orientation(a,b,c,q)) {
        case CGAL::POSITIVE:
            // q sees the triangle in counterclockwise order
            return checkEdges(a,b,c,q,p,coplanar);
        case CGAL::COPLANAR:
            // the segment is coplanar with the triangle's supporting plane
            // as we know that it is inside the tetrahedron it intersects the face
            //coplanar[0] = coplanar[1] = coplanar[2] = 3;
            return 3;
        case CGAL::NEGATIVE:
            // q sees the triangle in clockwise order
            return checkEdges(a,b,c,p,q,coplanar);
        default:
            break;
        }
    }
    ASSERT("should not happen" == NULL);
    return -1;
}
bool DELAUNAY::intersect(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
{
    // ASSERT(!in_facets.empty());
    if (in_facets.empty()) {
        return false;
    }
    static const int facet_vertex_order[] = {2,1,3,2,2,3,0,2,0,3,1,0,0,1,2,0};
    int coplanar[3];
    const REAL prevDist(inter.dist);
    for (const facet_t& in_facet: in_facets) {
        // ASSERT(!Tr.is_infinite(in_facet));
        if (Tr.is_infinite(in_facet)) {
            continue;
        }
        const int nb_coplanar(intersect(Tr.triangle(in_facet), seg, coplanar));
        if (nb_coplanar >= 0) {
            // skip this cell if the intersection is not in the desired direction
            const REAL interDist(inter.ray.IntersectsDist(getFacetPlane(in_facet)));
            if ((interDist > prevDist) != inter.bigger)
                continue;
            // vertices of facet i: j = 4 * i, vertices = facet_vertex_order[j,j+1,j+2] negative orientation
            inter.facet = in_facet;
            inter.dist = interDist;
            switch (nb_coplanar) {
            case 0: {
                // face intersection
                inter.type = intersection_t::FACET;
                // now find next facets to be checked as
                // the three faces in the neighbor cell different than the origin face
                out_facets.clear();
                const cell_handle_t nc(inter.facet.first->neighbor(inter.facet.second));
                // ASSERT(!Tr.is_infinite(nc));
                if (Tr.is_infinite(nc)) {
                    continue;
                }
                for (int i=0; i<4; ++i)
                    if (nc->neighbor(i) != inter.facet.first)
                        out_facets.push_back(facet_t(nc, i));
                return true; }
            case 1: {
                // coplanar with 1 edge = intersect edge
                const int j(4 * inter.facet.second);
                const int i1(j + coplanar[0]);
                inter.type = intersection_t::EDGE;
                inter.v1 = inter.facet.first->vertex(facet_vertex_order[i1+0]);
                inter.v2 = inter.facet.first->vertex(facet_vertex_order[i1+1]);
                // now find next facets to be checked as
                // the two faces in this cell opposing this edge
                out_facets.clear();
                const edge_t out_edge(inter.facet.first, facet_vertex_order[i1+0], facet_vertex_order[i1+1]);
                const typename delaunay_t::Cell_circulator efc(Tr.incident_cells(out_edge));
                typename delaunay_t::Cell_circulator ifc(efc);
                do {
                    const cell_handle_t c(ifc);
                    if (c == inter.facet.first) continue;
                    const facet_t f1(c, c->index(inter.v1));
                    if (!Tr.is_infinite(f1))
                        out_facets.push_back(f1);
                    const facet_t f2(c, c->index(inter.v2));
                    if (!Tr.is_infinite(f2))
                        out_facets.push_back(f2);
                } while (++ifc != efc);
                return true; }
            case 2: {
                // coplanar with 2 edges = hit a vertex
                // find vertex index
                const int j(4 * inter.facet.second);
                const int i1(j + coplanar[0]);
                const int i2(j + coplanar[1]);
                int i;
                if (facet_vertex_order[i1] == facet_vertex_order[i2] || facet_vertex_order[i1] == facet_vertex_order[i2+1]) {
                    i = facet_vertex_order[i1];
                } else
                if (facet_vertex_order[i1+1] == facet_vertex_order[i2] || facet_vertex_order[i1+1] == facet_vertex_order[i2+1]) {
                    i = facet_vertex_order[i1+1];
                } else {
                    ASSERT("2 edges intersections without common vertex" == NULL);
                }
                inter.type = intersection_t::VERTEX;
                inter.v1 = inter.facet.first->vertex(i);
                ASSERT(!Tr.is_infinite(inter.v1));
                if (inter.v1->point() == seg.target()) {
                    // target reached
                    out_facets.clear();
                    return false;
                }
                // now find next facets to be checked as
                // the faces in the cells around opposing this common vertex
                out_facets.clear();
                struct cell_back_inserter_t {
                    const delaunay_t& Tr;
                    const vertex_handle_t v;
                    const cell_handle_t current_cell;
                    std::vector<facet_t>& out_facets;
                    inline cell_back_inserter_t(const delaunay_t& _Tr, const intersection_t& inter, std::vector<facet_t>& _out_facets)
                        : Tr(_Tr), v(inter.v1), current_cell(inter.facet.first), out_facets(_out_facets) {}
                    inline cell_back_inserter_t& operator*() { return *this; }
                    inline cell_back_inserter_t& operator++(int) { return *this; }
                    inline void operator=(cell_handle_t c) {
                        if (c == current_cell)
                            return;
                        const facet_t f(c, c->index(v));
                        if (Tr.is_infinite(f))
                            return;
                        out_facets.push_back(f);
                    }
                };
                Tr.finite_incident_cells(inter.v1, cell_back_inserter_t(Tr, inter, out_facets));
                return true; }
            }
            // coplanar with 3 edges = tangent = impossible?
            break;
        }
    }
    // Bad end: no intersection found and we are not at the end of the segment (very rarely, but it happens)!
    out_facets.clear();
    return false;
}
bool DELAUNAY::intersectFace(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
{
    int coplanar[3];
    for (std::vector<facet_t>::const_iterator it=in_facets.cbegin(); it!=in_facets.cend(); ++it) {
        ASSERT(!Tr.is_infinite(*it));
        if (intersect(Tr.triangle(*it), seg, coplanar) == 0) {
            // face intersection
            inter.facet = *it;
            inter.type = intersection_t::FACET;
            // now find next facets to be checked as
            // the three faces in the neighbor cell different than the origin face
            out_facets.clear();
            inter.ncell = inter.facet.first->neighbor(inter.facet.second);
            if (Tr.is_infinite(inter.ncell))
                return false;
            for (int i=0; i<4; ++i)
                if (inter.ncell->neighbor(i) != inter.facet.first)
                    out_facets.push_back(facet_t(inter.ncell, i));
            return true;
        }
    }
    out_facets.clear();
    return false;
}
inline bool DELAUNAY::intersectFace(const delaunay_t& Tr, const segment_t& seg, const vertex_handle_t& v, const cell_handle_t& cell, std::vector<facet_t>& out_facets, intersection_t& inter)
{
    if (cell == cell_handle_t())
        return false;
    if (Tr.is_infinite(cell)) {
        inter.ncell = inter.facet.first = cell;
        return true;
    }
    std::vector<facet_t>& in_facets = out_facets;
    ASSERT(in_facets.empty());
    in_facets.push_back(facet_t(cell, cell->index(v)));
    return intersectFace(Tr, seg, in_facets, out_facets, inter);
}
DELAUNAY::edge_cap_t DELAUNAY::freeSpaceSupport(const delaunay_t& Tr, const std::vector<cell_info_t>& infoCells, const cell_handle_t& cell)
{
    // sum up all 4 incoming weights
    // (corresponding to the 4 facets of the neighbor cells)
    edge_cap_t wf(0);
    for (int i=0; i<4; ++i) {
        const facet_t& mfacet(Tr.mirror_facet(facet_t(cell, i)));
        wf += infoCells[mfacet.first->info()].f[mfacet.second];
    }
    return wf;
}
inline DELAUNAY::triangle_vhandles_t DELAUNAY::getTriangle(cell_handle_t cell, int i)
{
    ASSERT(i >= 0 && i <= 3);
    if ((i&1) == 0)
        return triangle_vhandles_t(
            cell->vertex((i+2)&3),
            cell->vertex((i+1)&3),
            cell->vertex((i+3)&3) );
    return triangle_vhandles_t(
        cell->vertex((i+1)&3),
        cell->vertex((i+2)&3),
        cell->vertex((i+3)&3) );
}
float DELAUNAY::computePlaneSphereAngle(const delaunay_t& Tr, const facet_t& facet)
{
    // compute facet normal
    if (Tr.is_infinite(facet.first))
        return 1.f;
    const triangle_vhandles_t tri(getTriangle(facet.first, facet.second));
    const Eigen::Vector3f v0(CGAL2EIGEN<float>(tri.verts[0]->point()));
    const Eigen::Vector3f v1(CGAL2EIGEN<float>(tri.verts[1]->point()));
    const Eigen::Vector3f v2(CGAL2EIGEN<float>(tri.verts[2]->point()));
    const Eigen::Vector3f fn((v1-v0).cross(v2-v0));
        const float fnLenSq(fn.squaredNorm());
        if (fnLenSq == 0.f)
            return 0.5f;

    // compute the co-tangent to the circumscribed sphere in one of the vertices
    #if CGAL_VERSION_NR < 1041101000
    const Eigen::Vector3f cc(CGAL2EIGEN<float>(facet.first->circumcenter(Tr.geom_traits())));
    #else
    struct Tools {
        static point_t circumcenter(const delaunay_t& Tr, const facet_t& facet) {
            return Tr.geom_traits().construct_circumcenter_3_object()(
                facet.first->vertex(0)->point(),
                facet.first->vertex(1)->point(),
                facet.first->vertex(2)->point(),
                facet.first->vertex(3)->point()
            );
        }
    };
    const Eigen::Vector3f cc(CGAL2EIGEN<float>(Tools::circumcenter(Tr, facet)));
    #endif
    const Eigen::Vector3f ct(cc-v0);
    const float ctLenSq(ct.squaredNorm());
    if (ctLenSq == 0.f)
        return 0.5f;

    // compute the angle between the two vectors
    return CLAMP((fn.dot(ct))/SQRT(fnLenSq*ctLenSq), -1.f, 1.f);
}

static inline uint32_t FindVertex(const Eigen::Vector3i& f, uint32_t v) { 
    for (uint32_t i=0; i<3; ++i) 
        if (f[i] == v) 
            return i; 
    return NO_ID; 
}
static inline int GetVertex(const Eigen::Vector3i& f, uint32_t v) { 
    const uint32_t idx(FindVertex(f, v)); 
    ASSERT(idx != NO_ID); 
    return f[idx]; 
}
static inline int& GetVertex(Eigen::Vector3i& f, uint32_t v) { 
    const uint32_t idx(FindVertex(f, v)); 
    ASSERT(idx != NO_ID); 
    return f[idx]; 
}

#define DEFINE_FACE_VERTS(n) \
    const Eigen::Vector3i& f##n = faces[*itFace++]; \
    const uint32_t idx##n(FindVertex(f##n, v)); \
    const uint32_t v##n##1(f##n[(idx##n+1)%3]); \
    const uint32_t v##n##2(f##n[(idx##n+2)%3])
#define IS_LOOP_FACE3(a, b, c) \
    (v##a##2 == v##b##1 && v##b##2 == v##c##1 && v##c##2 == v##a##1)
#define IS_LINK_FACE3(a, b, c) \
    (v##a##2 == v##b##1 && v##b##2 == v##c##1)
#define DEFINE_FACES4(a, b, c, d, go2) \
    if (IS_LINK_FACE3(a,b,c)) { \
        if (!IS_LINK_FACE3(c,d,a)) \
            goto go2; \
        faces.emplace_back(v##a##1, v##a##2, v##b##2); \
        faces.emplace_back(v##c##1, v##c##2, v##d##2); \
    }
#define DEFINE_REMOVE3(go2) \
    /* check that 3 faces form a loop */ \
    itFace = componentFaces.cbegin(); \
    DEFINE_FACE_VERTS(0); \
    DEFINE_FACE_VERTS(1); \
    DEFINE_FACE_VERTS(2); \
    if (!IS_LOOP_FACE3(0,1,2) && !IS_LOOP_FACE3(0,2,1)) \
        goto go2; \
    /* to find the right vertex order for the new face, */ \
    /* set first two vertices in the order appearing in any of the three existing faces, */ \
    /* and the third as the remaining one */ \
    faces.emplace_back( \
        v01, \
        v02, \
        (v02 == v11 ? v12 : v22) \
    ); \
    /* remove component faces and create a new face from the three component vertices */ \
    for (auto fIdx: componentFaces) \
        removeFaces.insert(fIdx); \
    ++nPyramid3
#define DEFINE_REMOVE4(go2) \
    /* check that 3 faces form a loop */ \
    itFace = componentFaces.cbegin(); \
    DEFINE_FACE_VERTS(0); \
    DEFINE_FACE_VERTS(1); \
    DEFINE_FACE_VERTS(2); \
    DEFINE_FACE_VERTS(3); \
    /* to find the right vertex order for the new faces, */ \
    /* find the link order of the face */ \
    DEFINE_FACES4(0,1,2,3, go2) else \
    DEFINE_FACES4(0,1,3,2, go2) else \
    DEFINE_FACES4(0,2,1,3, go2) else \
    DEFINE_FACES4(0,2,3,1, go2) else \
    DEFINE_FACES4(0,3,1,2, go2) else \
    DEFINE_FACES4(0,3,2,1, go2) else \
        goto go2; \
    /* remove component faces and create two new faces from the four component vertices */ \
    for (auto fIdx: componentFaces) \
        removeFaces.insert(fIdx); \
    ++nPyramid4
#define DEFINE_REMOVE_FACES \
    if (!removeFaces.empty()) { \
        /* remove old faces; */ \
        /* delete them in reverse order since the remove operation is simply replacing the removed item with the last item */ \
        std::vector<uint32_t> orderedRemoveFaces; \
        orderedRemoveFaces.reserve(removeFaces.size()); \
        nRemoveFaces += (unsigned)removeFaces.size(); \
        for (uint32_t fIdx: removeFaces) \
            orderedRemoveFaces.push_back(fIdx); \
        removeFaces.clear(); \
        std::sort(orderedRemoveFaces.begin(), orderedRemoveFaces.end()); \
        std::vector<uint32_t>::const_iterator it(orderedRemoveFaces.cend()); \
        do { \
            faces.erase(faces.begin() + *(--it)); \
        } while (it != orderedRemoveFaces.cbegin()); \
    }

struct VertexInfo {
    typedef uint32_t VIndex;
    typedef uint32_t FIndex;
    typedef boost::property<boost::vertex_index1_t, VIndex> VertexProperty;
    typedef boost::property<boost::edge_index_t, FIndex> EdgeProperty;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperty, EdgeProperty> Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_descriptor Edge;
    typedef boost::property_map<Graph, boost::vertex_index1_t>::type VertexIndex1Map;
    typedef boost::property_map<Graph, boost::edge_index_t>::type EdgeIndexMap;
    typedef boost::graph_traits<Graph>::vertex_iterator VertexIter;
    typedef boost::graph_traits<Graph>::out_edge_iterator EdgeIter;
    typedef std::unordered_map<Vertex, Graph::vertices_size_type> Components;
    typedef boost::associative_property_map<Components> ComponentMap;
    typedef std::unordered_map<VIndex, Vertex> VertexMap;
    typedef std::unordered_set<Vertex> VertexSet;
    struct FilterVertex {
        FilterVertex() {}
        FilterVertex(const VertexSet* _filterVerts) : filterVerts(_filterVerts) {}
        template <typename Vertex>
        bool operator()(const Vertex& v) const {
            return (filterVerts->find(v) == filterVerts->cend());
        }
        const VertexSet* filterVerts;
    };
    struct FilterEdge {
        FilterEdge() {}
        FilterEdge(const Graph* _graph, const VertexSet* _filterVerts) : graph(_graph), filterVerts(_filterVerts) {}
        template <typename Edge>
        bool operator()(const Edge& e) const {
            return (filterVerts->find(boost::source(e,*graph)) == filterVerts->cend() &&
                    filterVerts->find(boost::target(e,*graph)) == filterVerts->cend());
        }
        const Graph* graph;
        const VertexSet* filterVerts;
    };

    VertexMap index2idx; // useful/valid only during graph creation
    Graph graph;
    VertexIndex1Map vertexIndex1;
    EdgeIndexMap edgeIndex;
    Components components;
    VertexSet filterVerts;

    inline VertexInfo() {
        vertexIndex1 = boost::get(boost::vertex_index1, graph);
        edgeIndex = boost::get(boost::edge_index, graph);
    }
    Vertex AddVertex(VIndex v) {
        auto vert(index2idx.insert(std::make_pair(v, Vertex())));
        if (vert.second) {
            vert.first->second = boost::add_vertex(graph);
            vertexIndex1[vert.first->second] = v;
        }
        return vert.first->second;
    }
    void AddEdge(VIndex v0, VIndex v1, FIndex f) {
        boost::add_edge(AddVertex(v0), AddVertex(v1), f, graph);
    }
    size_t ComputeComponents() {
        components.clear();
        ComponentMap componentMap(components);
        return boost::connected_components(graph, componentMap);
    }
    size_t ComputeFilteredComponents() {
        ASSERT(!filterVerts.empty());
        FilterEdge filterEdge(&graph, &filterVerts);
        FilterVertex filterVertex(&filterVerts);
        boost::filtered_graph<Graph, FilterEdge, FilterVertex> filterGraph(graph, filterEdge, filterVertex);
        components.clear();
        ComponentMap componentMap(components);
        const size_t nComponents(boost::connected_components(filterGraph, componentMap));
        filterVerts.clear();
        return nComponents;
    }
    void Clear() {
        graph.clear();
        index2idx.clear();
    }
};

void ListIncidenteFaces(const std::vector<Eigen::Vector3d>& vertices,
                        const std::vector<Eigen::Vector3i>& faces,
                        std::vector<std::vector<uint32_t> >& vertexFaces) {
    vertexFaces.clear();
    vertexFaces.resize(vertices.size());
    FOREACH(i, faces) {
        const Eigen::Vector3i& face = faces[i];
        for (int v=0; v<3; ++v) {
            // ASSERT(vertexFaces[face[v]].Find(i) == FaceIdxArr::NO_INDEX);
            vertexFaces[face[v]].push_back(i);
        }
    }
}

// find all non-manifold edges/vertices and for each, duplicate the vertex,
// assigning the new vertex to the smallest connected set of faces;
// return true if problems were found
bool FixNonManifold(TriangleMesh& mesh)
{
    auto& vertices = mesh.vertices_;
    auto& colors = mesh.vertex_colors_;
    auto& vertices_visibilities = mesh.vertex_visibilities_;
    auto& vertex_labels = mesh.vertex_labels_;
    auto& faces = mesh.faces_;
    std::vector<std::vector<uint32_t> > vertexFaces;

    // TD_TIMER_STARTD();
    ASSERT(!vertices.empty() && !faces.empty());
    if (vertexFaces.size() != vertices.size()) {
        ListIncidenteFaces(vertices, faces, vertexFaces);
        // vertexFaces.clear();
        // vertexFaces.resize(vertices.size());
        // FOREACH(i, faces) {
        //     const Eigen::Vector3i& face = faces[i];
        //     for (int v=0; v<3; ++v) {
        //         // ASSERT(vertexFaces[face[v]].Find(i) == FaceIdxArr::NO_INDEX);
        //         vertexFaces[face[v]].push_back(i);
        //     }
        // }
    }
    VertexInfo vertexInfo;
    // IntArr sizes;
    std::vector<int> sizes;
    unsigned nNonManifoldVertices(0), nNonManifoldEdges(0), nRemoveFaces(0), nPyramid3(0), nPyramid4(0);
    std::unordered_set<uint32_t> seenFaces;
    std::unordered_set<uint32_t> removeFaces;
    std::unordered_set<uint32_t> componentFaces;
    std::unordered_set<uint32_t>::const_iterator itFace;
    VertexInfo::EdgeIter ei, eie;
    // fix non-manifold edges
    ASSERT(seenFaces.empty());
    FOREACH(v, vertices) {
        const std::vector<uint32_t>& vFaces = vertexFaces[v];
        if (vFaces.size() < 3)
            continue;
        for(auto pFIdx : vFaces) {
            const Eigen::Vector3i& f(faces[pFIdx]);
            const uint32_t i(FindVertex(f, v));
            vertexInfo.AddEdge(f[(i+1)%3], f[(i+2)%3], pFIdx);
        }
        for (const auto& idx2id: vertexInfo.index2idx) {
            boost::tie(ei, eie) = boost::out_edges(idx2id.second, vertexInfo.graph);
            if (std::distance(ei, eie) >= 4) {
                ASSERT(vertexInfo.filterVerts.empty());
                // do not proceed, if any of the faces was removed
                for(auto pFIdx : vFaces) {
                    if (seenFaces.find(pFIdx) != seenFaces.cend())
                        goto ABORT_EDGE;
                }
                {
                // current vertex and this vertex form the non-manifold edge
                if (vertexInfo.ComputeComponents() > 1) {
                    // filter-out all vertices not belonging to this component
                    const size_t mainComp(vertexInfo.components[idx2id.second]);
                    for (const auto& idx2id: vertexInfo.index2idx) {
                        if (vertexInfo.components[idx2id.second] != mainComp)
                            vertexInfo.filterVerts.insert(idx2id.second);
                    }
                }
                // filter-out this vertex to find the two components to be split
                vertexInfo.filterVerts.insert(idx2id.second);
                const size_t nComponents(vertexInfo.ComputeFilteredComponents());
                if (nComponents < 2)
                    break; // something is wrong, the vertex configuration is not as expected
                // find all vertices in the smallest component
                // sizes.Resize(nComponents);
                // sizes.Memset(0);
                sizes.resize(nComponents, 0);
                for (const auto& comp: vertexInfo.components)
                    ++sizes[comp.second];
                size_t nLongestCompIdx(0);
                for (size_t s=1; s<sizes.size(); ++s) {
                    if (sizes[nLongestCompIdx] < sizes[s])
                        nLongestCompIdx = s;
                }
                FOREACH(s, sizes) {
                    if (s == nLongestCompIdx)
                        continue;
                    ASSERT(componentFaces.empty());
                    // Mesh::Vertex pos(vertices[idx2id.first]);
                    Eigen::Vector3d pos(vertices[idx2id.first]);
                    Eigen::Vector3d col(colors[idx2id.first]);
                    
                    // update semantic label.
                    std::unordered_map<uint8_t, int> m_vertex_labels;
                    m_vertex_labels[vertex_labels[idx2id.first]]++;

                    // update visibility.
                    std::vector<uint32_t> vvis = vertices_visibilities[idx2id.first];
                    std::unordered_set<uint32_t> unique_visibilities;
                    unique_visibilities.insert(vvis.begin(), vvis.end());

                    for (const auto& comp: vertexInfo.components) {
                        if (comp.second == s) {
                            for (boost::tie(ei, eie) = boost::out_edges(comp.first, vertexInfo.graph); ei != eie; ++ei)
                                componentFaces.insert(vertexInfo.edgeIndex[*ei]);
                            pos += vertices[vertexInfo.vertexIndex1[comp.first]];
                            col += colors[vertexInfo.vertexIndex1[comp.first]];

                            m_vertex_labels[vertexInfo.vertexIndex1[comp.first]]++;

                            vvis = vertices_visibilities[vertexInfo.vertexIndex1[comp.first]];
                            unique_visibilities.insert(vvis.begin(), vvis.end());
                        }
                    }
                    const size_t nComponentVertices(sizes[s]+1); // including intersection vertex (this vertex)
                    if (componentFaces.size() != nComponentVertices) {
                        componentFaces.clear();
                        break; // something is wrong, the vertex configuration is not as expected
                    }
                    if (componentFaces.size() == 3/* && vertexInfo.components.size() > 6*/) {
                        // this is the case of a legitimate vertex on the surface and a pyramid rising from the neighboring surface
                        // having the apex the current vertex and the base formed by 3 vertices - one being this vertex;
                         DEFINE_REMOVE3(GENERAL_EDGE);
                    #if 1
                    } else if (componentFaces.size() == 4/* && vertexInfo.components.size() > 8*/) {
                        // this is the case of a legitimate vertex on the surface and a pyramid rising from the neighboring surface
                        // having the apex the current vertex and the base formed by 4 vertices - one being this vertex;
                        DEFINE_REMOVE4(GENERAL_EDGE);
                    #endif
                    } else {
                        GENERAL_EDGE:
                        // simply duplicate the vertex and assign it to the component faces
                        const uint32_t newIndex(vertices.size());
                        for (auto fIdx: componentFaces)
                            GetVertex(faces[fIdx], v) = newIndex;
                        // vertices.Insert(pos / nComponentVertices);
                        vertices.push_back(pos / nComponentVertices);
                        colors.push_back(col / nComponentVertices);
                        
                        int max_samples = 0;
                        int best_label = -1;
                        for(auto vertex_label : m_vertex_labels) {
                            if (vertex_label.second > max_samples) {
                                max_samples = vertex_label.second;
                                best_label = vertex_label.first;
                            }
                        }
                        vertex_labels.push_back(best_label);

                        vvis.clear();
                        vvis.insert(vvis.begin(), unique_visibilities.begin(), unique_visibilities.end());
                        vertices_visibilities.push_back(vvis);
                    }
                    for (auto fIdx: componentFaces)
                        seenFaces.insert(fIdx);
                    componentFaces.clear();
                }
                ++nNonManifoldEdges;
                }
                ABORT_EDGE:
                break;
            }
        }
        vertexInfo.Clear();
    }
    seenFaces.clear();
    DEFINE_REMOVE_FACES;
    // fix non-manifold vertices
    if (nNonManifoldEdges) {
        // ListIncidenteFaces();
        ListIncidenteFaces(vertices, faces, vertexFaces);
    }
    ASSERT(seenFaces.empty());
    FOREACH(v, vertices) {
        // const FaceIdxArr& vFaces = vertexFaces[v];
        const std::vector<uint32_t>& vFaces = vertexFaces[v];
        if (vFaces.size() < 2)
            continue;
        for(auto pFIdx : vFaces) {
            const Eigen::Vector3i& f(faces[pFIdx]);
            const uint32_t i(FindVertex(f, v));
            vertexInfo.AddEdge(f[(i+1)%3], f[(i+2)%3], pFIdx);
        }
        // find all connected sub-graphs
        const size_t nComponents(vertexInfo.ComputeComponents());
        if (nComponents == 1)
            goto ABORT_VERTEX;
        // do not proceed, if any of the faces was removed
        for(auto pFIdx : vFaces) {
            if (seenFaces.find(pFIdx) != seenFaces.cend())
                goto ABORT_VERTEX;
        }
        {
        // there are at least two connected components (usually exactly two);
        // duplicate the vertex and assign the duplicate to the smallest component
        ASSERT(nComponents > 1);
        // sizes.Resize(nComponents);
        // sizes.Memset(0);
        sizes.resize(nComponents, 0);
        for (const auto& comp: vertexInfo.components)
            ++sizes[comp.second];
        size_t nLongestCompIdx(0);
        for (size_t s=1; s<sizes.size(); ++s) {
            if (sizes[nLongestCompIdx] < sizes[s])
                nLongestCompIdx = s;
        }
        FOREACH(s, sizes) {
            if (s == nLongestCompIdx)
                continue;
            ASSERT(componentFaces.empty());
            for (const auto& idx2id: vertexInfo.index2idx) {
                if (vertexInfo.components[idx2id.second] == s) {
                    for (boost::tie(ei, eie) = boost::out_edges(idx2id.second, vertexInfo.graph); ei != eie; ++ei)
                        componentFaces.insert(vertexInfo.edgeIndex[*ei]);
                }
            }
            if (componentFaces.size() == 3 && sizes[s] == 3 && nComponents == 2/* && vFaces.GetSize() > 6*/) {
                // this is the case of a legitimate vertex on the surface and a pyramid rising from the neighboring surface
                // having the apex this vertex and the base formed by 3 vertices;
                DEFINE_REMOVE3(GENERAL_VERTEX);
            #if 1
            } else if (componentFaces.size() == 4 && sizes[s] == 4 && nComponents == 2/* && vFaces.GetSize() > 8*/) {
                // this is the case of a legitimate vertex on the surface and a pyramid rising from the neighboring surface
                // having the apex this vertex and the base formed by 4 vertices;
                DEFINE_REMOVE4(GENERAL_VERTEX);
            #endif
            } else {
                GENERAL_VERTEX:
                // simply duplicate the vertex and assign it to the component faces
                const uint32_t newIndex(vertices.size());
                // Vertex& pos(vertices.AddEmpty());
                // pos = vertices[v];
                Eigen::Vector3d pos = vertices[v];
                vertices.emplace_back(pos);
                Eigen::Vector3d col = colors[v];
                colors.emplace_back(col);
                uint8_t s_id = vertex_labels[v];
                vertex_labels.emplace_back(s_id);
                std::vector<uint32_t> vvis = vertices_visibilities[v];
                vertices_visibilities.emplace_back(vvis);
                for (auto fIdx: componentFaces)
                    GetVertex(faces[fIdx], v) = newIndex;
            }
            for (auto fIdx: componentFaces)
                seenFaces.insert(fIdx);
            componentFaces.clear();
        }
        ++nNonManifoldVertices;
        }
        ABORT_VERTEX:;
        vertexInfo.Clear();
    }
    seenFaces.clear();
    if (nNonManifoldVertices)
        vertexFaces.clear();
    DEFINE_REMOVE_FACES;
    printf("Fixed %u/%u non-manifold edges/vertices and %u faces removed: %u pyramid3 and %u pyramid4\n", 
           nNonManifoldEdges, nNonManifoldVertices, nRemoveFaces, nPyramid3, nPyramid4);
    return (nNonManifoldEdges > 0 || nNonManifoldVertices > 0);
} // FixNonManifold
#undef DEFINE_REMOVE_FACES
#undef DEFINE_REMOVE3
#undef DEFINE_REMOVE4
#undef DEFINE_FACES4
#undef IS_LINK_FACE3
#undef IS_LOOP_FACE3
#undef DEFINE_FACE_VERTS

using namespace DELAUNAY;
bool DelaunayInsert(PointCloud& pointcloud, 
                    delaunay_t& delaunay,
                    const std::vector<mvs::Image>& images,
                    const bool sampInsert,
                    const float distInsert,
                    const float diffDepth,
                    const float plane_insert_factor,
                    const float plane_score_thred,
                    const float b_resave = false){
    // TD_TIMER_STARTD();
    const bool is_score = (!pointcloud.scores.IsEmpty() 
                                          && plane_score_thred < 1.0f 
                                          && plane_score_thred > 1e-6);
    std::cout << "DelaunayInsert " << pointcloud.GetSize() 
        << " points, param: sampInsert, is_score, distInsert, diffDepth, plane_insert_factor, plane_score_thred, b_resave: " 
        << sampInsert << ", " <<  is_score << ", " << distInsert << ", " << diffDepth << ", " << plane_insert_factor 
        << ", " << plane_score_thred << ", " << b_resave << std::endl;

    std::vector<point_t> vertices(pointcloud.points.GetSize());
    std::vector<std::ptrdiff_t> indices(pointcloud.points.GetSize());
    // fetch points
    FOREACH(i, pointcloud.points) {
        const PointCloud::Point& X(pointcloud.points[i]);
        vertices[i] = point_t(X.x(), X.y(), X.z());
        indices[i] = i;
    }
    // sort vertices
    typedef CGAL::Spatial_sort_traits_adapter_3<delaunay_t::Geom_traits, point_t*> Search_traits;
    CGAL::spatial_sort(indices.begin(), indices.end(), Search_traits(&vertices[0], delaunay.geom_traits())); // insert vertices // Util::Progress progress(_T("Points inserted"), indices.size());
    // std::cout << std::endl;
    int progress = 0;
    float distInsertSq(SQUARE(distInsert));
    float depthInsertSq(SQUARE(diffDepth));

    vertex_handle_t hint;
    delaunay_t::Locate_type lt;
    int li, lj;

    float begin_memroy, max_memory, min_memory, end_memory;
    GetAvailableMemory(begin_memroy);

    uint64_t indices_print_step = indices.size() / 10 + 1;
    std::for_each(indices.cbegin(), indices.cend(), [&](size_t idx) {
        float score_factore = 1.0;
        if (is_score){
            const float score = pointcloud.scores[idx];
            if(score > plane_score_thred){
                // score_factore = 1.0f + (score - plane_score_thred) * 2 / (1.0f - plane_score_thred);
                // score_factore = 2.0f + std::cos((1.0f - score) / (1.0f - plane_score_thred) * M_PI);
                float dist_score = std::cos((1.0f - score) / (1.0f - plane_score_thred) * M_PI);
                score_factore = plane_insert_factor * (1.0f  + dist_score) - dist_score ;
                distInsertSq = SQUARE(distInsert * score_factore);
                depthInsertSq = SQUARE(diffDepth * score_factore);
            } else if (score > 0.5){
                score_factore = 1.1;
            }
        }

        const point_t& p = vertices[idx];
        if (std::isnan(p.x()) || std::isinf(p.x()) ||
            std::isnan(p.y()) || std::isinf(p.y()) ||
            std::isnan(p.z()) || std::isinf(p.z())) {
            return;
        }
        const PointCloud::Point& point = pointcloud.points[idx];
        const PointCloud::ViewArr& views = pointcloud.pointViews[idx];
        bool is_insert = true;
        
        // ASSERT(!views.IsEmpty());
        if (views.IsEmpty()) {
            return;
        }
        if (hint == vertex_handle_t()) {
            // this is the first point,
            // insert it
            hint = delaunay.insert(p);
            ASSERT(hint != vertex_handle_t());
        } else
        if (!sampInsert && (!is_score || score_factore < 1.0f + EPSILON)) {
            // insert all points
            hint = delaunay.insert(p, hint);
            ASSERT(hint != vertex_handle_t());
        } else {
            // locate cell containing this point
            const cell_handle_t c(delaunay.locate(p, lt, li, lj, hint->cell()));
            if (lt == delaunay_t::VERTEX) {
                // duplicate point, nothing to insert,
                // just update its visibility info
                hint = c->vertex(li);
                ASSERT(hint != delaunay.infinite_vertex());
            } else {
                // locate the nearest vertex
                vertex_handle_t nearest;
                if (delaunay.dimension() < 3) {
                    // use a brute-force algorithm if dimension < 3
                    delaunay_t::Finite_vertices_iterator vit = delaunay.finite_vertices_begin();
                    nearest = vit;
                    ++vit;
                    adjacent_vertex_back_inserter_t inserter(delaunay, p, nearest);
                    for (delaunay_t::Finite_vertices_iterator end = delaunay.finite_vertices_end(); vit != end; ++vit)
                        inserter = vit;
                } else {
                    // - start with the closest vertex from the located cell
                    // - repeatedly take the nearest of its incident vertices if any
                    // - if not, we're done
                    ASSERT(c != cell_handle_t());
                    nearest = delaunay.nearest_vertex_in_cell(p, c);
                    while (true) {
                        const vertex_handle_t v(nearest);
                        delaunay.adjacent_vertices(nearest, adjacent_vertex_back_inserter_t(delaunay, p, nearest));
                        if (v == nearest)
                            break;
                    }
                }
                ASSERT(nearest == delaunay.nearest_vertex(p, hint->cell()));
                hint = nearest;
                if (is_score){
                    is_insert = false;
                }
                // check if point is far enough to all existing points
                FOREACHPTR(pViewID, views) {
                    const mvs::Image& imageData = images[*pViewID];
                    const Eigen::Map<const Eigen::RowMatrix3f> R(imageData.GetR());
                    const Eigen::Map<const Eigen::Vector3f> T(imageData.GetT());
                    const Eigen::Vector3f pn(R * point + T);
                    const Eigen::Vector3f pe(R * (CGAL2EIGEN<float>(nearest->point())) + T);

                    bool b_insert = false;
                    if (!imageData.IsLidar()){
                        b_insert = (pn.z() - pe.z()) * (pn.z() - pe.z()) / (pn.z() * pn.z()) >= depthInsertSq ||
                                    (((pn / pn.z()).head<2>() - (pe / pe.z()).head<2>()) * 
                                    (imageData.GetK()[0] + imageData.GetK()[4]) * 
                                    0.5).squaredNorm() > distInsertSq;
                    } else {
                        b_insert = (pn - pe).norm() > 0.1 || 
                            (pn - pe).norm() * (pn - pe).norm() / (pn.norm() * pn.norm()) > 0.1 * depthInsertSq ;
                    }
                    if (b_insert){
                        // point far enough to an existing point,
                        // insert as a new point
                        hint = delaunay.insert(p, lt, c, li, lj);
                        ASSERT(hint != vertex_handle_t());
                        is_insert = true;
                        break;
                    }
                }
            }
        }
        // update point visibility info
        if (is_insert){
            // float wgt_factor = score_factore * score_factore;
            float wgt_factor = 1.0f;
            hint->info().InsertViews(pointcloud, idx, wgt_factor);
        }
        
        if (progress % indices_print_step == 0) {
            std::cout<<"\r";
            std::cout<<"Points inserted ["<<progress<<" / "<<indices.size()<<"]"<<std::flush;
        }
        ++progress;
    });
    std::cout << std::endl;

    GetAvailableMemory(max_memory);
    
    // progress.close();
    bool has_scores = !pointcloud.scores.IsEmpty();
    pointcloud.Release();

    GetAvailableMemory(min_memory);

    // Save points
    if (b_resave){
        const size_t num_point(delaunay.number_of_vertices());
        pointcloud.pointViews.Resize(num_point);
        pointcloud.pointWeights.Resize(num_point);
        vert_size_t viID(0);
        for (delaunay_t::All_vertices_iterator vi=delaunay.vertices_begin(), evi=delaunay.vertices_end(); vi!=evi; ++vi ){
            if (delaunay.is_infinite(vi)){
                continue;
            }
            const point_t& p(vi->point());
            const Eigen::Vector3f pt(CGAL2EIGEN<REAL>(p));
            pointcloud.points.Insert(std::move(pt));

            vert_info_t& vert(vi->info());
            if (has_scores){
                std::vector<float > temp_scores(vert.scores);
                std::sort(temp_scores.begin(), temp_scores.end());
                float temp_score = vert.scores.at((float)vert.scores.size() / 2.0f);
                pointcloud.scores.Insert(temp_score);
            }

            for (auto & view : vert.views) {
                pointcloud.pointViews[viID].Insert(view.idxView);
                pointcloud.pointWeights[viID].Insert(view.weight);
            }
            pointcloud.normals.Insert(std::move(vert.normal));
            pointcloud.colors.Insert(std::move(Eigen::Vector3ub(vert.color.x(), 
                                                                vert.color.y(),
                                                                vert.color.z())));
            pointcloud.pointTypes.Insert(vert.point_type);
            ++viID;
        }
    }

    GetAvailableMemory(end_memory);
    // std::cout << "DelaunayInsert begin / end freem-memory :" << begin_memroy 
    //     << " / " << end_memory << "\t max / min memory: " 
    //     << max_memory << " / " << min_memory << std::endl;
    
    return true;
};

bool DelaunayGraphCut(TriangleMesh& mesh, 
                      delaunay_t& delaunay,
                      const std::vector<mvs::Image>& images,
                      const bool bUseFreeSpaceSupport, 
                      const unsigned nItersFixNonManifold,
                      const float kSigma, const float kQual, const float kb,
                      const float kf, const float kRel, const float kAbs, 
                      const float kOutl, const float kInf) {
    float begin_memroy, memory_1, memory_2, memory_3, memory_4, memory_5, memory_6, memory_7, end_memory;
    GetAvailableMemory(begin_memroy);
    std::cout << "free memory: " << begin_memroy << std::endl;

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    // create the Delaunay triangulation
    std::vector<cell_info_t> infoCells;
    std::vector<camera_cell_t> camCells;
    std::vector<facet_t> hullFacets;
    {
        // init cells weights and
        // loop over all cells and store the finite facet of the infinite cells
        const size_t numNodes(delaunay.number_of_cells());
        infoCells.resize(numNodes);
        memset(&infoCells[0], 0, sizeof(cell_info_t)*numNodes);
        cell_size_t ciID(0);
        for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), eci=delaunay.all_cells_end(); ci!=eci; ++ci, ++ciID) {
            ci->info() = ciID;
            // skip the finite cells
            if (!delaunay.is_infinite(ci))
                continue;
            // find the finite face
            for (int f=0; f<4; ++f) {
                const facet_t facet(ci, f);
                if (!delaunay.is_infinite(facet)) {
                    // store face
                    hullFacets.push_back(facet);
                    break;
                }
            }
        }
        // find all cells containing a camera
        camCells.resize(images.size());
#if 0
        FOREACH(i, images) {
            const mvs::Image& imageData = images[i];
            // if (!imageData.IsValid())
            //     continue;
            const Eigen::Vector3f C(imageData.GetC());

            camera_cell_t& camCell = camCells[i];
            camCell.cell = delaunay.locate(EIGEN2CGAL(C));
            ASSERT(camCell.cell != cell_handle_t());
            fetchCellFacets<CGAL::POSITIVE>(delaunay, hullFacets, camCell.cell, imageData, camCell.facets);
            // link all cells contained by the camera to the source
            for (const facet_t& f: camCell.facets)
                infoCells[f.first->info()].s = kInf;
        }
#else
        auto FetchCameraCellFacets = [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                const mvs::Image& imageData = images[i];
                // if (!imageData.IsValid())
                //     continue;
                const Eigen::Vector3f C(imageData.GetC());

                camera_cell_t& camCell = camCells[i];
                camCell.cell = delaunay.locate(EIGEN2CGAL(C));
                ASSERT(camCell.cell != cell_handle_t());
                fetchCellFacets<CGAL::POSITIVE>(delaunay, hullFacets, camCell.cell, imageData, camCell.facets);
                // link all cells contained by the camera to the source
                for (const facet_t& f: camCell.facets)
                    infoCells[f.first->info()].s = kInf;
            }
        };
        const size_t num_slice = (images.size() + num_eff_threads - 1) / num_eff_threads;
        for (int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
            int start = thread_idx * num_slice;
            int end = std::min(num_slice * (thread_idx+ 1), images.size());
            thread_pool->AddTask(FetchCameraCellFacets, start, end);
        }
        thread_pool->Wait();
#endif
        GetAvailableMemory(memory_1);
        // printf("Delaunay tetrahedralization completed: %u points -> %u vertices, %u (+%u) cells, %u (+%u) faces (%s)", indices.size(), delaunay.number_of_vertices(), delaunay.number_of_finite_cells(), delaunay.number_of_cells()-delaunay.number_of_finite_cells(), delaunay.number_of_finite_facets(), delaunay.number_of_facets()-delaunay.number_of_finite_facets(), TD_TIMER_GET_FMT().c_str());
        printf("Delaunay tetrahedralization completed: %u vertices, %u (+%u) cells, %u (+%u) faces (free memory: %f)\n", 
            delaunay.number_of_vertices(), delaunay.number_of_finite_cells(), delaunay.number_of_cells()-delaunay.number_of_finite_cells(), 
            delaunay.number_of_finite_facets(), delaunay.number_of_facets()-delaunay.number_of_finite_facets(), memory_1);
    }

    // for every camera-point ray intersect it with the tetrahedrons and
    // add alpha_vis(point) to cell's directed edge in the graph
    {
        // TD_TIMER_STARTD();

        // estimate the size of the smallest reconstructible object
        FloatArr distsSq(0, delaunay.number_of_edges());
        for (delaunay_t::Finite_edges_iterator ei=delaunay.finite_edges_begin(), eei=delaunay.finite_edges_end(); ei!=eei; ++ei) {
            const cell_handle_t& c(ei->first);
            distsSq.Insert((CGAL2EIGEN<float>(c->vertex(ei->second)->point()) - CGAL2EIGEN<float>(c->vertex(ei->third)->point())).squaredNorm());
        }
        const float sigma(SQRT(distsSq.GetMedian())*kSigma);
        const float inv2SigmaSq(0.5f/(sigma*sigma));
        distsSq.Release();

        std::vector<facet_t> facets;

        size_t num_vert = delaunay.number_of_vertices();
        std::cout<<"Points weighted ["<< num_vert <<"]"<<std::endl;

#if 0
        // compute the weights for each edge
        {
        // TD_TIMER_STARTD();
        // Util::Progress progress(_T("Points weighted"), delaunay.number_of_vertices());
        int progress = 0;
        for (delaunay_t::Vertex_iterator vi=delaunay.vertices_begin(), vie=delaunay.vertices_end(); vi!=vie; ++vi) {
            vert_info_t& vert(vi->info());
            if (vert.views.IsEmpty())
                continue;
            #ifdef DELAUNAY_WEAKSURF
            vert.AllocateInfo();
            #endif
            const point_t& p(vi->point());
            const Eigen::Vector3f pt(CGAL2EIGEN<REAL>(p));
            FOREACH(v, vert.views) {
                const typename vert_info_t::view_t view(vert.views[v]);
                const uint32_t imageID(view.idxView);
                const edge_cap_t alpha_vis(view.weight);
                const mvs::Image& imageData = images[imageID];
                // ASSERT(imageData.IsValid());
                const Eigen::Vector3f C(imageData.GetC());
                const camera_cell_t& camCell = camCells[imageID];
                // compute the ray used to find point intersection
                const Eigen::Vector3f vecCamPoint(pt-C);
                const float norm = vecCamPoint.norm();
                if (norm < 1e-12 || std::isnan(norm) || std::isinf(norm)) {
                    continue;
                }
                const REAL invLenCamPoint(REAL(1)/norm);
                intersection_t inter(pt, Eigen::Vector3f(vecCamPoint*invLenCamPoint));
                // find faces intersected by the camera-point segment
                const segment_t segCamPoint(EIGEN2CGAL(C), p);
                if (!intersect(delaunay, segCamPoint, camCell.facets, facets, inter))
                    continue;
                do {
                    // assign score, weighted by the distance from the point to the intersection
                    const float dist_factor = 
                        EXP(-SQUARE((float)inter.dist)*inv2SigmaSq);
                    const edge_cap_t w(alpha_vis * (1.f - dist_factor));
                    edge_cap_t& f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
                    f += w;

                    // vert_info_t& v0 = inter.facet.first->vertex(0)->info();
                    // vert_info_t& v1 = inter.facet.first->vertex(1)->info();
                    // vert_info_t& v2 = inter.facet.first->vertex(2)->info();
                    // if (v0.point_type && v1.point_type && v2.point_type) {
                    //     // float wl = std::min(v0.views.GetSize(),
                    //     //     std::min(v1.views.GetSize(), v2.views.GetSize()));
                    //     f = f + 16 * (1.f - dist_factor);
                    // }
                } while (intersect(delaunay, segCamPoint, facets, facets, inter));
                // ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
                if (!(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi)) {
                    continue;
                }
                #ifdef DELAUNAY_WEAKSURF
                ASSERT(vert.viewsInfo[v].cell2Cam == NULL);
                vert.viewsInfo[v].cell2Cam = inter.facet.first;
                #endif
                // find faces intersected by the endpoint-point segment
                inter.dist = FLT_MAX; inter.bigger = false;
                const Eigen::Vector3f endPoint(pt+vecCamPoint*(invLenCamPoint*sigma));
                const segment_t segEndPoint(EIGEN2CGAL(endPoint), p);
                const cell_handle_t endCell(delaunay.locate(segEndPoint.source(), vi->cell()));
                ASSERT(endCell != cell_handle_t());
                fetchCellFacets<CGAL::NEGATIVE>(delaunay, hullFacets, endCell, imageData, facets);
                edge_cap_t& t(infoCells[endCell->info()].t);
                // if (endCell->info() == 436) {
                //     std::cout << t << std::endl;
                // }
                t += alpha_vis;
                while (intersect(delaunay, segEndPoint, facets, facets, inter)) {
                    // assign score, weighted by the distance from the point to the intersection
                    const facet_t& mf(delaunay.mirror_facet(inter.facet));
                    const float dist_factor = 
                        EXP(-SQUARE((float)inter.dist)*inv2SigmaSq);
                    const edge_cap_t w(alpha_vis * (1.f - dist_factor));
                    edge_cap_t& f(infoCells[mf.first->info()].f[mf.second]);
                    f += w;

                    // vert_info_t& v0 = mf.first->vertex(0)->info();
                    // vert_info_t& v1 = mf.first->vertex(1)->info();
                    // vert_info_t& v2 = mf.first->vertex(2)->info();
                    // if (v0.point_type && v1.point_type && v2.point_type) {
                    //     // float wl = std::min(v0.views.GetSize(),
                    //     //     std::min(v1.views.GetSize(), v2.views.GetSize()));
                    //     f = f + 16 * (1.f - dist_factor);
                    // }
                }
                // ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
                if (!(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi)) {
                    continue;
                }
                #ifdef DELAUNAY_WEAKSURF
                ASSERT(vert.viewsInfo[v].cell2End == NULL);
                vert.viewsInfo[v].cell2End = inter.facet.first;
                #endif
            }
            ++progress;
            
            // std::cout<<"\r";
            // std::cout<<"Points weighted ["<<progress<<" / "<<delaunay.number_of_vertices()<<"]"<<std::flush;
        }
        // progress.close();
        // printf("\tweighting completed in %s", TD_TIMER_GET_FMT().c_str());
        printf("\tweighting completed\n");
        }
#else
        {
        int progress = 0;
        uint64_t indices_print_step = delaunay.number_of_vertices() / 100 + 1;
        auto ComputeEdgeWeight = [&](delaunay_t::Vertex_iterator vs, delaunay_t::Vertex_iterator ve) {
            std::vector<facet_t> in_facets, out_facets;
            for (delaunay_t::Vertex_iterator vi = vs; vi != ve; ++vi) {
                vert_info_t& vert(vi->info());
                if (vert.views.IsEmpty())
                    continue;
                #ifdef DELAUNAY_WEAKSURF
                vert.AllocateInfo();
                #endif
                const point_t& p(vi->point());
                const Eigen::Vector3f pt(CGAL2EIGEN<REAL>(p));
                FOREACH(v, vert.views) {
                    const typename vert_info_t::view_t view(vert.views[v]);
                    const uint32_t imageID(view.idxView);
                    const edge_cap_t alpha_vis(view.weight);
                    const mvs::Image& imageData = images[imageID];
                    // ASSERT(imageData.IsValid());
                    const Eigen::Vector3f C(imageData.GetC());
                    const camera_cell_t& camCell = camCells[imageID];
                    // compute the ray used to find point intersection
                    const Eigen::Vector3f vecCamPoint(pt-C);
                    const float norm = vecCamPoint.norm();
                    if (norm < 1e-12 || std::isnan(norm) || std::isinf(norm)) {
                        continue;
                    }
                    const REAL invLenCamPoint(REAL(1)/norm);
                    intersection_t inter(pt, Eigen::Vector3f(vecCamPoint*invLenCamPoint));
                    // find faces intersected by the camera-point segment
                    const segment_t segCamPoint(EIGEN2CGAL(C), p);
                    
                    std::vector<intersection_t> inters;
                    if (!intersect(delaunay, segCamPoint, camCell.facets, out_facets, inter))
                        continue;
                    do {
                        inters.emplace_back(inter);
                        in_facets = out_facets;
                    } while (intersect(delaunay, segCamPoint, in_facets, out_facets, inter));

                    for (auto inter_ : inters) {
                        // assign score, weighted by the distance from the point to the intersection
                        const float dist_factor = 
                            EXP(-SQUARE((float)inter_.dist)*inv2SigmaSq);
                        const edge_cap_t w(alpha_vis * (1.f - dist_factor));
                        edge_cap_t& f(infoCells[inter_.facet.first->info()].f[inter_.facet.second]);
                        f += w;

                        // vert_info_t& v0 = inter_.facet.first->vertex(0)->info();
                        // vert_info_t& v1 = inter_.facet.first->vertex(1)->info();
                        // vert_info_t& v2 = inter_.facet.first->vertex(2)->info();
                        // if (v0.point_type && v1.point_type && v2.point_type) {
                        //     f = f + 16 * (1.f - dist_factor);
                        // }
                    }
                    
                    // ASSERT(out_facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
                    if (!(out_facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi)) {
                        continue;
                    }
                    #ifdef DELAUNAY_WEAKSURF
                    ASSERT(vert.viewsInfo[v].cell2Cam == NULL);
                    vert.viewsInfo[v].cell2Cam = inter.facet.first;
                    #endif
                    // find faces intersected by the endpoint-point segment
                    inter.dist = FLT_MAX; inter.bigger = false;
                    const Eigen::Vector3f endPoint(pt+vecCamPoint*(invLenCamPoint*sigma));
                    const segment_t segEndPoint(EIGEN2CGAL(endPoint), p);
                    const cell_handle_t endCell(delaunay.locate(segEndPoint.source(), vi->cell()));
                    ASSERT(endCell != cell_handle_t());
                    fetchCellFacets<CGAL::NEGATIVE>(delaunay, hullFacets, endCell, imageData, out_facets);
                    edge_cap_t& t(infoCells[endCell->info()].t);

                    t += alpha_vis;

                    inters.clear();
                    in_facets = out_facets;
                    while (intersect(delaunay, segEndPoint, in_facets, out_facets, inter)) {
                        inters.emplace_back(inter);
                        in_facets = out_facets;
                    }

                    // assign score, weighted by the distance from the point to the intersection
                    for (auto inter_ : inters) {
                        const facet_t& mf(delaunay.mirror_facet(inter_.facet));
                        const float dist_factor = 
                            EXP(-SQUARE((float)inter_.dist)*inv2SigmaSq);
                        const edge_cap_t w(alpha_vis * (1.f - dist_factor));
                        edge_cap_t& f(infoCells[mf.first->info()].f[mf.second]);
                        f += w;

                        // vert_info_t& v0 = mf.first->vertex(0)->info();
                        // vert_info_t& v1 = mf.first->vertex(1)->info();
                        // vert_info_t& v2 = mf.first->vertex(2)->info();
                        // if (v0.point_type && v1.point_type && v2.point_type) {
                        //     // float wl = std::min(v0.views.GetSize(),
                        //     //     std::min(v1.views.GetSize(), v2.views.GetSize()));
                        //     f = f + 16 * (1.f - dist_factor);
                        // }
                    }

                    // ASSERT(out_facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
                    if (!(out_facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi)) {
                        continue;
                    }
                    #ifdef DELAUNAY_WEAKSURF
                    ASSERT(vert.viewsInfo[v].cell2End == NULL);
                    vert.viewsInfo[v].cell2End = inter.facet.first;
                    #endif
                }
                ++progress;
                if (progress % indices_print_step == 0) {
                    std::cout<<"\r";
                    std::cout<<"Points weighted ["<<progress<<" / "<<delaunay.number_of_vertices()<<"]"<<std::flush;
                }
            }
        };

            const size_t num_slice = (num_vert + num_eff_threads - 1) / num_eff_threads;
            std::vector<delaunay_t::Vertex_iterator> iters(num_eff_threads + 1);
            delaunay_t::Vertex_iterator vi = delaunay.vertices_begin();
            for (int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
                iters[thread_idx] = vi;
                size_t num_limit = std::min(num_slice * (thread_idx+ 1), num_vert);
                for (size_t i = thread_idx * num_slice; i < num_limit; ++i) {
                    ++vi;
                }
            }
            iters[num_eff_threads] = delaunay.vertices_end();
            for (int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
                delaunay_t::Vertex_iterator vi = iters[thread_idx]; 
                delaunay_t::Vertex_iterator ve = iters[thread_idx + 1];
                thread_pool->AddTask(ComputeEdgeWeight, vi, ve);
            }
            thread_pool->Wait();
            GetAvailableMemory(memory_2);
            std::cout << "free memory 2: " << memory_2 << std::endl;
        }
#endif
        camCells.clear();

        #ifdef DELAUNAY_WEAKSURF
        // enforce t-edges for each point-camera pair with free-space support weights
        if (bUseFreeSpaceSupport) {
        // TD_TIMER_STARTD();
        for (delaunay_t::Vertex_iterator vi=delaunay.vertices_begin(), vie=delaunay.vertices_end(); vi!=vie; ++vi) {
            const vert_info_t& vert(vi->info());
            if (vert.views.IsEmpty())
                continue;
            const point_t& p(vi->point());
            const Eigen::Vector3f pt(CGAL2EIGEN<float>(p));
            FOREACH(v, vert.views) {
                const uint32_t imageID(vert.views[(vert_info_t::view_vec_t::IDX)v]);
                const mvs::Image& imageData = images[imageID];
                // ASSERT(imageData.IsValid());
                const Eigen::Vector3f C(imageData.GetC());
                // compute the ray used to find point intersection
                const Eigen::Vector3f vecCamPoint(pt-C);
                const float invLenCamPoint(1.f/(vecCamPoint).norm());
                // find faces intersected by the point-camera segment and keep the max free-space support score
                const Eigen::Vector3f bgnPoint(pt-vecCamPoint*(invLenCamPoint*sigma*kf));
                const segment_t segPointBgn(p, EIGEN2CGAL(bgnPoint));
                intersection_t inter;
                if (!intersectFace(delaunay, segPointBgn, vi, vert.viewsInfo[v].cell2Cam, facets, inter))
                    continue;
                edge_cap_t beta(0);
                do {
                    const edge_cap_t fs(freeSpaceSupport(delaunay, infoCells, inter.facet.first));
                    if (beta < fs)
                        beta = fs;
                } while (intersectFace(delaunay, segPointBgn, facets, facets, inter));
                // find faces intersected by the point-endpoint segment
                const Eigen::Vector3f endPoint(pt+vecCamPoint*(invLenCamPoint*sigma*kb));
                const segment_t segPointEnd(p, EIGEN2CGAL(endPoint));
                if (!intersectFace(delaunay, segPointEnd, vi, vert.viewsInfo[v].cell2End, facets, inter))
                    continue;
                edge_cap_t gammaMin(FLT_MAX), gammaMax(0);
                do {
                    const edge_cap_t fs(freeSpaceSupport(delaunay, infoCells, inter.facet.first));
                    if (gammaMin > fs)
                        gammaMin = fs;
                    if (gammaMax < fs)
                        gammaMax = fs;
                } while (intersectFace(delaunay, segPointEnd, facets, facets, inter));
                const edge_cap_t gamma((gammaMin+gammaMax)*0.5f);
                // if the point can be considered an interface point,
                // enforce the t-edge weight of the end cell
                const edge_cap_t epsAbs(beta-gamma);
                const edge_cap_t epsRel(gamma/beta);
                if (epsRel < kRel && epsAbs > kAbs && gamma < kOutl) {
                    edge_cap_t& t(infoCells[inter.ncell->info()].t);
                    t *= epsAbs;
                }
            }
        }
        printf("\tt-edge reinforcement completed\n");
        }
        #endif
        GetAvailableMemory(memory_3);
        printf("Delaunay tetrahedras weighting completed: %u cells, %u faces (free memory: %f)\n", delaunay.number_of_cells(), delaunay.number_of_facets(), memory_3);
    }

    // run graph-cut and extract the mesh
    {
        // TD_TIMER_STARTD();

        std::cout << "Runing GraphCut and Extract the mesh" << std::endl;
        // create graph
        MaxFlow<cell_size_t,edge_cap_t> graph(delaunay.number_of_cells());
        
        size_t progress = 0;
        // set weights
        for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), ce=delaunay.all_cells_end(); ci!=ce; ++ci) {
            const cell_size_t ciID(ci->info());
            const cell_info_t& ciInfo(infoCells[ciID]);
            edge_cap_t s = ciInfo.s;
            edge_cap_t t = ciInfo.t;
            if (!(ISFINITE(s) && s >= 0 && ISFINITE(t) && t >= 0)) {
                s = t = kInf;
            }

            graph.AddNode(ciID, s, t);
            for (int i=0; i<4; ++i) {
                const cell_handle_t cj(ci->neighbor(i));
                const cell_size_t cjID(cj->info());
                if (cjID < ciID) continue;
                const cell_info_t& cjInfo(infoCells[cjID]);
                const int j(cj->index(ci));
                const edge_cap_t q((1.f - MINF(computePlaneSphereAngle(delaunay, facet_t(ci,i)), computePlaneSphereAngle(delaunay, facet_t(cj,j))))*kQual);
                edge_cap_t fi = ciInfo.f[i]+q;
                edge_cap_t fj = cjInfo.f[j]+q;
                if (!(ISFINITE(fi) && fi >= 0 && ISFINITE(fj) && fj >= 0)) {
                    fi = fj = kInf;
                }
                graph.AddEdge(ciID, cjID, fi, fj);
            }
            ++progress;
            // std::cout << "\r";
            // std::cout << "Adding Node [" << progress << " / " << delaunay.number_of_cells() << "]" << std::flush;
        }
        GetAvailableMemory(memory_4);
        std::cout << "free memory 4: " << memory_4 << std::endl;
        infoCells.clear();

        std::cout << "\nCompute Max Flow\n";
        // find graph-cut solution
        const float maxflow(graph.ComputeMaxFlow());
        GetAvailableMemory(memory_5);
        std::cout << "free memory 5: " << memory_5 << std::endl;

        std::cout << "Extract Mesh" << std::endl;        
        // TriangleMesh mesh;
        // extract surface formed by the facets between inside/outside cells
        const size_t nEstimatedNumVerts(delaunay.number_of_vertices());
        std::unordered_map<void*,int> mapVertices;
        #if defined(_MSC_VER) && (_MSC_VER > 1600)
        mapVertices.reserve(nEstimatedNumVerts);
        #endif
        mesh.vertices_.reserve(nEstimatedNumVerts);
        mesh.faces_.reserve(nEstimatedNumVerts*2);
        mesh.vertex_colors_.reserve(nEstimatedNumVerts);
        mesh.vertex_visibilities_.reserve(nEstimatedNumVerts);
        mesh.vertex_labels_.reserve(nEstimatedNumVerts);
        for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), ce=delaunay.all_cells_end(); ci!=ce; ++ci) {
            const cell_size_t ciID(ci->info());
            for (int i=0; i<4; ++i) {
                if (delaunay.is_infinite(ci, i)) continue;
                const cell_handle_t cj(ci->neighbor(i));
                const cell_size_t cjID(cj->info());
                if (ciID < cjID) continue;
                const bool ciType(graph.IsNodeOnSrcSide(ciID));
                if (ciType == graph.IsNodeOnSrcSide(cjID)) continue;
                Eigen::Vector3i face;
                const triangle_vhandles_t tri(getTriangle(ci, i));
                for (int v=0; v<3; ++v) {
                    const vertex_handle_t vh(tri.verts[v]);
                    ASSERT(vh->point() == delaunay.triangle(ci,i)[v]);
                    const auto pairItID(mapVertices.insert(std::make_pair(vh.for_compact_container(), mesh.vertices_.size())));
                    if (pairItID.second) {
                        mesh.vertices_.emplace_back(CGAL2EIGEN<double>(vh->point()));
                        mesh.vertex_colors_.emplace_back(vh->info().color.cast<double>());

                        auto views = vh->info().views;
                        std::vector<uint32_t> vvis;
                        for (size_t idx = 0; idx < views.GetSize(); ++idx) {
                            vvis.emplace_back(views[idx].idxView);
                        }
                        mesh.vertex_visibilities_.emplace_back(vvis);
                        mesh.vertex_labels_.emplace_back(vh->info().point_type);
                    }
                    ASSERT(pairItID.first->second < mesh.vertices_.size());
                    face[v] = pairItID.first->second;
                }
                // correct face orientation
                if (!ciType)
                    std::swap(face[0], face[2]);
                mesh.faces_.emplace_back(face);
            }
        }
        GetAvailableMemory(memory_6);
        std::cout << "free memory 6: " << memory_6 << std::endl;
        delaunay.clear();
        GetAvailableMemory(memory_7);
        printf("Delaunay tetrahedras graph-cut completed (%g flow): %u vertices, %u faces (free memory: %f)\n", maxflow, mesh.vertices_.size(), mesh.faces_.size(), memory_7);
    }
    if (mesh.vertices_.size() < 3 || mesh.faces_.size() < 1){
        std::cout << "Error: mesh.vertices_.size() < 3 || mesh.faces_.size() < 1" << std::endl;
        return false;
    }
    // // fix non-manifold vertices and edges
    // for (unsigned i=0; i<nItersFixNonManifold; ++i)
    //     if (!FixNonManifold(mesh))
    //         break;
    GetAvailableMemory(end_memory);

    std::cout << "free memory end: " << end_memory << std::endl;

    return true;
}


bool DelaunayMeshing(TriangleMesh& mesh,
                     PointCloud& pointcloud, 
                     const std::vector<mvs::Image>& images,
                    const bool sampInsert,
                     const float distInsert,
                     const float diffDepth,
                    const float plane_insert_factor,
                    const float plane_score_thred,
                     const bool bUseFreeSpaceSupport, 
                     const unsigned nItersFixNonManifold,
                     const float kSigma, const float kQual, const float kb,
                     const float kf, const float kRel, const float kAbs, 
                     const float kOutl, const float kInf) {
    ASSERT(!pointcloud.IsEmpty());

    delaunay_t delaunay;

    bool b_resave = false;
#ifdef DELAUNAY_SAVE_POINTS
    b_resave = true;
#endif
    Timer delaunay_timer;
    delaunay_timer.Start();

    DelaunayInsert(pointcloud, delaunay, images, sampInsert, distInsert, diffDepth, plane_insert_factor, plane_score_thred, b_resave);

    std::cout << StringPrintf("DelaunayInsert Done in  %.3fs\n", 
        delaunay_timer.ElapsedSeconds());

    DelaunayGraphCut(mesh, delaunay, images, bUseFreeSpaceSupport, 
                     nItersFixNonManifold, kSigma, kQual, kb, kf, 
                     kRel, kAbs, kOutl,kInf);

    return true;
}

int InitCells(std::vector<std::vector<std::size_t> > &point_cluster_map,
            std::vector<struct Box> &cluster_bound_box,
            const std::vector<PlyPoint> &ply_points,
            const Model& model, const int max_num_points,
            const float overlap_factor,
            std::vector<Eigen::Vector3f>& transformed_points,
            std::vector<std::size_t>& cell_point_count,
            std::vector<std::vector<std::size_t>>& point_cell_map,
            // std::vector<std::vector<std::size_t>>& cell_points_map_set,
            std::vector<struct Box>& cell_bound_boxs,
            std::size_t& grid_size_x, std::size_t& grid_size_y,
            float& cell_size, const float num_points_factor = 1.0f, 
            MeshBox roi_meshbox = MeshBox()){
    Timer init_cells_timer;
    init_cells_timer.Start();
    std::cout << "Init Cell ..."  << std::endl;
    bool has_roi = roi_meshbox.IsValid();
    if (!has_roi){
        for(int i = 0; i < ply_points.size(); i ++){
            roi_meshbox.x_min = std::min(roi_meshbox.x_min, ply_points.at(i).x);
            roi_meshbox.y_min = std::min(roi_meshbox.y_min, ply_points.at(i).y);
            roi_meshbox.z_min = std::min(roi_meshbox.z_min, ply_points.at(i).z);
            roi_meshbox.x_max = std::max(roi_meshbox.x_max, ply_points.at(i).x);
            roi_meshbox.y_max = std::max(roi_meshbox.y_max, ply_points.at(i).y);
            roi_meshbox.z_max = std::max(roi_meshbox.z_max, ply_points.at(i).z);
        }
        roi_meshbox.SetBoundary();
    }

    Eigen::Matrix3f pivot;
    Eigen::Vector3f centroid(Eigen::Vector3f::Zero());
    std::vector<Eigen::Vector3f> sfm_points;
    sfm_points.reserve(model.points.size());
    for (int i = 0; i < model.points.size(); i++){
        Eigen::Vector3f xyz(model.points.at(i).x, model.points.at(i).y, model.points.at(i).z);
        Eigen::Vector3f xyz_t = roi_meshbox.rot * xyz;
        if (xyz_t.x() < roi_meshbox.box_x_min || xyz_t.x() > roi_meshbox.box_x_max ||
            xyz_t.y() < roi_meshbox.box_y_min || xyz_t.y() > roi_meshbox.box_y_max ||
            xyz_t.z() < roi_meshbox.box_z_min || xyz_t.z() > roi_meshbox.box_z_max){
            continue;
        }
        sfm_points.push_back(xyz);
    }
    sfm_points.shrink_to_fit();
    if (has_roi){
        pivot = roi_meshbox.rot;
    } else {
        if (sfm_points.size() < 3){
            sfm_points.clear();
            sfm_points.reserve(model.points.size());
            for (int i = 0; i < model.points.size(); i++){
                std::cout << "Warning: sfm points is empty " << std::endl;
                Eigen::Vector3f xyz(model.points.at(i).x, model.points.at(i).y, model.points.at(i).z);
                sfm_points.push_back(xyz);
            }
        }
        pivot = ComputePovitMatrix(sfm_points);
    }
    std::cout << "has roi, model.points.size(), sfm_point.size: " << has_roi << ", "<< model.points.size() << " -> " << sfm_points.size() << std::endl;
    roi_meshbox.Print();

    std::vector<float> point_spacings;
    float average_spacing = ComputeAvergeSapcing(point_spacings, sfm_points);

    // Transform points & Calculate BoundingBox.
    Eigen::Vector3f box_min, box_max;
    box_min = box_max = pivot * sfm_points[0];
    for (int i = 0; i < sfm_points.size(); ++i) {
        if (point_spacings[i] > 5 * average_spacing){
            continue;
        }
        Eigen::Vector3f point = pivot * sfm_points[i];
        
        box_min[0] = std::min(box_min[0], point[0]);
        box_min[1] = std::min(box_min[1], point[1]);
        box_min[2] = std::min(box_min[2], point[2]);
        box_max[0] = std::max(box_max[0], point[0]);
        box_max[1] = std::max(box_max[1], point[1]);
        box_max[2] = std::max(box_max[2], point[2]);
    }
    std::cout << "Box(sparse): " << box_min.transpose() << " / " << box_max.transpose() << std::endl;

    int point_num = ply_points.size();
    int min_num_cluster = 1.0 + 
        point_num / (max_num_points * num_points_factor);
    cell_size = std::sqrt((box_max.x() - box_min.x()) * 
                (box_max.y() - box_min.y()) / min_num_cluster);
    const float inv_cell_size = 1.0f / cell_size;

    transformed_points.resize(ply_points.size());
    for(int i = 0; i < ply_points.size(); i ++){
        Eigen::Vector3f ori_point;
        ori_point.x() = ply_points.at(i).x;
        ori_point.y() = ply_points.at(i).y;
        ori_point.z() = ply_points.at(i).z;
        transformed_points[i] = pivot * ori_point;
    }
    Eigen::Vector3f ply_box_min, ply_box_max;
    ply_box_min = ply_box_max = transformed_points[0];
    for (int i = 0; i < point_num; ++i) {
        auto &point = transformed_points[i];        
        ply_box_min[0] = std::min(ply_box_min[0], point[0]);
        ply_box_min[1] = std::min(ply_box_min[1], point[1]);
        ply_box_min[2] = std::min(ply_box_min[2], point[2]);
        ply_box_max[0] = std::max(ply_box_max[0], point[0]);
        ply_box_max[1] = std::max(ply_box_max[1], point[1]);
        ply_box_max[2] = std::max(ply_box_max[2], point[2]);
    }

    if (ply_points.size() < max_num_points){
        point_cluster_map.resize(ply_points.size());
        for (std::size_t i = 0; i < ply_points.size(); ++i) {
            point_cluster_map[i].push_back(std::size_t(0));
        }
        struct Box box;
        box.x_min = ply_box_min.x();
        box.y_min = ply_box_min.y();
        box.z_min = ply_box_min.z();
        box.x_max = ply_box_max.x();
        box.y_max = ply_box_max.y();
        box.z_max = ply_box_max.z();

        box.x_box_min = ply_box_min.x();
        box.y_box_min = ply_box_min.y();
        box.z_box_min = ply_box_min.z();
        box.x_box_max = ply_box_max.x();
        box.y_box_max = ply_box_max.y();
        box.z_box_max = ply_box_max.z();

        box.rot = pivot;
        cluster_bound_box.resize(1);
        cluster_bound_box[0] = box;
        std::cout << "Num Cluster = 1 (ply_points.size() < max_num_points)" << std::endl;
        return 1;
    }

    grid_size_x = 
        static_cast<std::size_t>((box_max.x() - box_min.x()) * inv_cell_size) + 1;
    grid_size_y = 
        static_cast<std::size_t>((box_max.y() - box_min.y()) * inv_cell_size) + 1;
    const std::size_t grid_side = grid_size_x;
    std::size_t grid_slide = grid_side * grid_size_y;

    double delt_x = (grid_size_x * cell_size - box_max.x() + box_min.x()) / 2;
    double delt_y = (grid_size_y * cell_size - box_max.y() + box_min.y()) / 2;
    box_min[0] -= delt_x;
    box_min[1] -= delt_y;
    box_max[0] += delt_x;
    box_max[1] += delt_y;

    std::vector<std::size_t> temp(grid_slide, 0);
    cell_point_count.swap(temp);
    point_cell_map.resize(point_num);
    // cell_points_map_set.resize(grid_slide);
    double delt_dist = cell_size * overlap_factor;
    int neighbor_x[8] = {1, 0, -1 ,0, 1, 1, -1, -1};
    int neighbor_y[8] = {0, 1, 0, -1, 1, -1, 1, -1};

    // for (std::size_t i = 0; i < point_num; ++i) {
    auto InsertCell = [&](std::size_t begin, std::size_t end){
        for (std::size_t i = begin; i < end; ++i){
            const auto &point = transformed_points[i];
            int x_cell_temp = static_cast<int>((point.x() - box_min.x()) * inv_cell_size);
            int y_cell_temp = static_cast<int>((point.y() - box_min.y()) * inv_cell_size);
            std::size_t x_cell = static_cast<std::size_t>(
                std::max(0, std::min(x_cell_temp, (int)grid_size_x-1)));
            std::size_t y_cell = static_cast<std::size_t>(
                std::max(0, std::min(y_cell_temp, (int)grid_size_y-1)));

            std::size_t cell_idx = y_cell * grid_side + x_cell;
            point_cell_map.at(i).push_back(cell_idx);

            if (overlap_factor > 1e-6){
                for (int j = 0; j < 8; j++){
                    int x_cell_temp_neigh = static_cast<int>((point.x() + 
                        neighbor_x[j] * delt_dist - box_min.x()) * inv_cell_size);
                    int y_cell_temp_neigh = static_cast<int>((point.y() + 
                        neighbor_y[j] * delt_dist - box_min.y()) * inv_cell_size);
                    std::size_t x_cell_neigh = static_cast<std::size_t>(
                        std::max(0, std::min(x_cell_temp_neigh, (int)grid_size_x-1)));
                    std::size_t y_cell_neigh = static_cast<std::size_t>(
                        std::max(0, std::min(y_cell_temp_neigh, (int)grid_size_y-1)));
                    if (x_cell_neigh != x_cell || y_cell_neigh != y_cell){
                        std::size_t cell_idx_neigh = y_cell_neigh * grid_side + x_cell_neigh;
                        point_cell_map.at(i).push_back(cell_idx_neigh);
                    }
                }
            }
        }
    };

    Timer insert_cells_timer;
    insert_cells_timer.Start();

    const int num_valid_thread = GetEffectiveNumThreads(-1);
    // const int num_valid_thread = 1;
    std::unique_ptr<ThreadPool> insert_thread_pool;
    insert_thread_pool.reset(new ThreadPool(num_valid_thread));

    std::size_t num_points_per_thread = ceil((float)point_num * 3 / (float)num_valid_thread);
    int num_cluster = ceil((float)point_num / (float)num_points_per_thread);
    for (int i = 0; i < num_cluster; i++){
        std::size_t begin = i * num_points_per_thread;
        std::size_t end = std::min((std::size_t)(i+1) * num_points_per_thread, (std::size_t)point_num);
        insert_thread_pool->AddTask(InsertCell, begin, end);
    }
    insert_thread_pool->Wait();

    std::cout << StringPrintf("num_cluster: %d, num_points_per_thread: %d in %.3fs\n", 
        num_cluster, num_points_per_thread, insert_cells_timer.ElapsedSeconds());

    for (std::size_t i = 0; i < point_num; ++i) {
        for (auto cell_id : point_cell_map.at(i)){
            cell_point_count[cell_id]++;
            // cell_points_map_set.at(cell_id).emplace(i);
        }
    }
    
    // bound box
    // std::vector<struct Box> cell_bound_boxs(grid_slide);
    cell_bound_boxs.resize(grid_slide);
    for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){
        int y_cell = cell_idx / grid_size_x;
        int x_cell = cell_idx % grid_size_x;

        struct Box box;
        box.x_min = 
            x_cell == 0 ? ply_box_min.x() : x_cell * cell_size + box_min.x();
        box.y_min = 
            y_cell == 0 ? ply_box_min.y() : y_cell * cell_size + box_min.y();
        box.z_min = ply_box_min.z();
        box.x_max = x_cell == (grid_size_x - 1) ? 
            ply_box_max.x() : (x_cell + 1) * cell_size + box_min.x();
        box.y_max = y_cell == (grid_size_y - 1) ? 
            ply_box_max.y() : (y_cell + 1) * cell_size + box_min.y();
        box.z_max = ply_box_max.z();

        box.x_box_min = x_cell * cell_size + box_min.x();
        box.y_box_min = y_cell * cell_size + box_min.y();
        box.z_box_min = ply_box_min.z();
        box.x_box_max = (x_cell + 1) * cell_size + box_min.x();
        box.y_box_max = (y_cell + 1) * cell_size + box_min.y();
        box.z_box_max = ply_box_max.z();

        box.rot = pivot;
        cell_bound_boxs[cell_idx] = box;
    }

    std::cout << "num cluster: " << grid_slide << "   "
              << "cell size: " << cell_size 
              << " in " << init_cells_timer.ElapsedSeconds() << " [s]" << std::endl;
    return grid_slide;
}

bool WhetherMerge(
    const std::vector<struct Box> &cell_bound_box,
    const std::vector<struct Box> &cluster_bound_box,
    std::vector<std::size_t> &cell_point_count,
    std::vector<std::size_t> &cluster_point_count,
    const int cell_idx, const int merge_cluster_idx,
    const int max_num_points){
    bool x_flag = 
        (std::abs(cell_bound_box[cell_idx].x_min - 
        cluster_bound_box[merge_cluster_idx].x_min) < EPSILON) && 
        (std::abs(cell_bound_box[cell_idx].x_max - 
        cluster_bound_box[merge_cluster_idx].x_max) < EPSILON) &&
        ((std::abs(cell_bound_box[cell_idx].y_min - 
        cluster_bound_box[merge_cluster_idx].y_max) < EPSILON) ||
        std::abs(cell_bound_box[cell_idx].y_max - 
        cluster_bound_box[merge_cluster_idx].y_min) < EPSILON);
    bool y_flag = 
        (std::abs(cell_bound_box[cell_idx].y_min - 
        cluster_bound_box[merge_cluster_idx].y_min) < EPSILON) && 
        (std::abs(cell_bound_box[cell_idx].y_max - 
        cluster_bound_box[merge_cluster_idx].y_max) < EPSILON) &&
        ((std::abs(cell_bound_box[cell_idx].x_min - 
        cluster_bound_box[merge_cluster_idx].x_max) < EPSILON) ||
        std::abs(cell_bound_box[cell_idx].x_max - 
        cluster_bound_box[merge_cluster_idx].x_min) < EPSILON);
    bool points_flag = (cell_point_count[cell_idx] + 
        cluster_point_count[merge_cluster_idx] < max_num_points);
    if ((x_flag || y_flag) && points_flag){
        // std::cout << "cell idx:" << cell_idx << "\tcluster_idx:"
        //           << merge_cluster_idx << std::endl;
        return true;
    }
    return false;
}

std::size_t Cell2Cluster(
    std::vector<int> &cell_cluster_map,
    std::vector<struct Box> &cluster_bound_box,
    std::vector<std::size_t> &cell_point_count,
    std::vector<std::vector<std::size_t> >& point_cell_map,
    std::vector<Eigen::Vector3f>& points ,
    std::vector<struct Box> &cell_bound_box,
    const std::size_t grid_size_x,
    const std::size_t grid_size_y,
    const double max_cell_size,
    const int max_num_points,
    const float overlap_factor){
    std::cout << "Cell2Cluster Start..." << std::endl;
    Timer c2c_timer;
    c2c_timer.Start();

    const std::size_t ori_cell_num = cell_bound_box.size();
    std::size_t grid_slide = cell_bound_box.size();
    const int point_num = points .size();
    for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){      
        while(cell_point_count.at(cell_idx) > max_num_points && 
              (cell_bound_box[cell_idx].x_box_max - cell_bound_box[cell_idx].x_box_min) > max_cell_size * 0.24 &&
              (cell_bound_box[cell_idx].y_box_max - cell_bound_box[cell_idx].y_box_min) > max_cell_size * 0.24){
            cell_point_count.push_back(0);
            cell_bound_box.push_back(Box());
            if (((cell_bound_box[cell_idx].x_box_max - 
                cell_bound_box[cell_idx].x_box_min) - 
                (cell_bound_box[cell_idx].y_box_max - 
                cell_bound_box[cell_idx].y_box_min)) < 
                max_cell_size * 0.01){
                // Split in Y direction
                float split_y = (cell_bound_box[cell_idx].y_box_min + 
                                cell_bound_box[cell_idx].y_box_max) / 2;
                struct Box box1 = cell_bound_box[cell_idx];
                struct Box box2 = cell_bound_box[cell_idx];
                box1.y_max = split_y;
                box2.y_min = split_y;
                box1.y_box_max = split_y;
                box2.y_box_min = split_y;
                cell_bound_box[cell_idx] = box1;
                cell_bound_box[grid_slide] = box2;
                
                for(std::size_t i = 0; i < point_num; ++i){
                    auto iter = std::find(point_cell_map[i].begin(), point_cell_map[i].end(), cell_idx);
                    if (iter == point_cell_map[i].end()){
                        continue;
                    }
                    // if (point_cell_map_set[i].find(cell_idx) == 
                    //     point_cell_map_set[i].end()){
                    //     continue;
                    // }
                    double delt_y = std::min((cell_bound_box[cell_idx].y_box_max - 
                        cell_bound_box[cell_idx].y_box_min), (float)max_cell_size)
                        * overlap_factor;
                    auto &point = points [i];
                    if (point.y() > split_y + delt_y){
                        // point_cell_map_set[i].erase(cell_idx);
                        point_cell_map[i].erase(iter);
                        cell_point_count[cell_idx]--;
                    }
                    if (point.y() > split_y - delt_y){
                        // point_cell_map_set[i].emplace(grid_slide);
                        point_cell_map[i].push_back(grid_slide);
                        cell_point_count[grid_slide]++;
                    }
                }
            } else {
                // Split in X direction
                float split_x = (cell_bound_box[cell_idx].x_box_min + 
                                cell_bound_box[cell_idx].x_box_max) / 2;
                struct Box box1 = cell_bound_box[cell_idx];
                struct Box box2 = cell_bound_box[cell_idx];
                box1.x_max = split_x;
                box2.x_min = split_x;
                box1.x_box_max = split_x;
                box2.x_box_min = split_x;
                cell_bound_box[cell_idx] = box1;
                cell_bound_box[grid_slide] = box2;
                
                for(std::size_t i = 0; i < point_num; ++i){
                    auto iter = std::find(point_cell_map[i].begin(), point_cell_map[i].end(), cell_idx);
                    if (iter == point_cell_map[i].end()){
                        continue;
                    }
                    // if (point_cell_map_set[i].find(cell_idx) == 
                    //     point_cell_map_set[i].end()){
                    //     continue;
                    // }
                    double delt_x = std::min((cell_bound_box[cell_idx].x_box_max - 
                        cell_bound_box[cell_idx].x_box_min), (float)max_cell_size) 
                        * overlap_factor;
                    auto &point = points [i];
                    if (point.x() > split_x + delt_x){
                        // point_cell_map_set[i].erase(cell_idx);
                        point_cell_map[i].erase(iter);
                        cell_point_count[cell_idx]--;
                    }
                    if (point.x() > split_x - delt_x){
                        // point_cell_map_set[i].emplace(grid_slide);
                        point_cell_map[i].push_back(grid_slide);
                        cell_point_count[grid_slide]++;
                    }
                }
            }
            grid_slide++;
        } 
        std::cout << "Cell "<< cell_idx << "   point size: " 
                  << cell_point_count[cell_idx] << std::endl;
    }

    std::size_t cell_num = cell_point_count.size();
    cell_cluster_map.resize(cell_num);
    std::fill(cell_cluster_map.begin(), cell_cluster_map.end(), -1);

    std::vector<std::pair<std::size_t, std::size_t> > dense_cells;
    dense_cells.reserve(cell_num);
    std::vector<std::size_t> sparse_cells;
    sparse_cells.reserve(cell_num);
    std::vector<unsigned char> cell_type_map(cell_num, 0);
    for (std::size_t i = 0; i < cell_num; i++) {
        if (cell_point_count[i] == 0){
            continue;
        }
        if (cell_point_count[i] < max_num_points / 4.0) {
            sparse_cells.push_back(i);
            cell_type_map[i] = 128;
            continue;
        }

        dense_cells.emplace_back(cell_point_count[i], i);
        cell_type_map[i] = 255;
    }
    dense_cells.shrink_to_fit();
    sparse_cells.shrink_to_fit();

    std::size_t cluster_idx = 0;
    std::size_t dense_cell_num = dense_cells.size();
    cluster_bound_box.resize(dense_cell_num);
    std::vector<std::size_t> cluster_point_count;
    std::vector<std::vector<std::size_t> > cluster_cells_map;
    cluster_cells_map.reserve(cell_num);

    for (std::size_t i = 0; i < dense_cell_num; i++) {
        int cell_idx = dense_cells[i].second;
        if (cell_cluster_map[cell_idx] != -1) {
            continue;
        }

        std::size_t num_visited_points = 0;
        std::vector<std::size_t> cluster_cell;
        // cluster_cells_map.push_back(std::vector<std::size_t>());
        // auto &cluster_cells = cluster_cells_map.back();

        cell_cluster_map[cell_idx] = cluster_idx;
        cluster_cell.push_back(cell_idx);
        num_visited_points = cell_point_count[cell_idx];

        cluster_bound_box[i] = cell_bound_box[cell_idx];

        cluster_point_count.push_back(num_visited_points);
        cluster_cells_map.emplace_back(cluster_cell);
        cluster_idx++;
    }

    std::size_t cluster_num = cluster_idx;
    std::vector<std::size_t> cluster_idx_map;
    cluster_idx_map.reserve(cluster_num);
    for (std::size_t i = 0; i < cluster_num; i++) {
        cluster_idx_map.push_back(i);
    }
    cluster_idx_map.shrink_to_fit();

    for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
        const auto &cluster_idx = cluster_idx_map[i];
        cluster_point_count[i] = cluster_point_count[cluster_idx];
        cluster_cells_map[i] = cluster_cells_map[cluster_idx];
        for (auto cell_idx : cluster_cells_map[i]) {
            cell_cluster_map[cell_idx] = i;
        }
    }

    int inc_num_cluster = cluster_idx_map.size();
    for (auto cell_idx : sparse_cells) {
        bool merge_flag = false;
        int merge_cluster_idx = -1;
        for (std::size_t i = 0; i < inc_num_cluster; i++) {
            merge_cluster_idx = i;
            if (!WhetherMerge(cell_bound_box, cluster_bound_box, cell_point_count, 
                cluster_point_count, cell_idx, merge_cluster_idx, max_num_points)){
                continue;
            }

            cell_cluster_map[cell_idx] = merge_cluster_idx;
            cluster_point_count[merge_cluster_idx] += cell_point_count[cell_idx];
            cluster_cells_map[merge_cluster_idx].push_back(cell_idx);
            cluster_bound_box[merge_cluster_idx] += cell_bound_box[cell_idx];
            cell_type_map[cell_idx] = 255;
            merge_flag = true;
            std::cout << "merge: cell_idx: " << cell_idx 
                      << "to Cluster " << merge_cluster_idx << std::endl;
            break;
        }
          
        if (!merge_flag){            
            std::size_t num_visited_points = 0;
            // cluster_cells_map.push_back(std::vector<std::size_t>());
            // auto &cluster_cells = cluster_cells_map.back();
            std::vector<std::size_t> cluster_cell;

            cell_cluster_map[cell_idx] = cluster_idx;
            // cluster_cells.push_back(cell_idx);
            cluster_cell.push_back(cell_idx);
            num_visited_points = cell_point_count[cell_idx];

            cluster_idx_map.push_back(cluster_idx);
            inc_num_cluster++;

            cluster_point_count.push_back(num_visited_points);
            cluster_bound_box.push_back(cell_bound_box[cell_idx]);
            cluster_cells_map.emplace_back(cluster_cell);
            cell_type_map[cell_idx] = 255;
            // std::cout << num_visited_points << " points clustered" << std::endl;  

            cluster_idx++;
            cluster_num = cluster_idx;
        }
    }

    cluster_num = cluster_idx_map.size();
    cluster_point_count.resize(cluster_num);
    cluster_cells_map.resize(cluster_num);
    cluster_bound_box.resize(cluster_num);

    std::cout << StringPrintf("%d Cell -> %d Cluster in %.3fs\n", 
                            ori_cell_num, cluster_num, c2c_timer.ElapsedSeconds());
    return cluster_num;
}

int PointsCluster(std::vector<std::vector<std::size_t> > &point_cluster_map,
                  std::vector<struct Box> &cluster_bound_box,
                  const std::vector<PlyPoint> &ply_points,
                  const Model& model, const int max_num_points,
                  const float overlap_factor, const MeshBox& roi_meshbox = MeshBox()){
    Timer cluster_timer;
    cluster_timer.Start();

    std::vector<Eigen::Vector3f> transformed_points;
    std::vector<std::size_t> cell_point_count;
    std::vector<std::vector<std::size_t> > point_cell_map;
    std::vector<struct Box> cell_bound_boxs;
    std::size_t grid_size_x, grid_size_y;
    float cell_size;
    int grid_slide = InitCells(point_cluster_map, cluster_bound_box, ply_points, 
                  model, max_num_points, overlap_factor, transformed_points,
                  cell_point_count, point_cell_map, 
                  cell_bound_boxs, grid_size_x, grid_size_y, cell_size, 1.0f, roi_meshbox);
    if (grid_slide == 1){
        return 1;
    }

    std::vector<int> cell_cluster_map;
    int num_cluster = 
        Cell2Cluster(cell_cluster_map, cluster_bound_box, cell_point_count, 
        point_cell_map, transformed_points, cell_bound_boxs, 
        grid_size_x, grid_size_y, cell_size, max_num_points, 
        overlap_factor);

    std::size_t point_num = ply_points.size();
    point_cluster_map.clear();
    point_cluster_map.resize(point_num);
    for (std::size_t i = 0; i < point_num; ++i) {
        for (auto id : point_cell_map[i]){
            point_cluster_map[i].push_back(cell_cluster_map[id]);
        }
    }
    
    std::cout << StringPrintf("Point CLuster End in %.3fs\n",  cluster_timer.ElapsedSeconds());
    return num_cluster;
}

int PointsFastCluster(std::vector<std::vector<std::size_t> > &point_cluster_map,
                  std::vector<struct Box> &cluster_bound_box,
                  const std::vector<PlyPoint> &ply_points,
                  const Model& model, const int max_num_points,
                  const float overlap_factor){

    Timer cluster_timer;
    cluster_timer.Start();

    std::vector<Eigen::Vector3f> transformed_points;
    std::vector<std::size_t> cell_point_count;
    std::vector<std::vector<std::size_t> > point_cell_map;
    std::vector<struct Box> cell_bound_boxs;
    std::size_t grid_size_x, grid_size_y;
    float cell_size;
    int grid_slide = InitCells(point_cluster_map, cluster_bound_box, ply_points, 
                  model, max_num_points, overlap_factor, transformed_points,
                  cell_point_count, point_cell_map, 
                  cell_bound_boxs, grid_size_x, grid_size_y, cell_size, 0.1f);
    if (grid_slide == 1){
        return 1;
    }

    std::size_t num_cluster = grid_slide;
    std::vector<std::size_t> cell_cluster_map;

    if (1){
        Timer split_timer;
        split_timer.Start();

        std::size_t point_num = ply_points.size();
        std::vector<std::vector<std::size_t>> cell_points_map_vec(grid_slide);
        std::vector<std::size_t> cell_points_num_vec(grid_slide, 0);
        for (int cell_id = 0; cell_id < grid_slide; cell_id++){
            cell_points_map_vec.at(cell_id).resize(cell_point_count.at(cell_id));
        }
        for (std::size_t i = 0; i < point_num; i++){
            for (auto cell_id : point_cell_map.at(i)){
                cell_points_map_vec.at(cell_id).at(cell_points_num_vec.at(cell_id)) = i;
                cell_points_num_vec.at(cell_id)++;
            }
        }
        for (int cell_id = 0; cell_id < grid_slide; cell_id++){
            if (cell_points_map_vec.at(cell_id).size() != cell_points_num_vec.at(cell_id)){
                std::cout << "!!!cell_points_map_vec.at(cell_id).size() != cell_points_num_vec.at(cell_id)" << std::endl;
                exit(-1);
            }
        }

        std::vector<std::vector<std::size_t>> split_cell_point_count(grid_slide);
        std::vector<std::vector<std::vector<std::size_t>>> split_cell_points_map_set(grid_slide);
        std::vector<std::vector<struct Box>> split_cell_bound_boxs(grid_slide);

        auto SplitCell = [&](std::size_t cell_idx){
            std::size_t num_cell_point = cell_points_map_vec.at(cell_idx).size();

            int split_axis = 0;
            if (1){
                int pnt_step = 100;
                int all_point_id = 0;
                std::size_t cell_point_id = 0;
                double mean_x = 0, mean_y = 0, mean_z = 0;
                std::vector<float> cell_x(num_cell_point),cell_y(num_cell_point),cell_z(num_cell_point); 
                for (auto point_id : cell_points_map_vec.at(cell_idx)){
                    all_point_id++;
                    if (all_point_id % pnt_step != 0){
                        continue;
                    }
                    cell_x.at(cell_point_id) = transformed_points[point_id].x();
                    cell_y.at(cell_point_id) = transformed_points[point_id].y();
                    cell_z.at(cell_point_id) = transformed_points[point_id].z();
                    cell_point_id++;
                    mean_x += transformed_points[point_id].x();
                    mean_y += transformed_points[point_id].y();
                    mean_z += transformed_points[point_id].z();
                }
                mean_x /= (double)cell_point_id;
                mean_y /= (double)cell_point_id;
                mean_z /= (double)cell_point_id;
                double accum_x  = 0.0, accum_y = 0.0, accum_z = 0.0;
                for (auto point_id : cell_points_map_vec.at(cell_idx)){
                    accum_x += (transformed_points[point_id].x() - mean_x) 
                            * (transformed_points[point_id].x() - mean_x);
                    accum_y += (transformed_points[point_id].y() - mean_y) 
                            * (transformed_points[point_id].y() - mean_y);
                    accum_z += (transformed_points[point_id].z() - mean_z) 
                            * (transformed_points[point_id].z() - mean_z);
                }
                if (accum_x > accum_y && accum_x > accum_z){
                    split_axis = 0;
                } else if (accum_y > accum_x && accum_y > accum_z) {
                    split_axis = 1;
                } else {
                    split_axis = 2;
                }
            }

            std::size_t num_points = 0;
            std::vector<std::pair<std::size_t, float>> v_id_axis(num_cell_point);
            for (auto point_id : cell_points_map_vec.at(cell_idx)){
                v_id_axis.at(num_points) = 
                    std::pair<std::size_t, float>(point_id,  transformed_points[point_id][split_axis]);
                num_points++;
            }

            std::sort(v_id_axis.begin(), v_id_axis.end(), 
                [](std::pair<std::size_t, float> a, std::pair<std::size_t, float> b){
                    return a.second < b.second; });

            // if (num_points != cell_point_count[cell_idx]){
            //     std::cerr << "num_points != cell_point_count[cell_idx]" << std::endl;
            //     exit(-1);
            // }

            int num_split = ceil((float)num_points / max_num_points);
            // std::cout << "num_points, max_num_points, num_split: " << num_points << ", " 
            //     << max_num_points << ", " << num_split << std::endl;

            split_cell_point_count.at(cell_idx) = std::vector<std::size_t>(num_split, 0);
            split_cell_points_map_set.at(cell_idx).resize(num_split);
            split_cell_bound_boxs.at(cell_idx).resize(num_split);
            
            size_t num_points_per_cluster = ceil((float)num_points / num_split);
            for(std::size_t idx = 0; idx < num_points; idx++){
                int cluster_id = idx / num_points_per_cluster;
                split_cell_point_count.at(cell_idx).at(cluster_id)++;
                split_cell_points_map_set.at(cell_idx).at(cluster_id).push_back(v_id_axis.at(idx).first);
            }

            for (int cluster_id = 0; cluster_id < num_split; cluster_id++){
                struct Box box;
                box = cell_bound_boxs.at(cell_idx);
                std::size_t begin_id = std::max(cluster_id * num_points_per_cluster, (std::size_t)0);
                std::size_t end_id = std::min((cluster_id + 1) * num_points_per_cluster, num_points - 1);
                if (split_axis == 0){
                    box.x_min = v_id_axis.at(begin_id).second;
                    box.x_max = v_id_axis.at(end_id).second;
                    box.x_box_min = v_id_axis.at(begin_id).second;
                    box.x_box_max = v_id_axis.at(end_id).second;
                } else if (split_axis == 1){
                    box.y_min = v_id_axis.at(begin_id).second;
                    box.y_max = v_id_axis.at(end_id).second;
                    box.y_box_min = v_id_axis.at(begin_id).second;
                    box.y_box_max = v_id_axis.at(end_id).second;
                } else {
                    box.z_min = v_id_axis.at(begin_id).second;
                    box.z_max = v_id_axis.at(end_id).second;
                    box.z_box_min = v_id_axis.at(begin_id).second;
                    box.z_box_max = v_id_axis.at(end_id).second;
                }
                split_cell_bound_boxs.at(cell_idx).at(cluster_id) = box;
            }
        };

        std::vector<std::size_t> split_cell_ids;
        for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){
            if ( cell_points_map_vec.at(cell_idx).size() < max_num_points){
                continue;
            }
            split_cell_ids.push_back(cell_idx);
        }
        const int num_valid_thread = std::min(GetEffectiveNumThreads(-1), (int)split_cell_ids.size());
        std::unique_ptr<ThreadPool> split_thread_pool;
        split_thread_pool.reset(new ThreadPool(num_valid_thread));
        std::cout << "split cell with " << num_valid_thread << " thread." << std::endl;

        for (std::size_t idx = 0; idx < split_cell_ids.size(); idx++){
            split_thread_pool->AddTask(SplitCell, split_cell_ids[idx]);
            // SplitCell(cell_idx);
        }
        split_thread_pool->Wait();

        for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){
            if (split_cell_point_count.at(cell_idx).empty() || split_cell_bound_boxs.at(cell_idx).empty()){
                continue;
            }
            num_cluster += (split_cell_point_count.at(cell_idx).size() - 1);
        }

        cell_point_count.resize(num_cluster);
        // cell_points_map_set.resize(num_cluster);
        cell_bound_boxs.resize(num_cluster);
        int cluster_num = grid_slide;
        for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){
            if (split_cell_point_count.at(cell_idx).empty() || split_cell_bound_boxs.at(cell_idx).empty()){
                continue;
            }
            
            for (int idx = 0; idx < split_cell_point_count.at(cell_idx).size(); idx++){
                int cell_cluster_id = cell_idx;
                if (idx > 0){
                    cell_cluster_id = cluster_num;
                    cluster_num++;
                } 
                cell_point_count.at(cell_cluster_id) = split_cell_point_count.at(cell_idx).at(idx);
                // cell_points_map_set.at(cell_cluster_id) = split_cell_points_map_set.at(cell_idx).at(idx);
                cell_bound_boxs.at(cell_cluster_id) = split_cell_bound_boxs.at(cell_idx).at(idx);
                if (idx > 0){
                    for (auto point_id : split_cell_points_map_set.at(cell_idx).at(idx)){
                        // point_cell_map_set.at(point_id).clear();
                        std::vector<std::size_t>::iterator cell_it;
                        cell_it = std::find(point_cell_map.at(point_id).begin(), point_cell_map.at(point_id).end(), cell_idx);
                        *cell_it = cell_cluster_id;
                    }
                }
            }
        }
        std::cout << "split cell in " << split_timer.ElapsedSeconds() << "[s]" << std::endl;
    }

    {
        std::size_t cell_num = cell_point_count.size();
        cell_cluster_map.resize(cell_num);
        std::fill(cell_cluster_map.begin(), cell_cluster_map.end(), -1);

        std::vector<std::pair<std::size_t, std::size_t> > dense_cells;
        dense_cells.reserve(cell_num);
        std::vector<std::size_t> sparse_cells;
        sparse_cells.reserve(cell_num);
        std::vector<unsigned char> cell_type_map(cell_num, 0);
        for (std::size_t i = 0; i < cell_num; i++) {
            if (cell_point_count[i] == 0){
                continue;
            }
            if (cell_point_count[i] < max_num_points / 4.0) {
                sparse_cells.push_back(i);
                cell_type_map[i] = 128;
                continue;
            }

            dense_cells.emplace_back(cell_point_count[i], i);
            cell_type_map[i] = 255;
        }
        dense_cells.shrink_to_fit();
        sparse_cells.shrink_to_fit();

        std::size_t cluster_idx = 0;
        std::size_t dense_cell_num = dense_cells.size();
        cluster_bound_box.resize(dense_cell_num);
        std::vector<std::size_t> cluster_point_count;
        std::vector<std::vector<std::size_t> > cluster_cells_map;

        for (std::size_t i = 0; i < dense_cell_num; i++) {
            int cell_idx = dense_cells[i].second;
            if (cell_cluster_map[cell_idx] != -1) {
                continue;
            }

            std::size_t num_visited_points = 0;
            cluster_cells_map.push_back(std::vector<std::size_t>());
            auto &cluster_cells = cluster_cells_map.back();

            cell_cluster_map[cell_idx] = cluster_idx;
            cluster_cells.push_back(cell_idx);
            num_visited_points = cell_point_count[cell_idx];

            cluster_bound_box[i] = cell_bound_boxs[cell_idx];

            cluster_point_count.push_back(num_visited_points);
            cluster_idx++;
        }

        num_cluster = cluster_idx;
        std::vector<std::size_t> cluster_idx_map;
        cluster_idx_map.reserve(num_cluster);
        for (std::size_t i = 0; i < num_cluster; i++) {
            cluster_idx_map.push_back(i);
        }
        cluster_idx_map.shrink_to_fit();

        for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
            const auto &cluster_idx = cluster_idx_map[i];
            cluster_point_count[i] = cluster_point_count[cluster_idx];
            cluster_cells_map[i] = cluster_cells_map[cluster_idx];
            for (auto cell_idx : cluster_cells_map[i]) {
                cell_cluster_map[cell_idx] = i;
            }
        }
        
        for (auto cell_idx : sparse_cells) {
            bool merge_flag = false;
            int merge_cluster_idx = -1;
            for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
                merge_cluster_idx = i;
                if (!WhetherMerge(cell_bound_boxs, cluster_bound_box, cell_point_count, 
                    cluster_point_count, cell_idx, merge_cluster_idx, max_num_points)){
                    continue;
                }

                cell_cluster_map[cell_idx] = merge_cluster_idx;
                cluster_point_count[merge_cluster_idx] += cell_point_count[cell_idx];
                cluster_cells_map[merge_cluster_idx].push_back(cell_idx);
                cluster_bound_box[merge_cluster_idx] += cell_bound_boxs[cell_idx];
                cell_type_map[cell_idx] = 255;
                merge_flag = true;
                // std::cout << "merge: cell_idx: " << cell_idx 
                //         << "to Cluster " << merge_cluster_idx << std::endl;
                break;
            }
            
            if (!merge_flag){            
                std::size_t num_visited_points = 0;
                cluster_cells_map.push_back(std::vector<std::size_t>());
                auto &cluster_cells = cluster_cells_map.back();

                cell_cluster_map[cell_idx] = cluster_idx;
                cluster_cells.push_back(cell_idx);
                num_visited_points = cell_point_count[cell_idx];

                cluster_idx_map.push_back(cluster_idx);

                cluster_point_count.push_back(num_visited_points);
                cluster_bound_box.push_back(cell_bound_boxs[cell_idx]);
                cell_type_map[cell_idx] = 255;
                // std::cout << num_visited_points << " points clustered" << std::endl;  

                cluster_idx++;
                num_cluster = cluster_idx;
            }
        }

        num_cluster = cluster_idx_map.size();
        cluster_point_count.resize(num_cluster);
        cluster_bound_box.resize(num_cluster);

        cell_point_count.swap(cluster_point_count);
        cell_bound_boxs.swap(cluster_bound_box);

        cluster_point_count.clear();
        cluster_bound_box.clear();
        std::cout << "Merge Cell ( " << cell_num << " -> " << num_cluster << " ) ... Done!"  << std::endl;
    }

    // std::unordered_map<std::size_t, std::size_t> cell_2_cluster;
    std::vector<std::size_t> cell_2_cluster(num_cluster);
    std::vector<std::pair<std::size_t, std::size_t>> cell_id_num(num_cluster);
    for (std::size_t cell_idx = 0; cell_idx < num_cluster; cell_idx++){
        cell_id_num.at(cell_idx) = std::pair<std::size_t, std::size_t>(cell_idx, cell_point_count.at(cell_idx) );
    }
    std::sort(cell_id_num.begin(), cell_id_num.end(), 
        [](std::pair<std::size_t, std::size_t> a, std::pair<std::size_t, std::size_t> b){
            return a.second > b.second; });
    cluster_bound_box.resize(num_cluster);
    for(std::size_t idx = 0; idx < num_cluster; idx++){
        cell_2_cluster[cell_id_num.at(idx).first] = idx;
        cluster_bound_box[idx] = cell_bound_boxs.at(cell_id_num.at(idx).first);
    }
    {
        Timer cell2cluster_timer;
        cell2cluster_timer.Start();

        std::size_t point_num = ply_points.size();
        point_cluster_map.clear();
        point_cluster_map.resize(point_num);
        auto InsertCell = [&](std::size_t begin, std::size_t end){
            for (std::size_t i = begin; i < end; ++i) {
                for (auto id : point_cell_map[i]) {
                    point_cluster_map[i].push_back(std::move(cell_2_cluster[cell_cluster_map[id]]));
                }
            }
        };
        const int num_valid_thread = GetEffectiveNumThreads(-1);
        std::unique_ptr<ThreadPool> convert_thread_pool;
        convert_thread_pool.reset(new ThreadPool(num_valid_thread));

        std::size_t num_points_per_thread = ceil((float)point_num * 3 / (float)num_valid_thread);
        int num_convert_cluster = ceil((float)point_num / (float)num_points_per_thread);
        for (int i = 0; i < num_convert_cluster; i++){
            std::size_t begin = i * num_points_per_thread;
            std::size_t end = std::min((std::size_t)(i+1) * num_points_per_thread, (std::size_t)point_num);
            convert_thread_pool->AddTask(InsertCell, begin, end);
        }
        convert_thread_pool->Wait();

        std::cout << StringPrintf("Cell2Cluster Cost Time: %.3fs\n", cell2cluster_timer.ElapsedSeconds());
    }
    std::cout << StringPrintf("Point Fast %d CLuster End in %.3fs\n",  num_cluster, cluster_timer.ElapsedSeconds());
    return num_cluster;
}

void PointsSample(std::vector<PlyPoint>& points,
                  std::vector<float>& points_score,
                  std::vector<std::vector<uint32_t> >& points_views,
                  std::vector<std::vector<float> >& points_weights,
                  const Model& model,
                  const float distInsert,
                  const float diffDepth){
    std::cout << "Point Sample Start..." << std::endl;
    Timer samp_timer;
    samp_timer.Start();

    float begin_memroy, end_memory;
    GetAvailableMemory(begin_memroy);

    const uint64_t num_points = points.size();
    const uint64_t min_num_points = 2e6;
    bool is_score = (points_score.size() == points.size());
    bool is_weight = (points_weights.size() == points_views.size());

    int num_eff_threads = GetEffectiveNumThreads(-1);
    // std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    uint64_t max_num_points = std::max((uint64_t)(num_points / num_eff_threads + 1), min_num_points);
    float overlap_factor = 0.0f;
    // max_num_points = 5e5;
    std::cout << "max_num_points(num_points): " << max_num_points << "(" << num_points << ")" << std::endl;

    std::vector<std::vector<std::size_t> > point_cluster_map;
    std::vector<struct Box> cluster_bound_box;
    int num_cluster = PointsFastCluster(point_cluster_map, cluster_bound_box, 
                                    points, model, max_num_points, overlap_factor);

    Timer preproc_timer;
    preproc_timer.Start();
    std::cout << StringPrintf("Cluster Data Preprocess\n");
    std::vector<std::vector<uint32_t> > cluster_point_ids(num_cluster);
    std::vector<std::vector<PlyPoint>> cluster_points_ply(num_cluster);
    std::vector<std::vector<float>> cluster_points_score(num_cluster);
    std::vector<std::vector<std::vector<uint32_t>>> cluster_points_vis(num_cluster);
    std::vector<std::vector<std::vector<float>>> cluster_points_weight(num_cluster);
    int dou_points_num = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        for (auto id : point_cluster_map[i]){
            cluster_point_ids.at(id).push_back(i);
            dou_points_num++;
        }
    }
    std::cout << "repeat points num / ply points num: " 
                      << dou_points_num - points.size() 
                      << " / " << points.size() << std::endl;

    std::cout << StringPrintf("Cluster Data Preprocess Cost Time: %.3fs\n", preproc_timer.ElapsedSeconds());

    const auto& images = model.images;
    auto DelaunaySample = [&](int id){
        uint64_t num_cluster_point = cluster_point_ids.at(id).size();
        mvs::PointCloud pointcloud;
        {
            pointcloud.pointViews.Resize(num_cluster_point);
            if (is_weight){
                pointcloud.pointWeights.Resize(num_cluster_point);
            }
            for (size_t i = 0; i < num_cluster_point; ++i) {
                // if (i % 100 == 0) {
                //     std::cout << "\r" << i << " / " << num_cluster_point;
                // }
                auto point_id = cluster_point_ids.at(id)[i];
                const PlyPoint& pt = points.at(point_id);// cluster_points_ply.at(id)[i];
                const std::vector<uint32_t>& vis_pt = points_views.at(point_id);// cluster_points_vis.at(id)[i];
                Eigen::Vector3f p(&pt.x);
                pointcloud.points.Insert(p);
                pointcloud.normals.Insert(Eigen::Vector3f(&pt.nx));
                pointcloud.colors.Insert(Eigen::Vector3ub(&pt.r));
                pointcloud.pointTypes.Insert(pt.s_id);
                if (is_score){
                    pointcloud.scores.Insert(points_score.at(point_id));
                }
                // for (const auto& view : vis_pt) {                 
                for (size_t j = 0; j < vis_pt.size(); j++) {                 
                    pointcloud.pointViews[i].Insert(vis_pt[j]);
                }
                if (is_weight && points_weights.at(point_id).size() == vis_pt.size()){
                    const std::vector<float>& weight_pt = points_weights.at(point_id);
                    for (size_t j = 0; j < vis_pt.size(); j++) {                 
                        pointcloud.pointWeights[i].Insert(weight_pt[j]);
                    }
                }
            }
        }

        {
            delaunay_t delaunay;
            DelaunayInsert(pointcloud, delaunay, images, true, distInsert, diffDepth, 2.0f, -1.0f, true);
            delaunay.clear();
        }

        uint32_t num_delaunay_points = pointcloud.points.size();
        cluster_points_ply.at(id).reserve(num_delaunay_points);
        if (is_score){
            cluster_points_score.at(id).reserve(num_delaunay_points);
        }
        cluster_points_vis.at(id).reserve(num_delaunay_points);
        cluster_points_weight.at(id).reserve(num_delaunay_points);

        for (int i = 0; i < num_delaunay_points; i++){
            PlyPoint pnt;
            pnt.x = pointcloud.points[i].x();
            pnt.y = pointcloud.points[i].y();
            pnt.z = pointcloud.points[i].z();
            pnt.nx = pointcloud.normals[i].x();
            pnt.ny = pointcloud.normals[i].y();
            pnt.nz = pointcloud.normals[i].z();
            pnt.r = pointcloud.colors[i].x();
            pnt.g = pointcloud.colors[i].y();
            pnt.b = pointcloud.colors[i].z();
            pnt.s_id = (uint8_t)pointcloud.pointTypes[i];
            cluster_points_ply.at(id).push_back(std::move(pnt));

            const auto& pnt_views = pointcloud.pointViews[i];
            const auto& pnt_weights = pointcloud.pointWeights[i];

            int num_views = pnt_views.size();
            CHECK_EQ(num_views, pnt_weights.size());
            std::vector<uint32_t> views(num_views);
            std::vector<float> weights(num_views);
            float sum_weight = 0;
            for (int j = 0; j < num_views; j++){
                views[j] = pnt_views[j];
                weights[j] = pnt_weights[j];
                sum_weight += pnt_weights[j];
            }
            cluster_points_vis.at(id).push_back(std::move(views));
            cluster_points_weight.at(id).push_back(std::move(weights));
            if (is_score){
                float score = 0.0f;
                if (views.size() < 3){
                    float vis_score = 1.0 - std::min(sum_weight / 8.0f, 1.0f);
                    score = 0.7 * pointcloud.scores[i] + 0.3 * vis_score;
                }
                cluster_points_score.at(id).push_back(score);
            }
        }
        pointcloud.Release();
        // std::cout << StringPrintf("Cluster %d Delaunay Sample %d points -> %d points in %.3fs\n", 
        //                           id, num_delaunay_point, cluster_points_ply.at(id).size(), 
        //                           samp_timer.ElapsedSeconds());

        return;
    };

    {
        Timer dela_timer;
        dela_timer.Start();
        num_eff_threads = std::min(num_eff_threads, num_cluster);
        std::unique_ptr<ThreadPool> thread_pool;
        std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
        thread_pool.reset(new ThreadPool(num_eff_threads));
        for (int id = 0; id < num_cluster; id++){
            thread_pool->AddTask(DelaunaySample, id);
        }
        thread_pool->Wait();
        std::cout << "DelaunaySample Elapsed: " << dela_timer.ElapsedSeconds() << " (s)" << std::endl;
    }

    Timer data_move_timer;
    data_move_timer.Start();
    std::cout << StringPrintf("Cluster Data Move\n");

    points.clear();
    // points.shrink_to_fit();
    points_score.clear();
    // points_score.shrink_to_fit();
    points_views.clear();
    // points_views.shrink_to_fit();
    points_weights.clear();
    // points_weights.shrink_to_fit();
    for (int id = 0; id < num_cluster; id++){
        points.insert(points.end(), 
                      cluster_points_ply.at(id).begin(),
                      cluster_points_ply.at(id).end());
        std::vector<PlyPoint>().swap( cluster_points_ply.at(id));

        if (is_score){
            points_score.insert(points_score.end(), 
                                cluster_points_score.at(id).begin(),
                                cluster_points_score.at(id).end());
            std::vector<float>().swap(cluster_points_score.at(id));
        }

        points_views.insert(points_views.end(), 
                            cluster_points_vis.at(id).begin(),
                            cluster_points_vis.at(id).end());
        std::vector<std::vector<uint32_t>>().swap(cluster_points_vis.at(id));

        points_weights.insert(points_weights.end(), 
                            cluster_points_weight.at(id).begin(),
                            cluster_points_weight.at(id).end());
        std::vector<std::vector<float>>().swap(cluster_points_weight.at(id));
    }
    std::cout << StringPrintf("Move Data Cost Time: %.3fs\n", data_move_timer.ElapsedSeconds());

    {
        std::vector<std::vector<PlyPoint>>().swap(cluster_points_ply);
        std::vector<std::vector<float>>().swap(cluster_points_score);
        std::vector<std::vector<std::vector<uint32_t>>>().swap(cluster_points_vis);
        std::vector<std::vector<std::vector<float>>>().swap(cluster_points_weight);
    }

    GetAvailableMemory(end_memory);

    std::cout << "Point Sample , begin / end freem-memory :" 
            << begin_memroy << " / " << end_memory 
            << "\t max-memory: " 
            << end_memory - begin_memroy << std::endl;
    
    std::cout << StringPrintf("Point Sample %d points -> %d points in %.3fs\n", 
                            num_points, points.size(), samp_timer.ElapsedSeconds());
    return;
}

} // namespace mvs
} // namespace sensem
