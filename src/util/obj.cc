//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "obj.h"
#include "util/string.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>

// VCG: mesh reconstruction post-processing
#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/stat.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/smooth.h>
#include <vcg/complex/algorithms/hole.h>
#include <vcg/complex/algorithms/polygon_support.h>
// VCG: mesh simplification
#include <vcg/complex/algorithms/update/position.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/selection.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_items_with_id_3.h> 
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel, 
						CGAL::Polyhedron_items_with_id_3>   Polyhedron;
typedef Polyhedron::Halfedge_handle    						Halfedge_handle;
typedef Polyhedron::Facet_handle       						Facet_handle;
typedef Polyhedron::Vertex_handle      						Vertex_handle;
typedef Polyhedron::HalfedgeDS                              HalfedgeDS;

#define CGAL_EIGEN3_ENABLED

#ifdef _DEBUG

#ifdef _MSC_VER
#define _DEBUGINFO
#define _CRTDBG_MAP_ALLOC	//enable this to show also the filename (DEBUG_NEW should also be defined in each file)
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _INC_CRTDBG
#define ASSERT(exp)	{if (!(exp) && 1 == _CrtDbgReport(_CRT_ASSERT, __FILE__, __LINE__, NULL, #exp)) _CrtDbgBreak();}
#else
#define ASSERT(exp)	{if (!(exp)) __debugbreak();}
#endif // _INC_CRTDBG
#define TRACE(...) {TCHAR buffer[2048];	_sntprintf(buffer, 2048, __VA_ARGS__); OutputDebugString(buffer);}
#else // _MSC_VER
#include <assert.h>
#define ASSERT(exp)	assert(exp)
#define TRACE(...)
#endif // _MSC_VER

#else

#ifdef _RELEASE
#define ASSERT(exp)
#else
#ifdef _MSC_VER
#define ASSERT(exp) {if (!(exp)) __debugbreak();}
#else // _MSC_VER
#define ASSERT(exp) {if (!(exp)) __builtin_trap();}
#endif // _MSC_VER
#endif
#define TRACE(...)

#endif // _DEBUG

#ifndef MAX_LINE_LENGTH
#define MAX_LINE_LENGTH 512
#endif

namespace sensemap {
namespace {
// A modifier creating a triangle with the incremental builder.
template <class HDS>
class Build_triangle : public CGAL::Modifier_base<HDS> {
public:
    Build_triangle(const TriangleMesh& mesh) : mesh_(mesh) {}
    void operator()( HDS& hds) {
        // Postcondition: hds is a valid polyhedral surface.
        CGAL::Polyhedron_incremental_builder_3<HDS> B( hds, true);

        typedef typename HDS::Vertex   Vertex;
        typedef typename Vertex::Point VPoint;

        B.begin_surface(mesh_.vertices_.size(), mesh_.faces_.size());
        for (auto vert : mesh_.vertices_) {
			B.add_vertex(VPoint(vert[0], vert[1], vert[2]));
        }
        for (auto facet : mesh_.faces_) {
            B.begin_facet();
            B.add_vertex_to_facet(facet[0]);
            B.add_vertex_to_facet(facet[1]);
            B.add_vertex_to_facet(facet[2]);
            B.end_facet();
        }
        B.end_surface();
    }
private:
    TriangleMesh mesh_;
};

int Find(int x, std::vector<int>& pre) {
    int r = x;
    while(pre[r] != r) {
        r = pre[r];
    }
    // 路径压缩
    int i = x, j;
    while(i != r) {
        j = pre[i];
        pre[i] = r;
        i = j;
    }
    return r;
}
void UnionSet(int x, int y, std::vector<int>& pre, std::vector<int>& rank) {
    int fx = Find(x, pre);
    int fy = Find(y, pre);
    if (fx != fy) {
        if (rank[fx] >= rank[fy]) {
            pre[fy] = fx;
            rank[fx] += rank[fy];
        } else {
            pre[fx] = fy;
            rank[fy] += rank[fx];
        }
    }
}
}

void TriangleMesh::Clear() {
    vertices_.clear();
    vertices_.shrink_to_fit();
    vertex_normals_.clear();
    vertex_normals_.shrink_to_fit();
    vertex_colors_.clear();
    vertex_colors_.shrink_to_fit();
    faces_.clear();
    faces_.shrink_to_fit();
    face_normals_.clear();
    face_normals_.shrink_to_fit();
    vertex_labels_.clear();
    vertex_labels_.shrink_to_fit();
    vertex_status_.clear();
    vertex_status_.shrink_to_fit();
    vertex_visibilities_.clear();
    vertex_visibilities_.shrink_to_fit();
}

void TriangleMesh::RemoveIsolatedPieces(const int min_num_facet) {
    const int num_vertex = vertices_.size();
    const int num_facet = faces_.size();
    std::vector<int> rank(num_vertex, 1);
    std::vector<int> pre(num_vertex);
    std::generate(pre.begin(), pre.end(), [n = 0]() mutable {
        return n++;
    });
    for (const auto& facet : faces_) {
        UnionSet(facet[0], facet[1], pre, rank);
        UnionSet(facet[0], facet[2], pre, rank);
        UnionSet(facet[1], facet[2], pre, rank);
    }

    bool has_color = vertex_colors_.size() != 0;
	bool has_status = vertex_status_.size() != 0;

    int i, j;
    std::vector<int> vtx_map(num_vertex, -1);
    for (i = 0, j = 0; i < num_vertex; ++i) {
        int r = Find(i, pre);
        if (rank[r] > min_num_facet) {
            vertices_[j] = vertices_[i];
            // vertex_normals_[j] = vertex_normals_[i];
            if (has_color) {
                vertex_colors_[j] = vertex_colors_[i];
            }
			if (has_status) {
				vertex_status_[j] = vertex_status_[i];
			}
            vtx_map[i] = j;
            j++;
        }
    }
    vertices_.resize(j);
    if (has_color) {
        vertex_colors_.resize(j);
    }
	if (has_status) {
		vertex_status_.resize(j);
	}

    for (i = 0, j = 0; i < num_facet; ++i) {
        Eigen::Vector3i facet = faces_[i];
        if (vtx_map[facet[0]] == -1 ||
            vtx_map[facet[1]] == -1 ||
            vtx_map[facet[2]] == -1) {
            continue;
        }
        facet[0] = vtx_map[facet[0]];
        facet[1] = vtx_map[facet[1]];
        facet[2] = vtx_map[facet[2]];
        faces_[j] = facet;
        j++;
    }
    faces_.resize(j);
    std::cout << "Remove " << num_vertex - vertices_.size() 
              << " isolated vertices" << std::endl;
}

void TriangleMesh::RemoveAbnormalFacets(std::vector<int>& border_verts_idx) {
	std::map<std::pair<int, int>, int> shared_edges;
	for (const auto& facet : faces_) {
		int v0 = facet.x();
		int v1 = facet.y();
		int v2 = facet.z();
		std::pair<int, int> edge01 = std::make_pair(v0, v1);
		shared_edges[edge01]++;
		std::pair<int, int> edge12 = std::make_pair(v1, v2);
		shared_edges[edge12]++;
		std::pair<int, int> edge20 = std::make_pair(v2, v0);
		shared_edges[edge20]++;
	}

	std::vector<unsigned char> bad_verts_flag(vertices_.size(), 0);
	for (auto edge : shared_edges) {
		if (edge.second != 1) {
			bad_verts_flag[edge.first.first] = 1;
			bad_verts_flag[edge.first.second] = 1;
		}
	}

	// Detect border vertices of the hole.
	std::set<int> border_verts_idx_set;
	for (auto& facet : faces_) {
		if (!bad_verts_flag.at(facet[0]) &&
			!bad_verts_flag.at(facet[1]) &&
			!bad_verts_flag.at(facet[2])) {
			continue;
		}
		if (!bad_verts_flag.at(facet[0])) {
			border_verts_idx_set.insert(facet[0]);
		}
		if (!bad_verts_flag.at(facet[1])) {
			border_verts_idx_set.insert(facet[1]);
		}
		if (!bad_verts_flag.at(facet[2])) {
			border_verts_idx_set.insert(facet[2]);
		}
	}
	border_verts_idx.clear();
	std::copy(border_verts_idx_set.begin(), border_verts_idx_set.end(), 
		std::back_inserter(border_verts_idx));

	const bool has_color = vertex_colors_.size() != 0;
	const bool has_normal = vertex_normals_.size() != 0;
	const bool has_label = vertex_labels_.size() != 0;

	int num_vert = vertices_.size();
	int num_facet = faces_.size();

	int i, j;
	std::vector<int> vert_idx_map(num_vert, -1);
	for (i = 0, j = 0; i < num_vert; ++i) {
		if (!bad_verts_flag.at(i)) {
			vertices_[j] = vertices_[i];
			if (has_normal) {
				vertex_normals_[j] = vertex_normals_[i];
			}
			if (has_color) {
				vertex_colors_[j] = vertex_colors_[i];
			}
			if (has_label) {
				vertex_labels_[j] = vertex_labels_[i];
			}

			vert_idx_map[i] = j;
			j = j + 1;
		}
	}
	vertices_.resize(j);
	if (has_normal) {
		vertex_normals_.resize(j);
	}
	if (has_color) {
		vertex_colors_.resize(j);
	}
	if (has_label) {
		vertex_labels_.resize(j);
	}

	// Update border vertex idx.
	for (auto& idx : border_verts_idx) {
		idx = vert_idx_map[idx];
	}

	std::vector<int> ref_vert_idxs(vertices_.size(), -1);
	for (i = 0, j = 0; i < num_facet; ++i) {
		auto facet = faces_.at(i);
		int v0 = facet.x();
		int v1 = facet.y();
		int v2 = facet.z();
		if (vert_idx_map.at(v0) == -1 || 
			vert_idx_map.at(v1) == -1 ||
			vert_idx_map.at(v2) == -1) {
			continue;
		}
		faces_.at(j) = facet;
		faces_.at(j).x() = vert_idx_map[v0];
		ref_vert_idxs.at(vert_idx_map[v0]) = 1;
		faces_.at(j).y() = vert_idx_map[v1];
		ref_vert_idxs.at(vert_idx_map[v1]) = 1;
		faces_.at(j).z() = vert_idx_map[v2];
		ref_vert_idxs.at(vert_idx_map[v2]) = 1;
		j = j + 1;
	}
	faces_.resize(j);
	std::cout << 
		StringPrintf("Detect %d bad vertices!\n", num_vert - vertices_.size());

	// Remove unreference vertex.
	num_vert = vertices_.size();
	num_facet = faces_.size();
	std::fill(vert_idx_map.begin(), vert_idx_map.end(), -1);
	for (i = 0, j = 0; i < num_vert; ++i) {
		if (ref_vert_idxs.at(i) != -1) {
			vertices_[j] = vertices_[i];
			if (has_normal) {
				vertex_normals_[j] = vertex_normals_[i];
			}
			if (has_color) {
				vertex_colors_[j] = vertex_colors_[i];
			}
			if (has_label) {
				vertex_labels_[j] = vertex_labels_[i];
			}

			vert_idx_map[i] = j;
			j = j + 1;
		}
	}
	vertices_.resize(j);
	if (has_normal) {
		vertex_normals_.resize(j);
	}
	if (has_color) {
		vertex_colors_.resize(j);
	}
	if (has_label) {
		vertex_labels_.resize(j);
	}
	for (i = 0, j = 0; i < num_facet; ++i) {
		auto facet = faces_.at(i);
		int v0 = facet.x();
		int v1 = facet.y();
		int v2 = facet.z();
		if (vert_idx_map.at(v0) == -1 || 
			vert_idx_map.at(v1) == -1 ||
			vert_idx_map.at(v2) == -1) {
			continue;
		}
		faces_.at(j) = facet;
		faces_.at(j).x() = vert_idx_map[v0];
		faces_.at(j).y() = vert_idx_map[v1];
		faces_.at(j).z() = vert_idx_map[v2];
		j = j + 1;
	}
	faces_.resize(j);

	std::cout << StringPrintf(
			"Detect %d unreference vertices", num_vert - vertices_.size()) 
		      << std::endl;

	for (i = 0, j = 0; i < border_verts_idx.size(); ++i) {
		auto idx = border_verts_idx.at(i);
		if (vert_idx_map.at(idx) == -1) {
			continue;
		}
		border_verts_idx.at(j) = vert_idx_map.at(idx);
		j = j + 1;
	}
	border_verts_idx.resize(j);
}
void TriangleMesh::RemoveSelectFaces(const std::set<size_t>& face_set){

     // save filtered mesh
    std::vector<std::size_t > filtered_faces;
    for (int i = 0; i < faces_.size(); i++){
        if (face_set.find(i) != face_set.end()){
            continue;
        }
        filtered_faces.push_back(i);
    }

    TriangleMesh filtered_mesh;
    filtered_mesh.vertices_.reserve(vertices_.size());
    filtered_mesh.vertex_normals_.reserve(vertex_normals_.size());
    filtered_mesh.vertex_colors_.reserve(vertex_colors_.size());
    filtered_mesh.vertex_labels_.reserve(vertex_labels_.size());
    filtered_mesh.vertex_status_.reserve(vertex_status_.size());
    filtered_mesh.vertex_visibilities_.reserve(vertex_visibilities_.size());
    filtered_mesh.faces_.reserve(faces_.size());
    filtered_mesh.face_normals_.reserve(face_normals_.size());

    std::size_t vertex_num = vertices_.size();
    for (auto face_id : filtered_faces) {
        filtered_mesh.faces_.push_back(faces_[face_id]);
    }
    if (!face_normals_.empty()) {
        for (auto face_id : filtered_faces) {
            filtered_mesh.face_normals_.push_back(face_normals_[face_id]);
        }
    }
    std::vector<int> vertex_idx_map(vertex_num, -1);
    for (auto face_id : filtered_faces) {
        const auto &face = faces_[face_id];
        vertex_idx_map[face[0]] = 0;
        vertex_idx_map[face[1]] = 0;
        vertex_idx_map[face[2]] = 0;
    }
    for (std::size_t i = 0; i < vertex_num; i++) {
        if (vertex_idx_map[i] == -1) {
            continue;
        }
        vertex_idx_map[i] = filtered_mesh.vertices_.size();
        filtered_mesh.vertices_.push_back(vertices_[i]);
    }
    if (!vertex_normals_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }
            filtered_mesh.vertex_normals_.push_back(vertex_normals_[i]);
        }
    }
    if (!vertex_colors_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }
            filtered_mesh.vertex_colors_.push_back(vertex_colors_[i]);
        }
    }
    if (!vertex_labels_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }
            filtered_mesh.vertex_labels_.push_back(vertex_labels_[i]);
        }
    }
    if (!vertex_status_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }
            filtered_mesh.vertex_status_.push_back(vertex_status_[i]);
        }
    }
    if (!vertex_visibilities_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }
            filtered_mesh.vertex_visibilities_.push_back(vertex_visibilities_[i]);
        }
    }
    filtered_mesh.vertices_.shrink_to_fit();
    filtered_mesh.vertex_normals_.shrink_to_fit();
    filtered_mesh.vertex_colors_.shrink_to_fit();
    filtered_mesh.vertex_labels_.shrink_to_fit();
    filtered_mesh.vertex_status_.shrink_to_fit();
    filtered_mesh.vertex_visibilities_.shrink_to_fit();
    filtered_mesh.faces_.shrink_to_fit();
    filtered_mesh.face_normals_.shrink_to_fit();
    for (auto &face : filtered_mesh.faces_) {
        face[0] = vertex_idx_map[face[0]];
        face[1] = vertex_idx_map[face[1]];
        face[2] = vertex_idx_map[face[2]];
    }
    Swap(filtered_mesh);
    return;
}

void TriangleMesh::HollFill() {
    std::cout << "Contruct Polyhedron" << std::endl;
    // Contruct Polyhedron.
    Polyhedron P;
    Build_triangle<HalfedgeDS> poly(*this);
    P.delegate(poly);

    int vert_id = 0;
    for (Polyhedron::Vertex_iterator it = P.vertices_begin();
         it != P.vertices_end(); ++it) {
        it->id() = vert_id++;
    }

    int facet_id = 0;
    for (Polyhedron::Facet_iterator it = P.facets_begin();
         it != P.facets_end(); ++it) {
        it->id() = facet_id++;
    }

    // Incrementally fill the holes
    unsigned int nb_holes = 0;
    for(Halfedge_handle h : CGAL::halfedges(P)) {
        if(h->is_border()) {
            std::vector<Facet_handle>  patch_facets;
            std::vector<Vertex_handle> patch_vertices;
            bool success = std::get<0>(
                    CGAL::Polygon_mesh_processing::triangulate_refine_and_fair_hole(
                            P, h,
                            std::back_inserter(patch_facets),
                            std::back_inserter(patch_vertices),
                            CGAL::Polygon_mesh_processing::parameters::vertex_point_map(CGAL::get(CGAL::vertex_point, P)).geom_traits(Kernel())) );
            std::cout << " Number of facets in constructed patch: " << patch_facets.size() << std::endl;
            std::cout << " Number of vertices in constructed patch: " << patch_vertices.size() << std::endl;
            std::cout << " Fairing : " << (success ? "succeeded" : "failed") << std::endl;
            ++nb_holes;
        }
    }
    std::cout << std::endl;
    std::cout << nb_holes << " holes have been filled" << std::endl;

    // std::ofstream out("filled.off", std::ofstream::out);
    // out << P << std::endl;

    faces_.clear();
    face_normals_.clear();
    typedef typename Polyhedron::Facet_iterator Facet_iterator;
    for (Facet_iterator f = P.facets_begin(); f != P.facets_end(); ++f) {
        Polyhedron::Halfedge_around_facet_circulator circ = f->facet_begin();
        Eigen::Vector3i facet;
        int i = 0;
        do {
            int id = circ->vertex()->id();
            if (id < vertices_.size()) {
                facet[i++] = id;
            }
        } while(++circ != f->facet_begin());
        faces_.emplace_back(facet);
    }
}


void TriangleMesh::HollFill(std::vector<std::vector<int> > &lists) {

    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef Kernel::Point_3 Point;

    for(auto &list : lists) {
        std::vector<Point> polyline;
        for(auto &id : list){
            auto vert = vertices_.at(id);
            polyline.push_back(Point(vert[0], vert[1], vert[2]));
        }

        if(polyline.size() < 4)
            continue;

        // any type, having Type(int, int, int) constructor available, can be used to hold output triangles
        typedef CGAL::Triple<int, int, int> Triangle_int;
        std::vector<Triangle_int> patch;
        patch.reserve(polyline.size() - 2); // there will be exactly n-2 triangles in the patch
        CGAL::Polygon_mesh_processing::triangulate_hole_polyline(
                polyline,
                std::back_inserter(patch));

        for (std::size_t i = 0; i < patch.size(); ++i) {
//            std::cout << "Triangle " << i << ": "
//                      << patch[i].first << " " << patch[i].second << " " << patch[i].third
//                      << std::endl;

            this->faces_.push_back(Eigen::Vector3i(list[patch[i].third], list[patch[i].second], list[patch[i].first]));
        }
    }
}

/*----------------------------------------------------------------*/

namespace CLEAN {
// define mesh type
class Vertex; class Edge; class Face;
struct UsedTypes : public vcg::UsedTypes<
	vcg::Use<Vertex>::AsVertexType,
	vcg::Use<Edge>  ::AsEdgeType,
	vcg::Use<Face>  ::AsFaceType   > {};

class Vertex : public vcg::Vertex<UsedTypes, vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::VFAdj, vcg::vertex::Mark, vcg::vertex::BitFlags, vcg::vertex::Color4b> {};
class Face   : public vcg::Face<  UsedTypes, vcg::face::VertexRef, vcg::face::Normal3f, vcg::face::FFAdj, vcg::face::VFAdj, vcg::face::Mark, vcg::face::BitFlags> {};
class Edge   : public vcg::Edge<  UsedTypes, vcg::edge::VertexRef> {};

class Mesh : public vcg::tri::TriMesh< std::vector<Vertex>, std::vector<Face>, std::vector<Edge> > {};

// decimation helper classes
typedef	vcg::SimpleTempData< Mesh::VertContainer, vcg::math::Quadric<double> > QuadricTemp;

class QHelper
{
public:
	QHelper() {}
	static void Init() {}
	static vcg::math::Quadric<double> &Qd(Vertex &v) { return TD()[v]; }
	static vcg::math::Quadric<double> &Qd(Vertex *v) { return TD()[*v]; }
	static Vertex::ScalarType W(Vertex * /*v*/) { return 1.0; }
	static Vertex::ScalarType W(Vertex & /*v*/) { return 1.0; }
	static void Merge(Vertex & /*v_dest*/, Vertex const & /*v_del*/) {}
	static QuadricTemp* &TDp() { static QuadricTemp *td; return td; }
	static QuadricTemp &TD() { return *TDp(); }
};

typedef BasicVertexPair<Vertex> VertexPair;

class TriEdgeCollapse : public vcg::tri::TriEdgeCollapseQuadric<Mesh, VertexPair, TriEdgeCollapse, QHelper> {
public:
	typedef vcg::tri::TriEdgeCollapseQuadric<Mesh, VertexPair, TriEdgeCollapse, QHelper> TECQ;
	inline TriEdgeCollapse(const VertexPair &p, int i, vcg::BaseParameterClass *pp) :TECQ(p, i, pp) {}
};
}

// decimate, clean and smooth mesh
// fDecimate factor is in range (0..1], if 1 no decimation takes place
void TriangleMesh::Clean(const float fDecimate, const float fSpurious, 
                         const bool bRemoveSpikes, const unsigned nCloseHoles, 
                         const unsigned nSmooth, const bool bLastClean)
{
	if (vertices_.empty() || faces_.empty())
		return;
	// TD_TIMER_STARTD();
	// create VCG mesh
    std::cout << "Clean begin (fSpurious: " << fSpurious << ")" << std::endl;

    bool has_label = !vertex_labels_.empty();
	CLEAN::Mesh mesh;
	{
		CLEAN::Mesh::VertexIterator vi = vcg::tri::Allocator<CLEAN::Mesh>::AddVertices(mesh, vertices_.size());
		// for (const auto& pVert : vertices_) {
		for (int i = 0; i < vertices_.size(); ++i) {
			// const Vertex& p(*pVert);
            const Eigen::Vector3d& p = vertices_[i];
			const Eigen::Vector3d& c = vertex_colors_[i];
            const int8_t s_id = has_label ? vertex_labels_[i] : -1;
			CLEAN::Vertex::CoordType& P((*vi).P());
			CLEAN::Vertex::ColorType& C((*vi).C());
			P[0] = p.x();
			P[1] = p.y();
			P[2] = p.z();
			C[0] = c.x();
			C[1] = c.y();
			C[2] = c.z();
            C[3] = s_id;
			++vi;
		}
		// vertices.Release();
        vertices_.clear();
		vertex_colors_.clear();
        vertex_labels_.clear();
		vi = mesh.vert.begin();
		std::vector<CLEAN::Mesh::VertexPointer> indices(mesh.vert.size());
		for (CLEAN::Mesh::VertexPointer& idx: indices) {
			idx = &*vi;
			++vi;
		}
		CLEAN::Mesh::FaceIterator fi = vcg::tri::Allocator<CLEAN::Mesh>::AddFaces(mesh, faces_.size());
		for (const auto& pFace : faces_) {
			// const Face& f(*pFace);
            const Eigen::Vector3i& f(pFace);
			ASSERT((*fi).VN() == 3);
			ASSERT(f[0]<(uint32_t)mesh.vn);
			(*fi).V(0) = indices[f[0]];
			ASSERT(f[1]<(uint32_t)mesh.vn);
			(*fi).V(1) = indices[f[1]];
			ASSERT(f[2]<(uint32_t)mesh.vn);
			(*fi).V(2) = indices[f[2]];
			++fi;
		}
		// faces.Release();
        faces_.clear();
	}

	// decimate mesh
	if (fDecimate < 1) {
		ASSERT(fDecimate > 0);
		vcg::tri::TriEdgeCollapseQuadricParameter pp;
		pp.QualityThr = 0.3; // Quality Threshold for penalizing bad shaped faces: the value is in the range [0..1], 0 accept any kind of face (no penalties), 0.5 penalize faces with quality < 0.5, proportionally to their shape
		pp.PreserveBoundary = true; // the simplification process tries to not affect mesh boundaries during simplification
		pp.BoundaryWeight = 1; // the importance of the boundary during simplification: the value is in the range (0..+inf), default (1.0) means that the boundary has the same importance as the rest; values greater than 1.0 raise boundary importance and has the effect of removing less vertices on the border
		pp.PreserveTopology = true; // avoid all collapses that cause a topology change in the mesh (like closing holes, squeezing handles, etc); if checked the genus of the mesh should stay unchanged
		pp.QualityWeight = false; // use the Per-Vertex quality as a weighting factor for the simplification: the weight is used as an error amplification value, so a vertex with a high quality value will not be simplified and a portion of the mesh with low quality values will be aggressively simplified
		pp.NormalCheck = false; // try to avoid face flipping effects and try to preserve the original orientation of the surface
		pp.OptimalPlacement = true; // each collapsed vertex is placed in the position minimizing the quadric error; it can fail (creating bad spikes) in case of very flat areas; if disabled edges are collapsed onto one of the two original vertices and the final mesh is composed by a subset of the original vertices
		pp.QualityQuadric = true; // add additional simplification constraints that improves the quality of the simplification of the planar portion of the mesh
		// decimate
		vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
		vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromVF(mesh);
		const int TargetFaceNum(std::floor(fDecimate * mesh.fn + 0.5f));
		vcg::math::Quadric<double> QZero;
		QZero.SetZero();
		CLEAN::QuadricTemp TD(mesh.vert, QZero);
		CLEAN::QHelper::TDp()=&TD;
		// if (pp.PreserveBoundary) {
		// 	pp.FastPreserveBoundary = true;
		// 	pp.PreserveBoundary = false;
		// }
		if (pp.NormalCheck)
			pp.NormalThrRad = M_PI/4.0;
		const int OriginalFaceNum(mesh.fn);
        printf("Decimated faces %d\n", OriginalFaceNum-TargetFaceNum);
		vcg::LocalOptimization<CLEAN::Mesh> DeciSession(mesh, &pp);
		DeciSession.Init<CLEAN::TriEdgeCollapse>();
		DeciSession.SetTargetSimplices(TargetFaceNum);
		DeciSession.SetTimeBudget(0.1f); // this allow to update the progress bar 10 time for sec...
		while (mesh.fn>TargetFaceNum && DeciSession.DoOptimization())
            printf("\r%d", OriginalFaceNum - mesh.fn);
		DeciSession.Finalize<CLEAN::TriEdgeCollapse>();
		printf("Mesh decimated: %d -> %d faces\n", OriginalFaceNum, TargetFaceNum);
	}

	// clean mesh
	{
        vcg::tri::Allocator<CLEAN::Mesh>::CompactFaceVector(mesh);
		vcg::tri::Allocator<CLEAN::Mesh>::CompactVertexVector(mesh);
		for (int i=0; i<10; ++i) {
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
			const int nSplitNonManifoldVertices = vcg::tri::Clean<CLEAN::Mesh>::SplitNonManifoldVertex(mesh, 0.1f);
			printf("Split %d non-manifold vertices\n", nSplitNonManifoldVertices);
			if (nSplitNonManifoldVertices == 0)
				break;
		}

		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		int nZeroAreaFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveZeroAreaFace(mesh);
		printf("Removed %d zero-area faces\n", nZeroAreaFaces);
		int nDuplicateFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveDuplicateFace(mesh);
		printf("Removed %d duplicate faces\n", nDuplicateFaces);
		int nNonManifoldFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveNonManifoldFace(mesh);
		printf("Removed %d non-manifold faces\n", nNonManifoldFaces);
		int nDegenerateFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveDegenerateFace(mesh);
		printf("Removed %d degenerate faces\n", nDegenerateFaces);
		int nNonManifoldVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveNonManifoldVertex(mesh);
		printf("Removed %d non-manifold vertices\n", nNonManifoldVertices);
		int nDegenerateVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveDegenerateVertex(mesh);
		printf("Removed %d degenerate vertices\n", nDegenerateVertices);
		int nDuplicateVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveDuplicateVertex(mesh);
		printf("Removed %d duplicate vertices\n", nDuplicateVertices);
		int nUnreferencedVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
		printf("Removed %d unreferenced vertices\n", nUnreferencedVertices);
		#if 1
		vcg::tri::Allocator<CLEAN::Mesh>::CompactFaceVector(mesh);
		vcg::tri::Allocator<CLEAN::Mesh>::CompactVertexVector(mesh);
		for (int i=0; i<10; ++i) {
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
			const int nSplitNonManifoldVertices = vcg::tri::Clean<CLEAN::Mesh>::SplitNonManifoldVertex(mesh, 0.1f);
			printf("Split %d non-manifold vertices\n", nSplitNonManifoldVertices);
			if (nSplitNonManifoldVertices == 0)
				break;
		}
		nZeroAreaFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveZeroAreaFace(mesh);
		printf("Removed %d zero-area faces\n", nZeroAreaFaces);
		nDuplicateFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveDuplicateFace(mesh);
		printf("Removed %d duplicate faces\n", nDuplicateFaces);
		nNonManifoldFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveNonManifoldFace(mesh);
		printf("Removed %d non-manifold faces\n", nNonManifoldFaces);
		nDegenerateFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveDegenerateFace(mesh);
		printf("Removed %d degenerate faces\n", nDegenerateFaces);
		nNonManifoldVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveNonManifoldVertex(mesh);
		printf("Removed %d non-manifold vertices\n", nNonManifoldVertices);
		nDegenerateVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveDegenerateVertex(mesh);
		printf("Removed %d degenerate vertices\n", nDegenerateVertices);
		nDuplicateVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveDuplicateVertex(mesh);
		printf("Removed %d duplicate vertices\n", nDuplicateVertices);
		nUnreferencedVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
		printf("Removed %d unreferenced vertices\n", nUnreferencedVertices);
		#else
		const int nNonManifoldVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveNonManifoldVertex(mesh);
		printf("Removed %d non-manifold vertices\n", nNonManifoldVertices);
		vcg::tri::Allocator<CLEAN::Mesh>::CompactFaceVector(mesh);
		vcg::tri::Allocator<CLEAN::Mesh>::CompactVertexVector(mesh);
		#endif
		vcg::tri::UpdateTopology<CLEAN::Mesh>::AllocateEdge(mesh);
	}

	// remove spurious components
	if (fSpurious > 0) {
		// FloatArr edgeLens(0, mesh.EN());
		std::vector<float> edgeLens;
		for (CLEAN::Mesh::EdgeIterator ei=mesh.edge.begin(); ei!=mesh.edge.end(); ++ei) {
			const CLEAN::Vertex::CoordType& P0((*ei).V(0)->P());
			const CLEAN::Vertex::CoordType& P1((*ei).V(1)->P());
			edgeLens.emplace_back((P1-P0).Norm());
		}
		#if 0
		const auto ret(ComputeX84Threshold<float,float>(edgeLens.Begin(), edgeLens.GetSize(), 3.f*fSpurious));
		const float thLongEdge(ret.first+ret.second);
		#else
		// const float thLongEdge(edgeLens.GetNth(edgeLens.GetSize()*95/100)*fSpurious);
        int nth = edgeLens.size() * 99 / 100;
		std::nth_element(edgeLens.begin(), edgeLens.begin() + nth, edgeLens.end());
		const float thLongEdge(edgeLens[nth] * fSpurious);
		#endif
		// remove faces with too long edges
		const size_t numLongFaces(vcg::tri::UpdateSelection<CLEAN::Mesh>::FaceOutOfRangeEdge(mesh, 0, thLongEdge));
		for (CLEAN::Mesh::FaceIterator fi=mesh.face.begin(); fi!=mesh.face.end(); ++fi)
			if (!(*fi).IsD() && (*fi).IsS())
				vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fi);
		printf("Removed %d faces with edges longer than %f\n", numLongFaces, thLongEdge);
		// remove isolated components
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		const std::pair<int, int> delInfo(vcg::tri::Clean<CLEAN::Mesh>::RemoveSmallConnectedComponentsDiameter(mesh, thLongEdge));
		printf("Removed %d connected components out of %d\n", delInfo.second, delInfo.first);
	}

	// remove spikes
	if (bRemoveSpikes) {
		int nTotalSpikes(0);
		vcg::tri::RequireVFAdjacency(mesh);
		while (true) {
			if (fSpurious <= 0 || nTotalSpikes != 0)
				vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
			int nSpikes(0);
			for (CLEAN::Mesh::VertexIterator vi=mesh.vert.begin(); vi!=mesh.vert.end(); ++vi) {
				if (vi->IsD())
					continue;
				CLEAN::Face* const start(vi->cVFp());
				if (start == NULL) {
					vcg::tri::Allocator<CLEAN::Mesh>::DeleteVertex(mesh, *vi);
					continue;
				}
				vcg::face::JumpingPos<CLEAN::Face> p(start, vi->cVFi(), &*vi);
				int count(0);
				do {
					++count;
					p.NextFE();
				} while (p.f!=start);
				if (count == 1) {
					vcg::tri::Allocator<CLEAN::Mesh>::DeleteVertex(mesh, *vi);
					++nSpikes;
				}
			}
			if (nSpikes == 0)
				break;
			for (CLEAN::Mesh::FaceIterator fi=mesh.face.begin(); fi!=mesh.face.end(); ++fi) {
				if (!fi->IsD() &&
					(fi->V(0)->IsD() ||
					 fi->V(1)->IsD() ||
					 fi->V(2)->IsD()))
					vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fi);
			}
			nTotalSpikes += nSpikes;
		}
		printf("Removed %d spikes\n", nTotalSpikes);
	}

	// close holes
	if (nCloseHoles > 0) {
		if (fSpurious <= 0 && !bRemoveSpikes)
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		vcg::tri::UpdateNormal<CLEAN::Mesh>::PerFaceNormalized(mesh);
		vcg::tri::UpdateNormal<CLEAN::Mesh>::PerVertexAngleWeighted(mesh);
		ASSERT(vcg::tri::Clean<CLEAN::Mesh>::CountNonManifoldEdgeFF(mesh) == 0);
		const int OriginalSize(mesh.fn);
		#if 1
		// When closing holes it tries to prevent the creation of faces that intersect faces adjacent to
		// the boundary of the hole. It is an heuristic, non intersecting hole filling can be NP-complete.
		const int holeCnt(vcg::tri::Hole<CLEAN::Mesh>::EarCuttingIntersectionFill< vcg::tri::SelfIntersectionEar<CLEAN::Mesh> >(mesh, (int)nCloseHoles, false));
		#else
		const int holeCnt = vcg::tri::Hole<CLEAN::Mesh>::EarCuttingFill< vcg::tri::MinimumWeightEar<CLEAN::Mesh> >(mesh, (int)nCloseHoles, false);
		#endif
		printf("Closed %d holes and added %d new faces\n", holeCnt, mesh.fn-OriginalSize);
	}

	// smooth mesh
	if (nSmooth > 0) {
		if (fSpurious <= 0 && !bRemoveSpikes && nCloseHoles <= 0)
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
		#if 1
		// vcg::tri::Smooth<CLEAN::Mesh>::VertexCoordLaplacian(mesh, (int)nSmooth, false, false);
        vcg::tri::Smooth<CLEAN::Mesh>::VertexCoordTaubin(mesh, (int)nSmooth, 0.5, -0.53);
		#else
		vcg::tri::Smooth<CLEAN::Mesh>::VertexCoordLaplacianHC(mesh, (int)nSmooth, false);
		#endif
		printf("Smoothed %d vertices\n", mesh.vn);
	}

	// clean mesh
	if (bLastClean && (fSpurious > 0 || bRemoveSpikes || nCloseHoles > 0 || nSmooth > 0)) {
		const int nNonManifoldFaces = vcg::tri::Clean<CLEAN::Mesh>::RemoveNonManifoldFace(mesh);
		printf("Removed %d non-manifold faces\n", nNonManifoldFaces);
		#if 0 // not working
		vcg::tri::Allocator<CLEAN::Mesh>::CompactEveryVector(mesh);
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
		const int nSplitNonManifoldVertices = vcg::tri::Clean<CLEAN::Mesh>::SplitNonManifoldVertex(mesh, 0.1f);
		printf("Split %d non-manifold vertices", nSplitNonManifoldVertices);
		#else
		const int nNonManifoldVertices = vcg::tri::Clean<CLEAN::Mesh>::RemoveNonManifoldVertex(mesh);
		printf("Removed %d non-manifold vertices\n", nNonManifoldVertices);
		#endif
	}

	// import VCG mesh
	{
		ASSERT(vertices_.empty() && faces_.empty());
		// vertices.Reserve(mesh.VN());
		vertices_.reserve(mesh.VN());
		vertex_colors_.reserve(mesh.VN());
        vertex_labels_.reserve(mesh.VN());
		vcg::SimpleTempData<CLEAN::Mesh::VertContainer, int> indices(mesh.vert);
		int idx(0);
		for (CLEAN::Mesh::VertexIterator vi=mesh.vert.begin(); vi!=mesh.vert.end(); ++vi) {
			if (vi->IsD())
				continue;
			// Vertex& p(vertices.AddEmpty());
            Eigen::Vector3d p, c;
			const CLEAN::Vertex::CoordType& P((*vi).P());
			const CLEAN::Vertex::ColorType& C((*vi).C());
			p[0] = P[0];
			p[1] = P[1];
			p[2] = P[2];
			c[0] = C[0];
			c[1] = C[1];
			c[2] = C[2];
            vertices_.emplace_back(p);
			vertex_colors_.emplace_back(c);
            vertex_labels_.emplace_back(C[3]);
			indices[vi] = idx++;
		}
		faces_.reserve(mesh.FN());
		for (CLEAN::Mesh::FaceIterator fi=mesh.face.begin(); fi!=mesh.face.end(); ++fi) {
			if (fi->IsD())
				continue;
			CLEAN::Mesh::FacePointer fp(&(*fi));
			// Face& f(faces.AddEmpty());
            Eigen::Vector3i f;
			f[0] = indices[fp->cV(0)];
			f[1] = indices[fp->cV(1)];
			f[2] = indices[fp->cV(2)];
            faces_.emplace_back(f);
		}
	}
	printf("Cleaned mesh: %u vertices, %u faces\n", vertices_.size(), faces_.size());
} // Clean

void TriangleMesh::ModifyNonMainfoldFace(){
    std::unordered_map<uint64_t, std::vector<size_t>> edge_2_faces;
    size_t num_vert = vertices_.size();
    for (size_t i = 0; i < faces_.size(); i++){
        const Eigen::Vector3i& facet = faces_.at(i);
        for (int k = 0; k < 3; k++){
            size_t vert_id0 = facet[k];
            size_t vert_id1 = facet[(k + 1) % 3];
            if (vert_id0 > vert_id1){
                size_t vert_temp = vert_id0;
                vert_id0 = vert_id1;
                vert_id1 = vert_temp;
            }
            
            uint64_t key = vert_id0 * num_vert + vert_id1;
            edge_2_faces[key].push_back(i);
        }
    }

    size_t num_modify = 0;
#if 0
    std::unordered_set<size_t> delete_facet_id;
    for (const auto edge : edge_2_faces){
        if (edge.second.size() > 2){
            delete_facet_id.insert(edge.second.begin(), edge.second.end());
            num_modify++;
        }
    }
    size_t i, j;
    for (i = 0, j = 0; i < faces_.size(); i++){
        if (delete_facet_id.find(i) != delete_facet_id.end()){
            continue;
        }
        faces_.at(j) = faces_.at(i);
        j++;
    }
    faces_.resize(j);

    std::cout << "Modify Non-Mainfold ( remove " << num_modify << " face)" << std::endl;
#else 
    // ToDo
    for (auto non_main_edge : edge_2_faces){
        if (non_main_edge.second.size() < 3 ) {
            continue;
        }
        num_modify++;

        uint64_t key = non_main_edge.first;
         size_t vert_id0 = key / num_vert;
        size_t vert_id1 = key % num_vert;
        // std::vector<bool > facet_order(non_main_edge.second.size(), false);
        std::vector<size_t> pos_faces;
        std::vector<size_t> neg_faces;
        for (int k = 0; k < non_main_edge.second.size(); k++ ){
            size_t facet_id = non_main_edge.second.at(k);
            const Eigen::Vector3i& facet = faces_.at(facet_id);
            for (int i = 0; i < 3; i++){
                if (facet[i] == vert_id0 && facet[(i + 1) % 3] == vert_id1){
                    pos_faces.push_back(non_main_edge.second.at(k));
                    break;
                }
            }
            if (pos_faces.empty() || pos_faces.back() != facet_id){
                neg_faces.push_back(facet_id);
            }
        }

        for (int k = 1; k < pos_faces.size() || k < neg_faces.size(); k++){
            size_t num_vertices = vertices_.size();
            vertices_.push_back(vertices_.at(vert_id0));
            vertices_.push_back(vertices_.at(vert_id1));
            if (vertex_colors_.size() == num_vertices){
                vertex_colors_.push_back(vertex_colors_.at(vert_id0));
                vertex_colors_.push_back(vertex_colors_.at(vert_id1));
            }
            if (vertex_labels_.size() == num_vertices){
                vertex_labels_.push_back(vertex_labels_.at(vert_id0));
                vertex_labels_.push_back(vertex_labels_.at(vert_id1));
            }
            if (vertex_normals_.size() == num_vertices){
                vertex_normals_.push_back(vertex_normals_.at(vert_id0));
                vertex_normals_.push_back(vertex_normals_.at(vert_id1));
            }
            if (vertex_status_.size() == num_vertices){
                vertex_status_.push_back(vertex_status_.at(vert_id0));
                vertex_status_.push_back(vertex_status_.at(vert_id1));
            }
            if (vertex_visibilities_.size() == num_vertices){
                vertex_visibilities_.push_back(vertex_visibilities_.at(vert_id0));
                vertex_visibilities_.push_back(vertex_visibilities_.at(vert_id1));
            }
            

            if (k < pos_faces.size()){
                size_t facet_id = pos_faces.at(k);
                Eigen::Vector3i& facet = faces_.at(facet_id);
                for (int i = 0; i < 3; i++){
                    if (facet[i] == vert_id0 && facet[(i + 1) % 3] == vert_id1){
                        facet[i] = num_vertices;
                        facet[(i+1) % 3] = num_vertices+1;
                        break;
                    }
                }
            }

            if (k < neg_faces.size()){
                size_t facet_id = neg_faces.at(k);
                Eigen::Vector3i& facet = faces_.at(facet_id);
                for (int i = 0; i < 3; i++){
                    if (facet[i]  == vert_id1&& facet[(i + 1) % 3] == vert_id0){
                        facet[i] = num_vertices + 1;
                        facet[(i+1) % 3] = num_vertices;
                        break;
                    }
                }
            }
        }
    }

    std::cout << "Modify Non-Mainfold ( split " << num_modify << " face)" << std::endl;
#endif

    // for (size_t i = 0; i < vertex_colors_.size(); i++){
    //     vertex_colors_.at(i) = Eigen::Vector3d(0,0,0);
    // }
    // for (const auto edge : edge_2_faces){
    //     if (edge.second.size() > 2){
    //         auto key = edge.first;
    //         size_t vert_id0 = key / num_vert;
    //         size_t vert_id1 = key % num_vert;
    //         vertex_colors_.at(vert_id0) = Eigen::Vector3d(255,255,255);
    //         vertex_colors_.at(vert_id1) = Eigen::Vector3d(255,255,255);
    //     }
    // }
    return;
}
void TraverseNeighbors(std::map<size_t, std::vector<size_t>>& face_neighbors_to_face,
                                                std::set<size_t>& boundary_face_set, 
                                                std::set<size_t >& outof_range_faceid,
                                                size_t face_id){
    if (boundary_face_set.find(face_id) != boundary_face_set.end() || 
        outof_range_faceid.find(face_id) == outof_range_faceid.end()){
        return;
    }
    boundary_face_set.insert(face_id);
    outof_range_faceid.erase(face_id);
    for (auto it : face_neighbors_to_face[face_id]){
        TraverseNeighbors(face_neighbors_to_face, 
                                              boundary_face_set,
                                              outof_range_faceid,
                                              it);
    }
};

bool TriangleMesh::FilterOutOfRangeFace(const float fSpurious){
    const size_t num_faces = faces_.size();
    std::vector<double > edge_lens(num_faces * 3);
    for (int i = 0; i < num_faces; i++){
        const Eigen::Vector3i& facet = faces_.at(i);
        for (int k = 0; k < 3; k++){
            size_t vert_id0 = facet[k];
            size_t vert_id1 = facet[(k + 1) % 3];

            double distance = (vertices_.at(vert_id0) - vertices_.at(vert_id1)).norm();
            edge_lens.at(3 * i + k ) = distance;
        }
    }

    auto edge_lens_temp = edge_lens;
    int nth = edge_lens_temp.size() * 98 / 100;
    std::nth_element(edge_lens_temp.begin(), edge_lens_temp.begin() + nth, edge_lens_temp.end());
    const double th_long_edge = edge_lens_temp[nth] * fSpurious;

    std::set<size_t > outof_range_faceid;
    for (int i = 0; i < edge_lens.size(); i++) { 
        if (edge_lens.at(i) > th_long_edge){
            outof_range_faceid.insert(i/3);
        }
    }
    // std::cout << "outof_range_faceid size: " << outof_range_faceid.size() << std::endl;

    // Face Neighbors
    std::set<size_t> boundary_face_ids;
    std::set<std::pair<size_t, size_t> > boundary_edge_ids;
    // int out_face_num = outof_range_faceid.size();
    std::map<size_t, std::vector<size_t>> face_neighbors_to_face;
    std::map<size_t, std::vector<size_t>> vertex_edge_to_vertex;
    std::map<std::pair<size_t, size_t>, std::vector<size_t>> edge_face_map;
    size_t vertex_id_1, vertex_id_2;
    // for(const auto out_face_id : outof_range_faceid) {
    for(int i = 0; i < faces_.size(); i++) {
        const auto &face = faces_[i];
        for(int j = 0; j < 3; ++j) {
            vertex_id_1 = face[j];
            vertex_id_2 = face[(j + 1) % 3];
            if(vertex_id_1 > vertex_id_2)
                std::swap(vertex_id_1, vertex_id_2);
            edge_face_map[std::make_pair(vertex_id_1, vertex_id_2)].push_back(i);
        }
    }
    size_t face_id_1, face_id_2;
    for (auto &it : edge_face_map) {
        if (it.second.size() == 1){
            boundary_face_ids.insert(it.second[0]);
            boundary_edge_ids.insert(it.first);
            const auto &face = faces_[it.second[0]];
            for(int j = 0; j < 3; ++j) {
                if (face[j] == it.first.first){
                    if (face[(j + 1) % 3] == it.first.second){
                        vertex_edge_to_vertex[it.first.first].push_back(it.first.second);
                    } else {
                        vertex_edge_to_vertex[it.first.second].push_back(it.first.first);
                    }
                }
            }
        }
        for(int i = 0; i < it.second.size(); ++i) {
            face_id_1 = it.second[i];
            for(int j = i + 1; j < it.second.size(); ++j) {
                face_id_2 = it.second[j];
                face_neighbors_to_face[face_id_1].push_back(face_id_2);
                face_neighbors_to_face[face_id_2].push_back(face_id_1);
            }
        }
    }
    
    std::pair<size_t, size_t> next_edge_id;
    size_t next_vert_id = -1;
    size_t last_vert_id = -1;
    std::vector<std::set<size_t> > edge_clusters;
    std::set<size_t> hole_edge_verts;
    std::vector<std::set<size_t> > face_clusters;
    std::set<size_t>  hole_face_ids;
    std::vector<std::pair<size_t, double>> cluster_dist;
    std::pair<size_t, double> edge_dist = std::pair<size_t, double>(0, 0.0);
    while (!boundary_edge_ids.empty()){
        if (hole_edge_verts.empty()){
            next_edge_id = *boundary_edge_ids.begin();
            const auto &face = faces_[edge_face_map[next_edge_id].at(0)];
            for(int j = 0; j < 3; ++j) {
                if (face[j] == next_edge_id.first){
                    if (face[(j + 1) % 3] == next_edge_id.second){
                        hole_edge_verts.insert(next_edge_id.first);
                        last_vert_id = next_edge_id.first;
                        next_vert_id = next_edge_id.second;
                    } else {
                        hole_edge_verts.insert(next_edge_id.second);
                        last_vert_id = next_edge_id.second;
                        next_vert_id = next_edge_id.first;
                    }
                }
            }
        }

        edge_dist.second += (vertices_.at(last_vert_id) - vertices_.at(next_vert_id)).norm();
        hole_edge_verts.insert(next_vert_id);
        hole_face_ids.insert(edge_face_map[next_edge_id].at(0));
        boundary_edge_ids.erase(next_edge_id);

        bool has_neig = false;
        for (const auto vert_id : vertex_edge_to_vertex[next_vert_id]){
            std::pair<size_t, size_t> temp_edge_id;
            if (next_vert_id > vert_id){
                temp_edge_id = std::make_pair(vert_id, next_vert_id);
            } else {
                temp_edge_id = std::make_pair(next_vert_id, vert_id);
            }
            if (boundary_edge_ids.find(temp_edge_id) == boundary_edge_ids.end()){
                continue;
            }
            last_vert_id = next_vert_id;
            next_edge_id = temp_edge_id;
            next_vert_id = vert_id;
            has_neig = true;
            break;
        }

        if (!has_neig){
            cluster_dist.push_back(edge_dist);
            edge_dist.first = cluster_dist.size();
            edge_dist.second = 0.0;
            edge_clusters.push_back(hole_edge_verts);
            hole_edge_verts.clear();
            face_clusters.push_back(hole_face_ids);
            hole_face_ids.clear();
        }
    }

    std::sort(cluster_dist.begin(), cluster_dist.end(), [](const std::pair<size_t, double> a, const std::pair<size_t, double> b){
         return a.second > b.second;
         });
    boundary_face_ids.clear();
    for (size_t i = 0; i < cluster_dist.size(); i++) {
        size_t idx = cluster_dist.at(i).first;
        if (i > 0 && edge_clusters[idx].size() < 6 && 100 * cluster_dist.at(idx).second < cluster_dist.at(0).second){
            continue;
        }
        boundary_face_ids.insert(face_clusters.at(idx).begin(), face_clusters.at(idx).end());
    }

    // Cluster
    std::set<size_t> boundary_face_set;
    for (size_t face_id : boundary_face_ids){
        TraverseNeighbors(face_neighbors_to_face, 
            boundary_face_set, outof_range_faceid, face_id);
    }

    RemoveSelectFaces(boundary_face_set);
     std::cout << "Remove " << boundary_face_set.size() << " out of range  face" << std::endl;
    return true;
}
/*----------------------------------------------------------------*/

void TriangleMesh::ComputeNormals() {
    // if (vertex_normals_.size() != 0) {
    //     return;
    // }
    std::cout << "Compute Vertex Normal" << std::endl;
    // Compute facet normal.
    face_normals_.resize(faces_.size());
    #pragma omp parallel for
    for (size_t i = 0; i < faces_.size(); ++i) {
        const Eigen::Vector3i& facet = faces_.at(i);
        const Eigen::Vector3d& vtx0 = vertices_.at(facet[0]);
        const Eigen::Vector3d& vtx1 = vertices_.at(facet[1]);
        const Eigen::Vector3d& vtx2 = vertices_.at(facet[2]);
        Eigen::Vector3d normal = (vtx1 - vtx0).cross(vtx2 - vtx0);
        face_normals_[i] = normal.normalized();
    }

    // Compute adjacent facets per vertex.
    std::vector<std::vector<int> > adj_facets(vertices_.size());
    for (size_t i = 0; i < faces_.size(); ++i) {
        const Eigen::Vector3i& facet = faces_.at(i);
        adj_facets[facet[0]].push_back(i);
        adj_facets[facet[1]].push_back(i);
        adj_facets[facet[2]].push_back(i);
    }

    // Compute vertex normal.
    vertex_normals_.resize(vertices_.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vertices_.size(); ++i) {
        const auto& adj_facet_per_vert = adj_facets.at(i);
        Eigen::Vector3d normal(0, 0, 0);
        for (const auto& facet_idx : adj_facet_per_vert) {
            normal = (normal + face_normals_[facet_idx]).normalized();
        }
        vertex_normals_[i] = normal;
    }
}

void TriangleMesh::AddMesh(const TriangleMesh& new_mesh) {
	int ori_num_face = faces_.size();
	vertices_.reserve(new_mesh.vertices_.size() + vertices_.size());
    vertex_normals_.reserve(new_mesh.vertex_normals_.size() + vertex_normals_.size());
    vertex_colors_.reserve(new_mesh.vertex_colors_.size() + vertex_colors_.size());
    vertex_labels_.reserve(new_mesh.vertex_labels_.size() + vertex_labels_.size());
    faces_.reserve(new_mesh.faces_.size() + faces_.size());
    face_normals_.reserve(new_mesh.face_normals_.size() + face_normals_.size());

    std::size_t vertex_num = new_mesh.vertices_.size();
	std::size_t new_face_num = new_mesh.faces_.size();
	for(std::size_t face_id = 0; face_id < new_face_num; face_id++){
		faces_.push_back(new_mesh.faces_[face_id]);
	}

    if (!new_mesh.face_normals_.empty()) {
        for (std::size_t face_id = 0; face_id < new_face_num; face_id++) {
            face_normals_.push_back(new_mesh.face_normals_[face_id]);
        }
    }
    std::vector<int> vertex_idx_map(vertex_num, -1);
    for (std::size_t face_id = 0; face_id < new_face_num; face_id++) {
        const auto &face = new_mesh.faces_[face_id];
        vertex_idx_map[face[0]] = 0;
        vertex_idx_map[face[1]] = 0;
        vertex_idx_map[face[2]] = 0;
    }
    for (std::size_t i = 0; i < vertex_num; i++) {
        if (vertex_idx_map[i] == -1) {
            continue;
        }

        vertex_idx_map[i] = vertices_.size();
        vertices_.push_back(new_mesh.vertices_[i]);
    }
    if (!new_mesh.vertex_normals_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }

            vertex_normals_.push_back(new_mesh.vertex_normals_[i]);
        }
    }
    if (!new_mesh.vertex_colors_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }

            vertex_colors_.push_back(new_mesh.vertex_colors_[i]);
        }
    }
    if (!new_mesh.vertex_labels_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }

            vertex_labels_.push_back(new_mesh.vertex_labels_[i]);
        }
    }
    vertices_.shrink_to_fit();
    vertex_normals_.shrink_to_fit();
    vertex_colors_.shrink_to_fit();
    vertex_labels_.shrink_to_fit();
    faces_.shrink_to_fit();
    face_normals_.shrink_to_fit();
    // for (auto &face : faces_) {
    for (int new_face_id = ori_num_face; new_face_id < faces_.size(); new_face_id++) {
		auto &face = faces_.at(new_face_id);
        face[0] = vertex_idx_map[face[0]];
        face[1] = vertex_idx_map[face[1]];
        face[2] = vertex_idx_map[face[2]];
    }

}
void TriangleMesh::Swap(TriangleMesh& new_mesh) {
    vertices_.swap(new_mesh.vertices_);
    vertex_normals_.swap(new_mesh.vertex_normals_);
    vertex_colors_.swap(new_mesh.vertex_colors_);
    faces_.swap(new_mesh.faces_);
    face_normals_.swap(new_mesh.face_normals_);
    vertex_labels_.swap(new_mesh.vertex_labels_);
    vertex_status_.swap(new_mesh.vertex_status_);
    vertex_visibilities_.swap(new_mesh.vertex_visibilities_);
}

bool WriteTriangleMeshObj(const std::string &filename, 
                          const TriangleMesh &mesh,
                          const bool write_as_rgb,
                          const bool write_as_sem) {
    FILE *obj_file = fopen(filename.c_str(), "w");

    bool write_rgb = (mesh.vertex_colors_.size() != 0) && write_as_rgb;
    bool write_normal = (mesh.vertex_normals_.size() != 0);

    printf("[Writing Triangle Mesh (.Obj) , rgb(%d), normal(%d), sem(%d)...", 
           int(write_rgb), int(write_normal), int(write_as_sem));
    if (write_as_sem) {
		for(int i = 0; i < mesh.vertices_.size(); ++i){
            fprintf(obj_file, "v %lf %lf %lf %lf %lf %lf %d\n",
                    mesh.vertices_[i](0), mesh.vertices_[i](1),
                    mesh.vertices_[i](2), 
                    mesh.vertex_colors_[i](0),
                    mesh.vertex_colors_[i](1),
                    mesh.vertex_colors_[i](2),
					mesh.vertex_labels_[i]);
        }
	} else if (write_rgb) {
        for(int i = 0; i < mesh.vertices_.size(); ++i){
            fprintf(obj_file, "v %lf %lf %lf %lf %lf %lf\n",
                    mesh.vertices_[i](0), mesh.vertices_[i](1),
                    mesh.vertices_[i](2), 
                    mesh.vertex_colors_[i](0),
                    mesh.vertex_colors_[i](1),
                    mesh.vertex_colors_[i](2));
        }
    } else {
        for(int i = 0; i < mesh.vertices_.size(); ++i){
            fprintf(obj_file, "v %lf %lf %lf\n",
                    mesh.vertices_[i](0), mesh.vertices_[i](1),
                    mesh.vertices_[i](2));
        }
    }

    for(int i = 0; i < mesh.vertex_normals_.size(); ++i){
        fprintf(obj_file, "vn %lf %lf %lf\n", mesh.vertex_normals_[i](0),
                mesh.vertex_normals_[i](1),mesh.vertex_normals_[i](2));
    }
    int num_skip = 0;
    for(int i = 0; i < mesh.faces_.size(); ++i){
        if ((mesh.vertices_[mesh.faces_[i](0)] - mesh.vertices_[mesh.faces_[i](1)]).norm() < 1e-6 ||
             (mesh.vertices_[mesh.faces_[i](1)] - mesh.vertices_[mesh.faces_[i](2)]).norm() < 1e-6 ||
             (mesh.vertices_[mesh.faces_[i](2)] - mesh.vertices_[mesh.faces_[i](0)]).norm() < 1e-6){
            num_skip++;
            continue;
        }
        if (write_normal) {
            fprintf(obj_file, "f %d//%d %d//%d %d//%d\n",
                    mesh.faces_[i](0) + 1, mesh.faces_[i](0) +1,
                    mesh.faces_[i](1) + 1, mesh.faces_[i](1) +1,
                    mesh.faces_[i](2) + 1, mesh.faces_[i](2) +1);
        } else {
            fprintf(obj_file, "f %d %d %d\n",
                    mesh.faces_[i](0) + 1, mesh.faces_[i](1) + 1,
                    mesh.faces_[i](2) + 1);
        }
    }
    fclose(obj_file);
    printf(", skip zero face(%d) ... done]\n", num_skip);
    return true;
}

bool ReadTriangleMeshObj(const std::string &filename, 
                         TriangleMesh &mesh,
                         const bool read_as_rgb,
                         const bool read_as_sem) {
    FILE *file = fopen(filename.c_str(), "r");
	if (!file) {
		std::cerr << "File does not exist! Please check the file path "
				  << filename << std::endl;
		return false;
	}
    char line_buf[MAX_LINE_LENGTH];
    while (fgets(line_buf, MAX_LINE_LENGTH, file))
    {
        if (!strncmp(line_buf, "v ", 2))
        {
            if (read_as_sem) {
                Eigen::Vector3d vertex, vertex_color;
				int vertex_label;
                sscanf(line_buf + 2, "%lf %lf %lf %lf %lf %lf %d",
                    &vertex[0], &vertex[1], &vertex[2], 
                    &vertex_color[0], &vertex_color[1], &vertex_color[2],
					&vertex_label);
                mesh.vertices_.push_back(vertex);
                mesh.vertex_colors_.push_back(vertex_color);
				mesh.vertex_labels_.push_back(vertex_label);
			} else if (read_as_rgb) {
                Eigen::Vector3d vertex, vertex_color;
                sscanf(line_buf + 2, "%lf %lf %lf %lf %lf %lf",
                    &vertex[0], &vertex[1], &vertex[2], 
                    &vertex_color[0], &vertex_color[1], &vertex_color[2]);
                mesh.vertices_.push_back(vertex);
                mesh.vertex_colors_.push_back(vertex_color);
            } else {
                Eigen::Vector3d vertex;
                sscanf(line_buf + 2, "%lf %lf %lf",
                    &vertex[0], &vertex[1], &vertex[2]);
                mesh.vertices_.push_back(vertex);
            }
        }
        if (!strncmp(line_buf, "vn ", 3))
        {
            Eigen::Vector3d vertex_norm;
            sscanf(line_buf + 3, "%lf %lf %lf",
                   &vertex_norm[0], &vertex_norm[1], &vertex_norm[2]);
            mesh.vertex_normals_.push_back(vertex_norm);
        }
        if (!strncmp(line_buf, "f ", 2))
        {
            Eigen::Vector3i face;
            char c;
			if (mesh.vertex_normals_.size() != 0) {
				sscanf(line_buf + 2, "%d %c %c %d %d %c %c %d %d %c %c %d",
					&face[0], &c, &c, &face[0],
					&face[1], &c, &c, &face[1],
					&face[2], &c, &c, &face[2]);
			} else {
				sscanf(line_buf + 2, "%d %d %d", &face[0], &face[1], &face[2]);
			}
            face[0] -= 1;
            face[1] -= 1;
            face[2] -= 1;
            mesh.faces_.push_back(face);
        }
    }
    fclose(file);
    std::cout << "vtxs.size(): " << mesh.vertices_.size() << std::endl;
    std::cout << "faces.size(): " << mesh.faces_.size() << std::endl;
	if (read_as_rgb) {
    	mesh.vertex_colors_.resize(mesh.vertices_.size());
	}
	if (mesh.vertex_normals_.size() != 0) {
		mesh.face_normals_.resize(mesh.faces_.size());
		for (int i = 0; i < mesh.face_normals_.size(); i++)
		{
			Eigen::Vector3d &N = mesh.face_normals_[i];
			N += mesh.vertex_normals_[mesh.faces_[i][0]];
			N += mesh.vertex_normals_[mesh.faces_[i][1]];
			N += mesh.vertex_normals_[mesh.faces_[i][2]];
			N.normalize();
		}
	}
    return true;
}

void MeshBox::ReadBox(const std::string& path){
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    // x_min, y_min, x_max, y_max
    std::getline(file, line);
    StringTrim(&line);
    while (line.empty() || line[0] == '#') {
        std::getline(file, line);
        StringTrim(&line);
    }
    std::stringstream line_stream(line);
    std::getline(line_stream, item, ' ');
    x_min = std::stof(item);
    std::getline(line_stream, item, ' ');
    y_min = std::stof(item);
    std::getline(line_stream, item, ' ');
    x_max = std::stof(item);
    std::getline(line_stream, item, ' ');
    y_max = std::stof(item);

    // rotation matrix
    std::getline(file, line);
    StringTrim(&line);
    while (line.empty() || line[0] == '#') {
        std::getline(file, line);
        StringTrim(&line);
    }
    std::stringstream rot_line_stream(line);
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            std::getline(rot_line_stream, item, ' ');
            rot(i,j) = std::stof(item);
        }
    }
}

void MeshBox::SetBoundary(){
    if (std::abs(x_min - x_max) < std::numeric_limits<float>::epsilon() || 
		std::abs(y_min - y_max) < std::numeric_limits<float>::epsilon()){
        std::cout << "Error: invalid box boundary" << std::endl;
        return;
    }

    if (border_width < 0 && border_factor < 0){
        border_width = std::min((x_max -x_min), (y_max -y_min)) / 100.0f;
    } else if (border_factor > 0){
        float border_width_1 = std::min((x_max -x_min), (y_max -y_min)) * border_factor;
        border_width = std::max(border_width, border_width_1);
    }

    box_x_min = x_min - border_width;
    box_y_min = y_min - border_width;
    box_x_max = x_max + border_width;
    box_y_max = y_max + border_width;
    if (z_min > z_max){
        box_z_min = -FLT_MAX;
        box_z_max = FLT_MAX;
    } else {
        float border_width_z = (z_max - z_min) / 100.f;
        box_z_min = z_min - border_width_z;
        box_z_max = z_max + border_width_z;
    }
}

void MeshBox::ResetBoundary(float scale_factor){
    if (border_width < 0 && border_factor < 0){
        border_factor = 0.01;
    }
    border_width *= scale_factor;
    border_factor *= scale_factor;
    SetBoundary();
}

void MeshBox::Print() const{
    PrintHeading2("MeshBox:");
    PrintOption(x_min);
    PrintOption(y_min);
    PrintOption(z_min);
    PrintOption(x_max);
    PrintOption(y_max);
    PrintOption(z_max);
    PrintOption(border_width);
    PrintOption(box_x_min);
    PrintOption(box_y_min);
    PrintOption(box_z_min);
    PrintOption(box_x_max);
    PrintOption(box_y_max);
    PrintOption(box_z_max);
    PrintOption(rot);
}

bool FilterWithBox(TriangleMesh &mesh,
                   const MeshBox box){
	TriangleMesh filtered_mesh;
    std::cout << "MeshFilterWithBox" << std::endl;
    if (mesh.vertices_.empty()|| mesh.faces_.empty()) {
        return 0;
    }

    Eigen::Matrix3f pivot = box.rot;
    Eigen::Vector3d centroid(Eigen::Vector3d::Zero());
    for (const auto &vertex : mesh.vertices_) {
        centroid += vertex;
    }
    std::size_t vertex_num = mesh.vertices_.size();
    centroid /= vertex_num;

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; ++i) {
        for (const auto &vertex : mesh.vertices_) {
            M(i, 0) += (vertex[i] - centroid[i]) * (vertex[0] - centroid[i]);
            M(i, 1) += (vertex[i] - centroid[i]) * (vertex[1] - centroid[i]);
            M(i, 2) += (vertex[i] - centroid[i]) * (vertex[2] - centroid[i]);
        }
    }
    std::size_t face_num = mesh.faces_.size();
    // std::vector<bool> box_face_map(face_num, false);
    std::vector<std::size_t > filtered_faces;
    std::vector<Eigen::Vector3f> transformed_tri_centroids(face_num);
    for (int i = 0; i < face_num; ++i) {
        auto &transformed_tri_centroid = transformed_tri_centroids[i];
        const auto &face = mesh.faces_[i];
        transformed_tri_centroid = pivot * (mesh.vertices_[face[0]] + mesh.vertices_[face[1]] + mesh.vertices_[face[2]]).cast<float>() / 3.0f;
    }

    for (std::size_t face_id = 0; face_id < face_num; ++face_id) {
        const auto &transformed_tri_centroid = transformed_tri_centroids[face_id];
        if (transformed_tri_centroid.x() < box.box_x_min ||
            transformed_tri_centroid.y() < box.box_y_min ||
            transformed_tri_centroid.x() > box.box_x_max ||
            transformed_tri_centroid.y() > box.box_y_max){
            continue;
        };
        filtered_faces.push_back(face_id);
        // box_face_map[i] = true;
    }

    // save filtered mesh
    filtered_mesh.vertices_.reserve(mesh.vertices_.size());
    filtered_mesh.vertex_normals_.reserve(mesh.vertex_normals_.size());
    filtered_mesh.vertex_colors_.reserve(mesh.vertex_colors_.size());
    filtered_mesh.vertex_labels_.reserve(mesh.vertex_labels_.size());
    filtered_mesh.faces_.reserve(mesh.faces_.size());
    filtered_mesh.face_normals_.reserve(mesh.face_normals_.size());

    for (auto face_id : filtered_faces) {
        filtered_mesh.faces_.push_back(mesh.faces_[face_id]);
    }
    if (!mesh.face_normals_.empty()) {
        for (auto face_id : filtered_faces) {
            filtered_mesh.face_normals_.push_back(mesh.face_normals_[face_id]);
        }
    }
    std::vector<int> vertex_idx_map(vertex_num, -1);
    for (auto face_id : filtered_faces) {
        const auto &face = mesh.faces_[face_id];
        vertex_idx_map[face[0]] = 0;
        vertex_idx_map[face[1]] = 0;
        vertex_idx_map[face[2]] = 0;
    }
    for (std::size_t i = 0; i < vertex_num; i++) {
        if (vertex_idx_map[i] == -1) {
            continue;
        }

        vertex_idx_map[i] = filtered_mesh.vertices_.size();
        filtered_mesh.vertices_.push_back(mesh.vertices_[i]);
    }
    if (!mesh.vertex_normals_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }

            filtered_mesh.vertex_normals_.push_back(mesh.vertex_normals_[i]);
        }
    }
    if (!mesh.vertex_colors_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }

            filtered_mesh.vertex_colors_.push_back(mesh.vertex_colors_[i]);
        }
    }
    if (!mesh.vertex_labels_.empty()) {
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }

            filtered_mesh.vertex_labels_.push_back(mesh.vertex_labels_[i]);
        }
    }
    filtered_mesh.vertices_.shrink_to_fit();
    filtered_mesh.vertex_normals_.shrink_to_fit();
    filtered_mesh.vertex_colors_.shrink_to_fit();
    filtered_mesh.vertex_labels_.shrink_to_fit();
    filtered_mesh.faces_.shrink_to_fit();
    filtered_mesh.face_normals_.shrink_to_fit();
    for (auto &face : filtered_mesh.faces_) {
        face[0] = vertex_idx_map[face[0]];
        face[1] = vertex_idx_map[face[1]];
        face[2] = vertex_idx_map[face[2]];
    }
	mesh = filtered_mesh;
    return true;
}

} // namespace sensemap