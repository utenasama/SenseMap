//
// Created by sensetime on 2020/1/9.
//

#ifndef SENSEMAP_MESH_INFO_H_
#define SENSEMAP_MESH_INFO_H_


#include <algorithm>
#include <vector>
#include <memory>
#include <unordered_map>

#include "obj.h"

namespace sensemap {

class MeshInfo {
public:
    /** Vertex classification according to adjacent triangles. */
    enum VertexClass {
        /** Vertex with a single closed fan of adjacent triangles. */
                VERTEX_CLASS_SIMPLE,
        /** Vertex with a single but open fan of triangles. */
                VERTEX_CLASS_BORDER,
        /** Vertex with more than one triangle fan. */
                VERTEX_CLASS_COMPLEX,
        /** Vertiex without any adjacent triangles. */
                VERTEX_CLASS_UNREF
    };

    typedef std::vector<int> AdjacentVertices;
    typedef std::vector<int> AdjacentFaces;

    /** Per-vertex classification and adjacency information. */
    struct VertexInfo {
        VertexClass vclass;
        AdjacentVertices verts;
        AdjacentFaces faces;

        void remove_adjacent_face(int face_id);

        void remove_adjacent_vertex(int vertex_id);

        void replace_adjacent_face(int old_id, int new_id);

        void replace_adjacent_vertex(int old_id, int new_id);
    };

    struct EdgeInfo {
        int num_shared_facet = 0;
    };

public:
    /** Constructor without initialization. */
    MeshInfo(void);

    /** Constructor with initialization for the given mesh. */
    MeshInfo(TriangleMesh &mesh);

    /** Initializes the data structure for the given mesh. */
    void initialize(TriangleMesh &mesh);

    /**
     * Updates the vertex info for a single vertex. It expects that the
     * list of adjacent faces is complete (but unorderd), and recomputes
     * adjacent face ordering, adjacent vertices and the vertex class.
     */
    void update_vertex(TriangleMesh const &mesh, int vertex_id);

    /** Checks for the existence of an edge between the given vertices. */
    bool is_mesh_edge(int v1,int v2) const;

    /** Returns faces adjacent to both vertices. */
    void get_faces_for_edge(int v1, int v2,
                            std::vector<int> *adjacent_faces) const;

    /** Returns border verices list for each border. */
    void get_border_lists(std::vector<std::vector<int> > &lists);

    /** Returns border verices list for each border from given ids. */
    void get_border_lists(const std::vector<int> &border_ids, std::vector<std::vector<int> > &lists);

    void get_clustered_adj_faces(const TriangleMesh &mesh, const int vertex_id,
                                 std::vector<std::vector<int> > &adj_facets);

    void resolve_complex_vertex(TriangleMesh &mesh);

    bool remove_adj_border_faces(TriangleMesh &mesh);

    bool remove_complex_faces(TriangleMesh &mesh);

    bool remove_unref_vertices(TriangleMesh &mesh);
public:
    VertexInfo &operator[](int id);

    VertexInfo const &operator[](int id) const;

    VertexInfo &at(int id);

    VertexInfo const &at(int id) const;

    int size(void) const;

    void clear(void);

private:
    std::vector<VertexInfo> vertex_info;
    std::unordered_map<uint64_t, EdgeInfo> edge_info;
};
    

/* ------------------------- Implementation ----------------------- */

inline
MeshInfo::MeshInfo (void)
{
}

inline
MeshInfo::MeshInfo (TriangleMesh & mesh)
{
    this->initialize(mesh);
}

inline MeshInfo::VertexInfo&
MeshInfo::operator[] (int id)
{
    return this->vertex_info[id];
}

inline MeshInfo::VertexInfo const&
MeshInfo::operator[] (int id) const
{
    return this->vertex_info[id];
}

inline MeshInfo::VertexInfo&
MeshInfo::at (int id)
{
    return this->vertex_info[id];
}

inline MeshInfo::VertexInfo const&
MeshInfo::at (int id) const
{
    return this->vertex_info[id];
}

inline int
MeshInfo::size (void) const
{
    return this->vertex_info.size();
}

inline void
MeshInfo::clear (void)
{
    std::vector<VertexInfo>().swap(this->vertex_info);
    edge_info.clear();
}

inline void
MeshInfo::VertexInfo::remove_adjacent_face (int face_id)
{
    this->faces.erase(std::remove(this->faces.begin(), this->faces.end(),
                                  face_id), this->faces.end());
}

inline void
MeshInfo::VertexInfo::remove_adjacent_vertex (int vertex_id)
{
    this->verts.erase(std::remove(this->verts.begin(), this->verts.end(),
                                  vertex_id), this->verts.end());
}

inline void
MeshInfo::VertexInfo::replace_adjacent_face (int old_id,
                                             int new_id)
{
    std::replace(this->faces.begin(), this->faces.end(), old_id, new_id);
}

inline void
MeshInfo::VertexInfo::replace_adjacent_vertex (int old_id,
                                               int new_id)
{
    std::replace(this->verts.begin(), this->verts.end(), old_id, new_id);
}

}  //namespace sensemap

#endif //SENSEMAP_MESH_INFO_H_
