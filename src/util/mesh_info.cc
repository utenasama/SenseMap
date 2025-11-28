//
// Created by sensetime on 2020/1/9.
//

#include "mesh_info.h"

#include <algorithm>
#include <list>
#include <set>
#include <iostream>

namespace sensemap {

const  size_t kMaxNumVerts =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());

uint64_t EdgePairToPairId(const int vert_id1, const int vert_id2) {
    if (vert_id1 > vert_id2) {
        return static_cast<uint64_t>(kMaxNumVerts) * vert_id2 + vert_id1;
    } 
    else {
        return static_cast<uint64_t>(kMaxNumVerts) * vert_id1 + vert_id2;
    }
}

void
MeshInfo::initialize(TriangleMesh & mesh) {
    auto &verts = mesh.vertices_;
    auto &faces = mesh.faces_;

    auto face_amount = faces.size();

    this->vertex_info.clear();
    this->vertex_info.resize(verts.size());

    /* Add faces to their three vertices. */
    for (int i = 0; i < face_amount; ++i)
        for (int j = 0; j < 3; ++j)
            this->vertex_info[faces[i][j]].faces.push_back(i);

    /* Classify each vertex and compute adjacenty info. */
    for (int i = 0; i < this->vertex_info.size(); ++i)
        this->update_vertex(mesh, i);

    this->edge_info.clear();

    for (const auto& facet : faces) {
        int idx0 = facet.x();
        int idx1 = facet.y();
        int idx2 = facet.z();
        
        uint64_t edge_id01 = EdgePairToPairId(idx0, idx1);
        this->edge_info[edge_id01].num_shared_facet++;
        uint64_t edge_id02 = EdgePairToPairId(idx0, idx2);
        this->edge_info[edge_id02].num_shared_facet++;
        uint64_t edge_id12 = EdgePairToPairId(idx1, idx2);
        this->edge_info[edge_id12].num_shared_facet++;
    }
}

/* ---------------------------------------------------------------- */

namespace {
    /* Adjacent face representation for the ordering algorithm. */
    struct AdjacentFace {
        int face_id;
        int first;
        int second;
    };

    typedef std::list<AdjacentFace> AdjacentFaceList;
}

/* ---------------------------------------------------------------- */

void
MeshInfo::update_vertex(TriangleMesh const &mesh, int vertex_id) {
    auto &verts = mesh.vertices_;
    auto &faces = mesh.faces_;
    VertexInfo &vinfo = this->vertex_info[vertex_id];

    /* Build new, temporary adjacent faces representation for ordering. */
    AdjacentFaceList adj_temp;
    for (int i = 0; i < vinfo.faces.size(); ++i) {
        int f_id = vinfo.faces[i];
        for (int j = 0; j < 3; ++j)
            if (faces[f_id][j] == vertex_id) {
                adj_temp.push_back(AdjacentFace());
                adj_temp.back().face_id = vinfo.faces[i];
                adj_temp.back().first = faces[f_id][(j + 1) % 3];
                adj_temp.back().second = faces[f_id][(j + 2) % 3];
                break;
            }
    }

    /* If there are no adjacent faces, the vertex is unreferenced. */
    if (adj_temp.empty()) {
        vinfo = VertexInfo();
        vinfo.vclass = VERTEX_CLASS_UNREF;
        return;
    }

    /* Sort adjacent faces by chaining them. */
    AdjacentFaceList adj_sorted;
    adj_sorted.push_back(adj_temp.front());
    adj_temp.pop_front();
    while (!adj_temp.empty()) {
        int const front_id = adj_sorted.front().first;
        int const back_id = adj_sorted.back().second;

        /* Find a faces that fits the back or front of sorted list. */
        bool found_face = false;
        for (AdjacentFaceList::iterator iter = adj_temp.begin();
             iter != adj_temp.end(); ++iter) {
            if (front_id == iter->second) {
                adj_sorted.push_front(*iter);
                adj_temp.erase(iter);
                found_face = true;
                break;
            }
            if (back_id == iter->first) {
                adj_sorted.push_back(*iter);
                adj_temp.erase(iter);
                found_face = true;
                break;
            }
        }

        /* If there is no next face, the vertex is complex. */
        if (!found_face)
            break;
    }

    /* If the vertex is complex, add unsorted adjacency information. */
    if (!adj_temp.empty()) {
        /* Transfer remaining adjacent faces. */
        adj_sorted.insert(adj_sorted.end(), adj_temp.begin(), adj_temp.end());
        adj_temp.clear();

        /* Create unique list of all adjacent vertices. */
        std::set<int> vset;
        for (AdjacentFaceList::iterator iter = adj_sorted.begin();
             iter != adj_sorted.end(); ++iter) {
            vset.insert(iter->first);
            vset.insert(iter->second);
        }
        vinfo.verts.insert(vinfo.verts.end(), vset.begin(), vset.end());
        vinfo.vclass = VERTEX_CLASS_COMPLEX;
        return;
    }

    /* If the vertex is not on the mesh boundary, the list is circular. */
    if (adj_sorted.front().first == adj_sorted.back().second)
        vinfo.vclass = VERTEX_CLASS_SIMPLE;
    else
        vinfo.vclass = VERTEX_CLASS_BORDER;

    /* Insert the face IDs in the adjacent faces list. */
    vinfo.faces.clear();
    for (AdjacentFaceList::iterator iter = adj_sorted.begin();
         iter != adj_sorted.end(); ++iter)
        vinfo.faces.push_back(iter->face_id);

    /* Insert vertex IDs in adjacent vertex list. */
    for (AdjacentFaceList::const_iterator iter = adj_sorted.begin();
         iter != adj_sorted.end(); ++iter)
        vinfo.verts.push_back(iter->first);
    if (vinfo.vclass == VERTEX_CLASS_BORDER)
        vinfo.verts.push_back(adj_sorted.back().second);
}

/* ---------------------------------------------------------------- */

bool
MeshInfo::is_mesh_edge(int v1, int v2) const {
    AdjacentVertices const &verts = this->vertex_info[v1].verts;
    return std::find(verts.begin(), verts.end(), v2) != verts.end();
}

/* ---------------------------------------------------------------- */

void
MeshInfo::get_faces_for_edge(int v1, int v2,
                             std::vector<int> *adjacent_faces) const {
    AdjacentFaces const &faces1 = this->vertex_info[v1].faces;
    AdjacentFaces const &faces2 = this->vertex_info[v2].faces;
    std::set<int> faces2_set(faces2.begin(), faces2.end());
    for (int i = 0; i < faces1.size(); ++i)
        if (faces2_set.find(faces1[i]) != faces2_set.end())
            adjacent_faces->push_back(faces1[i]);
}

void
MeshInfo::get_border_lists(std::vector<std::vector<int> > &lists){

    lists.clear();

    auto &vinfos = this->vertex_info;
    std::vector<bool> used(vinfos.size(), false);
    for(int i = 0; i < vinfos.size(); ++i){
        auto vinfo = vinfos[i];
        if(vinfo.vclass != VERTEX_CLASS_BORDER || used[i])
            continue;

        used[i] = true;

        int processed = 0;
        std::vector<int> border;
        border.push_back(i);
        while(processed != border.size()) {
            int id = border[processed];
            vinfo = vinfos[id];
            for (int j : vinfo.verts) {
                if (vinfos[j].vclass != VERTEX_CLASS_BORDER || used[j])
                    continue;

                uint64_t edge_id = EdgePairToPairId(id, j);
                if (this->edge_info[edge_id].num_shared_facet >= 2) {
                    continue;
                }

                used[j] = true;
                border.push_back(j);
                break; //sort
            }
            processed++;
        }
        lists.push_back(border);
    }
}


void
MeshInfo::get_border_lists(const std::vector<int> &border_ids, std::vector<std::vector<int> > &lists){
    lists.clear();
#if 1  // set 0 if border_ids covers all border vertices
    auto &vinfos = this->vertex_info;
    std::vector<bool> used(vinfos.size(), false);
    for(int i = 0; i < border_ids.size(); ++i){
        int v_i = border_ids[i];
        auto vinfo = vinfos[v_i];

        if(vinfo.vclass != VERTEX_CLASS_BORDER || used[v_i])
            continue;

        used[v_i] = true;

        int processed = 0;
        std::vector<int> border;
        border.push_back(v_i);
        while(processed != border.size()) {
            int id = border[processed];
            vinfo = vinfos[id];
            for (int j : vinfo.verts) {
                if (vinfos[j].vclass != VERTEX_CLASS_BORDER || used[j])
                    continue;

                uint64_t edge_id = EdgePairToPairId(id, j);
                if (this->edge_info[edge_id].num_shared_facet >= 2) {
                    continue;
                }

                used[j] = true;
                border.push_back(j);
                break; //sort
            }
            processed++;
        }
        lists.push_back(border);
    }
#else
    auto &vinfos = this->vertex_info;
    std::vector<bool> used(border_ids.size(), false);
    for(int i = 0; i < border_ids.size(); ++i){
        int v_i = border_ids[i];
        auto vinfo = vinfos[v_i];

        if(used[i])
            continue;

        used[i] = true;

        int processed = 0;
        std::vector<int> border;
        border.push_back(v_i);

        while(processed != border.size()) {
            int v_i = border[processed];
            vinfo = vinfos[v_i];

            for (int v_j : vinfo.verts) {
                auto it = std::find(border_ids.begin(), border_ids.end(), v_j);
                int j = it - border_ids.begin();
                if (it == border_ids.end() || used[j])
                    continue;
                used[j] = true;
                border.push_back(v_j);
                break;
            }
            processed++;
        }

         lists.push_back(border);


    }
#endif
}

void
MeshInfo::get_clustered_adj_faces(const TriangleMesh &mesh, const int vertex_id,
                                  std::vector<std::vector<int> > &adj_facets) {
    adj_facets.clear();

    auto &faces = mesh.faces_;
    VertexInfo& vinfo = this->at(vertex_id);

    /* Build new, temporary adjacent faces representation for ordering. */
    AdjacentFaceList adj_temp;
    for (int i = 0; i < vinfo.faces.size(); ++i) {
        int f_id = vinfo.faces[i];
        for (int j = 0; j < 3; ++j)
            if (faces[f_id][j] == vertex_id) {
                adj_temp.push_back(AdjacentFace());
                adj_temp.back().face_id = vinfo.faces[i];
                adj_temp.back().first = faces[f_id][(j + 1) % 3];
                adj_temp.back().second = faces[f_id][(j + 2) % 3];
                break;
            }
    }

    while(!adj_temp.empty()) {
        /* Sort adjacent faces by chaining them. */
        AdjacentFaceList adj_sorted;
        adj_sorted.push_back(adj_temp.front());
        adj_temp.pop_front();
        while (!adj_temp.empty()) {
            int const front_id = adj_sorted.front().first;
            int const back_id = adj_sorted.back().second;

            /* Find a faces that fits the back or front of sorted list. */
            bool found_face = false;
            for (AdjacentFaceList::iterator iter = adj_temp.begin();
                iter != adj_temp.end(); ++iter) {
                if (front_id == iter->second) {
                    adj_sorted.push_front(*iter);
                    adj_temp.erase(iter);
                    found_face = true;
                    break;
                }
                if (back_id == iter->first) {
                    adj_sorted.push_back(*iter);
                    adj_temp.erase(iter);
                    found_face = true;
                    break;
                }
            }

            /* If there is no next face, the vertex is complex. */
            if (!found_face)
                break;
        }
        std::vector<int> adj_facet_ids;
        for (auto adj_facet : adj_sorted) {
            adj_facet_ids.emplace_back(adj_facet.face_id);
        }
        adj_facets.emplace_back(adj_facet_ids);
    }
}

bool UpdateMesh(TriangleMesh &mesh,
                 std::vector<int> &face_idx,
                 bool inlier)
{
    if (face_idx.empty())
        return false;
    if (!std::is_sorted(face_idx.begin(), face_idx.end()))
        std::sort(face_idx.begin(), face_idx.end());
    if (inlier)
    {
        std::vector<Eigen::Vector3i> new_faces;
        std::vector<int> new_camera_for_faces;
        for (int i : face_idx)
        {
            new_faces.push_back(std::move(mesh.faces_[i]));
        }
        mesh.faces_.swap(new_faces);


        if(!mesh.face_normals_.empty())
        {
            std::vector<Eigen::Vector3d> new_face_normals;
            for(int i : face_idx)
                new_face_normals.push_back(std::move(mesh.face_normals_[i]));
            mesh.face_normals_.swap(new_face_normals);
        }
    }
    else
    {
        //delete faces and faces normal
        std::vector<Eigen::Vector3i> new_faces;
        std::vector<int> new_camera_for_faces;
        int lastIdx = face_idx[face_idx.size() - 1];
        int i = 0;
        for (int j = 0; i <= lastIdx; ++i)
        {
            auto faceIdx = face_idx[j];
            if (faceIdx != i)
            {
                new_faces.push_back(std::move(mesh.faces_[i]));
            }
            else
                ++j;
        }
        if (i < mesh.faces_.size())
        {
            new_faces.insert(new_faces.end(),
                             mesh.faces_.begin() + i, mesh.faces_.end());
        }
        mesh.faces_.swap(new_faces);
        if(!mesh.face_normals_.empty())
        {
            std::vector<Eigen::Vector3d> new_face_normals;
            lastIdx = face_idx[face_idx.size() - 1];
            i = 0;
            for (int j = 0; i <= lastIdx; ++i)
            {
                auto faceIdx = face_idx[j];
                if (faceIdx != i)
                    new_face_normals.push_back(
                            std::move(mesh.face_normals_[i]));
                else
                    ++j;
            }
            if (i < mesh.face_normals_.size())
                new_face_normals.insert(new_face_normals.end(),
                                        mesh.face_normals_.begin() + i,
                                        mesh.face_normals_.end());
            mesh.face_normals_.swap(new_face_normals);
        }
    }
//    //delete  vertices (not consider vertex_lablels yet)
//    std::vector<int> new_idx(mesh.vertices_.size(), 0);
//    for (auto f : mesh.faces_)
//    {
//        new_idx[f(0)] = 1;
//        new_idx[f(1)] = 1;
//        new_idx[f(2)] = 1;
//    }
//    std::vector<Eigen::Vector3d> new_vertices;
//    for (int i = 0; i < mesh.vertices_.size(); ++i)
//    {
//        if (new_idx[i] == 1)
//            new_vertices.push_back(mesh.vertices_[i]);
//    }
//    mesh.vertices_.swap(new_vertices);
//    if(!mesh.vertex_normals_.empty())
//    {
//        std::vector<Eigen::Vector3d> new_vertex_normals;
//        for (int i = 0; i < mesh.vertex_normals_.size(); ++i)
//        {
//            if (new_idx[i] == 1)
//                new_vertex_normals.push_back(mesh.vertex_normals_[i]);
//        }
//        mesh.vertex_normals_.swap(new_vertex_normals);
//    }
//    if(!mesh.vertex_colors_.empty())
//    {
//        std::vector<Eigen::Vector3d> new_vertex_colors;
//        for (int i = 0; i < mesh.vertex_colors_.size(); ++i)
//        {
//            if (new_idx[i] == 1)
//                new_vertex_colors.push_back(mesh.vertex_colors_[i]);
//        }
//        mesh.vertex_colors_.swap(new_vertex_colors);
//    }
//
//    //update faces
//    for (int i = 1; i<new_idx.size(); ++i)
//    {
//        new_idx[i] += new_idx[i - 1];
//    }
//
//    for (auto &f : mesh.faces_)
//    {
//        for (int j = 0; j < 3; ++j)
//            f[j] = new_idx[f[j]] - 1;
//    }
    return true;
}

void MeshInfo::resolve_complex_vertex(TriangleMesh &mesh) {
    bool has_normal = mesh.vertex_normals_.size() != 0;
    bool has_color = mesh.vertex_colors_.size() != 0;
    bool has_status = mesh.vertex_status_.size() != 0;
    size_t num_vert = mesh.vertices_.size();
    for (size_t i = 0; i < num_vert; ++i) {
        auto& vert = mesh.vertices_.at(i);
        MeshInfo::VertexInfo& vert_info = this->at(i);
        if (vert_info.vclass != MeshInfo::VERTEX_CLASS_COMPLEX) {
            continue;
        }
        std::vector<std::vector<int> > adj_facets;
        this->get_clustered_adj_faces(mesh, i, adj_facets);
        for (size_t j = 1; j < adj_facets.size(); ++j) {
            int vert_id = mesh.vertices_.size();
            mesh.vertices_.emplace_back(vert);
            if (has_normal) {
                mesh.vertex_normals_.emplace_back(mesh.vertex_normals_.at(i));
            }
            if (has_color) {
                mesh.vertex_colors_.emplace_back(mesh.vertex_colors_.at(i));
            }
            if (has_status) {
                mesh.vertex_status_.emplace_back(mesh.vertex_status_.at(i));
            }

            for (auto f_id : adj_facets.at(j)) {
                auto& facet = mesh.faces_.at(f_id);
                if (facet.x() == i) {
                    facet.x() = vert_id;
                }
                if (facet.y() == i) {
                    facet.y() = vert_id;
                }
                if (facet.z() == i) {
                    facet.z() = vert_id;
                }
            }
        }
    }
}

bool MeshInfo::remove_complex_faces(TriangleMesh &mesh) {
    auto &verts = mesh.vertices_;
    auto &faces = mesh.faces_;
    auto &vinfos = this->vertex_info;
    std::vector<int> faces_to_delete;
    for(int i = 0; i < vinfos.size(); ++i){
        auto &info = vinfos[i];
        if(info.vclass == VERTEX_CLASS_COMPLEX){
            for(auto face : info.faces){
                faces_to_delete.push_back(face);
            }
        }
    }

    std::sort(faces_to_delete.begin(), faces_to_delete.end());
    auto it = std::unique(faces_to_delete.begin(), faces_to_delete.end());
    faces_to_delete.erase(it, faces_to_delete.end());

    UpdateMesh(mesh, faces_to_delete, false);

    this->clear();
    this->initialize(mesh);

    return !faces_to_delete.empty();
}

bool MeshInfo::remove_adj_border_faces(TriangleMesh &mesh) {
    auto &verts = mesh.vertices_;
    auto &faces = mesh.faces_;
    auto &vinfos = this->vertex_info;
    std::vector<int> faces_to_delete;
    for(int i = 0; i < faces.size(); ++i){
        int count = 0;
        for(int j = 0; j < 3; ++j ){
            int vert_id = faces[i][j];
            auto &info = vinfos[vert_id];
            if(info.vclass == VERTEX_CLASS_BORDER)
                count++;
        }
        if(count == 3)
            faces_to_delete.push_back(i);
    }

    UpdateMesh(mesh, faces_to_delete, false);

    this->clear();
    this->initialize(mesh);

    return !faces_to_delete.empty();
}

bool MeshInfo::remove_unref_vertices(TriangleMesh &mesh) {

    auto &verts = mesh.vertices_;
    auto &faces = mesh.faces_;
    auto &colors = mesh.vertex_colors_;
    auto &labels = mesh.vertex_labels_;
    auto &normals = mesh.vertex_normals_;

    auto &vinfos = this->vertex_info;
    std::vector<int> vertices_to_delete;
    for(int i = 0; i < vinfos.size(); ++i){
        auto &info = vinfos[i];
        if(info.vclass == VERTEX_CLASS_UNREF){
            vertices_to_delete.push_back(i);
        }
    }

    if(vertices_to_delete.empty())
        return false;

    std::vector<int> new_idx(verts.size());
    int j = 0;
    if(vertices_to_delete[0] == 0){
        new_idx[0] = -1;
        j++;
    } else {
        new_idx[0] = 0;
    }
    for(int i = 1; i < new_idx.size(); ++i){
        if(i == vertices_to_delete[j]){
            new_idx[i] = new_idx[i - 1];
            j++;
        } else {
            new_idx[i] = new_idx[i - 1] + 1;
        }
    }

    //delete vertices
    for(int i = vertices_to_delete.size() - 1; i >= 0; --i){
        verts.erase(verts.begin() + vertices_to_delete[i]);
    }
    if(!colors.empty()){
        for(int i = vertices_to_delete.size() - 1; i >= 0; --i){
            colors.erase(colors.begin() + vertices_to_delete[i]);
        }
    }
    if(!labels.empty()){
        for(int i = vertices_to_delete.size() - 1; i >= 0; --i){
            labels.erase(labels.begin() + vertices_to_delete[i]);
        }
    }
    if(!normals.empty()){
        for(int i = vertices_to_delete.size() - 1; i >= 0; --i){
            normals.erase(normals.begin() + vertices_to_delete[i]);
        }
    }

    //update faces
    for(int i = 0; i < faces.size(); ++i){
        for(int j = 0; j < 3; ++j){
            auto &vert_id = faces[i][j];
            vert_id = new_idx[vert_id];
        }
    }

    return true;
}

} //namespace sensemap 