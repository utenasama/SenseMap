//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_OBJ_H_
#define SENSEMAP_UTIL_OBJ_H_

#include <float.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <set>
#include "util/misc.h"
#include <Eigen/Geometry>

#include "types.h"

namespace sensemap {

enum VERTEX_STATUS {
    NORMAL = 0,
    UPDATE = 1,
    DELETE = 2
};

class TriangleMesh {
public:
    TriangleMesh()  {};
    ~TriangleMesh() {};

    void Clear();

    void RemoveIsolatedPieces(const int min_num_facet);
    void RemoveAbnormalFacets(std::vector<int>& border_verts_idx);
    void RemoveSelectFaces(const std::set<size_t>& face_set);

    void HollFill();
    void HollFill(std::vector<std::vector<int> > &lists);

    void ComputeNormals();

    void Clean(const float fDecimate = 0.7f, const float fSpurious = 10.0f, 
               const bool bRemoveSpikes = true, const unsigned nCloseHoles = 30, 
               const unsigned nSmooth = 2, const bool bLastClean = true);
    void ModifyNonMainfoldFace();
    bool FilterOutOfRangeFace(const float fSpurious);

    void AddMesh(const TriangleMesh& new_mesh);
    void Swap(TriangleMesh& new_mesh);

public:
    std::vector<Eigen::Vector3d> vertices_;
    std::vector<Eigen::Vector3d> vertex_normals_;
    std::vector<Eigen::Vector3d> vertex_colors_;
    std::vector<Eigen::Vector3i> faces_;
    std::vector<Eigen::Vector3d> face_normals_;
    std::vector<int8_t> vertex_labels_;
    std::vector<int8_t> vertex_status_;
    std::vector<std::vector<uint32_t> > vertex_visibilities_;
};

bool WriteTriangleMeshObj(const std::string &filename,
                          const TriangleMesh &mesh,
                          const bool write_as_rgb = true,
                          const bool write_as_sem = false);

bool ReadTriangleMeshObj(const std::string &filename,
                         TriangleMesh &mesh,
                         const bool read_as_rgb = false,
                         const bool read_as_sem = false);

struct MeshBox {
    float x_min = FLT_MAX;
    float y_min = FLT_MAX;
    float z_min = FLT_MAX;
    float x_max = -FLT_MAX;
    float y_max = -FLT_MAX;
    float z_max = -FLT_MAX;

    float box_x_min = FLT_MAX;
    float box_y_min = FLT_MAX;
    float box_z_min = FLT_MAX;
    float box_x_max = -FLT_MAX;
    float box_y_max = -FLT_MAX;
    float box_z_max = -FLT_MAX;

    float border_width = -1;
    float border_factor = -1;
    Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();

    bool IsValid()const {
        return bool (x_min < x_max && y_min < y_max);
    };
    void ReadBox(const std::string& path);
    void SetBoundary();
    void ResetBoundary(float scale_factor);
    void Print() const;
};

bool FilterWithBox(TriangleMesh &mesh,
                   const MeshBox box);

} // namespace sensemap

#endif