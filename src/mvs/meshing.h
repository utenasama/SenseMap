//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_MESHING_H_
#define SENSEMAP_MVS_MESHING_H_

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <Eigen/Dense>

#include "util/types.h"
#include "mvs/depth_map.h"
#include "mvs/image.h"
#include "mvs/workspace.h"

#define MAX_NUM_VOXEL 1000000ull

namespace sensemap {
namespace mvs {

enum MeshStatus_ { VISIBLE = 1, BOUNDARY = 2, OVERLAP = 4, DIRTY = 8 };
typedef MeshStatus_ MeshStatus;

typedef struct Facet {
    unsigned char delay = 0;
    int mIdx = -1;
    Eigen::Vector3i vIdx;
    Eigen::Vector3f fNormal = Eigen::Vector3f::Zero();
} Facet;
typedef std::shared_ptr<Facet> FacetPtr;

typedef struct Vertex {
    unsigned char border = 0;
    int mIdx = -1;
    Eigen::Vector3f vCoord;
    Eigen::Vector3f vNormal = Eigen::Vector3f::Zero();
} Vertex;
typedef std::shared_ptr<Vertex> VertexPtr;

typedef struct Voxel_ {
    enum Status { NORMAL = 1, ADD = 2, DIRTY = 4, DELETED = 8 };
    unsigned char stat = Status::NORMAL;
    int pos[3];
    uint64_t hash_idx = 0xFFFFFFFFFFFFFFFF;
    float tsdf = 0;
    float weight = 0;
    int voxel_ref_count = 0;
} Voxel;

typedef struct HashEntry_ {
    enum Status { NORMAL = 1, ADD = 2, DIRTY = 4, DELETED = 8 };
    unsigned char stat = Status::NORMAL;
    int pos[3];
    int voxel_idx[8];
    HashEntry_() {
        memset(voxel_idx, -1, sizeof(int) * 8);
    }
} HashEntry;

typedef struct Edge_ {
    enum Status { NORMAL = 1, DELETED = 8 };
    unsigned char stat = Status::NORMAL;
    int vert_idx = -1;
} Edge;

class VoxelHashing {

public:
    VoxelHashing(const Workspace::Options& options)
        : workspace_options_(options) {}

    void Run();

    void ExportToObj(const std::string& filename);

private:
    void ComputeVoxelLength(const Image& cur_frm,
                            const DepthMap& depth_map);

    void IntegrateDepth(const Image& cur_frm,
                        const DepthMap& depth_map);

    void UpdateVolume(const DepthMap& depth_map,
                      const Eigen::Matrix4f& proj_matrix,
                      const Eigen::Vector3f& X,
                      std::unordered_map<int, std::pair<float, float> >& bkproj_depth);

    void UpdateHashEntry(const Eigen::Vector3f &X, int voxel_idx[8]);

    void ExtractISOSurface();

private:
    Workspace::Options workspace_options_;

    float one_over_voxel_size = 0.2f;

    std::vector<Vertex> m_surfaceMeshVtxs_;
    std::vector<Facet> m_surfaceMeshFacets_;

    std::unordered_map<uint64_t, int> m_hashEntryMapIdx_;
    std::vector<std::shared_ptr<HashEntry> > m_hashEntry_;
    std::unordered_map<uint64_t, int> m_voxelMapIdx_;
    std::vector<std::shared_ptr<Voxel> > m_voxelList_;
    std::unordered_map<uint64_t, int> m_edge_vert_map_;
    std::vector<std::shared_ptr<Edge> > m_edgeList_;
};

Eigen::Vector3f VertexInterp(const Eigen::Vector3f &p1, 
                             const Eigen::Vector3f &p2, 
                             float val1, float val2);

int Polygonise(const std::shared_ptr<HashEntry> pEntry,
                      const std::vector<std::shared_ptr<Voxel> > &voxelList,
                      const float one_over_voxel_size,
                      std::vector<Vertex> &vert_list,
                      std::vector<Facet> &facet_list,
                      std::unordered_map<uint64_t, int> &edge_vert_map,
                      std::vector<std::shared_ptr<Edge> > &edgeList);

int PolygoniseWoNormal(const std::shared_ptr<HashEntry> pEntry,
                      const std::vector<std::shared_ptr<Voxel> > &voxelList,
                      const float one_over_voxel_size,
                      std::vector<Vertex> &vert_list,
                      std::vector<Facet> &facet_list,
                      std::unordered_map<uint64_t, int> &edge_vert_map,
                      std::vector<std::shared_ptr<Edge> > &edgeList);

#if 0
struct PoissonMeshingOptions {
    // This floating point value specifies the importance that interpolation of
    // the point samples is given in the formulation of the screened Poisson
    // equation. The results of the original (unscreened) Poisson Reconstruction
    // can be obtained by setting this value to 0.
    double point_weight = 1.0;

    // This integer is the maximum depth of the tree that will be used for surface
    // reconstruction. Running at depth d corresponds to solving on a voxel grid
    // whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the
    // reconstructor adapts the octree to the sampling density, the specified
    // reconstruction depth is only an upper bound.
    int depth = 10;

    // If specified, the reconstruction code assumes that the input is equipped
    // with colors and will extrapolate the color values to the vertices of the
    // reconstructed mesh. The floating point value specifies the relative
    // importance of finer color estimates over lower ones.
    double color = 32.0;

    // This floating point values specifies the value for mesh trimming. The
    // subset of the mesh with signal value less than the trim value is discarded.
    double trim = 8.0;

    // .
    bool verbose = true;

    // The number of threads used for the Poisson reconstruction.
    int num_threads = -1;

    int num_isolated_pieces = 0;

    bool Check() const;
};

// Perform Poisson surface reconstruction and return true if successful.
bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::string& input_path,
                    const std::string& output_path);
#endif

} // namespace mvs
} // namespace sensemap

#endif