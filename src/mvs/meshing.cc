#include <fstream>

#if 0
#include <PoissonRecon/PoissonRecon.h>
#include <PoissonRecon/SurfaceTrimmer.h>
#endif

#include "mvs/meshing.h"
#include "mvs/model.h"

// #define ONE_OVER_VOXEL_SIZE 0.2
// #define TSDF_THRES 2.0 * ONE_OVER_VOXEL_SIZE

#define MAX_VOXEL_SIDE 20ll
#define MAX_VOXEL_SLIDE (MAX_VOXEL_SIDE * MAX_VOXEL_SIDE)
#define MAX_VOXEL_CUBE (MAX_VOXEL_SIDE * MAX_VOXEL_SIDE * MAX_VOXEL_SIDE)

namespace sensemap {
namespace mvs {
namespace {
uint64_t GetHashIndex(int x, int y, int z) {
    int x_step = x >= 0 ? x * 2 / MAX_VOXEL_SIDE : -(x * 2 + 1) / MAX_VOXEL_SIDE;
    int y_step = y >= 0 ? y * 2 / MAX_VOXEL_SIDE : -(y * 2 + 1) / MAX_VOXEL_SIDE;
    int z_step = z >= 0 ? z * 2 / MAX_VOXEL_SIDE : -(z * 2 + 1) / MAX_VOXEL_SIDE;
    x_step = x_step + 1;
    y_step = y_step + 1;
    z_step = z_step + 1;
    int max_step = std::max(x_step, std::max(y_step, z_step));
    uint64_t max_voxel_side_offset = max_step * MAX_VOXEL_SIDE;    

    int x_offset = x + max_voxel_side_offset;
    int y_offset = y + max_voxel_side_offset;
    int z_offset = z + max_voxel_side_offset;

    uint64_t hash_idx = 0ll;
    for (int i = 1; i < max_step; ++i) {
        hash_idx += i * i * i * MAX_VOXEL_CUBE;
    }
    hash_idx += z_offset * max_voxel_side_offset * max_voxel_side_offset +
                y_offset * max_voxel_side_offset +
                x_offset;
    return hash_idx;
}   
}

void VoxelHashing::Run() {
    std::cout << "Reading workspace..." << std::endl;

    std::shared_ptr<Workspace> workspace_;    
    workspace_.reset(new Workspace(workspace_options_));

    const Model& model = workspace_->GetModel();
    for (size_t i = 0; i < model.images.size(); ++i) {
        std::cout << "Fusing Frame#" << i << std::endl;
        const mvs::Image& image = model.images.at(i);
        const mvs::DepthMap& depth_map = workspace_->GetDepthMap(i);

        if (i == 0) {
            ComputeVoxelLength(image, depth_map);
        }

        IntegrateDepth(image, depth_map);
    }

    ExtractISOSurface();

    std::cout << "m_surfaceMeshVtxs_.size() = " << m_surfaceMeshVtxs_.size() 
              << std::endl;
    std::cout << "m_surfaceMeshFacets_.size() = " << m_surfaceMeshFacets_.size()
              << std::endl;
}

void VoxelHashing::ComputeVoxelLength(const Image& cur_frm,
                                      const DepthMap& depth_map) {
    typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3f;
    
    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();
    
    const float* K = cur_frm.GetK();
    const Matrix3f Kinv = Matrix3f(K).inverse();
    const float* R_c_w = cur_frm.GetR();
    const float* T_c_w = cur_frm.GetT();
    const Matrix3f R_w_c = Matrix3f(R_c_w).transpose();
    const Eigen::Vector3f T_w_c = -R_w_c * Eigen::Vector3f(T_c_w);

    float min_d = std::numeric_limits<float>::max();
    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float depth0 = depth_map.Get(y - 1, x);
            const float depth1 = depth_map.Get(y, x);
            if (depth0 > 0 && depth1 > 0) {
                Eigen::Vector3f Xc1 = Kinv * Eigen::Vector3f(x, y - 1, 1.0f) * depth0;
                Eigen::Vector3f Xw1 = R_w_c * Xc1 + T_w_c;
                Eigen::Vector3f Xc2 = Kinv * Eigen::Vector3f(x, y, 1.0f) * depth1;
                Eigen::Vector3f Xw2 = R_w_c * Xc2 + T_w_c;

                float d = (Xw2 - Xw1).norm();
                min_d = std::min(min_d, d);
            }
        }
    }
    min_d *= 10;
    one_over_voxel_size = std::min(min_d, one_over_voxel_size);
    std::cout << "one_over_voxel_size = " << one_over_voxel_size
              << " / " << min_d << std::endl;
}

void VoxelHashing::IntegrateDepth(const Image& cur_frm,
                                  const DepthMap& depth_map) {
    typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3f;
    
    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();
    const int step = 1;
    const int step_height = (height - 1) / step + 1;
    const int step_width = (width - 1) / step + 1;

    const float* K = cur_frm.GetK();
    const Matrix3f Kinv = Matrix3f(K).inverse();
    const float* R_c_w = cur_frm.GetR();
    const float* T_c_w = cur_frm.GetT();
    const Matrix3f R_w_c = Matrix3f(R_c_w).transpose();
    const Eigen::Vector3f T_w_c = -R_w_c * Eigen::Vector3f(T_c_w);

    const float* P = cur_frm.GetP();
    Eigen::Matrix4f P_m;
    P_m << P[0], P[1], P[2], P[3],
           P[4], P[5], P[6], P[7],
           P[8], P[9], P[10], P[11],
           0, 0, 0, 1;

    std::unordered_map<int, std::pair<float, float> > bkproj_depth;

    // static int no = 0;
    // char buf[128];
    // sprintf(buf, "./depth%04d.obj", no++);
    // FILE *fp = fopen(buf, "w");

    for (int y = step, sy = 0; y < height; y += step, ++sy) {
        for (int x = step, sx = 0; x < width; x += step, ++sx) {
            const float depth = depth_map.Get(y, x);
            Eigen::Vector3f Xc = Kinv * Eigen::Vector3f(x, y, 1.0f) * depth;
            Eigen::Vector3f Xw = R_w_c * Xc + T_w_c;
            UpdateVolume(depth_map, P_m, Xw, bkproj_depth);

            // fprintf(fp, "v %f %f %f\n", Xw[0], Xw[1], Xw[2]);
        }
    }

    // fclose(fp);
}

void VoxelHashing::UpdateVolume(
    const DepthMap& depth_map,
    const Eigen::Matrix4f& proj_matrix,
    const Eigen::Vector3f& X,
    std::unordered_map<int, std::pair<float, float> >& bkproj_depth) {
    
    Eigen::Vector3f pos;
    pos[0] = X.x() / one_over_voxel_size;
    pos[1] = X.y() / one_over_voxel_size;
    pos[2] = X.z() / one_over_voxel_size;

    int xmin = std::floor(pos[0]);
    int ymin = std::floor(pos[1]);
    int zmin = std::floor(pos[2]);

    int corners[8][3] = {
        { xmin,              ymin,              zmin },
        { xmin + 1,          ymin,              zmin },
        { xmin + 1,          ymin + 1,          zmin },
        { xmin,              ymin + 1,          zmin },
        { xmin,              ymin,              zmin + 1 },
        { xmin + 1,          ymin,              zmin + 1 },
        { xmin + 1,          ymin + 1,          zmin + 1 },
        { xmin,              ymin + 1,          zmin + 1 }
    };

    int voxel_idx[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };

    Eigen::Vector4f hX, proj;
    bool invalid_entry = false;
    int voxel_num = m_voxelList_.size();

    uint64_t entry_hash_idx = GetHashIndex(xmin, ymin, zmin);
    std::unordered_map<uint64_t, int>::iterator entry_it = 
        m_hashEntryMapIdx_.find(entry_hash_idx);
    int entryId = -1;
    if (entry_it != m_hashEntryMapIdx_.end())
    {
        entryId = entry_it->second;
    }
    int voxelId;
    for (int i = 0; !invalid_entry && (i < 8); ++i) {
        int* corner = corners[i];
        uint64_t voxel_hash_idx = GetHashIndex(corner[0], corner[1], corner[2]);

        voxelId = -1;
        if (-1 != entryId)
        {
            voxelId = m_hashEntry_[entryId]->voxel_idx[i];
        }
        else
        {
            std::unordered_map<uint64_t, int>::iterator voxel_it = 
                m_voxelMapIdx_.find(voxel_hash_idx);
            if (voxel_it != m_voxelMapIdx_.end()) {
                voxelId = voxel_it->second;
            }
        }

        if (voxelId != -1) {
            std::shared_ptr<Voxel> &pVoxel = m_voxelList_[voxelId];
            pVoxel->hash_idx = voxel_hash_idx;
            voxel_idx[i] = voxelId;

            float proj_depth, d = 0.0f;
            std::unordered_map<int, std::pair<float, float> >::iterator 
            bkproj_iter = bkproj_depth.find(voxelId);
            
            if (bkproj_iter != bkproj_depth.end()) {
                d = bkproj_iter->second.first;
                proj_depth = bkproj_iter->second.second;
            } else {
                hX = Eigen::Vector4f(corner[0] * one_over_voxel_size, 
                                     corner[1] * one_over_voxel_size, 
                                     corner[2] * one_over_voxel_size, 
                                     1.0f);
                proj = proj_matrix * hX;
                proj /= proj[3];

                proj_depth = proj[2];

                int u = proj[0] / proj[2] + 0.5f;
                int v = proj[1] / proj[2] + 0.5f;
                if (u >= 0 && u < depth_map.GetWidth() && 
                    v >= 0 && v < depth_map.GetHeight()) {
                    d = depth_map.Get(v, u);
                }

                bkproj_depth[voxelId] = std::make_pair(d, proj[2]);
            }

            if (d > 0.0f) {
                float tsdf = proj_depth - d;
                pVoxel->tsdf = (pVoxel->tsdf * pVoxel->weight + tsdf) / 
                               (pVoxel->weight + 1.0f);
                pVoxel->weight = pVoxel->weight + 1.0f;
                pVoxel->voxel_ref_count++;
                if (pVoxel->stat != Voxel::Status::ADD) {
                    pVoxel->stat = Voxel::Status::DIRTY;
                }
            }
        }
        else {
            hX = Eigen::Vector4f(corner[0] * one_over_voxel_size, 
                                 corner[1] * one_over_voxel_size, 
                                 corner[2] * one_over_voxel_size, 
                                 1.0f);
            proj = proj_matrix * hX;
            proj /= proj[3];

            int u = proj[0] / proj[2] + 0.5f;
            int v = proj[1] / proj[2] + 0.5f;
            float d = 0.0f;
            if (u >= 0 && u < depth_map.GetWidth() && 
                v >= 0 && v < depth_map.GetHeight()) {
                d = depth_map.Get(v, u);
            }
            if (d <= 0.0f) {
                invalid_entry = true;
                break;
            }
            
            voxelId = m_voxelList_.size();
            if (voxelId >= MAX_NUM_VOXEL) {
                invalid_entry = true;
                break;
            }
            std::shared_ptr<Voxel> pVoxel = std::shared_ptr<Voxel>(new Voxel());
            m_voxelList_.emplace_back(pVoxel);

            pVoxel->pos[0] = corner[0];
            pVoxel->pos[1] = corner[1];
            pVoxel->pos[2] = corner[2];
            pVoxel->hash_idx = voxel_hash_idx;
            pVoxel->tsdf = proj[2] - d;
            pVoxel->weight = 1.0f;
            pVoxel->voxel_ref_count = 1;
            pVoxel->stat = Voxel::Status::ADD;

            m_voxelMapIdx_[voxel_hash_idx] = voxelId;
            voxel_idx[i] = voxelId;
            bkproj_depth[voxelId] = std::make_pair(d, proj[2]);
        }
    }

    if (invalid_entry) {
        return;
    }
    
    UpdateHashEntry(pos, voxel_idx);
}

void VoxelHashing::UpdateHashEntry(const Eigen::Vector3f &X, int voxel_idx[8]) {
    int xmin = std::floor(X[0]);
    int ymin = std::floor(X[1]);
    int zmin = std::floor(X[2]);

    uint64_t entry_hash_idx = GetHashIndex(xmin, ymin, zmin);
    std::unordered_map<uint64_t, int>::iterator entry_it = 
        m_hashEntryMapIdx_.find(entry_hash_idx);

    if (entry_it != m_hashEntryMapIdx_.end()) {
        std::shared_ptr<HashEntry> &pEntry = m_hashEntry_[entry_it->second];
        if (pEntry->stat != HashEntry::Status::ADD) {
            pEntry->stat = HashEntry::Status::DIRTY;
        }
    }
    else {
        std::shared_ptr<HashEntry> pEntry = 
            std::shared_ptr<HashEntry>(new HashEntry());
        pEntry->stat = HashEntry::Status::ADD;
        pEntry->pos[0] = xmin;
        pEntry->pos[1] = ymin;
        pEntry->pos[2] = zmin;
        memcpy(pEntry->voxel_idx, voxel_idx, sizeof(int) * 8);

        m_hashEntryMapIdx_[entry_hash_idx] = m_hashEntry_.size();
        m_hashEntry_.emplace_back(pEntry);
    }
}

void VoxelHashing::ExtractISOSurface() {
    bool m_bComputeNormal = true;
    if (m_bComputeNormal)
    {
        for (auto pEntry : m_hashEntry_) {
            int* voxel_idx = pEntry->voxel_idx;
            char stat = ((m_voxelList_[voxel_idx[0]]->stat) |
                                    (m_voxelList_[voxel_idx[1]]->stat) |
                                    (m_voxelList_[voxel_idx[2]]->stat) |
                                    (m_voxelList_[voxel_idx[3]]->stat) |
                                    (m_voxelList_[voxel_idx[4]]->stat) |
                                    (m_voxelList_[voxel_idx[5]]->stat) |
                                    (m_voxelList_[voxel_idx[6]]->stat) |
                                    (m_voxelList_[voxel_idx[7]]->stat));

            if (stat & Voxel::Status::DELETED) {
                pEntry->stat = HashEntry::Status::DELETED;
            }
            if ((stat & Voxel::Status::DIRTY) && 
                (pEntry->stat != HashEntry::Status::DELETED)) {
                pEntry->stat = HashEntry::Status::DIRTY;
            }
            if (pEntry->stat == HashEntry::Status::NORMAL)
                continue;

            if (pEntry->stat != HashEntry::Status::DELETED) {
                Polygonise(pEntry, m_voxelList_, one_over_voxel_size, 
                    m_surfaceMeshVtxs_, m_surfaceMeshFacets_, m_edge_vert_map_, 
                    m_edgeList_);

                // reset entry status
                pEntry->stat = HashEntry::Status::NORMAL;
            }
        }
    }
    else
    {
        for (auto pEntry : m_hashEntry_) {
            int* voxel_idx = pEntry->voxel_idx;
            char stat = ((m_voxelList_[voxel_idx[0]]->stat) |
                                    (m_voxelList_[voxel_idx[1]]->stat) |
                                    (m_voxelList_[voxel_idx[2]]->stat) |
                                    (m_voxelList_[voxel_idx[3]]->stat) |
                                    (m_voxelList_[voxel_idx[4]]->stat) |
                                    (m_voxelList_[voxel_idx[5]]->stat) |
                                    (m_voxelList_[voxel_idx[6]]->stat) |
                                    (m_voxelList_[voxel_idx[7]]->stat));

            if (stat & Voxel::Status::DELETED) {
                pEntry->stat = HashEntry::Status::DELETED;
            }
            if ((stat & Voxel::Status::DIRTY) && 
                (pEntry->stat != HashEntry::Status::DELETED)) {
                pEntry->stat = HashEntry::Status::DIRTY;
            }
            if (pEntry->stat == HashEntry::Status::NORMAL)
                continue;

            if (pEntry->stat != HashEntry::Status::DELETED) {
                PolygoniseWoNormal(pEntry, m_voxelList_, one_over_voxel_size, 
                    m_surfaceMeshVtxs_, m_surfaceMeshFacets_, m_edge_vert_map_, 
                    m_edgeList_);

                // reset entry status
                pEntry->stat = HashEntry::Status::NORMAL;
            }
        }
    }

    for (auto & pVoxel : m_voxelList_) {
        if (pVoxel->stat == Voxel::Status::DELETED ||
            pVoxel->stat == Voxel::Status::NORMAL)
            continue;

        pVoxel->stat = Voxel::Status::NORMAL;
    }
}

void VoxelHashing::ExportToObj(const std::string& filename) {
    std::ofstream ofs(filename, std::ofstream::out);

    for (const auto& vtx : m_surfaceMeshVtxs_) {
        ofs << "v " << vtx.vCoord[0] << " " << vtx.vCoord[1]
            << " " << vtx.vCoord[2] << std::endl;
    }
    for (const auto& facet : m_surfaceMeshFacets_) {
        ofs << "f " << facet.vIdx[0] + 1 << " " << facet.vIdx[2] + 1
            << " " << facet.vIdx[1] + 1 << std::endl;
    }

    ofs.close();
}

#if 0
bool PoissonMeshingOptions::Check() const {
    CHECK_OPTION_GE(point_weight, 0);
    CHECK_OPTION_GT(depth, 0);
    CHECK_OPTION_GE(color, 0);
    CHECK_OPTION_GE(trim, 0);
    CHECK_OPTION_GE(num_threads, -1);
    CHECK_OPTION_NE(num_threads, 0);
    return true;
}

bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::string& input_path,
                    const std::string& output_path) {
    CHECK(options.Check());

    std::vector<std::string> args;

    args.push_back("./binary");

    args.push_back("--in");
    args.push_back(input_path);

    args.push_back("--out");
    args.push_back(output_path);

    args.push_back("--pointWeight");
    args.push_back(std::to_string(options.point_weight));

    args.push_back("--depth");
    args.push_back(std::to_string(options.depth));

    if (options.color > 0) {
        args.push_back("--color");
        args.push_back(std::to_string(options.color));
    }

    #ifdef OPENMP_ENABLED
    if (options.num_threads > 0) {
        args.push_back("--threads");
        args.push_back(std::to_string(options.num_threads));
    }
    #endif  // OPENMP_ENABLED

    if (options.trim > 0) {
        args.push_back("--density");
    }

    if (options.verbose) {
        args.push_back("--verbose");
    }

    std::vector<const char*> args_cstr;
    args_cstr.reserve(args.size());
    for (const auto& arg : args) {
        args_cstr.push_back(arg.c_str());
    }

    if (PoissonRecon(args_cstr.size(), const_cast<char**>(args_cstr.data())) !=
        EXIT_SUCCESS) {
        return false;
    }

    if (options.trim == 0) {
        return true;
    }

    args.clear();
    args_cstr.clear();

    args.push_back("./binary");

    args.push_back("--in");
    args.push_back(output_path);

    args.push_back("--out");
    args.push_back(output_path);

    args.push_back("--trim");
    args.push_back(std::to_string(options.trim));

    if (options.verbose) {
        args.push_back("--verbose");
    }

    args_cstr.reserve(args.size());
    for (const auto& arg : args) {
        args_cstr.push_back(arg.c_str());
    }

    return SurfaceTrimmer(args_cstr.size(),
                            const_cast<char**>(args_cstr.data())) == EXIT_SUCCESS;
}
#endif

}
}