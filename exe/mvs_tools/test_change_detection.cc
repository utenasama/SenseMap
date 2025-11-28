#include "util/misc.h"
#include "util/threading.h"
#include "util/depth_mesh.h"
#include "util/rgbd_helper.h"
#include "util/imageconvert.h"
#include "util/exception_handler.h"
#include "base/common.h"
#include "base/pose.h"
#include "base/undistortion.h"
#include "base/reconstruction.h"
#include "base/reconstruction_manager.h"
#include "mvs/workspace.h"
#include "mvs/integration/ScalableTSDFVolume.h"
#include "base/version.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

typedef tsdf::ScalableTSDFVolume<tsdf::ColoredTSDFVolume<uint8_t>> stsdfv_t;

std::shared_ptr<TriangleMesh> ExtractFlag(
    std::shared_ptr<stsdfv_t> volume,
    float weight_thresh)
{
    auto mesh = std::make_shared<TriangleMesh>();
    for (auto & v : volume->volume_units_) {
        const auto &volume0 = *v.second.volume_;
        Eigen::Vector3d origin = v.first.cast<double>() * volume->volume_unit_length_;
        for (int x = 0; x < volume0.resolution_; x++) {
            for (int y = 0; y < volume0.resolution_; y++) {
                for (int z = 0; z < volume0.resolution_; z++) {
                    Eigen::Vector3i idx0(x, y, z);
                    uint8_t u = volume0.extra_[volume0.IndexOf(idx0)];
                    float w = volume0.weight_[volume0.IndexOf(idx0)];

                    if (w > weight_thresh) {
                        Eigen::Vector3d pt = origin + idx0.cast<double>() * volume->voxel_length_;
                        if (u) {
                            mesh->vertices_.push_back(pt);
                            mesh->vertex_colors_.emplace_back(127, 0, 0);
                        }
                    }
                }
            }
        }
    }
    return mesh;
}

std::shared_ptr<TriangleMesh> ExtractFlagMesh(
    std::shared_ptr<stsdfv_t> volume,
    float weight_thresh)
{
    auto mesh = std::make_shared<TriangleMesh>();
    float half_voxel_length = volume->voxel_length_ * 0.5;
    std::unordered_map<Eigen::Vector4i, int, tsdf::hash<Eigen::Vector4i>>
        edgeindex_to_vertexindex;
    int edge_to_index[12];
    for (auto v : volume->volume_units_) {
        const auto index0 = v.first;
        const auto &volume0 = *v.second.volume_;
        for (int x = 0; x < volume0.resolution_; x++) {
            for (int y = 0; y < volume0.resolution_; y++) {
                for (int z = 0; z < volume0.resolution_; z++) {
                    Eigen::Vector3i idx0(x, y, z);
                    int cube_index = 0;
                    unsigned char u[8];
                    float w[8];
                    for (int i = 0; i < 8; i++) {
                        Eigen::Vector3i index1 = index0;
                        Eigen::Vector3i idx1 = idx0 + shift[i];
                        if (idx1(0) < volume->volume_unit_resolution_ &&
                            idx1(1) < volume->volume_unit_resolution_ &&
                            idx1(2) < volume->volume_unit_resolution_) {
                            u[i] = volume0.extra_[volume0.IndexOf(idx1)];
                            w[i] = volume0.weight_[volume0.IndexOf(idx1)];
                        } else {
                            for (int j = 0; j < 3; j++) {
                                if (idx1(j) >= volume->volume_unit_resolution_) {
                                    idx1(j) -= volume->volume_unit_resolution_;
                                    index1(j) += 1;
                                }
                            }
                            auto unit_itr1 = volume->volume_units_.find(index1);
                            if (unit_itr1 == volume->volume_units_.end()) {
                                u[i] = 0;
                                w[i] = std::numeric_limits<float>::max();
                            } else {
                                const auto &volume1 = *unit_itr1->second.volume_;
                                u[i] = volume1.extra_[volume1.IndexOf(idx1)];
                                w[i] = volume1.weight_[volume1.IndexOf(idx1)];
                            }
                        }
                        if (w[i] <= weight_thresh) {
                            cube_index = 0;
                            break;
                        } else {
                            if (u[i]) {
                                cube_index |= (1 << i);
                            }
                        }
                    }
                    if (cube_index == 0 || cube_index == 255) {
                        continue;
                    }
                    for (int i = 0; i < 12; i++) {
                        if (edge_table[cube_index] & (1 << i)) {
                            Eigen::Vector4i edge_index = Eigen::Vector4i(
                                    index0(0), index0(1), index0(2), 0) *
                                    volume->volume_unit_resolution_ +
                                    Eigen::Vector4i(x, y, z, 0) +
                                    edge_shift[i];
                            if (edgeindex_to_vertexindex.find(edge_index) ==
                                    edgeindex_to_vertexindex.end()) {
                                edge_to_index[i] =
                                        (int)mesh->vertices_.size();
                                edgeindex_to_vertexindex[edge_index] =
                                        (int)mesh->vertices_.size();
                                Eigen::Vector3d pt(
                                        volume->voxel_length_ * edge_index(0),
                                        volume->voxel_length_ * edge_index(1),
                                        volume->voxel_length_ * edge_index(2));
                                float f0 = u[edge_to_vert[i][0]] ? 1.0f : 0.0f;
                                float f1 = u[edge_to_vert[i][1]] ? 1.0f : 0.0f;
                                pt(edge_index(3)) += f0 * volume->voxel_length_ / (f0 + f1);
                                mesh->vertices_.push_back(pt);
                            } else {
                                edge_to_index[i] =
                                        edgeindex_to_vertexindex[edge_index];
                            }
                        }
                    }
                    for (int i = 0; tri_table[cube_index][i] != -1; i += 3)
                    {
                        mesh->faces_.push_back(Eigen::Vector3i(
                            edge_to_index[tri_table[cube_index][i]],
                            edge_to_index[tri_table[cube_index][i + 2]],
                            edge_to_index[tri_table[cube_index][i + 1]]));
                    }
                }
            }
        }
    };
    return mesh;
}

bool FilterFlag3D(
    std::shared_ptr<stsdfv_t> volume,
    int window_size = 1,
    int zero_thresh = 12
) {
    const int resolution = volume->volume_unit_resolution_;
    std::unordered_map<Eigen::Vector3i, std::vector<uint8_t>, tsdf::hash<Eigen::Vector3i>> temp_flags;
    for (auto & v : volume->volume_units_) {
        auto & v0 = v.second;
        temp_flags[v.first] = std::vector<uint8_t>(resolution * resolution * resolution, 0);
    }

    int nonzero_thresh = std::max(0, 1 + window_size + window_size) * (1 + window_size + window_size) * (1 + window_size + window_size) - zero_thresh;
    #pragma omp parallel
    {
    #pragma omp single
    {
    for (auto & v : volume->volume_units_) {
        const Eigen::Vector3i index0 = v.first;
        auto & volume0 = *v.second.volume_;
        auto & flags0 = temp_flags[v.first];

        auto work = [&]() {
        for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    const Eigen::Vector3i idx0(x, y, z);
                    uint8_t final_flag = 0;
                    
                    int count_zero = 0;
                    int count_nonzero = 0;
                    for (int xx = x-window_size; xx <= x+window_size; xx++) {
                        if (count_zero >= zero_thresh) break;
                        if (count_nonzero > nonzero_thresh) break;
                        for (int yy = y-window_size; yy <= y+window_size; yy++) {
                            if (count_zero >= zero_thresh) break;
                            if (count_nonzero > nonzero_thresh) break;
                            for (int zz = z-window_size; zz <= z+window_size; zz++) {
                                Eigen::Vector3i index1 = index0;
                                Eigen::Vector3i idx1(xx, yy, zz);
                                bool changed = false;
                                for (int j = 0; j < 3; j++) {
                                    while (idx1(j) >= resolution) {
                                        idx1(j) -= resolution;
                                        index1(j) += 1;
                                        changed = true;
                                    }
                                    while (idx1(j) < 0) {
                                        idx1(j) += resolution;
                                        index1(j) -= 1;
                                        changed = true;
                                    }
                                }

                                uint8_t flag = 0;
                                if (changed) {
                                    auto unit_itr1 = volume->volume_units_.find(index1);
                                    if (unit_itr1 != volume->volume_units_.end()) {
                                        const auto &volume1 = *unit_itr1->second.volume_;
                                        flag = volume1.extra_[volume1.IndexOf(idx1)];
                                    }
                                }
                                else {
                                    flag = volume0.extra_[volume0.IndexOf(idx1)];
                                }

                                if (flag == 0) {
                                    count_zero++;
                                } else {
                                    count_nonzero++;
                                    final_flag = flag;
                                }
                            }
                        }
                    }

                    if (count_zero >= zero_thresh) final_flag = 0;
                    flags0[volume0.IndexOf(idx0)] = final_flag;
                }
            }
        }
        };

        #pragma omp task
        work();
    }
    #pragma omp taskwait
    }
    }

    for (auto & v : volume->volume_units_) {
        std::swap(v.second.volume_->extra_, temp_flags[v.first]);
    }

    return true;
}

bool ChangeDetectionMarkFlag(
    std::shared_ptr<stsdfv_t> volume0, 
    std::shared_ptr<stsdfv_t> volume1,
    float coarse_weight_thresh, float coarse_change_thresh
) {
    if (volume0->sdf_trunc_ != volume1->sdf_trunc_) return false;
    if (volume0->voxel_length_ != volume1->voxel_length_) return false;

    for (auto & v : volume1->volume_units_) {
        if (volume0->volume_units_.count(v.first)) {
            auto & v0 = volume0->volume_units_[v.first];
            auto & v1 = v.second;
            for (int i = 0; i < v0.volume_->resolution_ * v0.volume_->resolution_ * v0.volume_->resolution_; i++) {
                float t0 = v0.volume_->tsdf_[i];
                float t1 = v1.volume_->tsdf_[i];
                float w0 = v0.volume_->weight_[i];
                float w1 = v1.volume_->weight_[i];
                if (w1 > coarse_weight_thresh) {
                    if (t0 - t1 > coarse_change_thresh / volume0->sdf_trunc_) {
                        // Dynamic, 0 > 1
                        v0.volume_->extra_[i] = 1;
                    } else if (t1 - t0 > coarse_change_thresh / volume0->sdf_trunc_) {
                        // Dynamic, 1 > 0
                        v0.volume_->extra_[i] = 1;
                    }
                }
            }
        }
    }

    return true;
}

bool ChangeDetectionMergeTSDF(
    std::shared_ptr<stsdfv_t> volume0, 
    std::shared_ptr<stsdfv_t> volume1,
    float coarse_weight_thresh, float coarse_change_thresh
) {
    if (volume0->sdf_trunc_ != volume1->sdf_trunc_) return false;
    if (volume0->voxel_length_ != volume1->voxel_length_) return false;

    auto resetFlag = [&]() {
        for (auto & v : volume0->volume_units_) {
            auto & v0 = v.second;
            for (int i = 0; i < v0.volume_->resolution_ * v0.volume_->resolution_ * v0.volume_->resolution_; i++) {
                v0.volume_->extra_[i] = 0;
            }
        }
    };
    resetFlag();

    // // Debug
    // auto resetColorByFlag = [&]() {
    //     for (auto & v : volume0->volume_units_) {
    //         auto & v0 = v.second;
    //         for (int i = 0; i < v0.volume_->resolution_ * v0.volume_->resolution_ * v0.volume_->resolution_; i++) {
    //             if (v0.volume_->extra_[i]) {
    //                 v0.volume_->color_[i] = Eigen::Vector3f(255, 0, 0);
    //             } else {
    //                 v0.volume_->color_[i] = Eigen::Vector3f(127, 127, 127);
    //             }
    //         }
    //     }
    // };

    for (auto & v : volume1->volume_units_) {
        if (volume0->volume_units_.count(v.first)) {
            auto & v0 = volume0->volume_units_[v.first];
            auto & v1 = v.second;
            for (int i = 0; i < v0.volume_->resolution_ * v0.volume_->resolution_ * v0.volume_->resolution_; i++) {
                float t0 = v0.volume_->tsdf_[i];
                float t1 = v1.volume_->tsdf_[i];
                float w0 = v0.volume_->weight_[i];
                float w1 = v1.volume_->weight_[i];
                if (w1 > coarse_weight_thresh) {
                    if (t0 - t1 > coarse_change_thresh / volume0->sdf_trunc_) {
                        // Dynamic, 0 > 1
                        v0.volume_->extra_[i] = 1;
                    } else if (t1 - t0 > coarse_change_thresh / volume0->sdf_trunc_) {
                        // Dynamic, 1 > 0
                        v0.volume_->extra_[i] = 1;
                    }
                }
            }
        }
    }

    // Debug
    // static int iter = -1;
    // std::string name;
    // iter++;
    // name = "iter_" + std::to_string(iter) + "_volume0a.obj";
    // WriteTriangleMeshObj(name.c_str(), *volume0->ExtractColoredTriangleMesh());
    // // resetColorByFlag();
    // name = "iter_" + std::to_string(iter) + "_volume0b.obj";
    // WriteTriangleMeshObj(name.c_str(), *ExtractFlag(volume0, 1.0f));
    // // FastFilterFlag3D<true>(volume0, 1, 1);
    // // FastFilterFlag3D<false>(volume0);
    FilterFlag3D(volume0, 1, 12);
    FilterFlag3D(volume0, 1, 25);
    // // resetColorByFlag();
    // name = "iter_" + std::to_string(iter) + "_volume0c.obj";
    // WriteTriangleMeshObj(name.c_str(), *ExtractFlag(volume0, 1.0f));

    size_t merge = 0, replace = 0, copy = 0;
    for (auto & v : volume1->volume_units_) {
        if (volume0->volume_units_.count(v.first)) {
            auto & v0 = volume0->volume_units_[v.first];
            auto & v1 = v.second;
            for (int i = 0; i < v0.volume_->resolution_ * v0.volume_->resolution_ * v0.volume_->resolution_; i++) {
                float t0 = v0.volume_->tsdf_[i];
                float t1 = v1.volume_->tsdf_[i];
                float w0 = v0.volume_->weight_[i];
                float w1 = v1.volume_->weight_[i];
                auto u0 = v0.volume_->extra_[i];
                if (u0 != 0) {
                    // Dynamic
                    if (t1 > t0) {
                        replace++;
                        v0.volume_->weight_[i] = w1;
                        v0.volume_->tsdf_[i] = t1;
                    }
                } else {
                    // Static
                    merge++;
                    v0.volume_->weight_[i] = w0 + w1;
                    v0.volume_->tsdf_[i] = (t0 * w0 + t1 * w1) / (w0 + w1);
                }
            }
        }
        else {
            // Copy
            copy += 1;
            volume0->OpenVolumeUnit(v.first);
            *volume0->volume_units_[v.first].volume_ = *v.second.volume_;
        }
    }

    // // Debug
    // name = "iter_" + std::to_string(iter) + "_volume0d.obj";
    // WriteTriangleMeshObj(name.c_str(), *volume0->ExtractColoredTriangleMesh());
    // name = "iter_" + std::to_string(iter) + "_volume1.obj";
    // WriteTriangleMeshObj(name.c_str(), *volume1->ExtractColoredTriangleMesh());

    std::cout << merge << " voxels merged, " << replace << " voxels replaced, " << copy << " volumes copied. " << std::endl;
    return true;
}

std::vector<cv::Mat> ProjectFlagMesh(
    const Image & image,
    const Camera & camera,
    const TriangleMesh & mesh,
    const TriangleMesh & fmesh
) {
    std::vector<cv::Mat> masks;

    Eigen::Matrix3d R = QuaternionToRotationMatrix(image.Qvec());
    Eigen::Vector3d t = image.Tvec();
    std::vector<Eigen::Vector3f> vertices(mesh.vertices_.size());
    std::vector<Eigen::Vector3f> fvertices(fmesh.vertices_.size());
    if (camera.NumLocalCameras() > 1) {
        for (int local_cam_id = 0; local_cam_id < camera.NumLocalCameras(); local_cam_id++) {
            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;
            camera.GetLocalCameraExtrinsic(local_cam_id, local_qvec, local_tvec);
            Eigen::Matrix3d R2 = QuaternionToRotationMatrix(local_qvec) * R;
            Eigen::Vector3d t2 = QuaternionToRotationMatrix(local_qvec) * t + local_tvec;

            std::vector<double> params;
            camera.GetLocalCameraIntrisic(local_cam_id, params);
            Camera camera2 = camera;
            camera2.SetParams(params);

            auto st = std::chrono::steady_clock::now();
            for (int i = 0; i < vertices.size(); i++) {
                const Eigen::Vector3d proj_point3D = R2 * mesh.vertices_[i] + t2;
                if (proj_point3D.z() > 0.0) {
                    Eigen::Vector2d point2D = camera2.WorldToImage(proj_point3D.hnormalized());
                    vertices[i] = Eigen::Vector3f(point2D.x(), point2D.y(), proj_point3D.z());
                } else {
                    vertices[i] = Eigen::Vector3f::Zero();
                }
            }
            for (int i = 0; i < fvertices.size(); i++) {
                const Eigen::Vector3d proj_point3D = R2 * fmesh.vertices_[i] + t2;
                if (proj_point3D.z() > 0.0) {
                    Eigen::Vector2d point2D = camera2.WorldToImage(proj_point3D.hnormalized());
                    fvertices[i] = Eigen::Vector3f(point2D.x(), point2D.y(), proj_point3D.z());
                } else {
                    fvertices[i] = Eigen::Vector3f::Zero();
                }
            }
            auto ed1 = std::chrono::steady_clock::now();

            cv::Mat depth(camera.Height(), camera.Width(), CV_32FC1);
            cv::Mat fdepth(camera.Height(), camera.Width(), CV_32FC1);
            WarpFrameBuffer(depth, vertices, mesh.faces_);
            WarpFrameBuffer(fdepth, fvertices, fmesh.faces_);
            auto ed2 = std::chrono::steady_clock::now();

            cv::Mat mask = cv::Mat::zeros(camera.Height(), camera.Width(), CV_8UC1);
            for (int y = 0; y < depth.rows; y++) {
                for (int x = 0; x < depth.cols; x++) {
                    float d = depth.at<float>(y, x);
                    float f = fdepth.at<float>(y, x);
                    if (f > 0.3f && d > 0.0f && f < d) {
                        mask.at<uchar>(y, x) = 255;
                    }
                }
            }
            masks.emplace_back(mask);
        }
    } else {
        auto st = std::chrono::steady_clock::now();
        for (int i = 0; i < vertices.size(); i++) {
            const Eigen::Vector3d proj_point3D = R * mesh.vertices_[i] + t;
            if (proj_point3D.z() > 0.0) {
                Eigen::Vector2d point2D = camera.WorldToImage(proj_point3D.hnormalized());
                vertices[i] = Eigen::Vector3f(point2D.x(), point2D.y(), proj_point3D.z());
            } else {
                vertices[i] = Eigen::Vector3f::Zero();
            }
        }
        for (int i = 0; i < fvertices.size(); i++) {
            const Eigen::Vector3d proj_point3D = R * fmesh.vertices_[i] + t;
            if (proj_point3D.z() > 0.0) {
                Eigen::Vector2d point2D = camera.WorldToImage(proj_point3D.hnormalized());
                fvertices[i] = Eigen::Vector3f(point2D.x(), point2D.y(), proj_point3D.z());
            } else {
                fvertices[i] = Eigen::Vector3f::Zero();
            }
        }
        auto ed1 = std::chrono::steady_clock::now();

        cv::Mat depth(camera.Height(), camera.Width(), CV_32FC1);
        cv::Mat fdepth(camera.Height(), camera.Width(), CV_32FC1);
        WarpFrameBuffer(depth, vertices, mesh.faces_);
        WarpFrameBuffer(fdepth, fvertices, fmesh.faces_);
        auto ed2 = std::chrono::steady_clock::now();

        cv::Mat mask = cv::Mat::zeros(camera.Height(), camera.Width(), CV_8UC1);
        for (int y = 0; y < depth.rows; y++) {
            for (int x = 0; x < depth.cols; x++) {
                float d = depth.at<float>(y, x);
                float f = fdepth.at<float>(y, x);
                if (f > 0.3f && d > 0.0f && f < d) {
                    mask.at<uchar>(y, x) = 255;
                }
            }
        }
        masks.emplace_back(mask);

        cv::Mat color;
        std::string name = JoinPaths("normal-images", image.Name());
        if (IsFileRGBD(name)) {
            Bitmap bitmap;
            ExtractRGBDData(name, bitmap);
            FreeImage2Mat(&bitmap, color);
        } else {
            color = cv::imread(name);
        }
        for (int y = 0; y < depth.rows; y++) {
            for (int x = 0; x < depth.cols; x++) {
                if (mask.at<uchar>(y, x)) {
                    color.at<cv::Vec3b>(y, x)[2] = std::min(255, color.at<cv::Vec3b>(y, x)[2] + 64);
                }
            }
        }
        std::string output_name = JoinPaths("masked-images", image.Name());
        if (!ExistsDir(GetParentDir(output_name))) {
            boost::filesystem::create_directories(boost::filesystem::path(GetParentDir(output_name)));
        }
        cv::imwrite(output_name, color);
    }

    return masks;
}

std::vector<cv::Mat> ProjectMeshMakeMask(
    const Image & image,
    const Camera & camera,
    const TriangleMesh & mesh,
    const TriangleMesh & bmesh,
    float relative_thresh = 0.05f,
    float absolute_thresh = 0.0f
) {
    std::vector<cv::Mat> masks;

    Eigen::Matrix3d R = QuaternionToRotationMatrix(image.Qvec());
    Eigen::Vector3d t = image.Tvec();
    std::vector<Eigen::Vector3f> vertices(mesh.vertices_.size(), Eigen::Vector3f::Zero());
    std::vector<Eigen::Vector3f> bvertices(bmesh.vertices_.size(), Eigen::Vector3f::Zero());
    if (camera.NumLocalCameras() > 1) {
        for (int local_cam_id = 0; local_cam_id < camera.NumLocalCameras(); local_cam_id++) {
            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;
            camera.GetLocalCameraExtrinsic(local_cam_id, local_qvec, local_tvec);
            Eigen::Matrix3d R2 = QuaternionToRotationMatrix(local_qvec) * R;
            Eigen::Vector3d t2 = QuaternionToRotationMatrix(local_qvec) * t + local_tvec;

            std::vector<double> params;
            camera.GetLocalCameraIntrisic(local_cam_id, params);
            Camera camera2 = camera;
            camera2.SetParams(params);

            auto st = std::chrono::steady_clock::now();
            for (int i = 0; i < vertices.size(); i++) {
                const Eigen::Vector3d proj_point3D = R2 * mesh.vertices_[i] + t2;
                if (proj_point3D.z() > 0.0) {
                    Eigen::Vector2d point2D = camera2.WorldToImage(proj_point3D.hnormalized());
                    vertices[i] = Eigen::Vector3f(point2D.x(), point2D.y(), proj_point3D.z());
                }
            }
            for (int i = 0; i < bvertices.size(); i++) {
                const Eigen::Vector3d proj_point3D = R2 * bmesh.vertices_[i] + t2;
                if (proj_point3D.z() > 0.0) {
                    Eigen::Vector2d point2D = camera2.WorldToImage(proj_point3D.hnormalized());
                    bvertices[i] = Eigen::Vector3f(point2D.x(), point2D.y(), proj_point3D.z());
                }
            }
            auto ed1 = std::chrono::steady_clock::now();

            cv::Mat depth(camera.Height(), camera.Width(), CV_32FC1);
            cv::Mat bdepth(camera.Height(), camera.Width(), CV_32FC1);
            WarpFrameBuffer(depth, vertices, mesh.faces_);
            WarpFrameBuffer(bdepth, bvertices, bmesh.faces_);
            auto ed2 = std::chrono::steady_clock::now();

            cv::Mat mask = cv::Mat::zeros(camera.Height(), camera.Width(), CV_8UC1);
            for (int y = 0; y < depth.rows; y++) {
                for (int x = 0; x < depth.cols; x++) {
                    float d = depth.at<float>(y, x);
                    float b = bdepth.at<float>(y, x);
                    if (d > std::numeric_limits<float>::epsilon() && b > std::numeric_limits<float>::epsilon() && 
                        std::min(std::abs(d / b - 1.0f), std::abs(b / d - 1.0f)) > relative_thresh &&
                        std::abs(d - b) > absolute_thresh
                    ) {
                        mask.at<uchar>(y, x) = 255;
                    }
                }
            }
            masks.emplace_back(mask);
        }
    } else {
        // The undistort_camera is used to test whether a 3D point is visible to an image,
        // because a distorted camera's WorldToImage may behave weird on a far-away point. 
        Camera undistort_camera = camera;
        if (camera.ModelName() != "PINHOLE" && camera.ModelName() != "SIMPLE_PINHOLE") {
            UndistortOptions options;
            Undistorter::UndistortCamera(options, camera, &undistort_camera);
        }

        auto st = std::chrono::steady_clock::now();
        for (int i = 0; i < vertices.size(); i++) {
            const Eigen::Vector3d proj_point3D = R * mesh.vertices_[i] + t;
            if (proj_point3D.z() > 0.0) {
                Eigen::Vector2d undistort_point2D = undistort_camera.WorldToImage(proj_point3D.hnormalized());
                if (undistort_point2D.x() >= 0 && undistort_point2D.y() >= 0 &&
                    undistort_point2D.x() < undistort_camera.Width() &&
                    undistort_point2D.y() < undistort_camera.Height()
                ) {
                    Eigen::Vector2d point2D = camera.WorldToImage(proj_point3D.hnormalized());
                    vertices[i] = Eigen::Vector3f(point2D.x(), point2D.y(), proj_point3D.z());
                }
            }
        }
        for (int i = 0; i < bvertices.size(); i++) {
            const Eigen::Vector3d proj_point3D = R * bmesh.vertices_[i] + t;
            if (proj_point3D.z() > 0.0) {
                Eigen::Vector2d undistort_point2D = undistort_camera.WorldToImage(proj_point3D.hnormalized());
                if (undistort_point2D.x() >= 0 && undistort_point2D.y() >= 0 &&
                    undistort_point2D.x() < undistort_camera.Width() &&
                    undistort_point2D.y() < undistort_camera.Height()
                ) {
                    Eigen::Vector2d point2D = camera.WorldToImage(proj_point3D.hnormalized());
                    bvertices[i] = Eigen::Vector3f(point2D.x(), point2D.y(), proj_point3D.z());
                }
            }
        }
        auto ed1 = std::chrono::steady_clock::now();

        cv::Mat depth(camera.Height(), camera.Width(), CV_32FC1);
        cv::Mat bdepth(camera.Height(), camera.Width(), CV_32FC1);
        WarpFrameBuffer(depth, vertices, mesh.faces_);
        WarpFrameBuffer(bdepth, bvertices, bmesh.faces_);
        auto ed2 = std::chrono::steady_clock::now();

        cv::Mat mask = cv::Mat::zeros(camera.Height(), camera.Width(), CV_8UC1);
        for (int y = 0; y < depth.rows; y++) {
            for (int x = 0; x < depth.cols; x++) {
                float d = depth.at<float>(y, x);
                float b = bdepth.at<float>(y, x);
                if (d > std::numeric_limits<float>::epsilon() && b > std::numeric_limits<float>::epsilon() && 
                    std::min(std::abs(d / b - 1.0f), std::abs(b / d - 1.0f)) > relative_thresh &&
                    std::abs(d - b) > absolute_thresh
                ) {
                    mask.at<uchar>(y, x) = 255;
                }
            }
        }
        masks.emplace_back(mask);
    }

    return masks;
}

void DepthNoiseFilter(
  cv::Mat & mask,
  mvs::DepthMap & depth,
  float depth_error_thresh,
  int statistic_count_thresh
) {
  mask.create(depth.GetHeight(), depth.GetWidth(), CV_32SC1);
  mask.setTo(0);

  for (int y = 0; y < depth.GetHeight(); y++) {
    for (int x = 0; x < depth.GetWidth(); x++) {
      float d = depth.Get(y, x);
      if (d <= 0) continue;

      double bad_count = 0;
      for (int dy = -2; dy <= 2; dy += 1) {
        const int y2 = y + dy;
        if (y2 < 0 || y2 >= depth.GetHeight()) continue;

        for (int dx = -2; dx <= 2; dx += 1) {
          const int x2 = x + dx;
          if (x2 < 0 || x2 >= depth.GetWidth()) continue;

          float d2 = depth.Get(y2, x2);
          if (d2 <= 0) {
            // bad_count += 0.5f;
          }
          else if (std::abs(d2 - d) > depth_error_thresh) {
            bad_count += 1.0f;
          }
        }
      }

      if (bad_count >= statistic_count_thresh) {
        mask.at<int>(y, x) = 1;
      }
    }
  }
  for (int y = 0; y < depth.GetHeight(); y++) {
    for (int x = 0; x < depth.GetWidth(); x++) {
      float d = depth.Get(y, x);
      if (d <= 0) continue;

      double error_sum = 0;
      int error_weight = 0;
      for (int dy = -2; dy <= 2; dy += 1) {
        const int y2 = y + dy;
        if (y2 < 0 || y2 >= depth.GetHeight()) continue;

        for (int dx = -2; dx <= 2; dx += 1) {
          const int x2 = x + dx;
          if (x2 < 0 || x2 >= depth.GetWidth()) continue;

          float d2 = depth.Get(y2, x2);
          if (d2 > 0) {
            error_sum += (d2 - d) * (d2 - d);
            error_weight += 1;
          }
        }
      }

      if (error_sum / error_weight >= 2.0 * depth_error_thresh * depth_error_thresh) {
        mask.at<int>(y, x) = 1;
      }
    }
  }
  for (int y = 0; y < depth.GetHeight(); y++) {
    for (int x = 0; x < depth.GetWidth(); x++) {
      if (mask.at<int>(y, x)) {
        depth.Set(y, x, 0.0f);
      }
    }
  }
}

int main(int argc, char *argv[])
{
	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	std::string configuration_file_path_1;
	std::string configuration_file_path_2;
    std::string sfm_export;
    std::string mask_export;
    std::vector<std::string> export_subpaths;
    std::string debug_images;
    if (argc == 5) {
        configuration_file_path_1 = argv[1];
        configuration_file_path_2 = argv[2];
        sfm_export = argv[3];
        mask_export = argv[4];
    } else if (argc == 6) {
        configuration_file_path_1 = argv[1];
        configuration_file_path_2 = argv[2];
        sfm_export = argv[3];
        mask_export = argv[4];
        export_subpaths = CSVToVector<std::string>(argv[5]);
    } else if (argc == 7) {
        configuration_file_path_1 = argv[1];
        configuration_file_path_2 = argv[2];
        sfm_export = argv[3];
        mask_export = argv[4];
        export_subpaths = CSVToVector<std::string>(argv[5]);
        debug_images = argv[6];
    } else {
        printf("Usage: %s\n", argv[0]);
        printf("       <OLD_MVS_YAML> <NEW_MVS_YAML> <SFM_EXPORT> <MASK_EXPORT> [SELECTED_SUBPATHS] [DEBUG_IMAGES]\n");
        return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    Configurator param1;
    Configurator param2;
    param1.Load(configuration_file_path_1.c_str());
    param2.Load(configuration_file_path_2.c_str());

	std::string workspace_path1 = param1.GetArgument("workspace_path", "");
	std::string workspace_path2 = param2.GetArgument("workspace_path", "");
	std::string image_path1 = param1.GetArgument("image_path", "");
	std::string image_path2 = param2.GetArgument("image_path", "");
	std::string image_type1 = param1.GetArgument("image_type", "perspective");
	std::string image_type2 = param2.GetArgument("image_type", "perspective");
	bool geom_consistency1 = param1.GetArgument("geom_consistency", 1);
	bool geom_consistency2 = param2.GetArgument("geom_consistency", 1);
	std::string input_type1 = geom_consistency1 ? GEOMETRIC_TYPE : PHOTOMETRIC_TYPE;
	std::string input_type2 = geom_consistency1 ? GEOMETRIC_TYPE : PHOTOMETRIC_TYPE;
    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    std::shared_ptr<TriangleMesh> base_mesh = std::make_shared<TriangleMesh>();
    
    {
        auto reconstruction_path = 
            JoinPaths(workspace_path1, "0");
        auto dense_reconstruction_path = 
            JoinPaths(reconstruction_path, DENSE_DIR);

        auto undistort_image_path =
            JoinPaths(dense_reconstruction_path, IMAGES_DIR);
        auto stereo_reconstruction_path = 
            JoinPaths(dense_reconstruction_path, STEREO_DIR);
        auto depth_maps_path = 
            JoinPaths(stereo_reconstruction_path, DEPTHS_DIR);
        auto prior_depth_maps_path = 
            JoinPaths(dense_reconstruction_path, DEPTHS_DIR);
        auto mesh_path = 
            JoinPaths(dense_reconstruction_path, MODEL_NAME);
        auto sparse_path = 
            JoinPaths(dense_reconstruction_path, SPARSE_DIR);

        ReadTriangleMeshObj(mesh_path, *base_mesh);
    }

    float scale_factor = -1.0f;
    float relative_depth_diff = 0.1f;
    float absolute_depth_diff = 0.0f;
    {
        auto reconstruction_path = 
            JoinPaths(workspace_path2, "0");
        auto dense_reconstruction_path = 
            JoinPaths(reconstruction_path, DENSE_DIR);

        auto undistort_image_path =
            JoinPaths(dense_reconstruction_path, IMAGES_DIR);
        auto stereo_reconstruction_path = 
            JoinPaths(dense_reconstruction_path, STEREO_DIR);
        auto depth_maps_path = 
            JoinPaths(stereo_reconstruction_path, DEPTHS_DIR);
        auto prior_depth_maps_path = 
            JoinPaths(dense_reconstruction_path, DEPTHS_DIR);
        auto mesh_path = 
            JoinPaths(dense_reconstruction_path, MODEL_NAME);
        auto scale_path = 
            JoinPaths(dense_reconstruction_path, "scale.txt");

        ReadTriangleMeshObj(mesh_path, *mesh);
        if (ExistsFile(scale_path)) {
            std::ifstream(scale_path) >> scale_factor;
        }
        if (scale_factor > 0.0f) {
            absolute_depth_diff = 0.2f / scale_factor;
            std::cout << "Scale: " << scale_factor << std::endl;
            std::cout << "Relative thresh: " << relative_depth_diff << std::endl;
            std::cout << "Absolute thresh: " << absolute_depth_diff << std::endl;
        }
    }

    Reconstruction export_reconstruction;
    export_reconstruction.ReadBinary(sfm_export, false);

    if (export_subpaths.size() != 0) {
        std::cout << "Exporting subpaths:" << std::endl;
        for (auto subpath : export_subpaths) {
            std::cout << subpath << std::endl;
        }
    }

    std::vector<image_t> export_images;
    for (auto image : export_reconstruction.Images()) {
        if (export_subpaths.size() == 0) {
            export_images.emplace_back(image.first);
        } else {
            for (auto & subpath : export_subpaths) {
                if (IsInsideSubpath(image.second.Name(), subpath)) {
                    export_images.emplace_back(image.first);
                    break;
                }
            }
        }
    }

    if (ExistsDir(mask_export)) {
        std::cout << "Inplace update of existing masks: " << mask_export << std::endl;
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < export_images.size(); i++) {
        const auto image_id = export_images[i];
        auto & image = export_reconstruction.Image(image_id);
        auto & camera = export_reconstruction.Camera(image.CameraId());
        std::vector<cv::Mat> masks = ProjectMeshMakeMask(image, camera, *mesh, *base_mesh, relative_depth_diff, absolute_depth_diff);

        for (int cam_id = 0; cam_id < masks.size(); cam_id++) {
            if (cv::countNonZero(masks[cam_id]) > camera.Width() * camera.Height() * 0.005) {
                std::string output_path = JoinPaths(mask_export, image.Name() + "." + MASK_EXT);
                if (cam_id > 0) {
                    auto find = output_path.find("cam0");
                    output_path.replace(find, find + strlen("cam0"), "cam" + std::to_string(cam_id));
                }

                double image_thresh = std::min(camera.Width(), camera.Height()) * 0.01;
                const int element_size = std::min(10, std::max(1, (int)image_thresh));
                cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * element_size + 1, 2 * element_size + 1));

                cv::Mat save_mask(masks[cam_id].rows, masks[cam_id].cols, CV_8UC1);
                save_mask.setTo(0);

                // component filter
                for (int y = 0; y < save_mask.rows; y++) {
                    for (int x = 0; x < save_mask.cols; x++) {
                        size_t count = 0;
                        size_t thresh = (2 * element_size + 1) * (2 * element_size + 1) * 0.2;
                        for (int yy = y - element_size; yy <= y + element_size; yy++) {
                            int y2 = yy;
                            if (y2 < 0) y2 = -y2;
                            if (y2 >= save_mask.rows) y2 = 2 * save_mask.rows - 1 - y2;
                            for (int xx = x - element_size; xx <= x + element_size; xx++) {
                                int x2 = xx;
                                if (x2 < 0) x2 = -x2;
                                if (x2 >= save_mask.cols) x2 = 2 * save_mask.cols - 1 - x2;
                                if (masks[cam_id].at<uchar>(y2, x2)) {
                                    count++;
                                }
                                if (count > thresh) break;
                            }
                            if (count > thresh) break;
                        }
                        if (count > thresh) {
                            save_mask.at<uchar>(y, x) = 255;
                        }
                    }
                }

                // noise filtering
                cv::erode(save_mask, save_mask, element);
                cv::erode(save_mask, save_mask, element);
                cv::dilate(save_mask, save_mask, element);
                cv::dilate(save_mask, save_mask, element);

                // hole filling
                cv::dilate(save_mask, save_mask, element);
                cv::dilate(save_mask, save_mask, element);
                cv::erode(save_mask, save_mask, element);

                if (cv::countNonZero(save_mask) > 0) {
                    if (ExistsFile(output_path)) {
                        cv::Mat prev_mask = cv::imread(output_path, cv::IMREAD_GRAYSCALE);
                        CHECK(prev_mask.size == save_mask.size);
                        save_mask |= prev_mask;
                    }

                    if (!ExistsPath(GetParentDir(output_path))) {
                        boost::filesystem::create_directories(GetParentDir(output_path));
                    }
                    cv::imwrite(output_path, save_mask);

                    if (!debug_images.empty()) {
                        cv::Mat color;
                        std::string name = JoinPaths(debug_images, image.Name());
                        if (IsFileRGBD(name)) {
                            Bitmap bitmap;
                            ExtractRGBDData(name, bitmap);
                            FreeImage2Mat(&bitmap, color);
                        } else {
                            color = cv::imread(name);
                        }
                        if (color.empty()) continue;
                        for (int y = 0; y < save_mask.rows; y++) {
                            for (int x = 0; x < save_mask.cols; x++) {
                                if (save_mask.at<uchar>(y, x)) {
                                    color.at<cv::Vec3b>(y, x)[2] = std::min(255, color.at<cv::Vec3b>(y, x)[2] + 64);
                                }
                            }
                        }

                        std::string debug_output_path = debug_images;
                        while (debug_output_path.length() > 0 && debug_output_path[debug_output_path.length() - 1] == '/') {
                            debug_output_path = debug_output_path.substr(0, debug_output_path.length() - 1);
                        }
                        debug_output_path += "-masked";
                        std::string debug_output_name = JoinPaths(debug_output_path, image.Name());
                        if (!ExistsDir(GetParentDir(debug_output_name))) {
                            boost::filesystem::create_directories(boost::filesystem::path(GetParentDir(debug_output_name)));
                        }
                        cv::imwrite(debug_output_name, color);
                    }
                }
            }
        }
    }

    return 0;
}