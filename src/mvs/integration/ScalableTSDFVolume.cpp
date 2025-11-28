#include "ScalableTSDFVolume.h"
#include "UniformTSDFVolume.h"
#include "MarchingCubesConst.h"
#include <unordered_set>
#include <iostream>
#include <fstream>

namespace sensemap {

namespace tsdf {

std::shared_ptr<TriangleMesh> ScalableTSDFVolumeBase::ExtractTriangleMesh(float weight_thresh)
{
    // implementation of marching cubes, based on
    // http://paulbourke.net/geometry/polygonise/
    auto mesh = std::make_shared<TriangleMesh>();
    float half_voxel_length = voxel_length_ * 0.5;
    std::unordered_map<Eigen::Vector4i, int, hash<Eigen::Vector4i>>
        edgeindex_to_vertexindex;
    int edge_to_index[12];

    ForeachVolumeUnit([&](Eigen::Vector3i index0, std::shared_ptr<TSDFVolume<>> volume) {
        const auto &volume0 = *volume;
        for (int x = 0; x < volume0.resolution_; x++) {
            for (int y = 0; y < volume0.resolution_; y++) {
                for (int z = 0; z < volume0.resolution_; z++) {
                    Eigen::Vector3i idx0(x, y, z);
                    int cube_index = 0;
                    float f[8];
                    float w[8];
                    for (int i = 0; i < 8; i++) {
                        Eigen::Vector3i index1 = index0;
                        Eigen::Vector3i idx1 = idx0 + shift[i];
                        if (idx1(0) < volume_unit_resolution_ &&
                            idx1(1) < volume_unit_resolution_ &&
                            idx1(2) < volume_unit_resolution_) {
                            f[i] = volume0.tsdf_[volume0.IndexOf(idx1)];
                            w[i] = volume0.weight_[volume0.IndexOf(idx1)];
                        } else {
                            for (int j = 0; j < 3; j++) {
                                if (idx1(j) >= volume_unit_resolution_) {
                                    idx1(j) -= volume_unit_resolution_;
                                    index1(j) += 1;
                                }
                            }
                            auto unit_itr1 = FindVolumeUnit(index1);
                            if (unit_itr1 == nullptr) {
                                f[i] = TSDF_DEFAULT_TSDF;
                                w[i] = TSDF_DEFAULT_WEIGHT;
                            } else {
                                const auto &volume1 = *unit_itr1;
                                f[i] = volume1.tsdf_[volume1.IndexOf(idx1)];
                                w[i] = volume1.weight_[volume1.IndexOf(idx1)];
                            }
                        }
                        if (w[i] <= weight_thresh) {
                            cube_index = 0;
                            break;
                        } else {
                            if (f[i] < 0) {
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
                                    volume_unit_resolution_ +
                                    Eigen::Vector4i(x, y, z, 0) +
                                    edge_shift[i];
                            if (edgeindex_to_vertexindex.find(edge_index) ==
                                    edgeindex_to_vertexindex.end()) {
                                edge_to_index[i] =
                                        (int)mesh->vertices_.size();
                                edgeindex_to_vertexindex[edge_index] =
                                        (int)mesh->vertices_.size();
                                Eigen::Vector3d pt(
                                        half_voxel_length + voxel_length_ * edge_index(0),
                                        half_voxel_length + voxel_length_ * edge_index(1),
                                        half_voxel_length + voxel_length_ * edge_index(2));
                                float f0 = std::fabs(f[edge_to_vert[i][0]]);
                                float f1 = std::fabs(f[edge_to_vert[i][1]]);
                                pt(edge_index(3)) += f0 * voxel_length_ / (f0 + f1);
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
    });
    return mesh;
}

std::shared_ptr<TriangleMesh> ScalableTSDFVolumeBase::ExtractColoredTriangleMesh(float weight_thresh)
{
    // implementation of marching cubes, based on
    // http://paulbourke.net/geometry/polygonise/
    auto mesh = std::make_shared<TriangleMesh>();
    float half_voxel_length = voxel_length_ * 0.5;
    std::unordered_map<Eigen::Vector4i, int, hash<Eigen::Vector4i>>
        edgeindex_to_vertexindex;
    int edge_to_index[12];
    ForeachColoredVolumeUnit([&](Eigen::Vector3i index0, std::shared_ptr<ColoredTSDFVolume<>> volume) {
        const auto &volume0 = *volume;
        for (int x = 0; x < volume0.resolution_; x++) {
            for (int y = 0; y < volume0.resolution_; y++) {
                for (int z = 0; z < volume0.resolution_; z++) {
                    Eigen::Vector3i idx0(x, y, z);
                    int cube_index = 0;
                    float f[8];
                    float w[8];
                    Eigen::Vector3f c[8];
                    for (int i = 0; i < 8; i++) {
                        Eigen::Vector3i index1 = index0;
                        Eigen::Vector3i idx1 = idx0 + shift[i];
                        if (idx1(0) < volume_unit_resolution_ &&
                            idx1(1) < volume_unit_resolution_ &&
                            idx1(2) < volume_unit_resolution_) {
                            f[i] = volume0.tsdf_[volume0.IndexOf(idx1)];
                            c[i] = volume0.color_[volume0.IndexOf(idx1)];
                            w[i] = volume0.weight_[volume0.IndexOf(idx1)];
                        } else {
                            for (int j = 0; j < 3; j++) {
                                if (idx1(j) >= volume_unit_resolution_) {
                                    idx1(j) -= volume_unit_resolution_;
                                    index1(j) += 1;
                                }
                            }
                            auto unit_itr1 = FindColoredVolumeUnit(index1);
                            if (unit_itr1 == nullptr) {
                                f[i] = TSDF_DEFAULT_TSDF;
                                c[i] = TSDF_DEFAULT_COLOR;
                                w[i] = TSDF_DEFAULT_WEIGHT;
                            } else {
                                const auto &volume1 = *unit_itr1;
                                f[i] = volume1.tsdf_[volume1.IndexOf(idx1)];
                                c[i] = volume1.color_[volume1.IndexOf(idx1)];
                                w[i] = volume1.weight_[volume1.IndexOf(idx1)];
                            }
                        }
                        if (w[i] <= weight_thresh) {
                            cube_index = 0;
                            break;
                        } else {
                            if (f[i] < 0) {
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
                                    volume_unit_resolution_ +
                                    Eigen::Vector4i(x, y, z, 0) +
                                    edge_shift[i];
                            if (edgeindex_to_vertexindex.find(edge_index) ==
                                    edgeindex_to_vertexindex.end()) {
                                edge_to_index[i] =
                                        (int)mesh->vertices_.size();
                                edgeindex_to_vertexindex[edge_index] =
                                        (int)mesh->vertices_.size();
                                Eigen::Vector3d pt(
                                        half_voxel_length + voxel_length_ * edge_index(0),
                                        half_voxel_length + voxel_length_ * edge_index(1),
                                        half_voxel_length + voxel_length_ * edge_index(2));
                                float f0 = std::fabs(f[edge_to_vert[i][0]]);
                                float f1 = std::fabs(f[edge_to_vert[i][1]]);
                                pt(edge_index(3)) += f0 * voxel_length_ / (f0 + f1);
                                mesh->vertices_.push_back(pt);
                                Eigen::Vector3f c0 = c[edge_to_vert[i][0]];
                                Eigen::Vector3f c1 = c[edge_to_vert[i][1]];
                                mesh->vertex_colors_.push_back(((f1 * c0.cast<double>() + f0 * c1.cast<double>()) / (f0 + f1)) * (1.0f / 255.0f));

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
    });
    return mesh;
}

std::shared_ptr<TriangleMesh> ScalableTSDFVolumeBase::ExtractTSDF(float weight_thresh)
{
    auto mesh = std::make_shared<TriangleMesh>();
    ForeachVolumeUnit([&](Eigen::Vector3i index0, std::shared_ptr<TSDFVolume<>> volume) {
        const auto &volume0 = *volume;
        Eigen::Vector3d origin = index0.cast<double>() * volume_unit_length_;
        for (int x = 0; x < volume0.resolution_; x++) {
            for (int y = 0; y < volume0.resolution_; y++) {
                for (int z = 0; z < volume0.resolution_; z++) {
                    Eigen::Vector3i idx0(x, y, z);
                    float f = volume0.tsdf_[volume0.IndexOf(idx0)];
                    float w = volume0.weight_[volume0.IndexOf(idx0)];

                    if (w > weight_thresh) {
                        Eigen::Vector3d pt = origin + idx0.cast<double>() * voxel_length_;
                        mesh->vertices_.push_back(pt);

                        double r = std::max(0.0, -f * 127.0);
                        double g = std::max(0.0, f * 127.0);
                        double b = (1.0 - std::abs(f)) * 127.0;
                        mesh->vertex_colors_.emplace_back(r, g, b);
                    }
                }
            }
        }
    });
    return mesh;
}

std::vector<Eigen::Vector3f> ScalableTSDFVolumeBase::GetWorldPoints(const cv::Mat &depth, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic)
{
    std::vector<Eigen::Vector3f> pcd_w;
    const Eigen::Matrix3f &RT = extrinsic.block<3, 3>(0, 0).transpose();
    const Eigen::Vector3f &T = extrinsic.block<3, 1>(0, 3);
    const Eigen::Matrix3f &inv_K = intrinsic.inverse();
    for (int y = 0; y < depth.rows; y += depth_sampling_stride_)
    {
        for (int x = 0; x < depth.cols; x += depth_sampling_stride_)
        {
            const float z_f = depth.at<float>(y, x);
            if (z_f > min_depth_ && z_f < max_depth_)
            {
                Eigen::Vector3f pt = RT * (inv_K * Eigen::Vector3f(x * z_f, y * z_f, z_f) - T);
                pcd_w.emplace_back(pt);
            }
        }
    }

    return pcd_w;
}

void ScalableTSDFVolumeBase::Integrate(const cv::Mat &depth, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic, float base_weight)
{
    /// create point cloud map
    std::vector<Eigen::Vector3f> pcd_w = GetWorldPoints(depth, intrinsic, extrinsic);

    /// integrate
    std::unordered_set<Eigen::Vector3i, hash<Eigen::Vector3i>> touched_volume_units;

    for (const auto &point : pcd_w) {
        auto min_bound = LocateVolumeUnit(point - Eigen::Vector3f(
                sdf_trunc_, sdf_trunc_, sdf_trunc_));
        auto max_bound = LocateVolumeUnit(point + Eigen::Vector3f(
                sdf_trunc_, sdf_trunc_, sdf_trunc_));
        for (auto x = min_bound(0); x <= max_bound(0); x++) {
            for (auto y = min_bound(1); y <= max_bound(1); y++) {
                for (auto z = min_bound(2); z <= max_bound(2); z++) {
                    auto loc = Eigen::Vector3i(x, y, z);
                    if (touched_volume_units.find(loc) ==
                            touched_volume_units.end()) {
                        touched_volume_units.insert(loc);

                        std::shared_ptr<TSDFVolume<>> volume;
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            volume = OpenVolumeUnit(Eigen::Vector3i(x, y, z));
                            cv_.wait(lock, [&] {
                                return volume->flag == 0;
                            });
                            volume->flag = 1;
                        }
                        
                        volume->Integrate(depth, intrinsic, extrinsic, true, false, base_weight);

                        {
                            mutex_.lock();
                            volume->flag = 0;
                            mutex_.unlock();
                            cv_.notify_all();
                        }
                    }
                }
            }
        }
    }
}

void ScalableTSDFVolumeBase::Integrate(const cv::Mat &depth, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic, bool de_integrate)
{
    /// create point cloud map
    std::vector<Eigen::Vector3f> pcd_w = GetWorldPoints(depth, intrinsic, extrinsic);

    /// integrate
    std::unordered_set<Eigen::Vector3i, hash<Eigen::Vector3i>> touched_volume_units;

    for (const auto &point : pcd_w) {
        auto min_bound = LocateVolumeUnit(point - Eigen::Vector3f(
                sdf_trunc_, sdf_trunc_, sdf_trunc_));
        auto max_bound = LocateVolumeUnit(point + Eigen::Vector3f(
                sdf_trunc_, sdf_trunc_, sdf_trunc_));
        for (auto x = min_bound(0); x <= max_bound(0); x++) {
            for (auto y = min_bound(1); y <= max_bound(1); y++) {
                for (auto z = min_bound(2); z <= max_bound(2); z++) {
                    auto loc = Eigen::Vector3i(x, y, z);
                    if (touched_volume_units.find(loc) ==
                            touched_volume_units.end()) {
                        touched_volume_units.insert(loc);

                        std::shared_ptr<TSDFVolume<>> volume;
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            volume = OpenVolumeUnit(Eigen::Vector3i(x, y, z));
                            cv_.wait(lock, [&] {
                                return volume->flag == 0;
                            });
                            volume->flag = 1;
                        }
                        
                        volume->Integrate(depth, intrinsic, extrinsic, false, de_integrate);

                        {
                            mutex_.lock();
                            volume->flag = 0;
                            mutex_.unlock();
                            cv_.notify_all();
                        }
                    }
                }
            }
        }
    }
}

void ScalableTSDFVolumeBase::Integrate(const cv::Mat &depth, const cv::Mat &color, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic, float base_weight)
{
    if (color.empty()) return Integrate(depth, intrinsic, extrinsic, base_weight);

    /// create point cloud map
    std::vector<Eigen::Vector3f> pcd_w = GetWorldPoints(depth, intrinsic, extrinsic);

    /// integrate
    std::unordered_set<Eigen::Vector3i, hash<Eigen::Vector3i>> touched_volume_units;

    for (const auto &point : pcd_w) {
        auto min_bound = LocateVolumeUnit(point - Eigen::Vector3f(
                sdf_trunc_, sdf_trunc_, sdf_trunc_));
        auto max_bound = LocateVolumeUnit(point + Eigen::Vector3f(
                sdf_trunc_, sdf_trunc_, sdf_trunc_));
        for (auto x = min_bound(0); x <= max_bound(0); x++) {
            for (auto y = min_bound(1); y <= max_bound(1); y++) {
                for (auto z = min_bound(2); z <= max_bound(2); z++) {
                    auto loc = Eigen::Vector3i(x, y, z);
                    if (touched_volume_units.find(loc) ==
                            touched_volume_units.end()) {
                        touched_volume_units.insert(loc);

                        std::shared_ptr<ColoredTSDFVolume<>> volume;
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            volume = OpenColoredVolumeUnit(Eigen::Vector3i(x, y, z));
                            cv_.wait(lock, [&] {
                                return volume->flag == 0;
                            });
                            volume->flag = 1;
                        }
                        
                        volume->Integrate(depth, color, intrinsic, extrinsic, true, false, base_weight);

                        {
                            mutex_.lock();
                            volume->flag = 0;
                            mutex_.unlock();
                            cv_.notify_all();
                        }
                    }
                }
            }
        }
    }
}

void ScalableTSDFVolumeBase::Integrate(const cv::Mat &depth, const cv::Mat &color, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic, bool de_integrate)
{
    if (color.empty()) return Integrate(depth, intrinsic, extrinsic, de_integrate);

    /// create point cloud map
    std::vector<Eigen::Vector3f> pcd_w = GetWorldPoints(depth, intrinsic, extrinsic);

    /// integrate
    std::unordered_set<Eigen::Vector3i, hash<Eigen::Vector3i>> touched_volume_units;

    for (const auto &point : pcd_w) {
        auto min_bound = LocateVolumeUnit(point - Eigen::Vector3f(
                sdf_trunc_, sdf_trunc_, sdf_trunc_));
        auto max_bound = LocateVolumeUnit(point + Eigen::Vector3f(
                sdf_trunc_, sdf_trunc_, sdf_trunc_));
        for (auto x = min_bound(0); x <= max_bound(0); x++) {
            for (auto y = min_bound(1); y <= max_bound(1); y++) {
                for (auto z = min_bound(2); z <= max_bound(2); z++) {
                    auto loc = Eigen::Vector3i(x, y, z);
                    if (touched_volume_units.find(loc) ==
                            touched_volume_units.end()) {
                        touched_volume_units.insert(loc);

                        std::shared_ptr<ColoredTSDFVolume<>> volume;
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            volume = OpenColoredVolumeUnit(Eigen::Vector3i(x, y, z));
                            cv_.wait(lock, [&] {
                                return volume->flag == 0;
                            });
                            volume->flag = 1;
                        }
                        
                        if (de_integrate) {
                            volume->Integrate(depth, color, intrinsic, extrinsic, false, true);
                        }
                        else {
                            volume->Integrate(depth, color, intrinsic, extrinsic, false, true);
                        }

                        {
                            mutex_.lock();
                            volume->flag = 0;
                            mutex_.unlock();
                            cv_.notify_all();
                        }
                    }
                }
            }
        }
    }
}

float ScalableTSDFVolumeBase::GetTSDFAt(const Eigen::Vector3f & p)
{
    Eigen::Vector3f p_locate = p - Eigen::Vector3f(0.5, 0.5, 0.5) * voxel_length_;
    Eigen::Vector3i index0 = LocateVolumeUnit(p_locate);
    auto unit_itr = FindVolumeUnit(index0);
    if (unit_itr == nullptr) {
        return TSDF_DEFAULT_TSDF;
    }
    const auto &volume0 = *unit_itr;
    Eigen::Vector3i idx0;
    Eigen::Vector3f p_grid = (p_locate - index0.cast<float>() * volume_unit_length_) / voxel_length_;
    for (int i = 0; i < 3; i++) {
        idx0(i) = (int)std::floor(p_grid(i));
        if (idx0(i) < 0) idx0(i) = 0;
        if (idx0(i) >= volume_unit_resolution_)
            idx0(i) = volume_unit_resolution_ - 1;
    }
    Eigen::Vector3f r = p_grid - idx0.cast<float>();
    float f[8];
    for (int i = 0; i < 8; i++) {
        Eigen::Vector3i index1 = index0;
        Eigen::Vector3i idx1 = idx0 + shift[i];
        if (idx1(0) < volume_unit_resolution_ &&
            idx1(1) < volume_unit_resolution_ &&
            idx1(2) < volume_unit_resolution_) {
            f[i] = volume0.tsdf_[volume0.IndexOf(idx1)];
        } else {
            for (int j = 0; j < 3; j++) {
                if (idx1(j) >= volume_unit_resolution_) {
                    idx1(j) -= volume_unit_resolution_;
                    index1(j) += 1;
                }
            }
            auto unit_itr1 = FindVolumeUnit(index1);
            if (unit_itr1 == nullptr) {
                f[i] = TSDF_DEFAULT_TSDF;
            } else {
                const auto &volume1 = *unit_itr1;
                f[i] = volume1.tsdf_[volume1.IndexOf(idx1)];
            }
        }
    }
    return (1 - r(0)) * ( (1 - r(1)) * ((1 - r(2)) * f[0] + r(2) * f[4]) +
            r(1) * ((1 - r(2)) * f[3] + r(2) * f[7])) +
            r(0) * ((1 - r(1)) * ((1 - r(2)) * f[1] + r(2) * f[5]) +
            r(1) * ((1 - r(2)) * f[2] + r(2) * f[6]));
}

std::tuple<float, float> ScalableTSDFVolumeBase::GetWeightedTSDFAt(const Eigen::Vector3f &p)
{
    Eigen::Vector3f p_locate = p - Eigen::Vector3f(0.5, 0.5, 0.5) * voxel_length_;
    Eigen::Vector3i index0 = LocateVolumeUnit(p_locate);
    auto unit_itr = FindVolumeUnit(index0);
    if (unit_itr == nullptr) {
        return std::make_tuple(TSDF_DEFAULT_TSDF, TSDF_DEFAULT_WEIGHT);
    }
    const auto &volume0 = *unit_itr;
    Eigen::Vector3i idx0;
    Eigen::Vector3f p_grid = (p_locate - index0.cast<float>() * volume_unit_length_) / voxel_length_;
    for (int i = 0; i < 3; i++) {
        idx0(i) = (int)std::floor(p_grid(i));
        if (idx0(i) < 0) idx0(i) = 0;
        if (idx0(i) >= volume_unit_resolution_)
            idx0(i) = volume_unit_resolution_ - 1;
    }
    Eigen::Vector3f r = p_grid - idx0.cast<float>();
    float f[8], w[8];
    for (int i = 0; i < 8; i++) {
        Eigen::Vector3i index1 = index0;
        Eigen::Vector3i idx1 = idx0 + shift[i];
        if (idx1(0) < volume_unit_resolution_ &&
            idx1(1) < volume_unit_resolution_ &&
            idx1(2) < volume_unit_resolution_) {
            f[i] = volume0.tsdf_[volume0.IndexOf(idx1)];
            w[i] = volume0.weight_[volume0.IndexOf(idx1)];
        } else {
            for (int j = 0; j < 3; j++) {
                if (idx1(j) >= volume_unit_resolution_) {
                    idx1(j) -= volume_unit_resolution_;
                    index1(j) += 1;
                }
            }
            auto unit_itr1 = FindVolumeUnit(index1);
            if (unit_itr1 == nullptr) {
                f[i] = TSDF_DEFAULT_TSDF;
                w[i] = TSDF_DEFAULT_WEIGHT;
            } else {
                const auto &volume1 = *unit_itr1;
                f[i] = volume1.tsdf_[volume1.IndexOf(idx1)];
                w[i] = volume1.weight_[volume1.IndexOf(idx1)];
            }
        }
    }

    float sdf = (1 - r(0)) * ( (1 - r(1)) * ((1 - r(2)) * f[0] + r(2) * f[4]) +
                               r(1) * ((1 - r(2)) * f[3] + r(2) * f[7])) +
                r(0) * ((1 - r(1)) * ((1 - r(2)) * f[1] + r(2) * f[5]) +
                        r(1) * ((1 - r(2)) * f[2] + r(2) * f[6]));

    float weight = (1 - r(0)) * ( (1 - r(1)) * ((1 - r(2)) * w[0] + r(2) * w[4]) +
                               r(1) * ((1 - r(2)) * w[3] + r(2) * w[7])) +
                r(0) * ((1 - r(1)) * ((1 - r(2)) * w[1] + r(2) * w[5]) +
                        r(1) * ((1 - r(2)) * w[2] + r(2) * w[6]));

    return std::make_tuple(sdf, weight);
}

std::tuple<float, Eigen::Vector3f> ScalableTSDFVolumeBase::GetColoredTSDFAt(const Eigen::Vector3f &p)
{
    Eigen::Vector3f p_locate = p - Eigen::Vector3f(0.5, 0.5, 0.5) * voxel_length_;
    Eigen::Vector3i index0 = LocateVolumeUnit(p_locate);
    auto unit_itr = FindColoredVolumeUnit(index0);
    if (unit_itr == nullptr) {
        return std::make_tuple(TSDF_DEFAULT_TSDF, TSDF_DEFAULT_COLOR);
    }
    const auto &volume0 = *unit_itr;
    Eigen::Vector3i idx0;
    Eigen::Vector3f p_grid = (p_locate - index0.cast<float>() * volume_unit_length_) / voxel_length_;
    for (int i = 0; i < 3; i++) {
        idx0(i) = (int)std::floor(p_grid(i));
        if (idx0(i) < 0) idx0(i) = 0;
        if (idx0(i) >= volume_unit_resolution_)
            idx0(i) = volume_unit_resolution_ - 1;
    }
    Eigen::Vector3f r = p_grid - idx0.cast<float>();
    float f[8];
    Eigen::Vector3f c[8];
    for (int i = 0; i < 8; i++) {
        Eigen::Vector3i index1 = index0;
        Eigen::Vector3i idx1 = idx0 + shift[i];
        if (idx1(0) < volume_unit_resolution_ &&
            idx1(1) < volume_unit_resolution_ &&
            idx1(2) < volume_unit_resolution_) {
            f[i] = volume0.tsdf_[volume0.IndexOf(idx1)];
            c[i] = volume0.color_[volume0.IndexOf(idx1)];
        } else {
            for (int j = 0; j < 3; j++) {
                if (idx1(j) >= volume_unit_resolution_) {
                    idx1(j) -= volume_unit_resolution_;
                    index1(j) += 1;
                }
            }
            auto unit_itr1 = FindColoredVolumeUnit(index1);
            if (unit_itr1 == nullptr) {
                f[i] = TSDF_DEFAULT_TSDF;
                c[i] = TSDF_DEFAULT_COLOR;
            } else {
                const auto &volume1 = *unit_itr1;
                f[i] = volume1.tsdf_[volume1.IndexOf(idx1)];
                c[i] = volume1.color_[volume1.IndexOf(idx1)];
            }
        }
    }

    float sdf = (1 - r(0)) * ( (1 - r(1)) * ((1 - r(2)) * f[0] + r(2) * f[4]) +
                               r(1) * ((1 - r(2)) * f[3] + r(2) * f[7])) +
                r(0) * ((1 - r(1)) * ((1 - r(2)) * f[1] + r(2) * f[5]) +
                        r(1) * ((1 - r(2)) * f[2] + r(2) * f[6]));

    Eigen::Vector3f color = (1 - r(0)) * ( (1 - r(1)) * ((1 - r(2)) * c[0] + r(2) * c[4]) +
                                          r(1) * ((1 - r(2)) * c[3] + r(2) * c[7])) +
                           r(0) * ((1 - r(1)) * ((1 - r(2)) * c[1] + r(2) * c[5]) +
                                   r(1) * ((1 - r(2)) * c[2] + r(2) * c[6]));

    return std::make_tuple(sdf, color);
}

std::tuple<float, float, Eigen::Vector3f> ScalableTSDFVolumeBase::GetWeightedColoredTSDFAt(const Eigen::Vector3f &p)
{
    Eigen::Vector3f p_locate = p - Eigen::Vector3f(0.5, 0.5, 0.5) * voxel_length_;
    Eigen::Vector3i index0 = LocateVolumeUnit(p_locate);
    auto unit_itr = FindColoredVolumeUnit(index0);
    if (unit_itr == nullptr) {
        return std::make_tuple(TSDF_DEFAULT_TSDF, TSDF_DEFAULT_WEIGHT, TSDF_DEFAULT_COLOR);
    }
    const auto &volume0 = *unit_itr;
    Eigen::Vector3i idx0;
    Eigen::Vector3f p_grid = (p_locate - index0.cast<float>() * volume_unit_length_) / voxel_length_;
    for (int i = 0; i < 3; i++) {
        idx0(i) = (int)std::floor(p_grid(i));
        if (idx0(i) < 0) idx0(i) = 0;
        if (idx0(i) >= volume_unit_resolution_)
            idx0(i) = volume_unit_resolution_ - 1;
    }
    Eigen::Vector3f r = p_grid - idx0.cast<float>();
    float f[8], w[8];
    Eigen::Vector3f c[8];
    for (int i = 0; i < 8; i++) {
        Eigen::Vector3i index1 = index0;
        Eigen::Vector3i idx1 = idx0 + shift[i];
        if (idx1(0) < volume_unit_resolution_ &&
            idx1(1) < volume_unit_resolution_ &&
            idx1(2) < volume_unit_resolution_) {
            f[i] = volume0.tsdf_[volume0.IndexOf(idx1)];
            c[i] = volume0.color_[volume0.IndexOf(idx1)];
            w[i] = volume0.weight_[volume0.IndexOf(idx1)];
        } else {
            for (int j = 0; j < 3; j++) {
                if (idx1(j) >= volume_unit_resolution_) {
                    idx1(j) -= volume_unit_resolution_;
                    index1(j) += 1;
                }
            }
            auto unit_itr1 = FindColoredVolumeUnit(index1);
            if (unit_itr1 == nullptr) {
                f[i] = TSDF_DEFAULT_TSDF;
                c[i] = TSDF_DEFAULT_COLOR;
                w[i] = TSDF_DEFAULT_WEIGHT;
            } else {
                const auto &volume1 = *unit_itr1;
                f[i] = volume1.tsdf_[volume1.IndexOf(idx1)];
                c[i] = volume1.color_[volume1.IndexOf(idx1)];
                w[i] = volume1.weight_[volume1.IndexOf(idx1)];
            }
        }
    }

    float sdf = (1 - r(0)) * ( (1 - r(1)) * ((1 - r(2)) * f[0] + r(2) * f[4]) +
                               r(1) * ((1 - r(2)) * f[3] + r(2) * f[7])) +
                r(0) * ((1 - r(1)) * ((1 - r(2)) * f[1] + r(2) * f[5]) +
                        r(1) * ((1 - r(2)) * f[2] + r(2) * f[6]));

    Eigen::Vector3f color = (1 - r(0)) * ( (1 - r(1)) * ((1 - r(2)) * c[0] + r(2) * c[4]) +
                                           r(1) * ((1 - r(2)) * c[3] + r(2) * c[7])) +
                            r(0) * ((1 - r(1)) * ((1 - r(2)) * c[1] + r(2) * c[5]) +
                                    r(1) * ((1 - r(2)) * c[2] + r(2) * c[6]));

    float weight = (1 - r(0)) * ( (1 - r(1)) * ((1 - r(2)) * w[0] + r(2) * w[4]) +
                               r(1) * ((1 - r(2)) * w[3] + r(2) * w[7])) +
                r(0) * ((1 - r(1)) * ((1 - r(2)) * w[1] + r(2) * w[5]) +
                        r(1) * ((1 - r(2)) * w[2] + r(2) * w[6]));

    return std::make_tuple(sdf, weight, color);
}

Eigen::Vector3f ScalableTSDFVolumeBase::GetNormalAt(const Eigen::Vector3f &p)
{
    Eigen::Vector3f n;
    const double half_gap = 0.99 * voxel_length_;
    for (int i = 0; i < 3; i++) {
        Eigen::Vector3f p0 = p;
        p0(i) -= half_gap;
        Eigen::Vector3f p1 = p;
        p1(i) += half_gap;
        n(i) = GetTSDFAt(p1) - GetTSDFAt(p0);
    }
    return n.normalized();
}

std::pair<Eigen::Vector3f, bool> ScalableTSDFVolumeBase::RayCasting(
    int x, int y, 
    const Eigen::Vector2f &min_max_depth, 
    const Eigen::Matrix3f &intrinsic_inv, 
    const Eigen::Matrix4f &extrinsic
)
{
    const Eigen::Matrix3f &inv_r = extrinsic.block<3, 3>(0, 0).transpose();
    const Eigen::Vector3f &t = extrinsic.block<3, 1>(0, 3);

    /// start pt
    Eigen::Vector3f pt_w_s = inv_r * (
        intrinsic_inv * Eigen::Vector3f(x * min_max_depth[0], y * min_max_depth[0], min_max_depth[0]) - t);

    /// end pt
    Eigen::Vector3f pt_w_e = inv_r * (
        intrinsic_inv * Eigen::Vector3f(x * min_max_depth[1], y * min_max_depth[1], min_max_depth[1]) - t);

    Eigen::Vector3f ray_dir = (pt_w_e - pt_w_s).normalized();

    float total_length = (pt_w_e - pt_w_s).norm();

    float length = 0.0f;

    Eigen::Vector3f pt_res = pt_w_s;
    float sdf = TSDF_DEFAULT_TSDF;
    while (length < total_length) {
        sdf = GetTSDFAt(pt_res);
        float step;

        if (std::fabs(sdf) < FLT_EPSILON) {
            /// not allocated
            step = volume_unit_length_;
        } else {

            /// find
            if (sdf <= 0.0f) return std::make_pair(pt_res, true);

            step = std::max(sdf * sdf_trunc_, voxel_length_);
        }

        length += step;

        pt_res += step * ray_dir;
    }

    return std::make_pair(pt_res, false);
}

std::tuple<Eigen::Vector3f, Eigen::Vector3f, bool> ScalableTSDFVolumeBase::ColoredRayCasting(
    int x, int y, 
    const Eigen::Vector2f &min_max_depth, 
    const Eigen::Matrix3f &intrinsic_inv, 
    const Eigen::Matrix4f &extrinsic
) {
    const Eigen::Matrix3f &inv_r = extrinsic.block<3, 3>(0, 0).transpose();
    const Eigen::Vector3f &t = extrinsic.block<3, 1>(0, 3);

    /// start pt
    Eigen::Vector3f pt_w_s = inv_r * (
            intrinsic_inv * Eigen::Vector3f(x * min_max_depth[0], y * min_max_depth[0], min_max_depth[0]) - t);

    /// end pt
    Eigen::Vector3f pt_w_e = inv_r * (
            intrinsic_inv * Eigen::Vector3f(x * min_max_depth[1], y * min_max_depth[1], min_max_depth[1]) - t);

    Eigen::Vector3f ray_dir = (pt_w_e - pt_w_s).normalized();

    float total_length = (pt_w_e - pt_w_s).norm();

    float length = 0.0f;

    Eigen::Vector3f pt_res = pt_w_s;

    Eigen::Vector3f color = TSDF_DEFAULT_COLOR;
    float sdf = TSDF_DEFAULT_TSDF, weight = TSDF_DEFAULT_WEIGHT;


    const float step_scale = sdf_trunc_;
    float step;

    while (length < total_length) {
        std::tie(sdf, color) = GetColoredTSDFAt(pt_res);

        if (std::fabs(sdf) < FLT_EPSILON) {
            /// not allocated
            step = volume_unit_length_;
        } else {

            /// find
            if (sdf <= FLT_EPSILON) break;

            step = std::max(sdf * sdf_trunc_, voxel_length_);
        }

        length += step;
        pt_res += step * ray_dir;
    }

    if (sdf < FLT_EPSILON){
        step = sdf * step_scale;
        pt_res += step * ray_dir;
        std::tie(sdf, weight, color) = GetWeightedColoredTSDFAt(pt_res);
        step = sdf * step_scale;
        pt_res += step * ray_dir;
        return std::make_tuple(pt_res, color, true);

    }

    return std::make_tuple(pt_res, color, false);
}

}

}