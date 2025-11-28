#pragma once

#include <vector>
#include <memory>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <util/obj.h>
#include <util/roi_box.h>
#include "UniformTSDFVolume.h"
#include "MarchingCubesConst.h"

namespace sensemap {

namespace tsdf {

template <typename T>
struct hash : std::unary_function<T, size_t> {
    std::size_t operator()(T const& matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class ScalableTSDFVolumeBase {
public:
    float voxel_length_;
    float volume_unit_length_;
    const float min_depth_;
    const float max_depth_;
    const float sdf_trunc_;
    const int volume_unit_resolution_;
    const int depth_sampling_stride_;

public:
    ScalableTSDFVolumeBase(
        float voxel_length, float sdf_trunc,
        float min_depth, float max_depth,
        int volume_unit_resolution,
        int depth_sampling_stride
    ) : voxel_length_(voxel_length),
        sdf_trunc_(sdf_trunc),
        min_depth_(min_depth),
        max_depth_(max_depth),
        volume_unit_resolution_(volume_unit_resolution),
        volume_unit_length_(voxel_length * volume_unit_resolution),
        depth_sampling_stride_(depth_sampling_stride)
    {
    }

    std::shared_ptr<TriangleMesh> ExtractTriangleMesh(float weight_thresh = 1.0f);
    std::shared_ptr<TriangleMesh> ExtractColoredTriangleMesh(float weight_thresh = 1.0f);
    std::shared_ptr<TriangleMesh> ExtractTSDF(float weight_thresh = 1.0f);
    void Integrate(const cv::Mat &depth, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic, float base_weight = 1.0f);
    void Integrate(const cv::Mat &depth, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic, bool de_integrate);
    void Integrate(const cv::Mat &depth, const cv::Mat &color, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic, float base_weight = 1.0f);
    void Integrate(const cv::Mat &depth, const cv::Mat &color, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic, bool de_integrate);
    
    float GetTSDFAt(const Eigen::Vector3f & p);
    std::tuple<float, float> GetWeightedTSDFAt(const Eigen::Vector3f & p);
    std::tuple<float, Eigen::Vector3f> GetColoredTSDFAt(const Eigen::Vector3f & p);
    std::tuple<float, float, Eigen::Vector3f> GetWeightedColoredTSDFAt(const Eigen::Vector3f & p);
    Eigen::Vector3f GetNormalAt(const Eigen::Vector3f & p);

    std::pair<Eigen::Vector3f, bool> RayCasting(
        int x, int y, 
        const Eigen::Vector2f &min_max_depth, 
        const Eigen::Matrix3f &intrinsic_inv, 
        const Eigen::Matrix4f &extrinsic
    );
    std::tuple<Eigen::Vector3f, Eigen::Vector3f, bool> ColoredRayCasting(
        int x, int y, 
        const Eigen::Vector2f &min_max_depth, 
        const Eigen::Matrix3f &intrinsic_inv, 
        const Eigen::Matrix4f &extrinsic
    );

    virtual std::shared_ptr<TSDFVolume<>> OpenVolumeUnit(const Eigen::Vector3i & index) = 0;
    virtual std::shared_ptr<TSDFVolume<>> FindVolumeUnit(const Eigen::Vector3i & index) = 0;
    virtual void ForeachVolumeUnit(std::function<void(Eigen::Vector3i, std::shared_ptr<TSDFVolume<>>)> func) = 0;
    virtual std::shared_ptr<ColoredTSDFVolume<>> OpenColoredVolumeUnit(const Eigen::Vector3i & index) = 0;
    virtual std::shared_ptr<ColoredTSDFVolume<>> FindColoredVolumeUnit(const Eigen::Vector3i & index) = 0;
    virtual void ForeachColoredVolumeUnit(std::function<void(Eigen::Vector3i, std::shared_ptr<ColoredTSDFVolume<>>)> func) = 0;

protected:
    std::mutex mutex_;
    std::condition_variable cv_;

protected:
    std::vector<Eigen::Vector3f> GetWorldPoints(const cv::Mat &depth, const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &extrinsic);
    Eigen::Vector3i LocateVolumeUnit(const Eigen::Vector3f &point, float volume_unit_length) {
        return Eigen::Vector3i((int)std::floor(point(0) / volume_unit_length),
                (int)std::floor(point(1) / volume_unit_length),
                (int)std::floor(point(2) / volume_unit_length));
    }
    Eigen::Vector3i LocateVolumeUnit(const Eigen::Vector3f &point) {
        return LocateVolumeUnit(point, volume_unit_length_);
    }
};

template<typename T>
struct VolumeUnit {
public:
    VolumeUnit() : volume_(nullptr) {}
public:
    std::shared_ptr<T> volume_;
    Eigen::Vector3i index_;
};

template<typename T=TSDFVolume<>>
class ScalableTSDFVolume : public ScalableTSDFVolumeBase
{
public:
    ScalableTSDFVolume(
        float voxel_length, float sdf_trunc,
        float min_depth = 0.0f, float max_depth = std::numeric_limits<float>::max(),
        int volume_unit_resolution = 16,
        int depth_sampling_stride = 4
    ) : ScalableTSDFVolumeBase(voxel_length, sdf_trunc, min_depth, max_depth, volume_unit_resolution, depth_sampling_stride)
    {
    }

public:
    void Reset()
    {
        volume_units_.clear();
    }
    void ReSample(float new_voxel_length)
    {
        const float new_volume_unit_length = new_voxel_length * volume_unit_resolution_;
        decltype(volume_units_) new_volume_units;

        for (auto & volume_unit : volume_units_) {
            Eigen::Vector3f volume_corners[8];
            volume_corners[0] = Eigen::Vector3f(
                volume_unit.first.x() * volume_unit_length_,
                volume_unit.first.y() * volume_unit_length_,
                volume_unit.first.z() * volume_unit_length_);
            volume_corners[1] = Eigen::Vector3f(
                volume_corners[0].x() + volume_unit_length_, 
                volume_corners[0].y(), 
                volume_corners[0].z());
            volume_corners[2] = Eigen::Vector3f(
                volume_corners[0].x(), 
                volume_corners[0].y() + volume_unit_length_, 
                volume_corners[0].z());
            volume_corners[3] = Eigen::Vector3f(
                volume_corners[0].x(), 
                volume_corners[0].y(), 
                volume_corners[0].z() + volume_unit_length_);
            volume_corners[4] = Eigen::Vector3f(
                volume_corners[0].x() + volume_unit_length_, 
                volume_corners[0].y() + volume_unit_length_, 
                volume_corners[0].z());
            volume_corners[5] = Eigen::Vector3f(
                volume_corners[0].x(), 
                volume_corners[0].y() + volume_unit_length_, 
                volume_corners[0].z() + volume_unit_length_);
            volume_corners[6] = Eigen::Vector3f(
                volume_corners[0].x() + volume_unit_length_, 
                volume_corners[0].y(), 
                volume_corners[0].z() + volume_unit_length_);
            volume_corners[7] = Eigen::Vector3f(
                volume_corners[0].x() + volume_unit_length_, 
                volume_corners[0].y() + volume_unit_length_, 
                volume_corners[0].z() + volume_unit_length_);

            for (int i = 0; i < 8; i++) {
                Eigen::Vector3i new_volume_index = LocateVolumeUnit(volume_corners[i], new_volume_unit_length);
                OpenVolumeUnit(new_volume_index, new_volume_units, new_volume_unit_length);
            }
        }

        std::cout << volume_units_.size() << " -> " << new_volume_units.size() << std::endl;

        for (auto& new_volume_unit : new_volume_units) {
            const Eigen::Vector3f volume_position(
                new_volume_unit.first.x() * new_volume_unit_length,
                new_volume_unit.first.y() * new_volume_unit_length,
                new_volume_unit.first.z() * new_volume_unit_length);

            auto& volume = *new_volume_unit.second.volume_;
            for (int x = 0; x < volume_unit_resolution_; x++) {
                for (int y = 0; y < volume_unit_resolution_; y++) {
                    for (int z = 0; z < volume_unit_resolution_; z++) {
                        const Eigen::Vector3f volume_position_xyz(
                            volume_position.x() + new_voxel_length * x + 0.5f * voxel_length_,
                            volume_position.y() + new_voxel_length * y + 0.5f * voxel_length_,
                            volume_position.z() + new_voxel_length * z + 0.5f * voxel_length_);

                        const int index = volume.IndexOf(x, y, z);
                        auto tw = GetWeightedTSDFAt(volume_position_xyz);
                        volume.tsdf_[index] = std::get<0>(tw);
                        volume.weight_[index] = std::get<1>(tw);
                    }
                }
            }
        }

        voxel_length_ = new_voxel_length;
        volume_unit_length_ = new_volume_unit_length;
        volume_units_.swap(new_volume_units);
    }
    void ColoredReSample(float new_voxel_length)
    {
        const float new_volume_unit_length = new_voxel_length * volume_unit_resolution_;
        decltype(volume_units_) new_volume_units;

        for (auto & volume_unit : volume_units_) {
            Eigen::Vector3f volume_corners[8];
            volume_corners[0] = Eigen::Vector3f(
                volume_unit.first.x() * volume_unit_length_,
                volume_unit.first.y() * volume_unit_length_,
                volume_unit.first.z() * volume_unit_length_);
            volume_corners[1] = Eigen::Vector3f(
                volume_corners[0].x() + volume_unit_length_, 
                volume_corners[0].y(), 
                volume_corners[0].z());
            volume_corners[2] = Eigen::Vector3f(
                volume_corners[0].x(), 
                volume_corners[0].y() + volume_unit_length_, 
                volume_corners[0].z());
            volume_corners[3] = Eigen::Vector3f(
                volume_corners[0].x(), 
                volume_corners[0].y(), 
                volume_corners[0].z() + volume_unit_length_);
            volume_corners[4] = Eigen::Vector3f(
                volume_corners[0].x() + volume_unit_length_, 
                volume_corners[0].y() + volume_unit_length_, 
                volume_corners[0].z());
            volume_corners[5] = Eigen::Vector3f(
                volume_corners[0].x(), 
                volume_corners[0].y() + volume_unit_length_, 
                volume_corners[0].z() + volume_unit_length_);
            volume_corners[6] = Eigen::Vector3f(
                volume_corners[0].x() + volume_unit_length_, 
                volume_corners[0].y(), 
                volume_corners[0].z() + volume_unit_length_);
            volume_corners[7] = Eigen::Vector3f(
                volume_corners[0].x() + volume_unit_length_, 
                volume_corners[0].y() + volume_unit_length_, 
                volume_corners[0].z() + volume_unit_length_);

            for (int i = 0; i < 8; i++) {
                Eigen::Vector3i new_volume_index = LocateVolumeUnit(volume_corners[i], new_volume_unit_length);
                OpenVolumeUnit(new_volume_index, new_volume_units, new_volume_unit_length);
            }
        }

        std::cout << volume_units_.size() << " -> " << new_volume_units.size() << std::endl;

        for (auto& new_volume_unit : new_volume_units) {
            const Eigen::Vector3f volume_position(
                new_volume_unit.first.x() * new_volume_unit_length,
                new_volume_unit.first.y() * new_volume_unit_length,
                new_volume_unit.first.z() * new_volume_unit_length);

            auto& volume = *std::static_pointer_cast<ColoredTSDFVolume<>>(new_volume_unit.second.volume_);
            for (int x = 0; x < volume_unit_resolution_; x++) {
                for (int y = 0; y < volume_unit_resolution_; y++) {
                    for (int z = 0; z < volume_unit_resolution_; z++) {
                        const Eigen::Vector3f volume_position_xyz(
                            volume_position.x() + new_voxel_length * x + 0.5f * voxel_length_,
                            volume_position.y() + new_voxel_length * y + 0.5f * voxel_length_,
                            volume_position.z() + new_voxel_length * z + 0.5f * voxel_length_);

                        const int index = volume.IndexOf(x, y, z);
                        auto twc = GetWeightedColoredTSDFAt(volume_position_xyz);
                        volume.tsdf_[index] = std::get<0>(twc);
                        volume.color_[index] = std::get<2>(twc);
                        volume.weight_[index] = std::get<1>(twc);
                    }
                }
            }
        }

        voxel_length_ = new_voxel_length;
        volume_unit_length_ = new_volume_unit_length;
        volume_units_.swap(new_volume_units);
    }

    /// Assume the index of the volume unit is (x, y, z), then the unit spans
    /// from (x, y, z) * volume_unit_length_
    /// to (x + 1, y + 1, z + 1) * volume_unit_length_
    std::unordered_map<Eigen::Vector3i, VolumeUnit<T>, hash<Eigen::Vector3i>> volume_units_;

    std::shared_ptr<TSDFVolume<>> OpenVolumeUnit(const Eigen::Vector3i &index) {
        return std::static_pointer_cast<TSDFVolume<>>(OpenVolumeUnit(index, volume_units_, volume_unit_length_));
    }

    std::shared_ptr<TSDFVolume<>> FindVolumeUnit(const Eigen::Vector3i & index) {
        auto find = volume_units_.find(index);
        if (find == volume_units_.end()) return nullptr;
        return std::static_pointer_cast<TSDFVolume<>>(find->second.volume_);
    };

    void ForeachVolumeUnit(std::function<void(Eigen::Vector3i, std::shared_ptr<TSDFVolume<>>)> func) {
        for (auto & volume : volume_units_) {
            func(volume.first, std::static_pointer_cast<TSDFVolume<>>(volume.second.volume_));
        }
    }

    std::shared_ptr<ColoredTSDFVolume<>> OpenColoredVolumeUnit(const Eigen::Vector3i &index) {
        return std::static_pointer_cast<ColoredTSDFVolume<>>(OpenVolumeUnit(index, ScalableTSDFVolume<T>::volume_units_, volume_unit_length_));
    }

    std::shared_ptr<ColoredTSDFVolume<>> FindColoredVolumeUnit(const Eigen::Vector3i & index) {
        auto find = ScalableTSDFVolume<T>::volume_units_.find(index);
        if (find == ScalableTSDFVolume<T>::volume_units_.end()) return nullptr;
        return std::static_pointer_cast<ColoredTSDFVolume<>>(find->second.volume_);
    };

    void ForeachColoredVolumeUnit(std::function<void(Eigen::Vector3i, std::shared_ptr<ColoredTSDFVolume<>>)> func) {
        for (auto & volume : ScalableTSDFVolume<T>::volume_units_) {
            func(volume.first, std::static_pointer_cast<ColoredTSDFVolume<>>(volume.second.volume_));
        }
    }

protected:
    std::shared_ptr<T> OpenVolumeUnit(
        const Eigen::Vector3i &index, 
        std::unordered_map<Eigen::Vector3i, VolumeUnit<T>, hash<Eigen::Vector3i>> &volume_units,
        float volume_unit_length
    ) {
        auto &unit = volume_units[index];
        if (!unit.volume_) {
            Eigen::Vector3f orig = index.cast<float>() * volume_unit_length;
            unit.volume_.reset(new T(
                    voxel_length_, volume_unit_resolution_, sdf_trunc_, orig, min_depth_, max_depth_));
            unit.index_ = index;
        }
        return unit.volume_;
    }
};

}

}