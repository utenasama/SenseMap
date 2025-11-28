#pragma once

#include <vector>
#include <limits>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <limits>

#define TSDF_USE_16BIT_DATA 1

namespace sensemap {

namespace tsdf {

static constexpr float TSDF_DEFAULT_TSDF = 1.0f;
static constexpr float TSDF_DEFAULT_WEIGHT = 0.0f;
static const Eigen::Vector3f TSDF_DEFAULT_COLOR = Eigen::Vector3f(255, 255, 255);

#if TSDF_USE_16BIT_DATA
class tsdf_t {
    static constexpr int TSDF_DECIMAL_BITS   = 15;
    int16_t value_;
public:
    tsdf_t(float value) {
        int temp = std::round(value * (1 << TSDF_DECIMAL_BITS));
        // temp = std::min(temp, (int)std::numeric_limits<int16_t>::max());
        // temp = std::max(temp, (int)std::numeric_limits<int16_t>::min());
        value_ = (int16_t)temp;
    }
    tsdf_t() : tsdf_t(TSDF_DEFAULT_TSDF) {};
    operator float() const {
        return value_ * 1.0f / (1 << TSDF_DECIMAL_BITS);
    }
};
class weight_t {
    static constexpr int WEIGHT_DECIMAL_BITS = 6;
    uint16_t value_;
public:
    weight_t(float value) {
        int temp = std::round(value * (1 << WEIGHT_DECIMAL_BITS));
        temp = std::min(temp, (int)std::numeric_limits<uint16_t>::max());
        // temp = std::max(temp, (int)std::numeric_limits<uint16_t>::min());
        value_ = (uint16_t)temp;
    }
    weight_t() : weight_t(TSDF_DEFAULT_WEIGHT) {};
    operator float() const {
        return value_ * 1.0f / (1 << WEIGHT_DECIMAL_BITS);
    }
};
class color_t {
    // color format details
    static constexpr float UnpackR(uint32_t packed_rgb) {
        // 11111111 11100000 00000000 00000000
        // 8 bit int + 3 bit decimal
        return (
            ((packed_rgb) >> 21)
        ) * 0.125f;
    }
    static constexpr float UnpackG(uint32_t packed_rgb) {
        // 00000000 00011111 11111100 00000000
        // 8 bit int + 3 bit decimal
        return (
            ((packed_rgb & 0x001ffc00) >> 10)
        ) * 0.125f;
    }
    static constexpr float UnpackB(uint32_t packed_rgb) {
        // 00000000 00000000 00000011 11111111
        // 8 bit int + 2 bit decimal
        return (
            ((packed_rgb & 0x000003ff))
        ) * 0.25f;
    }
    static constexpr uint32_t PackRGB(float r, float g, float b) {
        return (
            (((uint32_t)((r * 8.0f) + 0.5f)) << 21) |
            (((uint32_t)((g * 8.0f) + 0.5f)) << 10) |
            (((uint32_t)((b * 4.0f) + 0.5f))      )
        );
    }
    uint32_t value_;
public:
    color_t(const Eigen::Vector3f & value) {
        value_ = PackRGB(value[0], value[1], value[2]);
    }
    color_t() : color_t(TSDF_DEFAULT_COLOR) {};
    operator Eigen::Vector3f() const {
        return Eigen::Vector3f(UnpackR(value_), UnpackG(value_), UnpackB(value_));
    }
};
#else
typedef float tsdf_t;
typedef float weight_t;
typedef Eigen::Vector3f color-t;
#endif

template<typename... Ts> class TSDFVolume;
template<typename... Ts> class ColoredTSDFVolume;

template<>
class TSDFVolume<> {
public:
    TSDFVolume(
        float voxel_length, int resolution, float sdf_trunc,
        const Eigen::Vector3f &origin,
        float min_depth = 0.0f, 
        float max_depth = std::numeric_limits<float>::max()) :
        voxel_length_(voxel_length),
        resolution_(resolution),
        sdf_trunc_(sdf_trunc),
        origin_(origin),
        min_depth_(min_depth),
        max_depth_(max_depth),
        tsdf_(resolution * resolution * resolution),
        weight_(resolution * resolution * resolution)
    {
        Reset();
    }

    TSDFVolume & operator=(const TSDFVolume & volume) {
        tsdf_ = volume.tsdf_;
        weight_ = volume.weight_;
        return *this;
    }

public:
    void Reset() {
        for (int i = 0; i < tsdf_.size(); i++) tsdf_[i] = TSDF_DEFAULT_TSDF;
        for (int i = 0; i < weight_.size(); i++) weight_[i] = TSDF_DEFAULT_WEIGHT;
    }

    void Integrate(
        const cv::Mat &depth,
        const Eigen::Matrix3f &intrinsic, 
        const Eigen::Matrix4f &extrinsic, 
        bool use_sdf_weight = true,
        bool de_integrate = false,
        float base_weight = 1.0f);

    inline int IndexOf(int x, int y, int z) const {
        return x * resolution_ * resolution_ + y * resolution_ + z;
    }

    inline int IndexOf(const Eigen::Vector3i &xyz) const {
        return IndexOf(xyz(0), xyz(1), xyz(2));
    }

public:
    const int resolution_;
    const float voxel_length_;
    const float sdf_trunc_;
    const float min_depth_;
    const float max_depth_;
    const Eigen::Vector3f origin_;
    
    std::vector<tsdf_t> tsdf_;
    std::vector<weight_t> weight_;
    int flag = 0;
};

template<typename T, typename... Ts>
class TSDFVolume<T, Ts...> : public TSDFVolume<Ts...>
{
public:
    TSDFVolume(
        float voxel_length, int resolution, float sdf_trunc,
        const Eigen::Vector3f &origin,
        float min_depth = 0.0f, 
        float max_depth = std::numeric_limits<float>::max()) :
        TSDFVolume<Ts...>(voxel_length, resolution, sdf_trunc, origin, min_depth, max_depth),
        extra_(resolution * resolution * resolution)
    {
        Reset();
    }

    TSDFVolume<T, Ts...> & operator=(const TSDFVolume<T, Ts...> & volume) {
        TSDFVolume<Ts...>::operator=(volume);
        extra_ = volume.extra_;
        return *this;
    }

public:
    void Reset() {
        TSDFVolume<Ts...>::Reset();
        for (int i = 0; i < extra_.size(); i++) extra_[i] = T();
    }

public:
    std::vector<T> extra_;
};

template<>
class ColoredTSDFVolume<> : public TSDFVolume<> {
public:
    ColoredTSDFVolume(
        float voxel_length, int resolution, float sdf_trunc,
        const Eigen::Vector3f &origin,
        float min_depth = 0.0f, 
        float max_depth = std::numeric_limits<float>::max()) :
        TSDFVolume<>(voxel_length, resolution, sdf_trunc, origin, min_depth, max_depth),
        color_(resolution * resolution * resolution)
    {
        Reset();
    }

    ColoredTSDFVolume<> & operator=(const ColoredTSDFVolume<> & volume) {
        TSDFVolume<>::operator=(volume);
        color_ = volume.color_;
        return *this;
    }

public:
    void Reset() {
        TSDFVolume<>::Reset();
        for (int i = 0; i < color_.size(); i++) color_[i] = TSDF_DEFAULT_COLOR;
    }

    void Integrate(
        const cv::Mat &depth, const cv::Mat &color, 
        const Eigen::Matrix3f &intrinsic, 
        const Eigen::Matrix4f &extrinsic, 
        bool use_sdf_weight = true,
        bool de_integrate = false,
        float base_weight = 1.0f);

public:
    std::vector<color_t> color_;
};

template<typename T, typename... Ts>
class ColoredTSDFVolume<T, Ts...> : public ColoredTSDFVolume<Ts...>
{
public:
    ColoredTSDFVolume(
        float voxel_length, int resolution, float sdf_trunc,
        const Eigen::Vector3f &origin,
        float min_depth = 0.0f, 
        float max_depth = std::numeric_limits<float>::max()) :
        ColoredTSDFVolume<Ts...>(voxel_length, resolution, sdf_trunc, origin, min_depth, max_depth),
        extra_(resolution * resolution * resolution)
    {
        Reset();
    }

    ColoredTSDFVolume<T, Ts...> & operator=(const ColoredTSDFVolume<T, Ts...> & volume) {
        ColoredTSDFVolume<Ts...>::operator=(volume);
        extra_ = volume.extra_;
        return *this;
    }

public:
    void Reset() {
        ColoredTSDFVolume<Ts...>::Reset();
        for (int i = 0; i < extra_.size(); i++) extra_[i] = T();
    }

public:
    std::vector<T> extra_;
};

}

}
