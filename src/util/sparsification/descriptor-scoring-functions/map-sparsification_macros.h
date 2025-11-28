//
// Created by SENSETIME\caoquan on 19-11-1.
//
#include <glog/logging.h>
#include <util/types.h>
#include <unordered_set>

#ifndef SENSESLAM_MAP_SPARSIFICATION_MACROS_H
#define SENSESLAM_MAP_SPARSIFICATION_MACROS_H

constexpr size_t kBitsPerByte = 8u;
typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> DescriptorType;
typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> DescriptorsType;

namespace DescriptorUtils {

inline bool getBit(unsigned int bit, const DescriptorType &descriptor) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(DescriptorType);
    int byte = bit / kBitsPerByte;
    int bit_in_byte = bit % kBitsPerByte;
    CHECK_LT(byte, descriptor.rows());

    return descriptor(byte) & (1 << bit_in_byte);
}

template <typename MeanType>
void floatDescriptorMean(const DescriptorsType &descriptors, Eigen::Matrix<MeanType, Eigen::Dynamic, 1> *mean) {
    CHECK_GT(descriptors.rows(), 0);
    CHECK_GT(descriptors.cols(), 0);
    CHECK_NOTNULL(mean)->resize(descriptors.rows() * kBitsPerByte, Eigen::NoChange);
    mean->setZero();

    std::vector<int> sums;
    sums.resize(descriptors.rows() * kBitsPerByte, 0);
    for (int i = 0; i < descriptors.cols(); ++i) {
        for (size_t bit = 0u; bit < descriptors.rows() * kBitsPerByte; ++bit) {
            if (getBit(bit, descriptors.col(i))) {
                ++sums[bit];
            }
        }
    }

    for (size_t bit = 0u; bit < sums.size(); ++bit) {
        (*mean)(bit) = static_cast<MeanType>(sums[bit]) / descriptors.cols();
    }
}

template <typename FloatDescriptorType>
double differenceToMeanSquaredNorm(const DescriptorType &descriptor, const FloatDescriptorType &mean) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(FloatDescriptorType);
    CHECK_EQ(mean.size(), static_cast<int>(descriptor.size() * kBitsPerByte));

    typedef typename FloatDescriptorType::Scalar FloatScalarType;
    Eigen::Matrix<FloatScalarType, Eigen::Dynamic, 1> bits_of_descriptor;
    bits_of_descriptor.resize(descriptor.size() * kBitsPerByte, Eigen::NoChange);
    for (size_t bit = 0u; bit < descriptor.size() * kBitsPerByte; ++bit) {
        if (getBit(bit, descriptor)) {
            bits_of_descriptor[bit] = 1;
        } else {
            bits_of_descriptor[bit] = 0;
        }
    }

    return (bits_of_descriptor - mean).squaredNorm();
}

inline double descriptorMeanStandardDeviation(const DescriptorsType &descriptors) {
    if (descriptors.cols() < 2) {
        return 0.;
    }
    Eigen::VectorXd mean;
    mean.resize(descriptors.rows() * kBitsPerByte, Eigen::NoChange);
    floatDescriptorMean(descriptors, &mean);

    double deviation_squared = 0;
    for (int i = 0; i < descriptors.cols(); ++i) {
        deviation_squared += differenceToMeanSquaredNorm(descriptors.col(i), mean);
    }
    deviation_squared /= descriptors.cols();
    return sqrt(deviation_squared);
}

}  // namespace DescriptorUtils

#endif  // SENSESLAM_MAP_SPARSIFICATION_MACROS_H
