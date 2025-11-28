//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_FEATURE_UTILS_H_
#define SENSEMAP_FEATURE_UTILS_H_

#include "feature/types.h"

namespace sensemap {

// Convert feature keypoints to vector of points.
std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
		const FeatureKeypoints& keypoints);

// L2-normalize feature descriptor, where each row represents one feature.
Eigen::MatrixXf L2NormalizeFeatureDescriptors(
		const Eigen::MatrixXf& descriptors);


// L2-normalize feature descriptor, where each row represents one feature.
Eigen::MatrixXd L2NormalizeFeatureDescriptorsD(
		const Eigen::MatrixXd& descriptors);

// L1-Root-normalize feature descriptors, where each row represents one feature.
// See "Three things everyone should know to improve object retrieval",
// Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
Eigen::MatrixXf L1RootNormalizeFeatureDescriptors(
		const Eigen::MatrixXf& descriptors);

Eigen::MatrixXf L1NormalizeFeatureDescriptors(
		const Eigen::MatrixXf& descriptors);

// Convert normalized floating point feature descriptor to unsigned byte
// representation by linear scaling from range [0, 0.5] to [0, 255]. Truncation
// to a maximum value of 0.5 is used to avoid precision loss and follows the
// common practice of representing SIFT vectors.
FeatureDescriptors FeatureDescriptorsToUnsignedByte(
		const Eigen::MatrixXf& descriptors);



// Convert unsigned byte point feature descriptor to float
void FeatureDescriptorsTofloat(const FeatureDescriptors& descriptors,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& float_descriptors);


// Convert unsigned byte point feature descriptor to float
void CompressedFeatureDescriptorsTofloat(const CompressedFeatureDescriptors& descriptors,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& float_descriptors);


// Convert unsigned byte point feature descriptor to double
void FeatureDescriptorsToDouble(const FeatureDescriptors& descriptors,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& double_descriptors);

void PcaTraining(const FeatureDescriptors training_descriptors, Eigen::Matrix<double, 128, 128>& pca_matrix,
                 Eigen::Matrix<double, 128, 1>& embedding_thresholds);

void ReadPcaProjectionMatrix(Eigen::Matrix<double, 128, 128>& pca_matrix, 
							 Eigen::Matrix<double, 128, 1>& embedding_thresholds,const std::string& save_path);


// Convert float descriptors to binary descriptors
void GenerateDimPiarsForBinarization(std::vector<std::pair<int, int>>& dim_index_pairs, int binary_feature_dim=256);
void WriteDimPiarsForBinarization(const std::vector<std::pair<int, int>>& dim_index_pairs,
                                  const std::string& save_path);
void ReadDimPiarsForBinarization(std::vector<std::pair<int, int>>& dim_index_pairs, const std::string& save_path);


void ConvertFeatureDescriptorsToBinary(const FeatureDescriptors& descriptors,
									   const std::vector<std::pair<int,int>> &dim_index_pairs, 
                                       BinaryFeatureDescriptors& binary_descriptors);

void ConvertFeatureDescriptorsToBinary(const FeatureDescriptors& descriptors,
                                       const Eigen::Matrix<double, 128, 128>& pca_matrix,
                                       const Eigen::Matrix<double, 128, 1>& embedding_thresholds,
                                       BinaryFeatureDescriptors& binary_descriptors);

void ConvertFloatDescriptorsToBinary(const FeatureDescriptors& descriptors,
									   const std::vector<std::pair<int,int>> &dim_index_pairs, 
                                       BinaryFeatureDescriptors& binary_descriptors);

void CompressFeatureDescriptors(const FeatureDescriptors& descriptors,
                                CompressedFeatureDescriptors& compressed_descriptors,
                                const Eigen::Matrix<double, 128, 128>& pca_matrix,
                                const Eigen::Matrix<double, 128, 1>& embedding_thresholds,
								const int compressed_dimension = 32);


// Extract the descriptors corresponding to the largest-scale features.
void ExtractTopScaleFeatures(FeatureKeypoints* keypoints,
                             FeatureDescriptors* descriptors,
                             const size_t num_features);

} // namespace sensemap


#endif //SENSEMAP_FEATURE_UTILS_H
