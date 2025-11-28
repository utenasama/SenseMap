//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "feature/utils.h"
#include "util/math.h"
#include <fstream>
#include <Eigen/Eigenvalues>

namespace sensemap {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DoubleDescriptors;

std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
		const FeatureKeypoints& keypoints) {
	std::vector<Eigen::Vector2d> points(keypoints.size());
	for (size_t i = 0; i < keypoints.size(); ++i) {
		points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
	}
	return points;
}

Eigen::MatrixXf L2NormalizeFeatureDescriptors(
		const Eigen::MatrixXf& descriptors) {
	return descriptors.rowwise().normalized();
}

Eigen::MatrixXd L2NormalizeFeatureDescriptorsD(
		const Eigen::MatrixXd& descriptors) {
	return descriptors.rowwise().normalized();
}

Eigen::MatrixXf L1RootNormalizeFeatureDescriptors(
		const Eigen::MatrixXf& descriptors) {
	Eigen::MatrixXf descriptors_normalized(descriptors.rows(),
	                                       descriptors.cols());
	for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
		const float norm = descriptors.row(r).lpNorm<1>();
		descriptors_normalized.row(r) = descriptors.row(r) / norm;
		descriptors_normalized.row(r) =
				descriptors_normalized.row(r).array().sqrt();
	}

	return descriptors_normalized;
}


Eigen::MatrixXf L1NormalizeFeatureDescriptors(
		const Eigen::MatrixXf& descriptors) {
	Eigen::MatrixXf descriptors_normalized(descriptors.rows(),
	                                       descriptors.cols());
	for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
		const float norm = descriptors.row(r).lpNorm<1>();
		descriptors_normalized.row(r) = descriptors.row(r) / norm;
		
		for(size_t c = 0; c< descriptors.cols(); ++c){
			descriptors_normalized(r,c) = descriptors_normalized(r,c) + 1.0;
		}
	}

	return descriptors_normalized;
}

FeatureDescriptors FeatureDescriptorsToUnsignedByte(
		const Eigen::MatrixXf& descriptors) {
	FeatureDescriptors descriptors_unsigned_byte(descriptors.rows(),
	                                             descriptors.cols());
	for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
		for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {

			const float scaled_value = std::round(512.0f * descriptors(r, c));

			descriptors_unsigned_byte(r, c) =
					TruncateCast<float, uint8_t>(scaled_value);
		}
	}
	return descriptors_unsigned_byte;
}

// Convert unsigned byte point feature descriptor to float
void FeatureDescriptorsTofloat(const FeatureDescriptors& descriptors,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& float_descriptors){
	
	float_descriptors.resize(descriptors.rows(),descriptors.cols());

	for (size_t r = 0; r < descriptors.rows(); ++r) {
		for (size_t c = 0; c < descriptors.cols(); ++c) {
			float_descriptors(r, c) = static_cast<float> (descriptors(r,c));
		}
	}

}

// Convert unsigned byte point feature descriptor to float
void CompressedFeatureDescriptorsTofloat(const CompressedFeatureDescriptors& descriptors,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& float_descriptors){
	
	float_descriptors.resize(descriptors.rows(),descriptors.cols());

	if(descriptors.cols() == 128){
		for (size_t r = 0; r < descriptors.rows(); ++r) {
			for (size_t c = 0; c < descriptors.cols(); ++c) {
				float_descriptors(r, c) = static_cast<float> (descriptors(r,c));
			}
		}
	}
	else{
		for (size_t r = 0; r < descriptors.rows(); ++r) {
			for (size_t c = 0; c < descriptors.cols(); ++c) {
				float_descriptors(r, c) = static_cast<float> ((int8_t)(descriptors(r,c)));
			}
		}
	}
	
}

void FeatureDescriptorsToDouble(const FeatureDescriptors& descriptors,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& double_descriptors){
	
	double_descriptors.resize(descriptors.rows(),descriptors.cols());
	for (size_t r = 0; r < descriptors.rows(); ++r) {
		for (size_t c = 0; c < descriptors.cols(); ++c) {
			double_descriptors(r, c) = static_cast<double> (descriptors(r,c));
		}
	}
}

void ExtractTopScaleFeatures(FeatureKeypoints* keypoints,
                             FeatureDescriptors* descriptors,
                             const size_t num_features) {
	CHECK_EQ(keypoints->size(), descriptors->rows());
	CHECK_GT(num_features, 0);

	if (static_cast<size_t>(descriptors->rows()) <= num_features) {
		return;
	}

	FeatureKeypoints top_scale_keypoints;
	FeatureDescriptors top_scale_descriptors;

	std::vector<std::pair<size_t, float>> scales;
	scales.reserve(static_cast<size_t>(keypoints->size()));
	for (size_t i = 0; i < keypoints->size(); ++i) {
		scales.emplace_back(i, (*keypoints)[i].ComputeScale());
	}

	std::partial_sort(scales.begin(), scales.begin() + num_features,
	                  scales.end(),
	                  [](const std::pair<size_t, float> scale1,
	                     const std::pair<size_t, float> scale2) {
		                  return scale1.second > scale2.second;
	                  });

	top_scale_keypoints.resize(num_features);
	top_scale_descriptors.resize(num_features, descriptors->cols());
	for (size_t i = 0; i < num_features; ++i) {
		top_scale_keypoints[i] = (*keypoints)[scales[i].first];
		top_scale_descriptors.row(i) = descriptors->row(scales[i].first);
	}

	*keypoints = top_scale_keypoints;
	*descriptors = top_scale_descriptors;
}


void GenerateDimPiarsForBinarization(std::vector<std::pair<int, int>>& dim_index_pairs, int binary_feature_dim){
	
	dim_index_pairs.resize(binary_feature_dim);
	
	std::vector<std::pair<int,int>> all_pairs;
	for(int i = 0; i< 128; ++i){
		for(int j = i+1; j< 128; ++j){
			all_pairs.emplace_back(i,j);
		}
	}

	std::vector<bool> b_selected(all_pairs.size(),false);

	srand(int(time(0)));
	
	for(size_t pair_id = 0; pair_id < binary_feature_dim; pair_id ++){

		while(1){
			int j  = rand()%all_pairs.size();
			if(b_selected[j]){
				continue;
			}
			else{
				dim_index_pairs[pair_id] = all_pairs[j];
				b_selected[j] = true;
				break;
			}
		}													
	}
}


void WriteDimPiarsForBinarization(const std::vector<std::pair<int, int>>& dim_index_pairs,
                                  const std::string& save_path){
	
	std::ofstream file_binary_pattern(save_path, std::ios::binary);
	CHECK(file_binary_pattern.is_open());

	int pair_count = dim_index_pairs.size();
	file_binary_pattern.write((char*)&pair_count, sizeof(int));

	for(size_t i = 0; i< dim_index_pairs.size(); ++i){
		int dim1 = dim_index_pairs[i].first;
		int dim2 = dim_index_pairs[i].second;

		file_binary_pattern.write((char*)&dim1, sizeof(int));
		file_binary_pattern.write((char*)&dim2, sizeof(int));
	}
	file_binary_pattern.close();
}


void ReadDimPiarsForBinarization(std::vector<std::pair<int, int>>& dim_index_pairs, const std::string& save_path){
	
	std::ifstream file_binary_pattern(save_path, std::ios::binary);
	CHECK(file_binary_pattern.is_open());

	int pair_count;
	file_binary_pattern.read(reinterpret_cast<char*>(&pair_count), sizeof(int));

	dim_index_pairs.resize(pair_count);

	for(size_t i = 0; i<pair_count; ++i){
		int dim1, dim2;
		file_binary_pattern.read(reinterpret_cast<char*>(&dim1), sizeof(int));
		file_binary_pattern.read(reinterpret_cast<char*>(&dim2), sizeof(int));
		dim_index_pairs[i].first = dim1;
		dim_index_pairs[i].second = dim2;
	}

	file_binary_pattern.close();
}

void ReadPcaProjectionMatrix(Eigen::Matrix<double, 128, 128> &pca_matrix, 
							 Eigen::Matrix<double, 128, 1>& embedding_thresholds,const std::string& save_path){
	
	std::ifstream file_pca_matrix(save_path, std::ios::binary);
	CHECK(file_pca_matrix.is_open());

	for(int i = 0; i< 128; ++i){
        for(int j= 0; j< 128; ++j){
            double elem; 
            file_pca_matrix.read(reinterpret_cast<char*>(&elem), sizeof(double));
			pca_matrix(i,j) = elem;
		}
    }

    for(int i = 0; i < 128; ++i){
        double elem;
        file_pca_matrix.read(reinterpret_cast<char*>(&elem),sizeof(double));
		embedding_thresholds(i) = elem;
    }

	file_pca_matrix.close();
}


void ConvertFeatureDescriptorsToBinary(const FeatureDescriptors& descriptors,
									   const std::vector<std::pair<int,int>> &dim_index_pairs, 
                                       BinaryFeatureDescriptors& binary_descriptors){

	size_t section_dim = sizeof(uint64_t)*8;
	binary_descriptors.resize(descriptors.rows(),dim_index_pairs.size()/section_dim);
	
	for (size_t r = 0; r < descriptors.rows(); ++r) {

		for(size_t section = 0; section < binary_descriptors.cols(); section ++){	
			uint64_t long_descriptor = 0;
			for(size_t pair_id = 0; pair_id < section_dim; ++pair_id){
				uint8_t org_dim1 = descriptors(r,dim_index_pairs[section * section_dim + pair_id].first);
				uint8_t org_dim2 = descriptors(r,dim_index_pairs[section * section_dim + pair_id].second);

				if(org_dim1 > org_dim2){
					long_descriptor += (1<<pair_id);
				}
			}
			binary_descriptors(r,section) = long_descriptor;
		}
	}
}

void ConvertFeatureDescriptorsToBinary(const FeatureDescriptors& descriptors,
                                       const Eigen::Matrix<double, 128, 128>& pca_matrix,
                                       const Eigen::Matrix<double, 128, 1>& embedding_thresholds,
                                       BinaryFeatureDescriptors& binary_descriptors){

	
	size_t section_dim = sizeof(uint64_t)*8;
	binary_descriptors.resize(descriptors.rows(), 1);

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> double_descriptors;
	FeatureDescriptorsToDouble(descriptors, double_descriptors);

	Eigen::Matrix<double, Eigen::Dynamic,128> projected_descriptors = double_descriptors * (pca_matrix.transpose());


	for (size_t r = 0; r < projected_descriptors.rows(); ++r) {

		for(size_t section = 0; section < binary_descriptors.cols(); section ++){	
			uint64_t long_descriptor = 0;
			for(size_t dim = 0; dim < section_dim; ++ dim){
				if(projected_descriptors(r, section * section_dim + dim) >= embedding_thresholds(section * section_dim + dim)){
					long_descriptor += (1<<dim);
				}
			}
			binary_descriptors(r,section) = long_descriptor;
		}
	}
}

void CompressFeatureDescriptors(const FeatureDescriptors& descriptors,
								CompressedFeatureDescriptors& compressed_descriptors,
                                const Eigen::Matrix<double, 128, 128>& pca_matrix,
                                const Eigen::Matrix<double, 128, 1>& embedding_thresholds,
								const int compressed_dimension){
    
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> double_descriptors;
    FeatureDescriptorsToDouble(descriptors, double_descriptors);


    Eigen::Matrix<double, Eigen::Dynamic, 128> compressed_pca_matrix;
    compressed_pca_matrix.resize(compressed_dimension, 128);

    for (int i = 0; i < compressed_dimension; ++i) {
        compressed_pca_matrix.row(i) = pca_matrix.row(i);
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> projected_descriptors =
        double_descriptors * (compressed_pca_matrix.transpose());


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> normalized_projected_descriptors =
        L2NormalizeFeatureDescriptorsD(projected_descriptors);

    compressed_descriptors.resize(normalized_projected_descriptors.rows(), normalized_projected_descriptors.cols());
    for (Eigen::MatrixXd::Index r = 0; r < normalized_projected_descriptors.rows(); ++r) {
        for (Eigen::MatrixXd::Index c = 0; c < normalized_projected_descriptors.cols(); ++c) {
            const float scaled_value = std::round(128.0 * normalized_projected_descriptors(r, c));
            compressed_descriptors(r, c) = TruncateCast<float, int8_t>(scaled_value);
        }
    }
}

void ConvertFloatDescriptorsToBinary(const Eigen::MatrixXf& descriptors,
									   const std::vector<std::pair<int,int>> &dim_index_pairs, 
                                       BinaryFeatureDescriptors& binary_descriptors){

	size_t section_dim = sizeof(uint64_t)*8;
	binary_descriptors.resize(descriptors.rows(),dim_index_pairs.size()/section_dim);
	
	for (size_t r = 0; r < descriptors.rows(); ++r) {

		for(size_t section = 0; section < binary_descriptors.cols(); section ++){	
			uint64_t long_descriptor = 0;
			for(size_t pair_id = 0; pair_id < section_dim; ++pair_id){
				float org_dim1 = descriptors(r,dim_index_pairs[section * section_dim + pair_id].first);
				float org_dim2 = descriptors(r,dim_index_pairs[section * section_dim + pair_id].second);

				if(org_dim1 > org_dim2){
					long_descriptor += (1<<pair_id);
				}
			}
			binary_descriptors(r,section) = long_descriptor;
		}
	}
}

void PcaTraining(const FeatureDescriptors training_descriptors, Eigen::Matrix<double, 128, 128>& pca_matrix,
                 Eigen::Matrix<double, 128, 1>& embedding_thresholds) {
    DoubleDescriptors double_training_descriptors;
    FeatureDescriptorsToDouble(training_descriptors, double_training_descriptors);
    std::cout << "training descriptor count and dimension: " << double_training_descriptors.rows() << " "
              << double_training_descriptors.cols() << std::endl;

    Eigen::Matrix<double, 1, 128, Eigen::RowMajor> mean_descriptor =
        Eigen::Matrix<double, 1, 128, Eigen::RowMajor>::Zero();

    for (int i = 0; i < double_training_descriptors.rows(); ++i) {
        mean_descriptor += double_training_descriptors.row(i);
    }

    mean_descriptor /= double_training_descriptors.rows();

    Eigen::Matrix<double, 128, 128> cov = Eigen::Matrix<double, 128, 128>::Zero();

    for (int i = 0; i < double_training_descriptors.rows(); ++i) {
        cov += (double_training_descriptors.row(i) - mean_descriptor).transpose() *
               (double_training_descriptors.row(i) - mean_descriptor);
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 128, 128>> eigenSolver(cov);

    Eigen::Matrix<double, 128, 1> eigen_values = eigenSolver.eigenvalues();
    Eigen::Matrix<double, 128, 128> eigen_vectors = eigenSolver.eigenvectors();

    std::cout << "eigen_values: " << std::endl;
    std::cout << eigen_values << std::endl;

    for (int i = 0; i < 128; ++i) {
        pca_matrix.col(i) = eigen_vectors.col(127 - i);
    }

    Eigen::Matrix<double, 128, 128> pca_matrix_trans = pca_matrix.transpose();

    Eigen::Matrix<double, 128, Eigen::Dynamic> projected_descriptors =
        pca_matrix_trans * (double_training_descriptors.transpose());
    for (int i = 0; i < 128; ++i) {
        std::vector<double> values(projected_descriptors.cols());
        double min_value = std::numeric_limits<double>::max();
        double max_value = std::numeric_limits<double>::min();

        for (int j = 0; j < projected_descriptors.cols(); ++j) {
            values[j] = projected_descriptors(i, j);
            if (values[j] > max_value) {
                max_value = values[j];
            }
            if (values[j] < min_value) {
                min_value = values[j];
            }
        }

        int nth = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + nth, values.end());

        embedding_thresholds(i) = values[nth];
    }

    pca_matrix = pca_matrix_trans;
}

}  // namespace sensemap
