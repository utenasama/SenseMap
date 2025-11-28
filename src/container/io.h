//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CONTAINER_IO_H
#define SENSEMAP_CONTAINER_IO_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/image.h"
#include "base/track.h"
#include "util/types.h"
#include "feature_data_container.h"

namespace sensemap {

// Camrera text file in the following format:
//    LINE_1:            CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
//    LINE_I:            ...
//    LINE_NUM_CAMERA:   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
// For example:
//    1 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
//	  2 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
//	  3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531
void ReadCamerasText(const std::string& path, CameraPtrUmap* camera_data_ump,
                     bool camera_rig);
void WriteCamerasText(const std::string& path, const CameraPtrUmap& camera_data_ump);
void ReadCamerasBinary(const std::string& path, CameraPtrUmap *camera_data_ump,
                       bool camera_rig);
void WriteCamerasBinary(const std::string& path, const CameraPtrUmap& camera_data_ump);

// Serialization for camera-rig model.
// Serialization for camera-rig model.
void ReadLocalCamerasText(const std::string& path, CameraPtrUmap *camera_data_ump);
void WriteLocalCamerasText(const std::string& path, const CameraPtrUmap& camera_data_ump);
void ReadLocalCamerasBinary(const std::string& path, CameraPtrUmap *camera_data_ump);
void WriteLocalCamerasBinary(const std::string& path, const CameraPtrUmap& camera_data_ump);
void ReadLocalImagesBinary(const std::string& path, FeatureDataPtrUmap *image_data_ump);
void WriteLocalImagesBinary(const std::string& path, const FeatureDataPtrUmap& image_data_ump);

// FeatureData text file in the following format:
//    LINE_0_0:                     IMAGE_PATH NUM_IMAGE
//    LINE_1_0:                     IMAGE_ID, CAMERA_ID, NAME, NUM_FEATURES DIM
//    LINE_1_1:                     X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//    LINE_1_J:                     ...
//    LINE_1_NUM_FEATURES:          X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//    ...                           ...
//    LINE_I_0:                     ...
//    LINE_I_1:                     ...
//    LINE_I_J:                     ...
//    LINE_I_NUM_FEATURES:          ...
//    ...                           ...
//    LINE_NUM_IMAGE_0:             IMAGE_ID, CAMERA_ID, NAME, NUM_FEATURES DIM
//    LINE_NUM_IMAGE_1:             X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//    LINE_NUM_IMAGE_J:             ...
//    LINE_NUM_IMAGE_NUM_FEATURES:  X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
// For example:
//    3 /mnt/d/Benchmark/ETH3D/pipes/images/dslr_images
//    1 1 DSC_1.jpg 2 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//    2 1 DSC_2.jpg 3 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//    3 1 DSC_3.jpg 1 4
//    0.32 0.12 1.23 1.0 1 2 3 4
void ReadFeatureDataText(const std::string& path, std::string* image_path,
                         FeatureDataPtrUmap* feature_data_ump,
                         FeatureNameUmap* feature_data_name);
void WriteFeatureDataText(const std::string& path,
                          const std::string& image_path,
                          const FeatureDataPtrUmap& feature_data_ump);

void WriteFeatureDataBinary(const std::string &path,
                          const std::string& image_path,
                          const FeatureDataPtrUmap &feature_data_ump, bool write_descriptors = true);

void WriteBinaryDescriptorsIO(const std::string &path,
                          const FeatureDataPtrUmap &feature_data_ump);

void ReadFeatureDataBinary(const std::string &path, std::string* image_path,
                         FeatureDataPtrUmap *feature_data_ump,
                         FeatureNameUmap* feature_data_name);

void ReadFeatureBinaryDataWithoutDescriptor(const std::string &path, std::string* image_path,
                         FeatureDataPtrUmap *feature_data_ump,
                         FeatureNameUmap* feature_data_name);

void ReadBinaryDescriptorsIO(const std::string &path, 
                         FeatureDataPtrUmap *feature_data_ump);



void ReadSubPanoramaDataText(const std::string& path, 
                             FeatureDataPtrUmap* feature_data_ump);
void WriteSubPanoramaDataText(const std::string& path, 
                              const FeatureDataPtrUmap& feature_data_ump);

void ReadSubPanoramaDataBinary(const std::string& path, 
                               FeatureDataPtrUmap* feature_data_ump);
void WriteSubPanoramaDataBinary(const std::string& path, 
                                const FeatureDataPtrUmap& feature_data_ump);



void ReadGlobalFeaturesDataBinary(const std::string& path, FeatureDataPtrUmap* feature_data_ump);
void WriteGlobalFeaturesDataBinary(const std::string& path, const FeatureDataPtrUmap& feature_data_ump);




// IO for AprilTag Data
void ReadAprilTagDataText(const std::string& path,
                          FeatureDataPtrUmap* feature_data_ump);
void WriteAprilTagDataText(const std::string& path,
                           const FeatureDataPtrUmap& feature_data_ump);

void ReadAprilTagDataBinary(const std::string& path,
                            FeatureDataPtrUmap* feature_data_ump);
void WriteAprilTagDataBinary(const std::string& path,
                             const FeatureDataPtrUmap& feature_data_ump);

// IO for GPS Data
void ReadGPSDataBinary(const std::string& path, FeatureDataPtrUmap* feature_data_ump);
void WriteGPSDataBinary(const std::string& path, const FeatureDataPtrUmap& feature_data_ump);


void ReadPieceIndicesDataText(const std::string& path, 
                               FeatureDataPtrUmap* feature_data_ump);
void WritePieceIndicesDataText(const std::string& path, 
                                const FeatureDataPtrUmap& feature_data_ump); 

void ReadPieceIndicesDataBinary(const std::string& path, 
                               FeatureDataPtrUmap* feature_data_ump);
void WritePieceIndicesDataBinary(const std::string& path, 
                                const FeatureDataPtrUmap& feature_data_ump); 

} // namespace sensemap

#endif //SENSEMAP_CONTAINER_IO_H
