//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CONTAINER_FEATURE_DATA_CONTAINER_H_
#define SENSEMAP_CONTAINER_FEATURE_DATA_CONTAINER_H_

#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "feature/types.h"
#include "util/types.h"
#include "util/mat.h"
#include "util/bitmap.h"
#include "base/image.h"
#include "base/camera.h"
#include "retrieval/vlad_visual_index.h"


namespace sensemap {

struct FeatureData {
	Image image;
	std::vector<Bitmap> bitmap;
	std::vector<Bitmap> mask;
	std::vector<std::string> bitmap_paths;
	FeatureKeypoints keypoints;
	FeatureDescriptors descriptors;
	CompressedFeatureDescriptors compressed_descriptors;
	BinaryFeatureDescriptors binary_descriptors;
	PanoramaIndexs panoramaidxs;
	PieceIndexs pieceidxs;
    AprilTagDetections detections;
	VladVisualIndex::VLAD vlad_vector;
};
typedef std::shared_ptr<FeatureData> FeatureDataPtr;
typedef EIGEN_STL_UMAP(image_t, FeatureDataPtr) FeatureDataPtrUmap;
typedef EIGEN_STL_UMAP(std::string, image_t) FeatureNameUmap;
typedef EIGEN_STL_UMAP(camera_t, std::shared_ptr<Camera>) CameraPtrUmap;
typedef EIGEN_STL_UMAP(label_t, std::vector<image_t>) LabelledImagePtrUmap;

//This class holds the data required for feature operation, i.e. extraction and
//matching.
class FeatureDataContainer {

public:

	FeatureDataContainer() {};

	explicit FeatureDataContainer(std::string& image_path) :
			image_path_(image_path) {};

	FeatureDataContainer(std::string& image_path,
	                    FeatureDataPtrUmap feature_data,
	                    CameraPtrUmap cameras_data) :
		 image_path_(image_path),
		 feature_data_(std::move(feature_data)),
		 cameras_data_(std::move(cameras_data)) {};

	const std::string& GetImagePath() const;
	Camera& GetCamera(const camera_t camera_id);
	const Camera& GetCamera(const camera_t camera_id) const;
	const Image& GetImage(const image_t image_id) const;

	// This function can only be used for monocular image
	const Bitmap& GetBitmap(const image_t image_id) const;
	const FeatureKeypoints& GetKeypoints(const image_t image_id);
	const FeatureDescriptors& GetDescriptors(const image_t image_id) const;
	FeatureDescriptors& GetDescriptors(const image_t image_id);
	const CompressedFeatureDescriptors& GetCompressedDescriptors(const image_t image_id) const;
	CompressedFeatureDescriptors& GetCompressedDescriptors(const image_t image_id);
	
	const BinaryFeatureDescriptors& GetBinaryDescriptors(const image_t image_id);

	const PanoramaIndexs& GetPanoramaIndexs(const image_t image_id);
	const PieceIndexs& GetPieceIndexs(const image_t image_id);
	const AprilTagDetections& GetAprilTagDetections(const image_t image_id);
	std::vector<image_t> GetImageIds() const;
    std::vector<image_t> GetNewImageIds() const;
    std::vector<image_t> GetOldImageIds() const;
    std::vector<image_t> GetOldPoseImageIds() const;
    std::vector<image_t> GetNewPoseImageIds() const;
    std::unordered_set<std::string> GetImageNames() const;
	FeatureDataPtrUmap GetFeatureData() const;

	std::unordered_set<image_t> GetSequentialNeighbor(const image_t image_id, const label_t label_id, int scope);

    camera_t NumCamera() const;
	image_t GetGeoImageIndex() const;

	bool ExistAprilTagDetection() const;

    bool ExistImage(std::string image_name);
    bool ExistImage(image_t image_id);
    Image& GetImage(const std::string image_name);
    Image& GetImage(const image_t image_id);
    image_t GetImageId(const std::string image_name);

	VladVisualIndex::VLAD& GetVladVector(const image_t image_id);
	const VladVisualIndex::VLAD& GetVladVector(const image_t image_id) const;

	void emplace(image_t image_id, FeatureDataPtr image_data);
	void emplace(camera_t camera_id, std::shared_ptr<Camera> camera_data);
	void emplace(std::string image_name, image_t image_id);
	void SetImagePath(const std::string& image_path);
	void SetGeoImageIndex(const image_t image_index);

	void ReadImagesData(const std::string& path);
	void ReadImagesBinaryData(const std::string& path);
	void ReadImagesBinaryDataWithoutDescriptor(const std::string& path); // -- For triangulator
	void ReadSubPanoramaData(const std::string& path);
	void ReadSubPanoramaBinaryData(const std::string& path);
	void ReadCameras(const std::string& path, bool camera_rig = false);
	void ReadCamerasBinaryData(const std::string& path, bool camera_rig = false);
	void ReadLocalCameras(const std::string& path);
	void ReadLocalCamerasBinaryData(const std::string& path);
	void ReadLocalImagesBinaryData(const std::string& path);
	void ReadPieceIndicesData(const std::string& path);
	void ReadPieceIndicesBinaryData(const std::string& path);
	void ReadGlobalFeaturesBinaryData(const std::string& path);
	void ReadBinaryDescriptors(const std::string& path);

    void ReadAprilTagData(const std::string& path);
    void ReadAprilTagBinaryData(const std::string& path);
	void ReadGPSBinaryData(const std::string& path);

	void WriteImagesData(const std::string& path) const;
	void WriteImagesBinaryData(const std::string& path, bool write_descriptors = true) const;
	void WriteSubPanoramaData(const std::string& path) const;
	void WriteSubPanoramaBinaryData(const std::string& path) const;
	void WriteCameras(const std::string& path) const;
	void WriteCamerasBinaryData(const std::string& path) const;
	void WriteLocalCameras(const std::string& path) const;
	void WriteLocalCamerasBinaryData(const std::string& path) const;
	void WriteLocalImagesBinaryData(const std::string& path) const;
	void WriteAprilTagData(const std::string& path) const;
	void WriteAprilTagBinaryData(const std::string& path) const;
	void WriteGPSBinaryData(const std::string& path) const;
	void WritePieceIndicesData(const std::string& path) const;
	void WritePieceIndicesBinaryData(const std::string& path) const;
	void WriteGlobalFeaturesBinaryData(const std::string& path) const;
	void WriteBinaryDescriptors(const std::string& path) const;

	// Update Mode.
	void DeleteImage(image_t image_id);

private:

	std::string image_path_;
	FeatureDataPtrUmap feature_data_;
	CameraPtrUmap cameras_data_;
    FeatureNameUmap feature_name_;
	image_t geo_image_index_;
};

} // namespace sensemap

#endif //SENSEMAP_CONTAINER_FEATURE_DATA_CONTAINER_H

