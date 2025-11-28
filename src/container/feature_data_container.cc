//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "feature_data_container.h"

#include "io.h"

namespace sensemap {

const std::string &FeatureDataContainer::GetImagePath() const {
	return image_path_;
}

Camera &FeatureDataContainer::GetCamera(const camera_t camera_id) {
	return *cameras_data_.at(camera_id);
}

const Camera &FeatureDataContainer::GetCamera(const camera_t camera_id) const {
	return *cameras_data_.at(camera_id);
}

const Image &FeatureDataContainer::GetImage(const image_t image_id) const {
	return feature_data_.at(image_id)->image;
}

const Bitmap& FeatureDataContainer::GetBitmap(const image_t image_id) const {
	return feature_data_.at(image_id)->bitmap[0];
}

const FeatureKeypoints & FeatureDataContainer::GetKeypoints(
		const image_t image_id) {
	return feature_data_.at(image_id)->keypoints;
}

const FeatureDescriptors & FeatureDataContainer::GetDescriptors(
		const image_t image_id) const {
	return feature_data_.at(image_id)->descriptors;
}

FeatureDescriptors & FeatureDataContainer::GetDescriptors(
		const image_t image_id) {
	return feature_data_.at(image_id)->descriptors;
}

const CompressedFeatureDescriptors & FeatureDataContainer::GetCompressedDescriptors(
		const image_t image_id) const {
	return feature_data_.at(image_id)->compressed_descriptors;
}

CompressedFeatureDescriptors & FeatureDataContainer::GetCompressedDescriptors(
		const image_t image_id){
	return feature_data_.at(image_id)->compressed_descriptors;
}

const BinaryFeatureDescriptors & FeatureDataContainer::GetBinaryDescriptors(
		const image_t image_id) {
	return feature_data_.at(image_id)->binary_descriptors;
}


const PanoramaIndexs & FeatureDataContainer::GetPanoramaIndexs(
		const image_t image_id) {
	return feature_data_.at(image_id)->panoramaidxs;
}

const PieceIndexs & FeatureDataContainer::GetPieceIndexs(
		const image_t image_id) {
	return feature_data_.at(image_id)->pieceidxs;
}

const AprilTagDetections& FeatureDataContainer::GetAprilTagDetections(
        const image_t image_id){
    return feature_data_.at(image_id)->detections;
}

std::vector<image_t> FeatureDataContainer::GetImageIds() const {
    std::vector<image_t> image_ids;
    image_ids.reserve(feature_data_.size());
    for (const auto& image_data : feature_data_) {
        image_ids.push_back(image_data.first);
    }
    return image_ids;
}

std::vector<image_t> FeatureDataContainer::GetNewImageIds() const {
    std::vector<image_t> image_ids;
    image_ids.reserve(feature_data_.size());
    for (const auto& image_data : feature_data_) {
        if(image_data.second->image.LabelId() != 0 && 
           image_data.second->image.LabelId() != kInvalidLabelId){
            image_ids.push_back(image_data.first);
        }
    }
    image_ids.shrink_to_fit();
    return image_ids;
}

std::unordered_set<std::string> FeatureDataContainer::GetImageNames() const {
    std::unordered_set<std::string> image_names;
    for (const auto& image_data : feature_data_) {
        image_names.insert(image_data.second->image.Name());
    }
    return image_names;
}

std::vector<image_t> FeatureDataContainer::GetOldImageIds() const {
    std::vector<image_t> image_ids;
    image_ids.reserve(feature_data_.size());
    for (const auto& image_data : feature_data_) {
        if(image_data.second->image.LabelId() == 0){
            image_ids.push_back(image_data.first);
        }
    }
    image_ids.shrink_to_fit();
    return image_ids;
}

std::vector<image_t> FeatureDataContainer::GetOldPoseImageIds() const {
    std::vector<image_t> image_ids;
    image_ids.reserve(feature_data_.size());
    for (const auto& image_data : feature_data_) {
        if(image_data.second->image.LabelId() == 0 && image_data.second->image.HasPose()){
            image_ids.push_back(image_data.first);
        }
    }
    image_ids.shrink_to_fit();
    return image_ids;
}

std::vector<image_t> FeatureDataContainer::GetNewPoseImageIds() const {
    std::vector<image_t> image_ids;
    image_ids.reserve(feature_data_.size());
    for (const auto& image_data : feature_data_) {
        if(image_data.second->image.LabelId() == 0 && image_data.second->image.HasPose()){
            image_ids.push_back(image_data.first);
        }
    }

    for (const auto& image_data : feature_data_) {
        if(image_data.second->image.LabelId() != 0){
            image_ids.push_back(image_data.first);
        }
    }

    image_ids.shrink_to_fit();

    return image_ids;
}

FeatureDataPtrUmap FeatureDataContainer::GetFeatureData() const {
    return feature_data_;
}

std::unordered_set<image_t> FeatureDataContainer::GetSequentialNeighbor(const image_t image_id, const label_t label_id,
                                                                        int scope) {
    std::unordered_set<image_t> neighbor_images;

    for (int i = -scope; i <= scope; ++i) {
        if (static_cast<int>(image_id) + i <= 0) {
            continue;
        }

        if (feature_data_.find(image_id + i) == feature_data_.end()) {
            continue;
        }

        const Image& neighbor_image = feature_data_.at(image_id)->image;

        if (neighbor_image.LabelId() == label_id) {
            neighbor_images.insert(image_id + i);
        }
    }

    return neighbor_images;
}

camera_t FeatureDataContainer::NumCamera() const {
    return cameras_data_.size();
}

image_t FeatureDataContainer::GetGeoImageIndex() const {
    return geo_image_index_;
}

bool FeatureDataContainer::ExistAprilTagDetection() const { 
    for (const auto& cur_feature : feature_data_){
        if(cur_feature.second->detections.size()> 0){
            return true;
        }
    }
    return false;
}

bool FeatureDataContainer::ExistImage(std::string image_name){
    return feature_name_.count(image_name);
}

bool FeatureDataContainer::ExistImage(image_t image_id){
    return feature_data_.count(image_id);
}

Image& FeatureDataContainer::GetImage(const std::string image_name){
    CHECK(feature_name_.count(image_name)) << "Do not contain this image";
    auto image_id = feature_name_[image_name];
    return feature_data_.at(image_id)->image;
}

Image& FeatureDataContainer::GetImage(const image_t image_id){
    CHECK(feature_data_.count(image_id)) << "Do not contain this image";
    return feature_data_.at(image_id)->image;
}

image_t FeatureDataContainer::GetImageId(const std::string image_name){
    CHECK(feature_name_.count(image_name)) << "Do not contain this image";
    return feature_name_[image_name];
}


VladVisualIndex::VLAD& FeatureDataContainer::GetVladVector(const image_t image_id){
    return feature_data_.at(image_id)->vlad_vector;
}

const VladVisualIndex::VLAD& FeatureDataContainer::GetVladVector(const image_t image_id) const{
    return feature_data_.at(image_id)->vlad_vector;
}


void FeatureDataContainer::emplace(image_t image_id, FeatureDataPtr image_data) {
	feature_data_.emplace(image_id, image_data);
}

void FeatureDataContainer::emplace(camera_t camera_id,
                                   std::shared_ptr<Camera> camera_data) {
	cameras_data_.emplace(camera_id, camera_data);
}

void FeatureDataContainer::emplace(std::string image_name, image_t image_id) {
	feature_name_.emplace(image_name, image_id);
}

void FeatureDataContainer::SetImagePath(const std::string &image_path) {
	image_path_ = image_path;
}

void FeatureDataContainer::SetGeoImageIndex(const image_t image_index) {
    geo_image_index_ = image_index;
}

void FeatureDataContainer::ReadImagesData(const std::string &path) {
	std::cout << "FeatureDataContainer::ReadImagesData" << std::endl;
	ReadFeatureDataText(path, &image_path_, &feature_data_, &feature_name_);
}

void FeatureDataContainer::ReadImagesBinaryData(const std::string &path) {
	std::cout << "FeatureDataContainer::ReadImagesData" << std::endl;
	ReadFeatureDataBinary(path, &image_path_, &feature_data_, &feature_name_);
}

void FeatureDataContainer::ReadImagesBinaryDataWithoutDescriptor(const std::string &path) {
	std::cout << "FeatureDataContainer::ReadImagesDataWithoutDescriptor" << std::endl;
	ReadFeatureBinaryDataWithoutDescriptor(path, &image_path_, &feature_data_, &feature_name_);
}

void FeatureDataContainer::ReadCameras(const std::string &path, 
                                       bool camera_rig) {
	std::cout << "FeatureDataContainer::ReadCameras" << std::endl;
	ReadCamerasText(path, &cameras_data_, camera_rig);
}

void FeatureDataContainer::ReadCamerasBinaryData(const std::string &path, 
                                                 bool camera_rig) {
	std::cout << "FeatureDataContainer::ReadCameras" << std::endl;
	ReadCamerasBinary(path, &cameras_data_, camera_rig);
}

void FeatureDataContainer::ReadLocalCameras(const std::string& path) {
    std::cout << "FeatureDataContainer::ReadLocalCameras" << std::endl;
	ReadLocalCamerasText(path, &cameras_data_);
}

void FeatureDataContainer::ReadLocalCamerasBinaryData(const std::string& path) {
    std::cout << "FeatureDataContainer::ReadLocalCameras" << std::endl;
	ReadLocalCamerasBinary(path, &cameras_data_);
}

void FeatureDataContainer::ReadLocalImagesBinaryData(const std::string& path) {
    std::cout << "FeatureDataContainer::ReadLocalImages" << std::endl;
    ReadLocalImagesBinary(path, &feature_data_);
}

void FeatureDataContainer::ReadAprilTagData(const std::string &path) {
    std::cout << "FeatureDataContainer::ReadAprilTagData" << std::endl;
    ReadAprilTagDataText(path, &feature_data_);
}

void FeatureDataContainer::ReadAprilTagBinaryData(const std::string &path) {
    std::cout << "FeatureDataContainer::ReadAprilTagData" << std::endl;
    ReadAprilTagDataBinary(path, &feature_data_);
}

void FeatureDataContainer::ReadSubPanoramaBinaryData(const std::string &path) {
	std::cout << "FeatureDataContainer::ReadPanoramaData" << std::endl;
	ReadSubPanoramaDataBinary(path, &feature_data_);
}

void FeatureDataContainer::ReadSubPanoramaData(const std::string &path) {
	std::cout << "FeatureDataContainer::ReadPanoramaData" << std::endl;
	ReadSubPanoramaDataText(path, &feature_data_);
}

void FeatureDataContainer::ReadPieceIndicesData(const std::string &path) {
	std::cout << "FeatureDataContainer::ReadPieceIndicesData" << std::endl;
	ReadPieceIndicesDataText(path, &feature_data_);
}

void FeatureDataContainer::ReadPieceIndicesBinaryData(const std::string &path) {
	std::cout << "FeatureDataContainer::ReadPieceIndicesData" << std::endl;
	ReadPieceIndicesDataBinary(path, &feature_data_);
}
void FeatureDataContainer::ReadGlobalFeaturesBinaryData(const std::string& path){
    std::cout << "FeatureDataContainer::ReadGlobalFeaturesData" << std::endl;
    ReadGlobalFeaturesDataBinary(path, &feature_data_);
}

void FeatureDataContainer::ReadBinaryDescriptors(const std::string& path){
    std::cout << "FeatureDataContainer::ReadBinaryDescriptors" << std::endl;
    ReadBinaryDescriptorsIO(path,&feature_data_);
}


void FeatureDataContainer::ReadGPSBinaryData(const std::string& path) {
    std::cout << "FeatureDataContainer::ReadGPSDataBinary" << std::endl;
	ReadGPSDataBinary(path, &feature_data_);
}

void FeatureDataContainer::WriteImagesData(const std::string &path) const {
	std::cout << "FeatureDataContainer::WriteImagesData" << std::endl;
	WriteFeatureDataText(path, image_path_, feature_data_);
}

void FeatureDataContainer::WriteImagesBinaryData(const std::string &path, bool write_descriptors) const {
	std::cout << "FeatureDataContainer::WriteImagesData" << std::endl;
	WriteFeatureDataBinary(path, image_path_, feature_data_, write_descriptors);
}

void FeatureDataContainer::WriteCameras(const std::string &path) const {
	std::cout << "FeatureDataContainer::WriteCameras" << std::endl;
	WriteCamerasText(path, cameras_data_);
}

void FeatureDataContainer::WriteCamerasBinaryData(const std::string &path) const {
	std::cout << "FeatureDataContainer::WriteCameras" << std::endl;
	WriteCamerasBinary(path, cameras_data_);
}

void FeatureDataContainer::WriteLocalCameras(const std::string& path) const {
	std::cout << "FeatureDataContainer::WriteLocalCameras" << std::endl;
    WriteLocalCamerasText(path, cameras_data_);    
}

void FeatureDataContainer::WriteLocalCamerasBinaryData(const std::string& path) const {
	std::cout << "FeatureDataContainer::WriteLocalCameras" << std::endl;
    WriteLocalCamerasBinary(path, cameras_data_);
}

void FeatureDataContainer::WriteLocalImagesBinaryData(const std::string& path) const {
    std::cout << "FeatureDataContainer::WriteLocalImages" << std::endl;
    WriteLocalImagesBinary(path, feature_data_);
}

void FeatureDataContainer::WriteSubPanoramaData(const std::string &path) const {
	std::cout << "FeatureDataContainer::WritePanoramaData" << std::endl;
	WriteSubPanoramaDataText(path, feature_data_);
}

void FeatureDataContainer::WriteSubPanoramaBinaryData(const std::string &path) const {
	std::cout << "FeatureDataContainer::WritePanoramaData" << std::endl;
	WriteSubPanoramaDataBinary(path, feature_data_);
}

void FeatureDataContainer::WriteAprilTagData(const std::string& path) const {
    std::cout << "FeatureDataContainer::WriteAprilTagData" << std::endl;
    WriteAprilTagDataText(path, feature_data_);
}
void FeatureDataContainer::WriteAprilTagBinaryData(const std::string& path) const{
    std::cout << "FeatureDataContainer::WriteAprilTagData" << std::endl;
    WriteAprilTagDataBinary(path, feature_data_);
}

void FeatureDataContainer::WriteGPSBinaryData(const std::string& path) const {
    std::cout << "FeatureDataContainer::WriteGPSBinaryData" << std::endl;
    WriteGPSDataBinary(path, feature_data_);
}

void FeatureDataContainer::WritePieceIndicesData(const std::string& path) const{
    std::cout << "FeatureDataContainer::WritePieceIndicesData" << std::endl;
    WritePieceIndicesDataText(path, feature_data_);
}

void FeatureDataContainer::WritePieceIndicesBinaryData(const std::string& path) const{
    std::cout << "FeatureDataContainer::WritePieceIndicesData" << std::endl;
    WritePieceIndicesDataBinary(path, feature_data_);
}


void FeatureDataContainer::WriteGlobalFeaturesBinaryData(const std::string& path) const{
    std::cout << "FeatureDataContainer::WriteGlobalFeaturesBinaryData"<<std::endl;
    WriteGlobalFeaturesDataBinary(path, feature_data_);
}

void FeatureDataContainer::WriteBinaryDescriptors(const std::string& path) const {
    std::cout << "FeatureDataContainer::WriteBinaryDescriptors" << std::endl;
    WriteBinaryDescriptorsIO(path, feature_data_);
}


void FeatureDataContainer::DeleteImage(image_t image_id){
    if(feature_data_.find(image_id)!= feature_data_.end()){
        Image& image = feature_data_.at(image_id)->image;
        std::string name = image.Name();
        
        feature_data_.erase(image_id);

        CHECK(feature_name_.find(name)!=feature_name_.end());
        feature_name_.erase(name);
    }
}

} // namespace sensemap

