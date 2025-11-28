//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "reconstruction_manager.h"
#include "util/misc.h"


namespace sensemap {

ReconstructionManager::ReconstructionManager() {}

ReconstructionManager::ReconstructionManager(ReconstructionManager&& other)
    : ReconstructionManager() {
    reconstructions_ = std::move(other.reconstructions_);
}

ReconstructionManager& ReconstructionManager::operator = (ReconstructionManager&& other) {
    if (this != &other) {
        reconstructions_ = std::move(other.reconstructions_);
    }
    return *this;
}

size_t ReconstructionManager::Size() const { return reconstructions_.size(); }

const std::shared_ptr<Reconstruction> ReconstructionManager::Get(const size_t idx) const {
    return reconstructions_.at(idx);
}

std::shared_ptr<Reconstruction> ReconstructionManager::Get(const size_t idx) {
    return reconstructions_.at(idx);
}

size_t ReconstructionManager::Add() {
    const size_t idx = Size();
    reconstructions_[idx] = std::make_shared<Reconstruction>();
    return idx;
}

bool ReconstructionManager::Add(size_t idx) {
    if (reconstructions_.count(idx)) {
        return false;
    } else {
        reconstructions_[idx] = std::make_shared<Reconstruction>();
        return true;
    }

    return false;
}

std::vector<int> ReconstructionManager::getReconstructionIds() {
    std::vector<int> ids;
    for(auto reconstruction: reconstructions_){
        ids.push_back(reconstruction.first);
    }
    std::sort(ids.begin(),ids.end());
    return ids;
}

void ReconstructionManager::Delete(const size_t idx) {
    CHECK(reconstructions_.count(idx));
    reconstructions_.erase(idx);
}

void ReconstructionManager::Clear() {
    reconstructions_.clear();
}

int ReconstructionManager::Read(const std::string &path) {
    if (!ExistsDir(path)) {
        return -1;
    }

    reconstructions_.clear();
    size_t reconstruction_idx = 0;
    while (true) {
	    auto reconstruction_path = JoinPaths(path, std::to_string(reconstruction_idx));
	    if (!ExistsDir(reconstruction_path)) {
            break;
	    }
        reconstructions_[reconstruction_idx] = std::make_shared<Reconstruction>();
        std::shared_ptr<Reconstruction> reconstruction = reconstructions_[reconstruction_idx];
        reconstruction->ReadReconstruction(reconstruction_path);
        reconstruction_idx++;
	}
	return reconstruction_idx;
}

void ReconstructionManager::Write(const std::string &path, bool write_binary) {
    CreateDirIfNotExists(path);
    for (auto reconstruction : reconstructions_) {
        const std::string reconstruction_path = JoinPaths(path, std::to_string(reconstruction.first));
        CreateDirIfNotExists(reconstruction_path);
        reconstructions_[reconstruction.first]->WriteReconstruction(reconstruction_path, write_binary);
    }
}

}
