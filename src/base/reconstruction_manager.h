//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_RECONSTRUCTION_MANAGER_H_
#define SENSEMAP_BASE_RECONSTRUCTION_MANAGER_H_

#include <vector>
#include <memory>

#include "util/types.h"
#include "reconstruction.h"

namespace sensemap {

class ReconstructionManager {
public:
    ReconstructionManager();

    // Move constructor and assignment, rvalue reference
    ReconstructionManager(ReconstructionManager&& other);
    ReconstructionManager& operator = (ReconstructionManager&& other);

    // The number of reconstructions managed.
    size_t Size() const;

    // Get a reference to a specific reconstruction.
    const std::shared_ptr<Reconstruction> Get(const size_t idx) const;
    std::shared_ptr<Reconstruction> Get(const size_t idx);

    // Add a new empty reconstruction and return its index.
    size_t Add();
    bool Add(size_t idx);

    // Delete a specific reconstruction.
    void Delete(const size_t idx);

    // Delete all reconstruction.
    void Clear();

    // Read reconstructions from a workspace path
    int Read(const std::string &path);

    // Write reconstructions to a workspace path
    void Write(const std::string &path, bool write_binary = true);

    std::vector<int> getReconstructionIds();

private:
    // // disable destructor
    // ReconstructionManager(ReconstructionManager& other) = delete;

    std::unordered_map<int, std::shared_ptr<Reconstruction> > reconstructions_;
};

}

#endif