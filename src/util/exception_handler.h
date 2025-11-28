//Copyright (c) 2022, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_EXCEPTION_HANDLER_H
#define SENSEMAP_UTIL_EXCEPTION_HANDLER_H

#include <string>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace sensemap {
enum StateCode {
    SUCCESS                       = 0x0000,
    NO_MATCHING_INPUT_PARAM       = 0x0001,
    INVALID_INPUT_PARAM           = 0x0002,
    ENCYPT_CHECK_FAILED           = 0x0004,
    VARIABLE_CONFLICT             = 0x0005,
    IMAGE_LOAD_FAILED             = 0x0006,
    COLLAPSED_IMAGE_DIMENSION     = 0x0007,
    INVALID_IMAGE_DIMENSION       = 0x0008,
    POINT_CLOUD_IS_EMPTY          = 0x0009,
    INVALID_INPUT_FORMAT          = 0x000a,
    INVALID_PMVS_FORMAT           = 0x000b,
    MODEL_FILE_IS_NOT_EXIST       = 0x000c,
    MODEL_MERGE_FAILED            = 0x000d,
    RGBD_CALIB_FAILED             = 0x000e,
    RGBD_ALIGNMENT_FAILED         = 0x000f,
    CUDA_ERROR                    = 0x0010,
    LIMITED_GPU_MEMORY            = 0x0011,
    LIMITED_CPU_MEMORY            = 0x0012
};

class ExceptionHandler {
public:
    ExceptionHandler(const StateCode error_code, const std::string& filepath, const std::string& taskname)
    : error_code_(error_code),
      filepath_(filepath),
      taskname_(taskname) {
        msg_map_[(int)SUCCESS]                   = "success";
        msg_map_[(int)NO_MATCHING_INPUT_PARAM]   = "No matched input parameters!";
        msg_map_[(int)INVALID_INPUT_PARAM]       = "Invalid input parameters!";
        msg_map_[(int)ENCYPT_CHECK_FAILED]       = "Encypt check failed!";
        msg_map_[(int)VARIABLE_CONFLICT]         = "Variable conflict!";
        msg_map_[(int)IMAGE_LOAD_FAILED]         = "Image load failed!";
        msg_map_[(int)COLLAPSED_IMAGE_DIMENSION] = "Collapsed image dimension!";
        msg_map_[(int)INVALID_IMAGE_DIMENSION]   = "Invalid image dimension!";
        msg_map_[(int)POINT_CLOUD_IS_EMPTY]      = "The point cloud file is empty!";
        msg_map_[(int)INVALID_INPUT_FORMAT]      = "Invalid input model type!";
        msg_map_[(int)INVALID_PMVS_FORMAT]       = "Invalid PMVS format!";
        msg_map_[(int)MODEL_FILE_IS_NOT_EXIST]   = "Model files are not exists!";
        msg_map_[(int)MODEL_MERGE_FAILED]        = "Failed to merge reconstructions!";
        msg_map_[(int)RGBD_CALIB_FAILED]         = "Failed to calibrate RGBD cameras!";
        msg_map_[(int)RGBD_ALIGNMENT_FAILED]     = "Failed to reconstruct RGBD cameras!";
        msg_map_[(int)CUDA_ERROR]                = "Cuda error!";
        msg_map_[(int)LIMITED_GPU_MEMORY]        = "Limited GPU Memory!";
        msg_map_[(int)LIMITED_CPU_MEMORY]        = "Limited CPU Memory!";
    }
    
    void Dump();

private:
    static std::mutex g_exception_mutex_;
    StateCode error_code_;
    std::string filepath_;
    std::string taskname_;
    std::unordered_map<int, std::string> msg_map_;
};

}

#endif
