//Copyright (c) 2022, SenseTime Group.
//All rights reserved.

#include "exception_handler.h"

#include "misc.h"
#include "proc.h"
#include "base/version.h"

namespace sensemap {

std::mutex ExceptionHandler::g_exception_mutex_;

void ExceptionHandler::Dump() {
    std::unique_lock<std::mutex> lock(g_exception_mutex_);
    {
        if (!boost::filesystem::exists(filepath_)) {
            boost::filesystem::create_directories(filepath_);
        }
        auto filename = StringPrintf("%s/%s-%d.txt", filepath_.c_str(), taskname_.c_str(), GetProcessId());
        if (boost::filesystem::exists(filename)) {
            return;
        }
        std::cout << filename << std::endl << std::flush;
        std::ofstream file(filename, std::ofstream::out);
        file << "code: "<< std::hex << error_code_ << std::endl;
        file << "msg: " << msg_map_.at(error_code_) << std::endl;
        file << "version: " << __VERSION__ << std::endl;
        file.close();
    }
}

}
