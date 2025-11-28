#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <util/rgbd_helper.h>
#include <util/threading.h>
#include <util/imageconvert.h>
#include <util/misc.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Geometry> 
#include <iomanip>
#include <base/camera_models.h>
#include <base/undistortion.h>
#include <util/TinyEXIF.h>
#include "base/version.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

int main(int argc, char* argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: sfm-bin-packer-hybrid-")+__VERSION__);

    std::string input_color_path, input_depth_path, configuration_file_path;
    int index;
    int depth_search_range = 0;
    if (argc == 5) {
        input_color_path = argv[1];
        input_depth_path = argv[2];
        configuration_file_path = argv[3];
        index = std::atoi(argv[4]);
    } else if (argc == 6) {
        input_color_path = argv[1];
        input_depth_path = argv[2];
        configuration_file_path = argv[3];
        index = std::atoi(argv[4]);
        depth_search_range = std::atoi(argv[5]);
    } else {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "    <COLOR_PATH> <DEPTH_PATH> <SFM_YAML> <CAMERA_INDEX> [DEPTH_SEARCH_RANGE=" << depth_search_range << "]" << std::endl;
        std::cout << std::endl;
        return 1;
    }

    Configurator param;
    param.Load(configuration_file_path.c_str());
    std::string image_path = param.GetArgument("image_path", "");
    std::string camera_param_file = param.GetArgument("camera_param_file", "");
    CHECK(!image_path.empty());
    CHECK(!camera_param_file.empty());

    // cv::FileStorage fs;
    // CHECK(boost::filesystem::exists(camera_param_file));
    // CHECK(fs.open(camera_param_file, cv::FileStorage::READ));
    YAML::Node node = YAML::LoadFile(camera_param_file);
    CHECK(!node.IsNull());
    std::cout << "Camera " << index << std::endl;

    std::string sub_path;
    std::string bin_path;
    {
        // fs["sub_path_" + std::to_string(index)] >> sub_path;
        sub_path = node["sub_path_" + std::to_string(index)].as<std::string>();
        bin_path = (boost::filesystem::path(image_path) / boost::filesystem::path(sub_path)).string();
    }
    std::cout << "Subpath: " << sub_path << std::endl;
    std::cout << "Writing to: " << bin_path << std::endl << std::endl;
    CHECK(!sub_path.empty());

    Eigen::Matrix4d RT;
    {
        // cv::Mat RT_mat;
        // if(!fs["RT_" + std::to_string(index)].empty())
        YAML::Node cv_mat_node = node["RT_" + std::to_string(index)];
        if (cv_mat_node.IsDefined())
        {
            std::vector<double> mat_data = cv_mat_node["data"].as<std::vector<double> >();
            int mat_rows = cv_mat_node["rows"].as<int>();
            int mat_cols = cv_mat_node["cols"].as<int>();
            CHECK_EQ(mat_data.size(), mat_rows * mat_cols);
            cv::Mat1d RT_mat(mat_rows, mat_cols, mat_data.data());
            cv::cv2eigen(RT_mat, RT);
        }
        else
        {
            RT = Eigen::Matrix4d::Identity();
        }
    }
    std::cout << "Input RT: " << std::endl;
    std::cout << RT << std::endl << std::endl;

    std::string image_intrinsic_string;
    std::string depth_intrinsic_string;
    {
        std::string cam_model;
        std::string cam_params;
        // fs["camera_model_" + std::to_string(index)] >> cam_model;
        // fs["camera_params_" + std::to_string(index)] >> cam_params;
        cam_model = node["camera_model_" + std::to_string(index)].as<std::string>();
        cam_params = node["camera_params_" + std::to_string(index)].as<std::string>();
        image_intrinsic_string = cam_model + ", " + cam_params;

        // fs["rgbd_params_" + std::to_string(index)] >> depth_intrinsic_string;
        depth_intrinsic_string = node["rgbd_params_" + std::to_string(index)].as<std::string>();
    }
    CHECK(!image_intrinsic_string.empty());
    CHECK(!depth_intrinsic_string.empty());

    if (boost::filesystem::exists(bin_path)) {
        boost::filesystem::remove_all(bin_path);
    }
    boost::filesystem::create_directories(bin_path);

    std::vector<boost::filesystem::path> image_files;
    std::map<std::string, boost::filesystem::path> depth_files;
    boost::filesystem::recursive_directory_iterator end_iter;

    boost::filesystem::path image_full_path(input_color_path);
    for (boost::filesystem::recursive_directory_iterator iter(image_full_path); iter != end_iter; iter++) {
        if (boost::filesystem::is_regular_file(*iter)) {
            boost::filesystem::path filepath(iter->path().string());
            image_files.emplace_back(std::move(filepath));
        }
    }

    boost::filesystem::path depth_full_path(input_depth_path);
    for (boost::filesystem::recursive_directory_iterator iter(depth_full_path); iter != end_iter; iter++) {
        if (boost::filesystem::is_regular_file(*iter)) {
            boost::filesystem::path filepath(iter->path().string());
            depth_files[filepath.stem().string()] = filepath;
        }
    }

    Camera image_camera;
    CHECK(IntrinsicStringToCamera(image_intrinsic_string, image_camera)) << "Invalid intrinsic for color camera";
    std::cout << "Image Camera: " << image_camera.ModelName() << std::endl;
    std::cout << image_camera.ParamsToString() << std::endl << std::endl;
    
    Camera depth_camera;
    CHECK(IntrinsicStringToCamera(depth_intrinsic_string, depth_camera)) << "Invalid intrinsic for depth camera";
    std::cout << "Depth Camera: " << depth_camera.ModelName() << std::endl;
    std::cout << depth_camera.ParamsToString() << std::endl << std::endl;

    const int num_eff_threads = sensemap::GetEffectiveNumThreads(-1);
    sensemap::ThreadPool thread_pool(num_eff_threads);
    for (size_t index = 0; index < image_files.size(); ++index)
    thread_pool.AddTask([&](size_t i) {
        auto & image_file = image_files[i];
        std::string stem = image_file.stem().string();
        auto find = depth_files.find(stem);
        if (find == depth_files.end()) {
            find = depth_files.find(stem + image_file.extension().string());
        }
        for (int j = 1; j <= depth_search_range && find == depth_files.end(); j++) {
            int64_t stem_value = std::atoll(stem.c_str());
            if (find == depth_files.end()) {
                find = depth_files.find(std::to_string(stem_value + j));
            }
            if (find == depth_files.end()) {
                find = depth_files.find(std::to_string(stem_value - j));
            }
            if (find == depth_files.end()) {
                find = depth_files.find(std::to_string(stem_value + j) + image_file.extension().string());
            }
            if (find == depth_files.end()) {
                find = depth_files.find(std::to_string(stem_value - j) + image_file.extension().string());
            }
        }

        cv::Mat depth;
        if (find != depth_files.end()) {
            auto & depth_file = find->second;
            depth = cv::imread(depth_file.string(), cv::IMREAD_ANYDEPTH);
        } else if (!depth_files.empty()) {
            depth = cv::imread(depth_files.begin()->second.string(), cv::IMREAD_ANYDEPTH);
            depth.setTo(0);
            std::cout << "Warning: depth file of " << image_file.string() << " not found! " << std::endl;
        }
        if (depth.empty()) return;

        if (depth.type() != CV_32FC1) {
            depth.convertTo(depth, CV_32FC1, 0.001f);
        }
        MatXf depthmap(depth.cols, depth.rows, 1);
        for (int y = 0; y < depth.rows; y++) {
            for (int x = 0; x < depth.cols; x++) {
                float d = depth.at<float>(y, x);
                if (d > MAX_VALID_DEPTH_IN_M) d = 0;
                depthmap.Set(y, x, d);
            }
        }

        RGBDWriteData w_data;
        w_data.color_camera = image_camera;
        w_data.depth = depthmap;
        w_data.depth_camera = depth_camera;
        w_data.depth_RT = RT;
        if (image_file.extension().string() == ".txt") {
            // Sony .txt raw image
            FILE * fp = fopen(image_file.string().c_str(), "rb");
            if (!fp) return;

            int width, height;
            double timestamp;
            fread(&width, sizeof(int), 1, fp);
            fread(&height, sizeof(int), 1, fp);
            fread(&timestamp, sizeof(double), 1, fp);

            cv::Mat yuv(height * 3 / 2, width, CV_8UC1, width);;
            fread(yuv.data, 1, width * height * 3 / 2, fp);

            w_data.color_mat = yuv;
            w_data.color_type_hint = cv::COLOR_YUV2BGR_NV12;
        } else {
            // conventional image
            Bitmap image;
            image.Read(image_file.string(), true);

            uint16_t orientation = 0;
            if (0) {
                FILE * fp = fopen(image_file.string().c_str(), "rb");
                if (fp) {
                    long length = 0;
                    fseek(fp, 0, SEEK_END);
                    length = ftell(fp);
                    fseek(fp, 0, SEEK_SET);
                    std::vector<uint8_t> buffer(length);
                    fread(buffer.data(), 1, length, fp);
                    fclose(fp);

                    TinyEXIF::EXIFInfo info(buffer.data(), length);
                    orientation = info.Orientation;
                }
            }

            w_data.color = image;
            w_data.orientation = orientation;
        }

        std::string packed_file = JoinPaths(bin_path, 
            boost::filesystem::path(GetRelativePath(input_color_path, image_file.string()))
                .replace_extension(boost::filesystem::path(".binx"))
                .string());
        if (!boost::filesystem::exists(boost::filesystem::path(packed_file).parent_path())) {
            boost::filesystem::create_directories(boost::filesystem::path(packed_file).parent_path());
        }
        WriteRGBDBinxData(packed_file, w_data);

    }, index); 
    thread_pool.Wait();

    return 0;
}