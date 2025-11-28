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
#include <Eigen/Geometry> 
#include <iomanip>
#include <base/camera_models.h>
#include <base/undistortion.h>
#include <util/TinyEXIF.h>
#include "base/version.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

Camera IntrinsicStringToCamera(const std::string & str) {
    if (boost::filesystem::is_regular_file(str)) {
        std::ifstream ifs(str);
        char line[4096];
        ifs.getline(line, sizeof(line));

        std::string data(line);
        if (!data.empty()) {
            return IntrinsicStringToCamera(data);
        }
    }

    Camera camera;
    CHECK(IntrinsicStringToCamera(str, camera)) << "Intrinsic string invalid";
    return camera;
}

bool TryParseCalibYaml(
    const std::string & filename,
    std::string & image_intrinsic_string,
    std::string & depth_intrinsic_string,
    std::string & RT_string
) {
    if (!ExistsFile(filename)) return false;

    YAML::Node node = YAML::LoadFile(filename);
    YAML::Node calib_color = node["color"];
    YAML::Node calib_depth = node["depth"];

    if (!calib_color.IsDefined() || !calib_depth.IsDefined()) return false;
    image_intrinsic_string = calib_color["camera_model"].as<std::string>() + ", " + calib_color["camera_params"].as<std::string>();
    depth_intrinsic_string = calib_depth["camera_model"].as<std::string>() + ", " + calib_depth["camera_params"].as<std::string>();

    auto mat_data = calib_depth["RT"].as<std::vector<double>>();
    RT_string = VectorToCSV(mat_data);

    return true;
}

void WarpDepthBetweenCameras(const Camera& source_camera,
                             const Camera& target_camera,
                             const MatXf& source_depth,
                             MatXf* target_depth) {
    CHECK_EQ(source_camera.Width(), source_depth.GetWidth());
    CHECK_EQ(source_camera.Height(), source_depth.GetHeight());

    *target_depth = MatXf(target_camera.Width(), target_camera.Height(), 1);
    
    Eigen::Vector2d image_point;
    for (int y = 0; y < target_depth->GetHeight(); ++y) {
        image_point.y() = y + 0.5;
        for (int x = 0; x < target_depth->GetWidth(); ++x) {
            image_point.x() = x + 0.5;

            const Eigen::Vector2d world_point = 
                target_camera.ImageToWorld(image_point);
            const Eigen::Vector2d source_point =
                source_camera.WorldToImage(world_point);
            
            const double yf = source_point.y();
            const double xf = source_point.x();
            const int y0 = std::floor(yf);
            const int x0 = std::floor(xf);
            const int y1 = std::ceil(yf);
            const int x1 = std::ceil(xf);

            if (y0 < 0 || y0 >= source_depth.GetHeight() ||
                x0 < 0 || x0 >= source_depth.GetWidth() ||
                y1 < 0 || y1 >= source_depth.GetHeight() ||
                x1 < 0 || x1 >= source_depth.GetWidth()
            ) {
                target_depth->Set(y, x, 0);
                continue;
            }

            float depth = 0.0f;
            float depth_weight = 0.0f;
            float depth00 = source_depth.Get(y0, x0);
            float depth01 = source_depth.Get(y0, x1);
            float depth10 = source_depth.Get(y1, x0);
            float depth11 = source_depth.Get(y1, x1);
            if (depth00 > 0.0f) {
                float w  = 1.0f - (xf - x0);
                      w *= 1.0f - (yf - y0);
                depth += w * depth00;
                depth_weight += w;
            }
            if (depth01 > 0.0f) {
                float w  = 1.0f - (xf - x0);
                      w *= 1.0f - (y1 - yf);
                depth += w * depth01;
                depth_weight += w;
            }
            if (depth10 > 0.0f) {
                float w  = 1.0f - (x1 - xf);
                      w *= 1.0f - (yf - y0);
                depth += w * depth10;
                depth_weight += w;
            }
            if (depth11 > 0.0f) {
                float w  = 1.0f - (x1 - xf);
                      w *= 1.0f - (y1 - yf);
                depth += w * depth11;
                depth_weight += w;
            }

            if (depth_weight <= 0.5f) {
                target_depth->Set(y, x, 0);
                continue;
            }

            depth /= depth_weight;
            target_depth->Set(y, x, depth);
        }
    }
}

int main(int argc, char* argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: sfm-bin-packer-")+__VERSION__);

    std::string image_path, depth_path, bin_path;
    std::string image_intrinsic_string, depth_intrinsic_string;
    std::string RT_string;
    bool do_undistort = false;
    if (argc == 5) {
        image_path = argv[1];
        depth_path = argv[2];
        bin_path = argv[3];
        if (!TryParseCalibYaml(argv[4], image_intrinsic_string, depth_intrinsic_string, RT_string)) {
            std::cout << "Cannot read " << argv[4] << std::endl;
            return 1;
        }
    } else if (argc == 6) {
        image_path = argv[1];
        depth_path = argv[2];
        bin_path = argv[3];
        do_undistort = std::atoi(argv[5]);
        if (!TryParseCalibYaml(argv[4], image_intrinsic_string, depth_intrinsic_string, RT_string)) {
            std::cout << "Cannot read " << argv[4] << std::endl;
            return 1;
        }
    } else if (argc == 7) {
        image_path = argv[1];
        depth_path = argv[2];
        bin_path = argv[3];
        image_intrinsic_string = argv[4];
        depth_intrinsic_string = argv[5];
        RT_string = argv[6];
    } else if (argc == 8) {
        image_path = argv[1];
        depth_path = argv[2];
        bin_path = argv[3];
        image_intrinsic_string = argv[4];
        depth_intrinsic_string = argv[5];
        RT_string = argv[6];
        do_undistort = std::atoi(argv[7]);
    } else {
        std::cout << "Usage 1: " << argv[0] << std::endl;
        std::cout << "    <IMAGE_PATH> <DEPTH_PATH> <BIN_PATH>" << std::endl;
        std::cout << "    <IMAGE_INTRINSIC_STRING> <DEPTH_INTRINSIC_STRING>" << std::endl;
        std::cout << "    <DEPTH_TO_IMAGE_RT> [DO_UNDISTORT="<< do_undistort << "]" << std::endl;
        std::cout << std::endl;
        std::cout << "    Note: " << std::endl;
        std::cout << "    *_INTRINSIC_STRING: " << std::endl;
        std::cout << "        An array of name following floats denoted as \"MODEL_NAME,f,f,f,f,...\". " << std::endl;
        std::cout << "        OR " << std::endl;
        std::cout << "        A path to an ASCII file containing such data. "  << std::endl;
        std::cout << "        The MODEL_NAME, e.g., PINHOLE(4), RADIAL(5), OPENCV(8) or FULL_OPENCV(12). " << std::endl;
        std::cout << "    DEPTH_TO_IMAGE_RT: " << std::endl;
        std::cout << "        An array of 16 floats denoted as \"f,f,f,f,...\", row-majored. " << std::endl;
        std::cout << "        OR " << std::endl;
        std::cout << "        A path to an ASCII file containing 16 floats denoted as \"f f f f ...\", row-majored.. "  << std::endl;
        std::cout << std::endl;
        std::cout << "Usage 2: " << argv[0] << std::endl;
        std::cout << "    <IMAGE_PATH> <DEPTH_PATH> <BIN_PATH>" << std::endl;
        std::cout << "    <RGBD_CALIB_YAML> [DO_UNDISTORT="<< do_undistort << "]" << std::endl;
        std::cout << std::endl;
        return 1;
    }

    if (boost::filesystem::exists(bin_path)) {
        boost::filesystem::remove_all(bin_path);
    }
    boost::filesystem::create_directories(bin_path);

    Eigen::Matrix4d RT;
    if (boost::filesystem::is_regular_file(RT_string)) {
        std::ifstream ifs(RT_string);
        ifs >> RT(0, 0) >> RT(0, 1) >> RT(0, 2) >> RT(0, 3);
        ifs >> RT(1, 0) >> RT(1, 1) >> RT(1, 2) >> RT(1, 3);
        ifs >> RT(2, 0) >> RT(2, 1) >> RT(2, 2) >> RT(2, 3);
        ifs >> RT(3, 0) >> RT(3, 1) >> RT(3, 2) >> RT(3, 3);
    } else {
        auto RT_vector = CSVToVector<double>(RT_string);
        if (RT_vector.size() != 16) {
            std::cout << "RT must have 16 elements!" << std::endl;
            std::abort();
        }
        RT(0, 0) = RT_vector[0];  RT(0, 1) = RT_vector[1];  RT(0, 2) = RT_vector[2];  RT(0, 3) = RT_vector[3];
        RT(1, 0) = RT_vector[4];  RT(1, 1) = RT_vector[5];  RT(1, 2) = RT_vector[6];  RT(1, 3) = RT_vector[7];
        RT(2, 0) = RT_vector[8];  RT(2, 1) = RT_vector[9];  RT(2, 2) = RT_vector[10]; RT(2, 3) = RT_vector[11];
        RT(3, 0) = RT_vector[12]; RT(3, 1) = RT_vector[13]; RT(3, 2) = RT_vector[14]; RT(3, 3) = RT_vector[15];
    }
    std::cout << "Input RT: " << std::endl;
    std::cout << RT << std::endl << std::endl;

    std::vector<boost::filesystem::path> image_files;
    std::map<std::string, boost::filesystem::path> depth_files;
    boost::filesystem::recursive_directory_iterator end_iter;

    boost::filesystem::path image_full_path(image_path);
    for (boost::filesystem::recursive_directory_iterator iter(image_full_path); iter != end_iter; iter++) {
        if (boost::filesystem::is_regular_file(*iter)) {
            boost::filesystem::path filepath(iter->path().string());
            image_files.emplace_back(std::move(filepath));
        }
    }

    boost::filesystem::path depth_full_path(depth_path);
    for (boost::filesystem::recursive_directory_iterator iter(depth_full_path); iter != end_iter; iter++) {
        if (boost::filesystem::is_regular_file(*iter)) {
            boost::filesystem::path filepath(iter->path().string());
            depth_files[filepath.stem().string()] = filepath;
        }
    }

    Camera image_camera = IntrinsicStringToCamera(image_intrinsic_string);
    for (size_t i = 0; i < image_files.size(); ++i) {
        auto & image_file = image_files[i];
        cv::Mat rgb = cv::imread(image_file.string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (!rgb.empty()) {
            image_camera.SetWidth(rgb.cols);
            image_camera.SetHeight(rgb.rows);
            break;
        }
    }
    if (image_camera.Width() * image_camera.Height() == 0) {
        std::abort();
    }
    std::cout << "Image Camera: " << image_camera.ModelName() << std::endl;
    std::cout << image_camera.ParamsToString() << std::endl << std::endl;
    
    Camera depth_camera = IntrinsicStringToCamera(depth_intrinsic_string);
    for (auto & depth_item : depth_files) {
        auto & depth_file = depth_item.second;
        cv::Mat depth = cv::imread(depth_file.string(), cv::IMREAD_ANYDEPTH);
        if (!depth.empty()) {
            depth_camera.SetWidth(depth.cols);
            depth_camera.SetHeight(depth.rows);
            break;
        }
    }
    if (depth_camera.Width() * depth_camera.Height() == 0) {
        std::abort();
    }
    std::cout << "Depth Camera: " << depth_camera.ModelName() << std::endl;
    std::cout << depth_camera.ParamsToString() << std::endl << std::endl;

    UndistortOptions options;
    Camera undistorted_depth_camera;
    if (do_undistort) {
        Undistorter::UndistortCamera(options, depth_camera, &undistorted_depth_camera);
    } else {
        undistorted_depth_camera = depth_camera;
    }

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

        MatXf warped_depthmap(undistorted_depth_camera.Width(), undistorted_depth_camera.Height(), 1);
        if (do_undistort) {
            UniversalWarpDepthMap(warped_depthmap, depthmap, undistorted_depth_camera, depth_camera, Eigen::Matrix4f::Identity());
            // WarpDepthBetweenCameras(depth_camera, undistorted_depth_camera, depthmap, &warped_depthmap);
            depth = cv::Mat(warped_depthmap.GetHeight(), warped_depthmap.GetWidth(), CV_16UC1);
            for (int y = 0; y < depth.rows; y++) {
                for (int x = 0; x < depth.cols; x++) {
                    float d = warped_depthmap.Get(y, x);
                    depth.at<ushort>(y, x) = d * 1000.0 + 0.5;
                }
            }
        } else {
            warped_depthmap = depthmap;
        }

        // std::ofstream ofs(stem + ".obj");
        // for (int y = 0; y < undistorted_depth_camera.Height(); y++) {
        //     for (int x = 0; x < undistorted_depth_camera.Width(); x++) {
        //         float d = warped_depthmap.Get(y, x);
                
        //         Eigen::Vector2d depth_pt_image_2d(x, y);
        //         Eigen::Vector3d pt = undistorted_depth_camera.ImageToWorld(depth_pt_image_2d).homogeneous();
        //         pt *= (double)d;
        //         ofs << "v " << pt.transpose() << std::endl;
        //     }
        // }

        Bitmap image, undistorted_image;
        image.Read(image_file.string(), true);

        uint16_t orientation = 0;
        {
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

        Camera undistorted_image_camera;
        if (do_undistort) {
            Undistorter::UndistortImage(options, image, image_camera, &undistorted_image, &undistorted_image_camera);
        } else {
            undistorted_image = image;
            undistorted_image_camera = image_camera;
        }

        if (i == 0) {
            std::cout << "Output Image Camera: " << undistorted_image_camera.ModelName() << std::endl;
            std::cout << undistorted_image_camera.ParamsToString() << std::endl << std::endl;

            std::cout << "Output Depth Camera: " << undistorted_depth_camera.ModelName() << std::endl;
            std::cout << undistorted_depth_camera.ParamsToString() << std::endl << std::endl;
        }

        cv::Mat rgb;
        FreeImage2Mat(&undistorted_image, rgb);
        std::string packed_file = bin_path + "/" + stem + ".binx";

        RGBDWriteData w_data;
        w_data.color = undistorted_image;
        w_data.color_camera = undistorted_image_camera;
        w_data.depth = warped_depthmap;
        w_data.depth_camera = undistorted_depth_camera;
        w_data.depth_RT = RT;
        w_data.orientation = orientation;
        WriteRGBDBinxData(packed_file, w_data);
        
        // {
        //     RGBDData r_data;
        //     RGBDReadOption r_option;
        //     ExtractRGBDBinxAllData(packed_file, r_option, r_data);
        //     r_data.color.Write("test.png");

        //     MatXf warped_depthmap2(undistorted_image_camera.Width(), undistorted_image_camera.Height(), 1);
        //     FastWarpDepthMap(warped_depthmap2, r_data.depth, 
        //         r_data.color_K.cast<float>(), 
        //         r_data.depth_K.cast<float>(), 
        //         r_data.depth_RT.cast<float>());

        //     FILE * fp = fopen("test.obj", "w");
        //     for (int y = 0; y < r_data.color.Height(); y += 3) {
        //         for (int x = 0; x < r_data.color.Width(); x += 3) {
        //             auto c = r_data.color.GetPixel(x, y);
        //             float d = warped_depthmap2.Get(y, x);
        //             if (d <= 0) continue;

        //             Eigen::Vector2d pt2d(x, y);
        //             Eigen::Vector3d pt3d = undistorted_image_camera.ImageToWorld(pt2d).homogeneous();
        //                             pt3d *= d;
        //             fprintf(fp, "v %f %f %f %d %d %d\n",
        //                 pt3d.x(), pt3d.y(), pt3d.z(),
        //                 c.r, c.g, c.b
        //             );
        //         }
        //     }
        //     fclose(fp);
        //     exit(0);
        // }

        undistorted_image.Deallocate();
    }, index); 
    thread_pool.Wait();

    return 0;
}