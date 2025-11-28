#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <util/rgbd_helper.h>
#include <util/misc.h>
#include <util/threading.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Geometry> 
#include <iomanip>

#include "yaml-cpp/yaml.h"

using namespace sensemap;

std::unordered_map<std::string, double> ReadTimestampsFromCsv(const std::string & file) {
    std::unordered_map<std::string, double> map;
    std::ifstream ifs(file);
    if (!ifs.is_open()) return map;

    std::vector<char> line(1024);
    while (ifs.getline(line.data(), line.size())) {
        double timestamp;
        std::vector<char> filename(1024);
        if (sscanf(line.data(), "%lf,%s", &timestamp, filename.data()) == 2) {
            boost::filesystem::path filepath(filename.data());
            map.emplace(filepath.stem().string(), timestamp);
        }
    }

    return map;
}

std::vector<std::pair<double, Eigen::Vector3d>> ReadGravitysFromeCsv(const std::string & file) {
    std::vector<std::pair<double, Eigen::Vector3d>> result;
    std::ifstream ifs(file);
    if (!ifs.is_open()) return result;

    std::string line;

    std::string s=",";
    char *token;
    double time;
    Eigen::Vector3d gravity;
    while(std::getline(ifs, line)){
        if(line[0]=='#') continue;
        for(int i =0; i<4; i++){
            char *data = i == 0 ? const_cast<char *>(line.c_str()) : nullptr;

            token = strtok(data, s.c_str());
            std::stringstream ss;
            ss<<token;
            if(i==0) ss>>time;
            else ss>>gravity[i-1];
        }

        gravity.normalize();
        result.push_back(std::make_pair(time, gravity));
    }
    return result;
}

Eigen::Matrix3d ReadIntrinsicFromYaml(const std::string & file) {
    YAML::Node node = YAML::LoadFile(file);
    std::vector<double> camera = node["intrinsic"]["camera"].as<std::vector<double>>();
    CHECK_EQ(camera.size(), 4);

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = camera[0];
    K(1, 1) = camera[1];
    K(0, 2) = camera[2];
    K(1, 2) = camera[3];

    return K;
}

Eigen::Matrix4d ReadExtrisicFromYaml(const std::string & file) {
    YAML::Node node = YAML::LoadFile(file);
    std::vector<double> q = node["extrinsic"]["q"].as<std::vector<double>>();
    std::vector<double> p = node["extrinsic"]["p"].as<std::vector<double>>();
    CHECK_EQ(q.size(), 4);
    CHECK_EQ(p.size(), 3);

    Eigen::Vector4d qvec = Eigen::Vector4d(0, 0, 0, 1);
    Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
    qvec[0] = q[0];
    qvec[1] = q[1];
    qvec[2] = q[2];
    qvec[3] = q[3];
    tvec[0] = p[0];
    tvec[1] = p[1];
    tvec[2] = p[2];

    Eigen::Matrix4d RT = Eigen::Matrix4d::Identity();
    RT.block<3, 3>(0, 0) = Eigen::Quaterniond(
        qvec.w(), qvec.x(), 
        qvec.y(), qvec.z()
    ).toRotationMatrix();
    RT.block<3, 1>(0, 3) = tvec;

    return RT;
}

Eigen::Vector3d GetInterpolationGravity(std::vector<std::pair<double, Eigen::Vector3d>> gravitys, double time){
    Eigen::Vector3d g(0,0,0);

    if(gravitys.empty()) return g;

    int pos=-1, pos2 = -1;
    for(int i=0; i<gravitys.size(); i++){
        if(gravitys[i].first>=time){
            pos = i;
            break;
        }
    }

    if (pos==-1 && std::fabs(gravitys.back().first-time)>0.01) return g;
    if (pos==0 && std::abs(gravitys[0].first-time)>0.01) return g;


    if (pos==0) pos2 = 1;
    else  pos2 = pos-1;

    double t0 = gravitys[pos].first;
    double t1 = gravitys[pos2].first;
    Eigen::Vector3d v0 = gravitys[pos].second;
    Eigen::Vector3d v1 = gravitys[pos2].second;

    if (std::fabs(time-t0)<0.005) return v0;
    if (std::fabs(time-t1)<0.005) return v1;

    Eigen::Vector3d rot_axis = v0.cross(v1);

    if (rot_axis.norm()<0.001) {
        g= v0 + (v1-v0)/(t1-t0)*(time - t0);
        g.normalize();
        return g;
    }

    rot_axis /= rot_axis.norm();
    double rot_angle = acos(v0.dot(v1));
    double new_angle = rot_angle/(t1-t0)*(time - t0);

    Eigen::AngleAxisd rot_vec(new_angle, rot_axis);
    Eigen::Matrix3d rot_mat = rot_vec.toRotationMatrix();

    g = rot_mat * v0;

    g.normalize();
    return g;
}



int main(int argc, char* argv[]) {
    std::string path;
    std::string packed_path;
    int transpose = -1;

    if (argc == 3) {
        path = argv[1];
        packed_path = argv[2];
    } else if (argc == 4) {
        path = argv[1];
        packed_path = argv[2];
        transpose = std::atoi(argv[3]);
    } else {
        std::cout << "Usage 1: " << argv[0] << std::endl;
        std::cout << "    <DIOR_PATH> <OUTPUT_PATH> [TRANSPOSE=" << transpose << "]" << std::endl;
        return 1;
    }

    if (!ExistsDir(path)) {
        std::cout << path << " doesn't exists! " << std::endl;
        return 1;
    }

    bool high_res = false;
    std::string camera_path;
    std::string rgb_image_path;
    std::string depth_path = JoinPaths(path, "depth");
    std::string attitude_path = JoinPaths(path, "attitude");
    if (ExistsDir(JoinPaths(path, "high_camera", "raw_images"))) {
        high_res = true;
        camera_path = JoinPaths(path, "high_camera");
        rgb_image_path = JoinPaths(camera_path, "raw_images");
        std::cout << "Packing images in " << camera_path << std::endl;
    } else {
        camera_path = JoinPaths(path, "camera");
        rgb_image_path = JoinPaths(camera_path, "images");
        std::cout << "Packing images in " << camera_path << std::endl;
    }

    if (!ExistsDir(camera_path)) {
        std::cout << camera_path << " doesn't exists! " << std::endl;
        return 1;
    }
    if (!ExistsDir(depth_path)) {
        std::cout << depth_path << " doesn't exists! " << std::endl;
        return 1;
    }
    if (ExistsDir(packed_path)) {
        boost::filesystem::remove_all(packed_path);
    }
    boost::filesystem::create_directories(packed_path);

    const std::string depth_ext = ".png";
    const std::string packed_ext = ".binx";
    std::vector<std::string> camera_images = GetRecursiveFileList(rgb_image_path);

    std::unordered_map<std::string, double> color_timestamps = ReadTimestampsFromCsv(JoinPaths(camera_path, "data.csv"));
    std::unordered_map<std::string, double> depth_timestamps = ReadTimestampsFromCsv(JoinPaths(depth_path, "data.csv"));
    Eigen::Matrix3d K_rgb_small = ReadIntrinsicFromYaml(JoinPaths(camera_path, "sensor.yaml"));
    Eigen::Matrix3d K_tof = ReadIntrinsicFromYaml(JoinPaths(depth_path, "sensor.yaml"));
    Eigen::Matrix4d RT = ReadExtrisicFromYaml(JoinPaths(depth_path, "sensor.yaml"));

    int width_small_rgb = 0;
    int height_small_rgb = 0;
    int width_depth = 0;
    int height_depth = 0;
    {
        std::vector<std::string> low_camera_images = GetRecursiveFileList(JoinPaths(path, "camera", "images"));
        for (const auto & low_camera_image : low_camera_images) {
            cv::Mat rgb_small = cv::imread(low_camera_image, cv::IMREAD_UNCHANGED);
            if (!rgb_small.empty()) {
                width_small_rgb = rgb_small.cols;
                height_small_rgb = rgb_small.rows;
                break;
            }
        }

        if (width_small_rgb == 0 || height_small_rgb == 0) {
            std::cerr << "Cannot find RGB images" << std::endl;
            return 1;
        }

        std::vector<std::string> depth_images = GetRecursiveFileList(JoinPaths(path, "depth", "images"));
        for (const auto & depth_image : depth_images) {
            cv::Mat depth = cv::imread(depth_image, cv::IMREAD_UNCHANGED);
            if (!depth.empty()) {
                width_depth = depth.cols;
                height_depth = depth.rows;
                break;
            }
        }

        if (width_depth == 0 || height_depth == 0) {
            std::cerr << "Cannot find depth images" << std::endl;
            return 1;
        }
    }

    switch (transpose) {
    // 90deg clock-wise
    case 1: {
        // fx <=> fy
        std::swap(K_rgb_small(0, 0), K_rgb_small(1, 1));
        std::swap(K_tof(0, 0), K_tof(1, 1));

        // cx => cy
        // height - 1 - cy => cx
        std::swap(K_rgb_small(0, 2), K_rgb_small(1, 2));
        std::swap(K_tof(0, 2), K_tof(1, 2));
        K_rgb_small(0, 2) = height_small_rgb - 1 - K_rgb_small(0, 2);
        K_tof(0, 2) = height_depth - 1 - K_tof(0, 2);

        // width <=> height
        std::swap(width_small_rgb, height_small_rgb);
        std::swap(width_depth, height_depth);
    } break;
    }

    std::vector<std::pair<double, Eigen::Vector3d>> gravitys = ReadGravitysFromeCsv(JoinPaths(attitude_path, "data.csv"));
    Eigen::Matrix4d imu2cameraRT = ReadExtrisicFromYaml(JoinPaths(attitude_path, "sensor.yaml"));

    const int num_eff_threads = sensemap::GetEffectiveNumThreads(-1);
    sensemap::ThreadPool thread_pool(num_eff_threads);

    for (size_t index = 0; index < camera_images.size(); ++index) {
        thread_pool.AddTask([&](size_t i) {
            const auto & color_file = camera_images[i];
            std::string color_name = boost::filesystem::path(color_file).stem().string();

            std::string depth_file = JoinPaths(depth_path, "images", color_name + depth_ext);
            std::string packed_file = JoinPaths(packed_path, color_name + packed_ext);
            if (!boost::filesystem::exists(depth_file)) return;

            RGBDWriteData w_data;
            Eigen::Matrix3d K_rgb = K_rgb_small;
            if (high_res) {
                std::ifstream file(color_file, std::ifstream::binary);
                double timestamp;
                int width, height;
                file.read(reinterpret_cast<char*>(&width), sizeof(int));
                file.read(reinterpret_cast<char*>(&height), sizeof(int));
                file.read(reinterpret_cast<char*>(&timestamp), sizeof(double));
                K_rgb.row(0) *= 1.0 * width / width_small_rgb;
                K_rgb.row(1) *= 1.0 * height / height_small_rgb;

                w_data.color_type_hint = cv::COLOR_YUV2BGR_NV21;
                w_data.color_mat = cv::Mat(height / 2 * 3, width, CV_8UC1);
                file.read(reinterpret_cast<char*>(w_data.color_mat.data), sizeof(char)* width * height / 2 * 3);
                file.close();
            } else {
                w_data.color_mat = cv::imread(color_file, cv::IMREAD_COLOR);
            }

            w_data.depth_mat = cv::imread(depth_file, cv::IMREAD_UNCHANGED);
            if (w_data.depth_mat.type() != CV_16UC1) return;

            switch (transpose) {
            // 90deg clock-wise
            case 1: {
                CHECK(!high_res) << "transpose option for high_res images not implemented. ";
                cv::transpose(w_data.color_mat, w_data.color_mat);
                cv::transpose(w_data.depth_mat, w_data.depth_mat);
                cv::flip(w_data.color_mat, w_data.color_mat, 1);
                cv::flip(w_data.depth_mat, w_data.depth_mat, 1);
            } break;
            }

            std::ofstream file(packed_file, std::ios::binary);
            if (!file.is_open()) {
                std::cout << "fail to open data file" << std::endl;
                return;
            }

            /// rgb

            double rgb_timestamp = color_timestamps.find(color_name) != color_timestamps.end() ?
                                   color_timestamps.at(color_name) : 0.0;
            double depth_timestamp = depth_timestamps.find(color_name) != depth_timestamps.end() ?
                                     depth_timestamps.at(color_name) : 0.0; 
            Eigen::Vector3d gravity = GetInterpolationGravity(gravitys, rgb_timestamp);
            if(!gravity.isZero()) gravity.applyOnTheLeft(imu2cameraRT.block<3,3>(0,0).transpose());

            Camera camera_rgb;
            camera_rgb.SetModelIdFromName("PINHOLE");
            camera_rgb.SetFocalLengthX(K_rgb(0, 0));
            camera_rgb.SetFocalLengthY(K_rgb(1, 1));
            camera_rgb.SetPrincipalPointX(K_rgb(0, 2));
            camera_rgb.SetPrincipalPointY(K_rgb(1, 2));

            Camera camera_tof;
            camera_tof.SetModelIdFromName("PINHOLE");
            camera_tof.SetFocalLengthX(K_tof(0, 0));
            camera_tof.SetFocalLengthY(K_tof(1, 1));
            camera_tof.SetPrincipalPointX(K_tof(0, 2));
            camera_tof.SetPrincipalPointY(K_tof(1, 2));

            if (i == 0) {
                std::cout << "Input RT: " << std::endl;
                std::cout << RT << std::endl << std::endl;

                std::cout << "Image Camera: " << camera_rgb.ModelName() << std::endl;
                std::cout << camera_rgb.ParamsToString() << std::endl << std::endl;
                
                std::cout << "Depth Camera: " << camera_tof.ModelName() << std::endl;
                std::cout << camera_tof.ParamsToString() << std::endl << std::endl;
            }

            w_data.color_camera = camera_rgb;
            w_data.color_timestamp = rgb_timestamp;
            w_data.depth_camera = camera_tof;
            w_data.depth_RT = RT;
            w_data.depth_timestamp = depth_timestamp;
            w_data.timestamp = rgb_timestamp;
            Eigen::Quaterniond Calib_QIC = Eigen::Quaterniond{0.0, 0.707107, -0.707107, 0.0};
            Calib_QIC.normalize();
            w_data.gravity << gravity[0], gravity[1], gravity[2];
            w_data.gravity.applyOnTheLeft(Calib_QIC.toRotationMatrix().transpose());
            w_data.gravity.normalize();
            WriteRGBDData(packed_file, w_data);
        }, index); 
    }
    thread_pool.Wait();

    return 0;
}