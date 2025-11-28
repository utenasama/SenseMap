// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/sparsification/sampler-base.h"
#include "util/sparsification/sampler-factory.h"

#include "cluster/fast_community.h"
#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include <dirent.h>
#include <sys/stat.h>

using namespace sensemap;

std::string image_path;
std::string workspace_path;
std::string input_path;
std::string output_path;

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

// Read ORB feature
void ReadORBFeatureBinary(std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features,
                          const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open());

    // read image number
    image_t num_images = ReadBinaryLittleEndian<image_t>(&file);
    // std::cout << "Image number = " << num_images << std::endl;

    for (int i = 0; i < num_images; i++) {
        // Image id
        image_t image_id = ReadBinaryLittleEndian<image_t>(&file);
        // std::cout << "Image id = " << image_id << std::endl;

        // Get Keypoint and descriptor
        // Keypoint number
        int num_kp = ReadBinaryLittleEndian<int>(&file);
        // std::cout << "kp num = " << num_kp << std::endl;

        std::vector<cv::KeyPoint> cur_keypoints;
        // Read Keypoints
        for (size_t i = 0; i != num_kp; ++i) {
            cv::KeyPoint keypoint;
            keypoint.pt.x = ReadBinaryLittleEndian<float>(&file);
            keypoint.pt.y = ReadBinaryLittleEndian<float>(&file);
            keypoint.size = ReadBinaryLittleEndian<float>(&file);
            keypoint.angle = ReadBinaryLittleEndian<float>(&file);
            keypoint.response = ReadBinaryLittleEndian<float>(&file);
            keypoint.octave = ReadBinaryLittleEndian<int>(&file);
            keypoint.class_id = ReadBinaryLittleEndian<int>(&file);

            // std::cout << "x = " << keypoint.pt.x << std::endl;
            // std::cout << "y = " << keypoint.pt.y << std::endl;
            // std::cout << "scale = " << keypoint.size << std::endl;
            // std::cout << "ori = " << keypoint.angle << std::endl;
            // std::cout << "response = " << keypoint.response << std::endl;
            // std::cout << "octave = " << keypoint.octave << std::endl;
            // std::cout << "class_id = " << keypoint.class_id << std::endl;

            cur_keypoints.emplace_back(keypoint);
        }

        // Read descriptors
        cv::Mat mat(num_kp, 32, CV_8UC1);
        file.read((char *)(mat.data), mat.elemSize() * mat.total());

        features[image_id] = std::make_pair(cur_keypoints, mat);
    }

    file.close();
}

// Write ORB Feature
void WriteORBFeatureBinary(const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features,
                           const std::string &path) {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open());

    // write image number
    auto num_images = static_cast<image_t>(features.size());
    // std::cout << "Image number = " << num_images << std::endl;
    file.write((char *)&num_images, sizeof(image_t));

    for (const auto &it : features) {
        // Image id
        image_t image_id = static_cast<image_t>(it.first);
        file.write((char *)&image_id, sizeof(image_id));

        // Get Keypoint and descriptor
        auto kp = it.second.first;
        auto ds = it.second.second;

        // Keypoint number
        int num_kp = static_cast<int>(kp.size());
        file.write((char *)&num_kp, sizeof(num_kp));
        // std::cout << "kp num = " << num_kp << std::endl;

        // Output Keypoints
        for (size_t i = 0; i != kp.size(); ++i) {
            float size, angle, response, x, y;
            int octave, class_id;
            x = kp[i].pt.x;
            y = kp[i].pt.y;
            size = kp[i].size;
            angle = kp[i].angle;
            response = kp[i].response;
            octave = kp[i].octave;
            class_id = kp[i].class_id;

            // std::cout << "x = " << x << std::endl;
            // std::cout << "y = " << y << std::endl;
            // std::cout << "size = " << size << std::endl;
            // std::cout << "angle = " << angle << std::endl;

            file.write((char *)&x, sizeof(x));
            file.write((char *)&y, sizeof(y));
            file.write((char *)&size, sizeof(size));
            file.write((char *)&angle, sizeof(angle));
            file.write((char *)&response, sizeof(response));
            file.write((char *)&octave, sizeof(octave));
            file.write((char *)&class_id, sizeof(class_id));
        }

        // Output descriptors
        // std::cout << "ds rows = " << ds.rows << " , " << ds.cols << std::endl;
        file.write((const char *)(ds.data), ds.elemSize() * ds.total());
    }

    file.close();
}

void WriteORBMapBinary(std::shared_ptr<Reconstruction> reconstruction, std::vector<image_t> image_ids,
                       std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features,
                       const std::string &path) {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open());

    //////////////////////////////////////////////////////////////////////
    // 1. param
    //////////////////////////////////////////////////////////////////////
    // read map version
    int map_version = 0;  // 0-no img  1-with img
    WriteBinaryLittleEndian<int>(&file, map_version);
    std::cout << "map version = " << map_version << std::endl;
    int img_width = 1920;
    int img_height = 1080;
    WriteBinaryLittleEndian<int>(&file, img_width);
    WriteBinaryLittleEndian<int>(&file, img_height);

    double fx = 400;
    double fy = 400;
    double cx = 960;
    double cy = 540;
    WriteBinaryLittleEndian<double>(&file, fx);
    WriteBinaryLittleEndian<double>(&file, fy);
    WriteBinaryLittleEndian<double>(&file, cx);
    WriteBinaryLittleEndian<double>(&file, cy);

    std::cout << "img_width = " << img_width << std::endl;
    std::cout << "img_height = " << img_height << std::endl;
    std::cout << "fx = " << fx << std::endl;
    std::cout << "fy = " << fy << std::endl;
    std::cout << "cx = " << cx << std::endl;
    std::cout << "cy = " << cy << std::endl;

    double tic_x = 0;
    double tic_y = 0;
    double tic_z = 0;
    WriteBinaryLittleEndian<double>(&file, tic_x);
    WriteBinaryLittleEndian<double>(&file, tic_y);
    WriteBinaryLittleEndian<double>(&file, tic_z);

    std::cout << "tic = " << tic_x << " , " << tic_y << " , " << tic_z << std::endl;

    double qic_w = 1;
    double qic_x = 0;
    double qic_y = 0;
    double qic_z = 0;

    WriteBinaryLittleEndian<double>(&file, qic_w);
    WriteBinaryLittleEndian<double>(&file, qic_x);
    WriteBinaryLittleEndian<double>(&file, qic_y);
    WriteBinaryLittleEndian<double>(&file, qic_z);

    std::cout << "qic = " << qic_w << " , " << qic_x << " , " << qic_y << ", " << qic_z << std::endl;

    //////////////////////////////////////////////////////////////////////
    // 2. Keyframe entrence
    //////////////////////////////////////////////////////////////////////
    auto rec_image_ids = reconstruction->RegisterImageIds(image_ids);
    int keyframe_entrence_num = rec_image_ids.size();
    WriteBinaryLittleEndian<int>(&file, keyframe_entrence_num);
    std::cout << "keyframe_entrence_num = " << keyframe_entrence_num << std::endl;

    for (int i = 0; i < keyframe_entrence_num; i++) {
        WriteBinaryLittleEndian<int>(&file, i);
    }

    //////////////////////////////////////////////////////////////////////
    // 3. Keyframe
    //////////////////////////////////////////////////////////////////////
    // read keyframe number
    int keyframe_num = rec_image_ids.size();
    std::cout << "keyframe_num = " << keyframe_num << std::endl;
    WriteBinaryLittleEndian<int>(&file, keyframe_num);

    // Store all the mappoint ids
    std::unordered_set<mappoint_t> mappoints_set;

    for (int i = 0; i < keyframe_num; i++) {
        int keyframe_id = rec_image_ids[i];
        auto keyframe = features[keyframe_id];
        int keypoint_num = keyframe.first.size();

        WriteBinaryLittleEndian<int>(&file, keyframe_id);
        WriteBinaryLittleEndian<int>(&file, keypoint_num);

        // std::cout << "keyframe_id = " << keyframe_id << std::endl;
        // std::cout << "keypoint_num = " << keypoint_num << std::endl;

        for (int j = 0; j < keypoint_num; j++) {
            auto keypoint = keyframe.first[j];
            WriteBinaryLittleEndian<float>(&file, keypoint.pt.x);
            WriteBinaryLittleEndian<float>(&file, keypoint.pt.y);
            WriteBinaryLittleEndian<float>(&file, keypoint.size);
            WriteBinaryLittleEndian<float>(&file, keypoint.angle);
            WriteBinaryLittleEndian<float>(&file, keypoint.response);
            WriteBinaryLittleEndian<int>(&file, keypoint.octave);
            WriteBinaryLittleEndian<int>(&file, keypoint.class_id);

            // std::cout << "keypoint.pt.x = " << keypoint.pt.x << std::endl;
            // std::cout << "keypoint.pt.y = " << keypoint.pt.y << std::endl;
            // std::cout << "keypoint.size = " << keypoint.size << std::endl;
            // std::cout << "keypoint.angle = " << keypoint.angle << std::endl;
            // std::cout << "keypoint.response = " << keypoint.response << std::endl;
            // std::cout << "keypoint.octave = " << keypoint.octave << std::endl;
            // std::cout << "keypoint.class_id = " << keypoint.class_id << std::endl;
        }

        // Save descriptor
        file.write((const char *)(keyframe.second.data), keyframe.second.elemSize() * keyframe.second.total());

        // Get current image from reconstruction
        auto rec_keyframe = reconstruction->Image(keyframe_id);

        // Read Keyframe pose
        WriteBinaryLittleEndian<double>(&file, rec_keyframe.Tvec()[0]);
        WriteBinaryLittleEndian<double>(&file, rec_keyframe.Tvec()[1]);
        WriteBinaryLittleEndian<double>(&file, rec_keyframe.Tvec()[2]);

        // std::cout << "Pose -- translation = " << rec_keyframe.Tvec()[0] << " , " << rec_keyframe.Tvec()[1]
        //           << " , " << rec_keyframe.Tvec()[2] << std::endl;

        double w, x, y, z;
        w = rec_keyframe.Qvec()[0];
        x = rec_keyframe.Qvec()[1];
        y = rec_keyframe.Qvec()[2];
        z = rec_keyframe.Qvec()[3];

        // std::cout << "Pose -- rotation = " << w << " , " << x << " , " << y << " , " << z << std::endl;
        WriteBinaryLittleEndian<double>(&file, w);
        WriteBinaryLittleEndian<double>(&file, x);
        WriteBinaryLittleEndian<double>(&file, y);
        WriteBinaryLittleEndian<double>(&file, z);

        const auto &all_point2ds = rec_keyframe.Points2D();
        for (const auto &point_2d : all_point2ds) {
            if (point_2d.HasMapPoint()) {
                mappoints_set.insert(point_2d.MapPointId());
            }
        }

        // FIXME: DO not need. Load Image
        // auto image_name = feature_data_container->GetImage(keyframe_id).Name();
        // cv::Mat cur_img = cv::imread(image_path + "/" + image_name);
        // cv::cvtColor(cur_img, cur_img, CV_BGR2GRAY);
        // file.write((const char*)(cur_img.data), cur_img.elemSize() * cur_img.total());
        // // Display img
        // imshow("cur_img", cur_img);
        // cv::waitKey(1);
        // std::string img_path = "./images/" + std::to_string(keyframe_id) + ".jpg";
        // cv::imwrite(img_path, cur_img);
    }

    //////////////////////////////////////////////////////////////////////
    // 4. Mappoint
    //////////////////////////////////////////////////////////////////////
    // read keyframe number
    // auto mappoint_ids = reconstruction->MapPointIds();
    auto mappoint_ids = mappoints_set;
    int mappoint_num = mappoint_ids.size();
    WriteBinaryLittleEndian<int>(&file, mappoint_num);
    std::cout << "mappoint_num = " << mappoint_num << std::endl;

    std::unordered_set<image_t> cur_image_ids(image_ids.begin(), image_ids.end());

    for (auto mappoint_id : mappoint_ids) {
        // std::cout << "mappoint_id = " << mappoint_id << std::endl;
        WriteBinaryLittleEndian<int>(&file, mappoint_id);

        bool mappoint_valid = true;
        bool mappoint_bad = false;
        // std::cout << "mappoint_valid = " << mappoint_valid << std::endl;
        // std::cout << "mappoint_bad = " << mappoint_bad << std::endl;
        WriteBinaryLittleEndian<bool>(&file, mappoint_valid);
        WriteBinaryLittleEndian<bool>(&file, mappoint_bad);

        // Save mappoint pose
        auto mappoint = reconstruction->MapPoint(mappoint_id);
        Eigen::Vector3d mappoint_pose;
        mappoint_pose[0] = mappoint.X();
        mappoint_pose[1] = mappoint.Y();
        mappoint_pose[2] = mappoint.Z();
        // std::cout << "mappoint_pose = " << mappoint_pose[0] << " , " << mappoint_pose[1] << " , " << mappoint_pose[2]
        //           << std::endl;
        WriteBinaryLittleEndian<double>(&file, mappoint_pose[0]);
        WriteBinaryLittleEndian<double>(&file, mappoint_pose[1]);
        WriteBinaryLittleEndian<double>(&file, mappoint_pose[2]);

        // TODO:
        int mappoint_visible = 0;
        int mappoint_found = 0;
        WriteBinaryLittleEndian<int>(&file, mappoint_visible);
        WriteBinaryLittleEndian<int>(&file, mappoint_found);

        // Related keyframe id
        int mappoint_refrence_keyframe = mappoint.Track().Elements().begin()->image_id;
        WriteBinaryLittleEndian<int>(&file, mappoint_refrence_keyframe);

        // Get observations
        int observation_num = mappoint.Track().Elements().size();

        std::vector<std::pair<int, int>> cur_observations;
        for (int j = 0; j < observation_num; j++) {
            int frame_id = mappoint.Track().Elements()[j].image_id;
            int point_idx = mappoint.Track().Elements()[j].point2D_idx;

            if (!cur_image_ids.count(frame_id)) {
                continue;
            }

            cur_observations.emplace_back(std::make_pair(frame_id, point_idx));
        }

        observation_num = cur_observations.size();
        WriteBinaryLittleEndian<int>(&file, observation_num);

        for (int j = 0; j < observation_num; j++) {
            int frame_id = cur_observations[j].first;
            int point_idx = cur_observations[j].second;
            WriteBinaryLittleEndian<int>(&file, frame_id);
            WriteBinaryLittleEndian<int>(&file, point_idx);
        }
    }

    file.close();
}

void WriteORBMapBinary(std::shared_ptr<Reconstruction> reconstruction, std::vector<image_t> image_ids,
                       std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features,
                       std::vector<char> &vdata, const std::string &path) {
    vdata.clear();
    //////////////////////////////////////////////////////////////////////
    // 1. param
    //////////////////////////////////////////////////////////////////////
    // read map version
    int map_version = 0;  // 0-no img  1-with img
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&map_version),
                 reinterpret_cast<const char *>(&map_version) + sizeof(map_version));
    std::cout << "map version = " << map_version << std::endl;
    int img_width = 1920;
    int img_height = 1080;
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&img_width),
                 reinterpret_cast<const char *>(&img_width) + sizeof(img_width));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&img_height),
                 reinterpret_cast<const char *>(&img_height) + sizeof(img_height));

    double fx = 400;
    double fy = 400;
    double cx = 960;
    double cy = 540;
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&fx), reinterpret_cast<const char *>(&fx) + sizeof(fx));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&fy), reinterpret_cast<const char *>(&fy) + sizeof(fy));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&cx), reinterpret_cast<const char *>(&cx) + sizeof(cx));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&cy), reinterpret_cast<const char *>(&cy) + sizeof(cy));

    std::cout << "img_width = " << img_width << std::endl;
    std::cout << "img_height = " << img_height << std::endl;
    std::cout << "fx = " << fx << std::endl;
    std::cout << "fy = " << fy << std::endl;
    std::cout << "cx = " << cx << std::endl;
    std::cout << "cy = " << cy << std::endl;

    double tic_x = 0;
    double tic_y = 0;
    double tic_z = 0;
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&tic_x),
                 reinterpret_cast<const char *>(&tic_x) + sizeof(tic_x));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&tic_y),
                 reinterpret_cast<const char *>(&tic_y) + sizeof(tic_y));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&tic_z),
                 reinterpret_cast<const char *>(&tic_z) + sizeof(tic_z));

    std::cout << "tic = " << tic_x << " , " << tic_y << " , " << tic_z << std::endl;

    double qic_w = 1;
    double qic_x = 0;
    double qic_y = 0;
    double qic_z = 0;

    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&qic_w),
                 reinterpret_cast<const char *>(&qic_w) + sizeof(qic_w));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&qic_x),
                 reinterpret_cast<const char *>(&qic_x) + sizeof(qic_x));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&qic_y),
                 reinterpret_cast<const char *>(&qic_y) + sizeof(qic_y));
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&qic_z),
                 reinterpret_cast<const char *>(&qic_z) + sizeof(qic_z));

    std::cout << "qic = " << qic_w << " , " << qic_x << " , " << qic_y << ", " << qic_z << std::endl;

    //////////////////////////////////////////////////////////////////////
    // 2. Keyframe entrence
    //////////////////////////////////////////////////////////////////////
    auto rec_image_ids = reconstruction->RegisterImageIds(image_ids);
    int keyframe_entrence_num = rec_image_ids.size();
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keyframe_entrence_num),
                 reinterpret_cast<const char *>(&keyframe_entrence_num) + sizeof(keyframe_entrence_num));
    std::cout << "keyframe_entrence_num = " << keyframe_entrence_num << std::endl;

    for (int i = 0; i < keyframe_entrence_num; i++) {
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&i), reinterpret_cast<const char *>(&i) + sizeof(i));
    }

    //////////////////////////////////////////////////////////////////////
    // 3. Keyframe
    //////////////////////////////////////////////////////////////////////
    // read keyframe number
    int keyframe_num = rec_image_ids.size();
    std::cout << "keyframe_num = " << keyframe_num << std::endl;
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keyframe_num),
                 reinterpret_cast<const char *>(&keyframe_num) + sizeof(keyframe_num));

    // Store all the mappoint ids
    std::unordered_set<mappoint_t> mappoints_set;

    for (int i = 0; i < keyframe_num; i++) {
        int keyframe_id = rec_image_ids[i];
        auto keyframe = features[keyframe_id];
        int keypoint_num = keyframe.first.size();

        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keyframe_id),
                     reinterpret_cast<const char *>(&keyframe_id) + sizeof(keyframe_id));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keypoint_num),
                     reinterpret_cast<const char *>(&keypoint_num) + sizeof(keypoint_num));

        // std::cout << "keyframe_id = " << keyframe_id << std::endl;
        // std::cout << "keypoint_num = " << keypoint_num << std::endl;

        for (int j = 0; j < keypoint_num; j++) {
            auto keypoint = keyframe.first[j];
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keypoint.pt.x),
                         reinterpret_cast<const char *>(&keypoint.pt.x) + sizeof(keypoint.pt.x));
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keypoint.pt.y),
                         reinterpret_cast<const char *>(&keypoint.pt.y) + sizeof(keypoint.pt.y));
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keypoint.size),
                         reinterpret_cast<const char *>(&keypoint.size) + sizeof(keypoint.size));
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keypoint.angle),
                         reinterpret_cast<const char *>(&keypoint.angle) + sizeof(keypoint.angle));
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keypoint.response),
                         reinterpret_cast<const char *>(&keypoint.response) + sizeof(keypoint.response));
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keypoint.octave),
                         reinterpret_cast<const char *>(&keypoint.octave) + sizeof(keypoint.octave));
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&keypoint.class_id),
                         reinterpret_cast<const char *>(&keypoint.class_id) + sizeof(keypoint.class_id));

            // std::cout << "keypoint.pt.x = " << keypoint.pt.x << std::endl;
            // std::cout << "keypoint.pt.y = " << keypoint.pt.y << std::endl;
            // std::cout << "keypoint.size = " << keypoint.size << std::endl;
            // std::cout << "keypoint.angle = " << keypoint.angle << std::endl;
            // std::cout << "keypoint.response = " << keypoint.response << std::endl;
            // std::cout << "keypoint.octave = " << keypoint.octave << std::endl;
            // std::cout << "keypoint.class_id = " << keypoint.class_id << std::endl;
        }

        // Save descriptor
        // std::cout << "Write cv mat" << std::endl;
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(keyframe.second.data),
                     reinterpret_cast<const char *>(keyframe.second.data) +
                         keyframe.second.elemSize() * keyframe.second.total());
        // std::cout << "Finish write cv mat " << std::endl;

        // Get current image from reconstruction
        auto rec_keyframe = reconstruction->Image(keyframe_id);

        // Read Keyframe pose
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&rec_keyframe.Tvec()[0]),
                     reinterpret_cast<const char *>(&rec_keyframe.Tvec()[0]) + sizeof(rec_keyframe.Tvec()[0]));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&rec_keyframe.Tvec()[1]),
                     reinterpret_cast<const char *>(&rec_keyframe.Tvec()[1]) + sizeof(rec_keyframe.Tvec()[1]));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&rec_keyframe.Tvec()[2]),
                     reinterpret_cast<const char *>(&rec_keyframe.Tvec()[2]) + sizeof(rec_keyframe.Tvec()[2]));

        // std::cout << "Pose -- translation = " << rec_keyframe.Tvec()[0] << " , " << rec_keyframe.Tvec()[1]
        //           << " , " << rec_keyframe.Tvec()[2] << std::endl;

        double w, x, y, z;
        w = rec_keyframe.Qvec()[0];
        x = rec_keyframe.Qvec()[1];
        y = rec_keyframe.Qvec()[2];
        z = rec_keyframe.Qvec()[3];

        // std::cout << "Pose -- rotation = " << w << " , " << x << " , " << y << " , " << z << std::endl;
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&w), reinterpret_cast<const char *>(&w) + sizeof(w));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&x), reinterpret_cast<const char *>(&x) + sizeof(x));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&y), reinterpret_cast<const char *>(&y) + sizeof(y));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&z), reinterpret_cast<const char *>(&z) + sizeof(z));

        const auto &all_point2ds = rec_keyframe.Points2D();
        for (const auto &point_2d : all_point2ds) {
            if (point_2d.HasMapPoint()) {
                mappoints_set.insert(point_2d.MapPointId());
            }
        }

        // FIXME: DO not need. Load Image
        // auto image_name = feature_data_container->GetImage(keyframe_id).Name();
        // cv::Mat cur_img = cv::imread(image_path + "/" + image_name);
        // cv::cvtColor(cur_img, cur_img, CV_BGR2GRAY);
        // file.write((const char*)(cur_img.data), cur_img.elemSize() * cur_img.total());
        // // Display img
        // imshow("cur_img", cur_img);
        // cv::waitKey(1);
        // std::string img_path = "./images/" + std::to_string(keyframe_id) + ".jpg";
        // cv::imwrite(img_path, cur_img);
    }

    //////////////////////////////////////////////////////////////////////
    // 4. Mappoint
    //////////////////////////////////////////////////////////////////////
    // read keyframe number
    // auto mappoint_ids = reconstruction->MapPointIds();
    auto mappoint_ids = mappoints_set;
    int mappoint_num = mappoint_ids.size();
    vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_num),
                 reinterpret_cast<const char *>(&mappoint_num) + sizeof(int));
    std::cout << "mappoint_num = " << mappoint_num << std::endl;

    std::unordered_set<image_t> cur_image_ids(image_ids.begin(), image_ids.end());

    for (auto mappoint_id : mappoint_ids) {
        // std::cout << "mappoint_id = " << mappoint_id << std::endl;
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_id),
                     reinterpret_cast<const char *>(&mappoint_id) + sizeof(int));

        bool mappoint_valid = true;
        bool mappoint_bad = false;
        // std::cout << "mappoint_valid = " << mappoint_valid << std::endl;
        // std::cout << "mappoint_bad = " << mappoint_bad << std::endl;
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_valid),
                     reinterpret_cast<const char *>(&mappoint_valid) + sizeof(bool));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_bad),
                     reinterpret_cast<const char *>(&mappoint_bad) + sizeof(bool));

        // Save mappoint pose
        auto mappoint = reconstruction->MapPoint(mappoint_id);
        Eigen::Vector3d mappoint_pose;
        mappoint_pose[0] = mappoint.X();
        mappoint_pose[1] = mappoint.Y();
        mappoint_pose[2] = mappoint.Z();
        // std::cout << "mappoint_pose = " << mappoint_pose[0] << " , " << mappoint_pose[1] << " , " << mappoint_pose[2]
        //           << std::endl;
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_pose[0]),
                     reinterpret_cast<const char *>(&mappoint_pose[0]) + sizeof(double));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_pose[1]),
                     reinterpret_cast<const char *>(&mappoint_pose[1]) + sizeof(double));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_pose[2]),
                     reinterpret_cast<const char *>(&mappoint_pose[2]) + sizeof(double));

        // TODO:
        int mappoint_visible = 0;
        int mappoint_found = 0;
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_visible),
                     reinterpret_cast<const char *>(&mappoint_visible) + sizeof(int));
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_found),
                     reinterpret_cast<const char *>(&mappoint_found) + sizeof(int));

        // Related keyframe id
        int mappoint_refrence_keyframe = mappoint.Track().Elements().begin()->image_id;
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&mappoint_refrence_keyframe),
                     reinterpret_cast<const char *>(&mappoint_refrence_keyframe) + sizeof(int));

        // Get observations
        int observation_num = mappoint.Track().Elements().size();

        std::vector<std::pair<int, int>> cur_observations;
        for (int j = 0; j < observation_num; j++) {
            int frame_id = mappoint.Track().Elements()[j].image_id;
            int point_idx = mappoint.Track().Elements()[j].point2D_idx;

            if (!cur_image_ids.count(frame_id)) {
                continue;
            }

            cur_observations.emplace_back(std::make_pair(frame_id, point_idx));
        }

        observation_num = cur_observations.size();
        vdata.insert(vdata.end(), reinterpret_cast<const char *>(&observation_num),
                     reinterpret_cast<const char *>(&observation_num) + sizeof(int));

        for (int j = 0; j < observation_num; j++) {
            int frame_id = cur_observations[j].first;
            int point_idx = cur_observations[j].second;
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&frame_id),
                         reinterpret_cast<const char *>(&frame_id) + sizeof(int));
            vdata.insert(vdata.end(), reinterpret_cast<const char *>(&point_idx),
                         reinterpret_cast<const char *>(&point_idx) + sizeof(int));
        }
    }

    // FIXME: Tmp code for image id to image name written
    // std::string out_log_file_path = "/home/SENSETIME/jiaofei/data2/Backup/image_id.txt";
    std::string out_log_file_path = JoinPaths(workspace_path, "/image_id.txt");
    std::ofstream file(out_log_file_path, std::ios::trunc);
    CHECK(file.is_open()) << out_log_file_path;
    for (const auto &rec_image_id : rec_image_ids) {
        auto cur_image_name = reconstruction->Image(rec_image_id).Name();
        file << rec_image_id << " " << cur_image_name << std::endl;
    }
    file.close();
}

void ExportORBMap(std::shared_ptr<Reconstruction> reconstruction) {
    std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> orb_feature;
    // Read ORB feature
    ReadORBFeatureBinary(orb_feature, JoinPaths(workspace_path, "/orb_feature.bin"));

    // Get all the image ids
    auto image_ids = reconstruction->RegisterImageIds();

    // Write ORB Map
    WriteORBMapBinary(reconstruction, image_ids, orb_feature, JoinPaths(workspace_path, "/Out_Map.txt"));
}

void ExportORBMap(std::shared_ptr<Reconstruction> reconstruction, std::vector<image_t> image_ids) {
    std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> orb_feature;
    // Read ORB feature
    ReadORBFeatureBinary(orb_feature, JoinPaths(workspace_path, "/orb_feature.bin"));

    // Write ORB Map
    // WriteORBMapBinary(reconstruction, image_ids, orb_feature, JoinPaths(workspace_path, "/Out_Map.txt"));
    std::vector<char> buffer_data;
    WriteORBMapBinary(reconstruction, image_ids, orb_feature, buffer_data, JoinPaths(workspace_path, "/Out_Map.txt"));

    char *buffer;
    buffer = new char[buffer_data.size()];
    std::copy(buffer_data.begin(), buffer_data.end(), buffer);

    std::cout << "Save file" << std::endl;
    std::ofstream outfile(JoinPaths(workspace_path, "/Out_Map.txt"), std::ofstream::binary);
    outfile.write(buffer, buffer_data.size());
}

// Remove all the point2d which do not has mappoint
void RemoveUselessPoint2D(std::shared_ptr<Reconstruction> reconstruction_in,
                          std::shared_ptr<Reconstruction> reconstruction_out,
                          const std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features_in,
                          std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features_out) {
    std::unordered_map<image_t, std::unordered_map<point2D_t, point2D_t>> update_point2d_map;

    ///////////////////////////////////////////////////////////////////////////////////////
    // 1. Set camera bin
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "1. Set camera bin ... " << std::endl;
    for (const auto& cur_camera : reconstruction_in->Cameras()) {
        reconstruction_out->AddCamera(cur_camera.second);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 2. Update feature, remove all the unresgisted image and point2d
    ///////////////////////////////////////////////////////////////////////////////////////
    // Reset feature_out
    std::cout << "2. Update feature, remove all the unresgisted image and point2d ... " << std::endl;
    features_out.clear();
    size_t bad_image_counter = 0;
    for (const auto &feature_in : features_in) {
        const auto image_id_in = feature_in.first;
        const auto &image_feature = feature_in.second;

        if (!reconstruction_in->ExistsImage(image_id_in)) {
            //            std::cout << " Image not exist in reconstruction ... " << std::endl;
            continue;
        }

        // Check the image id is registed or not
        if (!reconstruction_in->IsImageRegistered(image_id_in)) {
            std::cout << " Image not registed ... " << std::endl;
            continue;
        }
        // Get current image
        const auto cur_image = reconstruction_in->Image(image_id_in);
        const auto old_point2ds = cur_image.Points2D();

        // Set old Keypoint and descriptor
        const auto &old_keypoints = image_feature.first;
        const auto &old_descriptors = image_feature.second;

        // Create new Keypoint and descriptor
        std::vector<cv::KeyPoint> new_keypoints;
        cv::Mat new_descriptors;
        point2D_t new_point_id = 0;
        for (point2D_t point_id = 0; point_id < image_feature.first.size(); point_id++) {
            // Skip all the point2d do not has mappoint
            if (!old_point2ds[point_id].HasMapPoint()) {
                continue;
            }

            new_keypoints.emplace_back(old_keypoints[point_id]);
            update_point2d_map[image_id_in][point_id] = new_point_id;
            new_point_id++;
        }

        // Update new descriptor
        new_descriptors.create(new_keypoints.size(), 32, CV_8U);
        for (point2D_t point_id = 0; point_id < image_feature.first.size(); point_id++) {
            if (update_point2d_map[image_id_in].count(point_id)) {
                old_descriptors.row(point_id).copyTo(new_descriptors.row(update_point2d_map[image_id_in][point_id]));
            }
        }

        if (new_keypoints.empty()) {
            // std::cout << "Find image has not mappoint..." << std::endl;
            bad_image_counter++;
            update_point2d_map.erase(image_id_in);
            continue;
        }
        features_out[image_id_in] = std::make_pair(new_keypoints, new_descriptors);
    }

    std::cout << " Image number = " << features_in.size() << " ,  bad image number = " << bad_image_counter
              << std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////
    // 3. Update Point2d
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "3. Update Point2d ... " << std::endl;
    const auto &image_ids = reconstruction_in->RegisterImageIds();
    for (const auto &image_id : image_ids) {
        const auto cur_image = reconstruction_in->Image(image_id);
        // Get all the old point2ds
        const auto old_point2ds = cur_image.Points2D();
        std::vector<class Point2D> new_point2Ds;  // Store new point2d

        for (point2D_t point_id = 0; point_id < old_point2ds.size(); point_id++) {
            if (!old_point2ds[point_id].HasMapPoint()) {
                continue;
            }

            class Point2D new_point2D;
            new_point2D.SetXY(old_point2ds[point_id].XY());
            auto point2d_index = update_point2d_map[image_id][point_id];
            new_point2Ds.emplace_back(new_point2D);
        }

        if (new_point2Ds.empty()) {
            continue;
        }

        Image new_image;
        // Update image id
        new_image.SetImageId(cur_image.ImageId());
        new_image.SetCameraId(cur_image.CameraId());

        // Update the camera rotation
        new_image.SetQvec(cur_image.Qvec());
        new_image.SetTvec(cur_image.Tvec());

        new_image.SetName(cur_image.Name());
        new_image.SetPoints2D(new_point2Ds);

        // Update reconstruction
        reconstruction_out->AddImage(new_image);
        reconstruction_out->RegisterImage(new_image.ImageId());
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 4. Update 3d point track id
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "4. Update 3d point track id ... " << std::endl;
    const auto &mappoint_ids = reconstruction_in->MapPointIds();
    for (const auto &mappoint_id : mappoint_ids) {
        class MapPoint new_mappoint;
        // Get old mappoint
        const auto old_mappoint = reconstruction_in->MapPoint(mappoint_id);

        // Use the old mappoint position
        new_mappoint.SetXYZ(old_mappoint.XYZ());
        new_mappoint.SetColor(old_mappoint.Color());
        new_mappoint.SetError(old_mappoint.Error());

        // Update the old mappoint track with new image id and point2d id
        class Track new_track;
        for (const auto &track_el : old_mappoint.Track().Elements()) {
            if (!update_point2d_map.count(track_el.image_id)) {
                continue;
            }

            if (!update_point2d_map[track_el.image_id].count(track_el.point2D_idx)) {
                continue;
            }
            auto new_point2d_id = update_point2d_map[track_el.image_id][track_el.point2D_idx];

            // std::cout << "Has mappoint ? " <<
            // update_reconstruction->Image(new_image_to_point.first).Point2D(new_image_to_point.second).HasMapPoint()
            // << " image id = " <<new_image_to_point.first << " , point id = " << new_image_to_point.second <<
            // std::endl;
            new_track.AddElement(track_el.image_id, new_point2d_id);
        }
        new_mappoint.SetTrack(new_track);

        reconstruction_out->AddMapPointWithError(new_mappoint.XYZ(), std::move(new_mappoint.Track()), new_mappoint.Color(),
                                                 new_mappoint.Error());
    }
}

void ExportUpdateReconstruction(std::shared_ptr<Reconstruction> reconstruction) {
    std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> orb_feature_in, orb_feature_out;
    auto reconstruction_out = std::make_shared<Reconstruction>();

    // Read ORB feature
    ReadORBFeatureBinary(orb_feature_in, JoinPaths(workspace_path, "/orb_feature.bin"));
    RemoveUselessPoint2D(reconstruction, reconstruction_out, orb_feature_in, orb_feature_out);

    WriteORBFeatureBinary(orb_feature_out, JoinPaths(workspace_path, "/orb_feature_new.bin"));
    if (boost::filesystem::exists(workspace_path + "/update/")) {
        boost::filesystem::remove_all(workspace_path + "/update/");
    }
    boost::filesystem::create_directories(workspace_path + "/update/");
    reconstruction_out->WriteReconstruction(JoinPaths(workspace_path, "/update/"), true);
}

void SimplifiedORBMap(std::shared_ptr<Reconstruction> reconstruction, double ratio) {
    std::unordered_map<image_t, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> orb_feature_in, orb_feature_out;
    auto reconstruction_out = std::make_shared<Reconstruction>();

    std::cout << "Load ORB Feature data ... " << std::endl;
    // Read ORB feature
    ReadORBFeatureBinary(orb_feature_in, JoinPaths(workspace_path, "/orb_feature.bin"));

    // Simplified Reconstruction
    int num_mappoint_to_keep = reconstruction->NumMapPoints() * ratio;  //! FIXME:
    SamplerBase::Ptr sampler = createSampler(SamplerBase::Type::kHeuristic);
    std::unordered_set<mappoint_t> mappoints_to_keep;
    std::cout << "Start Reconstruction simplified..." << std::endl;
    sampler->sample(*reconstruction.get(), orb_feature_in, num_mappoint_to_keep, mappoints_to_keep);

    std::cout << "Remove " << reconstruction->NumMapPoints() - mappoints_to_keep.size() << " mappoints" << std::endl;
    std::cout << "Keep " << mappoints_to_keep.size() << " mappoints" << std::endl;

    std::cout << "Before mean track length = " << reconstruction->ComputeMeanTrackLength() << std::endl;
    std::cout << "Before mean reprojection error = " << reconstruction->ComputeMeanReprojectionError() << std::endl;
    // Remove 3d mappoint from the reconstruction
    auto all_mappoint = reconstruction->MapPointIds();
    for (auto mappoint_id : all_mappoint) {
        // Skip these mappoint
        if (mappoints_to_keep.count(mappoint_id)) {
            continue;
        }

        // Remove this mappoint
        reconstruction->DeleteMapPoint(mappoint_id);
    }

    RemoveUselessPoint2D(reconstruction, reconstruction_out, orb_feature_in, orb_feature_out);

    std::cout << "After mean track length = " << reconstruction_out->ComputeMeanTrackLength() << std::endl;
    std::cout << "After mean reprojection error = " << reconstruction_out->ComputeMeanReprojectionError() << std::endl;

    WriteORBFeatureBinary(orb_feature_out, JoinPaths(workspace_path, "/orb_feature_new.bin"));
    if (boost::filesystem::exists(workspace_path + "/Simplified/")) {
        boost::filesystem::remove_all(workspace_path + "/Simplified/");
    }
    boost::filesystem::create_directories(workspace_path + "/Simplified/");
    reconstruction_out->WriteReconstruction(JoinPaths(workspace_path, "/Simplified/"), true);
}

int main(int argc, char *argv[]) {
    workspace_path = std::string(argv[1]);
    const int mode = atoi(argv[2]);
    double ratio;
    if (argc == 4) {
        ratio = atof(argv[3]);
    }

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto reconstruction = std::make_shared<Reconstruction>();
    std::cout << "Load ORB Reconstruction ... " << std::endl;
    reconstruction->ReadReconstruction(JoinPaths(workspace_path, "/0/"));

    // std::vector<image_t> test_images = {1,2,3,4,5,6,7};
    // std::vector<image_t> test_images;
    // for (int i = 0; i < 100; i++) {
    //     test_images.push_back(i);
    // }

    if (mode == 0) {
        std::vector<image_t> test_images = reconstruction->RegisterImageIds();
        auto start = std::chrono::high_resolution_clock::now();
        // Export ORB map
        ExportORBMap(reconstruction, test_images);
        auto time =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
                .count() /
            (1e3);
        std::cout << "Save ORB Map cost : " << time << " ms " << std::endl;

    } else if (mode == 1) {
        auto start = std::chrono::high_resolution_clock::now();
        ExportUpdateReconstruction(reconstruction);

        auto time =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
                .count() /
            (1e6);
        std::cout << "Save ORB Map cost : " << time << " s " << std::endl;
    } else if (mode == 2) {  // -- Simplify ORB Map
        std::cout << "ratio = " << ratio << std::endl;
        SimplifiedORBMap(reconstruction, ratio);
    }

    return 0;
}