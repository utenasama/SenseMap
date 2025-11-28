// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <Eigen/Eigen>
#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>
#include <boost/filesystem.hpp>
#include <ethz_apriltag2/include/apriltags/TagDetection.h>
#include <ethz_apriltag2/include/apriltags/TagDetector.h>
#include <ethz_apriltag2/include/apriltags/TagFamily.h>
#include <ethz_apriltag2/include/apriltags/Tag16h5.h>
#include <ethz_apriltag2/include/apriltags/Tag25h7.h>
#include <ethz_apriltag2/include/apriltags/Tag25h9.h>
#include <ethz_apriltag2/include/apriltags/Tag36h9.h>
#include <ethz_apriltag2/include/apriltags/Tag36h11.h>

std::vector<std::string> GetRecursiveFileList(const std::string &path) {
    std::vector<std::string> file_list;
    for (auto it = boost::filesystem::recursive_directory_iterator(path);
         it != boost::filesystem::recursive_directory_iterator(); ++it) {
        if (boost::filesystem::is_regular_file(*it)) {
            const boost::filesystem::path file_path = *it;
            if (file_path.string().substr(file_path.string().length() - 4, file_path.string().length()) != ".txt") {
                file_list.push_back(file_path.string());
            }
        }
    }

    std::sort(file_list.begin(), file_list.end(),
              [](const std::string& s_1, const std::string& s_2) {
                  return s_1 < s_2;
              });
    return file_list;
}

struct TagTracker {
    std::shared_ptr<AprilTags::TagDetector> m_tag_detector;
    AprilTags::TagCodes m_tag_codes;

    bool m_draw;  // draw image and April tag detections?
    int m_width;  // image size in pixels
    int m_height;
    double m_tag_size;  // April tag side length in meters of square black frame
    double m_fx;        // camera focal length in pixels
    double m_fy;
    double m_px;  // camera principal point
    double m_py;
    cv::Mat m_curr_image;
    double m_curr_timestamp{};
    std::map<int, std::vector<cv::Point3d>> m_corner_pos;
    int m_frame_cnt{0};
    std::vector<std::string> image_list;

public:
    // default constructor
    TagTracker()
            :  // default settings, most can be modified through command line
    // options (see below)
            m_tag_codes(AprilTags::tagCodes36h11),
            m_draw(true),
            m_width(640),
            m_height(480),
            m_tag_size(0.04125),
            m_fx(600),
            m_fy(600),
            m_px(m_width / 2),
            m_py(m_height / 2) {
    }

    // changing the tag family
    void set_tag_code(string s) {
        if (s == "16h5") {
            m_tag_codes = AprilTags::tagCodes16h5;
        } else if (s == "25h7") {
            m_tag_codes = AprilTags::tagCodes25h7;
        } else if (s == "25h9") {
            m_tag_codes = AprilTags::tagCodes25h9;
        } else if (s == "36h9") {
            m_tag_codes = AprilTags::tagCodes36h9;
        } else if (s == "36h11") {
            m_tag_codes = AprilTags::tagCodes36h11;
        } else {
            std::cout << "Invalid tag family specified" << endl;
            exit(1);
        }
    }

    void setup(const std::string &image_path) {
        m_tag_detector = std::make_shared<AprilTags::TagDetector>(m_tag_codes);
        // prepare window for drawing the camera images
        if (m_draw) {
            cv::namedWindow("apriltags_TagTracker", 1);
        }
        image_list = GetRecursiveFileList(image_path);
    }

    // The processing loop where images are retrieved, tags detected,
    // and information about detections generated
    void loop() {
        cv::Mat image;
        cv::Mat image_gray;
        cv::Mat map1, map2;

        for (auto query_image_path : image_list) {
            // capture frame
            cv::Mat image_raw = cv::imread(query_image_path);
            if (image_raw.empty()) {
                std::cerr << "Cannot read " << query_image_path << "\n";
                continue;
            } else {
                // dirty fix unknown intrinsics
                m_width = image_raw.cols;
                m_height = image_raw.rows;
            }
            image = image_raw.clone();

            m_curr_image = image.clone();

            // detect April tags (requires a gray scale image)
            if (image.empty()) break;
            cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
            std::vector<AprilTags::TagDetection> detections = m_tag_detector->extractTags(image_gray);

            // print out each detection
            std::cout << detections.size() << " tags detected:" << endl;
            if (detections.size() >= 1){
                std::cout << "Id = " << detections[0].id << std::endl;
                std::cout << "Coordinate = \n 0: " << detections[0].p[0].first  << " , " << detections[0].p[0].second
                          << "\n 1:" << detections[0].p[0].first  << " , " << detections[0].p[0].second
                          << "\n 2:" << detections[0].p[1].first  << " , " << detections[0].p[1].second
                          << "\n 2:" << detections[0].p[2].first  << " , " << detections[0].p[2].second << std::endl;
            }


            // calculate_pose(detections);
            // show the current image including any detections
            if (m_draw) {
                for (auto &detection : detections) {
                    // also highlight in the image
                    detection.draw(image);
                }
                imshow("apriltag_TagTracker", image);  // OpenCV call
            }

            // print out the frame rate at which image frames are being
            // processed
            m_frame_cnt++;

            // exit if any key is pressed
            if (cv::waitKey(1) >= 0) break;
        }
    }
};  // TagTracker

// always use tagCodes36h11 in vibration
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage ./test_apriltag image_path \n";
    }
    const std::string TAG_CODE = "36h11";

    TagTracker tag_tracker;
    tag_tracker.set_tag_code(TAG_CODE);

    // setup image source, window for drawing, serial port...
    tag_tracker.setup(argv[1]);

    // the actual processing loop where tags are detected and visualized
    tag_tracker.loop();

    return 0;
}
