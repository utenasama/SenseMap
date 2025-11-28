// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstring>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include "feature/utils.h"
#include "opencv2/opencv.hpp"
#include "util/panorama.h"
#include "util/mat2freeimage.h"
#include "util/freeimage2mat.h"

using namespace sensemap;

enum ImageType { Perspective = 0, Panorama = 1 };

std::vector<std::string> GetRecursiveFileList(const std::string& path) {
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
              [](const std::string& s_1, const std::string& s_2) { return s_1 < s_2; });
    return file_list;
}

// changing the tag family
bool set_tag_code(const std::string& tag_string, AprilTags::TagCodes& tag_codes) {
    if (tag_string == "16h5") {
        tag_codes = AprilTags::tagCodes16h5;
    } else if (tag_string == "25h7") {
        tag_codes = AprilTags::tagCodes25h7;
    } else if (tag_string == "25h9") {
        tag_codes = AprilTags::tagCodes25h9;
    } else if (tag_string == "36h9") {
        tag_codes = AprilTags::tagCodes36h9;
    } else if (tag_string == "36h11") {
        tag_codes = AprilTags::tagCodes36h11;
    } else {
        std::cout << "Invalid tag family specified" << endl;
        return false;
    }

    return true;
}

cv::Mat combineImages(vector<cv::Mat> imgs, int col, int row, bool hasMargin) {
    int imgAmount = imgs.size();
    int width = imgs[0].cols;
    int height = imgs[0].rows;
    int newWidth, newHeight;
    if (!hasMargin) {
        newWidth = col * imgs[0].cols;
        newHeight = row * imgs[0].rows;
    } else {
        newWidth = (col + 1) * 20 + col * width;
        newHeight = (row + 1) * 20 + row * height;
    }

    cv::Mat newImage;
    if (imgs[0].channels() == 1) {
        newImage = cv::Mat(newHeight, newWidth, CV_8UC1, cv::Scalar(255, 255, 255));
    } else if (imgs[0].channels() == 3) {
        newImage = cv::Mat(newHeight, newWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    }

    int x, y, imgCount;
    if (hasMargin) {
        imgCount = 0;
        x = 0;
        y = 0;
        while (imgCount < imgAmount) {
            cv::Mat imageROI = newImage(cv::Rect(x * width + (x + 1) * 20, y * height + (y + 1) * 20, width, height));
            imgs[imgCount].copyTo(imageROI);
            imgCount++;
            if (x == (col - 1)) {
                x = 0;
                y++;
            } else {
                x++;
            }
        }
    } else {
        imgCount = 0;
        x = 0;
        y = 0;
        while (imgCount < imgAmount) {
            cv::Mat imageROI = newImage(cv::Rect(x * width, y * height, width, height));
            imgs[imgCount].copyTo(imageROI);
            imgCount++;
            if (x == (col - 1)) {
                x = 0;
                y++;
            } else {
                x++;
            }
        }
    }
    return newImage;
};

// always use tagCodes36h11 in vibration
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage ./test_apriltag 1.image_path 2.out_image_path (Option)3.image_type\n";
        exit(-1);
    }
    std::string image_path = argv[1];
    std::string out_image_path = argv[2];

    int image_type = ImageType::Perspective;
    if (argc > 3) {
        image_type = atoi(argv[3]);
    }

    bool m_draw = false;
    if (argc == 5) {
        m_draw = bool(atoi(argv[4]));
    }
    const std::string TAG_CODE = "36h11";

    // Create tag code and april tag detector
    AprilTags::TagCodes tag_codes(AprilTags::tagCodes36h11);
    std::cout << "Tag code = " << TAG_CODE << std::endl;
    set_tag_code(TAG_CODE, tag_codes);
    std::shared_ptr<AprilTags::TagDetector> detector = std::make_shared<AprilTags::TagDetector>(tag_codes);

    // prepare window for drawing the camera images
    if (m_draw) {
        cv::namedWindow("apriltags_TagTracker", 1);
    }

    std::shared_ptr<sensemap::Panorama> panorama_convert = std::make_shared<sensemap::Panorama>();

    // Get Image list
    auto image_list = GetRecursiveFileList(image_path);
    cv::Mat image;
    cv::Mat image_gray;

    int counter = 0;

    for (int list_id = 0; list_id < image_list.size(); list_id++) {
        const auto& query_image_path = image_list[list_id];

        int img_width, img_height;

        // Get image name
        std::string cur_image_name =
            query_image_path.substr(image_path.size(), query_image_path.size() - image_path.size());
        std::cout << "Image name: " << cur_image_name << std::endl;

        // Get subfolder
        std::string sub_folder_name;
        if (cur_image_name.find('/') == std::string::npos) {
            sub_folder_name = out_image_path;
        } else {
            sub_folder_name = out_image_path + "/" + cur_image_name.substr(0, cur_image_name.rfind('/'));
        }
        std::cout << "Output folder name: " << sub_folder_name << std::endl;

        std::string output_image_name = out_image_path + "/" + cur_image_name;
        std::cout << "Output image name: " << output_image_name << std::endl;

        // detect April tags (requires a gray scale image)
        bool detected_tag = false;
        if (image_type == ImageType::Perspective) {
            // capture frame
            cv::Mat image_raw = cv::imread(query_image_path);
            if (image_raw.empty()) {
                std::cerr << "Cannot read " << query_image_path << "\n";
                continue;
            }
            image = image_raw.clone();
            cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
            std::vector<AprilTags::TagDetection> detections = detector->extractTags(image_gray);

            // print out each detection
            std::cout << detections.size() << " tags detected:" << endl;
            if (!detections.empty()) {
                std::cout << "Id = " << detections[0].id << std::endl;
                std::cout << "Coordinate = \n 0: " << detections[0].p[0].first << " , " << detections[0].p[0].second
                          << "\n 1:" << detections[0].p[0].first << " , " << detections[0].p[0].second
                          << "\n 2:" << detections[0].p[1].first << " , " << detections[0].p[1].second
                          << "\n 2:" << detections[0].p[2].first << " , " << detections[0].p[2].second << std::endl;
                detected_tag = true;
            }

            // show the current image including any detections
            if (m_draw && detected_tag) {
                for (auto& detection : detections) {
                    // also highlight in the image
                    detection.draw(image);
                }
                imshow("apriltag_TagTracker", image);  // OpenCV call
            }
            if (detected_tag) {
                if (!boost::filesystem::exists(sub_folder_name)) {
                    boost::filesystem::create_directories(sub_folder_name);
                }
                // Save the result in the output folder
                imwrite(output_image_name, image_raw);
            }
        } else if (image_type == ImageType::Panorama) {
            // Convert input image into bitmap
            Bitmap bitmap;
            if (!bitmap.Read(query_image_path)) {
                std::cout << "Image read failed ..." << std::endl;
                continue;
            } else {
                // dirty fix unknown intrinsics
                img_width = bitmap.Width();
                img_height = bitmap.Height();
            }

            if (counter == 0) {
                panorama_convert->PerspectiveParamsProcess(800, 800, 8, 60, img_width, img_height);
            }

            std::vector<Bitmap> out_bitmaps;
            panorama_convert->PanoramaToPerspectives(&bitmap, out_bitmaps);

            std::vector<std::vector<AprilTags::TagDetection>> all_detections;
            std::vector<cv::Mat> display_mats;
            for (auto out_bitmap : out_bitmaps) {
                cv::Mat out_mat;
                // Convert bitmap to mat
                FreeImage2Mat(&out_bitmap, out_mat);
                display_mats.emplace_back(out_mat);
                cv::cvtColor(out_mat, out_mat, cv::COLOR_BGR2GRAY);
                std::vector<AprilTags::TagDetection> detections = detector->extractTags(out_mat);

                // print out each detection

                std::cout << detections.size() << " tags detected:" << endl;
                if (!detections.empty()) {
                    std::cout << "Id = " << detections[0].id << std::endl;
                    std::cout << "Coordinate = \n 0: " << detections[0].p[0].first << " , " << detections[0].p[0].second
                              << "\n 1:" << detections[0].p[0].first << " , " << detections[0].p[0].second
                              << "\n 2:" << detections[0].p[1].first << " , " << detections[0].p[1].second
                              << "\n 2:" << detections[0].p[2].first << " , " << detections[0].p[2].second << std::endl;
                    detected_tag = true;
                }

                all_detections.emplace_back(detections);
            }

            if (m_draw && detected_tag) {
                for (int i = 0; i < all_detections.size(); i++) {
                    // Display the detection result
                    for (auto& detection : all_detections[i]) {
                        // also highlight in the image
                        detection.draw(display_mats[i]);
                    }
                }
                cv::Mat draw_image = combineImages(display_mats, 4, 2, false);
                cv::resize(draw_image, draw_image,
                           cv::Size((int)round(draw_image.cols * 0.5), (int)round(draw_image.rows * 0.5)));
                imshow("apriltag_TagTracker", draw_image);  // OpenCV call
            }
            if (detected_tag) {
                if (!boost::filesystem::exists(sub_folder_name)) {
                    boost::filesystem::create_directories(sub_folder_name);
                }
                // Save the result in the output folder
                bitmap.Write(output_image_name);
            }
        }

        // exit if any key is pressed
        counter++;
        //        if (cv::waitKey(1) >= 0) break;
    }

    return 0;
}
