// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "util/misc.h"

using namespace sensemap;

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }


int main(int argc, char *argv[])
{
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    std::string workspace_path(argv[1]);
    std::string image_path = argv[2];
    FeatureDataContainer data_container;

    data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
    if (dirExists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
        data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
    }
    data_container.ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path + "/features.bin"));
    // Read AprilTag file data
    data_container.ReadAprilTagBinaryData(JoinPaths(workspace_path + "/apriltags.bin"));

    //////////////////////////////////////////////////////////////////////////////
    // Test:
    auto camera = data_container.GetCamera(1);
    std::cout << StringPrintf("  Dimensions:      %d x %d", camera.Width(), camera.Height()) << std::endl;
    std::cout << StringPrintf("  Camera:          #%d - %s", camera.CameraId(), camera.ModelName().c_str())
              << std::endl;
    std::cout << StringPrintf("  Focal Length:    %.2fpx", camera.MeanFocalLength()) << std::endl;

    auto image_ids = data_container.GetImageIds();
    int counter = 0;
    for (auto image_id : image_ids)
    {
        auto detections = data_container.GetAprilTagDetections(image_id);
        if (detections.empty())
        {
            continue;
        }
        std::cout << "Image id = " << image_id << std::endl;
        
        auto image = data_container.GetImage(image_id);
        std::cout << "Image Name = " << image.Name() << std::endl;
        std::string input_image_path = JoinPaths(image_path, image.Name());
        //         Bitmap bitmap;
        //         if (bitmap.Read(input_image_path)) {
        // //            std::cout << bitmap.Width() << " " << bitmap.Height() << std::endl;
        //         }
        ////////////////////////////////////////////////////////////////////////////
        //draw keypoints
        cv::Mat mat = cv::imread(input_image_path);
        std::string cam1_path = input_image_path;

        cv::Mat mat_1;
        if (camera.NumLocalCameras() > 1) {
            cam1_path = cam1_path.replace(cam1_path.find("cam0"), 4, "cam1");
            std::cout << "cam1_path = " << cam1_path << std::endl;
            mat_1 = cv::imread(cam1_path);
        }
        
        // Check image exist
        if (!mat.data)
        { // Check for invalid input
            std::cout << "Could not open or find the image, path = " << input_image_path << std::endl;
            continue;
        }

        // auto keypoints = data_container.GetKeypoints(image_id);
        // std::vector<cv::KeyPoint> keypoints_show;
        // for (auto keypoint : keypoints) {
        //     keypoints_show.emplace_back(keypoint.x, keypoint.y, keypoint.ComputeScale(), keypoint.ComputeOrientation());
        // }
        // drawKeypoints(mat, keypoints_show, mat, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        //draw AprilTags
        //    auto detections = data_container.GetAprilTagDetections(image_id);
        std::cout << "Detection number = " << detections.size() << std::endl;
        for (auto tag_detection : detections) {
            auto p = tag_detection.p;
            std::cout << "tag_detection.local_camera_id = " << tag_detection.local_camera_id << std::endl;
            if (tag_detection.local_camera_id == 0 || tag_detection.local_camera_id == -1) {
                
                // Draw points
                cv::circle(mat, cv::Point2f(p[0].first, p[0].second), 4, cv::Scalar(255, 0, 0), -1);
                cv::circle(mat, cv::Point2f(p[1].first, p[1].second), 4, cv::Scalar(0, 255, 0), -1);
                cv::circle(mat, cv::Point2f(p[2].first, p[2].second), 4, cv::Scalar(0, 0, 255), -1);
                cv::circle(mat, cv::Point2f(p[3].first, p[3].second), 4, cv::Scalar(255, 255, 0), -1);

                // Draw edge
                cv::line(mat, cv::Point2f(p[0].first, p[0].second), cv::Point2f(p[1].first, p[1].second),
                        cv::Scalar(33, 33, 133), 2);
                cv::line(mat, cv::Point2f(p[1].first, p[1].second), cv::Point2f(p[2].first, p[2].second),
                        cv::Scalar(33, 33, 133), 2);
                cv::line(mat, cv::Point2f(p[2].first, p[2].second), cv::Point2f(p[3].first, p[3].second),
                        cv::Scalar(33, 33, 133), 2);
                cv::line(mat, cv::Point2f(p[3].first, p[3].second), cv::Point2f(p[0].first, p[0].second),
                        cv::Scalar(33, 33, 133), 2);

                std::cout << "p[0] = " << p[0].first << " , " << p[0].second << std::endl;
                std::cout << "p[1] = " << p[1].first << " , " << p[1].second << std::endl;
                std::cout << "p[2] = " << p[2].first << " , " << p[2].second << std::endl;
                std::cout << "p[3] = " << p[3].first << " , " << p[3].second << std::endl;


                cv::putText(mat,                         //target image
                            "Detection Id = "+std::to_string(tag_detection.id),            //text
                            cv::Point(10, mat.rows / 2), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            CV_RGB(118, 185, 0), //font color
                            2);
            } else {
                // Draw points
                cv::circle(mat_1, cv::Point2f(p[0].first, p[0].second), 4, cv::Scalar(255, 0, 0), -1);
                cv::circle(mat_1, cv::Point2f(p[1].first, p[1].second), 4, cv::Scalar(0, 255, 0), -1);
                cv::circle(mat_1, cv::Point2f(p[2].first, p[2].second), 4, cv::Scalar(0, 0, 255), -1);
                cv::circle(mat_1, cv::Point2f(p[3].first, p[3].second), 4, cv::Scalar(255, 255, 0), -1);

                // Draw edge
                cv::line(mat_1, cv::Point2f(p[0].first, p[0].second), cv::Point2f(p[1].first, p[1].second),
                        cv::Scalar(33, 33, 133), 2);
                cv::line(mat_1, cv::Point2f(p[1].first, p[1].second), cv::Point2f(p[2].first, p[2].second),
                        cv::Scalar(33, 33, 133), 2);
                cv::line(mat_1, cv::Point2f(p[2].first, p[2].second), cv::Point2f(p[3].first, p[3].second),
                        cv::Scalar(33, 33, 133), 2);
                cv::line(mat_1, cv::Point2f(p[3].first, p[3].second), cv::Point2f(p[0].first, p[0].second),
                        cv::Scalar(33, 33, 133), 2);

                cv::putText(mat_1,                         //target image
                            "Detection Id = "+std::to_string(tag_detection.id),             //text
                            cv::Point(10, mat_1.rows / 2), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            CV_RGB(118, 185, 0), //font color
                            2);
            }
            
        }

        const std::string ouput_image_path = JoinPaths(workspace_path + "/features", std::to_string(counter) + "_cam0.jpg");
        cv::imwrite(ouput_image_path, mat);

        if (camera.NumLocalCameras() > 1) {
            const std::string ouput_image_path_1 = JoinPaths(workspace_path + "/features", std::to_string(counter) + "_cam1.jpg");
            cv::imwrite(ouput_image_path_1, mat_1);
        }

        // cv::resize(mat, mat, cv::Size(round(0.25 * mat.cols), round(0.25 * mat.rows)), 0, 0, cv::INTER_AREA);
        // cv::imshow("test", mat);
        // cv::waitKey(0);
        counter++;
    }

    return 0;
}