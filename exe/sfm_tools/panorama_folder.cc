// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <chrono>

#include "util/panorama.h"
#include "util/misc.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;
using namespace cv;
using namespace sensemap;

string img_input_path;
string img_output_path;

//#define display_image

int main(int argc, char** argv) {
    //---------------------------
    cout << "Panorama Converter Folder" << endl;

    int image_count, image_width, image_height;
    double fov_vertical;

    if (argc > 6) {
        img_input_path = argv[1];
        img_output_path = argv[2];
        image_count = atoi(argv[3]);
        fov_vertical = strtod(argv[4], nullptr);
        image_width = atoi(argv[5]);
        image_height = atoi(argv[6]);
    } else {
        cout << "Input arg number error" << endl;
        return -1;
    }

    // Remove the last '/' in the input path and output path
    if (img_input_path.substr(img_input_path.size() - 1) == "/") {
        img_input_path = img_input_path.substr(0, img_input_path.size() - 1);
    }

    if (img_output_path.substr(img_output_path.size() - 1) == "/") {
        img_output_path = img_output_path.substr(0, img_output_path.size() - 1);
    }

    // Load image folder list
    std::cout << "Get image file list ..." << std::endl;
    auto image_list = GetRecursiveFileList(img_input_path);

    Panorama panorama;

    // Process each image
    size_t image_counter = 0;
    // double all_time = 0;
    // double all_save_time = 0;

    auto image_path = image_list[0];

    // Load panorama images
    Bitmap img_input_test;
    if (!img_input_test.Read(image_path, true)) {
        std::cerr << "seg image read fail. " << std::endl;
    }

    // Get the input image size
    int panorama_width = img_input_test.Width();
    int panorama_height = img_input_test.Height();
    panorama.PerspectiveParamsProcess(image_width, image_height, image_count, fov_vertical, panorama_width,
                                      panorama_height);

    auto start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < image_list.size(); i++) {
        auto image_path = image_list[i];

        // Load panorama images
        Bitmap img_input;
        if (!img_input.Read(image_path, true)) {
            std::cerr << "seg image read fail. " << std::endl;
            continue;
        }

#ifdef display_image
        // Display the original image
         cv::Mat image_input_mat = cv::imread(image_path);
         cv::imshow("Original Image", image_input_mat);
         cvWaitKey(0);
#endif

        // auto start = std::chrono::high_resolution_clock::now();
        // Process the original image from panorama to perspective projection
        std::vector<Bitmap> img_outs;
        panorama.PanoramaToPerspectives(&img_input, img_outs);

        std::string image_name = image_path;
        image_name = image_name.substr(img_input_path.size(), image_name.size() - img_input_path.size() - 4);
        std::string image_folder_name = image_name;
        image_folder_name = image_name.substr(0, image_name.rfind('/'));

        // cout << "image_name = " <<image_name  << std::endl;
        std::string result_image_folder_path = img_output_path + "/" + image_folder_name;
        std::string result_image_path = img_output_path + "/" + image_name;
        if (boost::filesystem::exists(result_image_folder_path)) {
            // boost::filesystem::remove_all(result_image_folder_path);
        } else {
            boost::filesystem::create_directories(result_image_folder_path);
        }

        std::string img_path = result_image_path + "_";
        int counter = 0;
        // start = std::chrono::high_resolution_clock::now();
        for (auto img_out : img_outs) {
            std::string cur_img_path = img_path + std::to_string(counter) + ".jpg";
            img_out.Write(cur_img_path, FIF_JPEG);
            counter++;
        }

#ifdef _OPENMP
#pragma omp critical(all_save_time)
#endif
        {
            // all_save_time = all_save_time + std::chrono::duration_cast<std::chrono::microseconds>(
            //                                     std::chrono::high_resolution_clock::now() - start)
            //                                     .count()/ 1e3;

            // Print the progress
            std::cout << "\r";
            std::cout << "Process Images [" << image_counter << " / " << image_list.size() - 1 << "]" << std::flush;

            // std::cout << "Process Images [" << image_counter << " / " << image_list.size() - 1
            //           << ", average convert time: " << all_time / image_counter << "ms"
            //           << " , average save time: " << all_save_time / image_counter << "ms ]" << std::flush;

            image_counter++;
        }
    }
    std::cout << std::endl;

    std::cout
            << "Cost time = "
            << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() /
               60
            << " min" << std::endl;

    return 0;
}
