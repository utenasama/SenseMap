// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "feature/extraction.h"
#include "util/misc.h"

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    std::string img_path(argv[1]);
    std::string workspace_path(argv[2]);

    ImageReaderOptions reader_options;
    reader_options.image_path = img_path;
    reader_options.single_camera = true;
    reader_options.fixed_camera = false;
    reader_options.single_camera_per_folder = true;
    reader_options.camera_model = "SIMPLE_RADIAL";
    // reader_options.camera_params = "2029.668943;1296.000000;864.000000;0.000000"; //street
    // reader_options.camera_params = "3456.000000;3024.000000;2016.000000;0.000000"; //dslr
    // reader_options.camera_params = "1520.000000,960.000000,540.000000,0.000000";//guobo
    // reader_options.camera_params = "1152.000000, 960.000000, 540.000000,0.000000";//Church

    FeatureDataContainer data_container;

    SiftExtractionOptions sift_extraction;
    sift_extraction.num_threads = -1;
    sift_extraction.use_gpu = true;
    sift_extraction.peak_threshold = 0.005 / sift_extraction.octave_resolution;

    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, &data_container);
    feature_extractor.Start();
    feature_extractor.Wait();

    /*//Singe thread version
    SingleSiftFeatureExtractor single_feature_extractor(reader_options,
                                                        &data_container);
    single_feature_extractor.Run();*/

    data_container.WriteCameras(workspace_path + "/cameras.txt");
    // data_container.WriteImagesData(workspace_path + "/features.txt");
    data_container.WriteImagesBinaryData(workspace_path + "/features.bin");

    //////////////////////////////////////////////////////////////////////////////
    // Test:
    auto camera = data_container.GetCamera(1);
    std::cout << StringPrintf("  Dimensions:      %d x %d", camera.Width(), camera.Height()) << std::endl;
    std::cout << StringPrintf("  Camera:          #%d - %s", camera.CameraId(), camera.ModelName().c_str())
              << std::endl;
    std::cout << StringPrintf("  Focal Length:    %.2fpx", camera.MeanFocalLength()) << std::endl;

    auto ids = data_container.GetImageIds();
    for (auto id : ids) {
        std::cout << id << std::endl;
        auto image = data_container.GetImage(id);

        const std::string input_image_path = JoinPaths(data_container.GetImagePath(), image.Name());
        Bitmap bitmap;
        if (bitmap.Read(input_image_path)) {
            std::cout << bitmap.Width() << " " << bitmap.Height() << std::endl;
        }
    }
    //////////////////////////////////////////////////////////////////////////////

    return 0;
}