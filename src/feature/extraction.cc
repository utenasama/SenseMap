// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "feature/extraction.h"

#include <numeric>

#include "feature/sift.h"
#include "util/cuda.h"
#include "util/large_fov_image.h"
#include "util/misc.h"
#include "util/string.h"
#include "util/panorama.h"
#include "util/piecewise_image.h"
#include "util/rgbd_helper.h"
#include "utils.h"

#ifdef CUDA_ENABLED
#include "SiftGPU/SiftGPU.h"
#endif

namespace sensemap {

namespace {

void ScaleKeypoints(const Bitmap& bitmap, const Camera& camera, FeatureKeypoints* keypoints) {
    std::cout << "bitmap size: " << bitmap.Width() << " " << bitmap.Height() << std::endl;
    std::cout << "camera size: " << camera.Width() << " " << camera.Height() << std::endl;
    if (static_cast<size_t>(bitmap.Width()) != camera.Width() ||
        static_cast<size_t>(bitmap.Height()) != camera.Height()) {
        const float scale_x = static_cast<float>(camera.Width()) / bitmap.Width();
        const float scale_y = static_cast<float>(camera.Height()) / bitmap.Height();
        for (auto& keypoint : *keypoints) {
            keypoint.Rescale(scale_x, scale_y);
        }
    }
    std::cout << "scale keypoints done" << std::endl;
}

void MaskKeypoints(const Bitmap& mask, FeatureKeypoints* keypoints, FeatureDescriptors* descriptors, 
                   PanoramaIndexs* panoramaidxs, AprilTagDetections* detections) {
    size_t out_index = 0;
    BitmapColor<uint8_t> color;
    size_t org_num_keypoint = keypoints->size();
    for (size_t i = 0; i < keypoints->size(); ++i) {
        if (!mask.GetPixel(static_cast<int>(keypoints->at(i).x), static_cast<int>(keypoints->at(i).y), &color) ||
            color.r == 0) {
            // Delete this keypoint by not copying it to the output.
        } else {
            // Retain this keypoint by copying it to the output index (in case this
            // index differs from its current position).
            if (out_index != i) {
                keypoints->at(out_index) = keypoints->at(i);
                panoramaidxs->at(out_index) = panoramaidxs->at(i);
                for (int col = 0; col < descriptors->cols(); ++col) {
                    (*descriptors)(out_index, col) = (*descriptors)(i, col);
                }
            }
            out_index += 1;
        }
    }

    keypoints->resize(out_index);
    panoramaidxs->resize(out_index);
    descriptors->conservativeResize(out_index, descriptors->cols());
    std::cout << StringPrintf("mask %d/%d keypoints done", out_index, org_num_keypoint) << std::endl;
}

void FilterKeypoints(const Bitmap& bitmap, const Camera& camera, FeatureKeypoints* keypoints, FeatureDescriptors* descriptors, 
                     PanoramaIndexs* panoramaidxs, AprilTagDetections* detections) {
    const size_t width = camera.Width();
    const size_t height = camera.Height();
    size_t out_index = 0;

    for (size_t i = 0; i < keypoints->size(); ++i) {
        int x = static_cast<int>(keypoints->at(i).x);
        int y = static_cast<int>(keypoints->at(i).y);
        if (x < 0 || x >= width || y < 0 || y >= height) {
            continue;
        }
        // Retain this keypoint by copying it to the output index (in case this
        // index differs from its current position).
        if (out_index != i) {
            keypoints->at(out_index) = keypoints->at(i);
            panoramaidxs->at(out_index) = panoramaidxs->at(i);
            for (int col = 0; col < descriptors->cols(); ++col) {
                (*descriptors)(out_index, col) = (*descriptors)(i, col);
            }
        }
        out_index += 1;
    }

    keypoints->resize(out_index);
    panoramaidxs->resize(out_index);
    descriptors->conservativeResize(out_index, descriptors->cols());
    std::cout << "filter keypoints done" << std::endl;
}

void BitmapToMat(const Bitmap& bitmap, cv::Mat& mat) {
    if (bitmap.Channels() == 3) {
        mat.create(bitmap.Height(), bitmap.Width(), CV_8UC3);
    } else {
        mat.create(bitmap.Height(), bitmap.Width(), CV_8UC1);
    }

#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < bitmap.Height(); ++r) {
        for (int c = 0; c < bitmap.Width(); ++c) {
            BitmapColor<uint8_t> color;
            bitmap.GetPixel(c, r, &color);
            cv::Vec3b color_in_mat;
            color_in_mat(0) = color.b;
            color_in_mat(1) = color.g;
            color_in_mat(2) = color.r;
            if (bitmap.Channels() == 3) {
                mat.at<cv::Vec3b>(r, c) = color_in_mat;
            } else {
                mat.at<uint8_t>(r, c) = color_in_mat(2);
            }
        }
    }
}

}  // unnamed namespace

SiftFeatureExtractor::SiftFeatureExtractor(const ImageReaderOptions& reader_options,
                                           const SiftExtractionOptions& sift_options,
                                           FeatureDataContainer* data_container, image_t image_index,
                                           camera_t camera_index, label_t label_index)
    : reader_options_(reader_options),
      sift_options_(sift_options),
      image_reader_(reader_options_, image_index, camera_index, label_index),
      data_container_(data_container) {
    CHECK(reader_options_.Check());
    CHECK(sift_options_.Check());

    data_container_->SetImagePath(reader_options.image_path);
    std::shared_ptr<Bitmap> camera_mask;
    if (!reader_options_.camera_mask_path.empty()) {
        camera_mask = std::shared_ptr<Bitmap>(new Bitmap());
        if (!camera_mask->Read(reader_options_.camera_mask_path,
                               /*as_rgb*/ false)) {
            std::cerr << "  ERROR: Cannot read camera mask file: " << reader_options_.camera_mask_path
                      << ". No mask is going to be used." << std::endl;
            camera_mask.reset();
        }
    }

    const int num_threads = GetEffectiveNumThreads(sift_options_.num_threads);
    CHECK_GT(num_threads, 0);
    std::cout << "num_threads: " << num_threads << std::endl;

    int bitmap_read_num_threads = (GetEffectiveNumThreads(reader_options_.bitmap_read_num_threads) <= num_threads)
                                          ? GetEffectiveNumThreads(reader_options_.bitmap_read_num_threads)
                                          : num_threads;
    std::cout<<"bitmap read num threads: "<<bitmap_read_num_threads<<std::endl;

    // Make sure that we only have limited number of objects in the queue to avoid
    // excess in memory usage since images and features take lots of memory.
    const int kQueueSize = 1;
    bitmap_reader_queue_.reset(new JobQueue<ImageData>(bitmap_read_num_threads));
    resizer_queue_.reset(new JobQueue<ImageData>(bitmap_read_num_threads));
    extractor_queue_.reset(new JobQueue<ImageData>(bitmap_read_num_threads));
    writer_queue_.reset(new JobQueue<ImageData>(kQueueSize));

    if (sift_options_.max_image_size > 0) {
        for (int i = 0; i < num_threads; ++i) {
            resizers_.emplace_back(
                new ImageResizerThread(sift_options_.max_image_size, resizer_queue_.get(), extractor_queue_.get()));
        }
    }
    if(reader_options_.read_image_info_first){

        for (int i = 0; i<bitmap_read_num_threads; ++i){
            if(sift_options_.max_image_size > 0){
                bitmap_readers_.emplace_back(new BitMapReaderThread(bitmap_reader_queue_.get(),resizer_queue_.get()));
            }
            else{
                bitmap_readers_.emplace_back(new BitMapReaderThread(bitmap_reader_queue_.get(),extractor_queue_.get()));
            }
        }        
    }

#ifdef CUDA_ENABLED
    if (!sift_options_.domain_size_pooling && !sift_options_.estimate_affine_shape && sift_options_.use_gpu) {
        std::vector<int> gpu_indices = CSVToVector<int>(sift_options_.gpu_index);
        CHECK_GT(gpu_indices.size(), 0);

        if (gpu_indices.size() == 1 && gpu_indices[0] == -1) {
            const int num_cuda_devices = GetNumCudaDevices();
            CHECK_GT(num_cuda_devices, 0);
            gpu_indices.resize(num_cuda_devices);
            std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
        }
        auto sift_gpu_options = sift_options_;
        for (const auto& gpu_index : gpu_indices) {
            sift_gpu_options.gpu_index = std::to_string(gpu_index);
            extractors_.emplace_back(new SiftFeatureExtractorThread(sift_gpu_options, camera_mask,
                                                                    extractor_queue_.get(), writer_queue_.get()));
        }
    } else {
        auto custom_sift_options = sift_options_;
        custom_sift_options.use_gpu = false;
        for (int i = 0; i < num_threads; ++i) {
            extractors_.emplace_back(new SiftFeatureExtractorThread(custom_sift_options, camera_mask,
                                                                    extractor_queue_.get(), writer_queue_.get()));
        }
    }
#else
    {
        auto custom_sift_options = sift_options_;
        custom_sift_options.use_gpu = false;
        for (int i = 0; i < num_threads; ++i) {
            extractors_.emplace_back(new SiftFeatureExtractorThread(custom_sift_options, camera_mask,
                                                                    extractor_queue_.get(), writer_queue_.get()));
        }
    }
#endif  // CUDA_ENABLED

    writer_.reset(new FeatureWriterThread(image_reader_.NumImages(), data_container_, writer_queue_.get()));
}

void SiftFeatureExtractor::Run() {
    PrintHeading1("Feature extraction");

    for(auto& bitmap_reader: bitmap_readers_){
        bitmap_reader->Start();
    }

    for (auto& resizer : resizers_) {
        resizer->Start();
    }

    for (auto& extractor : extractors_) {
        extractor->Start();
    }

    writer_->Start();

    for (auto& extractor : extractors_) {
        if (!extractor->CheckValidSetup()) {
            return;
        }
    }

    std::cout << "Next index = " << image_reader_.NextIndex() << std::endl;
    std::cout << "Number image = " << image_reader_.NumImages() << std::endl;
    std::cout << "Initial Index = " << image_reader_.InitialIndex() << std::endl;
    double total_read_image_time = 0.0;

    while (image_reader_.NextIndex() - image_reader_.InitialIndex() < image_reader_.NumImages()) {
        if (IsStopped()) {
            bitmap_reader_queue_->Stop();
            resizer_queue_->Stop();
            extractor_queue_->Stop();
            bitmap_reader_queue_->Clear();
            resizer_queue_->Clear();
            extractor_queue_->Clear();
            break;
        }

        ImageData image_data;
        auto& camera = image_data.camera;
        auto& feature_data = image_data.featuredata;
        feature_data.bitmap.resize(reader_options_.num_local_cameras);
        feature_data.mask.resize(reader_options_.num_local_cameras);

        std::chrono::high_resolution_clock::time_point start_time_read_image =
            std::chrono::high_resolution_clock::now();
        image_data.status = image_reader_.Next(&camera, &feature_data.image, &feature_data.bitmap, &feature_data.mask,
                                               &feature_data.bitmap_paths, reader_options_.read_image_info_first);

        std::chrono::high_resolution_clock::time_point end_time_read_image = std::chrono::high_resolution_clock::now();
        std::cout << StringPrintf("=> Read image info Elapsed time: %.3f [ms]",
                                  std::chrono::duration_cast<std::chrono::microseconds>(end_time_read_image -
                                                                                        start_time_read_image)
                                          .count() /
                                      1e3)
                         .c_str()
                  << std::endl;

        total_read_image_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(end_time_read_image - start_time_read_image).count() /
            1e6;

        if (image_data.status != ImageReader::Status::SUCCESS) {
            for (int i = 0; i < reader_options_.num_local_cameras; ++i) {
                feature_data.bitmap[i].Deallocate();
            }
        }
        if(reader_options_.read_image_info_first){
            CHECK(bitmap_reader_queue_->Push(std::move(image_data)));
        }
        else{
            if (sift_options_.max_image_size > 0) {
                CHECK(resizer_queue_->Push(std::move(image_data)));
            } else {
                CHECK(extractor_queue_->Push(std::move(image_data)));
            }
        }
    }
    std::cout <<StringPrintf("=> Read all image info Elapsed time: %.3f [s]",total_read_image_time);

    data_container_->SetGeoImageIndex(image_reader_.GeoImageIndex());

    bitmap_reader_queue_->Wait();
    bitmap_reader_queue_->Stop();
    for(auto& bitmap_reader: bitmap_readers_){
        bitmap_reader->Wait();
    }

    resizer_queue_->Wait();
    resizer_queue_->Stop();
    for (auto& resizer : resizers_) {
        resizer->Wait();
    }

    extractor_queue_->Wait();
    extractor_queue_->Stop();
    for (auto& extractor : extractors_) {
        extractor->Wait();
    }

    writer_queue_->Wait();
    writer_queue_->Stop();
    writer_->Wait();

    GetTimer().PrintMinutes();
}

BitMapReaderThread::BitMapReaderThread(JobQueue<ImageData>* input_queue,
                                       JobQueue<ImageData>* output_queue)
    : input_queue_(input_queue), output_queue_(output_queue) {}

void BitMapReaderThread::Run() {
    while (true) {
        if (IsStopped()) {
            break;
        }

        auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            auto& image_data = input_job.Data();

            if (image_data.status == ImageReader::Status::SUCCESS) {
                auto& feature_data = image_data.featuredata;

                for (size_t i = 0; i < feature_data.bitmap_paths.size(); ++i) {
                    if (IsFileRGBD(feature_data.bitmap_paths[i])) {
                        CHECK(ExtractRGBDData(feature_data.bitmap_paths[i], feature_data.bitmap[i], false))
                            << "Read rgbd image " << feature_data.bitmap_paths[i] << " failed" << std::endl;
                    } else {
                        CHECK(feature_data.bitmap[i].Read(feature_data.bitmap_paths[i], false))
                            << "Read bit map " << feature_data.bitmap_paths[i] << " failed" << std::endl;
                    }
                }
            }

            output_queue_->Push(std::move(image_data));
        } else {
            break;
        }
    }
}




ImageResizerThread::ImageResizerThread(const int max_image_size, JobQueue<ImageData>* input_queue,
                                       JobQueue<ImageData>* output_queue)
    : max_image_size_(max_image_size), input_queue_(input_queue), output_queue_(output_queue) {}

void ImageResizerThread::Run() {
    while (true) {
        if (IsStopped()) {
            break;
        }

        auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            auto& image_data = input_job.Data();

            if (image_data.status == ImageReader::Status::SUCCESS) {
                auto& feature_data = image_data.featuredata;

                for (int i = 0; i < feature_data.bitmap.size(); ++i) {
                    if (static_cast<int>(feature_data.bitmap[i].Width()) > max_image_size_ ||
                        static_cast<int>(feature_data.bitmap[i].Height()) > max_image_size_) {
                        // Fit the down-sampled version exactly into the max dimensions.
                        const double scale = static_cast<double>(max_image_size_) /
                                             std::max(feature_data.bitmap[i].Width(), feature_data.bitmap[i].Height());
                        const int new_width = static_cast<int>(feature_data.bitmap[i].Width() * scale);
                        const int new_height = static_cast<int>(feature_data.bitmap[i].Height() * scale);

                        feature_data.bitmap[i].Rescale(new_width, new_height);
                        feature_data.mask[i].Rescale(new_width, new_height);
                    }
                }
            }

            output_queue_->Push(std::move(image_data));
        } else {
            break;
        }
    }
}

SiftFeatureExtractorThread::SiftFeatureExtractorThread(const SiftExtractionOptions& sift_options,
                                                       const std::shared_ptr<Bitmap>& camera_mask,
                                                       JobQueue<ImageData>* input_queue,
                                                       JobQueue<ImageData>* output_queue)
    : sift_options_(sift_options), camera_mask_(camera_mask), input_queue_(input_queue), output_queue_(output_queue) {
    CHECK(sift_options_.Check());
}

void SiftFeatureExtractorThread::Run() {
    PrintHeading1("SiftFeatureExtractorThread start");

#ifdef CUDA_ENABLED
    std::unique_ptr<SiftGPU> sift_gpu;
    if (sift_options_.use_gpu) {
        sift_gpu.reset(new SiftGPU);
        if (!CreateSiftGPUExtractor(sift_options_, sift_gpu.get())) {
            std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;
            SignalInvalidSetup();
            return;
        }
    }
#endif

    // Initial Panorama
    std::shared_ptr<Panorama> panorama = std::make_shared<Panorama>();

    if (sift_options_.convert_to_perspective_image) {
        if (sift_options_.use_panorama_config) {
            panorama->PerspectiveParamsProcess(sift_options_.panorama_image_width, sift_options_.panorama_image_height,
                                               sift_options_.panorama_config_params);
        } else {
            panorama->PerspectiveParamsProcess(sift_options_.perspective_image_width,
                                               sift_options_.perspective_image_height,
                                               sift_options_.perspective_image_count, sift_options_.fov_w,
                                               sift_options_.panorama_image_width, sift_options_.panorama_image_height);
        }
    }

    // Initial AprilTag Detector
    apriltag_detector_t *tag_detector = apriltag_detector_create();
    apriltag_detector_add_family(tag_detector, sift_options_.apriltag_family);
    tag_detector->quad_decimate = 2;
    tag_detector->quad_sigma = 0.0;
    tag_detector->nthreads = 1;
    tag_detector->debug = 0;
    tag_detector->refine_edges = 1;

    // Initialize LargeFovImage
    std::shared_ptr<LargeFovImage> large_fov_image = std::make_shared<LargeFovImage>();
    bool large_fov_image_set = false;

    // Initialize LargeFovImage
    std::shared_ptr<PiecewiseImage> piecewise_image = std::make_shared<PiecewiseImage>();
    bool piecewise_image_set = false;


    SignalValidSetup();

    while (true) {
        if (IsStopped()) {
            break;
        }

        auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            auto& image_data = input_job.Data();
            auto& feature_data = image_data.featuredata;

            if (image_data.status == ImageReader::Status::SUCCESS) {
                bool success = false;
                if (sift_options_.estimate_affine_shape ||
                    sift_options_.domain_size_pooling) {
                    SiftExtractionOptions custom_options = sift_options_;
                    std::unordered_set<int> visited_peak_threshold_levels;
                    int peak_threshold_level = 0;
                    std::vector<Bitmap> perspective_images;
                    if (image_data.camera.ModelName() == "SPHERICAL") {
                        std::cout << "Panorama Perspective Image" << std::endl;
                        if (feature_data.bitmap[0].Height() != panorama->GetPanoramaHeight() ||
                            feature_data.bitmap[0].Width() != panorama->GetPanoramaWidth()) {
                            std::cout << "image name: "<<feature_data.image.Name()<<std::endl;
                            std::cout << "panorama size: "<<feature_data.bitmap[0].Height()<<" "<<feature_data.bitmap[0].Width()<<std::endl;
                            std::cout << "Panorama Image size incorrect, Reinitialize" << std::endl;

                            if (sift_options_.use_panorama_config) {
                                panorama->PerspectiveParamsProcess(feature_data.bitmap[0].Width(),
                                                                   feature_data.bitmap[0].Height(),
                                                                   sift_options_.panorama_config_params);
                            } else {
                                panorama->PerspectiveParamsProcess(
                                    sift_options_.perspective_image_width, sift_options_.perspective_image_height,
                                    sift_options_.perspective_image_count, sift_options_.fov_w,
                                    feature_data.bitmap[0].Width(), feature_data.bitmap[0].Height());
                            }
                        }
                        // Check panorama image size
                        panorama->PanoramaToPerspectives(&feature_data.bitmap[0], perspective_images);
                    } else if (
                        (image_data.camera.ModelName() == "OPENCV_FISHEYE") &&
                         image_data.camera.NumLocalCameras() > 2) {
                        if (sift_options_.convert_to_perspective_image) {
                            std::cout << "Pro2 Perspective Image" << std::endl;
                            if (feature_data.bitmap[0].Height() != large_fov_image->GetImageHeight() ||
                                feature_data.bitmap[0].Width() != large_fov_image->GetImageWidth() ||
                                !large_fov_image_set) {
                                large_fov_image->SetCamera(image_data.camera);
                                large_fov_image->ParamPreprocess(
                                    sift_options_.perspective_image_width, sift_options_.perspective_image_height,
                                    sift_options_.fov_w, feature_data.bitmap[0].Width(), feature_data.bitmap[0].Height());

                                large_fov_image_set = true;
                            }

                            for (int i = 0; 
                                i < image_data.camera.NumLocalCameras(); i++) {
                                Bitmap perspective_image;
                                large_fov_image->ToPerspective(feature_data.bitmap[i], perspective_image, i);
                                perspective_images.emplace_back(perspective_image);
                            }
                        } else {
                            std::cout << "Pro2 Normal Image" << std::endl;
                            for (int i = 0; 
                                 i < image_data.camera.NumLocalCameras(); i++) {
                                perspective_images.emplace_back(feature_data.bitmap[i]);
                            }
                        }
                    } else if (
                        (image_data.camera.ModelName() == "OPENCV_FISHEYE") &&
                         image_data.camera.NumLocalCameras() <= 2) {
                        std::cout << "Rig Perspective Image" << std::endl;
                        size_t num_local_camera = image_data.camera.NumLocalCameras();
                        if (feature_data.bitmap[0].Height() != piecewise_image->GetImageHeight() ||
                            feature_data.bitmap[0].Width() != piecewise_image->GetImageWidth() ||
                            !piecewise_image_set) {
                            piecewise_image->SetCamera(image_data.camera);
                            CHECK(sift_options_.perspective_image_count == 6 ||
                                  sift_options_.perspective_image_count == 8 ||
                                  sift_options_.perspective_image_count == 10 ||
                                  sift_options_.perspective_image_count == 30);
                            piecewise_image->ParamPreprocess(
                                sift_options_.perspective_image_width, sift_options_.perspective_image_height,
                                sift_options_.fov_w, feature_data.bitmap[0].Width(), feature_data.bitmap[0].Height(),
                                sift_options_.perspective_image_count / num_local_camera);

                            piecewise_image_set = true;
                        }

                        for (int i = 0; i < num_local_camera; i++) {
                            std::vector<Bitmap> splited_perspective_images;
                            piecewise_image->ToSplitedPerspectives(feature_data.bitmap[i], splited_perspective_images, i);
                                                        
                            for(size_t j = 0; j<splited_perspective_images.size(); ++j){
                                perspective_images.emplace_back(splited_perspective_images[j]);
                            }
                        }
                    } else {
                        std::cout << "Normal Perspective Image" << std::endl;
                        for (int i = 0; i < image_data.camera.NumLocalCameras(); i++) {
                            perspective_images.emplace_back(feature_data.bitmap[i]);
                        }
                    }

                    do {
                        if (visited_peak_threshold_levels.find(peak_threshold_level) !=
                            visited_peak_threshold_levels.end()) {
                            break;
                        }
                    
                        custom_options.max_num_features = 
                            1.3 * custom_options.max_num_features_customized + 1;
                        custom_options.peak_threshold =
                            sift_options_.peak_threshold * pow(2.0, static_cast<double>(peak_threshold_level));

                        std::cout << "peak threshold: " << custom_options.peak_threshold << std::endl;
                        if ((image_data.camera.ModelName() == "UNIFIED" ||
                             image_data.camera.ModelName() == "OPENCV_FISHEYE") &&
                            sift_options_.convert_to_perspective_image) {
                            if (image_data.camera.NumLocalCameras() > 2) {
                                // Pro2 Feature Extraction
                                success = ExtractCovariantSiftFeaturesCPU(
                                    custom_options, perspective_images, &feature_data.keypoints,
                                    &feature_data.descriptors, &feature_data.panoramaidxs, true, large_fov_image.get());
                            } else {
                                // Rig Feature Extraction
                                success = ExtractCovariantSiftFeaturesCPU(
                                    custom_options, perspective_images, &feature_data.keypoints,
                                    &feature_data.descriptors, &feature_data.panoramaidxs, &feature_data.pieceidxs, 
                                    true, piecewise_image.get(), image_data.camera.NumLocalCameras());
                            }
                        } else if (image_data.camera.ModelName() != "SPHERICAL" ||
                                   !sift_options_.convert_to_perspective_image) {
                            success = ExtractCovariantSiftFeaturesCPU(
                                custom_options, feature_data.bitmap, 
                                &feature_data.keypoints, 
                                &feature_data.descriptors,
                                &feature_data.panoramaidxs);
                        } else {
                            // Panorama Image Feature Extraction
                            CHECK(false) << "Not implemented with DSP-SIFT";
                        }

                        visited_peak_threshold_levels.insert(peak_threshold_level);
                        
                        std::cout << "keypoints size: " << feature_data.keypoints.size() << std::endl;

                        if (static_cast<double>(feature_data.keypoints.size()) <
                            static_cast<double>(custom_options.min_num_features_customized) * 0.8) {
                            peak_threshold_level--;
                        } else if (static_cast<double>(feature_data.keypoints.size()) >
                                   static_cast<double>(custom_options.max_num_features_customized) * 1.3) {
                            peak_threshold_level++;
                        } else {
                            break;
                        }
                    } while (peak_threshold_level >= -3 && peak_threshold_level <= 3);
                }
#ifdef CUDA_ENABLED
                else if (sift_options_.use_gpu) {
                    SiftExtractionOptions custom_options = sift_options_;
                    std::unordered_set<int> visited_peak_threshold_levels;
                    int peak_threshold_level = 0;
                    std::cout << "Detection starts ============== " << std::endl;
                    std::vector<Bitmap> perspective_images;

                    std::chrono::high_resolution_clock::time_point start_time_convert_image = std::chrono::high_resolution_clock::now();

                    if (image_data.camera.ModelName() == "SPHERICAL") {
                        std::cout << "Panorama Perspective Image" << std::endl;
                        if (feature_data.bitmap[0].Height() != panorama->GetPanoramaHeight() ||
                            feature_data.bitmap[0].Width() != panorama->GetPanoramaWidth()) {
                            std::cout << "image name: "<<feature_data.image.Name()<<std::endl;
                            std::cout << "panorama size: "<<feature_data.bitmap[0].Height()<<" "<<feature_data.bitmap[0].Width()<<std::endl;
                            std::cout << "Panorama Image size incorrect, Reinitialize" << std::endl;

                            if (sift_options_.use_panorama_config) {
                                panorama->PerspectiveParamsProcess(feature_data.bitmap[0].Width(),
                                                                   feature_data.bitmap[0].Height(),
                                                                   sift_options_.panorama_config_params);
                            } else {
                                panorama->PerspectiveParamsProcess(
                                    sift_options_.perspective_image_width, sift_options_.perspective_image_height,
                                    sift_options_.perspective_image_count, sift_options_.fov_w,
                                    feature_data.bitmap[0].Width(), feature_data.bitmap[0].Height());
                            }
                        }
                        // Check panorama image size
                        panorama->PanoramaToPerspectives(&feature_data.bitmap[0], perspective_images);
                    } else if (
                        (image_data.camera.ModelName() == "OPENCV_FISHEYE") &&
                         image_data.camera.NumLocalCameras() > 2) {
                        if (sift_options_.convert_to_perspective_image) {
                            std::cout << "Omni Perspective Image" << std::endl;
                            if (feature_data.bitmap[0].Height() != large_fov_image->GetImageHeight() ||
                                feature_data.bitmap[0].Width() != large_fov_image->GetImageWidth() ||
                                !large_fov_image_set) {
                                large_fov_image->SetCamera(image_data.camera);
                                large_fov_image->ParamPreprocess(
                                    sift_options_.perspective_image_width, sift_options_.perspective_image_height,
                                    sift_options_.fov_w, feature_data.bitmap[0].Width(), feature_data.bitmap[0].Height());

                                large_fov_image_set = true;
                            }

                            for (int i = 0; i < image_data.camera.NumLocalCameras(); i++) {
                                Bitmap perspective_image;
                                large_fov_image->ToPerspective(feature_data.bitmap[i], perspective_image, i);
                                perspective_images.emplace_back(perspective_image);
                            }
                        } else {
                            std::cout << "Omni Normal Image" << std::endl;
                            perspective_images = feature_data.bitmap;
                        }
                    } else if (
                        (image_data.camera.ModelName() == "OPENCV_FISHEYE") &&
                         image_data.camera.NumLocalCameras() == 2) {
                        std::cout << "Rig Perspective Image" << std::endl;
                        size_t num_local_camera = image_data.camera.NumLocalCameras();
                        if (feature_data.bitmap[0].Height() != piecewise_image->GetImageHeight() ||
                            feature_data.bitmap[0].Width() != piecewise_image->GetImageWidth() ||
                            !piecewise_image_set) {
                            piecewise_image->SetCamera(image_data.camera);
                            CHECK(sift_options_.perspective_image_count == 6 ||
                                sift_options_.perspective_image_count == 8 ||
                                sift_options_.perspective_image_count == 10 ||
                                sift_options_.perspective_image_count == 30);
                            piecewise_image->ParamPreprocess(
                                sift_options_.perspective_image_width, sift_options_.perspective_image_height,
                                sift_options_.fov_w, feature_data.bitmap[0].Width(), feature_data.bitmap[0].Height(),
                                sift_options_.perspective_image_count / num_local_camera);

                            piecewise_image_set = true;
                        }
                        if (sift_options_.convert_to_perspective_image) {
                            for (int i = 0; i < num_local_camera; i++) {
                                std::vector<Bitmap> splited_perspective_images;
                                piecewise_image->ToSplitedPerspectives(feature_data.bitmap[i], splited_perspective_images, i);
                                                            
                                for(size_t j = 0; j<splited_perspective_images.size(); ++j){
                                    perspective_images.emplace_back(std::move(splited_perspective_images[j]));
                                }
                            }
                        } else {
                            std::cout << "Rig Normal Image" << std::endl;
                            perspective_images = feature_data.bitmap;
                        }
                    } else {
                        std::cout << "Normal Perspective Image" << std::endl;
                        perspective_images = feature_data.bitmap;
                    }

                    std::chrono::high_resolution_clock::time_point end_time_convert_image =
                        std::chrono::high_resolution_clock::now();
                    std::cout << StringPrintf("=> Convert to perspective images Elapsed time: %.3f [ms]",
                                              std::chrono::duration_cast<std::chrono::microseconds>(
                                                  end_time_convert_image - start_time_convert_image)
                                                      .count() /
                                                  1e3)
                                     .c_str()
                              << std::endl;

                    if (sift_options_.detect_apriltag) {
                        std::chrono::high_resolution_clock::time_point start_time_detect_apriltag =
                            std::chrono::high_resolution_clock::now();
                        std::cout << "Detect apriltag" << std::endl;
                        if ((image_data.camera.ModelName() == "UNIFIED" ||
                             image_data.camera.ModelName() == "OPENCV_FISHEYE") 
                             && image_data.camera.NumLocalCameras() > 2 &&
                             large_fov_image_set) {
                            // Pro2 Camera
                            for (int i = 0; i < image_data.camera.NumLocalCameras(); i++) {
                                // Extract AprilTag
                                auto perspective_image = perspective_images[i];
                                if (perspective_image.IsRGB()) {
                                    perspective_image.ConvertToGray();
                                }

                                std::vector<uint8_t> bitmap_raw_bits = perspective_image.ConvertToRawBits();

                                // Make an image_u8_t header for the Mat data
                                image_u8_t im = { .width = perspective_image.Width(),
                                    .height = perspective_image.Height(),
                                    .stride = perspective_image.Width(),
                                    .buf = bitmap_raw_bits.data()
                                };

                                zarray_t *detections = apriltag_detector_detect(tag_detector, &im);

                                for (int j = 0; j < zarray_size(detections); j++) {
                                    apriltag_detection_t *det;
                                    zarray_get(detections, j, &det);

                                    AprilTagDetection cur_detection;
                                    cur_detection.local_camera_id = i;
                                    cur_detection.id = det->id;
                                    double cxy_x, cxy_y;
                                    large_fov_image->ConvertPerspectiveCoordToOriginal(
                                        det->c[0], det->c[1], i, cxy_x, cxy_y);
                                    cur_detection.cxy.first = cxy_x;
                                    cur_detection.cxy.second = cxy_y;

                                    double p_1_x, p_1_y, p_2_x, p_2_y, p_3_x, p_3_y, p_4_x, p_4_y;
                                    large_fov_image->ConvertPerspectiveCoordToOriginal(
                                        det->p[0][0], det->p[0][1], i, p_1_x, p_1_y);
                                    cur_detection.p[0].first = p_1_x;
                                    cur_detection.p[0].second = p_1_y;

                                    large_fov_image->ConvertPerspectiveCoordToOriginal(
                                        det->p[1][0], det->p[1][1], i, p_2_x, p_2_y);
                                    cur_detection.p[1].first = p_2_x;
                                    cur_detection.p[1].second = p_2_y;

                                    large_fov_image->ConvertPerspectiveCoordToOriginal(
                                        det->p[2][0], det->p[2][1], i, p_3_x, p_3_y);
                                    cur_detection.p[2].first = p_3_x;
                                    cur_detection.p[2].second = p_3_y;

                                    large_fov_image->ConvertPerspectiveCoordToOriginal(
                                        det->p[3][0], det->p[3][1], i, p_4_x, p_4_y);
                                    cur_detection.p[3].first = p_4_x;
                                    cur_detection.p[3].second = p_4_y;

                                    feature_data.detections.emplace_back(cur_detection);
                                }
                                std::cout << StringPrintf("Pro2 Camera: %d apriltags are detected!\n", zarray_size(detections));
                                apriltag_detections_destroy(detections);
                            }
                        } else if (
                            (image_data.camera.ModelName() == "UNIFIED" ||
                             image_data.camera.ModelName() == "OPENCV_FISHEYE") 
                             && image_data.camera.NumLocalCameras() == 2 
                             && sift_options_.convert_to_perspective_image) {
                            // Rig Camera
                            size_t num_local_camera = image_data.camera.NumLocalCameras();
                            size_t piece_num = sift_options_.perspective_image_count / num_local_camera;
                            for (int i = 0; i < num_local_camera; i++) {
                                for (int j = 0; j < piece_num; j++) {
                                    // Extract AprilTag
                                    auto perspective_image =
                                        perspective_images[i * piece_num + j];
                                    if (perspective_image.IsRGB()) {
                                        perspective_image.ConvertToGray();
                                    }

                                    std::vector<uint8_t> bitmap_raw_bits = perspective_image.ConvertToRawBits();

                                    // Make an image_u8_t header for the Mat data
                                    image_u8_t im = { .width = perspective_image.Width(),
                                        .height = perspective_image.Height(),
                                        .stride = perspective_image.Width(),
                                        .buf = bitmap_raw_bits.data()
                                    };

                                    zarray_t *detections = apriltag_detector_detect(tag_detector, &im);


                                    for (int k = 0; k < zarray_size(detections); k++) {
                                        apriltag_detection_t *det;
                                        zarray_get(detections, k, &det);

                                        AprilTagDetection cur_detection;
                                        cur_detection.id = det->id;
                                        cur_detection.local_camera_id = i;
                                        double cxy_x, cxy_y;
                                        piecewise_image->ConvertSplitedPerspectiveCoordToOriginal(
                                            det->c[0], det->c[1], i, j, cxy_x, cxy_y);
                                        cur_detection.cxy.first = cxy_x;
                                        cur_detection.cxy.second = cxy_y;

                                        double p_1_x, p_1_y, p_2_x, p_2_y, p_3_x, p_3_y, p_4_x, p_4_y;
                                        piecewise_image->ConvertSplitedPerspectiveCoordToOriginal(
                                            det->p[0][0], det->p[0][1], i, j, p_1_x, p_1_y);
                                        cur_detection.p[0].first = p_1_x;
                                        cur_detection.p[0].second = p_1_y;

                                        piecewise_image->ConvertSplitedPerspectiveCoordToOriginal(
                                            det->p[1][0], det->p[1][1], i, j, p_2_x, p_2_y);
                                        cur_detection.p[1].first = p_2_x;
                                        cur_detection.p[1].second = p_2_y;

                                        piecewise_image->ConvertSplitedPerspectiveCoordToOriginal(
                                            det->p[2][0], det->p[2][1], i, j, p_3_x, p_3_y);
                                        cur_detection.p[2].first = p_3_x;
                                        cur_detection.p[2].second = p_3_y;

                                        piecewise_image->ConvertSplitedPerspectiveCoordToOriginal(
                                            det->p[3][0], det->p[3][1], i, j, p_4_x, p_4_y);
                                        cur_detection.p[3].first = p_4_x;
                                        cur_detection.p[3].second = p_4_y;

                                        feature_data.detections.emplace_back(cur_detection);
                                    }
                                    std::cout << StringPrintf("Rig Camera: %d apriltags are detected!\n", zarray_size(detections));
                                    apriltag_detections_destroy(detections);
                                }
                            }
                        } else if (image_data.camera.ModelName() == "SPHERICAL") {
                            // Convert sub detection result to panorama detection result
                            for (int i = 0; i < perspective_images.size(); i++) {
                                
                                // std::chrono::high_resolution_clock::time_point start_time_detect_apriltag_preoperation =
                                //     std::chrono::high_resolution_clock::now();
                                auto& perspective_image = perspective_images[i];
                                // cv::Mat image_mat, image_gray_mat;
                                // BitmapToMat(perspective_image, image_mat);
                                // if (perspective_image.IsRGB()) {
                                //     cv::cvtColor(image_mat, image_gray_mat, cv::COLOR_BGR2GRAY);
                                // } else {
                                //     image_gray_mat = image_mat.clone();
                                // }


                                if (perspective_image.IsRGB()) {
                                    perspective_image.ConvertToGray();
                                }

                                std::vector<uint8_t> bitmap_raw_bits = perspective_image.ConvertToRawBits();

                                // std::chrono::high_resolution_clock::time_point end_time_detect_apriltag_preoperation =
                                //     std::chrono::high_resolution_clock::now();
                                // std::cout << StringPrintf("=> Detect apriltag Pre-operation Elapsed time: %.3f [ms]",
                                //                           std::chrono::duration_cast<std::chrono::microseconds>(
                                //                               end_time_detect_apriltag_preoperation - start_time_detect_apriltag_preoperation)
                                //                                   .count() /
                                //                               1e3)
                                //                  .c_str()
                                //           << std::endl;

                                // std::chrono::high_resolution_clock::time_point start_time_detect_apriltag_runtime =
                                //     std::chrono::high_resolution_clock::now();

                                

                                // Make an image_u8_t header for the Mat data
                                image_u8_t im = { .width = perspective_image.Width(),
                                    .height = perspective_image.Height(),
                                    .stride = perspective_image.Width(),
                                    .buf = bitmap_raw_bits.data()
                                };

                                zarray_t *detections = apriltag_detector_detect(tag_detector, &im);

                                // std::chrono::high_resolution_clock::time_point end_time_detect_apriltag_runtime =
                                //     std::chrono::high_resolution_clock::now();
                                // std::cout << StringPrintf("=> Detect apriltag run-time Elapsed time: %.3f [ms]",
                                //                           std::chrono::duration_cast<std::chrono::microseconds>(
                                //                               end_time_detect_apriltag_runtime - start_time_detect_apriltag_runtime)
                                //                                   .count() /
                                //                               1e3)
                                //                  .c_str()
                                //           << std::endl;

                                
                                // std::chrono::high_resolution_clock::time_point start_time_convert_apriltag_runtime =
                                //     std::chrono::high_resolution_clock::now();

                                AprilTagDetections process_detections;
                                for (int k = 0; k < zarray_size(detections); k++) {
                                    apriltag_detection_t *det;
                                    zarray_get(detections, k, &det);
                                    AprilTagDetection cur_detection;

                                    // id
                                    cur_detection.id = det->id;
                                    cur_detection.local_camera_id = 0;
                                    double u, v, u_in, v_in;
                                    // cxy
                                    u_in = det->c[0];
                                    v_in = det->c[1];
                                    panorama->ConvertPerspectiveCoordToPanorama(i, u_in, v_in, u, v);
                                    cur_detection.cxy = {(float)u, (float)v};
                                    // p[4]
                                    for (size_t j = 0; j < 4; ++j) {
                                        u_in = det->p[j][0];
                                        v_in = det->p[j][1];
                                        panorama->ConvertPerspectiveCoordToPanorama(i, u_in, v_in, u, v);
                                        cur_detection.p[j] = {(float)u, (float)v};
                                    }

                                    // Check the id exist in the panorama or not Note: Avoid record
                                    // AprilTag several times
                                    if (!process_detections.empty()) {
                                        bool id_exist = true;
                                        int id = 0;
                                        for (size_t j = 0; j < process_detections.size(); ++j) {
                                            auto detection = process_detections.at(j);
                                            if (detection.id == cur_detection.id) {
                                                id_exist = true;
                                                id = j;
                                                break;
                                            } else {
                                                id_exist = false;
                                            }
                                        }

                                        if (id_exist) {
                                            process_detections.at(id).cxy = {
                                                (process_detections.at(id).cxy.first + cur_detection.cxy.first) / 2,
                                                (process_detections.at(id).cxy.second + cur_detection.cxy.second) / 2};
                                            for (size_t j = 0; j < 4; ++j) {
                                                float mean_u =
                                                    (cur_detection.p[j].first + process_detections.at(id).p[j].first) /
                                                    2;
                                                float mean_v = (cur_detection.p[j].second +
                                                                process_detections.at(id).p[j].second) /
                                                               2;
                                                process_detections.at(id).p[j] = {(float)mean_u, (float)mean_v};
                                            }
                                        } else {
                                            process_detections.emplace_back(cur_detection);
                                        }
                                    } else {
                                        process_detections.emplace_back(cur_detection);
                                    }

                                    feature_data.detections = process_detections;
                                }
                                std::cout << StringPrintf("Sphere Camera: %d apriltags are detected!\n", zarray_size(detections));
                                apriltag_detections_destroy(detections);

                                // std::chrono::high_resolution_clock::time_point end_time_convert_apriltag_runtime =
                                //     std::chrono::high_resolution_clock::now();
                                // std::cout << StringPrintf("=> Convert apriltag run-time Elapsed time: %.3f [ms]",
                                //                           std::chrono::duration_cast<std::chrono::microseconds>(
                                //                               end_time_convert_apriltag_runtime - start_time_convert_apriltag_runtime)
                                //                                   .count() /
                                //                               1e3)
                                //                  .c_str()
                                //           << std::endl;
                            }
                        } else {
                            // Normal Camera
                            for (int i = 0; i < image_data.camera.NumLocalCameras(); i++) {
                                // Extract AprilTag
                                auto perspective_image = perspective_images[i];
                                cv::Mat image_mat, image_gray_mat;
                                BitmapToMat(perspective_image, image_mat);
                                if (perspective_image.IsRGB()) {
                                    cv::cvtColor(image_mat, image_gray_mat, cv::COLOR_BGR2GRAY);
                                } else {
                                    image_gray_mat = image_mat.clone();
                                }

                                // Make an image_u8_t header for the Mat data
                                image_u8_t im = { .width = image_gray_mat.cols,
                                    .height = image_gray_mat.rows,
                                    .stride = image_gray_mat.cols,
                                    .buf = image_gray_mat.data
                                };

                                zarray_t *detections = apriltag_detector_detect(tag_detector, &im);


                                for (int k = 0; k < zarray_size(detections); k++) {
                                    apriltag_detection_t *det;
                                    zarray_get(detections, k, &det);

                                    AprilTagDetection cur_detection;
                                    cur_detection.local_camera_id = i;
                                    cur_detection.id = det->id;
                                    cur_detection.cxy.first = det->c[0];
                                    cur_detection.cxy.second = det->c[1];

                                    cur_detection.p[0].first = det->p[0][0];
                                    cur_detection.p[0].second = det->p[0][1];

                                    cur_detection.p[1].first = det->p[1][0];
                                    cur_detection.p[1].second = det->p[1][1];

                                    cur_detection.p[2].first = det->p[2][0];
                                    cur_detection.p[2].second = det->p[2][1];

                                    cur_detection.p[3].first = det->p[3][0];
                                    cur_detection.p[3].second = det->p[3][1];

                                    feature_data.detections.emplace_back(cur_detection);
                                }
                                std::cout << StringPrintf("Normal Camera: %d apriltags are detected!\n", zarray_size(detections));
                                apriltag_detections_destroy(detections);
                            }
                        }
                        std::chrono::high_resolution_clock::time_point end_time_detect_apriltag =
                        std::chrono::high_resolution_clock::now();
                        std::cout << StringPrintf("=> Detect apriltag Elapsed time: %.3f [ms]",
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                end_time_detect_apriltag - start_time_detect_apriltag).count() / 1e3).c_str()
                              << std::endl;
                    }

                    std::chrono::high_resolution_clock::time_point start_time_sift_extraction =
                        std::chrono::high_resolution_clock::now();

                    do {
                        if (visited_peak_threshold_levels.find(peak_threshold_level) !=
                            visited_peak_threshold_levels.end()) {
                            break;
                        }

                        custom_options.peak_threshold =
                            sift_options_.peak_threshold * pow(1.6, static_cast<double>(peak_threshold_level));

                        std::cout << "peak threshold: " << custom_options.peak_threshold << std::endl;
                        if (image_data.camera.ModelName() == "SPHERICAL") {
                            if (sift_options_.convert_to_perspective_image) {
                                // Panorama Image Feature Extraction
                                CHECK(feature_data.bitmap.size() == 1);
                                success = ExtractSiftFeaturesGPUPanorama(
                                    custom_options, feature_data.bitmap[0],
                                    perspective_images, panorama.get(), 
                                    sift_gpu.get(), &feature_data.keypoints, 
                                    &feature_data.descriptors, 
                                    &feature_data.panoramaidxs);
                            } else {
                                success = ExtractSiftFeaturesGPU(
                                custom_options, feature_data.bitmap, 
                                sift_gpu.get(), &feature_data.keypoints, 
                                &feature_data.descriptors,
                                &feature_data.panoramaidxs);
                            }
                        } else if (image_data.camera.ModelName() == "OPENCV_FISHEYE") {
                            if (image_data.camera.NumLocalCameras() > 2) {
                                // Pro2 Feature Extraction
                                success = ExtractSiftFeaturesGPU(
                                    custom_options, perspective_images, sift_gpu.get(), &feature_data.keypoints,
                                    &feature_data.descriptors,
                                    &feature_data.panoramaidxs,
                                    !!large_fov_image_set, 
                                    large_fov_image_set ? large_fov_image.get() : nullptr);
                            } else if (sift_options_.convert_to_perspective_image && image_data.camera.NumLocalCameras() == 2) {
                                std::cout << "perspective_images: " << perspective_images.size() << std::endl;
                                // Rig Feature Extraction
                                success = ExtractSiftFeaturesGPU(
                                    custom_options, perspective_images, sift_gpu.get(), &feature_data.keypoints,
                                    &feature_data.descriptors, &feature_data.panoramaidxs, &feature_data.pieceidxs, 
                                    true, piecewise_image.get(), image_data.camera.NumLocalCameras());
                            } else {
                                success = ExtractSiftFeaturesGPU(
                                    custom_options, perspective_images, 
                                    sift_gpu.get(), &feature_data.keypoints, 
                                    &feature_data.descriptors,
                                    &feature_data.panoramaidxs);
                            }
                        } else {
                            success = ExtractSiftFeaturesGPU(
                                custom_options, perspective_images, 
                                sift_gpu.get(), &feature_data.keypoints, 
                                &feature_data.descriptors,
                                &feature_data.panoramaidxs);
                        }

                        visited_peak_threshold_levels.insert(peak_threshold_level);

                        std::cout << "keypoints size: " << feature_data.keypoints.size() << std::endl;

                        if (static_cast<double>(feature_data.keypoints.size()) <
                            static_cast<double>(custom_options.min_num_features_customized) * 0.8) {
                            peak_threshold_level--;
                        } else if (static_cast<double>(feature_data.keypoints.size()) >
                                   static_cast<double>(custom_options.max_num_features_customized) * 1.3) {
                            peak_threshold_level++;
                        } else {
                            break;
                        }
                    } while (peak_threshold_level >= -3 && peak_threshold_level <= 3);

                    std::chrono::high_resolution_clock::time_point end_time_sift_extraction =
                        std::chrono::high_resolution_clock::now();
                    std::cout << StringPrintf("=> Sift extraction Elapsed time: %.3f [ms]",
                                              std::chrono::duration_cast<std::chrono::microseconds>(
                                                  end_time_sift_extraction - start_time_sift_extraction)
                                                      .count() /
                                                  1e3)
                                     .c_str()
                              << std::endl;
                    std::cout << "Detection ends ==============" << std::endl;
                }
#endif
                else {
                    SiftExtractionOptions custom_options = sift_options_;
                    std::unordered_set<int> visited_peak_threshold_levels;
                    int peak_threshold_level = 0;
                    do {
                        if (visited_peak_threshold_levels.find(peak_threshold_level) !=
                            visited_peak_threshold_levels.end()) {
                            break;
                        }

                        custom_options.peak_threshold =
                            sift_options_.peak_threshold * pow(2.0, static_cast<double>(peak_threshold_level));

                        if (image_data.camera.ModelName() != "SPHERICAL" ||
                            !sift_options_.convert_to_perspective_image) {
                            success = ExtractSiftFeaturesCPU(custom_options, feature_data.bitmap, tag_detector,
                                                             &feature_data.keypoints, &feature_data.descriptors,
                                                             &feature_data.panoramaidxs, &feature_data.detections);
                        } else {
                            success = ExtractSiftFeaturesCPUPanorama(
                                custom_options, feature_data.bitmap[0], tag_detector, &feature_data.keypoints,
                                &feature_data.descriptors, &feature_data.panoramaidxs, &feature_data.detections);
                        }

                        visited_peak_threshold_levels.insert(peak_threshold_level);

                        if (static_cast<double>(feature_data.keypoints.size()) <
                            static_cast<double>(custom_options.min_num_features_customized) * 0.8) {
                            peak_threshold_level--;
                        } else if (static_cast<double>(feature_data.keypoints.size()) >
                                   static_cast<double>(custom_options.max_num_features_customized) * 1.3) {
                            peak_threshold_level++;
                        } else {
                            break;
                        }
                    } while (peak_threshold_level >= -3 && peak_threshold_level <= 3);
                }
                if (success) {
                    ScaleKeypoints(feature_data.bitmap[0], image_data.camera, &feature_data.keypoints);
                    if (camera_mask_) {
                        //std::cout << "mask keypoints" << std::endl;
                        MaskKeypoints(*camera_mask_, &feature_data.keypoints, &feature_data.descriptors, 
                                      &feature_data.panoramaidxs, &feature_data.detections);
                    }
                    if (feature_data.mask[0].Data()) {  // This is a wrong implementation if the multi-camera system
                                                        // want to use different masks for the local images.
                        //std::cout << "mask keypoints using data.mask" << std::endl;
                        MaskKeypoints(feature_data.mask[0], &feature_data.keypoints, &feature_data.descriptors, 
                                      &feature_data.panoramaidxs, &feature_data.detections);
                    }
                    FilterKeypoints(feature_data.bitmap[0], image_data.camera, &feature_data.keypoints, &feature_data.descriptors, 
                                    &feature_data.panoramaidxs, &feature_data.detections);
                } else {
                    image_data.status = ImageReader::Status::FAILURE;
                }
            }
            std::cout << "Deallocate bitmap" << std::endl << std::flush;
            for (size_t i = 0; i < feature_data.bitmap.size(); ++i) {
                feature_data.bitmap[i].Deallocate();
            }
            for (size_t i = 0; i < feature_data.mask.size(); ++i) {
                feature_data.mask[i].Deallocate();
            }
            output_queue_->Push(image_data);
        } else {
            break;
            
        }
    }

    // apriltag_detector_destroy(tag_detector);
}

FeatureWriterThread::FeatureWriterThread(const size_t num_images, FeatureDataContainer* data_container,
                                         JobQueue<ImageData>* input_queue)
    : num_images_(num_images), input_queue_(input_queue), data_container_(data_container) {}

void FeatureWriterThread::Run() {
    size_t image_index = 0;
    while (true) {
        if (IsStopped()) {
            break;
        }

        auto input_job = input_queue_->Pop();
        if (input_job.IsValid()) {
            std::cout << "receive image data" << std::endl;
            auto image_data = input_job.Data();
            auto feature_data = std::make_shared<FeatureData>(std::move(image_data.featuredata));
            auto camera_data = std::make_shared<Camera>(std::move(image_data.camera));
            image_index += 1;

            std::cout << StringPrintf("Processed file [%d/%d]", image_index, num_images_) << std::endl;
            std::cout << StringPrintf("  Id:              %d", feature_data->image.ImageId()) << std::endl;

            std::cout << StringPrintf("  Name:            %s", feature_data->image.Name().c_str()) << std::endl;

            if (feature_data->image.HasLabel()) {
                std::cout << StringPrintf("  Label:           %d", feature_data->image.LabelId()) << std::endl;
            }

            if (image_data.status == ImageReader::Status::IMAGE_EXISTS) {
                std::cout << "  SKIP: Features for image already extracted." << std::endl;
            } else if (image_data.status == ImageReader::Status::BITMAP_ERROR) {
                std::cout << "  ERROR: Failed to read image file format." << std::endl;
            } else if (image_data.status == ImageReader::Status::CAMERA_SINGLE_DIM_ERROR) {
                std::cout << "  ERROR: Single camera specified, "
                             "but images have different dimensions."
                          << std::endl;
            } else if (image_data.status == ImageReader::Status::CAMERA_EXIST_DIM_ERROR) {
                std::cout << "  ERROR: Image previously processed, but current image "
                             "has different dimensions."
                          << std::endl;
            } else if (image_data.status == ImageReader::Status::CAMERA_PARAM_ERROR) {
                std::cout << "  ERROR: Camera has invalid parameters." << std::endl;
            } else if (image_data.status == ImageReader::Status::FAILURE) {
                std::cout << "  ERROR: Failed to extract features." << std::endl;
            }

            if (image_data.status != ImageReader::Status::SUCCESS) {
                continue;
            }

            std::cout << StringPrintf("  Dimensions:      %d x %d", camera_data->Width(), camera_data->Height())
                      << std::endl;
            std::cout << StringPrintf("  Camera:          #%d - %s", camera_data->CameraId(),
                                      camera_data->ModelName().c_str())
                      << std::endl;
            std::cout << StringPrintf("  Focal Length:    %.2fpx", camera_data->MeanFocalLength());
            if (camera_data->HasPriorFocalLength()) {
                std::cout << " (Prior)" << std::endl;
            } else {
                std::cout << std::endl;
            }
            if (feature_data->image.HasTvecPrior()) {
                std::cout << StringPrintf("  GPS:             LAT=%.3f, LON=%.3f, ALT=%.3f",
                                          feature_data->image.TvecPrior(0), feature_data->image.TvecPrior(1),
                                          feature_data->image.TvecPrior(2))
                          << std::endl;
            }
            std::cout << StringPrintf("  Features:        %d", feature_data->keypoints.size()) << std::endl;

            std::cout << StringPrintf("  Tag Number:      %d", feature_data->detections.size()) << std::endl;

            //			if(feature_data->image.ImageId() == kInvalidImageId)
            //			{
            //				feature_data->image.SetImageId(static_cast<const image_t>(image_index));
            //				//feature_data->image.SetImageId(database_->WriteImage(imag_index));
            //			}

            data_container_->emplace(feature_data->image.ImageId(), feature_data);
            data_container_->emplace(camera_data->CameraId(), camera_data);
            data_container_->emplace(feature_data->image.Name(), feature_data->image.ImageId());
        } else {
            break;
        }
    }
}

}  // namespace sensemap
