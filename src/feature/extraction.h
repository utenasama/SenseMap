//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_FEATURE_EXTRACTION_H_
#define SENSEMAP_FEATURE_EXTRACTION_H_

#include "base/image_reader.h"
#include "feature/sift.h"
#include "util/threading.h"
#include "container/feature_data_container.h"

namespace sensemap {

struct FeatureData;

struct ImageData{
	ImageReader::Status status = ImageReader::Status::FAILURE;
	Camera camera;
	FeatureData featuredata;
};

// Simplified single thread version
class SingleSiftFeatureExtractor {

public:
	SingleSiftFeatureExtractor(const ImageReaderOptions &reader_options,
	                           FeatureDataContainer *data_container);

	void Run();

private:

	const ImageReaderOptions reader_options_;
	ImageReader image_reader_;
	FeatureDataContainer *data_container_;
};

// Feature extraction class to extract features for all images in a directory.
class SiftFeatureExtractor : public Thread {
public:
	SiftFeatureExtractor(const ImageReaderOptions& reader_options,
	                     const SiftExtractionOptions& sift_options,
	                     FeatureDataContainer *data_container,
                         image_t image_index = 0,
                         camera_t camera_index = 0, 
						 label_t label_index = 1);

private:
	void Run();

	const ImageReaderOptions reader_options_;
	const SiftExtractionOptions sift_options_;

	ImageReader image_reader_;
	FeatureDataContainer *data_container_;

	std::vector<std::unique_ptr<Thread>> resizers_;
	std::vector<std::unique_ptr<Thread>> extractors_;
	std::vector<std::unique_ptr<Thread>> bitmap_readers_;
	std::unique_ptr<Thread> writer_;

	std::unique_ptr<JobQueue<ImageData>> bitmap_reader_queue_;
	std::unique_ptr<JobQueue<ImageData>> resizer_queue_;
	std::unique_ptr<JobQueue<ImageData>> extractor_queue_;
	std::unique_ptr<JobQueue<ImageData>> writer_queue_;
};


class BitMapReaderThread : public Thread {
public:
	BitMapReaderThread(JobQueue<ImageData>* input_queue,
	                 JobQueue<ImageData>* output_queue);

private:
	void Run();

	JobQueue<ImageData>* input_queue_;
	JobQueue<ImageData>* output_queue_;
};



class ImageResizerThread : public Thread {
public:
	ImageResizerThread(const int max_image_size, JobQueue<ImageData>* input_queue,
	                   JobQueue<ImageData>* output_queue);

private:
	void Run();

	const int max_image_size_;

	JobQueue<ImageData>* input_queue_;
	JobQueue<ImageData>* output_queue_;
};

class SiftFeatureExtractorThread : public Thread {
public:
	SiftFeatureExtractorThread(const SiftExtractionOptions& sift_options,
	                           const std::shared_ptr<Bitmap>& camera_mask,
	                           JobQueue<ImageData>* input_queue,
	                           JobQueue<ImageData>* output_queue);

private:
	void Run();

	const SiftExtractionOptions sift_options_;
	std::shared_ptr<Bitmap> camera_mask_;

	JobQueue<ImageData>* input_queue_;
	JobQueue<ImageData>* output_queue_;
};

class FeatureWriterThread : public Thread {
public:
	FeatureWriterThread(const size_t num_images,
	                    FeatureDataContainer *data_container,
	                    JobQueue<ImageData>* input_queue);

private:
	void Run();

	const size_t num_images_;
	FeatureDataContainer *data_container_;
	JobQueue<ImageData>* input_queue_;
};

} // namespace sensemap

#endif //SENSEMAP_FEATURE_EXTRACTION_H
