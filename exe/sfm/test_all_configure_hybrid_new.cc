// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/pose.h"
#include "base/common.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/mat.h"
#include "util/rgbd_helper.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "feature/global_feature_extraction.h"

#include "controllers/incremental_mapper_controller.h"
#include "controllers/directed_mapper_controller.h"

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "base/version.h"
#ifdef DO_ENCRYPT_CHECK
#include "../check.h"
#endif

#include <dirent.h>
#include <sys/stat.h>

#include "util/gps_reader.h"
#include <unordered_set>
#include "../system_io.h"
#include "util/ply.h"

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

using namespace sensemap;

std::string configuration_file_path;
std::string gps_origin_str;
bool has_pca = false;
Eigen::Matrix<double, 128, 128> pca_matrix;

FILE *fs;

struct HybridOptions{
    bool debug_info = false;
    int child_id = -1;
    bool update_flag = false;
    bool save_flag = true;
    bool read_flag = true;
    HybridOptions(bool de = false, int ch = -1, bool up = false, bool sa = false, 
                  bool re = true):debug_info(de), child_id(ch), update_flag(up), 
                    save_flag(sa), read_flag(re){};
    void Print(){
        std::cout << "debug_info: " << debug_info << "\nchild_id: " << child_id
            << "\nupdate_flag: " << update_flag << "\nsave_flag: " << save_flag
            << "\nread_flag; " << read_flag << std::endl;
    };
}; 

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

void PrintReconSummary(const std::string &flog_name, const size_t num_total_image,
                       const std::shared_ptr<ReconstructionManager> &reconstruction_manager) {
    if (reconstruction_manager->Size() == 0) {
        return;
    }
    std::shared_ptr<Reconstruction> best_rec;
    for (int i = 0; i < reconstruction_manager->Size(); ++i) {
        const std::shared_ptr<Reconstruction> &rec = reconstruction_manager->Get(i);
        if (!best_rec || best_rec->NumRegisterImages() < rec->NumRegisterImages()) {
            best_rec = rec;
        }
    }
    FILE *fp = fopen(flog_name.c_str(), "w");

    size_t num_reg_image = best_rec->NumRegisterImages();
    fprintf(fp, "Registered / Total: %zu / %zu\n", num_reg_image, num_total_image);
    fprintf(fp, "Mean Track Length: %f\n", best_rec->ComputeMeanTrackLength());
    fprintf(fp, "Mean Reprojection Error: %f\n", best_rec->ComputeMeanReprojectionError());
    fprintf(fp, "Mean Observation Per Register Image: %f\n", best_rec->ComputeMeanObservationsPerRegImage());

    fclose(fp);
}

bool IsBluetoothFile(std::string filePath) {
    std::ifstream infile;
    infile.open(filePath, std::ios::in);
    if (!infile.is_open()) {
        std::cout << "Cant to Open file: " << filePath << std::endl;
        return false;
    }
    std::vector<std::string> items = StringSplit(filePath, ".");
    if (items.back() != "txt" && items.back() != "csv") {
        std::cout << filePath << "is not .txt or csv file" << std::endl;
        return false;
    }
    int count = 0;
    std::string line;
    while (getline(infile, line)) {
        if (line.size() == 0) {
            continue;
        } 
        items = StringSplit(line, ",");
        if (items.size() < 9) {
            std::cout << filePath << " is not Bluetooth file" << std::endl;
            return false;
        }
        if (std::stoi(items[8]) > -200 && std::stoi(items[8]) < 200){
             count++;
        }

        if (count > 5) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> GetRecursiveTMPFileList(const std::string& path) {
	std::vector<std::string> file_list;
	for (auto it = boost::filesystem::recursive_directory_iterator(path);
	     it != boost::filesystem::recursive_directory_iterator(); ++it) {
		if (boost::filesystem::is_regular_file(*it)) {
			const boost::filesystem::path file_path = *it;
            file_list.push_back(file_path.string());	
		}
	}
	return file_list;
}

std::string TimeStampFromStr(const std::string &time_str) {
    int year, month, day, hour, minute, second;
    std::vector<std::string> split_elems = StringSplit(time_str, " ");
    if (split_elems.size() == 2) {
        std::vector<std::string> ymd_elems = StringSplit(split_elems[0], "-");

        // std::cout << "ymd_elems[0] = " << ymd_elems[0] << std::endl;
        year = std::stoi(ymd_elems[0]) - 1900;
        // std::cout << "ymd_elems[1] = " << ymd_elems[1] << std::endl;
        month = std::stoi(ymd_elems[1]) - 1;
        // std::cout << "ymd_elems[2] = " << ymd_elems[2] << std::endl;
        day = std::stoi(ymd_elems[2]);


        // std::cout << "split_elems[1].substr(0, 2) = " << split_elems[1].substr(0, 2) << std::endl;
        hour = std::stoi(split_elems[1].substr(0, 2));
        // std::cout << "split_elems[1].substr(3, 2) = " << split_elems[1].substr(3, 2) << std::endl;
        minute = std::stoi(split_elems[1].substr(3, 2));
        // std::cout << "split_elems[1].substr(6, 2) = " << split_elems[1].substr(6, 2) << std::endl;
        second = std::stoi(split_elems[1].substr(6, 2));


        // std::cout << "split_elems[1].substr(9, 3) = " << split_elems[1].substr(9, 3) << std::endl;
        std::string mcro_sceond = split_elems[1].substr(9, 3);

        struct tm timeinfo;
        timeinfo.tm_year = year;
        timeinfo.tm_mon = month;
        timeinfo.tm_mday = day;
        timeinfo.tm_hour = hour;
        timeinfo.tm_min = minute;
        timeinfo.tm_sec = second;
        timeinfo.tm_isdst = 0;
        time_t t = mktime(&timeinfo);
        return std::to_string(t) + mcro_sceond;
    } else {
        std::cout << "Convert Unix time stamp failed" << std::endl;
    }

    return "1609430400000";  //  20210101_000000
}

bool LoadBlueToothFolder(const std::string& path, 
                         std::unordered_map<double, std::vector<std::pair<std::string, int>>>& prior_bluetooth_time_signal) {
    prior_bluetooth_time_signal.clear();

    std::string bluetooth_files_path = path;
    std::vector<std::string> bluetooth_files;
    std::vector<std::string> items = StringSplit(bluetooth_files_path, ".");
    if (items.back() == "txt" || items.back() == "csv") {
        if (IsBluetoothFile(bluetooth_files_path))
            bluetooth_files.push_back(bluetooth_files_path);
        else {
            std::cout << "Error: its not Bluetooth txt ,Check its path :" << bluetooth_files_path << std::endl;
            return false;
        }
    } else {
        std::string endchar = bluetooth_files_path.substr(bluetooth_files_path.length() - 1);
        if (endchar == "/") {
            bluetooth_files_path.pop_back();
        }
        std::vector<std::string> files = GetRecursiveTMPFileList(bluetooth_files_path);
        for (const auto& file : files) {
            std::cout <<"file = " << file<<std::endl;
            items = StringSplit(file, ".");
            if (items.back() == "txt" || items.back() == "csv") {
                if (IsBluetoothFile(file)) {
                    bluetooth_files.push_back(file);
                }
            }
        }
        if (bluetooth_files.size() == 0) {
            std::cout << "Error: no bluetooth file in path" << bluetooth_files_path << std::endl;
        }
    }

    
    if (bluetooth_files.empty()) {
        return false;
    }


    for (const auto& file_path : bluetooth_files) {
        if (!IsBluetoothFile(file_path)) {
            std::cout << "Warning: its not bluetooth file " << file_path << std::endl;
            continue;
        }

        std::ifstream testdatafile(file_path);
        std::string linedata;
        std::vector<std::string> one_signal_line;
        while (getline(testdatafile, linedata)) {
            one_signal_line.clear();
            one_signal_line = StringSplit(linedata, ",");
            //      BeaconSignaldata beacondata;
            //      GetdataFromLinedata(linedata ,beacondata);
            double timestamp = std::stod(TimeStampFromStr(one_signal_line[0]));
            // double timestamp = one_signal_line.size() == 9 ? std::stod(one_signal_line[1]) : std::stod(one_signal_line[9]);
            std::string major_minor = one_signal_line[4] + "-" + one_signal_line[5];
            int rssi = std::stoi(one_signal_line[8]);
            if (rssi == 0) {
                rssi = -97;
            }

            // if (prior_bluetooth_time_signal.count(timestamp)) {
            //     std::cout << "Warning: Dumplicated BlueTooth Collection!" << std::endl;
            // }

            prior_bluetooth_time_signal[timestamp].emplace_back(std::make_pair(major_minor, rssi));
        }
    }

    return true;
}

bool LoadFeatures(FeatureDataContainer &feature_data_container, Configurator &param, const std::string workspace_path) {
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);
    bool have_matched = boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"));

    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        if (!have_matched) {
            feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        } else {
            feature_data_container.ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path, "/features.bin"));
        }
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_images.bin"))) {
            feature_data_container.ReadLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
        }
        exist_feature_file = true;
    }

    // If the camera model is Spherical
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
        }
    }

    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
            feature_data_container.ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
        }
    }

    // Check the AprilTag detection file exist or not
    if (exist_feature_file && static_cast<bool>(param.GetArgument("detect_apriltag", 0))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/apriltags.bin"))) {
            feature_data_container.ReadAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain AprilTags data" << std::endl;
        }
    }

    // Check the GPS file exist or not.
    if (exist_feature_file && use_gps_prior) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/gps.bin"))) {
            feature_data_container.ReadGPSBinaryData(JoinPaths(workspace_path, "/gps.bin"));
        }
    }

    if (exist_feature_file) {
        return true;
    } else {
        return false;
    }
}

bool LoadGlobalFeatures(FeatureDataContainer &feature_data_container, Configurator &param, const std::string workspace_path){
    CHECK(!workspace_path.empty());

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/vlad_vectors.bin"))){
        feature_data_container.ReadGlobalFeaturesBinaryData(workspace_path + "/vlad_vectors.bin");
        return true;
    }
    else{
        return false;
    }
}


bool LoadMatches(FeatureDataContainer &feature_data_container, 
                            SceneGraphContainer &scene_graph,
                            Configurator &param, 
                            const std::string workspace_path) {
    CHECK(!workspace_path.empty());
    
    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

    if (!boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        std::cout << "input files is not exist ..." << std::endl;
        return false;
    }

    // load match data
    scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/two_view_geometry.bin"))) {
        scene_graph.ReadImagePairsBinaryData(JoinPaths(workspace_path, "/two_view_geometry.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/strong_loops.bin"))) {
        std::cout<<"read strong_loops file "<<std::endl;
        scene_graph.ReadStrongLoopsBinaryData(JoinPaths(workspace_path, "/strong_loops.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/loop_pairs.bin"))) {
        std::cout<<"read loop_pairs file "<<std::endl;
        scene_graph.ReadLoopPairsInfoBinaryData(JoinPaths(workspace_path, "/loop_pairs.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/normal_pairs.bin"))) {
        std::cout<<"read normal_pairs file "<<std::endl;
        scene_graph.ReadNormalPairsBinaryData(JoinPaths(workspace_path, "/normal_pairs.bin"));
    }

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph.Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph.Cameras();

    std::vector<image_t> image_ids = feature_data_container.GetImageIds();

    for (const auto image_id : image_ids) {
        const Image &image = feature_data_container.GetImage(image_id);
        if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        images[image_id] = image;

        const FeatureKeypoints &keypoints = feature_data_container.GetKeypoints(image_id);
        images[image_id].SetPoints2D(keypoints);
        const PanoramaIndexs &panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);

        const Camera &camera = feature_data_container.GetCamera(image.CameraId());

        std::vector<uint32_t> local_image_indices(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        images[image_id].SetLocalImageIndices(local_image_indices);

        if (!scene_graph.ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
        }

        if (scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(scene_graph.CorrespondenceGraph()->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(
                scene_graph.CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
        } else {
            std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
        }
    }
    scene_graph.CorrespondenceGraph()->Finalize();

    return true;
}

void ReadSlamPoses(const std::string path, std::map<std::string,std::vector<double>>& slam_poses){
    std::cout<<"Read "<<path<<std::endl;
    std::ifstream infile(path);
    CHECK(infile.is_open());

    std::string line;
    std::string item;

    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        StringTrim(&line);

        std::stringstream line_stream(line);

        std::getline(line_stream, item, ' ');
        std::string img_name = item;

        std::vector<double> pose;

        std::getline(line_stream, item, ' ');
        // std::cout<<"tx "<<item<<std::endl;
        double tx = std::stold(item);
        // std::cout<<std::setprecision(17)<<"tx "<<tx<<std::endl;
        pose.push_back(tx);
        std::getline(line_stream, item, ' ');
        double ty = std::stold(item);
        // std::cout<<"ty "<<ty<<std::endl;
        pose.push_back(ty);
        std::getline(line_stream, item, ' ');
        double tz = std::stold(item);
        // std::cout<<"tz "<<tz<<std::endl;
        pose.push_back(tz);


        std::getline(line_stream, item, ' ');
        double rw = std::stold(item);
        // std::cout<<"rw "<<rw<<std::endl;
        pose.push_back(rw);
        std::getline(line_stream, item, ' ');
        double rx = std::stold(item);
        // std::cout<<"rx "<<rx<<std::endl;
        pose.push_back(rx);
        std::getline(line_stream, item, ' ');
        double ry = std::stold(item);
        // std::cout<<"ry "<<ry<<std::endl;
        pose.push_back(ry);
        std::getline(line_stream, item, ' ');
        double rz = std::stold(item);
        // std::cout<<"rz "<<rz<<std::endl;
        pose.push_back(rz);

        slam_poses[img_name] = pose;
    }
    infile.close();
}

// SfM
bool FeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param,
                       struct HybridOptions hybrid_options = HybridOptions()) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    if (!boost::filesystem::exists(workspace_path)){
        boost::filesystem::create_directories(workspace_path);
    } 
    // CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    if (hybrid_options.child_id < 0){
        option_parser.GetImageReaderOptions(reader_options, param);
    } else {
        option_parser.GetImageReaderOptions(reader_options, param, hybrid_options.child_id);
        if (!hybrid_options.save_flag){
            workspace_path = JoinPaths(workspace_path, reader_options.child_path);
        }
        if (!boost::filesystem::exists(workspace_path)){
            boost::filesystem::create_directories(workspace_path);
        }
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        // only quit not in update mode
        CHECK(LoadFeatures(feature_data_container,param,workspace_path));
        std::cout << "Load Features from " << workspace_path << std::endl;
        return true;
    }

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

    std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");
    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

    SiftExtractionOptions sift_extraction;
    option_parser.GetFeatureExtractionOptions(sift_extraction, param);

    std::string panorama_config_file = param.GetArgument("panorama_config_file", "");
    if (!panorama_config_file.empty()) {
        std::vector<PanoramaParam> panorama_params;
        LoadParams(panorama_config_file, panorama_params);
        sift_extraction.panorama_config_params = panorama_params;
        sift_extraction.use_panorama_config = true;
    }

    //TODO: get start image_id & camera_id
    image_t start_image_id = 0;
    camera_t start_camera_id = 0;
    label_t start_label_id = 0;
    if (!feature_data_container.GetImageIds().empty()) {
        // std::string gps_origin_str = JoinPaths(workspace_path, "/gps_origin.txt");
        std::vector<double> vec_gps_origin;
        if (use_gps_prior){
            reader_options.gps_origin = gps_origin_str;
            std::cout << "ReaderOptions set ori_gps_origin: " << reader_options.gps_origin << std::endl;
        }
        for(const auto old_image_id: feature_data_container.GetImageIds()){
            if(start_image_id < old_image_id){
                start_image_id = old_image_id;
            }
            std::string image_name = feature_data_container.GetImage(old_image_id).Name();
            if (hybrid_options.update_flag){
                feature_data_container.GetImage(image_name).SetLabelId(0);
            } else {
                label_t label_id = feature_data_container.GetImage(old_image_id).LabelId();
                if(label_id > start_label_id){
                    start_label_id = label_id;
                }
            }
        }
        ++start_image_id;
        start_camera_id = feature_data_container.NumCamera();
    }
    ++start_label_id;


    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, &feature_data_container, start_image_id, start_camera_id, start_label_id);
    feature_extractor.Start();
    feature_extractor.Wait();
    fprintf(
        fs, "%s\n",
        StringPrintf("Feature Extraction Elapsed time: %.3f [minutes]", feature_extractor.GetTimer().ElapsedMinutes())
            .c_str());
    fflush(fs);

    Timer timer;

    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type != RetrieveType::SIFT) {
        // Pca training using the extrated feature descriptors
        timer.Start();

        std::cout << "Collect training descriptors " << std::endl;
        FeatureDescriptors training_descriptors;
        size_t training_descriptors_count = 0;

        const std::vector<image_t>& whole_image_ids = feature_data_container.GetImageIds();
        const std::vector<image_t>& new_image_ids = feature_data_container.GetNewImageIds();
        std::vector<image_t> old_image_ids;
        std::set_difference(whole_image_ids.begin(), whole_image_ids.end(), new_image_ids.begin(), new_image_ids.end(),
                            std::inserter(old_image_ids,old_image_ids.begin()));

        std::cout<<"whole_image_ids: " << whole_image_ids.size() << " new_image_ids: " << new_image_ids.size() << 
                    " old_image_ids: " << old_image_ids.size() << std::endl;

        Eigen::Matrix<double, 128, 1> embedding_thresholds;
        std::string pca_matrix_path = param.GetArgument("pca_matrix_path", "");

        //TODO: if exist pca matrix already & map update, read matrix
        bool load_pca = false;
        bool enable_whole_pca = true;
        size_t existed_feature_dimension;
        if(old_image_ids.size() == 0){
            existed_feature_dimension=static_cast<int>(param.GetArgument("compressed_feature_dimension", 128));;
        }else{
            existed_feature_dimension = feature_data_container.GetCompressedDescriptors(old_image_ids[0]).cols();
        } 
        std::cout<<"existed_feature_dimension: "<<existed_feature_dimension<<std::endl;
        if(old_image_ids.size() != 0 && existed_feature_dimension != 128){
            std::cout << "Original Map Feature already compressed, Cannot training pca with old descriptors" << std::endl;
            enable_whole_pca = false;
        }

        // CHECK(enable_whole_pca) << existed_feature_dimension;
        // compress descriptors using the trained PCA matrix
        PrintHeading1("Compressing descriptors");

        int compressed_feature_dimension = static_cast<int>(param.GetArgument("compressed_feature_dimension", 128));
        CHECK(compressed_feature_dimension == 128 || compressed_feature_dimension == 64 ||
            compressed_feature_dimension == 32);
        if(compressed_feature_dimension != existed_feature_dimension){
            std::cout<<" WARN!!!! existed_feature_dimension is "<<existed_feature_dimension<<" not equal to yaml setting "<<compressed_feature_dimension<<std::endl;
            compressed_feature_dimension = existed_feature_dimension;
            std::cout<<" set compressed_feature_dimension to "<<compressed_feature_dimension<<std::endl;
        }

        if(!has_pca && enable_whole_pca){
            auto image_ids = whole_image_ids;

            if(compressed_feature_dimension != 128){
                int pca_training_feature_count = static_cast<int> (param.GetArgument("pca_training_feature_count", 1000000));
                uint64_t total_feature_count = 0;
                for (int i = 0; i < image_ids.size(); ++i){
                    image_t current_id = image_ids[i];
                    const auto &keypoints = feature_data_container.GetKeypoints(current_id); 

                    total_feature_count += keypoints.size();
                }

                int sample_step = 1;
                if(pca_training_feature_count < total_feature_count){
                    sample_step = total_feature_count / pca_training_feature_count;
                }

                for (int i = 0; i < image_ids.size(); ++i) {
                    image_t current_id = image_ids[i];
                    const auto &descriptors = feature_data_container.GetDescriptors(current_id);
                    size_t sampled_descriptors_count = descriptors.rows() / sample_step;

                    training_descriptors.conservativeResize(training_descriptors_count + sampled_descriptors_count, descriptors.cols());

                    for (size_t j = 0; j < sampled_descriptors_count; j++) {
                        training_descriptors.row(training_descriptors_count + j) = descriptors.row(j * sample_step);
                    }
                    training_descriptors_count += sampled_descriptors_count;
                }
                
                // Eigen::Matrix<double, 128, 128> pca_matrix;
                // Eigen::Matrix<double, 128, 1> embedding_thresholds;

                std::cout << "PCA training " << std::endl;  
                PcaTraining(training_descriptors,pca_matrix,embedding_thresholds);
                has_pca = true;
                std::cout << StringPrintf("PCA training  in %.3f min", timer.ElapsedMinutes()) << std::endl;

                fprintf(fs, "%s\n", StringPrintf("PCA training Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
                fflush(fs);

                // std::string pca_matrix_path = param.GetArgument("pca_matrix_path", "");
                if (pca_matrix_path.empty()) {
                    pca_matrix_path = workspace_path + "pca_matrix.bin";
                }
                std::ofstream file(pca_matrix_path, std::ios::binary);
                CHECK(file.is_open()) << pca_matrix_path;

                for(int i = 0; i< 128; ++i){
                    for(int j= 0; j< 128; ++j){
                        double elem = pca_matrix(i,j);
                        file.write((char*)&elem, sizeof(double));
                    }
                }
                for(int i = 0; i < 128; ++i){
                    double elem = embedding_thresholds(i);
                    file.write((char*)&elem,sizeof(double));
                }

                file.close();
                CHECK(boost::filesystem::exists(pca_matrix_path));
            }
        
        }
        
        auto image_ids = new_image_ids;

        // int compressed_feature_dimension = static_cast<int>(param.GetArgument("compressed_feature_dimension", 128));
        // CHECK(compressed_feature_dimension == 128 || compressed_feature_dimension == 64 ||
        //       compressed_feature_dimension == 32);

        std::cout<<"compressing images size "<<image_ids.size()<<std::endl;
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            auto &descriptors = feature_data_container.GetDescriptors(current_id);
            auto &compressed_descriptors = feature_data_container.GetCompressedDescriptors(current_id);

            if (compressed_feature_dimension == 128) {
                compressed_descriptors = descriptors;
            } else {
                CompressFeatureDescriptors(descriptors, compressed_descriptors, pca_matrix, embedding_thresholds,
                                        compressed_feature_dimension);
            }
            descriptors.resize(0, 0);
        }
        std::cout << StringPrintf("Compressing descriptors in %.3f min", timer.ElapsedMinutes()) << std::endl;

        fprintf(fs, "%s\n", StringPrintf("Compressing descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
        fflush(fs);
    } else {
        const std::vector<image_t>& new_image_ids = feature_data_container.GetNewImageIds();
        for (int i = 0; i < new_image_ids.size(); ++i) {
            image_t current_id = new_image_ids[i];
            auto &descriptors = feature_data_container.GetDescriptors(current_id);
            auto &compressed_descriptors = feature_data_container.GetCompressedDescriptors(current_id);
            std::swap(descriptors, compressed_descriptors);
        }
    }

    timer.Start();
    bool write_feature = static_cast<bool>(param.GetArgument("write_feature", 1));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));
    if (write_feature) {
        // if map_update, save features to map_update folder
        feature_data_container.WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        feature_data_container.WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
        feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));

        if ((static_cast<bool>(param.GetArgument("map_update", 0)))||
            (reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE")) {
            feature_data_container.WritePieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        }
        if (sift_extraction.detect_apriltag) {
            // Check the Arpiltag Detect Result
            if (feature_data_container.ExistAprilTagDetection()) {
                feature_data_container.WriteAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
            } else {
                std::cout << "Warning: No Apriltag Detection has been found ... " << std::endl;
            }
        }
        if (use_gps_prior) {
            feature_data_container.WriteGPSBinaryData(JoinPaths(workspace_path, "/gps.bin"));
        }
    }
    fprintf(fs, "%s\n", StringPrintf("Write local descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    
    if (use_gps_prior) {
        image_t geo_image_idx = feature_data_container.GetGeoImageIndex();
        if (feature_data_container.ExistImage(geo_image_idx) || !reader_options.gps_origin.empty()) {
            double latitude, longitude, altitude;
            if (feature_data_container.ExistImage(geo_image_idx)){
                const class Image& image = feature_data_container.GetImage(geo_image_idx);
                std::string image_path = JoinPaths(reader_options.image_path, image.Name());
                Bitmap bitmap;
                bitmap.Read(image_path);
                // double latitude, longitude, altitude;
                bitmap.ExifLatitude(&latitude);
                bitmap.ExifLongitude(&longitude);
                bitmap.ExifAltitude(&altitude);
            } else {
                std::vector<double> gps_origin = CSVToVector<double>(reader_options.gps_origin);
                latitude = gps_origin[0];
                longitude = gps_origin[1];
                altitude = gps_origin[2];
            }
            GeodeticConverter geo_converter(latitude, longitude, altitude);
            Eigen::Matrix3x4d M = geo_converter.NedToEcefMatrix();

            std::ofstream file(JoinPaths(workspace_path, "/ned_to_ecef.txt"), std::ofstream::out);
            file << MAX_PRECISION << M(0, 0) << " " << M(0, 1) << " " 
                << M(0, 2) << " " << M(0, 3) << std::endl;
            file << MAX_PRECISION << M(1, 0) << " " << M(1, 1) << " " 
                << M(1, 2) << " " << M(1, 3) << std::endl;
            file << MAX_PRECISION << M(2, 0) << " " << M(2, 1) << " " 
                << M(2, 2) << " " << M(2, 3) << std::endl;
            file.close();
            std::ofstream file1(JoinPaths(workspace_path, "/gps_origin.txt"));
            file1 << MAX_PRECISION << latitude << " " << longitude << " " << altitude << std::endl;
            file1.close();

            std::stringstream gps_origin_stream;
            gps_origin_stream << MAX_PRECISION << latitude << "," << longitude << "," << altitude << std::endl;
            gps_origin_str = gps_origin_stream.str();
        }
    }

    return true;
}

bool GlobalFeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param,
                             bool new_local_feature_extraction, 
                             struct HybridOptions hybrid_options = HybridOptions()) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    GlobalFeatureExtractionOptions options;

    option_parser.GetGlobalFeatureExtractionOptions(options, param);

    // if (boost::filesystem::exists(JoinPaths(workspace_path, "/vlad_vectors.bin"))) {
    //     // only quit not in update mode
    //     if (!static_cast<bool>(param.GetArgument("map_update", 0))){
    //         std::cout << "Global feature already exists, skip feature extraction" << std::endl;
    //         return false;
    //     }
    // }

    // if(!new_local_feature_extraction){
    //     std::string feature_data_path;
    //     if (static_cast<bool>(param.GetArgument("map_update", 0))){
    //         feature_data_path = JoinPaths(workspace_path, "/map_update");
    //     }else{
    //         feature_data_path = workspace_path;
    //     }
    //     CHECK(LoadFeatures(feature_data_container,param,feature_data_path));
    // }

    const std::vector<image_t> &whole_image_ids = feature_data_container.GetImageIds();

    std::string vlad_code_book_path = options.vlad_code_book_path;

    //TODO: if exist vald matrix already & map update, read matrix
    bool load_vlad = false;
    // size_t existed_feature_dimension = feature_data_container.GetCompressedDescriptors(whole_image_ids[0]).cols();
    // if (static_cast<bool>(param.GetArgument("map_update", 0))) {
    //     if(boost::filesystem::exists(vlad_code_book_path)){
    //         load_vlad = true;
    //     }
    // }


    // vlad code book training using the extracted descriptors
    Timer timer;
    if(!load_vlad){
        timer.Start();
        VladVisualIndex vlad_visual_index;
        VladVisualIndex::CodeBookCreateOptions code_book_create_option;
        code_book_create_option.num_vocabulary = param.GetArgument("vlad_num_vocabulary",256);

        std::cout << "Collect training descriptors " << std::endl;

        CompressedFeatureDescriptors training_descriptors;
        size_t training_descriptors_count = 0;

        // const std::vector<image_t> &image_ids = feature_data_container.GetImageIds();
        auto image_ids = whole_image_ids;

        int vlad_training_feature_count = static_cast<int>(param.GetArgument("vlad_training_feature_count", 1000000));
        int total_feature_count = 0;
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            const auto &keypoints = feature_data_container.GetKeypoints(current_id);
            total_feature_count += keypoints.size();
        }

        int sample_step = 1;
        if (vlad_training_feature_count < total_feature_count) { 
            sample_step = total_feature_count / vlad_training_feature_count;
        }

        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            const auto &descriptors = feature_data_container.GetCompressedDescriptors(current_id);
            size_t sampled_descriptors_count = descriptors.rows() / sample_step;

            training_descriptors.conservativeResize(training_descriptors_count + sampled_descriptors_count,
                                                    descriptors.cols());

            for (size_t j = 0; j < sampled_descriptors_count; j++) {
                training_descriptors.row(training_descriptors_count + j) = descriptors.row(j * sample_step);
            }
            training_descriptors_count += sampled_descriptors_count;
        }
        VladVisualIndex::Descriptors float_training_descriptors;
        CompressedFeatureDescriptorsTofloat(training_descriptors, float_training_descriptors);
        std::cout << "training descriptor count and dimension: " << float_training_descriptors.rows() << " "
                << float_training_descriptors.cols() << std::endl;


        std::cout << "Kmeans to create code book " << std::endl;
        // create code book
        vlad_visual_index.CreateCodeBook(code_book_create_option, float_training_descriptors);
        std::cout << StringPrintf("Create code book in %.3f min", timer.ElapsedMinutes()) << std::endl;

        fprintf(fs, "%s\n", StringPrintf("Create code book in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
        fflush(fs);
        
        vlad_visual_index.SaveCodeBook(options.vlad_code_book_path);
    }

    timer.Start();
    PrintHeading1("Global feature extraction");
    GlobalFeatureExtractor global_feature_extractor(options, &feature_data_container);
    global_feature_extractor.Run();
    feature_data_container.WriteGlobalFeaturesBinaryData(workspace_path + "/vlad_vectors.bin");

    fprintf(fs, "%s\n", StringPrintf("Extract Global descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);

    return true;
}

void FeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     Configurator &param, struct HybridOptions hybrid_options = HybridOptions()) {
    using namespace std::chrono;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    std::string workspace_path = param.GetArgument("workspace_path", "");

    if (hybrid_options.child_id > -1){
        ImageReaderOptions reader_options;
        option_parser.GetImageReaderOptions(reader_options, param, hybrid_options.child_id);

        if (!hybrid_options.save_flag){
            workspace_path = JoinPaths(workspace_path, reader_options.child_path);
        }
    }
    CHECK(boost::filesystem::exists(workspace_path));

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        std::cout << "Scene graph already exists, skip feature matching" << std::endl;
        LoadMatches(feature_data_container, scene_graph, param, workspace_path);
        return;
    }

    FeatureMatchingOptions options;
    option_parser.GetFeatureMatchingOptions(options, param);

    if (hybrid_options.child_id > 0){
        options.match_between_reconstructions_ = true;
        options.delete_duplicated_images_ = false;
    }

    if (options.method_ == FeatureMatchingOptions::MatchMethod::SPATIAL) {
        size_t num_image_has_prior = 0;
        auto image_ids = feature_data_container.GetImageIds();
        for (auto image_id : image_ids) {
            Image image = feature_data_container.GetImage(image_id);
            num_image_has_prior += !!image.HasTvecPrior();
        }
        float prior_ratio = num_image_has_prior * 1.0 / image_ids.size();
        if (prior_ratio < 0.999) {
            options.method_ = FeatureMatchingOptions::MatchMethod::VOCABTREE;
            std::cout << StringPrintf("The number of image that has prior pose is %f(<0.999), switch to vocabtree!\n");
        }
    }

    // use intial sfm to filter far image pairs.
    bool use_initial_sfm = static_cast<bool>(param.GetArgument("use_initial_sfm", 0));

    bool has_initial_sfm = false;
    if (boost::filesystem::is_directory(workspace_path + "/initial_sfm") &&
        boost::filesystem::exists(workspace_path + "/initial_sfm/cameras.bin") &&
        boost::filesystem::exists(workspace_path + "/initial_sfm/images.bin") &&
        boost::filesystem::exists(workspace_path + "/initial_sfm/points3D.bin")) {
        has_initial_sfm = true;
    }

    if (use_initial_sfm && has_initial_sfm) {
        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;
        std::unordered_map<image_t, bool> prior_pose_validations;
        auto reconstruction = std::make_shared<Reconstruction>();
        reconstruction->ReadReconstruction(workspace_path + "/initial_sfm");

        std::vector<image_t> image_ids = feature_data_container.GetImageIds();

        for (const auto image_id : image_ids) {
            if (!(reconstruction->ExistsImage(image_id) && reconstruction->IsImageRegistered(image_id))) {
                prior_pose_validations.emplace(image_id, false);
                continue;
            }
            const auto &image = reconstruction->Image(image_id);
            prior_rotations.emplace(image_id, image.Qvec());
            prior_translations.emplace(image_id, image.Tvec());
            prior_pose_validations.emplace(image_id, true);
        }

        reconstruction->ComputeBaselineDistance();
        for (const auto image_id : image_ids) {
            if (!(reconstruction->ExistsImage(image_id) && reconstruction->IsImageRegistered(image_id))) {
                continue;
            }

            Eigen::Vector4d qvec = prior_rotations.at(image_id);
            Eigen::Vector3d tvec = prior_translations.at(image_id);

            Eigen::Vector3d C = ProjectionCenterFromPose(qvec, tvec);

            bool valid = false;
            if (image_id > 1 && prior_rotations.find(image_id - 1) != prior_rotations.end()) {
                Eigen::Vector4d qvec_previous = prior_rotations.at(image_id - 1);
                Eigen::Vector3d tvec_previous = prior_translations.at(image_id - 1);
                Eigen::Vector3d C_previous = ProjectionCenterFromPose(qvec_previous, tvec_previous);

                double distance = (C - C_previous).norm();
                if (distance < reconstruction->baseline_distance * 30) {
                    valid = true;
                }
            }
            if (!valid && prior_rotations.find(image_id + 1) != prior_rotations.end()) {
                Eigen::Vector4d qvec_next = prior_rotations.at(image_id + 1);
                Eigen::Vector3d tvec_next = prior_translations.at(image_id + 1);
                Eigen::Vector3d C_next = ProjectionCenterFromPose(qvec_next, tvec_next);

                double distance = (C - C_next).norm();
                if (distance < reconstruction->baseline_distance * 30) {
                    valid = true;
                }
            }

            if (!valid) {
                CHECK(prior_pose_validations.find(image_id) != prior_pose_validations.end());
                prior_pose_validations.at(image_id) = false;
            }
        }

        options.prior_neighbor_distance = reconstruction->baseline_distance;
        options.have_prior_pose_ = true;
        options.prior_rotations = prior_rotations;
        options.prior_translations = prior_translations;
        options.prior_pose_validations = prior_pose_validations;
        options.max_match_distance = param.GetArgument("max_match_distance", 20.0f);
    }
    
    // use bluetooth signal filter far image pairs.
    bool use_prior_bluetooth = static_cast<bool>(param.GetArgument("use_prior_bluetooth", 0));
    std::string bluetooth_prior_folder = param.GetArgument("prior_bluetooth_path", "");
    double prior_bluetooth_threshold_inside = param.GetArgument("prior_bluetooth_threshold_inside", 3.0f);
    double prior_bluetooth_threshold_outside = param.GetArgument("prior_bluetooth_threshold_outside", 7.0f);
    double prior_bluetooth_threshold_outlier = param.GetArgument("prior_bluetooth_threshold_outlier", 9.0f);
    
    if (use_prior_bluetooth) {
        options.have_prior_bluetooth_ = true;
        options.prior_bluetooth_threshold_inside = prior_bluetooth_threshold_inside;
        options.prior_bluetooth_threshold_outside = prior_bluetooth_threshold_outside;
        options.prior_bluetooth_threshold_outlier = prior_bluetooth_threshold_outlier;
        if (!LoadBlueToothFolder(bluetooth_prior_folder, options.prior_bluetooth_time_signal)){
            std::cout << "ERROR: Bluetooth Config Failed" << std::endl;
            // FIXME: May not need exit
            exit(-1);
        }
    }

    // If gps prior is available
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    std::string gps_prior_file = param.GetArgument("gps_prior_file", "");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";
    if (use_gps_prior) {
        if (boost::filesystem::exists(gps_prior_file)) {
            auto image_ids = feature_data_container.GetImageIds();
            std::vector<std::string> image_names;
            for (const auto image_id : image_ids) {
                const Image &image = feature_data_container.GetImage(image_id);
                std::string name = image.Name();
                image_names.push_back(name);
            }

            std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> gps_locations;
            LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, true);
            
            std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> image_locations;
            GPSLocationsToImages(gps_locations, image_names, image_locations);
            std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
            std::cout << image_locations.size() << " images have gps prior" << std::endl;

            std::vector<PlyPoint> gps_locations_ply;
            for (const auto image_id : image_ids) {
                const Image &image = feature_data_container.GetImage(image_id);
                std::string name = image.Name();

                if (image_locations.find(name) != image_locations.end()) {
                    prior_locations_gps.emplace(image_id, image_locations.at(name));

                    PlyPoint gps_location_ply;
                    gps_location_ply.r = 255;
                    gps_location_ply.g = 0;
                    gps_location_ply.b = 0;
                    gps_location_ply.x = image_locations.at(name).first[0];
                    gps_location_ply.y = image_locations.at(name).first[1];
                    gps_location_ply.z = image_locations.at(name).first[2];
                    gps_locations_ply.push_back(gps_location_ply);
                }
            }
            options.prior_locations = prior_locations_gps;
            sensemap::WriteBinaryPlyPoints(workspace_path + "/gps-match.ply", gps_locations_ply);
            options.have_prior_location_ = true;
            options.max_distance_for_loop = param.GetArgument("max_distance_for_loop", 20.0f);
        }
    }

    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    MatchDataContainer match_data;
    FeatureMatcher matcher(options, &feature_data_container, &match_data, &scene_graph);
    std::cout << "matching ....." << std::endl;
    matcher.Run();
    std::cout << "matching done" << std::endl;
    std::cout << "build graph" << std::endl;
    matcher.BuildSceneGraph();
    std::cout << "build graph done" << std::endl;

    high_resolution_clock::time_point end_time = high_resolution_clock::now();

    fprintf(fs, "%s\n",
            StringPrintf("Feature Matching Elapsed time: %.3f [minutes]",
                         duration_cast<microseconds>(end_time - start_time).count() / 6e7)
                .c_str());
    fflush(fs);

    scene_graph.CorrespondenceGraph()->ExportToGraph(workspace_path + "/scene_graph.png");
    std::cout << "ExportToGraph done!" << std::endl;

    bool write_match = static_cast<bool>(param.GetArgument("write_match", 1));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));

    if (write_match) {
        scene_graph.WriteSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
        // scene_graph.WriteImagePairsBinaryData(workspace_path + "/two_view_geometry.bin");
        scene_graph.WriteBlueToothPairsInfoBinaryData(workspace_path + "/bluetooth_info.bin");
        scene_graph.WriteLoopPairsInfoBinaryData(workspace_path + "/loop_pairs.bin");
        scene_graph.WriteNormalPairsBinaryData(workspace_path + "/normal_pairs.bin");
        scene_graph.WriteStrongLoopsBinaryData(workspace_path + "/strong_loops.bin");
    }

}

bool RTKReady(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container, double threshold = 0.8) {
    auto image_ids = scene_graph_container->GetImageIds();
    double ready_num = 0;
    for (auto id : image_ids) {
        auto image = scene_graph_container->Image(id);
        if (image.RtkFlag() == 50) {
            ready_num++;
        }
    }
    std::cout << "RTKReady: " << ready_num << " / " << image_ids.size() << std::endl;
    return ready_num / image_ids.size() > threshold;
}

void DirectedSFM(std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                 std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Dierected Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::DIRECTED;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    option_parser.GetMapperOptions(options->independent_mapper_options,param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";
    // if (static_cast<bool>(param.GetArgument("map_update", 0))){
    //     options->independent_mapper_options.map_update = true;
    //     auto reconstruction_idx = reconstruction_manager->Add();
    //     std::shared_ptr<Reconstruction> reconstruction = reconstruction_manager->Get(reconstruction_idx);
    //     auto cameras = scene_graph_container->Cameras();
    //     bool camera_rig = false;
    //     for(auto camera : cameras){
    //         if(camera.second.NumLocalCameras()>1){
    //             camera_rig = true;
    //         }
    //     }
    //     reconstruction->ReadReconstruction(workspace_path + "/0",camera_rig);
    //     // set original image to label 0
    //     auto old_image_ids = reconstruction->RegisterImageIds();
    //     for (auto old_image_id : old_image_ids) {
    //         reconstruction->Image(old_image_id).SetLabelId(0);
    //         reconstruction->Image(old_image_id).SetPoseFlag(true);
    //     }
    //     workspace_path = JoinPaths(workspace_path, "/map_update");
    // }

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    bool color_harmonization = param.GetArgument("color_harmonization", 0);
    
    size_t base_reconstruction_idx = reconstruction_manager->Size();

    auto independent_options = std::make_shared<IndependentMapperOptions>(
            options->independent_mapper_options);

    if(independent_options->direct_mapper_type == 0){
        return ;
    }

    while(true) {

        if(!RTKReady(scene_graph_container, 0.4) &&
            (independent_options->direct_mapper_type == 2 || independent_options->direct_mapper_type == 4)){
            break;
        }

        auto *mapper = new DirectedMapperController(independent_options, image_path, workspace_path,
                scene_graph_container, reconstruction_manager);
        mapper->Start();
        mapper->Wait();
        
        fprintf(
                fs, "%s\n",
                StringPrintf("Directed Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
        fflush(fs);

        if (mapper->IsSuccess()) {

            std::set<image_t> clustered_image_ids;
            for (auto & image_id : scene_graph_container->GetImageIds()) {
                clustered_image_ids.insert(image_id);
            }
            auto reconstruction = reconstruction_manager->Get(reconstruction_manager->Size() - 1);
            for (auto & image_id : reconstruction->RegisterImageIds()) {
                clustered_image_ids.erase(image_id);
            }

            std::shared_ptr<SceneGraphContainer> cluster_graph_container =
                std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
            scene_graph_container->ClusterSceneGraphContainer(clustered_image_ids, *cluster_graph_container.get());

            std::swap(scene_graph_container, cluster_graph_container);

            std::cout << StringPrintf("Remain %d images to be reconstructed!\n", scene_graph_container->NumImages());
        }
        if (!mapper->IsSuccess() || scene_graph_container->NumImages() == 0 || 
            !options->independent_mapper_options.multiple_models) {
            break;
        }
    }
    for (size_t i = base_reconstruction_idx; i < reconstruction_manager->Size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(i);
        if (reconstruction->RegisterImageIds().size() <= 0) {
            continue;
        }

        if (color_harmonization) {
            reconstruction->ColorHarmonization(image_path);
        }

        // CHECK(reconstruction->RegisterImageIds().size()>0);
        const image_t first_image_id = reconstruction->RegisterImageIds()[0]; 
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image& image = reconstruction->Image(first_image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (options->independent_mapper_options.extract_colors) {
            reconstruction->ExtractColorsForAllImages(image_path);
        }
        std::cout << "export to " << JoinPaths(rec_path, MAPPOINT_NAME) << std::endl;
        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));

        if (camera.NumLocalCameras() > 1) {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);
            rig_reconstruction.WriteReconstruction(export_rec_path,
                options->independent_mapper_options.write_binary_model);
        } else {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);
        }
    }
}

void IncrementalSFM(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                    std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, 
                    Configurator &param, struct HybridOptions hybrid_options = HybridOptions()) {
    using namespace sensemap;

    PrintHeading1("Incremental Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::INCREMENTAL;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetMapperOptions(options->independent_mapper_options, param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";
    if (hybrid_options.child_id > -1){
        option_parser.GetImageReaderOptions(reader_options, param, hybrid_options.child_id);
        if (!hybrid_options.save_flag){
            workspace_path = JoinPaths(workspace_path, reader_options.child_path);
        }
    } else {
        option_parser.GetImageReaderOptions(reader_options, param);
    }
    
    options->independent_mapper_options.map_update = hybrid_options.update_flag;
    if (hybrid_options.child_id > 0){
        // option_parser.GetImageReaderOptions(reader_options, param, hybrid_options.child_id);
        std::shared_ptr<Reconstruction> reconstruction = reconstruction_manager->Get(0);

        // set original image to label 0
        auto old_image_ids = reconstruction->RegisterImageIds();
        for (auto old_image_id : old_image_ids) {

            class Image & old_image = reconstruction->Image(old_image_id);
            old_image.SetLocalImageIndices(scene_graph_container->Image(old_image_id).LocalImageIndices());
            old_image.SetLabelId(0);
            old_image.SetPoseFlag(true);
        }
    }

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    // use prior pose from slam to constrain SfM
    bool use_slam_graph = static_cast<bool>(param.GetArgument("use_slam_graph", 0));
    std::string preview_pose_file = param.GetArgument("preview_pose_file", "");
    if (use_slam_graph && (!preview_pose_file.empty()) && boost::filesystem::exists(preview_pose_file)) {
        std::vector<Keyframe> keyframes;
        if (boost::filesystem::path(preview_pose_file).extension().string() == ".tum") {
            LoadPriorPoseFromTum(preview_pose_file, keyframes, (param.GetArgument("image_type", "").compare("rgbd") == 0));
        }
        else {
            LoadPirorPose(preview_pose_file, keyframes);
        }

        std::unordered_map<std::string, Keyframe> keyframe_map;
        for (auto const keyframe : keyframes) {
            keyframe_map.emplace(keyframe.name, keyframe);
        }

        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

        // std::string camera_rig_params_file = param.GetArgument("rig_params_file", "");
        // CameraRigParams rig_params;
        // bool camera_rig = false;
        // Eigen::Matrix3d R0;
        // Eigen::Vector3d t0;

        // if (!camera_rig_params_file.empty()) {
        //     if (rig_params.LoadParams(camera_rig_params_file)) {
        //         camera_rig = true;
        //         R0 = rig_params.local_extrinsics[0].block<3, 3>(0, 0);
        //         t0 = rig_params.local_extrinsics[0].block<3, 1>(0, 3);

        //     } else {
        //         std::cout << "failed to read rig params" << std::endl;
        //         exit(-1);
        //     }
        // }

        std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
        for (const auto image_id : image_ids) {
            const Image &image = scene_graph_container->Image(image_id);
            const Camera& camera = scene_graph_container->Camera(image.CameraId());
            bool camera_rig = false;
            Eigen::Matrix3d R0;
            Eigen::Vector3d t0;
            if(camera.NumLocalCameras()>1){
                camera_rig = true;
                Eigen::Vector4d qvec;
                Eigen::Vector3d tvec;
                camera.GetLocalCameraExtrinsic(0,qvec,tvec);
                R0 = QuaternionToRotationMatrix(qvec);
                t0 = tvec;
            }

            std::string name = image.Name();
            if (keyframe_map.find(name) != keyframe_map.end()) {
                Keyframe keyframe = keyframe_map.at(name);

                Eigen::Matrix3d r = keyframe.rot;
                Eigen::Vector4d q = RotationMatrixToQuaternion(r);
                Eigen::Vector3d t = keyframe.pos;

                if (camera_rig) {
                    Eigen::Matrix3d R_rig = R0.transpose() * r;
                    Eigen::Vector3d t_rig = R0.transpose() * (t - t0);
                    q = RotationMatrixToQuaternion(R_rig);
                    t = t_rig;
                }
                prior_rotations.emplace(image_id, q);
                prior_translations.emplace(image_id, t);
            }
        }

        options->independent_mapper_options.prior_rotations = prior_rotations;
        options->independent_mapper_options.prior_translations = prior_translations;
        options->independent_mapper_options.have_prior_pose = true;
    }

    // use gps location prior to constrain the image
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    std::string gps_prior_file = param.GetArgument("gps_prior_file", "");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";
    if (use_gps_prior) {
        if (boost::filesystem::exists(gps_prior_file)) {
            auto image_ids = scene_graph_container->GetImageIds();
            std::vector<std::string> image_names;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();
                image_names.push_back(name);
            }

            std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> gps_locations;
            if (options->independent_mapper_options.optimization_use_horizontal_gps_only) {
                LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, true);
            } else {
                LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, false);
            }
            std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> image_locations;
            GPSLocationsToImages(gps_locations, image_names, image_locations);
            std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
            std::cout << image_locations.size() << " images have gps prior" << std::endl;
            
            bool set_gps_info = true;
            if(image_locations.size() == 0){
                std::cout<<" WARNING!!!!! There has no gps related images!"<<std::endl;
                std::cout<<" WARNING!!!!! Set use_gps_prior to 0!"<<std::endl;
                use_gps_prior = false;
                set_gps_info = false;
            }

            if(set_gps_info){
                std::vector<PlyPoint> gps_locations_ply;
                for (const auto image_id : image_ids) {
                    const Image &image = scene_graph_container->Image(image_id);
                    std::string name = image.Name();

                    if (image_locations.find(name) != image_locations.end()) {
                        prior_locations_gps.emplace(image_id, image_locations.at(name));

                        PlyPoint gps_location_ply;
                        gps_location_ply.r = 255;
                        gps_location_ply.g = 0;
                        gps_location_ply.b = 0;
                        gps_location_ply.x = image_locations.at(name).first[0];
                        gps_location_ply.y = image_locations.at(name).first[1];
                        gps_location_ply.z = image_locations.at(name).first[2];
                        gps_locations_ply.push_back(gps_location_ply);
                    }
                }
                options->independent_mapper_options.prior_locations_gps = prior_locations_gps;
                options->independent_mapper_options.original_gps_locations = gps_locations;
                sensemap::WriteBinaryPlyPoints(workspace_path + "/gps.ply", gps_locations_ply, false, true);
                options->independent_mapper_options.has_gps_prior = true;
            }
        }

    }

    // rgbd mode
    int num_local_cameras = reader_options.num_local_cameras;
    bool with_depth = options->independent_mapper_options.with_depth;

    MapperController *mapper =
        MapperController::Create(options, workspace_path, image_path, scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Incremental Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    bool use_apriltag = static_cast<bool>(param.GetArgument("detect_apriltag", 0));
    double apriltag_size = param.GetArgument("apriltag_size", 0.113f);
    bool color_harmonization = param.GetArgument("color_harmonization", 0);
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(i);
        CHECK(reconstruction->RegisterImageIds().size() > 0);
        const image_t first_image_id = reconstruction->RegisterImageIds()[0];
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image &image = reconstruction->Image(first_image_id);
        const Camera &camera = reconstruction->Camera(image.CameraId());

        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (color_harmonization) {
            reconstruction->ColorHarmonization(image_path);
        }

        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));

        if (use_gps_prior && !(hybrid_options.child_id > 0)) {
            // Eigen::Matrix3x4d matrix_to_align =
            reconstruction->AlignWithPriorLocations(options->independent_mapper_options.max_error_gps);
            if(false && camera.NumLocalCameras() == 1){
                Reconstruction rec = *reconstruction.get();
                rec.AddPriorToResult();
                // rec.NormalizeWoScale();
                
                std::string trans_rec_path = rec_path + "-gps";
                if (!boost::filesystem::exists(trans_rec_path)) {
                    boost::filesystem::create_directories(trans_rec_path);
                }
                rec.WriteBinary(trans_rec_path);
            }
        }

        if (options->independent_mapper_options.extract_colors) {
            reconstruction->ExtractColorsForAllImages(image_path);
        }
        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));
        
        if (camera.NumLocalCameras() > 1) {
            reconstruction->WriteReconstruction(rec_path, options->independent_mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);

            rig_reconstruction.WriteReconstruction(export_rec_path,
                                                   options->independent_mapper_options.write_binary_model);
        } else {
            reconstruction->WriteReconstruction(rec_path, options->independent_mapper_options.write_binary_model);
        }
    }
}


///////////////////// Main ///////////////////////
int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: hd-sfm-")+__VERSION__);
    
    Timer timer;
    timer.Start();

    int param_idx = 1;
#ifdef DO_ENCRYPT_CHECK
    CHECK(argc >= 5);
    int ret = do_check(5, (const char**)argv);
    std::cout << "Check Status: " << ret << std::endl;
    if (ret) return ret;
    param_idx = 5;
#endif
    configuration_file_path = std::string(argv[param_idx]);
    std::cout << "configuration_file_path: " << configuration_file_path << std::endl;

    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    bool debug_info = static_cast<bool>(param.GetArgument("debug_info", 0));
    int register_sequential = static_cast<int>(param.GetArgument("register_sequential", 0));
    int num_cameras = static_cast<int>(param.GetArgument("num_cameras", -1));
    std::string cameras_param_file = param.GetArgument("camera_param_file", "");
    std::cout << "register_sequential, num_cameras, cameras_param_file: " << register_sequential << ", " << num_cameras << ", " << cameras_param_file << std::endl;

    if (num_cameras <= 1 || cameras_param_file.empty()){
        bool new_feature_extraction = FeatureExtraction(*feature_data_container.get(), param);

        typedef FeatureMatchingOptions::RetrieveType RetrieveType;
        RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
        if (retrieve_type == RetrieveType::VLAD) {
            GlobalFeatureExtraction(*feature_data_container.get(), param, new_feature_extraction);
        }
        FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);
        
        feature_data_container.reset();

        std::string mapper_method = param.GetArgument("mapper_method", "incremental");

        // if (mapper_method.compare("incremental") == 0) {
        //     IncrementalSFM(scene_graph_container, reconstruction_manager, param);
        // } else if (mapper_method.compare("cluster") == 0) {
        //     ClusterIncrementalMapperOptions(scene_graph_container, reconstruction_manager, param);
        // }
    } else if (!cameras_param_file.empty()) {
        struct HybridOptions hybrid_options = HybridOptions(debug_info);
        bool new_feature_extraction = true;
        if (register_sequential == 0){
            for (int idx = 0; idx < num_cameras; idx++){
                hybrid_options.child_id = idx;
                if (idx == num_cameras-1){
                    hybrid_options.save_flag = true;
                }
                new_feature_extraction = new_feature_extraction && 
                    FeatureExtraction(*feature_data_container.get(), param, hybrid_options);
            }

            typedef FeatureMatchingOptions::RetrieveType RetrieveType;
            RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
            if (retrieve_type == RetrieveType::VLAD) {
                GlobalFeatureExtraction(*feature_data_container.get(), param, new_feature_extraction);
            }

            FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);
            IncrementalSFM(scene_graph_container, reconstruction_manager, param);
        } else if (register_sequential == 2) {
            hybrid_options.child_id = 0;
            if (!hybrid_options.debug_info){
                hybrid_options.save_flag = false;
                hybrid_options.read_flag = false;
            }
            bool new_feature_extraction = FeatureExtraction(*feature_data_container.get(), param, hybrid_options);
            
            typedef FeatureMatchingOptions::RetrieveType RetrieveType;
            RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
            if (retrieve_type == RetrieveType::VLAD) {
                GlobalFeatureExtraction(*feature_data_container.get(), param, new_feature_extraction, hybrid_options);
            }

            FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param, hybrid_options);            

            for (int idx = 1; idx < num_cameras; idx++){
                hybrid_options.child_id = idx;
                hybrid_options.update_flag = true;
                
                FeatureExtraction(*feature_data_container.get(), param, hybrid_options);

                typedef FeatureMatchingOptions::RetrieveType RetrieveType;
                RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
                if (retrieve_type == RetrieveType::VLAD) {
                    // vlad is not good for different cameras
                    GlobalFeatureExtraction(*feature_data_container.get(), param, new_feature_extraction, hybrid_options);
                }

                FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), 
                                param, hybrid_options);

                if (idx == num_cameras-1){
                    hybrid_options.child_id = -1;
                    hybrid_options.save_flag = true;
                }
            }

            IncrementalSFM(scene_graph_container, reconstruction_manager, param, hybrid_options);
        } else {
            for (int idx = 0; idx < num_cameras; idx++){
                hybrid_options.child_id = idx;
                if (idx > 0){
                    hybrid_options.update_flag = true;
                }
                if (idx == num_cameras - 1){
                    hybrid_options.save_flag = true;
                }
                bool new_feature_extraction = FeatureExtraction(*feature_data_container.get(), param, hybrid_options);

                typedef FeatureMatchingOptions::RetrieveType RetrieveType;
                RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
                if (retrieve_type == RetrieveType::VLAD) {
                    // vlad is not good for different cameras
                    GlobalFeatureExtraction(*feature_data_container.get(), param, new_feature_extraction, hybrid_options);
                }

                FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), 
                                param, hybrid_options);

                // if (idx == num_cameras-1){
                //     hybrid_options.child_id = -1;
                // }
                IncrementalSFM(scene_graph_container, reconstruction_manager, param, hybrid_options);
            }
        }
    }

    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;

    PrintReconSummary(workspace_path + "/statistic.txt", scene_graph_container->NumImages(), reconstruction_manager);

    fclose(fs);

    return 0;
}