// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <stdlib.h>
#include <iostream>

#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <cmath>
#include <vector>
#include <map>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <time.h>
#include <float.h>
#include <boost/filesystem/path.hpp>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "../Configurator.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/similarity_transform.h"
#include "cluster/fast_community.h"
#include "container/feature_data_container.h"
#include "estimators/scale_selection.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "optim/ransac/loransac.h"
#include "util/misc.h"

using namespace sensemap;


void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    v.clear();
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) v.push_back(s.substr(pos1));
}

bool IsBluetoothFile(std::string filePath) {
    std::ifstream infile;
    infile.open(filePath, std::ios::in);
    if (!infile.is_open()) {
        std::cout << "Cant to Open file: " << filePath << std::endl;
        return false;
    }
    std::vector<std::string> items;
    SplitString(filePath, items, ".");
    if (items.back() != "txt" && items.back() != "csv") {
        std::cout << filePath << "is not .txt or csv file" << std::endl;
        return false;
    }
    int count = 0;
    std::string line;
    while (getline(infile, line)) {
        if (line.size() == 0) continue;
        SplitString(line, items, ",");
        if (items.size() < 9) {
            std::cout << filePath << " is not Bluetooth file" << std::endl;
            return false;
        }
        if (std::stoi(items[8]) > -200 && std::stoi(items[8]) < 200) count++;
        if (count > 5) return true;
    }
    return false;
}

std::vector<std::string> getFilesList(std::string dirpath, std::vector<std::string>& gridfile_names) {
    DIR* dir = opendir(dirpath.c_str());
    if (dir == NULL) {
        std::cout << "opendir error! Path: " << dirpath << std::endl;
    }
    std::vector<std::string> allPath;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {  // It's dir
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
            std::string dirNew = dirpath + "/" + entry->d_name;
            std::vector<std::string> tempPath = getFilesList(dirNew, gridfile_names);
            allPath.insert(allPath.end(), tempPath.begin(), tempPath.end());
            gridfile_names.push_back(entry->d_name);
        } else {
            std::string name = entry->d_name;
            std::string imgdir = dirpath + "/" + name;
            // sprintf("%s",imgdir.c_str());
            allPath.push_back(imgdir);
        }
    }
    closedir(dir);
    // system("pause");
    return allPath;
}

class BeaconSignaldata {
public:
    // Beacondata from collect line
    double unix_timestamp_ = 0;
    // Identity_str: Identity of Signal : Name or UUID_Major_Minor
    std::string Name_, UUID_, Major_, Minor_, Identity_str_, FactoryID_, MAC_;
    int RSSI_, num_items_;

    // stable data of Beacon
    // Sorted BeaconID in RSSI vec
    size_t ID_vec_;
    // Beacon's dominate AreaStr
    std::string Area_domin_;
    // RSSI data
    int rssi_mean, rssi_max, times_recv;
    // Site String
    std::string Site_Code_domin_;

    // init for 9 items
    BeaconSignaldata(double unix_timestamp, std::string Name, std::string UUID, std::string Major, std::string Minor,
                     std::string FactoryID, std::string MAC, int RSSI)
        : unix_timestamp_(unix_timestamp),
          Name_(Name),
          UUID_(UUID),
          Major_(Major),
          Minor_(Minor),
          Identity_str_(UUID + "_" + Major + "_" + Minor),
          FactoryID_(FactoryID),
          MAC_(MAC),
          RSSI_(RSSI),
          num_items_(9) {}

    // init for 5 items
    BeaconSignaldata(double unix_timestamp, std::string Name, int RSSI)
        : unix_timestamp_(unix_timestamp), Name_(Name), RSSI_(RSSI), num_items_(5) {}

    BeaconSignaldata(){};

    inline bool HaveName() const;

    inline bool HaveArea() const;

    ~BeaconSignaldata() {}

private:
};
bool BeaconSignaldata::HaveName() const {
    if (Name_ == "null" || Name_.empty())
        return false;
    else
        return true;
}
bool BeaconSignaldata::HaveArea() const {
    if (Area_domin_ == "null" || Area_domin_.empty())
        return false;
    else
        return true;
}

bool GetdataFromLinedata(const std::string& linestr, BeaconSignaldata& beacondata) {
    std::vector<std::string> One_Signal_line;
    SplitString(linestr, One_Signal_line, ",");
    // rule linedata
    // for 5 items
    if (One_Signal_line.size() == 5) {
        beacondata.Name_ = One_Signal_line[2];
        beacondata.RSSI_ = stoi(One_Signal_line[4]);
        // for IOS data
        if (beacondata.RSSI_ == 0) beacondata.RSSI_ = -98;

        beacondata.unix_timestamp_ = atof(One_Signal_line[1].c_str());
        // beacondata.Identity_str_ = beacondata.Name_;
        beacondata.num_items_ = 5;
        return true;
        // for 9 items
    } else if (One_Signal_line.size() == 9) {
        beacondata.Name_ = One_Signal_line[2];
        beacondata.MAC_ = One_Signal_line[7];
        beacondata.UUID_ = One_Signal_line[3];
        beacondata.Major_ = One_Signal_line[4];
        beacondata.Minor_ = One_Signal_line[5];
        ////Identity_str
        // beacondata.Identity_str_ = beacondata.Name_;
        // beacondata.Identity_str = One_Signal_line[3] + "_" + One_Signal_line[4] + "_" + One_Signal_line[6] ;
        // FactoryID = first number?
        beacondata.FactoryID_ = One_Signal_line[6];
        beacondata.unix_timestamp_ = atof(One_Signal_line[1].c_str());
        beacondata.RSSI_ = stoi(One_Signal_line[8]);
        if (beacondata.RSSI_ == 0) beacondata.RSSI_ = -100;
        beacondata.num_items_ = 9;
        return true;
    } else if (One_Signal_line.size() == 10) {
        beacondata.Name_ = One_Signal_line[2];
        beacondata.MAC_ = One_Signal_line[7];
        beacondata.UUID_ = One_Signal_line[3];
        beacondata.Major_ = One_Signal_line[4];
        beacondata.Minor_ = One_Signal_line[5];
        ////Identity_str
        // beacondata.Identity_str_ = beacondata.Name_;
        // beacondata.Identity_str = One_Signal_line[3] + "_" + One_Signal_line[4] + "_" + One_Signal_line[6] ;
        // FactoryID = first number?
        beacondata.FactoryID_ = One_Signal_line[6];
        beacondata.unix_timestamp_ = atof(One_Signal_line[9].c_str());
        beacondata.RSSI_ = stoi(One_Signal_line[8]);
        if (beacondata.RSSI_ == 0) beacondata.RSSI_ = -100;
        beacondata.num_items_ = 9;
        return true;
    }
    return false;
}

int BeaconDistance(const std::vector<std::string>& bluetooth_files, const double& timestamp1, const double& timestamp2,
                std::map<double, std::pair<std::string, int>>& time_signal,
                   double& distance) {
    double timestamp1ms, timestamp2ms;
    if (timestamp1 < 2000000000 && timestamp1 > 1000000000)
        timestamp1ms = timestamp1 * 1000;
    else if (timestamp1 < 2000000000000 && timestamp1 > 1000000000000)
        timestamp1ms = timestamp1;
    else {
        std::cout << "Wrong timestamp input : timestamp1: " << timestamp1 << std::endl;
        return -1;
    }
    if (timestamp2 < 2000000000 && timestamp2 > 1000000000)
        timestamp2ms = timestamp2 * 1000;
    else if (timestamp2 < 2000000000000 && timestamp2 > 1000000000000)
        timestamp2ms = timestamp2;
    else {
        std::cout << "Wrong timestamp input : timestamp2: " << timestamp2 << std::endl;
        return -1;
    }


    //  //compute vip rssi vec
    //  double neardis1 = DBL_MAX, neardis2 = DBL_MAX;
    //  double nearest_timestamp1 = -1,nearest_timestamp2 = -1;
    //
    //  for(auto& time_rssi : time_signal) {
    //
    //    double dis1 = fabs ( timestamp1ms - time_rssi.first );
    //    double dis2 = fabs ( timestamp2ms - time_rssi.first );
    //
    //    if (dis1 < neardis1) {
    //      neardis1 = dis1;
    //      nearest_timestamp1 = time_rssi.first;
    //    }
    //
    //    if (dis2 < neardis2) {
    //      neardis2 = dis2;
    //      nearest_timestamp2 = time_rssi.first;
    //    }
    //  }
    //
    //  /// No signal catched near 10s
    //  if (neardis1 > 10 * 1000  ||  neardis2 > 10 * 1000) {
    //    std::cout << "Error: No Beacon data in input timestamp: timestamp1ms "<<timestamp1ms<< " nearest_dis: "<<
    //    neardis1  << " "
    //              <<  "timestamp2ms "<<timestamp2ms<< " nearest_dis: "<< neardis2 << std::endl;
    //    return -1;
    //  }

    ////gather timestamp signal

    std::map<std::string, std::vector<int>> rssimap1, rssimap2;

    for (double timestamp = timestamp1ms - 3000; timestamp < timestamp1ms + 3000; timestamp++) {
        if (time_signal.count(timestamp)) {
            std::string unic_majorminor = time_signal[timestamp].first;
            rssimap1[unic_majorminor].push_back(time_signal[timestamp].second);
        }
    }
    for (double timestamp = timestamp2ms - 3000; timestamp < timestamp2ms + 3000; timestamp++) {
        if (time_signal.count(timestamp)) {
            std::string unic_majorminor = time_signal[timestamp].first;
            rssimap2[unic_majorminor].push_back(time_signal[timestamp].second);
        }
    }

    ////compare signal map
    int good_signal_num1 = 0, good_signal_num2 = 0;
    int signal_num1 = 0, signal_num2 = 0;
    int max_rssi1 = -100, max_rssi2 = -100;

    for (auto& beacon_data : rssimap1) {
        if (beacon_data.second.size() == 0) continue;

        int rssi_mean = (int)(std::accumulate(std::begin(beacon_data.second), std::end(beacon_data.second), 0.0) /
                              beacon_data.second.size());
        beacon_data.second.push_back(rssi_mean);

        if (rssi_mean < -85) {
            good_signal_num1++;
            signal_num1++;
        } else if (rssi_mean < -93)
            signal_num1++;
        if (rssi_mean > max_rssi1) max_rssi1 = rssi_mean;
    }

    for (auto& beacon_data : rssimap2) {
        if (beacon_data.second.size() == 0) continue;

        int rssi_mean = (int)(std::accumulate(std::begin(beacon_data.second), std::end(beacon_data.second), 0.0) /
                              beacon_data.second.size());
        beacon_data.second.push_back(rssi_mean);

        if (rssi_mean < -85) {
            good_signal_num2++;
            signal_num2++;
        } else if (rssi_mean < -93)
            signal_num2++;
        if (rssi_mean > max_rssi2) max_rssi2 = rssi_mean;
    }

    // for (auto& beacon_data : rssimap1) {
    //     std::cout << "rssimap1 : " << beacon_data.first << " " << beacon_data.second.back() << std::endl;
    // }
    // for (auto& beacon_data : rssimap2) {
    //     std::cout << "rssimap2 : " << beacon_data.first << " " << beacon_data.second.back() << std::endl;
    // }

    // if (signal_num1 < 3 && signal_num2 < 3 && max_rssi1 < -90 && max_rssi2 < -90) {
    //     // std::cout << "Error: both signal is too weak to compare" << std::endl;
    //     return -1;
    // }

    if (signal_num1 == 0 || signal_num2 == 0) {
        return -1;
    }

    // std::map<std::string,std::vector<int>> rssimap_stronger,rssimap_other;
    std::vector<double> Eucldis_vec;
    double rssi_distance = -100;

    // std::cout << "good_signal_num1 " << good_signal_num1 << " good_signal_num2: " << good_signal_num2 << std::endl;

    if (good_signal_num1 > good_signal_num2) {
        Eucldis_vec.clear();
        for (const auto& beacon_data : rssimap1) {
            if (beacon_data.second.back() < -95) continue;

            if (rssimap2.count(beacon_data.first)) {
                double Eucldis = (beacon_data.second.back() - rssimap2[beacon_data.first].back()) *
                                 (beacon_data.second.back() - rssimap2[beacon_data.first].back());
                // std::cout << " dis of Beacon " << beacon_data.first << " " << sqrt(Eucldis) << std::endl;
                Eucldis_vec.push_back(Eucldis);
            } else {
                double Eucldis = (-95 - beacon_data.second.back()) * (-95 - beacon_data.second.back());
                Eucldis_vec.push_back(Eucldis);
                // std::cout << " dis of Beacon " << beacon_data.first << " " << Eucldis << std::endl;
            }
        }
        rssi_distance =
            sqrt((std::accumulate(std::begin(Eucldis_vec), std::end(Eucldis_vec), 0.0)) / Eucldis_vec.size());

    } else {
        Eucldis_vec.clear();
        for (const auto& beacon_data : rssimap2) {
            if (beacon_data.second.back() < -95) continue;
            
            if (rssimap1.count(beacon_data.first)) {
                double Eucldis = (beacon_data.second.back() - rssimap1[beacon_data.first].back()) *
                                 (beacon_data.second.back() - rssimap1[beacon_data.first].back());
                // std::cout << " dis of Beacon " << beacon_data.first << " " << sqrt(Eucldis) << std::endl;
                Eucldis_vec.push_back(Eucldis);
            } else {
                double Eucldis = (-95 - beacon_data.second.back()) * (-95 - beacon_data.second.back());
                Eucldis_vec.push_back(Eucldis);
                // std::cout << " dis of Beacon " << beacon_data.first << " " << Eucldis << std::endl;
            }
        }
        rssi_distance =
            sqrt((std::accumulate(std::begin(Eucldis_vec), std::end(Eucldis_vec), 0.0)) / Eucldis_vec.size());
    }

    distance = rssi_distance;
    return 0;
}


std::vector<std::string> GetBlueToothFileList(const std::string& path) {
    std::string bluetooth_files_path = path;
    std::vector<std::string> bluetooth_files;
    std::vector<std::string> items;
    SplitString(bluetooth_files_path, items, ".");
    if (items.back() == "txt" || items.back() == "csv") {
        if (IsBluetoothFile(bluetooth_files_path))
            bluetooth_files.push_back(bluetooth_files_path);
        else {
            std::cout << "Error: its not Bluetooth txt ,Check its path :" << bluetooth_files_path << std::endl;
            return bluetooth_files;
        }
    } else {
        std::string endchar = bluetooth_files_path.substr(bluetooth_files_path.length() - 1);
        if (endchar == "/") {
            bluetooth_files_path.pop_back();
        }

        std::vector<std::string> files_name;
        std::vector<std::string> files = getFilesList(bluetooth_files_path, files_name);
        for (const auto& file : files) {
            SplitString(file, items, ".");
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

    return bluetooth_files;
}

double TimeStampFromName(const std::string& image_name) {
    std::vector<std::string> items;
    SplitString(image_name, items, "/");
    std::string name = items.back();
    SplitString(name, items, ".");
    // std::cout << "items[0] = " << items[0] << std::endl;
    return std::stod(items[0]);
}

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

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

int main(int argc, char *argv[]) {
    std::string workspace_path = std::string(argv[1]);
    std::string bluetooth_folder_path = std::string(argv[2]);

    bool camera_rig = true;
    // Load feature and scene_graph
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    // Read feature file
    PrintHeading1("Loading feature data ");
    Timer timer;

    timer.Start();
    if (dirExists(workspace_path + "/features.bin")) {
        feature_data_container->ReadImagesBinaryDataWithoutDescriptor(workspace_path + "/features.bin");
        if (dirExists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container->ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        feature_data_container->ReadSubPanoramaBinaryData(workspace_path + "/sub_panorama.bin");
    } else {
        std::cerr << "ERROR: Current workspace do not contain features.bin " << workspace_path << std::endl;
        return 0;
    }
    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    // Read scene graph .txt or .bin
    PrintHeading1("Loading scene graph matching data");
    timer.Start();
    if (dirExists(workspace_path + "/scene_graph.bin")) {
        scene_graph_container->ReadSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
    } else if (dirExists(workspace_path + "/scene_graph.txt")) {
        scene_graph_container->ReadSceneGraphData(workspace_path + "/scene_graph.txt");
    } else {
        std::cerr << "ERROR: Current workspace do not contain scene_graph.bin or scene_graph.txt " << workspace_path
                  << std::endl;
        return 0;
    }
    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    auto correspondence_graph = scene_graph_container->CorrespondenceGraph();

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph_container->Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph_container->Cameras();

    // FeatureDataContainer data_container;
    std::vector<image_t> image_ids = feature_data_container->GetImageIds();

    std::cout << "image_ids.size() = " << image_ids.size() << std::endl;

    for (const auto image_id : image_ids) {
        const Image &image = feature_data_container->GetImage(image_id);
        if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        images[image_id] = image;
        const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
        // const std::vector<Eigen::Vector2d> points = FeatureKeypointsToPointsVector(keypoints);
        // images[image_id].SetPoints2D(points);
        images[image_id].SetPoints2D(keypoints);

        // std::cout << image.CameraId() << std::endl;
        const Camera &camera = feature_data_container->GetCamera(image.CameraId());
        // std::cout << image.CameraId() << std::endl;
        if (!scene_graph_container->ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
        }

        const PanoramaIndexs &panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        images[image_id].SetLocalImageIndices(local_image_indices);

        if (correspondence_graph->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
        } else {
            // std::cout << "Do not contain this image" << std::endl;
        }
    }
    std::cout << "Load correspondece graph finished" << std::endl;
    correspondence_graph->Finalize();


    // Read Bluetooth File
    auto bluetooth_file_paths = GetBlueToothFileList(bluetooth_folder_path);

    /* Read the Bluetooth file */
    std::map<double, std::pair<std::string, int>> time_signal;

    for (const auto& file_path : bluetooth_file_paths) {
        if (!IsBluetoothFile(file_path)) {
            std::cout << "Warning: its not bluetooth file " << file_path << std::endl;
            continue;
        }

        std::ifstream testdatafile(file_path);
        std::string linedata;
        std::vector<std::string> one_signal_line;
        while (getline(testdatafile, linedata)) {
            one_signal_line.clear();
            SplitString(linedata, one_signal_line, ",");
            //      BeaconSignaldata beacondata;
            //      GetdataFromLinedata(linedata ,beacondata);
            // std::cout << "one_signal_line.size() = " <<one_signal_line.size() <<std::endl;
            // std::cout << "one_signal_line[2] = " <<one_signal_line[1] <<std::endl;



            double timestamp = std::stod(TimeStampFromStr(one_signal_line[0]));

            // double timestamp = one_signal_line.size() == 9 ? std::stod(one_signal_line[1]) : std::stod(one_signal_line[9]);
            std::string major_minor = one_signal_line[4] + "-" + one_signal_line[5];
            int rssi = std::stoi(one_signal_line[8]);
            if (rssi == 0) rssi = -97;
            time_signal[timestamp] = std::make_pair(major_minor, rssi);
        }
    }

    // Get all the correspondence pair
    const auto& image_pairs = correspondence_graph->ImagePairs();
    int delete_pair_num = 0;
    int counter = 0;
    std::unordered_map<image_pair_t, double> bluetooth_pair_info;
    for (auto image_pair : image_pairs) {
        image_t image_1_id, image_2_id;
        utility::PairIdToImagePair(image_pair.first, &image_1_id, &image_2_id);

        // std::cout << "Pair id = " << image_pair.first << std::endl;
        // std::cout << "Image 1 id = " << image_1_id << std::endl;
        // std::cout << "Image 2 id = " << image_2_id << std::endl;


        if (!feature_data_container->ExistImage(image_1_id)) {
            std::cout << "Image Not Exist ???? 1 " << image_1_id << std::endl;
            continue;
        }

        if (!feature_data_container->ExistImage(image_2_id)) {
            std::cout << "Image Not Exist ???? 2 " << image_2_id <<  std::endl;
            continue;
        }

        const Image &image_1 = feature_data_container->GetImage(image_1_id);
        const Image &image_2 = feature_data_container->GetImage(image_2_id);

        std::string image_1_name = image_1.Name();
        std::string image_2_name = image_2.Name();

        // std::cout << "Image 1 name = " << image_1_name << std::endl;
        // std::cout << "Image 2 name = " << image_2_name << std::endl;
        
        // Get the time stamp 
        double image_1_timestamp = TimeStampFromName(image_1_name);
        double image_2_timestamp = TimeStampFromName(image_2_name);

        double dis = -100;
       
        BeaconDistance(bluetooth_file_paths, image_1_timestamp, image_2_timestamp, time_signal, dis);

        if (dis > 5) {
            // std::cout << "Distance Larger than 5" << std::endl;
            correspondence_graph->DeleteCorrespondences(image_1_id, image_2_id);
            delete_pair_num++;
        }
        
        // bluetooth_pair_info[image_pair.first] = dis;

        counter ++;
        std::cout <<" [ " << counter << " / " << image_pairs.size() << " ]" << std::endl;
    }
    // correspondence_graph->SetBluetoothPairsInfo(bluetooth_pair_info);

    correspondence_graph->Finalize();
    std::cout << "delete_pair_num = " << delete_pair_num << std::endl;
    correspondence_graph->WriteCorrespondenceBinaryData(workspace_path+"/update_scene_graph.bin");
    // correspondence_graph->WriteBlueToothPairsInfoBinaryData(workspace_path+"/bluetooth_info.bin");
}
