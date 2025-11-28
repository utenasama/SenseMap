// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#include "util/tag_scale_recover.h"
#include "base/projection.h"
#include "estimators/scale_selection.h"
#include "optim/ransac/loransac.h"
#include "controllers/incremental_mapper_controller.h"

namespace sensemap {

bool TagVideoNameFromString(const std::string &image_name, std::string &video_name) {
    std::vector<std::string> split_elems = StringSplit(image_name, "/");

    // INSV
    // Pro2
    if (split_elems[0].find("cam") != std::string::npos && split_elems[1].find("VID") != std::string::npos) {
        video_name = split_elems[1];
        return true;
    }

    // MP4
    if (split_elems[0].find("VID") != std::string::npos) {
        video_name = split_elems[0];
        return true;
    }

    // Other
    video_name = "default";
    return false;
}

std::string TimeStampFromVideoName(const std::string &video_name) {
    int year, month, day, hour, minute, second;
    std::vector<std::string> split_elems = StringSplit(video_name, "_");
    // MP4  -- VID_20210828_143826_00_001
    // Insv -- VID_20210809_121305077
    // Pro2 -- VID_20210322_002941
    // std::cout << "video_name = " << video_name << std::endl;
    if (video_name.length() == 22 || video_name.length() == 19 || video_name.length() == 26) {
        // std::cout << "split_elems[1].substr(0, 4) = " << split_elems[1].substr(0, 4) << std::endl;
        year = std::stoi(split_elems[1].substr(0, 4)) - 1900;
        // std::cout << "split_elems[1].substr(4, 2) = " << split_elems[1].substr(4, 2) << std::endl;
        month = std::stoi(split_elems[1].substr(4, 2)) - 1;
        // std::cout << "split_elems[1].substr(6, 2) = " << split_elems[1].substr(6, 2) << std::endl;
        day = std::stoi(split_elems[1].substr(6, 2));
        // std::cout << "split_elems[2].substr(0, 2) = " << split_elems[2].substr(0, 2) << std::endl;
        hour = std::stoi(split_elems[2].substr(0, 2));
        // std::cout << "split_elems[2].substr(2, 2) = " << split_elems[2].substr(2, 2) << std::endl;
        minute = std::stoi(split_elems[2].substr(2, 2));
        // std::cout << "split_elems[2].substr(4, 2) = " << split_elems[2].substr(4, 2) << std::endl;
        second = std::stoi(split_elems[2].substr(4, 2));

        struct tm timeinfo;
        timeinfo.tm_year = year;
        timeinfo.tm_mon = month;
        timeinfo.tm_mday = day;
        timeinfo.tm_hour = hour;
        timeinfo.tm_min = minute;
        timeinfo.tm_sec = second;
        timeinfo.tm_isdst = 0;
        time_t t = mktime(&timeinfo);
        return std::to_string(t) + "000";
    }

    return "1609430400000";  //  20210101_000000
}

std::vector<double> ransac_sacle_estimate(std::vector<double> edges, double tag_size, double min_error) {
    int best_index = -1;
    int best_index_inlier_size = 0;
    std::vector<double> best_inlier_edges;
    double best_inlier_min_error;

    double sum = std::accumulate(edges.begin(), edges.end(), 0.0);
    double sum_number = edges.size();
    double init_scale = tag_size / (sum / sum_number);

    for (int i = 0; i < edges.size(); i++) {
        double current_cadidate = edges[i];
        std::vector<double> current_inlier_edges;
        std::vector<double> current_inlier_error;
        current_inlier_edges.emplace_back(edges[i]);
        for (int j = 0; j < edges.size(); j++) {
            if (i == j) {
                continue;
            }

            double current_error = init_scale * std::abs(edges[i] - edges[j]);
            if (current_error < min_error) {
                current_inlier_error.emplace_back(current_error);
                current_inlier_edges.emplace_back(edges[j]);
            }
        }

        if (current_inlier_edges.size() > best_index_inlier_size) {
            best_index_inlier_size = current_inlier_edges.size();
            best_inlier_edges = current_inlier_edges;
            best_inlier_min_error = std::accumulate(current_inlier_error.begin(), current_inlier_error.end(), 0.0) /
                                    current_inlier_error.size();
        }
    }

    std::cout << "Best Inlier size = " << best_inlier_edges.size() << std::endl;
    std::cout << "Best Inlier Min Error = " << best_inlier_min_error << std::endl;

    return best_inlier_edges;
}

std::vector<Eigen::Vector3d> ransac_pose_select(std::vector<Eigen::Vector3d> poses, double min_error) {
    int best_index = -1;
    int best_index_inlier_size = 0;
    std::vector<Eigen::Vector3d> best_inlier_poses;
    double best_inlier_min_error;

    for (int i = 0; i < poses.size(); i++) {
        Eigen::Vector3d current_cadidate = poses[i];
        std::vector<Eigen::Vector3d> current_inlier_poses;
        std::vector<double> current_inlier_error;
        current_inlier_poses.emplace_back(poses[i]);
        for (int j = 0; j < poses.size(); j++) {
            if (i == j) {
                continue;
            }

            double current_error = (poses[i] - poses[j]).norm();
            if (current_error < min_error) {
                current_inlier_error.emplace_back(current_error);
                current_inlier_poses.emplace_back(poses[j]);
            }
        }

        if (current_inlier_poses.size() > best_index_inlier_size) {
            best_index_inlier_size = current_inlier_poses.size();
            best_inlier_poses = current_inlier_poses;
            best_inlier_min_error = std::accumulate(current_inlier_error.begin(), current_inlier_error.end(), 0.0) /
                                    current_inlier_error.size();
        }
    }

    std::cout << "Best Inlier size = " << best_inlier_poses.size() << std::endl;
    std::cout << "Best Inlier Min Error = " << best_inlier_min_error << std::endl;

    return best_inlier_poses;
}

double scale_estimate(const std::map<std::pair<std::string, std::string>, double> &tag_measurements,
                      const std::map<std::string, Eigen::Vector3d> &marker_poses) {
    std::vector<double> scale_candidates;
    scale_candidates.clear();
    for (const auto tag_measurement : tag_measurements) {
        if (marker_poses.count(tag_measurement.first.first) && marker_poses.count(tag_measurement.first.second)) {
            double cur_scale =
                tag_measurement.second /
                (marker_poses.at(tag_measurement.first.first) - marker_poses.at(tag_measurement.first.second)).norm();
            std::cout << "current scale candidate = " << cur_scale << std::endl;
            scale_candidates.emplace_back(cur_scale);
        }
    }

    RANSACOptions ransac_options;
    ransac_options.max_error = 0.1;
    ransac_options.min_inlier_ratio = 0.2;
    LORANSAC<ScaleSelectionEstimator, ScaleSelectionEstimator> ransac(ransac_options);

    const auto report = ransac.Estimate(scale_candidates, scale_candidates);

    if (!report.success) {
        std::cout << "Estimate fail ..." << std::endl;
        return 1.0;
    }

    std::cout << "Inlier number = " << report.support.num_inliers << std::endl;

    std::cout << report.model << std::endl;
    return report.model;
}

double ComputeReconstructionScale(
    double tag_size,
    std::unordered_map<std::string, std::vector<image_t>> &apriltag_id_map,
    std::shared_ptr<FeatureDataContainer> feature_data_container,
    std::unordered_map<mappoint_t, std::unordered_map<std::string, int>> &mappoint_apriltag_map,
    std::unordered_map<std::string, std::vector<mappoint_t>> &apriltag_mappoint_map,
    std::shared_ptr<Reconstruction> reconstruction,
    std::vector<std::pair<std::string, Eigen::Vector3d>> &alignment_points,
    std::vector<Eigen::Vector3d> &alignment_tag_points) {
    // Calculate scale for each AprilTag
    std::vector<double> scales;
    alignment_points.clear();
    alignment_tag_points.clear();
    double e_sum = 0;
    double e_sq_sum = 0;
    int success_number = 0;
    int fail_number = 0;

    std::vector<double> marker_edges;
    for (const auto &apriltag_id_pair : apriltag_id_map) {
        std::string apriltag_id = apriltag_id_pair.first;
        const std::vector<image_t> &track_image_ids = apriltag_id_pair.second;
        std::vector<mappoint_t> apriltag_mappoints = apriltag_mappoint_map[apriltag_id];
        int min_track_length = 1000;
        bool all_mappoint_observation_correct = true;
        for (int i = 0; i < apriltag_mappoints.size(); i++) {
            mappoint_t mappoint_id = apriltag_mappoints[i];
            // Check mappoint exist or not
            if (!reconstruction->ExistsMapPoint(mappoint_id)) {
                // std::cout << " AprilTag Mappoint not exist ..." << std::endl;
                all_mappoint_observation_correct = false;
                continue;
            }

            // Check track length
            // if (sqrt(reconstruction->MapPoint(mappoint_id).Track().Length()) < min_track_length) {
            //     min_track_length = sqrt(reconstruction->MapPoint(mappoint_id).Track().Length());
            // }

            std::unordered_set<image_t> track_set;
            for (const auto &track : reconstruction->MapPoint(mappoint_id).Track().Elements()) {
                track_set.insert(track.image_id);
            }

            if (min_track_length > track_set.size()) {
                min_track_length = track_set.size();
            }
        }

        if (!all_mappoint_observation_correct) {
            std::cout << "Current Apriltag not reconstruct correctly" << std::endl;
            fail_number++;
            continue;
        }

        std::cout << "Min track length = " << min_track_length << std::endl;
        if (min_track_length < 3) {
            std::cout << "Min track length less than 3..." << std::endl;
            fail_number++;
            continue;
        }
        // TODO: Evaluate value after the reconstruction sim3 applied
        // std::cout << apriltag_id << std::endl;
        // std::cout << " " << reconstruction->MapPoint(apriltag_mappoints[0]).XYZ() << std::endl;
        alignment_points.emplace_back(apriltag_id, reconstruction->MapPoint(apriltag_mappoints[0]).XYZ());

        // Calculate the four edge of AprilTag
        auto mappoint_1 = reconstruction->MapPoint(apriltag_mappoints[1]).XYZ();
        auto mappoint_2 = reconstruction->MapPoint(apriltag_mappoints[2]).XYZ();
        auto mappoint_3 = reconstruction->MapPoint(apriltag_mappoints[3]).XYZ();
        auto mappoint_4 = reconstruction->MapPoint(apriltag_mappoints[4]).XYZ();

        alignment_tag_points.emplace_back(mappoint_4);
        alignment_tag_points.emplace_back(mappoint_1);
        alignment_tag_points.emplace_back(mappoint_2);
        alignment_tag_points.emplace_back(mappoint_3);

        double e1 = (mappoint_1 - mappoint_2).norm();
        double e2 = (mappoint_2 - mappoint_3).norm();
        double e3 = (mappoint_3 - mappoint_4).norm();
        double e4 = (mappoint_4 - mappoint_1).norm();

        marker_edges.emplace_back(e1);
        marker_edges.emplace_back(e2);
        marker_edges.emplace_back(e3);
        marker_edges.emplace_back(e4);
        double mean = (e1 + e2 + e3 + e4) / 4;
        // double stdev = sqrt()

        std::cout << "Id = " << apriltag_id << std::endl;
        std::cout << "e1 = " << e1 << " , e2 = " << e2 << " , e3 = " << e3 << " , e4 = " << e4 << std::endl;
        // double cur_scale = Real_AprilTag_Edge / ((e1 + e2 + e3 + e4) / 4);
        // std::cout << "cur_scale = " << cur_scale << std::endl;
        // e_sum = e_sum + e1 + e2 + e3 + e4;
        // e_sq_sum = e_sq_sum + e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4;

        // scales.emplace_back(cur_scale);
        success_number++;
    }

    std::cout << "Success tag number = " << success_number << std::endl;
    std::cout << "Fail tag number = " << fail_number << std::endl;

    if (success_number == 0) {
        return 0;
    }

    // Calculate Inlier Edge using Ransac
    std::cout << "Original Number = " << marker_edges.size() << std::endl;
    marker_edges = ransac_sacle_estimate(marker_edges, tag_size, 0.1);
    std::cout << "Inlier Number = " << marker_edges.size() << std::endl;

    e_sum = 0;
    e_sq_sum = 0;
    for (double edge : marker_edges) {
        e_sum += edge;
        e_sq_sum += edge * edge;
    }

    // Calculate the average scale
    double scale = 0;
    std::cout << " e_sq_sum / e_sum = " << e_sq_sum / e_sum << std::endl;
    scale = tag_size * e_sum / e_sq_sum;
    // for (auto cur_scale : scales){
    //     scale+= cur_scale;
    // }
    // scale = scale / scales.size();

    // // Calculate the scale using tag measurements
    // if (!tag_measurements.empty()) {
    //     // Output the center mappoint after transform
    //     std::map<std::string, Eigen::Vector3d> alignment_points_map;
    //     for (const auto &apriltag_id_pair : apriltag_id_map) {
    //         std::string apriltag_id = apriltag_id_pair.first;
    //         std::vector<mappoint_t> apriltag_mappoints = apriltag_mappoint_map[apriltag_id];
    //         if (!reconstruction->ExistsMapPoint(apriltag_mappoints[0])) {
    //             continue;
    //         }
    //         alignment_points_map[apriltag_id] = reconstruction->MapPoint(apriltag_mappoints[0]).XYZ();
    //     }

    //     // Check tag measurement validation
    //     int tag_measurement_counter = 0;
    //     for (auto tag_measurement : tag_measurements) {
    //         if (alignment_points_map.count(tag_measurement.first.first) &&
    //             alignment_points_map.count(tag_measurement.first.second)) {
    //             tag_measurement_counter++;
    //         }
    //     }

    //     if (tag_measurement_counter < 2) {
    //         std::cout << "Input Measurement TXT Error !!!" << std::endl;
    //         exit(-1);
    //     }

    //     double calculate_scale = scale_estimate(tag_measurements, alignment_points_map);
    //     if (calculate_scale != 1.0) {
    //         scale = calculate_scale;
    //     }
    // }

    std::cout << "Final scale = " << scale << std::endl;

    // // Construct the simaliraty transform
    // SimilarityTransform3 resize_map(scale, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d(0, 0, 0));

    // auto all_register_image_ids = reconstruction->RegisterImageIds();
    // // Transform Images
    // for (const auto image_id : all_register_image_ids) {
    //     auto &image = reconstruction->Image(image_id);
    //     resize_map.TransformPose(&image.Qvec(), &image.Tvec());  // Perfrom the calculated transform
    // }

    // // Transform Map Point
    // auto all_mappoint_ids = reconstruction->MapPointIds();
    // for (auto &mappoint : all_mappoint_ids) {
    //     auto &xyz = reconstruction->MapPoint(mappoint).XYZ();
    //     resize_map.TransformPoint(&xyz);
    //     reconstruction->MapPoint(mappoint).SetXYZ(xyz);
    // }

    // // Output the center mappoint after transform
    // for (const auto &apriltag_id_pair : apriltag_id_map) {
    //     std::string apriltag_id = apriltag_id_pair.first;
    //     std::vector<mappoint_t> apriltag_mappoints = apriltag_mappoint_map[apriltag_id];
    //     if (!reconstruction->ExistsMapPoint(apriltag_mappoints[0])) {
    //         continue;
    //     }
    //     alignment_points.emplace_back(
    //         std::make_pair(apriltag_id, reconstruction->MapPoint(apriltag_mappoints[0]).XYZ()));

    //     // FIXME: Remove the mappoint after transform
    //     // reconstruction->DeleteMapPoint(apriltag_mappoints[0]);
    // }

    return scale;
}

void CalculateMappointReprojectionError(
    std::shared_ptr<Reconstruction> reconstruction,
    std::unordered_map<mappoint_t, std::unordered_map<std::string, int>> &mappoint_apriltag_map) {
    for (const auto mappoint_id : mappoint_apriltag_map) {
        if (!reconstruction->ExistsMapPoint(mappoint_id.first)) {
            continue;
        }

        class MapPoint &mappoint = reconstruction->MapPoint(mappoint_id.first);
        double reproj_error_sum = 0.0;
        double tmp_error = 0.0, tmp_inlier_error = 0.0;
        int tmp_count = 0, tmp_inlier_count = 0;

        std::vector<TrackElement> track_elems_to_delete;

        for (const auto &track_el : mappoint.Track().Elements()) {
            const class Image &image = reconstruction->Image(track_el.image_id);
            const class Camera &camera = reconstruction->Camera(image.CameraId());
            const Point2D &point2D = image.Point2D(track_el.point2D_idx);
            const double squared_reproj_error =
                CalculateSquaredReprojectionError(point2D.XY(), mappoint.XYZ(), image.Qvec(), image.Tvec(), camera);
            reproj_error_sum += std::sqrt(squared_reproj_error);
        }
        mappoint.SetError(reproj_error_sum / mappoint.Track().Length());
    }
}

double SingleAreaFourPoints(const AprilTagDetection &tag_detection) {
    std::pair<float, float> p1 = tag_detection.p[0];
    std::pair<float, float> p2 = tag_detection.p[1];
    std::pair<float, float> p3 = tag_detection.p[2];
    std::pair<float, float> p4 = tag_detection.p[3];

    // std::cout << "Id = " << tag_detections[0].id << std::endl;
    // std::cout << "Coordinate = \n 0: " << tag_detections[0].p[0].first  << " , " << tag_detections[0].p[0].second
    //             << "\n 1:" << tag_detections[0].p[1].first  << " , " << tag_detections[0].p[1].second
    //             << "\n 2:" << tag_detections[0].p[2].first  << " , " << tag_detections[0].p[2].second
    //             << "\n 3:" << tag_detections[0].p[3].first  << " , " << tag_detections[0].p[3].second << std::endl;

    ////compute score
    float a = sqrt((p1.first - p2.first) * (p1.first - p2.first) + (p1.second - p2.second) * (p1.second - p2.second));
    float b = sqrt((p3.first - p2.first) * (p3.first - p2.first) + (p3.second - p2.second) * (p3.second - p2.second));
    float c = sqrt((p3.first - p4.first) * (p3.first - p4.first) + (p3.second - p4.second) * (p3.second - p4.second));
    float d = sqrt((p1.first - p4.first) * (p1.first - p4.first) + (p1.second - p4.second) * (p1.second - p4.second));
    float h = sqrt((p1.first - p3.first) * (p1.first - p3.first) + (p1.second - p3.second) * (p1.second - p3.second));
    float x1 = (a + b + h) / 2;
    float x2 = (c + d + h) / 2;
    float s1 = sqrt(x1 * (x1 - a) * (x1 - b) * (x1 - h));
    float s2 = sqrt(x2 * (x2 - c) * (x2 - d) * (x2 - h));
    double tag_area = s1 + s2;

    return tag_area;
}

bool CheckMarkerMoved(double tag_size,
                      std::unordered_map<std::string, std::vector<image_t>> &apriltag_id_map,
                      std::shared_ptr<FeatureDataContainer> feature_data_container,
                      std::unordered_map<mappoint_t, std::unordered_map<std::string, int>> &mappoint_apriltag_map,
                      std::unordered_map<std::string, std::vector<mappoint_t>> &apriltag_mappoint_map,
                      std::shared_ptr<Reconstruction> reconstruction,
                      std::vector<std::pair<std::string, Eigen::Vector3d>> &alignment_points,
                      std::shared_ptr<IncrementalMapper> mapper, IndependentMapperOptions mapper_options) {
    std::unordered_map<std::string, Eigen::Vector3d> alignment_points_map;
    for (const auto &tag : alignment_points) {
        alignment_points_map[tag.first] = tag.second;
    }

    auto options = mapper_options.IncrementalMapperOptions();
    options.abs_pose_min_num_inliers = 3;

    int correct_tag_counter = 0;
    for (const auto &apriltag_id_pair : apriltag_id_map) {
        std::string apriltag_id = apriltag_id_pair.first;
        const std::vector<image_t> &track_image_ids = apriltag_id_pair.second;
        std::vector<mappoint_t> apriltag_mappoints = apriltag_mappoint_map[apriltag_id];
        int min_track_length = 1000;
        bool all_mappoint_observation_correct = true;
        for (int i = 0; i < apriltag_mappoints.size(); i++) {
            mappoint_t mappoint_id = apriltag_mappoints[i];
            // Check mappoint exist or not
            if (!reconstruction->ExistsMapPoint(mappoint_id)) {
                // std::cout << " AprilTag Mappoint not exist ..." << std::endl;
                all_mappoint_observation_correct = false;
                continue;
            }

            // Check track length
            // if (sqrt(reconstruction->MapPoint(mappoint_id).Track().Length()) < min_track_length) {
            //     min_track_length = sqrt(reconstruction->MapPoint(mappoint_id).Track().Length());
            // }

            std::unordered_set<image_t> track_set;
            for (const auto &track : reconstruction->MapPoint(mappoint_id).Track().Elements()) {
                track_set.insert(track.image_id);
            }

            if (min_track_length > track_set.size()) {
                min_track_length = track_set.size();
            }
        }

        if (!all_mappoint_observation_correct) {
            std::cout << "Current Apriltag not reconstruct correctly" << std::endl;
            continue;
        }

        std::cout << "Min track length = " << min_track_length << std::endl;
        if (min_track_length < 3) {
            std::cout << "Min track length less than 3..." << std::endl;
            continue;
        }
        // // TODO: Evaluate value after the reconstruction sim3 applied
        // alignment_points[apriltag_id] = reconstruction->MapPoint(apriltag_mappoints[0]).XYZ();

        // Calculate the four edge of AprilTag
        auto mappoint_1 = reconstruction->MapPoint(apriltag_mappoints[1]).XYZ();
        auto mappoint_2 = reconstruction->MapPoint(apriltag_mappoints[2]).XYZ();
        auto mappoint_3 = reconstruction->MapPoint(apriltag_mappoints[3]).XYZ();
        auto mappoint_4 = reconstruction->MapPoint(apriltag_mappoints[4]).XYZ();

        // Check Each point average reproject error in all the tracked images
        std::vector<Eigen::Vector3d> marker_poses;
        for (const auto &track_image_id : track_image_ids) {
            const auto detections = feature_data_container->GetAprilTagDetections(track_image_id);
            const auto &recon_image = reconstruction->Image(track_image_id);
            const auto &recon_camera = reconstruction->Camera(recon_image.CameraId());

            const std::string &image_name = recon_image.Name();
            std::string video_name;
            TagVideoNameFromString(image_name, video_name);

            // Convert Video Name to Unix TimeStamp(ms)
            std::cout << "video_name = " << video_name << std::endl;
            std::string time_stamp = TimeStampFromVideoName(video_name);

            if (detections.empty()) {
                continue;
            }

            std::cout << "Image id = " << track_image_id << std::endl;

            // 2d detection
            for (auto tag_detection : detections) {
                auto p = tag_detection.p;

                std::vector<std::string> split_elems = StringSplit(apriltag_id, "_");
                if (tag_detection.id != std::stoi(split_elems[0])) {
                    continue;
                }

                // TODO: Check Multi Tag
                std::cout << "apriltag_id = " << apriltag_id << std::endl;
                std::cout << "time_stamp = " << time_stamp << std::endl;
                if (time_stamp != split_elems[1]) {
                    continue;
                }

                double area_size = SingleAreaFourPoints(tag_detection);

                if (area_size < 10 * 10) {
                    continue;
                }

                // Estimate T_cm
                Eigen::Matrix4d T_cm = Eigen::Matrix4d::Identity();
                ;

                std::vector<Eigen::Vector3d> objPts;
                double s = tag_size / 2.;
                objPts.emplace_back(-s, -s, 0);
                objPts.emplace_back(s, -s, 0);
                objPts.emplace_back(s, s, 0);
                objPts.emplace_back(-s, s, 0);

                std::pair<float, float> p1 = tag_detection.p[0];
                std::pair<float, float> p2 = tag_detection.p[1];
                std::pair<float, float> p3 = tag_detection.p[2];
                std::pair<float, float> p4 = tag_detection.p[3];

                std::vector<Eigen::Vector2d> imgPts;
                imgPts.emplace_back(p1.first, p1.second);
                imgPts.emplace_back(p2.first, p2.second);
                imgPts.emplace_back(p3.first, p3.second);
                imgPts.emplace_back(p4.first, p4.second);

                Eigen::Vector3d pose_tvec;
                Eigen::Vector4d pose_qvec;
                bool estimate_pose = false;
                if (recon_camera.NumLocalCameras() == 1) {
                    // Panorama or perspective
                    std::vector<char> inlier_mask;
                    estimate_pose = mapper->EstimateCameraPose(options, recon_camera, imgPts, objPts, pose_tvec,
                                                               pose_qvec, inlier_mask);
                } else {
                    // INSV Pro2
                    std::vector<char> inlier_mask;
                    std::vector<int> tri_camera_indices;
                    tri_camera_indices.emplace_back(tag_detection.local_camera_id);
                    tri_camera_indices.emplace_back(tag_detection.local_camera_id);
                    tri_camera_indices.emplace_back(tag_detection.local_camera_id);
                    tri_camera_indices.emplace_back(tag_detection.local_camera_id);
                    estimate_pose = mapper->EstimateCameraPoseRig(options, recon_camera, imgPts, tri_camera_indices,
                                                                  objPts, pose_tvec, pose_qvec, inlier_mask);
                }

                if (!estimate_pose) {
                    std::cout << "Estimate Pose Error " << std::endl;
                    continue;
                }

                const Eigen::Quaterniond quat_cm(pose_qvec(0), pose_qvec(1), pose_qvec(2), pose_qvec(3));
                T_cm.block(0, 0, 3, 3) = quat_cm.toRotationMatrix();
                T_cm.block(0, 3, 3, 1) = pose_tvec;
                // std::cout << "T_cm = " << T_cm << std::endl;

                Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
                auto q_cw = recon_image.Qvec();
                auto t_cw = recon_image.Tvec();
                const Eigen::Quaterniond quat_cw(q_cw(0), q_cw(1), q_cw(2), q_cw(3));
                T_cw.block(0, 0, 3, 3) = quat_cw.toRotationMatrix();
                // std::cout << "T_wc_r = " << T_wc << std::endl;
                T_cw.block(0, 3, 3, 1) = t_cw;
                // std::cout << "T_cw = " << T_cw << std::endl;

                // std::cout << "T_wc = " << T_wc << std::endl;
                Eigen::Matrix4d T_wm = T_cw.inverse() * T_cm;
                // std::cout << "T_wm = " << T_wm << std::endl;
                Eigen::Matrix3d r_wm = T_wm.block(0, 0, 3, 3);
                // std::cout << "T_wm_r = " << r_wm << std::endl;
                const Eigen::Quaterniond quat_wm(r_wm);
                Eigen::Vector4d q_wm = Eigen::Vector4d(quat_wm.w(), quat_wm.x(), quat_wm.y(), quat_wm.z());
                Eigen::Vector3d t_wm = T_wm.block(0, 3, 3, 1);

                marker_poses.emplace_back(t_wm);
            }
        }

        // // Calculate marker average pose
        // Eigen::Vector3d marker_average_pose(0, 0, 0);
        for (int i = 0; i < marker_poses.size(); i++) {
            // marker_average_pose = marker_average_pose + marker_poses[i];
            std::cout << "marker_poses[i] = " << marker_poses[i] << std::endl;
        }

        auto inlier_marker_poses = ransac_pose_select(marker_poses, 2);
        std::cout << "inlier_marker_poses.size() = " << inlier_marker_poses.size() << std::endl;
        std::cout << "marker_poses.size() / 2 = " << marker_poses.size() / 2 << std::endl;
        if (inlier_marker_poses.size() > marker_poses.size() / 2) {
            correct_tag_counter++;
            // return true;
        }

        // std::cout << "marker_poses.size() = " << marker_poses.size() << std::endl;
        // marker_average_pose = marker_average_pose/marker_poses.size();
        // std::cout << "marker_average_pose = " << marker_average_pose << std::endl;
        // std::cout << " alignment_points_map[apriltag_id] = " << alignment_points_map[apriltag_id] << std::endl;
        // std::cout << "(marker_average_pose - alignment_points_map[apriltag_id]).norm() = " << (marker_average_pose -
        // alignment_points_map[apriltag_id]).norm() << std::endl;

        // if ((marker_average_pose - alignment_points_map[apriltag_id]).norm() > 0.5) {

        //     // Exist Move Marker
        //     return true;
        // }
    }

    return correct_tag_counter == 0;
}

TagScaleRecover::TagScaleRecover(TagScaleRecoverOptions options) : options_(options) {}

double TagScaleRecover::ComputeScale(std::shared_ptr<FeatureDataContainer> feature_data_container,
                                     std::shared_ptr<SceneGraphContainer> scene_graph_container,
                                     std::shared_ptr<Reconstruction> reconstruction_input) {
    FILE *fs;
    fs = fopen((options_.workspace_path + "/tag_statistics.txt").c_str(), "w");

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();

    // Copy Recontruction
    *(reconstruction.get()) = *(reconstruction_input.get());

    const std::unordered_set<mappoint_t> &const_mappoint_ids = reconstruction->MapPointIds();
    const auto image_ids = reconstruction->RegisterImageIds();
    //    auto mappoint_num = reconstruction->MapPointIds().size() + 10;
    //    std::cout << "mappoint_num = " << mappoint_num << std::endl;

    //////////////////////////////////////////////////////////////////////////////
    // Calculate AprilTag for each image
    //////////////////////////////////////////////////////////////////////////////
    // april_tag_id -----> image_ids
    std::unordered_map<std::string, std::vector<image_t>> apriltag_id_map;
    for (const auto image_id : image_ids) {
        const Image &image = reconstruction->Image(image_id);
        const auto detections = feature_data_container->GetAprilTagDetections(image_id);
        if (detections.empty()) {
            continue;
        }
        const std::string &image_name = image.Name();
        std::string video_name;
        TagVideoNameFromString(image_name, video_name);

        // Convert Video Name to Unix TimeStamp(ms)
        std::string time_stamp = TimeStampFromVideoName(video_name);

        for (const auto &detection : detections) {
            // Check detection area size
            // double area_size = AreaFourPoints(detection);
            // if (area_size > 20 * 20) {
            std::string detection_label = std::to_string(detection.id) + "_" + time_stamp;
            std::cout << "detection_label = " << detection_label << std::endl;
            apriltag_id_map[detection_label].emplace_back(image_id);
            // }
        }
    }

    if (apriltag_id_map.empty()) {
        fprintf(fs, "%s\n", StringPrintf("ERROR: No Tag Detection. Failed to Recover Scale").c_str());
        fflush(fs);
        return 1.0;
    }

    // Print the raw track length for each apriltag
    for (const auto &apriltag : apriltag_id_map) {
        std::cout << "Tag label = " << apriltag.first << " , track length = " << apriltag.second.size() << std::endl;
    }

    // Add correspondence for each april tage point
    // mappoint_id -----> label_id ------> point_id
    std::unordered_map<mappoint_t, std::unordered_map<std::string, int>> mappoint_apriltag_map;
    std::unordered_map<std::string, std::vector<mappoint_t>> apriltag_mappoint_map;
    for (const auto &apriltag_id_pair : apriltag_id_map) {
        std::string label_id = apriltag_id_pair.first;

        // Add correspondence and update track
        std::vector<Track> mappoint_track;
        mappoint_track.resize(5);
        for (int i = 0; i < apriltag_id_pair.second.size(); i++) {
            for (int j = 0; j < apriltag_id_pair.second.size(); j++) {
                if (i == j || i > j) {
                    continue;
                }

                // Get image id from pair
                const image_t image_id_1 = apriltag_id_pair.second[i];
                const image_t image_id_2 = apriltag_id_pair.second[j];

                // Check two image id exist in the reconstruction or not
                if (!reconstruction->ExistsImage(image_id_1) || !reconstruction->ExistsImage(image_id_2)) {
                    // Skip all the image which do not contain pose
                    // std::cout << "Image not exist ... " << std::endl;
                    continue;
                }

                // Get correct detection from all the image detections
                AprilTagDetection detection_1, detection_2;
                detection_1.id = -1;
                detection_2.id = -1;

                // Get detect points
                std::vector<class Point2D> detect_points_1, detect_points_2;
                std::vector<point2D_t> point2d_ids1, point2d_ids2;
                for (const auto &detection : feature_data_container->GetAprilTagDetections(image_id_1)) {
                    std::vector<std::string> split_elems = StringSplit(label_id, "_");
                    if (detection.id == std::stoi(split_elems[0])) {
                        detect_points_1.clear();
                        detection_1 = detection;
                        // std::cout << "Detection image id = " << image_id_1 << std::endl;
                        // std::cout << "Detection local camera id = " << detection.local_camera_id << std::endl;

                        std::vector<uint32_t> local_image_indices;

                        Point2D cxy, p_0, p_1, p_2, p_3;
                        cxy.SetXY(Eigen::Vector2d(detection_1.cxy.first, detection_1.cxy.second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        cxy.SetMapPointId(mappoint_num);
                        detect_points_1.emplace_back(cxy);

                        p_0.SetXY(Eigen::Vector2d(detection_1.p[0].first, detection_1.p[0].second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        p_0.SetMapPointId(mappoint_num + 1);
                        detect_points_1.emplace_back(p_0);

                        p_1.SetXY(Eigen::Vector2d(detection_1.p[1].first, detection_1.p[1].second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        p_1.SetMapPointId(mappoint_num + 2);
                        detect_points_1.emplace_back(p_1);

                        p_2.SetXY(Eigen::Vector2d(detection_1.p[2].first, detection_1.p[2].second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        p_2.SetMapPointId(mappoint_num + 3);
                        detect_points_1.emplace_back(p_2);

                        p_3.SetXY(Eigen::Vector2d(detection_1.p[3].first, detection_1.p[3].second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        p_3.SetMapPointId(mappoint_num + 4);
                        detect_points_1.emplace_back(p_3);

                        std::vector<point2D_t> rec_point2d_ids1;

                        // Add point to scene graph container
                        point2d_ids1 =
                            scene_graph_container->Image(image_id_1).AddPoints2D(detect_points_1, local_image_indices);
                        // Add point2d to reconstruction
                        rec_point2d_ids1 =
                            reconstruction->Image(image_id_1).AddPoints2D(detect_points_1, local_image_indices);

                        // Check two point2d list is same or not
                        for (int point2d_ids_list_id = 0; point2d_ids_list_id < 5; point2d_ids_list_id++) {
                            if (point2d_ids1[point2d_ids_list_id] != rec_point2d_ids1[point2d_ids_list_id]) {
                                std::cout << "1 Error!! Point2d id not the same ... "
                                          << point2d_ids1[point2d_ids_list_id] << " .. "
                                          << rec_point2d_ids1[point2d_ids_list_id] << std::endl;
                                exit(-1);
                            }

#ifdef DEBUG_APRILTAG_TRI
                            // Convert vector to set
                            if (point2d_ids1[point2d_ids_list_id] <
                                reconstruction_old->Image(image_id_1).Points2D().size()) {
                                std::cout << "Point2d Already Exist .... " << point2d_ids1[point2d_ids_list_id]
                                          << std::endl;
                                exit(-1);
                            }
#endif
                            // Update Obervation number
                            scene_graph_container->Image(image_id_1)
                                .SetNumObservations(scene_graph_container->Image(image_id_1).NumObservations() + 5);
                            reconstruction->Image(image_id_1)
                                .SetNumObservations(reconstruction->Image(image_id_1).NumObservations() + 5);
                        }

                        break;
                    }
                }

                for (const auto &detection : feature_data_container->GetAprilTagDetections(image_id_2)) {
                    std::vector<std::string> split_elems = StringSplit(label_id, "_");
                    if (detection.id == std::stoi(split_elems[0])) {
                        detect_points_2.clear();
                        detection_2 = detection;

                        // std::cout << "Detection image id = " << image_id_2 << std::endl;
                        // std::cout << "Detection local camera id = " << detection.local_camera_id << std::endl;

                        std::vector<uint32_t> local_image_indices;
                        Point2D cxy, p_0, p_1, p_2, p_3;
                        cxy.SetXY(Eigen::Vector2d(detection_2.cxy.first, detection_2.cxy.second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        cxy.SetMapPointId(mappoint_num);
                        detect_points_2.emplace_back(cxy);

                        p_0.SetXY(Eigen::Vector2d(detection_2.p[0].first, detection_2.p[0].second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        p_0.SetMapPointId(mappoint_num + 1);
                        detect_points_2.emplace_back(p_0);

                        p_1.SetXY(Eigen::Vector2d(detection_2.p[1].first, detection_2.p[1].second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        p_1.SetMapPointId(mappoint_num + 2);
                        detect_points_2.emplace_back(p_1);

                        p_2.SetXY(Eigen::Vector2d(detection_2.p[2].first, detection_2.p[2].second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        p_2.SetMapPointId(mappoint_num + 3);
                        detect_points_2.emplace_back(p_2);

                        p_3.SetXY(Eigen::Vector2d(detection_2.p[3].first, detection_2.p[3].second));
                        local_image_indices.emplace_back(detection.local_camera_id);
                        //                        p_3.SetMapPointId(mappoint_num + 4);
                        detect_points_2.emplace_back(p_3);

                        std::vector<point2D_t> rec_point2d_ids2;

                        // Add point to scene graph container
                        point2d_ids2 =
                            scene_graph_container->Image(image_id_2).AddPoints2D(detect_points_2, local_image_indices);
                        // Add point2d to reconstruction
                        rec_point2d_ids2 =
                            reconstruction->Image(image_id_2).AddPoints2D(detect_points_2, local_image_indices);

                        // Check two point2d list is same or not
                        for (int point2d_ids_list_id = 0; point2d_ids_list_id < 5; point2d_ids_list_id++) {
                            if (point2d_ids2[point2d_ids_list_id] != rec_point2d_ids2[point2d_ids_list_id]) {
                                std::cout << "Error!! Point2d id not the same ... " << point2d_ids2[point2d_ids_list_id]
                                          << " .. " << rec_point2d_ids2[point2d_ids_list_id] << std::endl;
                                exit(-1);
                            }

#ifdef DEBUG_APRILTAG_TRI
                            // Convert vector to set
                            if (point2d_ids2[point2d_ids_list_id] <
                                reconstruction_old->Image(image_id_2).Points2D().size()) {
                                std::cout << "Point2d Already Exist .... " << point2d_ids2[point2d_ids_list_id]
                                          << std::endl;
                                exit(-1);
                            }
#endif

                            // Update Obervation number
                            scene_graph_container->Image(image_id_2)
                                .SetNumObservations(scene_graph_container->Image(image_id_2).NumObservations() + 5);
                            reconstruction->Image(image_id_2)
                                .SetNumObservations(reconstruction->Image(image_id_2).NumObservations() + 5);
                        }
                        break;
                    }
                }

                // Check detection_1 and detection_2
                if (detection_1.id == -1 || detection_2.id == -1) {
                    std::cout << "Error!!! AprilTag detection result error" << std::endl;
                    exit(-1);
                }

                // Add point2d, correspondence
                for (int point_id = 0; point_id < 5; point_id++) {
                    // std::cout << "Image id 1 = " << image_id_1 << " , Image id 2 = " << image_id_2 << std::endl;
                    scene_graph_container->CorrespondenceGraph()->UpdateCorrespondence(
                        image_id_1, image_id_2, point2d_ids1[point_id], point2d_ids2[point_id]);

                    // Update Correspondence
                    scene_graph_container->Image(image_id_1)
                        .SetNumCorrespondences(scene_graph_container->Image(image_id_1).NumCorrespondences() + 1);
                    reconstruction->Image(image_id_1)
                        .SetNumCorrespondences(reconstruction->Image(image_id_1).NumCorrespondences() + 1);

                    scene_graph_container->Image(image_id_2)
                        .SetNumCorrespondences(scene_graph_container->Image(image_id_2).NumCorrespondences() + 1);
                    reconstruction->Image(image_id_2)
                        .SetNumCorrespondences(reconstruction->Image(image_id_2).NumCorrespondences() + 1);

                    // Update track
                    mappoint_track[point_id].AddElement(image_id_1, point2d_ids1[point_id]);
                    mappoint_track[point_id].AddElement(image_id_2, point2d_ids2[point_id]);
                }
            }
        }

        // Update the Mappoint with track to reconstruction
        for (int i = 0; i < 5; i++) {
            mappoint_t mappoint_number =
                reconstruction->AddMapPoint(Eigen::Vector3d(0, 0, 0), mappoint_track[i], Eigen::Vector3ub(255, 0, 0));
            mappoint_apriltag_map[mappoint_number][label_id] = i;
            apriltag_mappoint_map[label_id].emplace_back(mappoint_number);
        }
    }

    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);

    IndependentMapperOptions mapper_options;

    // Display the camera basic info
    auto rec_cameras = reconstruction->Cameras();
    std::cout << "Camera number = " << rec_cameras.size() << std::endl;

    for (auto camera : rec_cameras) {
        std::cout << "  Camera index = " << camera.first << std::endl;
        std::cout << "  Camera model = " << camera.second.ModelName() << std::endl;
        std::cout << "  Camera param = ";
        for (auto param : camera.second.Params()) {
            std::cout << "  " << param;
        }
        std::cout << std::endl;

        // Sub Camera
        std::cout << " Sub Camera Number = " << camera.second.NumLocalCameras() << std::endl;
        if (camera.second.NumLocalCameras() > 1) {
            std::vector<double> params;
            camera.second.GetLocalCameraIntrisic(camera.first, params);
            std::cout << "  Sub Camera intrinsic = ";
            for (auto param : params) {
                std::cout << "  " << param;
            }
            std::cout << std::endl;
        }
    }

    //////////////////////////////////////////////////////////////////////////////
    // Triangulation AprilTag Points
    //////////////////////////////////////////////////////////////////////////////
    const auto tri_options = mapper_options.Triangulation();
    size_t mappoint_counter = 0;
    for (const auto mappoint_apriltag_pair : mappoint_apriltag_map) {
        const mappoint_t mappoint_id = mappoint_apriltag_pair.first;
        PrintHeading1(StringPrintf("Triangulating mappoint #%d", mappoint_id));

        size_t tri_num = mapper->TriangulateMappoint(tri_options, mappoint_id);
        std::cout << "Triangulate point number = " << tri_num << std::endl;
        mappoint_counter = mappoint_counter + tri_num;
    }

    if (mappoint_counter == 0) {
        std::cout << "ERROR: no mappoint has been triangulated ... " << std::endl;
        fprintf(fs, "%s\n", StringPrintf("ERROR: No Valid Tag Detection. Failed to Recover Scale").c_str());
        fflush(fs);
        return 1.0;
    }

    //////////////////////////////////////////////////////////////////////////////
    // Bundle adjustment
    //////////////////////////////////////////////////////////////////////////////

    BundleAdjustmentOptions ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.solver_options.minimizer_progress_to_stdout = false;
    ba_options.solver_options.max_num_iterations = 30;
    ba_options.solver_options.max_linear_solver_iterations = 100;

    ba_options.refine_focal_length = false;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = false;

    // Configure bundle adjustment.
    // Avoid degeneracies in bundle adjustment.
    reconstruction->FilterObservationsWithNegativeDepth();

    const size_t num_observations = reconstruction->ComputeNumObservations();

    BundleAdjustmentConfig ba_config;
    for (auto &image_id : image_ids) {
        ba_config.AddImage(image_id);
        const auto& image = reconstruction->Image(image_id);
        const Camera &camera = reconstruction->Camera(image.CameraId());
        ba_config.SetConstantCamera(image.CameraId());
        ba_config.SetConstantPose(image_id);
    }

    // for (const auto &mappoint_id : const_mappoint_ids) {
    //     if (reconstruction->ExistsMapPoint(mappoint_id)) {
    //         ba_config.AddConstantPoint(mappoint_id);
    //     }
    // }

    for (int i = 0; i < 3; i++) {
        PrintHeading1("AprilTag Bundle adjustment");

        if (ba_config.NumImages() > 1) {
            BundleAdjuster bundle_adjuster(ba_options, ba_config);
            if (!bundle_adjuster.Solve(reconstruction.get())) {
                std::cout << "Bundle Adjustment Failed!" << std::endl;
            }
        }

        const size_t num_filtered_observations = mapper->FilterPoints(mapper_options.IncrementalMapperOptions());
        std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;

        // Output the mappoint reprojection error
        for (auto mappoint_id_pair : mappoint_apriltag_map) {
            if (!reconstruction->ExistsMapPoint(mappoint_id_pair.first)) {
                continue;
            }
            mappoint_t mappoint_id = mappoint_id_pair.first;
            std::cout << "Mappoint id " << mappoint_id
                      << " , Reprojection error = " << reconstruction->MapPoint(mappoint_id).Error() << std::endl;
        }
    }

    // Calculate the reprojection error for each mappoint
    CalculateMappointReprojectionError(reconstruction, mappoint_apriltag_map);

    double scale = ComputeReconstructionScale(options_.tag_size, apriltag_id_map, feature_data_container, mappoint_apriltag_map,
                                              apriltag_mappoint_map, reconstruction, alignment_points_, alignment_tag_points_);

    if (scale == 0 || std::isnan(scale)) {
        //
        fprintf(fs, "%s\n", StringPrintf("ERROR: Failed to Recover Scale").c_str());
        fflush(fs);
        return 1.0;
    }

    // Check Marker is moved or not
    bool marker_moved =
        CheckMarkerMoved(options_.tag_size, apriltag_id_map, feature_data_container, mappoint_apriltag_map, apriltag_mappoint_map,
                         reconstruction, alignment_points_, mapper, mapper_options);

    if (marker_moved) {
        fprintf(fs, "%s\n", StringPrintf("ERROR: Marker is Moved").c_str());
        fflush(fs);
        return 1.0;
    }

    const bool kDiscardReconstruction = false;
    mapper->EndReconstruction(kDiscardReconstruction);

    fprintf(
        fs, "%s\n",
        StringPrintf("SUCCESS: Scale Recovery Finished. Scale = %.3f", scale)
            .c_str());
    fflush(fs);

    return scale;
}

}  // namespace sensemap