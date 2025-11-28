// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "gps_reader.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "math.h"
#include "util/misc.h"
#include <iomanip>

namespace sensemap {

static constexpr double sm_a = 6378137.0;
static constexpr double sm_b = 6356752.314;
static constexpr double UTMScaleFactor = 0.9996;

double UTMCentralMeridian (double zone) {
    double cmeridian;
    double deg = -183.0 + (zone * 6.0);
    cmeridian = deg / 180.0 * M_PI;
    return cmeridian;
}

double ArcLengthOfMeridian(double phi) {
    double alpha, beta, gamma, delta, epsilon, n;
    double result;

    /* Precalculate n */
    n = (sm_a - sm_b) / (sm_a + sm_b);

    /* Precalculate alpha */
    alpha = ((sm_a + sm_b) / 2.0)
        * (1.0 + (std::pow(n, 2.0) / 4.0) + (std::pow(n, 4.0) / 64.0));

    /* Precalculate beta */
    beta = (-3.0 * n / 2.0) + (9.0 * std::pow(n, 3.0) / 16.0)
        + (-3.0 * std::pow(n, 5.0) / 32.0);

    /* Precalculate gamma */
    gamma = (15.0 * std::pow(n, 2.0) / 16.0)
        + (-15.0 * std::pow(n, 4.0) / 32.0);

    /* Precalculate delta */
    delta = (-35.0 * std::pow(n, 3.0) / 48.0)
        + (105.0 * std::pow(n, 5.0) / 256.0);

    /* Precalculate epsilon */
    epsilon = (315.0 * std::pow(n, 4.0) / 512.0);

    /* Now calculate the sum of the series and return */
    result = alpha
        * (phi + (beta * std::sin (2.0 * phi))
            + (gamma * std::sin(4.0 * phi))
            + (delta * std::sin(6.0 * phi))
            + (epsilon * std::sin(8.0 * phi)));

    return result;
}

void MapLatLonToXY (double phi, double lambda, double lambda0, double* x, double* y) {
    double N, nu2, ep2, t, t2, l;
    double l3coef, l4coef, l5coef, l6coef, l7coef, l8coef;
    double tmp;

    /* Precalculate ep2 */
    ep2 = (std::pow(sm_a, 2.0) - std::pow(sm_b, 2.0)) / std::pow(sm_b, 2.0);

    /* Precalculate nu2 */
    nu2 = ep2 * std::pow(std::cos(phi), 2.0);

    /* Precalculate N */
    N = std::pow(sm_a, 2.0) / (sm_b * std::sqrt(1 + nu2));

    /* Precalculate t */
    t = std::tan (phi);
    t2 = t * t;
    tmp = (t2 * t2 * t2) - std::pow (t, 6.0);

    /* Precalculate l */
    l = lambda - lambda0;

    /* Precalculate coefficients for l**n in the equations below
        so a normal human being can read the expressions for easting
        and northing
        -- l**1 and l**2 have coefficients of 1.0 */
    l3coef = 1.0 - t2 + nu2;

    l4coef = 5.0 - t2 + 9 * nu2 + 4.0 * (nu2 * nu2);

    l5coef = 5.0 - 18.0 * t2 + (t2 * t2) + 14.0 * nu2
        - 58.0 * t2 * nu2;

    l6coef = 61.0 - 58.0 * t2 + (t2 * t2) + 270.0 * nu2
        - 330.0 * t2 * nu2;

    l7coef = 61.0 - 479.0 * t2 + 179.0 * (t2 * t2) - (t2 * t2 * t2);

    l8coef = 1385.0 - 3111.0 * t2 + 543.0 * (t2 * t2) - (t2 * t2 * t2);

    /* Calculate easting (x) */

    *x = N * std::cos (phi) * l
        + (N / 6.0 * std::pow (std::cos (phi), 3.0) * l3coef * std::pow (l, 3.0))
        + (N / 120.0 * std::pow(std::cos(phi), 5.0) * l5coef * std::pow(l, 5.0))
        + (N / 5040.0 * std::pow(std::cos(phi), 7.0) * l7coef * std::pow(l, 7.0));

    /* Calculate northing (y) */
    *y = ArcLengthOfMeridian (phi)
        + (t / 2.0 * N * std::pow(std::cos(phi), 2.0) * std::pow(l, 2.0))
        + (t / 24.0 * N * std::pow(std::cos(phi), 4.0) * l4coef * std::pow(l, 4.0))
        + (t / 720.0 * N * std::pow(std::cos(phi), 6.0) * l6coef * std::pow(l, 6.0))
        + (t / 40320.0 * N * std::pow(std::cos(phi), 8.0) * l8coef * std::pow(l, 8.0));

    return;
}

bool GPSReader::Load(const std::string& file_path, bool abuse_altitude, int accuracy_level_threshold) {
    if (file_path.empty()) return false;
    std::cout << "load gps info" << std::endl;
    // open file
    std::ifstream infile;
    infile.open(file_path, std::ios::binary | std::ios::in);
    if (!infile.is_open()) {
        return false;
    }

    std::string line;
    std::string item;
    time_locations_.clear();
    std::string head;
    std::getline(infile, head);
    while (std::getline(infile, line)) {
        StringTrim(&line);

        std::stringstream line_stream(line);

        // // time_stamp_slam
        // std::getline(line_stream, item, ',');
        // unsigned long slam_time = std::stoul(item);

        // // longitude
        // std::getline(line_stream, item, ',');
        // double longitude = std::stod(item);

        // // latitude
        // std::getline(line_stream, item, ',');
        // double latitude = std::stod(item);

        // // altitude
        // std::getline(line_stream, item, ',');
        // double altitude = std::stod(item);

        // // horizontal_accuracy
        // std::getline(line_stream, item, ',');
        // double horizontal_accuracy = std::stod(item);
        
        // // vertical_accuracy
        // std::getline(line_stream, item, ',');
        // double vertical_accuracy = std::stod(item);

        // // system time
        // std::getline(line_stream, item, ',');
        // unsigned long system_time = std::stoul(item);



        std::getline(line_stream, item, ',');
        unsigned long system_time = std::stoul(item);

        // latitude
        std::getline(line_stream, item, ',');
        double latitude = std::stod(item);

        // longitude
        std::getline(line_stream, item, ',');
        double longitude = std::stod(item);

        // altitude
        std::getline(line_stream, item, ',');
        double altitude = std::stod(item);

        // accuracy level
        std::getline(line_stream, item, ',');
        int accuracy_level = std::stoi(item);

        if(accuracy_level < accuracy_level_threshold){
            continue;
        }

        if (abuse_altitude) {
            altitude = 0.0;
        }
        Eigen::Vector3d coord = gpsToLocation(latitude, longitude, altitude);
        std::pair<Eigen::Vector3d,int> coord_accuracy(coord,accuracy_level);

        time_locations_.emplace(system_time, coord_accuracy);
    }
    infile.close();
    return true;
}

Eigen::Vector3d GPSReader::gpsToLocation(double latitude, double longitude, double altitude) {
    double cosLat = std::cos(latitude * M_PI / 180);
    double sinLat = std::sin(latitude * M_PI / 180);

    double cosLong = std::cos(longitude * M_PI / 180);
    double sinLong = std::sin(longitude * M_PI / 180);

    double c = 1 / std::sqrt(cosLat * cosLat + (1 - f_) * (1 - f_) * sinLat * sinLat);
    double s = (1 - f_) * (1 - f_) * c;

    double x = (R_ * c + altitude) * cosLat * cosLong;
    double y = (R_ * c + altitude) * cosLat * sinLong;
    double z = (R_ * s + altitude) * sinLat;

    return {x, y, z};
}

void GPSReader::LocationToGps(const Eigen::Vector3d X, double *latitude, double *longitude, double *altitude) {
    const double a = R_;
    const double b = a * (1 - f_);
    double c, d, p, q;
    double N;
    c = std::sqrt(((a * a) - (b * b)) / (a * a));
    d = std::sqrt(((a * a) - (b * b)) / (b * b));
    p = std::sqrt((X[0] * X[0]) + (X[1] * X[1]));
    q = std::atan2((X[2] * a), (p * b));
    *longitude = std::atan2(X[1], X[0]);
    *latitude = std::atan2((X[2] + (d * d) * b * std::pow(std::sin(q), 3)), (p - (c * c) * a * std::pow(std::cos(q), 3)));
    N = a / std::sqrt(1 - ((c * c) * std::pow(std::sin(*latitude), 2)));
    *altitude = (p / std::cos(*latitude)) - N;
    *longitude = *longitude * 180.0 / M_PI;
    *latitude = *latitude * 180.0 / M_PI;
}

Eigen::Vector3d GPSReader::gpsToUTM(double latitude, double longtitude) {
    double zone = std::floor((longtitude + 180.0) / 6) + 1;
    double cm = UTMCentralMeridian(zone);
 
    double x, y;
    MapLatLonToXY(latitude / 180.0 * M_PI, longtitude / 180 * M_PI, cm, &x, &y);

    /* Adjust easting and northing for UTM system. */
    x = x * UTMScaleFactor + 500000.0;
    y = y * UTMScaleFactor;
    if (y < 0.0) {
        y = y + 10000000.0;
    }

    return Eigen::Vector3d(x, y, zone);
}

GeodeticConverter::GeodeticConverter(double latitude, double longitude, double altitude) {
    ecef_origin_ref_ = GPSReader::gpsToLocation(latitude, longitude, altitude);

    const double latitude_rad = DEG2RAD(latitude);
    const double longitude_rad = DEG2RAD(longitude);
    const double sin_lat = std::sin(latitude_rad);
    const double cos_lat = std::cos(latitude_rad);
    const double sin_lon = std::sin(longitude_rad);
    const double cos_lon = std::cos(longitude_rad);
    ned_to_ecef_(0, 0) = -sin_lat * cos_lon;
    ned_to_ecef_(0, 1) = -sin_lon;
    ned_to_ecef_(0, 2) = -cos_lat * cos_lon;
    ned_to_ecef_(1, 0) = -sin_lat * sin_lon;
    ned_to_ecef_(1, 1) = cos_lon;
    ned_to_ecef_(1, 2) = -cos_lat * sin_lon;
    ned_to_ecef_(2, 0) = cos_lat;
    ned_to_ecef_(2, 1) = 0;
    ned_to_ecef_(2, 2) = -sin_lat;
}

void GeodeticConverter::LLAToNed(double latitude, double longitude, 
    double altitude, double *n, double *e, double *d) {
    double x, y, z;
    LLAToEcef(latitude, longitude, altitude, &x, &y, &z);
    EcefToNed(x, y, z, n, e, d);
}

void GeodeticConverter::LLAToEcef(double latitude, double longitude, 
    double altitude, double *x, double *y, double *z) {
    Eigen::Vector3d X = GPSReader::gpsToLocation(latitude, longitude, altitude);
    *x = X.x();
    *y = X.y();
    *z = X.z();
}

void GeodeticConverter::EcefToNed(double x, double y, double z, double *n, double *e, double *d) {
    x -= ecef_origin_ref_.x();
    y -= ecef_origin_ref_.y();
    z -= ecef_origin_ref_.z();
    *n = ned_to_ecef_(0, 0) * x + ned_to_ecef_(1, 0) * y + ned_to_ecef_(2, 0) * z;
    *e = ned_to_ecef_(0, 1) * x + ned_to_ecef_(1, 1) * y + ned_to_ecef_(2, 1) * z;
    *d = ned_to_ecef_(0, 2) * x + ned_to_ecef_(1, 2) * y + ned_to_ecef_(2, 2) * z;
}

Eigen::Matrix3x4d GeodeticConverter::NedToEcefMatrix() {
    Eigen::Matrix3x4d M;
    M.block<3, 3>(0, 0) = ned_to_ecef_;
    M.block<3, 1>(0, 3) = ecef_origin_ref_;
    return M;
}

Eigen::Matrix3x4d GeodeticConverter::EcefToNedMatrix() {
    Eigen::Matrix3x4d M;
    M.block<3, 3>(0, 0) = ned_to_ecef_.transpose();
    M.block<3, 1>(0, 3) = -ned_to_ecef_.transpose() * ecef_origin_ref_;
    return M;
}

bool TimeCompare(const std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>& p1,
                 const std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>& p2) {
    return p1.first < p2.first;
}

bool LoadOriginGPSinfo(const std::string& file_path,
                       std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d, int>>>& gps_locations,
                       const std::string& gps_trans_path, bool abuse_altitude, int accuracy_level_threshold) {
    std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d, int>>> gps_locations_horizontal;

    if (HasFileExtension(file_path, ".txt") || HasFileExtension(file_path, ".csv")) {
        GPSReader gps_reader;

        // load full gps
        if (!gps_reader.Load(file_path, false, accuracy_level_threshold)) {
            return false;
        }
        std::unordered_map<unsigned long, std::pair<Eigen::Vector3d, int>> time_locations =
            gps_reader.GetTimeLocations();
        std::cout << "gps locations size: " << time_locations.size() << std::endl;

        for (const auto& time_location : time_locations) {
            gps_locations.push_back(time_location);
        }

        // load gps without altitude
        if (!gps_reader.Load(file_path, true, accuracy_level_threshold)) {
            return false;
        }
        std::unordered_map<unsigned long, std::pair<Eigen::Vector3d, int>> time_locations_horizontal =
            gps_reader.GetTimeLocations();
        for (const auto& time_location_horizontal : time_locations_horizontal) {
            gps_locations_horizontal.push_back(time_location_horizontal);
        }

    } else {
        std::vector<std::string> file_list = sensemap::GetRecursiveFileList(file_path);
        for (auto file : file_list) {
            auto name = GetPathBaseName(file);
            if (name == "gps.csv" || name == "gps.txt") {
                GPSReader gps_reader;

                // load full gps
                if (!gps_reader.Load(file, false, accuracy_level_threshold)) {
                    return false;
                }
                std::unordered_map<unsigned long, std::pair<Eigen::Vector3d, int>> time_locations =
                    gps_reader.GetTimeLocations();
                std::cout << "gps locations size: " << time_locations.size() << std::endl;
                for (const auto& time_location : time_locations) {
                    gps_locations.push_back(time_location);
                }

                // load gps without altitude
                if (!gps_reader.Load(file, true, accuracy_level_threshold)) {
                    return false;
                }

                std::unordered_map<unsigned long, std::pair<Eigen::Vector3d, int>> time_locations_horizontal =
                    gps_reader.GetTimeLocations();
                for (const auto& time_location_horizontal : time_locations_horizontal) {
                    gps_locations_horizontal.push_back(time_location_horizontal);
                }
            }
        }
    }

    CHECK(gps_locations_horizontal.size() == gps_locations.size());

    Eigen::Vector3d mean_coord(0, 0, 0);
    for (const auto& location : gps_locations) {
        mean_coord = mean_coord + location.second.first;
    }
    mean_coord = mean_coord / static_cast<double>(gps_locations.size());

    for (auto& location : gps_locations) {
        location.second.first = location.second.first - mean_coord;
    }

    // record the mean coord in the file so as to use gps prior in ECEF frame
    std::ofstream gps_trans_file(gps_trans_path);
    CHECK(gps_trans_file.is_open()) << "gps trans file can not be opened";

    gps_trans_file << std::fixed << std::setprecision(15) << mean_coord(0) << " " << mean_coord(1) << " "
                   << mean_coord(2) << " " << std::endl;

    for (auto& location_horizontal : gps_locations_horizontal) {
        location_horizontal.second.first = location_horizontal.second.first - mean_coord;
    }

    // estimate gravity direction
    std::cout << "estimate gravity direction" << std::endl;
    Eigen::Vector3d mean_coord_horizontal(0, 0, 0);
    for (const auto& location_horizontal : gps_locations_horizontal) {
        mean_coord_horizontal = mean_coord_horizontal + location_horizontal.second.first;
    }
    mean_coord_horizontal = mean_coord_horizontal / static_cast<double>(gps_locations_horizontal.size());

    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();

    for (const auto& location_horizontal : gps_locations_horizontal) {
        ATA += (location_horizontal.second.first - mean_coord_horizontal) *
               (location_horizontal.second.first - mean_coord_horizontal).transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(ATA);
    Eigen::Vector3d eigenValues = eigenSolver.eigenvalues();
    Eigen::Matrix3d eigenVector = eigenSolver.eigenvectors();

    double min_eigenvalue = std::numeric_limits<double>::max();
    int min_idx = -1;
    for (unsigned int i = 0; i < 3; i++) {
        if (min_eigenvalue > eigenValues(i)) {
            min_eigenvalue = eigenValues(i);
            min_idx = i;
        }
    }

    Eigen::Vector3d vertical_vector = eigenVector.block<3, 1>(0, min_idx);
    std::cout << "gravity direction: " << vertical_vector[0] << " " << vertical_vector[1] << " " << vertical_vector[2]
              << std::endl;

    int positive_vote = 0;
    int negative_vote = 0;
    for (size_t i = 0; i < gps_locations_horizontal.size(); ++i) {
        Eigen::Vector3d up = gps_locations[i].second.first - gps_locations_horizontal[i].second.first;

        if (up.dot(vertical_vector) >= 0) {
            positive_vote++;
        } else {
            negative_vote++;
        }
    }
    if (negative_vote > positive_vote) {
        vertical_vector = -vertical_vector;
    }

    Eigen::Vector3d x_vector;
    for (unsigned int i = 0; i < 3; i++) {
        if (i != min_idx) {
            x_vector = eigenVector.block<3, 1>(0, i);
            break;
        }
    }

    //want z axis to point down

    vertical_vector = -vertical_vector; 

    Eigen::Vector3d y_vector = vertical_vector.cross(x_vector);

    Eigen::Matrix3d R;
    R.block<3, 1>(0, 0) = x_vector;
    R.block<3, 1>(0, 1) = y_vector;
    R.block<3, 1>(0, 2) = vertical_vector;
    R.transposeInPlace();

    // Rotate the gps locations to align with gravity direction
    for (auto& location : gps_locations) {
        location.second.first = R * location.second.first;
    }

    for(size_t i = 0; i<3; ++i){
        for(size_t j = 0; j<3; ++j){
            gps_trans_file<<R(i,j)<<" ";    
        }
        gps_trans_file<<std::endl;
    }
    gps_trans_file.close();

    std::stable_sort(gps_locations.begin(), gps_locations.end(), TimeCompare);

    if(abuse_altitude){
        for (auto& horizontal_location : gps_locations_horizontal) {
            horizontal_location.second.first = R * horizontal_location.second.first;
        }

        gps_locations = gps_locations_horizontal;
        std::stable_sort(gps_locations.begin(), gps_locations.end(), TimeCompare);  
    }


    return true;
}

bool GPSLocationsToImages(std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d, int>>>& gps_locations,
                          const std::vector<std::string>& image_names,
                          std::unordered_map<std::string, std::pair<Eigen::Vector3d, int>>& image_locations,
                          const long time_offset, const long max_gps_image_time_differ) {
    image_locations.clear();
    std::unordered_map<unsigned long, std::string> image_name_time_map;
    std::vector<unsigned long> image_time_stamps;
    std::unordered_map<unsigned long, std::pair<Eigen::Vector3d, int>> image_time_locations;
    std::unordered_map<unsigned long, double> image_time_differs;

    for (const auto& image_name : image_names) {
        size_t pos1 = image_name.find(".jpg");
        size_t pos2 = image_name.rfind("/");
        std::string image_time_str;
        if (pos2 != std::string::npos) {
            image_time_str = image_name.substr(pos2 + 1, pos1 - pos2 - 1);
        } else {
            image_time_str = image_name.substr(0, pos1);
        }

        unsigned long image_time = std::stol(image_time_str) + time_offset;

        image_name_time_map.emplace(image_time, image_name);
        image_time_stamps.push_back(image_time);
    }
    //std::cout << "max time differ: " << max_gps_image_time_differ << std::endl;
    std::stable_sort(image_time_stamps.begin(), image_time_stamps.end());

    unsigned long image_time, gps_time;
    double current_time_differ;

    size_t gps_idx = 0;
    size_t image_idx = 0;

    while (gps_idx < gps_locations.size() && image_idx < image_time_stamps.size()) {
        image_time = image_time_stamps[image_idx];
        gps_time = gps_locations[gps_idx].first;


        double current_time_differ = fabs(static_cast<double>(image_time) - static_cast<double>(gps_time));
        if (current_time_differ < max_gps_image_time_differ) {
            if (image_time_locations.find(image_time) == image_time_locations.end()) {
                image_time_locations.emplace(image_time, gps_locations[gps_idx].second);
                image_time_differs.emplace(image_time, current_time_differ);
            } else {
                CHECK(image_time_differs.find(image_time) != image_time_differs.end());
                if (image_time_differs.at(image_time) > current_time_differ) {
                    image_time_locations.at(image_time) = gps_locations[gps_idx].second;
                    image_time_differs.at(image_time) = current_time_differ;
                }
            }
        }

        if (gps_time > image_time && gps_idx > 0) {
            unsigned long previous_gps_time = gps_locations[gps_idx - 1].first;
            double previous_time_differ =
                fabs(static_cast<double>(image_time) - static_cast<double>(previous_gps_time));

            if (previous_time_differ < max_gps_image_time_differ) {
                if (image_time_locations.find(image_time) == image_time_locations.end()) {
                    image_time_locations.emplace(image_time, gps_locations[gps_idx - 1].second);
                    image_time_differs.emplace(image_time, previous_time_differ);
                } else {
                    CHECK(image_time_differs.find(image_time) != image_time_differs.end());
                    if (image_time_differs.at(image_time) > previous_time_differ) {
                        image_time_locations.at(image_time) = gps_locations[gps_idx - 1].second;
                        image_time_differs.at(image_time) = previous_time_differ;
                    }
                }
            }
        }

        if (gps_time <= image_time) {
            gps_idx++;
        } else {
            image_idx++;
        }
    }

    for (auto time_location : image_time_locations) {
        CHECK(image_name_time_map.find(time_location.first) != image_name_time_map.end());
        std::string image_name = image_name_time_map.at(time_location.first);
        image_locations.emplace(image_name, time_location.second);
    }
    return true;
}

bool LoadGpsOrigin(const std::string& gps_origin_path, std::vector<double>& vec_gps){
    vec_gps.clear();
    std::ifstream file(gps_origin_path);
    CHECK(file.is_open()) << gps_origin_path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);
        while (!line_stream.eof()) {
            std::getline(line_stream, item, ' ');
            vec_gps.push_back(std::stod(item));
        }
    }

    file.close();
    return (vec_gps.size() == 3);
};

}  // namespace sensemap
