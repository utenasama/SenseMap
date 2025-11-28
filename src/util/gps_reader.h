//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_SRC_UTIL_GPS_READER_H_
#define SENSEMAP_SRC_UTIL_GPS_READER_H_

#include <unordered_map>
#include <Eigen/Core>
#include "util/string.h"
#include "util/types.h"
#include "util/misc.h"

namespace sensemap{

class GPSReader {
public:
    bool Load(const std::string& file_path, bool abuse_altitude = false, int accuracy_level_threshold = 1);
    inline std::unordered_map<unsigned long, std::pair<Eigen::Vector3d,int>> GetTimeLocations();
    static Eigen::Vector3d gpsToLocation(double latitude, double longitude, double altitude);
    static void LocationToGps(const Eigen::Vector3d X, double *latitude, double *longitude, double *altitude);
    static Eigen::Vector3d gpsToUTM(double latitude, double longtitude);

private:
    std::unordered_map<unsigned long, std::pair<Eigen::Vector3d,int>> time_locations_;
    static const int R_ = 6378137;                   // earth radius
    static constexpr double f_inv_ = 298.257222101;  // 1/oblateness
    static constexpr double f_ = 0.003352811;
};

class GeodeticConverter {
public:
    GeodeticConverter(double latitude = 0.0, double longitude = 0.0, double altitude = 0.0);
    void LLAToNed(double latitude, double longitude, double altitude,
                  double *n, double *e, double *d);
    void LLAToEcef(double latitude, double longitude, double altitude,
                  double *x, double *y, double *z);
    void EcefToNed(double x, double y, double z, double *n, double *e, double *d);

    Eigen::Matrix3x4d NedToEcefMatrix();
    Eigen::Matrix3x4d EcefToNedMatrix();
private:
    Eigen::Vector3d ecef_origin_ref_;
    Eigen::Matrix3d ned_to_ecef_;
};

std::unordered_map<unsigned long, std::pair<Eigen::Vector3d,int>> GPSReader::GetTimeLocations() { return time_locations_; }

// load gps info
bool LoadOriginGPSinfo(const std::string& file_path,
                       std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>>& gps_locations,
                       const std::string & gps_trans_path, 
                       bool abuse_altitude = false,
                       int accuracy_level_threshold = 1);

bool GPSLocationsToImages(std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>>& gps_locations,
                          const std::vector<std::string>& image_names,
                          std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>>& image_locations,
                          const long time_offset = 0,
                          const long max_gps_image_time_differ = 300);

bool LoadGpsOrigin(const std::string& gps_origin_path, std::vector<double>& vec_gps);

}  // namespace sensemap

#endif