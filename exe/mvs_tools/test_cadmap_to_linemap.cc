//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "util/bitmap.h"
#include "base/common.h"
#include "mvs/workspace.h"

#include "base/version.h"
#include "../Configurator_yaml.h"


using namespace sensemap;

std::string configuration_file_path;

// Reference paper: 
// [1] Wall Extraction and Room Detection for Multi-Unit Architectural Floor Plans
//     (http://dspace.library.uvic.ca/bitstream/handle/1828/10111/Cabrera-Vargas_Dany_MSc_2018.pdf)
// [2] 多分辨率线段提取方法及线段语义分析


class Line{

public:
    float k;
    float b;
    float angle; // range in [0, PAI]
    int x0, y0, x1, y1;
    bool valid;
    float min_height, max_height;

    Line(float _k, float _b, int _x0, int _y0, int _x1, int _y1, float _min_height, float _max_height):
            k(_k), b(_b), x0(_x0), y0(_y0), x1(_x1), y1(_y1), min_height(_min_height), max_height(_max_height){
        angle = atan(k);
        if(angle < 0)
            angle += M_PI;
        valid = true;
    }

    float length() const {
        return sqrt((y1 - y0) * (y1 - y0) + (x1 - x0) * (x1 - x0));
    }
};


std::vector<std::vector<float> > height_max;
std::vector<std::vector<float> > height_min;
std::vector<std::vector<bool> >  height_valid;

template <typename T>
T CalAverage(const std::vector<T> &vec){
    
    if(vec.empty())
        return 0;

    float sum = 0;
    for(auto &t: vec){
        sum += t;
    }
    return sum / vec.size();
}

template <typename T>
bool ConditionalBlur(cv::Mat &angle_map){
    
    const int height = angle_map.rows;
    const int width = angle_map.cols;
    
    cv::Mat blur_map = cv::Mat::zeros(height, width, angle_map.type());
    std::vector<std::vector<int> > directions = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, 
                                                 {0, 1}, {-1, -1}, {-1, 0}, {-1, 1},
                                                 {-2, -2}, {-2, -1}, {-2, 0}, {-2, 1}, {-2, 2},
                                                 {-1, -2}, {-1, 2}, {0, -2}, {0, 2}, {1, -2}, {1, 2},
                                                 {2, -2}, {2, -1}, {2, 0}, {2, 1}, {2, 2}
                                                 };

    std::vector<T> similars;
    std::vector<T> differences;
    for(size_t y = 0; y < height; y++){
        for(size_t x = 0; x < width; x++){
            if(angle_map.at<T>(y, x) == -1){
                blur_map.at<T>(y, x) = -1;
                continue;
            }

            similars.clear();
            differences.clear();
            for(auto &dir: directions){
                int ty = y + dir[0];
                int tx = x + dir[1];
                if (ty < 0 || ty >= height || tx < 0 || tx >= width || angle_map.at<T>(ty, tx) == -1)
                    continue;
                
                if(fabs(angle_map.at<T>(y, x) - angle_map.at<T>(ty, tx)) < M_PI / 6){
                    similars.push_back(angle_map.at<T>(ty, tx));
                }else{
                    differences.push_back(angle_map.at<T>(ty, tx));
                }

                if(similars.size() > differences.size()){
                    similars.push_back(angle_map.at<T>(y, x));
                    blur_map.at<T>(y, x) = CalAverage<T>(similars);
                }else if(similars.size() < differences.size()){
                    blur_map.at<T>(y, x) = CalAverage<T>(differences);
                }else{
                    blur_map.at<T>(y, x) = angle_map.at<T>(y, x);
                }
            }
        }
    }
    blur_map.copyTo(angle_map);

    return true;
}


bool SliceRun(const cv::Mat &image, cv::Mat &slice_map, int x, int y, int dx, int dy){

    int height = image.rows;
    int width = image.cols;

    bool in_slice = false;
    int count = 0;
    while (0 <= x && x < width && 0 <= y && y < height){
        if (image.at<uint8_t>(y, x) != 0){
            count++;
            in_slice = true;
        }else if(in_slice){
            in_slice = false;
            int tx = x, ty = y;
            for(int step = 0; step < count; step++){
                ty -= dy;
                tx -= dx;
                slice_map.at<uint8_t>(ty, tx) = count;
            }
            count = 0;
        }
        x += dx;
        y += dy;
    }

    if(in_slice){
        int tx = x, ty = y;
        for(int step = 0; step < count; step++){
            ty -= dy;
            tx -= dx;
            slice_map.at<uint8_t>(ty, tx) = count;
        }
    }

    return true;
}

bool GetSliceMap(cv::Mat image, std::vector<cv::Mat> &slice_maps){
    
    slice_maps.clear();
    
    int height = image.rows;
    int width = image.cols;

    cv::Mat slice_map = cv::Mat::zeros(height, width, CV_8UC1);
    for(int y = 0; y < height; y++)
        SliceRun(image, slice_map, 0, y, 1, 0);
    // cv::imwrite("./slicemap0.jpg", slice_map);
    // ConditionalBlur<uint8_t>(slice_map);
    // cv::imwrite("./slicemap0_blur.jpg", slice_map);
    slice_maps.push_back(slice_map.clone());

    slice_map = cv::Mat::zeros(height, width, CV_8UC1);
    for(int x = 0; x < width; x++)
        SliceRun(image, slice_map, x, 0, 0, 1);
    // cv::imwrite("./slicemap1.jpg", slice_map);
    // ConditionalBlur<uint8_t>(slice_map);
    // cv::imwrite("./slicemap1_blur.jpg", slice_map);
    slice_maps.push_back(slice_map.clone());
    
    slice_map = cv::Mat::zeros(height, width, CV_8UC1);
    for(int y = 0; y < height; y++)
        SliceRun(image, slice_map, 0, y, 1, -1);
    for(int x = 1; x < width; x++)
        SliceRun(image, slice_map, x, height - 1, 1, -1);
    // cv::imwrite("./slicemap2.jpg", slice_map);
    // ConditionalBlur<uint8_t>(slice_map);
    // cv::imwrite("./slicemap2_blur.jpg", slice_map);
    slice_maps.push_back(slice_map.clone());

    slice_map = cv::Mat::zeros(height, width, CV_8UC1);
    for(int y = 0; y < height; y++)
        SliceRun(image, slice_map, 0, y, 1, 1);
    for(int x = 1; x < width; x++)
        SliceRun(image, slice_map, x, 0, 1, 1);
    // cv::imwrite("./slicemap3.jpg", slice_map);
    // ConditionalBlur<uint8_t>(slice_map);
    // cv::imwrite("./slicemap3_blur.jpg", slice_map);
    slice_maps.push_back(slice_map.clone());

    return true;
}

bool GetAngleMap(const cv::Mat &image, const std::vector<cv::Mat> &slice_maps, cv::Mat &angle_map){
    
    const int height = image.rows;
    const int width = image.cols;
    const int slice_count = slice_maps.size(); 
    std::vector<int> slices;

    angle_map = cv::Mat::zeros(height, width, CV_32FC1);
    for(size_t y = 0; y < height; y++){
        for(size_t x = 0; x < width; x++){
            slices.clear();
            for(size_t t = 0; t < slice_count; t++){
                if(slice_maps[t].at<uint8_t>(y, x) != 0)
                    slices.push_back(slice_maps[t].at<uint8_t>(y, x));
            }

            if(slices.size() != slice_count){
                angle_map.at<float>(y, x) = -1;
                continue;
            }
            
            float ph = slices[0], pv = slices[1], pd = slices[2], pe = slices[3];
            if(pd == pe && pd == ph || pd == pe && pv > ph){
                angle_map.at<float>(y, x) = M_PI_2;
            }else if(pd == pe && pd == pv || pd == pe && ph > pv){
                angle_map.at<float>(y, x) = 0;
            }else if(ph == pv && pe < 0.1 * pd){
                angle_map.at<float>(y, x) = M_PI_4;
            }else if(ph == pv && pd < 0.1 * pe){
                angle_map.at<float>(y, x) = M_PI_4 * 3;
            }else{
                
                float alpha0 = pd > pe ? atan(pv / ph) : -atan(pv / ph);
                float alpha1;
                if (ph > pv){
                    alpha1 = pd > pe ? atan(pd / pe) - M_PI_4 : atan(pd / pe) + M_PI_4 * 3;
                }else{
                    alpha1 = atan(pe / pd) + M_PI_4;
                }
                
                if (alpha0 < 0)
                    alpha0 += M_PI;
                if (alpha1 < 0)
                    alpha1 += M_PI;

                if(0.5 * fabs(ph - pv) > fabs(pd - pe)){
                    angle_map.at<float>(y, x) = alpha0;
                }else if (0.5 * fabs(pd - pe) > fabs(ph - pv)){
                    angle_map.at<float>(y, x) = alpha1;
                }else{
                    angle_map.at<float>(y, x) = (alpha0 + alpha1) / 2;
                }

                if (angle_map.at<float>(y, x) > M_PI / 8 * 7){
                        angle_map.at<float>(y, x) -= M_PI;
                }
            }
        }
    }

    return true;
}


bool GetSeperations(const cv::Mat &angle_map, std::vector<cv::Mat> &seperations){
    
    const int height = angle_map.rows;
    const int width = angle_map.cols;

    for(int i = 0; i < 4; i++){
        seperations.push_back(cv::Mat::zeros(height, width, CV_8UC1));
    }

    for(size_t y = 0; y < height; y++){
        for(size_t x = 0; x < width; x++){
            const float &angle = angle_map.at<float>(y, x);
            if(angle == -1)
                continue;

            if(angle < M_PI / 8 && angle > -M_PI / 8){
                seperations[0].at<uint8_t>(y, x) = 255;
            }else if(angle < M_PI / 8 * 3){
                seperations[1].at<uint8_t>(y, x) = 255;
            }else if(angle < M_PI / 8 * 5){
                seperations[2].at<uint8_t>(y, x) = 255;
            }else{
                seperations[3].at<uint8_t>(y, x) = 255;
            }
        }
    }
    return true;
}

Line FitLine(std::vector<cv::Point> &points){

    cv::Vec4f output;
    cv::fitLine(points, output, cv::DIST_L2, 0, 0.01, 0.01);

    float k = output[1] / output[0];
    float b = output[3] - k * output[2];
    float line_height_max = FLT_MIN;
    float line_height_min = FLT_MAX; 
    bool line_height_found = false;

    int x_min = INT_MAX, x_max = INT_MIN, y_min = INT_MAX, y_max = INT_MIN;
    for(auto &p: points){
        x_min = std::min(x_min, p.x);
        x_max = std::max(x_max, p.x);
        y_min = std::min(y_min, p.y);
        y_max = std::max(y_max, p.y);
        if(!height_valid.empty() && height_valid[p.y][p.x]){
            line_height_found = true;
            line_height_max = std::max(line_height_max, height_max[p.y][p.x]);
            line_height_min = std::min(line_height_min, height_min[p.y][p.x]);
        }
    }

    int x0, y0, x1, y1;
    if(fabs(k) > 1){
        y0 = y_min;
        y1 = y_max;
        x0 = (y0 - b) / k;
        x1 = (y1 - b) / k; 
    }else{
        x0 = x_min;
        x1 = x_max;
        y0 = k * x0 + b;
        y1 = k * x1 + b;
    }

    Line l(k, b, x0, y0, x1, y1, line_height_min, line_height_max);
    if(!height_valid.empty())
        l.valid = line_height_found;
    return l;
}

bool ExtractLines(const std::vector<cv::Mat> &seperations, std::vector<Line> &lines, int max_grid_size){

    for(int i = 0; i < seperations.size(); i++){
        
        cv::Mat labels;
        int n_components = cv::connectedComponents(seperations[i], labels, 8, CV_32S);
        
        std::vector<int> count(n_components + 1, 0);
        std::vector<std::vector<cv::Point> > line_points(n_components + 1);
        for(int y = 0; y < labels.rows; y++){
            for(int x = 0; x < labels.cols; x++){
                int label = labels.at<unsigned int>(y, x);
                count[label] ++;
                line_points[label].push_back(cv::Point(x, y));
            }
        }

        for(size_t label = 0; label < line_points.size(); label++){
            auto &points = line_points[label];
            if (label == 0)
                continue;
            if (max_grid_size > 2000 && points.size() < 150)
                continue;
            if (max_grid_size <= 2000 && points.size() < 50)
                continue;
            Line line = FitLine(points);
            if(line.valid)
                lines.push_back(line);
        }
    }

    std::vector<Line> filtered_lines;
    for(auto &line: lines){
        if(line.valid){
            filtered_lines.emplace_back(line);
        }
    }
    lines.swap(filtered_lines);

    return true;
}

float PointDist(int x0, int y0, int x1, int y1){
    return sqrt((y1 - y0) * (y1 - y0) + (x1 - x0) * (x1 - x0));
}

bool line_compare(const Line &x, const Line &y){
    return x.length() > y.length();
}

void MergeLineCluster(std::vector<Line> &lines, std::set<int> &cluster, cv::Mat &debug_img, cv::Scalar &cluster_color){

    if(cluster.size() < 2)
        return;

    std::vector<int> idxs;
    idxs.assign(cluster.begin(), cluster.end());

    Eigen::Vector2f center;
    Eigen::Vector2f dir;
    float center_weight = 0;

    Eigen::Vector2f s1(lines[idxs[0]].x0, lines[idxs[0]].y0);
    Eigen::Vector2f e1(lines[idxs[0]].x1, lines[idxs[0]].y1);

    center = (s1 + e1) * lines[idxs[0]].length();
    center_weight += lines[idxs[0]].length() * 2;
    dir = (e1 - s1) * lines[idxs[0]].length();

    for(int i = 1; i < idxs.size(); i++){
        Eigen::Vector2f s(lines[idxs[i]].x0, lines[idxs[i]].y0);
        Eigen::Vector2f e(lines[idxs[i]].x1, lines[idxs[i]].y1);
        Eigen::Vector2f se = e - s;

        center += (s + e) * lines[idxs[i]].length();
        center_weight += lines[idxs[i]].length() * 2;

        if(dir.dot(se) < 0){
            dir -= se * lines[idxs[i]].length();
        }else{
            dir += se * lines[idxs[i]].length();;
        }
    }

    center /= center_weight;
    dir = dir.normalized();

    cv::circle(debug_img, cv::Point2f(center[0], center[1]), 6, cluster_color, -1);

    float t_max = FLT_MIN;
    float t_min = FLT_MAX;

    for(int i = 0; i < idxs.size(); i++){
        Eigen::Vector2f s(lines[idxs[i]].x0, lines[idxs[i]].y0);
        Eigen::Vector2f e(lines[idxs[i]].x1, lines[idxs[i]].y1);

        float t_s = (s - center).dot(dir);
        float t_e = (e - center).dot(dir);
        t_min = std::min(t_min, t_s);
        t_min = std::min(t_min, t_e);
        t_max = std::max(t_max, t_s);
        t_max = std::max(t_max, t_e);
    }

    
    Eigen::Vector2f start = center + t_max * dir;
    Eigen::Vector2f end = center + t_min * dir;

    cv::line(debug_img, cv::Point(start[0], start[1]), cv::Point(end[0], end[1]), cv::Scalar(255,255,255), 1);

    float k = (end[1] - start[1]) / (end[0] - start[0]);
    float b = end[1] - k * end[0];

    float line_height_max = FLT_MIN;
    float line_height_min = FLT_MAX;
    for(int i = 0; i < idxs.size(); i++){
        line_height_min = std::min(line_height_min, lines[idxs[i]].min_height);
        line_height_max = std::max(line_height_max, lines[idxs[i]].max_height);
    }

    lines[idxs[0]] = Line(k, b, start[0], start[1], end[0], end[1], line_height_min, line_height_max);
    for(int i = 1; i < idxs.size(); i++){
        lines[idxs[i]].valid = false;
    }
}

bool MergeLines(std::vector<Line> &lines, int max_grid_size){
    
    // std::ofstream fs("line_dist.txt");

    int theta_thres, vertical_thres, horizon_thres;

    if (max_grid_size > 2000){
        theta_thres = 30;
        vertical_thres = 30;
        horizon_thres = 50;
    }else{
        theta_thres = 20;
        vertical_thres = 10;
        horizon_thres = 20;
    }


    std::sort(lines.begin(), lines.end(), line_compare);

    std::vector<std::set<int> > clusters;
    for(int i = 0; i < lines.size(); i++){
        if(lines[i].valid){
            clusters.push_back(std::set<int>{i});
        }
    }

    for(size_t i = 0; i < lines.size(); i++){
        for(size_t j = 0; j < lines.size(); j++){
            if(i != j && lines[i].valid && lines[j].valid){

                Line line0 = lines[i];
                Line line1 = lines[j];
                if (lines[i].length() < lines[j].length()){
                    line0 = lines[j];
                    line1 = lines[i];
                }

                float theta = fabs(line0.angle - line1.angle);
                if(theta > M_PI / 2)
                    theta = M_PI - theta;
                float dist_theta = line0.length() * sin(theta);

                float l0 = fabs(line0.k * line1.x0 - line1.y0 + line0.b) / sqrt(line0.k * line0.k + 1);
                float l1 = fabs(line0.k * line1.x1 - line1.y1 + line0.b) / sqrt(line0.k * line0.k + 1);
                float dist_ver = (l0 * l0 + l1 * l1) / (l0 + l1); // Lehmer Mean


                float x0 = (line0.k * line0.x0 + line1.x0 / line0.k + line1.y0 - line0.y0 ) / (1 / line0.k + line0.k);
                float y0 = -1 / line0.k * (x0 - line1.x0) + line1.y0;
                float x1 = (line0.k * line0.x0 + line1.x1 / line0.k + line1.y1 - line0.y0 ) / (1 / line0.k + line0.k);
                float y1 = -1 / line0.k * (x1 - line1.x1) + line1.y1;

                float dist_hor = 0;
                if((line0.x0 <= x0 && x0 <= line0.x1) || (line0.x0 >= x0 && x0 >= line0.x1) ||
                   (line0.x0 <= x1 && x1 <= line0.x1) || (line0.x0 >= x1 && x1 >= line0.x1)
                ){
                }else{
                    dist_hor = PointDist(x0, y0, line0.x0, line0.y0);
                    dist_hor = std::min(dist_hor, PointDist(x0, y0, line0.x1, line0.y1));
                    dist_hor = std::min(dist_hor, PointDist(x1, y1, line0.x0, line0.y0));
                    dist_hor = std::min(dist_hor, PointDist(x1, y1, line0.x1, line0.y1));
                }

                // fs << "L" << i << " vs L" << j << ": " << dist_theta << " " << dist_ver << " " << dist_hor << " " << std::endl;

                if(dist_theta < theta_thres && dist_ver < vertical_thres && dist_hor < horizon_thres){

                    int idx_i = -1;
                    int idx_j = -1;
                    for(int s_id = 0; s_id < clusters.size(); s_id++){
                        if(clusters[s_id].find(i) != clusters[s_id].end()){
                            idx_i = s_id;
                        }
                        if(clusters[s_id].find(j) != clusters[s_id].end()){
                            idx_j = s_id;
                        }
                        if(idx_i != -1 && idx_j != -1){
                            break;
                        }
                    }

                    if(idx_i != idx_j && idx_i != -1 && idx_j != -1){
                        clusters[idx_j].insert(clusters[idx_i].begin(), clusters[idx_i].end());
                        clusters[idx_i].clear();
                    }
                }
            }
        }
    }
    // fs.close();

    // remove empty clusters
    std::vector<std::set<int> > filtered_clusters;
    for(int i = 0; i < clusters.size(); i++){
        if(!clusters[i].empty()){
            filtered_clusters.emplace_back(clusters[i]);
        }
    }
    clusters.swap(filtered_clusters);

    cv::Mat cluster_img = cv::Mat::zeros(10000, 10000, CV_8UC3);
    for(int i = 0; i < clusters.size(); i++){
        
        cv::Scalar cluster_color(rand() % 256, rand() % 256, rand() % 256);
        for(auto it = clusters[i].begin(); it != clusters[i].end(); it++){
            cv::line(cluster_img, cv::Point(lines[*it].x0, lines[*it].y0), cv::Point(lines[*it].x1, lines[*it].y1), cluster_color, 4);
            cv::putText(cluster_img, std::to_string(i),
                        cv::Point2f((lines[*it].x0 + lines[*it].x1) / 2, (lines[*it].y0 + lines[*it].y1) / 2),
                        cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 255));
        }
        MergeLineCluster(lines, clusters[i], cluster_img, cluster_color);
    }
    // cv::imwrite("cluster_img.jpg", cluster_img);
    
    std::vector<Line> filtered_lines;
    for(auto &line: lines){
        if(line.valid){
            filtered_lines.emplace_back(line);
        }
    }
    lines.swap(filtered_lines);

    return true;
}


bool RemoveShortLines(std::vector<Line> &lines, int length_thres){
    
    std::vector<Line> filtered_lines;

    for(auto &line: lines){
        if(line.valid && line.length() > length_thres){
            filtered_lines.emplace_back(line);
        }
    }
    lines.swap(filtered_lines);

    return true;
}

Eigen::Vector4d CalcPlaneEquation(Eigen::Vector3d &X0,
                                  Eigen::Vector3d &X1, 
                                  Eigen::Vector3d &X2){
    
    Eigen::Vector4d plane;
    plane[0] = (X1[1] - X0[1]) * (X2[2] - X0[2]) - (X1[2] - X0[2]) * (X2[1] - X0[1]);
    plane[1] = (X2[0] - X0[0]) * (X1[2] - X0[2]) - (X1[0] - X0[0]) * (X2[2] - X0[2]);
    plane[2] = (X1[0] - X0[0]) * (X2[1] - X0[1]) - (X2[0] - X0[0]) * (X1[1] - X0[1]);
    plane[3] = -(plane[0] * X0[0] + plane[1] * X0[1] + plane[2] * X0[2]);
    return plane;
}

bool GenerateWallsFromLines(const std::string &align_param_path, 
                            const std::string &multiplane_path,
                            const std::string &trans_path,
                            const std::vector<Line> &lines,
                            const bool verbose,
                            const int layer_id){


    
    // Read align params
    Eigen::Matrix3d T, M;
    Eigen::Vector3d TT, MT;

    FILE *fp = fopen(align_param_path.c_str(), "r");

    if(fp == NULL){
        std::cout << "[ERROR] File not exists:" << align_param_path << std::endl;
        return false;
    }


    char buf[1000];

    for(int i = 0; i < 4; i++){
        fgets(buf, 1000, fp);
    }

    fscanf(fp,"%lf %lf %lf %lf\n", &(T(0, 0)), &(T(0, 1)), &(T(0, 2)), &(TT[0]));
    fscanf(fp,"%lf %lf %lf %lf\n", &(T(1, 0)), &(T(1, 1)), &(T(1, 2)), &(TT[1]));
    fscanf(fp,"%lf %lf %lf %lf\n", &(T(2, 0)), &(T(2, 1)), &(T(2, 2)), &(TT[2]));
    
    for(int i = 0; i < 2; i++){
        fgets(buf, 1000, fp);
    }

    fscanf(fp,"%lf %lf %lf %lf\n", &(M(0, 0)), &(M(0, 1)), &(M(0, 2)), &(MT[0]));
    fscanf(fp,"%lf %lf %lf %lf\n", &(M(1, 0)), &(M(1, 1)), &(M(1, 2)), &(MT[1]));
    fscanf(fp,"%lf %lf %lf %lf\n", &(M(2, 0)), &(M(2, 1)), &(M(2, 2)), &(MT[2]));

    fclose(fp);


    Eigen::RowMatrix3x4d trans;
    bool has_trans = ExistsFile(trans_path);
    std::cout << trans_path << std::endl;
    if (has_trans) {
        // Load transform matrix
        cv::FileStorage fs;
        fs.open(trans_path, cv::FileStorage::READ);
        cv::Mat trans_mat;
        if(fs["transMatrix"].type() != cv::FileNode::MAP){
            std::cout << "ERROR: Input yaml error !!" << std::endl;
            exit(-1);
        }
        fs["transMatrix"] >> trans_mat;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                trans(i, j) = trans_mat.at<double>(i, j);
            }
        }   

        std::cout << "trans: " << std::endl << trans << std::endl;
    }


    int mesh_num = 1;
    int g_poly_id = 0;

    FILE *obj_file, *txt_file;
    
    if(verbose){
        obj_file = fopen(JoinPaths(multiplane_path, "walls.obj").c_str(), "w");
    }
    txt_file = fopen(JoinPaths(multiplane_path, "wall_polys.txt").c_str(), "w");
    fprintf(txt_file, "\n");
    fprintf(txt_file, "%d\n", lines.size());

    for(int i = 0; i < lines.size(); i++){

        Eigen::Vector3d proj0(lines[i].x0, lines[i].y0, 1);
        Eigen::Vector3d proj1(lines[i].x1, lines[i].y1, 1);

        Eigen::Vector3d P0 = M.inverse() * (proj0 - MT);
        Eigen::Vector3d P1 = M.inverse() * (proj1 - MT);

        P0[2] = lines[i].min_height;
        Eigen::Vector3d X0 = T.inverse() * (P0 - TT);
        P0[2] = lines[i].max_height;
        Eigen::Vector3d X3 = T.inverse() * (P0 - TT);

        P1[2] = lines[i].min_height;
        Eigen::Vector3d X1 = T.inverse() * (P1 - TT);
        P1[2] = lines[i].max_height;
        Eigen::Vector3d X2 = T.inverse() * (P1 - TT);

        if(has_trans){
            X0 = trans * X0.homogeneous();
            X1 = trans * X1.homogeneous();
            X2 = trans * X2.homogeneous();
            X3 = trans * X3.homogeneous();
        }


        Eigen::Vector4d plane = CalcPlaneEquation(X0, X1, X2);
        fprintf(txt_file, "# plane parameter.\n");
        fprintf(txt_file, "%f %f %f %f\n", plane.x(), plane.y(), plane.z(), plane.w());
        fprintf(txt_file, "\n# count of outer polygon.\n");
        fprintf(txt_file, "1\n");
        fprintf(txt_file, "\n# polygon id.\n");
        fprintf(txt_file, "ID: %d\n", g_poly_id++);
        fprintf(txt_file, "# polygon vertex list(x y z list).\n");
        fprintf(txt_file, "4\n");
        fprintf(txt_file, "%f %f %f\n", X0[0], X0[1], X0[2]);
        fprintf(txt_file, "%f %f %f\n", X1[0], X1[1], X1[2]);
        fprintf(txt_file, "%f %f %f\n", X2[0], X2[1], X2[2]);
        fprintf(txt_file, "%f %f %f\n", X3[0], X3[1], X3[2]);
        fprintf(txt_file, "\n");
        fprintf(txt_file, "# count of inner polygon.\n");
        fprintf(txt_file, "0\n");


        if(verbose){
            fprintf(obj_file, "v %f %f %f\n", X0[0], X0[1], X0[2]);
            fprintf(obj_file, "v %f %f %f\n", X1[0], X1[1], X1[2]);
            fprintf(obj_file, "v %f %f %f\n", X2[0], X2[1], X2[2]);
            fprintf(obj_file, "v %f %f %f\n", X3[0], X3[1], X3[2]);
            fprintf(obj_file, "f %d %d %d\n", mesh_num, mesh_num + 1, mesh_num + 2);
            fprintf(obj_file, "f %d %d %d\n", mesh_num, mesh_num + 2, mesh_num + 3);
            mesh_num += 4;
        }

        
    }
    if(verbose){
        fclose(obj_file);
    }
    fclose(txt_file);
    
    return true;
}

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    int max_grid_size = param.GetArgument("max_cad_grid", 1000);
    bool output_ground = param.GetArgument("cad_output_ground", 0);
    bool output_wall = param.GetArgument("cad_output_wall", 0);
    std::string trans_path = param.GetArgument("trans_path", "");
    bool verbose = param.GetArgument("verbose", 0);

    int num_reconstruction = 0;
    for (size_t reconstruction_idx = 0; ; reconstruction_idx++) {
        auto reconstruction_path = JoinPaths(workspace_path, std::to_string(reconstruction_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
        if (!ExistsDir(dense_reconstruction_path)) {
            break;
        }
        
        int layer_id = 0;
        while(true){

            // input
            auto mesh_map_path = StringPrintf("%s/mesh_map%04d.jpg",
                                        dense_reconstruction_path.c_str(), layer_id);
            auto point_map_path = StringPrintf("%s/cadmap%04d.jpg",
                                        dense_reconstruction_path.c_str(), layer_id);
            auto align_param_path = StringPrintf("%s/align_param.txt",
                                        dense_reconstruction_path.c_str(), layer_id);
            auto height_range_path = StringPrintf("%s/heigh_range%04d.xml",
                                        dense_reconstruction_path.c_str(), layer_id);            

            // output
            const auto line_map_path = StringPrintf("%s/linemap%04d.jpg",
                                        dense_reconstruction_path.c_str(), layer_id);
            const auto line_file_path = StringPrintf("%s/lines%04d.txt",
                                        dense_reconstruction_path.c_str(), layer_id);
            // // input
            // const auto mesh_map_path = JoinPaths(dense_reconstruction_path, "meshmap.jpg");
            // const auto point_map_path = JoinPaths(dense_reconstruction_path, "cadmap.jpg");
            // const auto align_param_path = JoinPaths(dense_reconstruction_path, "align_param.txt");
            // const auto height_range_path = JoinPaths(dense_reconstruction_path, "height_range.xml");
            
            // // output
            // const auto line_map_path = JoinPaths(dense_reconstruction_path, "linemap.jpg");
            // const auto line_file_path = JoinPaths(dense_reconstruction_path, "lines.txt");

            if(!ExistsFile(point_map_path) || !ExistsFile(mesh_map_path)){
                break;
            }

            if(output_wall){

                if(!ExistsFile(height_range_path)){
                    std::cout << "[ERROR] Output_wall enabled but height range file " << height_range_path << " not exists, set cad_output_height_range:1 and re-run test_dense_to_cadmap." << std::endl;
                    output_wall = false;
                }else{

                    height_max = std::vector<std::vector<float> >(max_grid_size, std::vector<float>(max_grid_size, 0));
                    height_min = std::vector<std::vector<float> >(max_grid_size, std::vector<float>(max_grid_size, 0));
                    height_valid = std::vector<std::vector<bool> >(max_grid_size, std::vector<bool>(max_grid_size, false));

                    cv::Mat height_range(max_grid_size, max_grid_size, CV_32FC2);
                    cv::FileStorage fs1(height_range_path, cv::FileStorage::READ);
                    fs1["height_range"] >> height_range;
                    fs1.release();

                    for(int y = 0; y < max_grid_size; y++){
                        for(int x = 0; x < max_grid_size; x++){
                            if(height_range.at<cv::Vec2f>(y, x)[0] < 1000 && height_range.at<cv::Vec2f>(y, x)[0] > -1000){
                                height_min[y][x] = height_range.at<cv::Vec2f>(y, x)[0];
                                height_max[y][x] = height_range.at<cv::Vec2f>(y, x)[1];
                                height_valid[y][x] = true;
                            }
                        }
                    }
                }
            }


            cv::Mat point_map = cv::imread(point_map_path, 0);

            cv::Mat mesh_map = cv::Mat::zeros(point_map.rows, point_map.cols, CV_8UC1);
            // if(ExistsFile(mesh_map_path)){
            //     mesh_map = cv::imread(mesh_map_path, 0);
            // }

            // merge point_map and mesh_map
            cv::Mat image = cv::Mat::zeros(point_map.rows, point_map.cols, CV_8UC1);
            for (int y = 0; y < point_map.rows; y++){
                for (int x = 0; x < point_map.cols; x++){
                    if(point_map.at<uint8_t>(y, x) != 255 || mesh_map.at<uint8_t>(y, x) >= 5){
                        image.at<uint8_t>(y, x) = 255;;
                    }
                }
            }

            // remove noise points
            if(max_grid_size > 2000){
                auto kernel = cv::getStructuringElement(0, cv::Size(3, 3));
                cv::Mat tophat_img;
                cv::morphologyEx(image, tophat_img, 5, kernel);
                cv::bitwise_xor(image, tophat_img, image);
                cv::dilate(image, image, kernel, cv::Point(-1, -1), 5);
            }

            std::cout << "==> GetSliceMap" << std::endl;
            std::vector<cv::Mat> slice_maps;
            GetSliceMap(image, slice_maps);

            std::cout << "==> GetAngleMap" << std::endl;
            cv::Mat angle_map;
            GetAngleMap(image, slice_maps, angle_map);

            std::cout << "==> ConditionalBlur" << std::endl;
            ConditionalBlur<float>(angle_map);

            std::cout << "==> GetSeperations" << std::endl;
            std::vector<cv::Mat> seperations;
            GetSeperations(angle_map, seperations);

            std::cout << "==> ExtractLines" << std::endl;
            std::vector<Line> lines;
            ExtractLines(seperations, lines, max_grid_size);

            std::cout << lines.size() << " lines extracted." << std::endl;
            std::cout << "==> MergeLines" << std::endl;
            MergeLines(lines, max_grid_size);

            std::cout << "==> Remove short lines" << std::endl;
            if(max_grid_size > 2000){
                RemoveShortLines(lines, 50);
            }

            std::cout << lines.size() << " lines righted." << std::endl;
            std::cout << "==> Output results" << std::endl;

            cv::Mat line_map = cv::Mat::zeros(point_map.rows, point_map.cols, CV_8UC1);
            for(int i = 0; i < lines.size(); i++){
                if(lines[i].valid){
                    cv::line(line_map, cv::Point(lines[i].x0, lines[i].y0), cv::Point(lines[i].x1, lines[i].y1), 255, 4);
                    // cv::putText(line_map, std::to_string(i),
                    //             cv::Point2f((lines[i].x0 + lines[i].x1) / 2, (lines[i].y0 + lines[i].y1) / 2),
                    //             cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }
            }
            cv::imwrite(line_map_path, line_map);


            // // output lines params
            // std::ofstream fs(line_file_path);
            // fs << "# [num] [x0] [y0] [x1] [y1]" << std::endl;
            // for(int i = 0; i < lines.size(); i++){
            //     if(lines[i].valid){
            //         fs << i << " " << lines[i].x0 << " " << lines[i].y0 << " " << lines[i].x1 << " " << lines[i].y1 << std::endl;
            //     }
            // }
            // fs.close();

            if(output_wall){
                std::cout << "==> Generate walls" << std::endl;
                std::string multiplane_path = JoinPaths(dense_reconstruction_path, "multi-planes");
                CreateDirIfNotExists(multiplane_path);
                GenerateWallsFromLines(align_param_path, multiplane_path, trans_path, lines, verbose, layer_id);
            }

            layer_id++;
        }
    }
}
