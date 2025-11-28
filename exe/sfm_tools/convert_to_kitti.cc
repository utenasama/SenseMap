#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <Eigen/Dense>

#include "util/types.h"
#include "util/string.h"
#include "base/image.h"
#include "base/camera.h"

using namespace sensemap;

void ReadImagesText(const std::string& path,
                    std::map<image_t, class Image>& images) {
    images.clear();

    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)){
        sensemap::StringTrim(&line);

        if (line.empty() || line[0] == '#'){
            continue;
        }

        std::stringstream line_stream1(line);

        // ID
        std::getline(line_stream1, item, ' ');
        const image_t image_id = std::stoul(item);

        class Image image;
        image.SetImageId(image_id);

        image.SetRegistered(true);

        // QVEC (qw, qx, qy, qz)
        std::getline(line_stream1, item, ' ');
        image.Qvec(0) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(1) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(2) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(3) = std::stold(item);

        image.NormalizeQvec();

        // TVEC
        std::getline(line_stream1, item, ' ');
        image.Tvec(0) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Tvec(1) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Tvec(2) = std::stold(item);

        // CAMERA_ID
        std::getline(line_stream1, item, ' ');
        image.SetCameraId(std::stoul(item));

        // NAME
        std::getline(line_stream1, item, ' ');
        image.SetName(item);

        // POINTS2D
        if (!std::getline(file, line)){
            break;
        }

        StringTrim(&line);
        std::stringstream line_stream2(line);

        std::vector<Eigen::Vector2d> points2D;
        std::vector<mappoint_t> point3D_ids;

        if (!line.empty()){
            while (!line_stream2.eof()){
                Eigen::Vector2d point;

                std::getline(line_stream2, item, ' ');
                point.x() = std::stold(item);

                std::getline(line_stream2, item, ' ');
                point.y() = std::stold(item);

                points2D.push_back(point);

                std::getline(line_stream2, item, ' ');
                if (item == "-1"){
                    point3D_ids.push_back(kInvalidMapPointId);
                }
                else{
                    point3D_ids.push_back(std::stoll(item));
                }
            }
        }

        // image.SetUp(Camera(image.CameraId()));
        // image.SetPoints2D(points2D);

        // for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
        //      ++point2D_idx){
        //     if (point3D_ids[point2D_idx] != kInvalidMapPointId){
        //         image.SetMapPointForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
        //     }
        // }

        images.emplace(image.ImageId(), image);
    }
}

int main(int argc, char* argv[]) {

    std::string pose_dir(argv[1]);

    std::map<image_t, class Image> images;
    ReadImagesText(pose_dir, images);

    FILE *fp = fopen((pose_dir + "/pose.txt").c_str(), "w");
    std::map<image_t, class Image>::iterator it = images.begin();
    for (; it != images.end(); ++it) {
        Eigen::Matrix3x4d proj_matrix = it->second.ProjectionMatrix();
        fprintf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
            proj_matrix(0, 0), proj_matrix(0, 1), proj_matrix(0, 2), proj_matrix(0, 3),
            proj_matrix(1, 0), proj_matrix(1, 1), proj_matrix(1, 2), proj_matrix(1, 3),
            proj_matrix(2, 0), proj_matrix(2, 1), proj_matrix(2, 2), proj_matrix(2, 3)
        );
    }

    fclose(fp);

    return 0;
}

