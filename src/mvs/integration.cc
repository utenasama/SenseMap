#include <iomanip>
#include <numeric>

#include "util/misc.h"
#include "util/threading.h"

#include "estimators/plane_estimator.h"
#include "optim/ransac/loransac.h"
#include "util/obj.h"
#include "util/rgbd_helper.h"

#include "mvs/utils.h"
#include "mvs/integration.h"
#include "mvs/integration/ScalableTSDFVolume.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <vector>
#include <fstream>
#include <boost/tuple/tuple.hpp>


namespace sensemap {
namespace mvs {

using namespace utility;

void StereoIntegration::Options::Print() const {
  PrintHeading2("StereoIntegration::Options");
  PrintOption(max_image_size);
  PrintOption(check_num_images);
  PrintOption(cache_size);
  PrintOption(max_spacing_factor);
}

bool StereoIntegration::Options::Check() const {
  CHECK_OPTION_GT(check_num_images, 0);
  CHECK_OPTION_GT(cache_size, 0);
  CHECK_OPTION_GE(max_spacing_factor, 0.0);
  return true;
}

StereoIntegration::StereoIntegration(const Options& options,
                           const std::string& workspace_path,
                           const std::string& input_type)
    : num_reconstruction_(0),
      options_(options),
      workspace_path_(workspace_path), 
      input_type_(input_type) {
  CHECK(options_.Check());
}

void StereoIntegration::ReadWorkspace() {
  num_reconstruction_ = 0;
  std::cout << "Reading workspace..." << std::endl;
    for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
        const auto& dense_reconstruction_path = 
            JoinPaths(reconstruction_path, DENSE_DIR);
        if (!ExistsDir(dense_reconstruction_path)) {
            break;
        }

        num_reconstruction_++;
    }
}

void StereoIntegration::AddBlackList(const std::vector<uint8_t>& black_list) {
  semantic_label_black_list_ = black_list;
}

void StereoIntegration::NoiseFilter(
  cv::Mat & mask,
  cv::Mat & depth,
  float depth_error_thresh,
  int statistic_count_thresh
) {
  mask.create(depth.rows, depth.cols, CV_32SC1);
  mask.setTo(0);

  for (int y = 0; y < depth.rows; y++) {
    for (int x = 0; x < depth.cols; x++) {
      float d = depth.at<float>(y, x);
      if (d <= 0) continue;

      double bad_count = 0;
      for (int dy = -2; dy <= 2; dy += 1) {
        const int y2 = y + dy;
        if (y2 < 0 || y2 >= depth.rows) continue;

        for (int dx = -2; dx <= 2; dx += 1) {
          const int x2 = x + dx;
          if (x2 < 0 || x2 >= depth.cols) continue;

          float d2 = depth.at<float>(y2, x2);
          if (d2 <= 0) {
            // bad_count += 0.5f;
          }
          else if (std::abs(d2 - d) > depth_error_thresh) {
            bad_count += 1.0f;
          }
        }
      }

      if (bad_count >= statistic_count_thresh) {
        mask.at<int>(y, x) = 1;
      }
    }
  }
  for (int y = 0; y < depth.rows; y++) {
    for (int x = 0; x < depth.cols; x++) {
      float d = depth.at<float>(y, x);
      if (d <= 0) continue;

      double error_sum = 0;
      int error_weight = 0;
      for (int dy = -2; dy <= 2; dy += 1) {
        const int y2 = y + dy;
        if (y2 < 0 || y2 >= depth.rows) continue;

        for (int dx = -2; dx <= 2; dx += 1) {
          const int x2 = x + dx;
          if (x2 < 0 || x2 >= depth.cols) continue;

          float d2 = depth.at<float>(y2, x2);
          if (d2 > 0) {
            error_sum += (d2 - d) * (d2 - d);
            error_weight += 1;
          }
        }
      }

      if (error_sum / error_weight >= depth_error_thresh * depth_error_thresh) {
        mask.at<int>(y, x) = 1;
      }
    }
  }
  for (int y = 0; y < depth.rows; y++) {
    for (int x = 0; x < depth.cols; x++) {
      if (mask.at<int>(y, x)) {
        depth.at<float>(y, x) = 0;
      }
    }
  }
}

void StereoIntegration::ComponentFilter(
  cv::Mat & mask,
  cv::Mat & depth,
  float connectivity_thresh,
  int min_piece_thresh,
  bool with_depth
) {
  mask.create(depth.rows, depth.cols, CV_32SC1);
  mask.setTo(0);

  for (int y = 0, connectivity_id = 0; y < depth.rows; y++) {
    for (int x = 0; x < depth.cols; x++) {
      if (mask.at<int>(y, x)) continue;
      if (depth.at<float>(y, x) <= 0) continue;

      float min_depth = depth.at<float>(y, x);
      float avg_depth = depth.at<float>(y, x);
      int count = 1;
      int id = ++connectivity_id;
      std::queue<std::pair<int, int>> Q;
      std::vector<std::pair<int, int>> H;
      Q.push(std::make_pair(y, x));
      H.push_back(std::make_pair(y, x));
      mask.at<int>(y, x) = id;
      
      while (!Q.empty()) {
        auto coord = Q.front();
        Q.pop();

        float d = depth.at<float>(coord.first, coord.second);
        for (int y2 = coord.first - 5; y2 <= coord.first + 5; y2 += 2) {
          if (y2 < 0 || y2 >= depth.rows) continue;

          for (int x2 = coord.second - 5; x2 <= coord.second + 5; x2 += 2) {
            if (x2 < 0 || x2 >= depth.cols) continue;
            if (mask.at<int>(y2, x2)) continue;
            
            float d2 = depth.at<float>(y2, x2);
            if (d2 <= 0.0f) continue;

            if ((d - d2) * (d - d2) < connectivity_thresh * connectivity_thresh) {
              Q.push(std::make_pair(y2, x2));
              H.push_back(std::make_pair(y2, x2));
              mask.at<int>(y2, x2) = id;
              min_depth = std::min(min_depth, d2);
              avg_depth += d2;
              count += 1;
            }
          }
        }
      }
      avg_depth /= count;

      if (min_depth <= options_.min_depth || 
          (with_depth && avg_depth * avg_depth * H.size() < min_piece_thresh) ||
          (!with_depth && H.size() < min_piece_thresh)
      ) {
        for (const auto & coord : H) {
          depth.at<float>(coord.first, coord.second) = 0.0f;
        }
      }
    }
  }
}

void StereoIntegration::JointTrilateralFilter(
  const cv::Mat & color, const cv::Mat & src, cv::Mat & dst, 
  int d, double sigmaColor, double sigmaDepth, double sigmaSpace, double completionThresh, 
  int sampleStep, int borderType
) {
  if (src.size() != color.size()) std::abort();
  if (color.type() != CV_8UC1) std::abort();
  if (src.type() != CV_32FC1) std::abort();
  if (dst.type() != CV_32FC1) std::abort();

  if (sigmaColor <= 0)
      sigmaColor = 1;
  if (sigmaDepth <= 0)
      sigmaDepth = 1;
  if (sigmaSpace <= 0)
      sigmaSpace = 1;

  int radius;
  if (d <= 0)
      radius = cvRound(sigmaSpace*1.5);
  else
      radius = d / 2;
  radius = std::max(radius, 1);

  dst.create(src.size(), src.type());

  if (dst.data == src.data)
      dst.create(src.size(), src.type());

  int rd = 2 * radius + 1;

  double gaussColorCoeff = -0.5 / (sigmaColor*sigmaColor);
  double gaussSpaceCoeff = -0.5 / (sigmaSpace*sigmaSpace);
  double gaussDepthCoeff = -0.5 / (sigmaDepth*sigmaDepth);

  float colorWeights[255];
  for (int i = 0; i < 255; i++)
  {
    colorWeights[i] = (float) std::exp(i * i * gaussColorCoeff);
  }

  const int depthBits = 8;
  const float depthIndexScale = 1 << depthBits;
  const int depthCount = std::sqrt(std::log(std::numeric_limits<float>::epsilon()) / gaussDepthCoeff) * depthIndexScale + 0.5f;
  std::vector<float> depthWeights;
  depthWeights.resize(depthCount);
  for (int i = 0; i < depthWeights.size(); i++) 
  {
    const float d = i / depthIndexScale;
    depthWeights[i] = (float) std::exp(d * d * gaussDepthCoeff);
  }
  depthWeights.push_back(0);

  cv::Mat jointTemp, srcTemp;
  copyMakeBorder(color, jointTemp, radius, radius, radius, radius, borderType);
  copyMakeBorder(src, srcTemp, radius, radius, radius, radius, borderType);
  size_t srcElemStep = srcTemp.step / srcTemp.elemSize();
  size_t jElemStep = jointTemp.step / jointTemp.elemSize();
  CV_Assert(srcElemStep == jElemStep);

  std::vector<float> spaceWeightsv(rd*rd);
  std::vector<int> spaceOfsJointv(rd*rd);
  float *spaceWeights = &spaceWeightsv[0];
  int *spaceOfsJoint = &spaceOfsJointv[0];

  float wMax = 0.0f;
  int maxk = 0;
  for (int i = -radius; i <= radius; i += sampleStep)
  {
    for (int j = -radius; j <= radius; j += sampleStep)
    {
      double r2 = i*i + j*j;
      if (r2 > radius * radius)
        continue;

      spaceWeights[maxk] = (float) std::exp(r2 * gaussSpaceCoeff);
      spaceOfsJoint[maxk] = (int) (i*jElemStep + j);
      wMax += spaceWeights[maxk];
      maxk++;
    }
  }

  for (int i = radius; i < radius + color.rows; i++)
  {
    int j = radius;
    for ( ; j < srcTemp.cols - radius; j++)
    {
      const uchar * jointCenterPixPtr = jointTemp.ptr<uchar>(i) + j;
      const float * srcCenterPixPtr = srcTemp.ptr<float>(i) + j;
      float srcSum = 0.0f;
      float wSum = 0.0f;

      const float srcDepth = srcCenterPixPtr[0];
      for (int k = 0 ; k < maxk; k++)
      {
        int colorDiff = std::abs((int)jointCenterPixPtr[0] - (int)jointCenterPixPtr[spaceOfsJoint[k]]);
        
        const float refDepth = srcCenterPixPtr[spaceOfsJoint[k]];
        if (refDepth > 0) 
        {
          if (srcDepth > 0) 
          {
            int depthDiff = std::abs(srcDepth - refDepth) * depthIndexScale;
            depthDiff = std::min(depthDiff, (int)depthWeights.size() - 1);

            float weight = spaceWeights[k] * colorWeights[colorDiff] * depthWeights[depthDiff];
            srcSum += weight * refDepth;
            wSum += weight;
          }
          else 
          {
            float weight = spaceWeights[k] * colorWeights[colorDiff];
            srcSum += weight * refDepth;
            wSum += weight;
          }
        }
      }

      if (srcDepth > 0.0f || completionThresh >= 0 && wSum > wMax * completionThresh) 
      {
        dst.at<float>(i - radius, j - radius) = srcSum / wSum;
      }
      else 
      {
        dst.at<float>(i - radius, j - radius) = 0;
      }
    }
  }
}

void StereoIntegration::DepthMapFilter(cv::Mat &depth, const cv::Mat &gray) {
  auto save_depth_as_obj = [&](const std::string & file) {
    const float fx = (depth.rows + depth.cols) * 0.5f;
    const float fy = (depth.rows + depth.cols) * 0.5f;
    const float cx = (depth.rows) * 0.5f;
    const float cy = (depth.cols) * 0.5f;

    FILE * fp = fopen(file.c_str(), "w");
    if (!fp) return;

    for (int y = 0; y < depth.rows; y++) {
      for (int x = 0; x < depth.cols; x++) {
        float d = depth.at<float>(y, x);
        uchar g = gray.at<uchar>(y, x);
        if (d > 0.0f) {
          float xf = d * (x - cx) / fx;
          float yf = d * (y - cy) / fy;

          fprintf(fp, "v %f %f %f %d %d %d\n", xf, yf, d, g, g, g);
        }
      }
    }
    fclose(fp);
  };

  // save_depth_as_obj("filter_0.obj");

  cv::Mat mask(depth.rows, depth.cols, CV_32SC1);

  cv::medianBlur(depth, depth, 5);
  NoiseFilter(mask, depth, options_.noise_filter_depth_thresh, options_.noise_filter_count_thresh);
  // save_depth_as_obj(std::string("filter_") + std::to_string(1) + ".obj");

  ComponentFilter(mask, depth, options_.connectivity_filter_thresh, depth.rows * depth.cols * 0.002f, true);
  ComponentFilter(mask, depth, options_.connectivity_filter_thresh, depth.rows * depth.cols * 0.005f, false);
  // save_depth_as_obj(std::string("filter_") + std::to_string(2) + ".obj");

  if (options_.do_joint_filter) {
    cv::Mat depth2(depth.rows, depth.cols, CV_32FC1);
    JointTrilateralFilter(
      gray, depth, depth2, 
      -1, 17, 9, 11, options_.joint_filter_completion_thresh,
      2, cv::BorderTypes::BORDER_ISOLATED);
    depth2.copyTo(depth);
  }
  // save_depth_as_obj(std::string("filter_") + std::to_string(3) + ".obj");
}

void StereoIntegration::Run() {
  options_.Print();
  std::cout << std::endl;

  ReadWorkspace();

  for (size_t reconstruction_idx = 0; reconstruction_idx < num_reconstruction_; 
       reconstruction_idx++) {

    PrintHeading1(StringPrintf("Integrating# %d", reconstruction_idx));
    
    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    auto reconstruction_path = 
      JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
    auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
    if (!ExistsDir(dense_reconstruction_path)) {
      continue;
    }
    active_dense_path_ = dense_reconstruction_path;

    auto undistort_image_path =
      JoinPaths(dense_reconstruction_path, IMAGES_DIR);
    if (!ExistsDir(undistort_image_path)) {
      continue;
    }

    auto undistort_sparse_path = 
      JoinPaths(dense_reconstruction_path, SPARSE_DIR);
    if (!ExistsDir(undistort_sparse_path)) {
      continue;
    }

    auto stereo_reconstruction_path = 
      JoinPaths(dense_reconstruction_path, STEREO_DIR);

    auto depth_maps_path = JoinPaths(stereo_reconstruction_path, DEPTHS_DIR);
    if (!ExistsDir(depth_maps_path)) {
      continue;
    }

    auto normal_maps_path = JoinPaths(stereo_reconstruction_path, NORMALS_DIR);
    if (!ExistsDir(normal_maps_path)) {
      continue;
    }

    auto semantic_maps_path = JoinPaths(dense_reconstruction_path, SEMANTICS_DIR);

    Clear();

    if (options_.format.compare("panorama") == 0) {
      std::vector<std::pair<float, float> > depth_ranges;
      std::vector<image_t> image_ids;
      ImportPanoramaWorkspace(dense_reconstruction_path,
        image_names_, images_, image_ids, overlapping_images_,
        depth_ranges, false);
    } else {
      Workspace::Options workspace_options;
      workspace_options.max_image_size = options_.max_image_size;
      workspace_options.image_as_rgb = true;
      workspace_options.cache_size = options_.cache_size;
      workspace_options.workspace_path = dense_reconstruction_path;
      workspace_options.workspace_format = options_.format;
      workspace_options.image_path = undistort_image_path;
      workspace_options.input_type = input_type_;

      std::unique_ptr<Workspace> workspace_;
      workspace_.reset(new Workspace(workspace_options));
      const Model& model = workspace_->GetModel();

      const double kMinTriangulationAngle = 0;
      // overlapping_images_ = model.GetMaxOverlappingImages(
      //     options_.check_num_images, kMinTriangulationAngle);

      images_ = model.images;
      image_names_.resize(images_.size());
      for (int image_idx = 0; image_idx < images_.size(); ++image_idx) {
        const std::string image_name = model.GetImageName(image_idx);
        image_names_[image_idx] = image_name;
        // const Bitmap& bitmap = workspace_->GetBitmap(image_idx);
        // images_[image_idx].SetBitmap(bitmap);
      }
    }

    if (images_.empty()) {
      return;
    }
  
    const size_t image_num = images_.size();

    semantic_maps_.resize(image_num);
    depth_maps_.resize(image_num);
    tof_depth_maps_.resize(image_num);
    used_images_.resize(image_num, 0);
    // valid_pixel_masks_.resize(image_num);
    P_.resize(image_num);
    inv_P_.resize(image_num);
    inv_R_.resize(image_num);

    auto Init = [&](int image_idx) {
      const std::string image_name = image_names_[image_idx];
      std::cout << "\rconfigure image " << image_idx + 1 << "/"
                << image_num << std::flush;

      auto& image = images_.at(image_idx);
      const std::string file_name = StringPrintf("%s.%s.%s", 
          image_name.c_str(), input_type_.c_str(), DEPTH_EXT);
      const std::string image_path = 
          JoinPaths(undistort_image_path, image_name);
      const std::string semantic_path =
          JoinPaths(semantic_maps_path, image_name);
      const std::string depth_map_path =
          JoinPaths(depth_maps_path, file_name);
      const std::string tof_depth_map_path =
          JoinPaths(dense_reconstruction_path, DEPTHS_DIR, image_name + "." + DEPTH_EXT);

      Bitmap bitmap;
      bitmap.Read(image_path, true);
      image.SetBitmap(bitmap);
      cv::Mat color = cv::Mat(bitmap.Height(), bitmap.Width(), CV_8UC1);
      for (int y = 0; y < bitmap.Height(); y++) {
        for (int x = 0; x < bitmap.Width(); x++) {
          BitmapColor<uint8_t> c;
          image.GetBitmap().GetPixel(x, y, &c);
          int gray = (c.b + c.g + c.r) / 3.0 + 0.5;
          if (gray > 255) gray = 255;
          color.at<uchar>(y, x) = gray;
        }
      }
      bitmap.Deallocate();

      if (ExistsFile(depth_map_path)) {
        auto depth_map_ptr = std::unique_ptr<DepthMap>(new DepthMap);
        depth_map_ptr->Read(depth_map_path);

        if (options_.do_filter) {
          cv::Mat depth = cv::Mat(depth_map_ptr->GetHeight(), depth_map_ptr->GetWidth(), CV_32FC1, depth_map_ptr->GetPtr());
          DepthMapFilter(depth, color);
          if ((void *)depth.data != depth_map_ptr->GetPtr()) {
            std::memcpy(depth_map_ptr->GetPtr(), depth.data, sizeof(float) * depth_map_ptr->GetHeight() * depth_map_ptr->GetWidth());
          }
        }

        depth_maps_.at(image_idx).swap(depth_map_ptr);

      } else {
        std::cout << StringPrintf(
                "WARNING: Ignoring geometric image %s, because input does not "
                "exist.", image_name.c_str())
                  << std::endl;
        return;
      }

      if (options_.tof_weight > 0) {
        if (ExistsFile(tof_depth_map_path)) {
          auto depth_map_ptr = std::unique_ptr<DepthMap>(new DepthMap);
          depth_map_ptr->Read(tof_depth_map_path);

          const int width = depth_maps_.at(image_idx)->GetWidth();
          const int height = depth_maps_.at(image_idx)->GetHeight();
          auto resized_depth_map_ptr = std::unique_ptr<DepthMap>(new DepthMap(width, height, 0, MAX_VALID_DEPTH_IN_M));

          if (depth_map_ptr->GetWidth()  != width ||
              depth_map_ptr->GetHeight() != height
          ) {
            ResizeDepthMap(*resized_depth_map_ptr, *depth_map_ptr);
            tof_depth_maps_.at(image_idx).swap(resized_depth_map_ptr);
          } else {
            tof_depth_maps_.at(image_idx).swap(depth_map_ptr);
          }

        } else {
          std::cout << StringPrintf(
                  "WARNING: Ignoring ToF image %s, because input does not "
                  "exist.", image_name.c_str())
                    << std::endl;
          return;
        }
      }

      const std::string semantic_path_jpg = 
        semantic_path.substr(0, semantic_path.size() - 3) + "png";
      if (ExistsFile(semantic_path_jpg)) {
        semantic_maps_.at(image_idx).reset(new Bitmap);
        semantic_maps_.at(image_idx)->Read(semantic_path_jpg, false);
      }

      used_images_.at(image_idx) = true;

      Eigen::RowMatrix3f K = 
        Eigen::Map<const Eigen::RowMatrix3f>(image.GetK());
      ComposeProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                              P_.at(image_idx).data());
      ComposeInverseProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                                    inv_P_.at(image_idx).data());
      inv_R_.at(image_idx) =
          Eigen::Map<const Eigen::RowMatrix3f>(image.GetR()).transpose();
    };
    std::cout << std::endl;

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    for (int image_idx = 0; image_idx < image_num; ++image_idx) {
      thread_pool->AddTask(Init, image_idx);
    }
    thread_pool->Wait();

    Eigen::Matrix3f intrinsic = Eigen::Map<const Eigen::RowMatrix3f>(images_[0].GetK());
    auto volume = std::make_shared<tsdf::ScalableTSDFVolume<tsdf::ColoredTSDFVolume<>>>(
      options_.voxel_length, options_.sdf_trunc_precision,
      options_.min_depth, options_.max_depth);
    std::cout << "TSDF volume created" << std::endl;

    std::vector<std::vector<unsigned char> > yuv_datas(image_num);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int image_idx = 0; image_idx < image_num; ++image_idx) {
      const auto &image = images_[image_idx];
      int width = image.GetWidth();
      int height = image.GetHeight();

      Eigen::Matrix4f extrinsic(Eigen::Matrix4f::Identity());
      extrinsic.block<3, 3>(0, 0) = Eigen::Map<const Eigen::RowMatrix3f>(image.GetR());
      extrinsic.block<3, 1>(0, 3) = Eigen::Map<const Eigen::Vector3f>(image.GetT());

      cv::Mat color = cv::Mat(height, width, CV_8UC3);
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          BitmapColor<uint8_t> c;
          image.GetBitmap().GetPixel(x, y, &c);
          color.at<cv::Vec3b>(y, x) = cv::Vec3b(c.b, c.g, c.r);
        }
      }

      const auto &depth_map = depth_maps_[image_idx];
      cv::Mat depth = cv::Mat(depth_map->GetHeight(), depth_map->GetWidth(), CV_32FC1, depth_map->GetPtr());
      volume->Integrate(depth, color, intrinsic, extrinsic, 1.0f);

      if (options_.tof_weight > 0) {
        const auto &depth_map = tof_depth_maps_[image_idx];
        cv::Mat depth = cv::Mat(depth_map->GetHeight(), depth_map->GetWidth(), CV_32FC1, depth_map->GetPtr());
        volume->Integrate(depth, color, intrinsic, extrinsic, options_.tof_weight);
      }

      std::cout << "\rIntegrated " << image_idx << "/" << image_num << std::flush;
    }
    std::cout << std::endl;

    auto mesh_ptr = volume->ExtractColoredTriangleMesh(options_.extract_mesh_threshold);
    std::cout << "Mesh with " << mesh_ptr->vertices_.size() << " vertices and " << mesh_ptr->faces_.size() << " triangles extracted" << std::endl;
    auto model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);
    if (options_.num_isolated_pieces > 0) {
      mesh_ptr->RemoveIsolatedPieces(options_.num_isolated_pieces);
    }
    mesh_ptr->ComputeNormals();
    WriteTriangleMeshObj(model_path, *mesh_ptr);
  }
}

void StereoIntegration::Clear() {
    image_names_.clear();
    images_.clear();
    depth_maps_.clear();
    tof_depth_maps_.clear();
    used_images_.clear();
    overlapping_images_.clear();
    // valid_pixel_masks_.clear();
    P_.clear();
    inv_P_.clear();
    inv_R_.clear();
}

}  // namespace mvs
}  // namespace sensemap
