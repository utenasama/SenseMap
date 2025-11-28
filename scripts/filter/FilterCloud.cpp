#include <iostream> 
#include <time.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h> 
#include <pcl/point_cloud.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/search/flann_search.h>  
#include <pcl/search/kdtree.h>  

#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/extract_indices.h>

// #include "endian.h"
#include "../../src/util/endian.h"

using namespace pcl;
using namespace pcl::io;
using namespace sensemap;

double computeCloudResolution (const pcl::PointCloud<PointXYZ>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointXYZ> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

void ReadPointsVisibility(const std::string& path,
    std::vector<std::vector<uint32_t> >& points_visibility) {
    std::fstream file(path, std::ios::in | std::ios::binary);
    // CHECK(file.is_open()) << path;

    uint64_t num_point = ReadBinaryLittleEndian<uint64_t>(&file);
    points_visibility.resize(num_point);
        
    for (auto& visibility : points_visibility) {
        uint32_t num_vis = ReadBinaryLittleEndian<uint32_t>(&file);
        visibility.resize(num_vis);
        ReadBinaryLittleEndian<uint32_t>(&file, &visibility);
    }
    file.close();
}

void WritePointsVisibility(
    const std::string& path,
    const std::vector<std::vector<uint32_t> >& points_visibility) {
  std::fstream file(path, std::ios::out | std::ios::binary);
  // CHECK(file.is_open()) << path;

  WriteBinaryLittleEndian<uint64_t>(&file, points_visibility.size());

  for (const auto& visibility : points_visibility) {
    WriteBinaryLittleEndian<uint32_t>(&file, visibility.size());
    for (const auto& image_idx : visibility) {
      WriteBinaryLittleEndian<uint32_t>(&file, image_idx);
    }
  }
}

int main (int argc, char** argv) {

    clock_t start;
    start = clock();

    if (argc < 4) {
      std::cout << "the param is in path, out path, radius, (threadhold) " << std::endl;
    }

    std::cerr << "Start reading point cloud..." << std::endl;
    float cloud_resolution = 0.2;
    {
      // float leaf_size = 0.03f;  // 单位：m
      pcl::PointCloud<PointXYZ>::Ptr xyz_cloud (new pcl::PointCloud<PointXYZ>);
      pcl::PLYReader xyz_reader;
      xyz_reader.read (argv[1], *xyz_cloud); 
      cloud_resolution = computeCloudResolution (xyz_cloud);
      // float cloud_resolution = 0.02;
      std::cerr << "compute Cloud Resolution: " << cloud_resolution 
                << "\t time consuming: " << (double)(clock()-start) /60 /CLOCKS_PER_SEC <<" min" << std::endl;
    }

    pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
    pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2);
    pcl::PLYReader reader;
    reader.read (argv[1], *cloud); 
    // reader.read ("fused.ply", *cloud); 
    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height 
        << " data points (" << pcl::getFieldsList (*cloud) << ")." 
              << "\t time consuming: " << (double)(clock()-start) /60 /CLOCKS_PER_SEC <<" min" << std::endl;

    int radius = std::atoi( argv[3]);
    float min_neighbors = radius * radius / 4;
    if (argc == 5)
      min_neighbors = std::atof (argv[4]); 
    float leaf_size = cloud_resolution * radius;
    std::cerr << "OutlierRemoval param " << "(leaf_size: " << leaf_size 
              << "/  min_neighbors: " << min_neighbors << ")" 
              << "\t time consuming: " << (double)(clock()-start) /60/CLOCKS_PER_SEC <<" min" << std::endl;

    pcl::RadiusOutlierRemoval<pcl::PCLPointCloud2> sor(true);
    pcl::PointIndices filter_pi;
    sor.setInputCloud(cloud); 
    sor.setRadiusSearch(leaf_size); 
    sor.setMinNeighborsInRadius(min_neighbors);
    sor.filter(*cloud_filtered);
    sor.getRemovedIndices(filter_pi);

    std::cerr << "PointCloud after remove outlier: " 
        << cloud_filtered->width * cloud_filtered->height 
        << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." 
        << "\t time consuming: " << (double)(clock()-start) /60 /CLOCKS_PER_SEC 
        <<" min" << std::endl;

    // remove vis point
    std::string input_vis_path = argv[1];
    input_vis_path = input_vis_path + ".vis";
    std::cout << "input visibility: " << input_vis_path << std::endl;
    std::vector<std::vector<uint32_t> > vis_points;
    ReadPointsVisibility(input_vis_path, vis_points);

    int j = 0;
    filter_pi.indices.push_back(0);
    std::vector<std::vector<uint32_t> > filter_vis_points;
    for (int i=0; i <vis_points.size(); i++){
      if (i == filter_pi.indices.at(j)) {
        j++;
        continue;
      }
      std::vector<uint32_t> vis_point = vis_points.at(i);
      filter_vis_points.push_back(vis_point);
    }
    std::cerr << "vis_points size: " << vis_points.size() << "\t &&\t"
         << "filter_vis_points size: " << filter_vis_points.size() << std::endl;
 
    // write
    pcl::PLYWriter writer;
    writer.write( argv[2], cloud_filtered, Eigen::Vector4f::Zero(), 
                    Eigen::Quaternionf::Identity(),true, false);
    std::string output_vis_path = argv[2];
    output_vis_path = output_vis_path + ".vis";
    WritePointsVisibility(output_vis_path, filter_vis_points);

    std::cerr << "The End" 
              << "\t Total time consuming: " << (double)(clock()-start) /60 /CLOCKS_PER_SEC  <<" min" << std::endl;

    return (0);
}
