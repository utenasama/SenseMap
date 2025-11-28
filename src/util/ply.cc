//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "ply.h"
#include "util/semantic_table.h"

#include <fstream>
#include <iomanip>
#include <Eigen/Core>

#include <PoissonRecon/Ply.h>

#include "logging.h"
#include "misc.h"

namespace sensemap {

bool ReadPointRealType(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

  std::vector<PlyPoint> points;

  std::string line;

  // The index of the property for ASCII PLY files.
  int X_index = -1;
  int Y_index = -1;
  int Z_index = -1;
  int NX_index = -1;
  int NY_index = -1;
  int NZ_index = -1;
  int R_index = -1;
  int G_index = -1;
  int B_index = -1;

  // The position in number of bytes of the property for binary PLY files.
  int X_byte_pos = -1;
  int Y_byte_pos = -1;
  int Z_byte_pos = -1;
  int NX_byte_pos = -1;
  int NY_byte_pos = -1;
  int NZ_byte_pos = -1;
  int R_byte_pos = -1;
  int G_byte_pos = -1;
  int B_byte_pos = -1;

  // Flag to use double precision in binary PLY files
  bool X_double = false;
  bool Y_double = false;
  bool Z_double = false;
  bool NX_double = false;
  bool NY_double = false;
  bool NZ_double = false;

  bool in_vertex_section = false;
  bool is_binary = false;
  bool is_little_endian = false;
  size_t num_bytes_per_line = 0;
  size_t num_vertices = 0;

  int index = 0;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    if (line == "end_header") {
      break;
    }

    if (line.size() >= 6 && line.substr(0, 6) == "format") {
      if (line == "format ascii 1.0") {
        is_binary = false;
      } else if (line == "format binary_little_endian 1.0") {
        is_binary = true;
        is_little_endian = true;
      } else if (line == "format binary_big_endian 1.0") {
        is_binary = true;
        is_little_endian = false;
      }
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      in_vertex_section = false;
      if (line_elems[1] == "vertex") {
        num_vertices = std::stoll(line_elems[2]);
        in_vertex_section = true;
      } else if (std::stoll(line_elems[2]) > 0) {
        std::cout << "WARN: Only vertex elements supported; ignoring "
                  << line_elems[1] << std::endl;
      }
    }

    if (!in_vertex_section) {
      continue;
    }

    // Show diffuse, ambient, specular colors as regular colors.

    if (line_elems.size() >= 3 && line_elems[0] == "property") {
      CHECK(line_elems[1] == "float" || line_elems[1] == "float32" ||
            line_elems[1] == "double" || line_elems[1] == "float64" ||
            line_elems[1] == "uchar")
          << "PLY import only supports float, double, and uchar data types";

      if (line == "property float x" || line == "property float32 x" ||
          line == "property double x" || line == "property float64 x") {
        X_index = index;
        X_byte_pos = num_bytes_per_line;
        X_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float y" || line == "property float32 y" ||
                 line == "property double y" || line == "property float64 y") {
        Y_index = index;
        Y_byte_pos = num_bytes_per_line;
        Y_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float z" || line == "property float32 z" ||
                 line == "property double z" || line == "property float64 z") {
        Z_index = index;
        Z_byte_pos = num_bytes_per_line;
        Z_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float nx" || line == "property float32 nx" ||
                 line == "property double nx" ||
                 line == "property float64 nx") {
        NX_index = index;
        NX_byte_pos = num_bytes_per_line;
        NX_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float ny" || line == "property float32 ny" ||
                 line == "property double ny" ||
                 line == "property float64 ny") {
        NY_index = index;
        NY_byte_pos = num_bytes_per_line;
        NY_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float nz" || line == "property float32 nz" ||
                 line == "property double nz" ||
                 line == "property float64 nz") {
        NZ_index = index;
        NZ_byte_pos = num_bytes_per_line;
        NZ_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property uchar r" || line == "property uchar red" ||
                 line == "property uchar diffuse_red" ||
                 line == "property uchar ambient_red" ||
                 line == "property uchar specular_red") {
        R_index = index;
        R_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar g" || line == "property uchar green" ||
                 line == "property uchar diffuse_green" ||
                 line == "property uchar ambient_green" ||
                 line == "property uchar specular_green") {
        G_index = index;
        G_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar b" || line == "property uchar blue" ||
                 line == "property uchar diffuse_blue" ||
                 line == "property uchar ambient_blue" ||
                 line == "property uchar specular_blue") {
        B_index = index;
        B_byte_pos = num_bytes_per_line;
      }

      index += 1;
      if (line_elems[1] == "float" || line_elems[1] == "float32") {
        num_bytes_per_line += 4;
      } else if (line_elems[1] == "double" || line_elems[1] == "float64") {
        num_bytes_per_line += 8;
      } else if (line_elems[1] == "uchar") {
        num_bytes_per_line += 1;
      } else {
        LOG(FATAL) << "Invalid data type: " << line_elems[1];
      }
    }
  }

  return X_double;
}

std::vector<PlyPoint> ReadPly(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

  std::vector<PlyPoint> points;

  std::string line;

  // The index of the property for ASCII PLY files.
  int X_index = -1;
  int Y_index = -1;
  int Z_index = -1;
  int NX_index = -1;
  int NY_index = -1;
  int NZ_index = -1;
  int R_index = -1;
  int G_index = -1;
  int B_index = -1;

  // The position in number of bytes of the property for binary PLY files.
  int X_byte_pos = -1;
  int Y_byte_pos = -1;
  int Z_byte_pos = -1;
  int NX_byte_pos = -1;
  int NY_byte_pos = -1;
  int NZ_byte_pos = -1;
  int R_byte_pos = -1;
  int G_byte_pos = -1;
  int B_byte_pos = -1;

  // Flag to use double precision in binary PLY files
  bool X_double = false;
  bool Y_double = false;
  bool Z_double = false;
  bool NX_double = false;
  bool NY_double = false;
  bool NZ_double = false;

  bool in_vertex_section = false;
  bool is_binary = false;
  bool is_little_endian = false;
  size_t num_bytes_per_line = 0;
  size_t num_vertices = 0;

  int index = 0;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    if (line == "end_header") {
      break;
    }

    if (line.size() >= 6 && line.substr(0, 6) == "format") {
      if (line == "format ascii 1.0") {
        is_binary = false;
      } else if (line == "format binary_little_endian 1.0") {
        is_binary = true;
        is_little_endian = true;
      } else if (line == "format binary_big_endian 1.0") {
        is_binary = true;
        is_little_endian = false;
      }
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      in_vertex_section = false;
      if (line_elems[1] == "vertex") {
        num_vertices = std::stoll(line_elems[2]);
        in_vertex_section = true;
      } else if (std::stoll(line_elems[2]) > 0) {
        std::cout << "WARN: Only vertex elements supported; ignoring "
                  << line_elems[1] << std::endl;
      }
    }

    if (!in_vertex_section) {
      continue;
    }

    // Show diffuse, ambient, specular colors as regular colors.

    if (line_elems.size() >= 3 && line_elems[0] == "property") {
      CHECK(line_elems[1] == "float" || line_elems[1] == "float32" ||
            line_elems[1] == "double" || line_elems[1] == "float64" ||
            line_elems[1] == "uchar")
          << "PLY import only supports float, double, and uchar data types";

      if (line == "property float x" || line == "property float32 x" ||
          line == "property double x" || line == "property float64 x") {
        X_index = index;
        X_byte_pos = num_bytes_per_line;
        X_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float y" || line == "property float32 y" ||
                 line == "property double y" || line == "property float64 y") {
        Y_index = index;
        Y_byte_pos = num_bytes_per_line;
        Y_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float z" || line == "property float32 z" ||
                 line == "property double z" || line == "property float64 z") {
        Z_index = index;
        Z_byte_pos = num_bytes_per_line;
        Z_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float nx" || line == "property float32 nx" ||
                 line == "property double nx" ||
                 line == "property float64 nx") {
        NX_index = index;
        NX_byte_pos = num_bytes_per_line;
        NX_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float ny" || line == "property float32 ny" ||
                 line == "property double ny" ||
                 line == "property float64 ny") {
        NY_index = index;
        NY_byte_pos = num_bytes_per_line;
        NY_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float nz" || line == "property float32 nz" ||
                 line == "property double nz" ||
                 line == "property float64 nz") {
        NZ_index = index;
        NZ_byte_pos = num_bytes_per_line;
        NZ_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property uchar r" || line == "property uchar red" ||
                 line == "property uchar diffuse_red" ||
                 line == "property uchar ambient_red" ||
                 line == "property uchar specular_red") {
        R_index = index;
        R_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar g" || line == "property uchar green" ||
                 line == "property uchar diffuse_green" ||
                 line == "property uchar ambient_green" ||
                 line == "property uchar specular_green") {
        G_index = index;
        G_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar b" || line == "property uchar blue" ||
                 line == "property uchar diffuse_blue" ||
                 line == "property uchar ambient_blue" ||
                 line == "property uchar specular_blue") {
        B_index = index;
        B_byte_pos = num_bytes_per_line;
      }

      index += 1;
      if (line_elems[1] == "float" || line_elems[1] == "float32") {
        num_bytes_per_line += 4;
      } else if (line_elems[1] == "double" || line_elems[1] == "float64") {
        num_bytes_per_line += 8;
      } else if (line_elems[1] == "uchar") {
        num_bytes_per_line += 1;
      } else {
        LOG(FATAL) << "Invalid data type: " << line_elems[1];
      }
    }
  }

  const bool is_normal_missing =
      (NX_index == -1) || (NY_index == -1) || (NZ_index == -1);
  const bool is_rgb_missing =
      (R_index == -1) || (G_index == -1) || (B_index == -1);

  CHECK(X_index != -1 && Y_index != -1 && Z_index)
      << "Invalid PLY file format: x, y, z properties missing";

  points.reserve(num_vertices);

  if (is_binary) {
    std::vector<char> buffer(num_bytes_per_line);
    for (size_t i = 0; i < num_vertices; ++i) {
      file.read(buffer.data(), num_bytes_per_line);

      PlyPoint point;

      if (is_little_endian) {
        point.x = LittleEndianToNative(
            X_double ? *reinterpret_cast<double*>(&buffer[X_byte_pos])
                     : *reinterpret_cast<float*>(&buffer[X_byte_pos]));
        point.y = LittleEndianToNative(
            Y_double ? *reinterpret_cast<double*>(&buffer[Y_byte_pos])
                     : *reinterpret_cast<float*>(&buffer[Y_byte_pos]));
        point.z = LittleEndianToNative(
            Z_double ? *reinterpret_cast<double*>(&buffer[Z_byte_pos])
                     : *reinterpret_cast<float*>(&buffer[Z_byte_pos]));

        if (!is_normal_missing) {
          point.nx = LittleEndianToNative(
              NX_double ? *reinterpret_cast<double*>(&buffer[NX_byte_pos])
                        : *reinterpret_cast<float*>(&buffer[NX_byte_pos]));
          point.ny = LittleEndianToNative(
              NY_double ? *reinterpret_cast<double*>(&buffer[NY_byte_pos])
                        : *reinterpret_cast<float*>(&buffer[NY_byte_pos]));
          point.nz = LittleEndianToNative(
              NZ_double ? *reinterpret_cast<double*>(&buffer[NZ_byte_pos])
                        : *reinterpret_cast<float*>(&buffer[NZ_byte_pos]));
        }

        if (!is_rgb_missing) {
          point.r = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[R_byte_pos]));
          point.g = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[G_byte_pos]));
          point.b = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[B_byte_pos]));
        }
      } else {
        point.x = BigEndianToNative(
            X_double ? *reinterpret_cast<double*>(&buffer[X_byte_pos])
                     : *reinterpret_cast<float*>(&buffer[X_byte_pos]));
        point.y = BigEndianToNative(
            Y_double ? *reinterpret_cast<double*>(&buffer[Y_byte_pos])
                     : *reinterpret_cast<float*>(&buffer[Y_byte_pos]));
        point.z = BigEndianToNative(
            Z_double ? *reinterpret_cast<double*>(&buffer[Z_byte_pos])
                     : *reinterpret_cast<float*>(&buffer[Z_byte_pos]));

        if (!is_normal_missing) {
          point.nx = BigEndianToNative(
              NX_double ? *reinterpret_cast<double*>(&buffer[NX_byte_pos])
                        : *reinterpret_cast<float*>(&buffer[NX_byte_pos]));
          point.ny = BigEndianToNative(
              NY_double ? *reinterpret_cast<double*>(&buffer[NY_byte_pos])
                        : *reinterpret_cast<float*>(&buffer[NY_byte_pos]));
          point.nz = BigEndianToNative(
              NZ_double ? *reinterpret_cast<double*>(&buffer[NZ_byte_pos])
                        : *reinterpret_cast<float*>(&buffer[NZ_byte_pos]));
        }

        if (!is_rgb_missing) {
          point.r = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[R_byte_pos]));
          point.g = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[G_byte_pos]));
          point.b = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[B_byte_pos]));
        }
      }

      points.push_back(point);
    }
  } else {
    while (std::getline(file, line)) {
      StringTrim(&line);
      std::stringstream line_stream(line);

      std::string item;
      std::vector<std::string> items;
      while (!line_stream.eof()) {
        std::getline(line_stream, item, ' ');
        StringTrim(&item);
        items.push_back(item);
      }

      PlyPoint point;

      point.x = std::stold(items.at(X_index));
      point.y = std::stold(items.at(Y_index));
      point.z = std::stold(items.at(Z_index));

      if (!is_normal_missing) {
        point.nx = std::stold(items.at(NX_index));
        point.ny = std::stold(items.at(NY_index));
        point.nz = std::stold(items.at(NZ_index));
      }

      if (!is_rgb_missing) {
        point.r = std::stoi(items.at(R_index));
        point.g = std::stoi(items.at(G_index));
        point.b = std::stoi(items.at(B_index));
      }

      points.push_back(point);
    }
  }

  return points;
}

void WriteBinaryPlyPoints(const std::string& path,
                          const std::vector<PlyPoint>& points,
                          const bool write_normal, const bool write_rgb) {
  std::fstream text_file(path, std::ios::out);
  CHECK(text_file.is_open()) << path;

  text_file << "ply" << std::endl;
  text_file << "format binary_little_endian 1.0" << std::endl;
  text_file << "element vertex "  << std::setw(20) << points.size() << std::endl;

  text_file << "property float x" << std::endl;
  text_file << "property float y" << std::endl;
  text_file << "property float z" << std::endl;

  if (write_normal) {
    text_file << "property float nx" << std::endl;
    text_file << "property float ny" << std::endl;
    text_file << "property float nz" << std::endl;
  }

  if (write_rgb) {
    text_file << "property uchar red" << std::endl;
    text_file << "property uchar green" << std::endl;
    text_file << "property uchar blue" << std::endl;
  }

  text_file << "end_header" << std::endl;
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;

  for (const auto& point : points) {
    WriteBinaryLittleEndian<float>(&binary_file, point.x);
    WriteBinaryLittleEndian<float>(&binary_file, point.y);
    WriteBinaryLittleEndian<float>(&binary_file, point.z);

    if (write_normal) {
      WriteBinaryLittleEndian<float>(&binary_file, point.nx);
      WriteBinaryLittleEndian<float>(&binary_file, point.ny);
      WriteBinaryLittleEndian<float>(&binary_file, point.nz);
    }

    if (write_rgb) {
      WriteBinaryLittleEndian<uint8_t>(&binary_file, point.r);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, point.g);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, point.b);
    }
  }

  binary_file.close();
}

void WriteTextPlyPoints(const std::string& path,
                        const std::vector<PlyPoint>& points,
                        const bool write_normal, const bool write_rgb) {
  std::ofstream file(path);
  CHECK(file.is_open()) << path;

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "element vertex " << points.size() << std::endl;

  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;

  if (write_normal) {
    file << "property float nx" << std::endl;
    file << "property float ny" << std::endl;
    file << "property float nz" << std::endl;
  }

  if (write_rgb) {
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
  }

  file << "end_header" << std::endl;

  for (const auto& point : points) {
    file << point.x << " " << point.y << " " << point.z;

    if (write_normal) {
      file << " " << point.nx << " " << point.ny << " " << point.nz;
    }

    if (write_rgb) {
      file << " " << static_cast<int>(point.r) << " "
           << static_cast<int>(point.g) << " " << static_cast<int>(point.b);
    }

    file << std::endl;
  }

  file.close();
}

void AppendWriteBinaryPlyPoints(const std::string& path,
                                const std::vector<PlyPoint>& points){
  CHECK(ExistsFile(path)) << path;

  std::fstream text_file(path, std::ios::in|std::ios::out);
  CHECK(text_file.is_open()) << path;

  int NX_index = -1;
  int NY_index = -1;
  int NZ_index = -1;
  int R_index = -1;
  int G_index = -1;
  int B_index = -1;

  std::string line;
  bool in_vertex_section = false;
  bool write_normal = false;
  bool write_rgb = false;
  uint64_t ori_num_vertices = 0;
  int begin_pos = text_file.tellg();
  int seek_pos = text_file.tellg();

  int index = 0;
  while (std::getline(text_file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      begin_pos = text_file.tellg();
      continue;
    }

    if (line == "end_header") {
      break;
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      if (line_elems[1] == "vertex") {
        ori_num_vertices = std::stoll(line_elems[2]);
        seek_pos = begin_pos;
        in_vertex_section = true;
      }
    }

    if (!in_vertex_section){
      begin_pos = text_file.tellg();
      continue;
    }


    if (line_elems.size() >= 3 && line_elems[0] == "property") {
      CHECK(line_elems[1] == "float" || line_elems[1] == "float32" ||
            line_elems[1] == "uchar")
          << "PLY import only supports the float and uchar data types";

      if (line == "property float nx" || line == "property float32 nx") {
        NX_index = index;
      } else if (line == "property float ny" || line == "property float32 ny") {
        NY_index = index;
      } else if (line == "property float nz" || line == "property float32 nz") {
        NZ_index = index;
      } else if (line == "property uchar r" || line == "property uchar red" ||
                 line == "property uchar diffuse_red" ||
                 line == "property uchar ambient_red" ||
                 line == "property uchar specular_red") {
        R_index = index;
      } else if (line == "property uchar g" || line == "property uchar green" ||
                 line == "property uchar diffuse_green" ||
                 line == "property uchar ambient_green" ||
                 line == "property uchar specular_green") {
        G_index = index;
      } else if (line == "property uchar b" || line == "property uchar blue" ||
                 line == "property uchar diffuse_blue" ||
                 line == "property uchar ambient_blue" ||
                 line == "property uchar specular_blue") {
        B_index = index;
      }
      index += 1;
    }
  }
  text_file.seekp(seek_pos, std::ios::beg);
  text_file << "element vertex " << std::setw(20) << ori_num_vertices + points.size() << std::endl;
  text_file.close();

  const bool is_normal_missing =
      (NX_index == -1) || (NY_index == -1) || (NZ_index == -1);
  const bool is_rgb_missing =
      (R_index == -1) || (G_index == -1) || (B_index == -1);

  std::fstream binary_file(path,
                          std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;
  
  begin_pos = binary_file.tellg();
  // binary_file.clear(binary_file.rdstate() & ~std::ifstream::failbit);
  for (const auto& point : points) {
          WriteBinaryLittleEndian<float>(&binary_file, point.x);
          WriteBinaryLittleEndian<float>(&binary_file, point.y);
          WriteBinaryLittleEndian<float>(&binary_file, point.z);

      if (!is_normal_missing) {
          WriteBinaryLittleEndian<float>(&binary_file, point.nx);
          WriteBinaryLittleEndian<float>(&binary_file, point.ny);
          WriteBinaryLittleEndian<float>(&binary_file, point.nz);
      }

      if (!is_rgb_missing) {
          WriteBinaryLittleEndian<uint8_t>(&binary_file, point.r);
          WriteBinaryLittleEndian<uint8_t>(&binary_file, point.g);
          WriteBinaryLittleEndian<uint8_t>(&binary_file, point.b);
      }
  }
  binary_file.close();
}

void ReadPly(const std::string& path,
            std::vector<double>& Xs,
            std::vector<double>& Ys,
            std::vector<double>& Zs,
            std::vector<double>& nXs,
            std::vector<double>& nYs,
            std::vector<double>& nZs,
            std::vector<uint8_t>& rs,
            std::vector<uint8_t>& gs,
            std::vector<uint8_t>& bs) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

  double X;
  double Y;
  double Z;
  double nX;
  double nY;
  double nZ;
  uint8_t r;
  uint8_t g;
  uint8_t b;

  std::string line;

  // The index of the property for ASCII PLY files.
  int X_index = -1;
  int Y_index = -1;
  int Z_index = -1;
  int NX_index = -1;
  int NY_index = -1;
  int NZ_index = -1;
  int R_index = -1;
  int G_index = -1;
  int B_index = -1;

  // The position in number of bytes of the property for binary PLY files.
  int X_byte_pos = -1;
  int Y_byte_pos = -1;
  int Z_byte_pos = -1;
  int NX_byte_pos = -1;
  int NY_byte_pos = -1;
  int NZ_byte_pos = -1;
  int R_byte_pos = -1;
  int G_byte_pos = -1;
  int B_byte_pos = -1;

  bool in_vertex_section = false;
  bool is_binary = false;
  bool is_little_endian = false;
  size_t num_bytes_per_line = 0;
  size_t num_vertices = 0;

  int index = 0;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    if (line == "end_header") {
      break;
    }

    if (line.size() >= 6 && line.substr(0, 6) == "format") {
      if (line == "format ascii 1.0") {
        is_binary = false;
      } else if (line == "format binary_little_endian 1.0") {
        is_binary = true;
        is_little_endian = true;
      } else if (line == "format binary_big_endian 1.0") {
        is_binary = true;
        is_little_endian = false;
      }
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      in_vertex_section = false;
      if (line_elems[1] == "vertex") {
        num_vertices = std::stoll(line_elems[2]);
        in_vertex_section = true;
      } else if (std::stoll(line_elems[2]) > 0) {
        LOG(FATAL) << "Only vertex elements supported";
      }
    }

    if (!in_vertex_section) {
      continue;
    }

    // Show diffuse, ambient, specular colors as regular colors.

    if (line_elems.size() >= 3 && line_elems[0] == "property") {
      CHECK(line_elems[1] == "double" || line_elems[1] == "float32" ||
            line_elems[1] == "uchar")
          << "PLY import only supports the float and uchar data types";

      if (line == "property double x" || line == "property float32 x") {
        X_index = index;
        X_byte_pos = num_bytes_per_line;
      } else if (line == "property double y" || line == "property float32 y") {
        Y_index = index;
        Y_byte_pos = num_bytes_per_line;
      } else if (line == "property double z" || line == "property float32 z") {
        Z_index = index;
        Z_byte_pos = num_bytes_per_line;
      } else if (line == "property double nx" || line == "property float32 nx") {
        NX_index = index;
        NX_byte_pos = num_bytes_per_line;
      } else if (line == "property double ny" || line == "property float32 ny") {
        NY_index = index;
        NY_byte_pos = num_bytes_per_line;
      } else if (line == "property double nz" || line == "property float32 nz") {
        NZ_index = index;
        NZ_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar r" || line == "property uchar red" ||
                 line == "property uchar diffuse_red" ||
                 line == "property uchar ambient_red" ||
                 line == "property uchar specular_red") {
        R_index = index;
        R_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar g" || line == "property uchar green" ||
                 line == "property uchar diffuse_green" ||
                 line == "property uchar ambient_green" ||
                 line == "property uchar specular_green") {
        G_index = index;
        G_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar b" || line == "property uchar blue" ||
                 line == "property uchar diffuse_blue" ||
                 line == "property uchar ambient_blue" ||
                 line == "property uchar specular_blue") {
        B_index = index;
        B_byte_pos = num_bytes_per_line;
      }

      index += 1;
      if (line_elems[1] == "double" || line_elems[1] == "float64") {
        num_bytes_per_line += 8;
      } else if (line_elems[1] == "float" || line_elems[1] == "float32") {
        num_bytes_per_line += 4;
      } else if (line_elems[1] == "uchar") {
        num_bytes_per_line += 1;
      } else {
        LOG(FATAL) << "Invalid data type: " << line_elems[1];
      }
    }
  }

  const bool is_normal_missing =
      (NX_index == -1) || (NY_index == -1) || (NZ_index == -1);
  const bool is_rgb_missing =
      (R_index == -1) || (G_index == -1) || (B_index == -1);

  CHECK(X_index != -1 && Y_index != -1 && Z_index)
      << "Invalid PLY file format: x, y, z properties missing";

  // points.reserve(num_vertices);
  Xs.reserve(num_vertices);
  Ys.reserve(num_vertices);
  Zs.reserve(num_vertices);
  nXs.reserve(num_vertices);
  nYs.reserve(num_vertices);
  nZs.reserve(num_vertices);
  rs.reserve(num_vertices);
  gs.reserve(num_vertices);
  bs.reserve(num_vertices);

  if (is_binary) {
    std::vector<char> buffer(num_bytes_per_line);
    for (size_t i = 0; i < num_vertices; ++i) {
      file.read(buffer.data(), num_bytes_per_line);

      PlyPoint point;

      if (is_little_endian) {
        X = LittleEndianToNative(
            *reinterpret_cast<double*>(&buffer[X_byte_pos]));
        Y = LittleEndianToNative(
            *reinterpret_cast<double*>(&buffer[Y_byte_pos]));
        Z = LittleEndianToNative(
            *reinterpret_cast<double*>(&buffer[Z_byte_pos]));

        if (!is_normal_missing) {
          nX = LittleEndianToNative(
              *reinterpret_cast<double*>(&buffer[NX_byte_pos]));
          nY = LittleEndianToNative(
              *reinterpret_cast<double*>(&buffer[NY_byte_pos]));
          nZ = LittleEndianToNative(
              *reinterpret_cast<double*>(&buffer[NZ_byte_pos]));
        }

        if (!is_rgb_missing) {
          r = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[R_byte_pos]));
          g = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[G_byte_pos]));
          b = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[B_byte_pos]));
        }
      } else {
        X =
            BigEndianToNative(*reinterpret_cast<double*>(&buffer[X_byte_pos]));
        Y =
            BigEndianToNative(*reinterpret_cast<double*>(&buffer[Y_byte_pos]));
        Z =
            BigEndianToNative(*reinterpret_cast<double*>(&buffer[Z_byte_pos]));

        if (!is_normal_missing) {
          nX = BigEndianToNative(
              *reinterpret_cast<double*>(&buffer[NX_byte_pos]));
          nY = BigEndianToNative(
              *reinterpret_cast<double*>(&buffer[NY_byte_pos]));
          nZ = BigEndianToNative(
              *reinterpret_cast<double*>(&buffer[NZ_byte_pos]));
        }

        if (!is_rgb_missing) {
          r = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[R_byte_pos]));
          g = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[G_byte_pos]));
          b = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[B_byte_pos]));
        }
      }

      Xs.push_back(X);
      Ys.push_back(Y);
      Zs.push_back(Z);
      nXs.push_back(nX);
      nYs.push_back(nY);
      nZs.push_back(nZ);
      rs.push_back(r);
      gs.push_back(g);
      bs.push_back(b);
    }
  } else {
    while (std::getline(file, line)) {
      StringTrim(&line);
      std::stringstream line_stream(line);

      std::string item;
      std::vector<std::string> items;
      while (!line_stream.eof()) {
        std::getline(line_stream, item, ' ');
        StringTrim(&item);
        items.push_back(item);
      }

      PlyPoint point;

      X = std::stold(items.at(X_index));
      Y = std::stold(items.at(Y_index));
      Z = std::stold(items.at(Z_index));

      if (!is_normal_missing) {
        nX = std::stold(items.at(NX_index));
        nY = std::stold(items.at(NY_index));
        nZ = std::stold(items.at(NZ_index));
      }

      if (!is_rgb_missing) {
        r = std::stoi(items.at(R_index));
        g = std::stoi(items.at(G_index));
        b = std::stoi(items.at(B_index));
      }

      Xs.push_back(X);
      Ys.push_back(Y);
      Zs.push_back(Z);
      nXs.push_back(nX);
      nYs.push_back(nY);
      nZs.push_back(nZ);
      rs.push_back(r);
      gs.push_back(g);
      bs.push_back(b);
    }
  }

  return;
}

void WriteBinaryPlyPoints(const std::string& path,
                          const std::vector<double>& Xs,
                          const std::vector<double>& Ys,
                          const std::vector<double>& Zs,
                          const std::vector<double>& nXs,
                          const std::vector<double>& nYs,
                          const std::vector<double>& nZs,
                          const std::vector<uint8_t>& rs,
                          const std::vector<uint8_t>& gs,
                          const std::vector<uint8_t>& bs,
                          const bool write_normal, const bool write_rgb) {
  std::fstream text_file(path, std::ios::out);
  CHECK(text_file.is_open()) << path;

  text_file << "ply" << std::endl;
  text_file << "format binary_little_endian 1.0" << std::endl;
  text_file << "element vertex " << Xs.size() << std::endl;

  text_file << "property double x" << std::endl;
  text_file << "property double y" << std::endl;
  text_file << "property double z" << std::endl;

  if (write_normal) {
    text_file << "property double nx" << std::endl;
    text_file << "property double ny" << std::endl;
    text_file << "property double nz" << std::endl;
  }

  if (write_rgb) {
    text_file << "property uchar red" << std::endl;
    text_file << "property uchar green" << std::endl;
    text_file << "property uchar blue" << std::endl;
  }

  text_file << "end_header" << std::endl;
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;

  for (size_t i = 0; i < Xs.size(); ++i) {
    WriteBinaryLittleEndian<double>(&binary_file, Xs[i]);
    WriteBinaryLittleEndian<double>(&binary_file, Ys[i]);
    WriteBinaryLittleEndian<double>(&binary_file, Zs[i]);

    if (write_normal) {
      WriteBinaryLittleEndian<double>(&binary_file, nXs[i]);
      WriteBinaryLittleEndian<double>(&binary_file, nYs[i]);
      WriteBinaryLittleEndian<double>(&binary_file, nZs[i]);
    }

    if (write_rgb) {
      WriteBinaryLittleEndian<uint8_t>(&binary_file, rs[i]);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, gs[i]);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, bs[i]);
    }
  }

  binary_file.close();
}

void ReadPointsSemantic(const std::string& path,
                        std::vector<PlyPoint>& points,
                        const bool read_binary) {
    if (read_binary) {
      std::fstream binary_file(path, std::ios::in | std::ios::binary);
      CHECK(binary_file.is_open()) << path;

      for (auto& point : points) {
        point.s_id = ReadBinaryLittleEndian<uint8_t>(&binary_file);
      }
      binary_file.close();
    } else {
      std::fstream text_file(path, std::ios::in);
      CHECK(text_file.is_open()) << path;

      for (auto& point : points) {
        int s_id;
        text_file >> s_id;
        point.s_id = s_id;
      }
      text_file.close();
    }
}

void WritePointsSemantic(const std::string& path,
                         const std::vector<PlyPoint>& points,
                         const bool write_binary) {
  if (write_binary) {
    std::fstream binary_file(path, std::ios::out | std::ios::binary);
    CHECK(binary_file.is_open()) << path;

    for (const auto& point : points) {
      WriteBinaryLittleEndian<uint8_t>(&binary_file, point.s_id);
    }

    binary_file.close();
  } else {
    std::fstream text_file(path, std::ios::out);
    CHECK(text_file.is_open()) << path;
    for (const auto& point : points) {
      text_file << (int)point.s_id << " ";
    }
    text_file << std::endl;
    text_file.close();
  }
}

void AppendWritePointsSemantic(const std::string& path,
                         const std::vector<PlyPoint>& points,
                         const bool write_binary) {
  CHECK(ExistsFile(path)) << path;
  if (write_binary) {
    std::fstream binary_file(path, std::ios::out | std::ios::binary | std::ios::app);
    CHECK(binary_file.is_open()) << path;

    for (const auto& point : points) {
      WriteBinaryLittleEndian<uint8_t>(&binary_file, point.s_id);
    }

    binary_file.close();
  } else {
    std::fstream text_file(path, std::ios::out | std::ios::app);
    CHECK(text_file.is_open()) << path;
    // text_file.seekp(-4, std::ios::end);
    for (const auto& point : points) {
      text_file << (int)point.s_id << " ";
    }
    text_file << std::endl;
    text_file.close();
  }
}

void WritePointsSemanticColor(const std::string& path,
                              const std::vector<PlyPoint>& points) {
  std::fstream text_file(path, std::ios::out);
  CHECK(text_file.is_open()) << path;

  text_file << "ply" << std::endl;
  text_file << "format binary_little_endian 1.0" << std::endl;
  text_file << "element vertex " << std::setw(20) << points.size() << std::endl;

  text_file << "property float x" << std::endl;
  text_file << "property float y" << std::endl;
  text_file << "property float z" << std::endl;

  text_file << "property float nx" << std::endl;
  text_file << "property float ny" << std::endl;
  text_file << "property float nz" << std::endl;

  text_file << "property uchar red" << std::endl;
  text_file << "property uchar green" << std::endl;
  text_file << "property uchar blue" << std::endl;

  text_file << "end_header" << std::endl;
  text_file.close();

  std::fstream binary_file(path, 
                           std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;

  for (const auto& point : points) {
    WriteBinaryLittleEndian<float>(&binary_file, point.x);
    WriteBinaryLittleEndian<float>(&binary_file, point.y);
    WriteBinaryLittleEndian<float>(&binary_file, point.z);

    // if (write_normal) {
      WriteBinaryLittleEndian<float>(&binary_file, point.nx);
      WriteBinaryLittleEndian<float>(&binary_file, point.ny);
      WriteBinaryLittleEndian<float>(&binary_file, point.nz);
    // }

    // if (write_rgb) {
      uint8_t rgb[3];
      if (point.s_id != -1) {
        rgb[0] = adepallete[point.s_id * 3];
        rgb[1] = adepallete[point.s_id * 3 + 1];
        rgb[2] = adepallete[point.s_id * 3 + 2];
      } else {
        rgb[0] = 255;
        rgb[1] = 255;
        rgb[2] = 255;
      }

      WriteBinaryLittleEndian<uint8_t>(&binary_file, rgb[0]);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, rgb[1]);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, rgb[2]);
    // }
  }
  binary_file.close();
}

void AppendWritePointsSemanticColor(const std::string& path,
                                    const std::vector<PlyPoint>& points) {
  CHECK(ExistsFile(path)) << path;
  std::fstream text_file(path, std::ios::in | std::ios::out);
  CHECK(text_file.is_open()) << path;

  std::string line;
  bool in_vertex_section = false;
  uint64_t ori_num_vertices = 0;
  int begin_pos = text_file.tellg();
  int seek_pos = text_file.tellg();

  int index = 0;
  while (std::getline(text_file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      begin_pos = text_file.tellg();
      continue;
    }

    if (line == "end_header") {
      break;
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      if (line_elems[1] == "vertex") {
        ori_num_vertices = std::stoll(line_elems[2]);
        seek_pos = begin_pos;
        in_vertex_section = true;
      }
    }

    if (!in_vertex_section){
      begin_pos = text_file.tellg();
      continue;
    }
  }
  text_file.seekg(seek_pos, std::ios::beg);
  text_file << "element vertex " << std::setw(20) << ori_num_vertices + points.size() << std::endl;
  text_file.close();

  std::fstream binary_file(path, 
                           std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;

  for (const auto& point : points) {
    WriteBinaryLittleEndian<float>(&binary_file, point.x);
    WriteBinaryLittleEndian<float>(&binary_file, point.y);
    WriteBinaryLittleEndian<float>(&binary_file, point.z);

    WriteBinaryLittleEndian<float>(&binary_file, point.nx);
    WriteBinaryLittleEndian<float>(&binary_file, point.ny);
    WriteBinaryLittleEndian<float>(&binary_file, point.nz);

    uint8_t rgb[3];
    if (point.s_id != -1) {
      rgb[0] = adepallete[point.s_id * 3];
      rgb[1] = adepallete[point.s_id * 3 + 1];
      rgb[2] = adepallete[point.s_id * 3 + 2];
    } else {
      rgb[0] = 255;
      rgb[1] = 255;
      rgb[2] = 255;
    }

    WriteBinaryLittleEndian<uint8_t>(&binary_file, rgb[0]);
    WriteBinaryLittleEndian<uint8_t>(&binary_file, rgb[1]);
    WriteBinaryLittleEndian<uint8_t>(&binary_file, rgb[2]);
  }
  binary_file.close();
}

void ReadPointsVisibility(
    const std::string& path,
    std::vector<std::vector<uint32_t> >& points_visibility) {
    std::fstream file(path, std::ios::in | std::ios::binary);
    CHECK(file.is_open()) << path;

    uint64_t num_point = ReadBinaryLittleEndian<uint64_t>(&file);
    points_visibility.resize(num_point);

    for (auto &visibility : points_visibility) {
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
  CHECK(file.is_open()) << path;

  WriteBinaryLittleEndian<uint64_t>(&file, points_visibility.size());

  for (const auto &visibility : points_visibility) {
    WriteBinaryLittleEndian<uint32_t>(&file, visibility.size());
    for (const auto &image_idx : visibility) {
      WriteBinaryLittleEndian<uint32_t>(&file, image_idx);
    }
  }
}

void AppendWritePointsVisibility(
    const std::string& path,
    const std::vector<std::vector<uint32_t> >& points_visibility) {
  CHECK(ExistsFile(path)) << path;
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  CHECK(file.is_open()) << path;

  uint64_t num_point = ReadBinaryLittleEndian<uint64_t>(&file);
  num_point += points_visibility.size();

  file.seekp(0, std::ios::beg);
  WriteBinaryLittleEndian<uint64_t>(&file, num_point);

  file.seekp(0, std::ios::end);
  for (const auto &visibility : points_visibility) {
    WriteBinaryLittleEndian<uint32_t>(&file, visibility.size());
    for (const auto &image_idx : visibility) {
      WriteBinaryLittleEndian<uint32_t>(&file, image_idx);
    }
  }
}

void ReadPointsWeight(
    const std::string& path,
    std::vector<std::vector<float> > &points_weight) {
    std::fstream file(path, std::ios::in | std::ios::binary);
    CHECK(file.is_open()) << path;

    uint64_t num_point = ReadBinaryLittleEndian<uint64_t>(&file);
    points_weight.resize(num_point);

    for (auto &weight : points_weight) {
        uint32_t num_vis = ReadBinaryLittleEndian<uint32_t>(&file);
        weight.resize(num_vis);
        ReadBinaryLittleEndian<float>(&file, &weight);
    }
    file.close();
}

void WritePointsWeight(
    const std::string& path,
    const std::vector<std::vector<float> >& points_weight) {
  std::fstream file(path, std::ios::out | std::ios::binary);
  CHECK(file.is_open()) << path;

  WriteBinaryLittleEndian<uint64_t>(&file, points_weight.size());

  for (const auto &weight : points_weight) {
    WriteBinaryLittleEndian<uint32_t>(&file, weight.size());
    for (const auto &weight_pnt : weight) {
      WriteBinaryLittleEndian<float>(&file, weight_pnt);
    }
  }
}

void AppendWritePointsWeight(
    const std::string& path,
    const std::vector<std::vector<float> >& points_weight) {
  CHECK(ExistsFile(path)) << path;
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  CHECK(file.is_open()) << path;

  uint64_t num_point = ReadBinaryLittleEndian<uint64_t>(&file);
  num_point += points_weight.size();

  file.seekp(0, std::ios::beg);
  WriteBinaryLittleEndian<uint64_t>(&file, num_point);

  file.seekp(0, std::ios::end);
  for (const auto &weight : points_weight) {
    WriteBinaryLittleEndian<uint32_t>(&file, weight.size());
    for (const auto &weight_pnt : weight) {
      WriteBinaryLittleEndian<float>(&file, weight_pnt);
    }
  }
  file.close();
}

void ReadPointsScore(const std::string& path,
  std::vector<float>& points_score) {
  std::fstream binary_file(path, std::ios::in | std::ios::binary);
  CHECK(binary_file.is_open()) << path;

  int num_point;
  binary_file.read((char *)&num_point, sizeof(uint64_t));

  for (int i = 0; i < num_point; ++i) {
    float score;
    binary_file.read((char*)&score, sizeof(float));
    points_score.emplace_back(score);
  }
  binary_file.close();
}

void WritePointsScore(const std::string& path,
  const std::vector<float>& points_score) {
  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary);
  CHECK(binary_file.is_open()) << path;

  int num_point = points_score.size();
  binary_file.write((const char*)&num_point, sizeof(uint64_t));

  for (const auto sco : points_score) {
    binary_file.write((const char*)&sco, sizeof(float));
  }
  binary_file.close();
}

void AppendWWritePointsScore(const std::string& path,
  const std::vector<float>& points_score) {
  CHECK(ExistsFile(path)) << path;
  std::fstream binary_file(path,
                           std::ios::in | std::ios::out | std::ios::binary);
  CHECK(binary_file.is_open()) << path;

  uint64_t num_point;
  binary_file.read((char *)&num_point, sizeof(uint64_t));
  num_point += points_score.size();

  binary_file.seekp(0, std::ios::beg);
  binary_file.write((const char*)&num_point, sizeof(uint64_t));

  binary_file.seekp(0, std::ios::end);
  for (const auto sco : points_score) {
    binary_file.write((const char*)&sco, sizeof(float));
  }
  binary_file.close();
}

void WriteTextPlyMesh(const std::string& path, const PlyMesh& mesh, 
                      const bool write_as_normal, const bool write_as_rgb) {
  std::fstream file(path, std::ios::out);
  CHECK(file.is_open());

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "element vertex " << mesh.vertices.size() << std::endl;
  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;
  if (write_as_normal){
    file << "property float nx" << std::endl;
    file << "property float ny" << std::endl;
    file << "property float nz" << std::endl;
  }
  if (write_as_rgb){
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
    file << "property uchar alpha" << std::endl;
  }
  file << "element face " << mesh.faces.size() << std::endl;
  file << "property list uchar int vertex_index" << std::endl;
  file << "end_header" << std::endl;

  for (const auto& vertex : mesh.vertices) {
    file << vertex.x << " " << vertex.y << " " << vertex.z;
    if (write_as_normal){
      file << " " << vertex.nx << " " << vertex.ny << " " << vertex.nz;
    }
    if (write_as_rgb){
      file << " " << int(vertex.r) << " " << int(vertex.g) << " " << int(vertex.b) << " " << int(255);
    }
    file << std::endl;
  }

  for (const auto& face : mesh.faces) {
    file << StringPrintf("3 %d %d %d", face.vertex_idx1, face.vertex_idx2,
                         face.vertex_idx3)
         << std::endl;
  }
  file.close();
}

void WriteBinaryPlyMesh(const std::string& path, const PlyMesh& mesh, 
                        const bool write_as_normal, const bool write_as_rgb) {
  std::fstream text_file(path, std::ios::out);
  CHECK(text_file.is_open());

  text_file << "ply" << std::endl;
  text_file << "format binary_little_endian 1.0" << std::endl;
  text_file << "comment VCGLIB generated" << std::endl;
  text_file << "element vertex " << mesh.vertices.size() << std::endl;
  text_file << "property float x" << std::endl;
  text_file << "property float y" << std::endl;
  text_file << "property float z" << std::endl;
  if (write_as_normal){
    text_file << "property float nx" << std::endl;
    text_file << "property float ny" << std::endl;
    text_file << "property float nz" << std::endl;
  }
  if (write_as_rgb){
    text_file << "property uchar red" << std::endl;
    text_file << "property uchar green" << std::endl;
    text_file << "property uchar blue" << std::endl;
    text_file << "property uchar alpha" << std::endl;
  }
  text_file << "element face " << mesh.faces.size() << std::endl;
  text_file << "property list uchar int vertex_index" << std::endl;
  text_file << "end_header" << std::endl;
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;

  for (const auto& vertex : mesh.vertices) {
    WriteBinaryLittleEndian<float>(&binary_file, vertex.x);
    WriteBinaryLittleEndian<float>(&binary_file, vertex.y);
    WriteBinaryLittleEndian<float>(&binary_file, vertex.z);
    if (write_as_normal){
      WriteBinaryLittleEndian<float>(&binary_file, vertex.nx);
      WriteBinaryLittleEndian<float>(&binary_file, vertex.ny);
      WriteBinaryLittleEndian<float>(&binary_file, vertex.nz);
    }
    if (write_as_rgb){
      WriteBinaryLittleEndian<uint8_t>(&binary_file, vertex.r);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, vertex.g);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, vertex.b);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, 225);
    }
  }

  for (const auto& face : mesh.faces) {
    CHECK_LT(face.vertex_idx1, mesh.vertices.size());
    CHECK_LT(face.vertex_idx2, mesh.vertices.size());
    CHECK_LT(face.vertex_idx3, mesh.vertices.size());
    const uint8_t kNumVertices = 3;
    WriteBinaryLittleEndian<uint8_t>(&binary_file, kNumVertices);
    WriteBinaryLittleEndian<int>(&binary_file, face.vertex_idx1);
    WriteBinaryLittleEndian<int>(&binary_file, face.vertex_idx2);
    WriteBinaryLittleEndian<int>(&binary_file, face.vertex_idx3);
  }

  binary_file.close();
}

void ReadBinaryPlyMesh(const std::string& path, PlyMesh& mesh) {

  typedef PlyColorAndValueVertex< float > Vertex;

	bool readFlags[ Vertex::ReadComponents ];
	if(!PlyReadHeader((char *)path.c_str(), PlyColorAndValueVertex<float>::ReadProperties, 
                    PlyColorAndValueVertex<float>::ReadComponents, readFlags)) { 
    fprintf( stderr , "[ERROR] Failed to read ply header: %s\n" , path.c_str());
    exit( 0 );
  }

	bool hasValue = readFlags[3];
	bool hasColor = ( readFlags[4] || readFlags[7] ) && 
                  ( readFlags[5] || readFlags[8] ) && 
                  ( readFlags[6] || readFlags[9] );

  std::vector< Vertex > vertices;
  std::vector< std::vector< int > > polygons;

  int ft , commentNum = 0;
  char** comments;
  PlyReadPolygons((char *)path.c_str() , vertices , polygons , Vertex::ReadProperties , 
                  Vertex::ReadComponents , ft , &comments , &commentNum );
  
  mesh.vertices.resize(vertices.size());
  for (int i = 0; i < vertices.size(); ++i) {
    mesh.vertices[i].x = vertices[i].point[0];
    mesh.vertices[i].y = vertices[i].point[1];
    mesh.vertices[i].z = vertices[i].point[2];
    mesh.vertices[i].r = vertices[i].color[0];
    mesh.vertices[i].g = vertices[i].color[1];
    mesh.vertices[i].b = vertices[i].color[2];
  }

  mesh.faces.resize(polygons.size());
  for (int i = 0; i < polygons.size(); ++i) {
    mesh.faces[i].vertex_idx1 = polygons[i][0];
    mesh.faces[i].vertex_idx2 = polygons[i][1];
    mesh.faces[i].vertex_idx3 = polygons[i][2];
  }
  
}

}  // namespace sensemap
