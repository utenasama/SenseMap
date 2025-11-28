# SenseMap

SenseMap is a large scale Structure-from-Motion(SfM) system.
We are committed to building high-resolution maps and 3D reconstruction of large-scale scenes.


## Getting Started

### Docker 
Dockerfile can be used for easy installation.
Execute the following commands:
```
cd SenseMap
docker build -t sensemap -f Dockerfile .
```

### Dependencies

- c++11
- [CMake](https://cmake.org/) is a cross platform build system.
- [eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) is used extensively for doing nearly all the matrix and linear algebra operations.
- [FreeImage](http://freeimage.sourceforge.net/) is used to read and write image files.
- [Ceres Solver](http://ceres-solver.org/) is a library for solving non-linear least squares problems.
- [glog](https://code.google.com/archive/p/google-glog/) is used for error checking and logging.
- [boost-filesystem](https://www.boost.org/) is used for filesystem operation.
- [opencv](https://github.com/opencv/opencv) is an Open Source Computer Vision Library, just for visualization.
- [GeoTIFF](https://github.com/OSGeo/libgeotiff.git). This library is designed to permit the extraction and parsing of the "GeoTIFF" Key directories, as well as definition and installation of GeoTIFF keys in new files
- [PROJ>=6.0](https://proj.org/index.html) is a generic coordinate transformation software that transforms geospatial coordinates from one coordinate reference system (CRS) to another
- [GDAL](https://github.com/OSGeo/GDAL.git) is an open source MIT licensed translator library for raster and vector geospatial data formats

### Installing

- Dependencies from the default Ubuntu repositories:

```
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-test-dev \
    libboost-graph-dev \
    libsuitesparse-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libfreeimage-dev \
    libjsoncpp-dev \
    libyaml-cpp-dev \
    libsqlite3-dev \
    libtbb-dev \
    libtbb2
```
- Install CUDA Toolkit.
- Install Ceres Solver.
- Install OpenCV.
- Install CGAL 5.0 or later.
  git clone https://github.com/CGAL/cgal/tree/releases/CGAL-5.2
  cmake sensemap_dir/build -DCGAL_DIR="your cgal dir"
- Checkout the latest source code and compile.

### Build GeoTIFF
```
git clone https://github.com/OSGeo/libgeotiff.git
cd libgeotiff/libgeotiff
mkdir build
cd build
cmake ../
make -j
make install
```

### Build PROJ
```
https://proj.org/install.html#compilation-and-installation-from-source-code
git clone https://github.com/OSGeo/PROJ.git
git checkout 9.0.1
cd PROJ
mkdir build
cd build
cmake ..
cmake --build .
cmake --build . --target install
```

### Build GDAL
```
git clone https://github.com/OSGeo/GDAL.git
cd GDAL
mkdir build
cd build
cmake ..
cmake --build .
cmake --build . --target install
```

### Note
如果是gcc9的编译环境，需要安装其他版本的tbb，具体如下：
支持lidar重建的分支需要升级gcc编译器，具体gcc-9, boost1.82.0, cuda12.0
tbb大版本需要安装2020版，具体安装步骤可以参考https://blog.csdn.net/qq_39779233/article/details/126284595
- 下载tbb源码
- 编译
因为要使用 gcc-9 进行编译，所以需要编辑成 gcc-9 形式
cp build/linux.gcc.inc build/linux.gcc-9.inc 
1
编辑 linux.gcc-9.inc 文件：
第15、16行原来是
CPLUS ?= g++
CONLY ?= gcc
修改为
CPLUS ?= g++-9
CONLY ?= gcc-9

然后在文件夹 oneTBB-2020_U3/ 中编译
cd oneTBB-2020_U3
make compiler=gcc-9 stdver=c++17 tbb_build_prefix=my_tbb_build
编译完成后，在 builld/ 文件夹下会看到编译生成的文件夹 my_tbb_build_release/.

- 安装
将 tbb 编译生成的库文件放到对应的 /usr/ 文件夹下：
sudo cp -r oneTBB-2020_U3/include /usr/local/include/tbb

sudo cp -r oneTBB-2020_U3/build/my_tbb_build_release /usr/local/tbb-2020_U3/lib
# 建立新安装tbb版本的符号链接
sudo ln -s /usr/local/tbb-2020_U3/lib/libtbb.so.2 /usr/local/lib/libtbb.so

sudo ln -s /usr/local/tbb-2020_U3/lib/libtbbmalloc.so.2 /usr/local/lib/libtbbmalloc.so

sudo ln -s /usr/local/tbb-2020_U3/lib/libtbbmalloc_proxy.so.2 /usr/local/lib/libtbbmalloc_proxy.so

然后把 库文件的路径写入到 ~/.bashrc ：

echo 'export LD_LIBRARY_PATH=/usr/local/tbb-2020_U3/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
然后再次编译程序。
————————————————

版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
原文链接：https://blog.csdn.net/qq_39779233/article/details/126284595

### Building with OpenMP on OS X
Up to at least Xcode 8, OpenMP support was disabled in Apple’s version of Clang. However, you can install the latest version of the LLVM toolchain from Homebrew which does support OpenMP, and thus build Ceres with OpenMP support on OS X. To do this, you must install llvm via Homebrew:
```
# Install latest version of LLVM toolchain.
brew install llvm
```

As the LLVM formula in Homebrew is keg-only, it will not be installed to /usr/local to avoid conflicts with the standard Apple LLVM toolchain. To build Ceres with the Homebrew LLVM toolchain you should do the following:

```
tar zxf ceres-solver-1.14.0.tar.gz
mkdir ceres-bin
cd ceres-bin
# Configure the local shell only (not persistent) to use the Homebrew LLVM
# toolchain in favour of the default Apple version.  This is taken
# verbatim from the instructions output by Homebrew when installing the
# llvm formula.
export LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
export PATH="/usr/local/opt/llvm/bin:$PATH"
# Force CMake to use the Homebrew version of Clang.  OpenMP will be
# automatically enabled if it is detected that the compiler supports it.
cmake -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++ ../ceres-solver-1.14.0
make -j3
make test
# Optionally install Ceres.  It can also be exported using CMake which
# allows Ceres to be used without requiring installation.  See the
# documentation for the EXPORT_BUILD_DIR option for more information.
make install
```
```
cd SenseMap
cmake -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++ ./
make -j4
```

## Running the tests

### Benchmarks

Download from http://file.intra.sensetime.com/d/9f607954ba/ .

See the [Dataset.md](Dataset.md) file for details.

We list some useful benchmarks and use some of them to evaluate our algorithm.

#### VO Benchmarks

- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)
- [ICL-NUIM](http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)

#### MVS Benchmarks
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [ETH3D](https://www.eth3d.net/)
- [DTU MVS Data Set](http://roboimagedata.compute.dtu.dk/?page_id=36)

#### Other Datasets

- [Internet](http://www.cs.cornell.edu/projects/1dsfm/)







### Break down into end to end tests

TODO

```
Give an example
```

## Coding style

We use [C++ 风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/contents/).
