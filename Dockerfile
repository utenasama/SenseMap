FROM nvidia/cuda:9.0-devel-ubuntu16.04
MAINTAINER Mingxuan Jiang

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Update Source
RUN echo "deb https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse\n"\
"deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse\n"\
"deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\n"\
"deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\n"\
"deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\n"\
"deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\n"\
"deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse\n"\
"deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse\n"\
"deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse\n"\
"deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse\n\n"\
"deb http://mirrors.163.com/ubuntu/ bionic main restricted universe multiverse\n"\
"deb http://mirrors.163.com/ubuntu/ bionic-security main restricted universe multiverse\n"\
"deb http://mirrors.163.com/ubuntu/ bionic-updates main restricted universe multiverse\n"\
"deb http://mirrors.163.com/ubuntu/ bionic-proposed main restricted universe multiverse\n"\
"deb http://mirrors.163.com/ubuntu/ bionic-backports main restricted universe multiverse\n"\
"deb-src http://mirrors.163.com/ubuntu/ bionic main restricted universe multiverse\n"\
"deb-src http://mirrors.163.com/ubuntu/ bionic-security main restricted universe multiverse\n"\
"deb-src http://mirrors.163.com/ubuntu/ bionic-updates main restricted universe multiverse\n"\
"deb-src http://mirrors.163.com/ubuntu/ bionic-proposed main restricted universe multiverse\n"\
"deb-src http://mirrors.163.com/ubuntu/ bionic-backports main restricted universe multiverse\n\n"\
"deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse\n"\
"deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse\n"\ 
"deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\n"\
"deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\n"\
"deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\n"\
"deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\n"\
"deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse\n"\
"deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse\n"\
"deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse\n"\
"deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse\n"\
"deb http://cn.archive.ubuntu.com/ubuntu/ xenial main restricted\n"\
"deb http://cn.archive.ubuntu.com/ubuntu/ xenial-updates main restricted\n"\
"deb http://cn.archive.ubuntu.com/ubuntu/ xenial universe\n"\
"deb http://cn.archive.ubuntu.com/ubuntu/ xenial-updates universe\n"\
"deb http://cn.archive.ubuntu.com/ubuntu/ xenial multiverse\n"\
"deb http://cn.archive.ubuntu.com/ubuntu/ xenial-updates multiverse\n"\
"deb http://cn.archive.ubuntu.com/ubuntu/ xenial-backports main restricted universe multiverse\n"\
"deb http://security.ubuntu.com/ubuntu xenial-security main restricted\n"\
"deb http://security.ubuntu.com/ubuntu xenial-security universe\n"\
"deb http://security.ubuntu.com/ubuntu xenial-security universe\n"\
> /etc/apt/sources.list

# RUN cat /etc/apt/sources.list

# Install the basic packages
RUN set -x && \
  apt-get update -y -qq && \
  : "basic dependencies" && \
  apt-get install -y -qq \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    curl \
    tar \
    unzip \
    vim

# glog
RUN apt-get install -y \
    autoconf \
    automake \
    libtool

# OpenBLAS
RUN apt-get install -y \ 
    gfortran


# gflag
WORKDIR /tmp
RUN git clone https://github.com/gflags/gflags && \
    cd gflags && \
    cmake . -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON && \
    make -j && \
    make install && \
    cd /tmp && \
    rm -rf *


# glog
WORKDIR /tmp
RUN git clone https://github.com/google/glog && \
    cd glog && \
    ./autogen.sh && \
    ./configure && \
    make -j && \
    make install && \
    cd /tmp && \
    rm -rf *

# Boost
WORKDIR /tmp
RUN wget -cq http://file.intra.sensetime.com/d/93e9dc5985/files/\?p\=/boost_1_68_0.tar.gz\&dl\=1 -O boost_1_68_0.tar.gz && \
    tar -xzf boost_1_68_0.tar.gz && \
    cd boost_1_68_0 && \
    ./bootstrap.sh && \
    ./b2 && \
    ./b2 install && \
    cd /tmp && \
    rm -rf *

# FreeImage
WORKDIR /tmp
RUN wget -cq http://file.intra.sensetime.com/d/93e9dc5985/files/\?p\=/FreeImage3180.zip\&dl\=1 -O FreeImage3180.zip && \
    unzip -q FreeImage3180.zip && \
    rm -rf FreeImage3180.zip && \
    cd FreeImage && \
    make -j && \
    make install && \
    cd /tmp && \
    rm -rf *

# OpenBLAS
WORKDIR /tmp
RUN wget -cq http://file.intra.sensetime.com/d/93e9dc5985/files/\?p\=/OpenBLAS.tar.gz\&dl\=1 -O OpenBLAS.tar.gz && \
    tar -xzf OpenBLAS.tar.gz && \
    cd OpenBLAS && \
    make FC=gfortran && \
    make install && \
    cd /tmp && \
    rm -rf *

# Eigen
ARG EIGEN3_VERSION=3.3.6
WORKDIR /tmp
RUN set -x && \
    wget -q http://bitbucket.org/eigen/eigen/get/${EIGEN3_VERSION}.tar.bz2 && \
    tar xf ${EIGEN3_VERSION}.tar.bz2 && \
    mv eigen-eigen-* eigen-${EIGEN3_VERSION} && \
    cd eigen-${EIGEN3_VERSION} && \
    mkdir -p build && \
    cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        .. && \
    make -j && \
    make install && \
    cd /tmp && \
    rm -rf *

# Ceres    
WORKDIR /tmp
RUN git clone https://github.com/ceres-solver/ceres-solver.git && \
    cd ceres-solver && \
    mkdir build  && cd build && \ 
    cmake .. && make -j && \
    make install && \
    cd /tmp && \
    rm -rf *

# Opencv and SenseMap Denpendence
RUN apt-get install -y \
    gcc-4.9 \
    g++-4.9 \
    libsuitesparse-dev \
    libglew-dev

# TODO: Complie gcc 5.4



# Change gcc link
RUN rm -r /usr/bin/gcc && \
    ln -sf /usr/bin/gcc-5 /usr/bin/gcc && \
    rm -r /usr/bin/g++ && \
    ln -sf /usr/bin/g++-5 /usr/bin/g++ 

# OpenCV
WORKDIR /tmp
RUN wget -cq http://file.intra.sensetime.com/f/ce5839de70/\?raw\=1 -O OpenCV-3.4.5.tar.gz&& \
    tar xf OpenCV-3.4.5.tar.gz && \
    cd OpenCV-3.4.5 && cd opencv-3.4.5 && \
    mkdir build && cd build && \
    cmake \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-5 \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        # -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.5/modules \
        -DWITH_CUDA=ON \
        -DWITH_CUBLAS=ON \
        -DDCUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
        -DENABLE_CXX11=ON \
        # -DOPENCV_ENABLE_NONFREE=ON \
        .. && \
    make -j && \
    make install && \
    cd /tmp && \
    rm -rf *    

# Switch workdir back to /
WORKDIR /

ENTRYPOINT ["/bin/bash"]