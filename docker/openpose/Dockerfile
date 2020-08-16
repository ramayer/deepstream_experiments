# Openpose demo
#
# Based on https://hub.docker.com/r/cwaffles/openpose ; but doesn't seem to work anymore.
#
# Upgraded for 2019 based on:
#
#  https://outline.com/VazH8S
#
#
# Errors out with: 
#   CMake Error at cmake/Cuda.cmake:263 (message):
#   cuDNN version >3 is required.
# but 
#   libcudnn8-dev is already the newest version (8.0.1.13-1+cuda11.0).


FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
   autoconf automake libtool curl make g++ unzip

RUN pip3 install numpy opencv-python 

#replace cmake as old version has CUDA variable bugs
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

#get openpose
WORKDIR /openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

#build it
WORKDIR /openpose/build
RUN cmake -DBUILD_PYTHON=ON .. && make -j `nproc`
WORKDIR /openpose
