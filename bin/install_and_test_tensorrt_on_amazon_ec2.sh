#!/bin/bash
################################################################################
# Tensorrt installation and test Starting from:
#
#    Deep Learning AMI (Ubuntu 18.04) Version 30.0 (ami-029510cec6d69f121)
#
################################################################################


#!/bin/bash

usage="
Usage:
	 install_and_test_tensorrt_on_amazon_ec2.sh --download-prefix http://example.com/tmp
"

download_prefix=''
while [[ "$#" -gt 0 ]]; do
    echo "here $1"
    case $1 in
        -d|--download-prefix) download_prefix="$2"; shift ;;
        -h|--help) echo 'requires --tensorrt-url parameter' ;;
        *) echo "$usage"; exit 1 ;;
    esac
    shift
done

if [ "$download_prefix" == "" ]; then
    echo "$usage"
    exit
fi

tensorrt_url=$download_prefix'/cuda10.2-tensorrt7.0.tar'
deepstream_url=$download_prefix'/deepstream-5.0_5.0.0-1_amd64.deb'

echo "Will download TensorRT from" $tensorrt_url
echo "Will download Deepstream from" $deepstream_url


################################################################################
# Choose cuda:
#    https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html
################################################################################

echo "======== choosing cuda-10.2"
sudo rm /usr/local/cuda; sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda

###############################################################################
# check nvidia drivers
###############################################################################
echo "======== Checking nvidia drivers"
sudo apt-get -qq update
sudo apt-get -qq -y install ubuntu-drivers-common
ubuntu-drivers devices

################################################################################
# check cuda
################################################################################
echo "======== Checking cuda"
nvcc --version
nvidia-smi

cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

################################################################################
# Now try this python program from this pycuda tutorial
#
#  https://documen.tician.de/pycuda/tutorial.html
################################################################################
echo "======== Checking pycuda"
cd

pip install --quiet 'pycuda>=2019.1.1'

cat > test_pycuda.py <<EOL

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy
a = numpy.random.randn(4,4)
a = a.astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)


func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))
a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
EOL

python test_pycuda.py


################################################################################
# test tensorflow
#
# https://www.tensorflow.org/overview
################################################################################
echo "======== checking tensorflow from python"
cd 

pip install tensorflow

cat > test_tensorflow.py <<EOL
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
EOL

python test_tensorflow.py


################################################################################
# TensorRT seems to have missing dependencies 
# Alan got the dependencies to work by following the Windows instructions here: 
# https://docs.nvidia.com/cuda/wsl-user-guide/index.html
#
# 
# Without it, the command "sudo apt-get install tensorrt" was giving me errors like
#  "tensorrt : Depends: libnvinfer7 (= 7.0.0-1+cuda10.2) but it is not going to be installed".
# 
################################################################################
echo "======== Installing more TensorRT dependencies"
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list'
sudo apt-get -qq update
sudo apt-get -qq -y install cuda-toolkit-10-2


################################################################################
# Download and install tensorrt
#
# That tar file is just a copy of 
#
# wget 'https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/7.0/7.0.0.11/local_repo/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb?Hfy3BwfVFWD4b5HmPkNSalQCmKHmBCE74vg8vlA_uHhpvmGiA6xjvvR2_aPGbzhjCGPLX0HEGHgV4xJFr-jKh5Qe7K-FeW7xhFJfUYb6IdGEi1yp2DCtNH4F4vENp5bs_4YmLlIzyvIg8qU1bRKuNqlFiQbm2Av7AwXtJ38QD85zKHE301T32By2Q47BzuBI7-NY2TgaSwIf4fqRGq2wMz9PY-Jejp3LBkcnjDlr69hcrEEJmRoC7L1xQSBdelmiLENZi3ZHp0fsDw'
#
# that doesn't require filling out the damn survey each time.
################################################################################
echo "======== Installing TensorRT"
cd
wget -N $tensorrt_url
tar xvf cuda10.2-tensorrt7.0.tar
sudo dpkg -i ./nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-cuda10.2-trt7.0.0.11-ga-20191216/7fa2af80.pub
sudo apt-get -qq update
sudo apt-get -qq -y install tensorrt uff-converter-tf

################################################################################
# Compile and Run the tensorrt samples
################################################################################
echo "======== Testing TensorRT"
cd /usr/src/tensorrt/samples/sampleMNIST
sudo make 
cd /usr/src/tensorrt/data/mnist
pip3 install Pillow
sudo ./download_pgms.py
cd /usr/src/tensorrt
./bin/sample_mnist

################################################################################
# Install gstreamer
################################################################################
echo "======== Installing gstreamer"
sudo apt-get -qq -y install \
     libssl1.0.0 \
     libgstreamer1.0-0 \
     gstreamer1.0-tools \
     gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-libav \
     libgstrtspserver-1.0-0 \
     libjansson4

################################################################################
# Install libkafka (and related tools)
################################################################################
echo "======== Installing libkafka"
sudo apt-get -qq -y install librdkafka1 librdkafka-dev librdkafka++1 python3-confluent-kafka


################################################################################
# download the deepstream .deb
################################################################################
#
# wget 'https://developer.download.nvidia.com/assets/Deepstream/DeepStream_5.0/deepstream-5.0_5.0.0-1_amd64.deb?drgRRH6Ed8lLpGWnIUtsGQh9M2ucowfWHqGrlwFs-cPewdVrK-zlwKtuGK2_IYGjpcBTXp8wCZU-Wc7E3JD6mClBELkoSGYFqtDsxwbFUngRIThjUXYRSFA8HKMllw4zZTjrEWIEOb-VFLTXAvcLNvcnAM_kc3BUpjey6_Zma7c'

echo "======== Installing DeepStream"
cd
wget -N $deepstream_url
sudo apt-get -qq -y install ./deepstream-5.0_5.0.0-1_amd64.deb

################################################################################
# download the deepstream .deb
################################################################################
echo "======== Testing DeepStream"
cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-test1
make

./deepstream-test1-app /opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_1080p_h264.mp4


################################################################################
################################################################################
# Attempt the python gstreamer wrapper.
#
# The part below here is not yet working.
# Python can't find the gstreamer plugins.
################################################################################
################################################################################

exit
echo "======== Testing Python gstreamer"
cd
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
cd deepstream_python_apps/

# gives an error on 'import gi'.  
# https://askubuntu.com/questions/1057832/how-to-install-gi-for-anaconda-python3-6
#
# sudo apt-get install ubuntu-restricted-extras
sudo apt-get -qq -y install ubuntu-restricted-extras
conda install -c conda-forge pygobject
pip install gobject PyGObject
pip install pyds
cd ~/deepstream_python_apps/apps/deepstream-test1
python deepstream_test_1.py  /opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264
