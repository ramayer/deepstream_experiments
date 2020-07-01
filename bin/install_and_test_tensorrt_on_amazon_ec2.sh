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
	 install_and_test_tensorrt_on_amazon_ec2.sh --tensorrt-url http://example.com/tmp/cuda10.2-tensorrt7.0.tar
"

tensorrt_url=''
while [[ "$#" -gt 0 ]]; do
    echo "here $1"
    case $1 in
        -t|--tensorrt-url) tensorrt_url="$2"; shift ;;
        -h|--help) echo 'requires --tensorrt-url parameter' ;;
        *) echo "$usage"; exit 1 ;;
    esac
    shift
done

if ["$tensorrt_url" == ""]; then
    echo "$usage"
    exit
fi

################################################################################
# Choose cuda:
#    https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html
################################################################################

sudo rm /usr/local/cuda; sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda

################################################################################
# check cuda
################################################################################

nvcc --version
nvidia-smi

cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery



###############################################################################
# check nvidia drivers
###############################################################################
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
nvcc --version

################################################################################
# Now try this python program from this pycuda tutorial
#
#  https://documen.tician.de/pycuda/tutorial.html
################################################################################

pip install 'pycuda>=2019.1.1'

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

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get install -y cuda-toolkit-10-2


################################################################################
# Download and install tensorrt
#
# That tar file is just a copy of 
#
# wget 'https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/7.0/7.0.0.11/local_repo/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb?Hfy3BwfVFWD4b5HmPkNSalQCmKHmBCE74vg8vlA_uHhpvmGiA6xjvvR2_aPGbzhjCGPLX0HEGHgV4xJFr-jKh5Qe7K-FeW7xhFJfUYb6IdGEi1yp2DCtNH4F4vENp5bs_4YmLlIzyvIg8qU1bRKuNqlFiQbm2Av7AwXtJ38QD85zKHE301T32By2Q47BzuBI7-NY2TgaSwIf4fqRGq2wMz9PY-Jejp3LBkcnjDlr69hcrEEJmRoC7L1xQSBdelmiLENZi3ZHp0fsDw'
#
# that doesn't require filling out the damn survey each time.
################################################################################
cd
wget $tensorrt_url
tar xvf cuda10.2-tensorrt7.0.tar
sudo dpkg -i ./nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-cuda10.2-trt7.0.0.11-ga-20191216/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y tensorrt
sudo apt-get install -y uff-converter-tf

################################################################################
# Compile and Run the tensorrt samples
################################################################################
cd /usr/src/tensorrt/samples/sampleMNIST
sudo make 
cd /usr/src/tensorrt/data/mnist
pip3 install Pillow
sudo ./download_pgms.py
cd /usr/src/tensorrt
./bin/sample_mnist
