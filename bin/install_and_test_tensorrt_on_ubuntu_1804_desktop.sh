#!/bin/bash

################################################################################
# Tensorrt installation and test Starting from Ubuntu 18.04 on a desktop
################################################################################

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


###############################################################################
# check nvidia drivers
###############################################################################
echo "======== Checking nvidia drivers"
sudo apt-get -qq update
sudo apt-get -qq -y install ubuntu-drivers-common
ubuntu-drivers devices
nvidia-smi

###############################################################################
# Install various prerequesites
###############################################################################

sudo apt-get install -y python3-pip git

################################################################################
# Install Cuda 10.2
#
#     The following seems to work, even though it contradicts the
#     nvidia documentation for cuda on Linux.
#
#     It was found on the Cuda-for-Ubuntu-under-WSL pages on nvidia's
#     site.  https://docs.nvidia.com/cuda/wsl-user-guide/index.html
# 
#     Without it, the command "sudo apt-get install tensorrt" was giving me errors like
#     "tensorrt : Depends: libnvinfer7 (= 7.0.0-1+cuda10.2) but it is not going to be installed".
#
#   Do not: 
#     sudo apt install nvidia-cuda-toolkit
#   on an out-of-the-box Ubuntu-18.04 and hope to get DeepStream 5 working.
#   It will cause dependency hell.
#
################################################################################

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list'
sudo apt-get -qq update
sudo apt-get -qq -y install cuda-toolkit-10-2

if ! grep -q 'export PATH.*cuda' $HOME/.bashrc; then
   echo "Adding cuda to path.  You will need to source $HOME/.bashrc"
   echo 'export PATH=$PATH:/usr/local/cuda-10.2/bin' >> $HOME/.bashrc
fi

################################################################################
# check cuda compiler
################################################################################
echo "======== Checking cuda"
nvcc --version


################################################################################
# Test compiling and running cuda samples
################################################################################
mkdir ~/cuda_samples
rsync -aP /usr/local/cuda/samples/. ~/cuda_samples

cd ~/cuda_samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
################################################################################
#
# ./deviceQuery Starting...
# 
#  CUDA Device Query (Runtime API) version (CUDART static linking)
# 
# Detected 1 CUDA Capable device(s)
# 
# Device 0: "GeForce RTX 2060"
#   CUDA Driver Version / Runtime Version          10.2 / 10.2
#   CUDA Capability Major/Minor version number:    7.5
#   Total amount of global memory:                 5931 MBytes (6219563008 bytes)
#   (30) Multiprocessors, ( 64) CUDA Cores/MP:     1920 CUDA Cores
#   GPU Max Clock rate:                            1680 MHz (1.68 GHz)
#   Memory Clock rate:                             7001 Mhz
#   Memory Bus Width:                              192-bit
#   L2 Cache Size:                                 3145728 bytes
#   Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
#   Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
#   Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
#   Total amount of constant memory:               65536 bytes
#   Total amount of shared memory per block:       49152 bytes
#   Total number of registers available per block: 65536
#   Warp size:                                     32
#   Maximum number of threads per multiprocessor:  1024
#   Maximum number of threads per block:           1024
#   Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
#   Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
#   Maximum memory pitch:                          2147483647 bytes
#   Texture alignment:                             512 bytes
#   Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
#   Run time limit on kernels:                     Yes
#   Integrated GPU sharing Host Memory:            No
#   Support host page-locked memory mapping:       Yes
#   Alignment requirement for Surfaces:            Yes
#   Device has ECC support:                        Disabled
#   Device supports Unified Addressing (UVA):      Yes
#   Device supports Compute Preemption:            Yes
#   Supports Cooperative Kernel Launch:            Yes
#   Supports MultiDevice Co-op Kernel Launch:      Yes
#   Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
#   Compute Mode:
#      < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
# 
# deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.2, CUDA Runtime Version = 10.2, NumDevs = 1
# Result = PASS
# 
# 
################################################################################



################################################################################
# Now try this python program from this pycuda tutorial
#
#  https://documen.tician.de/pycuda/tutorial.html
################################################################################
echo "======== Checking pycuda"
cd

pip3 install --quiet 'pycuda>=2019.1.1'

cat > /tmp/test_pycuda.py <<EOL

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

python3 /tmp/test_pycuda.py


################################################################################
# test tensorflow
#
# https://www.tensorflow.org/overview
#
#
# The current version of tensorflow works in the Amazon cloud - but fails with
#  'Illegal instruction (core dumped)"
# on my desktop with an older CPU.
#  https://stackoverflow.com/questions/49094597/illegal-instruction-core-dumped-after-running-import-tensorflow
#  https://tech.amikelive.com/node-887/how-to-resolve-error-illegal-instruction-core-dumped-when-running-import-tensorflow-in-a-python-program/
#
# Workaround using tensorflow 1.5
#
################################################################################
echo "======== checking tensorflow from python"
cd 

# pip3 install tensorflow
pip3 install tensorflow==1.5

cat > /tmp/test_tensorflow.py <<EOL
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

python3 /tmp/test_tensorflow.py


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

./deepstream-test1-app /opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264


#
sudo apt-get install -y libgstrtspserver-1.0-dev
cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-test2
make
./deepstream-test2-app  /opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264
#

#
cd /opt/nvidia/deepstream/deepstream-5.0/bin
make
./deepstream-opencv-test file:///opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264
./deepstream-app -c /opt/nvidia/deepstream/deepstream-5.0/samples/configs/deepstream-app/source1_usb_dec_infer_resnet_int8.txt 

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
sudo apt install libgirepository1.0-dev
sudo apt install libcairo2-dev
sudo apt-get -qq -y install ubuntu-restricted-extras
sudo apt-get install python-gi-dev

conda install -c conda-forge pygobject
pip3 install gobject PyGObject
pip3 install pyds
cd ~/deepstream_python_apps/apps/deepstream-test1
python3 deepstream_test_1.py  /opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264


################################################################################
################################################################################
################################################################################

ron@ron-whitebox:/opt/nvidia/deepstream/deepstream-5.0/bin$ ./deepstream-app -c /opt/nvidia/deepstream/deepstream-5.0/samples/configs/deepstream-app/config_infer_primary.txt 

(deepstream-app:21131): GStreamer-CRITICAL **: 23:58:31.112: gst_element_get_static_pad: assertion 'GST_IS_ELEMENT (element)' failed
Segmentation fault (core dumped)




# kinda works

gst-launch-1.0 videotestsrc ! nveglglessink

gst-launch-1.0 filesrc location=/tmp/1.h264 ! h264parse ! nvv4l2decoder ! nveglglessink

# These mostly work
# 
# https://gist.github.com/strezh/9114204

gst-launch-1.0 v4l2src !     'video/x-raw, width=640, height=480, framerate=30/1' !     videoconvert !     x264enc pass=qual quantizer=20 tune=zerolatency !     rtph264pay !     udpsink host=127.0.0.1 port=1234

gst-launch-1.0 udpsrc port=1234 !     "application/x-rtp, payload=127" !     rtph264depay !     avdec_h264 !     videoconvert  !     xvimagesink 

#
# These mostly fail
#
# https://docs.nvidia.com/metropolis/deepstream/plugin-manual/index.html#page/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_faq.html#
#
#  dlopen error: /opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_mot_klt.so: cannot open shared object file: No such file or directory


# This works:

cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-dewarper-test
gst-launch-1.0 uridecodebin uri= file://`pwd`/../../../../samples/streams/sample_cam6.mp4 ! nvvideoconvert ! nvdewarper source-id=6 num-output-buffers=4 config-file=config_dewarper.txt ! m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=4 batched-push-timeout=100000 num-surfaces-per-frame=4 ! nvmultistreamtiler rows=1 columns=1 width=720 height=576 ! nvvideoconvert ! nveglglessink



cd /opt/nvidia/deepstream/deepstream-5.0/samples/
gst-launch-1.0 filesrc location = ./streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=1  ! nvinfer config-file-path= ./configs/deepstream-app/config_infer_primary.txt ! dsexample full-frame=1 ! nvvideoconvert ! nvdsosd ! nveglglessink sync=0
