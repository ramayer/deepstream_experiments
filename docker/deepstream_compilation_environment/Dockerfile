###############################################################################
#
# NVidia's nvcr.io/nvidia/deepstream:5.0-dp-20.04-triton doesn't
# automatically come with the dev version of gstreamer plugins required
# to complie their demos.
#
# Build this docker image with
#
#     docker build . -t deepstream-compilation-environment:0.1
#
# Run with
#
#     docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -e PUID="0" -e PGID="0" deepstream-compilation-environment:0.1
#

# Start with Nvidia's deepstream runtime

FROM nvcr.io/nvidia/deepstream:5.0-dp-20.04-triton

# Install libraries required for compiling their demos

RUN apt-get update
RUN apt-get install -y \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libgstreamer1.0-0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libssl1.0.0 \
    pkg-config
    
# Test compiling and running a sample.

RUN cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-test1; export CFLAGS=`pkg-config --cflags gstreamer-1.0`; make

ENTRYPOINT ["/bin/bash","-c","cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-test1; ./deepstream-test1-app /opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264"]




