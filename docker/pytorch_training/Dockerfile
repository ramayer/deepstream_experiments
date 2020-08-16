# A mashup of Python's
#
#    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#
#  And NVidia's 
#
#    nvcr.io/nvidia/pytorch:20.07-py3
#
#   Run with:
#
#     docker build . -t ron-pytorch-training:0.1
#
#     docker run --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e PUID="0" -e PGID="0"  ron-pytorch-test:latest
#
# or
#
#     docker run -p 0.0.0.0:8888:8888/TCP --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -e PUID="0" -e PGID="0"  ron-pytorch-test:latest
#

FROM nvcr.io/nvidia/pytorch:20.07-py3
RUN pip3 install jupyter
EXPOSE 8888

# Done inside the notebook 
# RUN git clone https://github.com/akmayer/RockPaperScissors

COPY notebooks/ notebooks/

