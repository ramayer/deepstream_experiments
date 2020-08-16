################################################################################
# Based on a combination of:
#
#        https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
#
# and NVidia's 
#
#        nvcr.io/nvidia/pytorch:20.07-py3
#    
# and
#
#        https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile
#
################################################################################
#
#  Run with:
#
#     docker build . -t detectron2-test:0.1
#
#     docker run --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e PUID="0" -e PGID="0"  detectron2-test:latest
#
# or
#
#     docker run -p 0.0.0.0:8888:8888/TCP --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -e PUID="0" -e PGID="0" detectron2-test:0.1
#
################################################################################


FROM nvcr.io/nvidia/pytorch:20.07-py3

ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_CUDA="1"

RUN pip3 install jupyter
EXPOSE 8888

RUN apt-get update
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN pip install --user -e detectron2

# runs a model
ENTRYPOINT []
CMD sh -c "cd detectron2; python3 demo/demo.py --webcam --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

