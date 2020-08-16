# Detectron2 compiled on Nvidia's  nvcr.io/nvidia/pytorch:20.07-py3 cuda-11 ontainer


### To use it, try

```
     docker build . -t detectron-pose-test:0.1
```

and run it witn 

```
     docker run --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -e PUID="0" -e PGID="0" detectron-pose-test:0.1
```
