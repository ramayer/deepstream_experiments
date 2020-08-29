

* Build with

`docker build . -t ron-pytorch-training:0.2`

* Launch with

```
docker run -p 0.0.0.0:8888:8888/TCP --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -e PUID="0" -e PGID="0" ron-pytorch-training:0.2

```

* Run 

`jupyter lab`


* Open

`http://127.0.0.1:8888/lab/tree/notebooks/HandGestureRecognition.ipynb`

It should recognize hand gestures.
