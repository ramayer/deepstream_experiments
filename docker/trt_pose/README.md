
Download model weights from here https://github.com/NVIDIA-AI-IOT/trt_pose

Build the image with:


```
mkdir -p modelweights
cp -a /home/ron/jupyter/trt_pose/tasks/human_pose/*.pth modelweights
docker build -t trt-pose-x86:0.1 .
```

Run the image with

```
docker run -p 0.0.0.0:8888:8888/TCP --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -e PUID="0" -e PGID="0" trt-pose-x86:0.1
```

Navigate to 

```
http://127.0.0.1:8888/lab/tree/trt_pose/tasks/human_pose/trt_pose_demo.ipynb
    
```

