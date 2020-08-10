FROM nvcr.io/nvidia/tensorflow:20.06-tf2-py3
RUN pip3 install jupyter
EXPOSE 8888
RUN pip3 install keras
RUN pip3 install tqdm cython pycocotools torch matplotlib
RUN git clone https://github.com/NVIDIA-AI-IOT/trt_pose
RUN cd trt_pose && python3 setup.py install
# note - you need to download the modelweights as cescribed here: https://github.com/NVIDIA-AI-IOT/trt_pose
ADD modelweights /workspace/trt_pose/tasks/human_pose
RUN pip3 install torchvision
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN cd torch2trt && python3 setup.py install
RUN apt-get update
RUN apt-get -y install libsm6 libxext6 libxrender-dev
RUN apt-get -y install libgl1-mesa-glx libgl1-mesa-dev
RUN pip3 install opencv-python
RUN curl 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/1991-Silje_Studio_11%C3%A5r_cropped.jpg/220px-1991-Silje_Studio_11%C3%A5r_cropped.jpg' > /workspace/dance.jpg
RUN curl 'https://upload.wikimedia.org/wikipedia/commons/7/72/Mia1997.JPG' > /workspace/soccer.jpg

# at this point the following should work:
#  (cd /workspace/trt_pose/tasks/human_pose/ ; jupyter-lab)&
