FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
	git \
	pkg-config \
	libopencv-dev \
	python3-pip python3-dev libsm6 libxext6 libxrender-dev \
	wget \
	nano vim

RUN mkdir src
WORKDIR src
RUN git clone https://github.com/AlexeyAB/darknet
WORKDIR darknet
RUN sed -i 's/GPU=0/GPU=1/; s/CUDNN=0/CUDNN=1/; s/OPENCV=0/OPENCV=1/' Makefile
RUN make

RUN mkdir weights
WORKDIR weights

RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

WORKDIR ../..
COPY . .
COPY yolo-obj.cfg darknet/cfg
COPY scripts/run_demo.py darknet/

RUN pip3 install requirements.txt
