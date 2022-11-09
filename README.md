# Object Detection Guide

### Run in Docker environment

Build a docker image
```sh
$ docker build -t darknet-yolo-4 .
```

Run the docker container and, optionally, connect volumes with training data or weightsf
```sh
$ docker run -it \
    --name darknet-yolo-4 \
    -v path/to/training/data/:/src/darknet/data \
    --rm darknet-yolo-4 /bin/bash
```

### Run in local environment

Check Dockerfile and repeat the steps:
- download darknet
- change compilation params (no need if you don't have GPU, CUDA, OpenCV)
- compile darknet
- download and store YOLO weights
- setup config file for YOLO model inside darknet/cfg
- install requirements

### Prepare data

Convert annotations to YOLO format
```sh
$ python convert_annotations.py \
    -d images \
    -a data.csv
```

Crop dataset to smaller pieces
```sh
$ python crop_dataset.py \
    -d images \
    -s cropped \
    -x 3 \
    -y 2 
```

Split dataset into train/test subsets
```sh
$ python prepare_dataset.py \
    -src cropped \
    -dst darknet/data \
    -ts 0.7
```

### Train and evaluate [YOLOv4](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

Configure your model `darknet/cfg/yolo-obj.cfg`

Your dataset have to be stored here `darknet/data`

Run training
```sh
$ cd darknet
$ ./darknet detector train data/obj.data cfg/yolo-obj.cfg weights/yolov4.conv.137 -dont_show -map
```

Run metric calculation on the validation subset
```sh
$ ./darknet detector map data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_best.weights -iou_thresh 0.25
```

You can check model performance on a certain image
```sh
$ ./darknet detector test cfg/yolo-obj.cfg backup/yolo-obj_best.weights data/obj/CAM2_14_2020-09-29T15_15_42+00_00_0_0.jpeg
```
or on a whole dataset
```sh
$ python run_demo.py \
    -w backup/yolo-obj_best.weights \
    -s ../demo \
    -td data/valid.txt 
```

Run predictions and save them in `darknet/result.txt`
```sh
$ ./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_best.weights -dont_show -ext_output < data/valid.txt > data/result.txt
```

Convert predictions to the YOLO annotation format. Add `-j` flag to join pieces.
```sh
$ python parse_results.py \
    -d result.txt \
    -s result 
```

Calculate metrics and find bad predictions
```sh
$ python calculate_metrics.py \
    -gt darknet/data/obj \
    -p result \
    -iou 0.25 \
    -bad 0.2
```
