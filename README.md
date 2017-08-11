## yolov2_tensorflow

### Requirements
1. Tensorflow
2. OpenCV

Tensorflow implementation of [YOLO](https://pjreddie.com/darknet/yolo/), including yolov1 and yolov2 demo.


### Installation

1. Clone yolov2_tensorflow repository
	```Shell
	$ git clone https://github.com/shishichang/yolov2-tensorflow.git
    $ cd yolov2_tensorflow
	```

2. Download [YOLO_v1](http://pan.baidu.com/s/1cGV694) [YOLO_v2_pb](http://pan.baidu.com/s/1hrRszrA) [YOLO_v2_meta](http://pan.baidu.com/s/1dEOaGPr) 
   put it in `models`

3. Modify configuration in `yolo/config.py` for yolov1

4. Run
	```Shell
	$ python demo_yolo_v1.py
	$ python demo_yolo_v2.py
	```
