# ObjectDetection

Object detection and classification. Based on the following papers: [YOLO](https://arxiv.org/pdf/1506.02640.pdf), [YOLOv2](https://arxiv.org/pdf/1612.08242.pdf)
## Results from my own trained from scratch model:
Weights can be found in /weights/mobilenetyolov2 or [here](https://puu.sh/F1XJe/190d024b50)
<br/>
Config can be found in /cfg/mobilenetyolov2.cfg or [here](https://pastebin.com/681E3JHg)

![Car](https://i.imgur.com/UVfHpFM.jpg) <br/>
![Person and plant](https://i.imgur.com/cqHkrGz.jpg) <br/>
![Aeroplane](https://i.imgur.com/Y4Vl36d.jpg) <br/>
![Bird](https://i.imgur.com/54tXl74.jpg) <br/>
![Train](https://i.imgur.com/caZiw7X.jpg) <br/>

### Tested on the following platform
* Windows 10.0.18362
* Nvidia GeForce GTX 1080 Ti
* CUDA 10.0.130
* CuDNN 7.6.0
* Anaconda 3.17.8
* Tensorflow 1.14.0
* Python 3.7.3

#### Dependencies
* Python
* Tensorflow
* PIL
* numpy
* pyclustering
* argparse
* scipy

# Usage
## Labels
You need to specify the labels that your model will be predicting. This is done by simply listing them in a .txt file like so: <br/>
```
person
car
bicycle
train
motorbike
```
An example labels.txt is provided.
## Config
You need to provide a config file for your model. The config contains all the parameters for the model like so:
```
[out]
image_width = 416
image_height = 416
boxes = 5
classes = 20

[loss]
object_scale = 200
noobject_scale = 1
class_scale = 200
coord_scale = 200
# use batch_size 1 if only testing and larger value if training
batch_size = 64

[decode]
anchors = 1.3, 2.12, 2.71, 4.77, 4.39, 9.0, 7.61, 5.21, 10.15, 10.45
threshhold = 0.1
nms_threshhold = 0.35

[optimizer]
name=Adam
learning_rate=0.00005
beta_1=0.9
beta_2=0.999
epsilon=0.00000001
decay = 0.0

[net]
predefined = mobilenet
```
An example config can be found in /cfg/mobilenetyolov2.cfg or [here](https://pastebin.com/681E3JHg)
##### The [out] section
Contains the number of labels, the number of anchor boxes (more on that [here](#anchors)) as well as the input dimensions of the backend network.

##### The [loss] section
Contains parameters used by the loss function (ignore if only using a pretrained network) and the batch size. Use a batch size of 1 for testing purposes and a larger one for training.

##### The [optimizer] section
The type of optimizer and the parameters of said optimizer. Supported optimizers are ADAM, RMSprop and SGD. Examples of their usage can be found in the example config.

##### The [net] section
The architecture of the network used as a backend for the model. Predefined supported architectures are mobilenet and VGG. Custom architectures are also supported.

##### The [decode] section
Includes the anchor points of the predefined anchor boxes (more on that [here](#anchors)) as well as the threshholds for predicting objects and the Non-Max Suppression.

## Anchors
