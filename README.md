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
#### The [out] section
Contains the number of labels, the number of anchor boxes (more on that [here](#anchors)) as well as the input dimensions of the backend network.

#### The [loss] section
Contains parameters used by the loss function (ignore if only using a pretrained network) and the batch size. Use a batch size of 1 for testing purposes and a larger one for training.

#### The [optimizer] section
The type of optimizer and the parameters of said optimizer. Supported optimizers are ADAM, RMSprop and SGD. Examples of their usage can be found in the example config.

#### The [net] section
The architecture of the network used as a backend for the model. Predefined supported architectures are mobilenet and VGG. Custom architectures are also supported.

#### The [decode] section
Includes the anchor points of the predefined anchor boxes (more on that [here](#anchors)) as well as the threshholds for predicting objects and the Non-Max Suppression.

### Anchors
The neural network predicts bounding boxes based on predefined anchor boxes. These anchor boxes can be hand-picked or (the better approach) can be calculated by an algorithm. </br>
Anchors boxes are boxes which closely resemble the most typical object shapes in the training set. Anchor boxes look something like this: </br> </br>
![Anchors](https://i.imgur.com/EvshViU.png)

In [kmeans.ipynb](kmeans.ipynb) you can find a K-Means implementation of calculating the best possible N anchor boxes and a graph of the average IoU for each N. Based on the graph you should pick your number of boxes. For the VOC dataset the graph looks like so, and 5 bounding boxes is appropriate. </br> </br>
![VOCkmeansgraph](https://i.imgur.com/65HTdY4.png)

## Detecting objects
Detecting objects is done the following way:
```
labels_dir = "./labels.txt"
cfg_path = r"./cfg/mobilenetyolov2.cfg"
weights = r'./weights/mobilenetyolov2'

cfg = Config(cfg_path)
encoder = LabelEncoder(read_labels(labels_dir)[0])
networkfactory = NetworkFactory()

yolo = YOLO(cfg, encoder, networkfactory, weights)

afk = r"./mytestimages/marian.jpg"
objs = yolo.feed_forward(afk, draw = True, supression="regular", save_image = True, save_json = True, onlyconf = True)
```
In order to detect objects you need to:
1. Specify the labels & create a LabelEncoder object
2. Specify the config & create a Config object
3. Specify the weights
4. Initialize a YOLO object with the config, encoder, weights & a NetworkFactory

You detect objects by calling the feed_forward method, which returns a list of all the detected objects. It can also display the image (with the boxes surrounding detected objects), save the image with drawn boxes and save the list with objects as a .json file.

A detection demo can be found [feed-forward.ipynb](feed-forward.ipynb).

## Training a new model / further training an existing model
Training your model is done the following way:
```
labels_dir = "./labels.txt"
cfg_path = r"./cfg/mobilenetyolov2.cfg"
weights = r'./weights/mobilenetyolov2'
annotation_folder = r'.\annotations'
images_folder = r'.\images'

annotations, images = get_annotations_images(annotation_folder, images_folder)

cfg = Config(cfg_path)
encoder = LabelEncoder(read_labels(labels_dir)[0])
networkfactory = NetworkFactory()

#yolo = YOLO(cfg, encoder, networkfactory)
yolo = YOLO(cfg, encoder, networkfactory, weights)

epochs = 2000
yolo.train(batch_generator_inmemory, annotations, images, epochs)

yolo.save(weights)
```
In order to train a model you need to:
1. Specify the labels & create a LabelEncoder object
2. Specify the config & create a Config object
3. Specify the weights (optional - if you do not specify any weights training will start from scratch)
4. Initialize a YOLO object with the config, encoder, weights (optional) & a NetworkFactory
5. Call the train method, specifying a batch generator, the training dataset (images & annotations) and the number of epochs
6. Save your newly trained network by calling the save method

A training demo can be found [train.ipynb](train.ipynb).
