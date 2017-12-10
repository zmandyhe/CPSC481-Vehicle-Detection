# Vehicle Detection Algorithm: Tensorflow + OpenCV
This project is inspired by CSUF CPSC481 Artificial Intelligence (AI) class, which introduced many classical AI and Machine Learning algorithms. The goal of this project is to utilize existing state-of-the-art libraries to develop a pipeline to detect vehicles from pictures captured in parking lots and street drivings (KITTI dataset). As an interesting experiment, this project also created an initial pipeline to detect vehicles from real-time street driving videos. This part is set as a very early stage for Driverless Car project that I will continue to work on.

The following in this file will describe the key steps of how to use and run the the vehicle detection pipeline in Jupyter Notebook, the serveral external libraries used, and some key functions used in the vehicle detection process.

## Convolutional Neural Networks (CNNs) Concepts

below content are according to Tensorflow CNNs, for further information, please refer to: https://www.tensorflow.org/tutorials/deep_cnn.
### Convolutional neural networks (CNNs) are the current state-of-the-art model architecture for image classification tasks. CNNs apply a series of filters to the raw pixel data of an image to extract and learn higher-level features, which the model can then use for classification. CNNs contains three components:
•    Convolutional layers, which apply a specified number of convolution filters to the image. For each sub-region, the layer performs a set of mathematical operations to produce a single value in the output feature map. Convolutional layers then typically apply a ReLU activation function to the output to introduce nonlinearities into the model.
•    Pooling layers, which downsample the image data extracted by the convolutional layers to reduce the dimensionality of the feature map in order to decrease processing time. A commonly used pooling algorithm is max pooling, which extracts subregions of the feature map (e.g., 2x2-pixel tiles), keeps their maximum value, and discards all other values.
•    Dense (fully connected) layers, which perform classification on the features extracted by the convolutional layers and downsampled by the pooling layers. In a dense layer, every node in the layer is connected to every node in the preceding layer.

Typically, a CNN is composed of a stack of convolutional modules that perform feature extraction. Each module consists of a convolutional layer followed by a pooling layer. The last convolutional module is followed by one or more dense layers that perform classification. The final dense layer in a CNN contains a single node for each target class in the model (all the possible classes the model may predict), with a softmax activation function to generate a value between 0–1 for each node (the sum of all these softmax values is equal to 1). We can interpret the softmax values for a given image as relative measurements of how likely it is that the image falls into each target class.

The SSD Network model used in this project pipeline utilizes CNNS with 7 Conv layers and 4 extra blocks/layers. This is an open source library provided by TF-Slim, under the project folder of ssd_vgg.py with the function of SSDNet.

## Setup and Installation

These instructions will go through the development environment, the dependent libraries, and process of running the pipeline in Jupyter Notebook on your local machine.

### Development Environment
Since this project depends on several specific external libararies, it is suggested to set up a tensorflow virtual environment to run tensorflow and other dependencies.

#### Python 3 environment:
The pipeline is developed and tested on Python 3 environment. Please make sure you have Python 3 and pip installed in your local computer.

#### Setup Virtual Environment:
Since the pipeline depends on open source and/or external public libraries, some of them might be throughly tested. It is suggested to run the pipeline in a virtual environment. In this file, it names the virtual environment as "tensorflow" to specify it is running on Tensforflow projects. The instruction has been tested on On macOS and Linux system.

##### Install Python 3 virtual environment named "tensorflow":
python3 -m virtualenv tensorflow

##### to activate virtual environment named "tensorflow":
source ~/tensorflow/bin/activate

##### Install Tensorflow under the virtual environment:
pip3 install tensorflow

##### Install Tensorflow Object Detection API from github:
The TensorFlow Object Detection API is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models. It needs to be installed separately by either clone or download the zip file to your local computer.

To use the download method, go to its github: https://github.com/tensorflow/models/tree/master/research/object_detection
Once it is downloaded, its file folder is usally named as models. Manually move this model folder under Tensorflow folder. When running some models, it will need to run from Tensorflow/Models/Reserach folder. The key libraries and pre-trained model of VGG16 is also located under the folder of "slim".

##### Install Jupyter Notebook:
This pipeline is demonstrated in Jupyter Notebook.
pip install jupyter

#### Protobuf Compilation
The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the tensorflow/models/research/ directory:

```From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

### Install the Remaining Dependent Libraries:
To run this pipeline, it needs to install the following libraries:
```pip3 install pillow
pip3 install lxml
pip3 install matplotlib
```

OpenCV libraries is used to processing images for this pipeline, to install OpenCV on python 3:
pip3 install opencv -python

### Add Libraries to PYTHONPATH
When running locally, the tensorflow/models/research/ and slim directories should be appended to PYTHONPATH. This can be done by running the following from tensorflow/models/research/:
From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
Note: This command needs to run from every new terminal you start. If you wish to avoid running this manually, you can add it as a new line to the end of your ~/.bashrc file.

### Test Installation:
You can test that you have correctly installed the Tensorflow Object Detection API by running the following command:
'''
python object_detection/builders/model_builder_test.py
'''

## Running the Pipeline in Jupyter Notebook

### Firstly running the following from tensorflow/models/research/:
From tensorflow/models/research/
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### cd project folder under the virtual environment
```
cd cpsc481project
~/tensorflow/bin/activate
```
### Launch Jupyter Notebook:
Run below command and it will launch the Jupyter Notebook in browser. Use the file of  "vehicle-detection-pic-count-video-sutmit.ipynb";
```
jupyter notebook
```

## External Libraries:

Creating accurate deep learning models capable of localizing and identifying multiple objects or vehicle objects for this project purpose in a single image remains a core challenge in computer vision and machine learning fields.  During the project assignment 1 and assignment 2 processes for evluating varied models and approaches, I have certainly found there are external open source codebases to be useful for my project need. There are two main open source libraries I used in my developing pipeline for my project.

#### Google's Tensorflow common deep CNNs in TF-Slim Libraries. 
When downloaded the Tensorflow Object Detection API form github mentioned above, the TF-Slim libraries had been include under the foler of "~/tensorflow/models/research/slim". t contains source files on how to convert the original raw data into TFRecords files. The project re-uses the pre-trained model for KITT dataset, the "kitt.py" is to perform the convertion, the "kitti_commom.py" is to implement the interface. The RF-Slim libraries also contains other source files that can work on Pascal VOC dataset, and so on. To visit the TF-Slim: https://github.com/tensorflow/models/tree/master/research/slim

#### SSD Pre-Processing: 
The TF-Slim library also contains the implementation of the pre-processing before training or evaluation using the source file of "ssd_vgg_preprocessing.py". To visit the TF-Slim: https://github.com/tensorflow/models/tree/master/research/slim

#### The pre-trained model of SSD Network: 
The TF-Slim under the same folder contains the SSD network implemented in this project pipeline used the the source file "ssd_vgg_300.py" which defines the 11 layers for the convelutional neural network model. To visit the TF-Slim: https://github.com/tensorflow/models/tree/master/research/slim

#### Post-processing/SDC-Vehicle-Detection: 
The SSD Network detected and draw one bounding box for each detected vehicle on every layers of convolutional neutral network, so it needs to impelment a post-processing process to average the bounding boxes tensor values to reduce the bounding boxed from multiple to one unique detection box and output to draw on the final image (or video). To implement this process, a reference and re-wirtten of the famous Non-Maximum Supression math model is used with a reference of an open source project "SDC-Vehicle-Detection" NMS algorithem. To visit this algorithm: https://github.com/balancap/SDC-Vehicle-Detection

## Key Functions:

### image pre-processing function:
Process an image through SSD network, inputs include 1) img: Numpy array containing an image, 2) select_threshold: Classification threshold (i.e. probability threshold for car detection), with default value of 0.5. The function returns classes, scores and bboxes of objects detected. It calls functions defined in ssd_commom.py:

### anchor boxes encoding:
Bounding box: a float.32 format, which contains coordinates of [y_min, x_min, y_max, x_max] that floats in the range of [0.0, 1.0] relative to an underlining image’s height and width.
tf.image.draw_bounding_boxes(images, boxes, name=None), it returns a Tensor

### Feature Layer: defined in net/ssd_vgg_300.py:
The TensorFlow layers module provides a high-level API that makes it easy to construct a neural network. It provides methods that facilitate the creation of dense (fully connected) layers and convolutional layers, adding activation functions, and applying dropout regularization. TF-Slim library, ssd_vgg_300.py defines a SSDNet class to extract features and CNN layers.

### Post-processing: Non-Maximum Supression Algorithm:
It firstly compute overlap score between bboxes1 and bboxes2 of their height and width, and return maximum value of them. After that it apply non-maximum selection to bounding boxes with score averaging: go over the list of boxes, and for each, see if boxes with lower score overlap. If yes, averaging their scores and coordinates, and consider it as a valid detection. It takes input of classes, scores and bboxes from SSD network output in the previous step, then return classes, scores and bboxes of objects detected after applying NMS. The bboxes will be used to draw on the image as an single output detection box.

### Extract Car Collection Function: 
it first tries to match cars from the collection with new bounding boxes. For every match, the car coordinates and speed are updated accordingly. If there are remaining boxes at the end of the matching process, every one of them is considered as a new car. Finally, the algorithm also checks in how many of the last N frames the object has been detected.
This allows to remove false positives, which usually only appear on a very few frames. It output a list of car objects updated.

### Video Process Frame function:
it is to take the image input from car collection, process through SSD network, then apply NMS algorithm and draw bounding boxes. The later part is similar with processing SSD and apply NMS for processing image.

## Author
This project is to fullfil the CPSC 481 class project requirement. The project is inspired by the class project, and is an initial project for a longer term project that might lead to a graduate project of Driverless Car. Furtuer works will include:
Implement video camera and/or sensors to detect objects
Implement multiple object detections driving on the street, e.g. vehicles, people, obstcles, traffice signs, animals, etc
Implement road lane detection
Train and evaluate own model
Compare the performance from other object detection models
Online prediction
Automate the whole input-training-detect-prediction-output process

## Acknowledgment
This project is inspired by and referred to the "SDC-Vehicle-Detection". To visit this algorithm: https://github.com/balancap/SDC-Vehicle-Detection.

