## For some arithmetic and Matrix Operations
import numpy as np

## Dataframe Manipulation
import pandas as pd

## For Visualization
import matplotlib.pyplot as plt

## For Visualization too
import seaborn as sns

## For Math
import math

## For image and video input
import cv2

## For acessing the directory
import os

# The shutil in Python is a module that offers several functions to deal with operations on files and their collections. It provides the ability to copy and removal of files.
# In a way, it is similar to the OS Module; however, the OS Module does have functions dealing with collections of files.
import shutil

## Splitting folders
import splitfolders

## Python Imaging Library
import PIL
import PIL.Image

## The requests module allows you to send HTTP requests using Python.
import requests

## Time module
import time
from datetime import datetime

## For Randomization
import random

## Regular expression
## A regular expression (or RE) specifies a set of strings that matches it
import re

## Scipy mododule
## SciPy is an open-source Python library which is used to solve
import scipy

## Deeplearning Packages
import tensorflow as tf
from tensorflow import keras

## Deeplearning models
## Models
from keras.models import Sequential
from keras.models import load_model 

## Layers
from keras import layers
## For setting the input layer
from keras.layers import InputLayer
## For the creation of different layer of neural networks
from keras.layers import Dense
## Convolution layer
from keras.layers import Conv2D
## For Pooling
from keras.layers import MaxPooling2D
## To turn the neural network layer into one dimensional
from keras.layers import Flatten
## Global Average Pooling
from keras.layers import GlobalAveragePooling2D
## Batch Normalization
from keras.layers import BatchNormalization
## Dropping some of the data in the layer of Neural Network
from keras.layers import Dropout

## Image Prerocessing layers
from keras.layers import Resizing
from keras.layers import Rescaling
from keras.layers import CenterCrop

## Image data augmentation layers
from keras.layers import RandomFlip
from keras.layers import RandomRotation

## Utils
## Importing Images and reading it
from keras.utils import load_img
## to convert the categorical variable of target and one hot encode it
from keras.utils import to_categorical
## Normalizing the data
from keras.utils import normalize
## Converting the loaded image to array
from keras.utils import img_to_array


## Applications. The pretrained models
## MobileNetV2
from keras.applications import MobileNetV2
## Preprocessing of mobilenetv2
from keras.applications import mobilenet_v2
## VGGFace
from keras_vggface.vggface import VGGFace
## Preprocessing Input of mobilenet. Scales values from -1 to 1. This is just the same as the one in mobilenetv2
from keras.applications.mobilenet import preprocess_input
## VGG 19
from keras.applications import VGG19
## VGG19 Preprocess Input
from keras.applications import vgg19
## ResNet50
from keras.applications import ResNet50
## Xception
from keras.applications import Xception
## Inception V3
from keras.applications import InceptionV3



## Preprocessing
from keras.preprocessing.image import ImageDataGenerator

## Metrics
from keras.metrics import Precision, Recall, BinaryAccuracy


## Callbacks
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


## To reset the session in keras models
from keras import backend as K


## For Image Localization
from matplotlib.patches import Rectangle #To add a rectangle overlay to the image
from skimage.feature.peak import peak_local_max  #To detect hotspots in 2D images. 


## to prevent error from appearing in tensorflow

# may put this function in another utility file
def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

# replace `import tensorflow as tf` with this line
# or insert this line at the beginning of the `__init__.py` of a package that depends on tensorflow
tf = import_tensorflow()    

