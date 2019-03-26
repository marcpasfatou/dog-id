from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas as pd
import os
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

execution_path = os.getcwd()
dataset_path = os.path.join(execution_path, "train")

tf.enable_eager_execution()

print(os.listdir())


train_label = pd.read_csv(os.path.join(execution_path,"label.csv"))
image_size = 224

def get_batch(files,label_file, batch_size = 32):
    """
        Create batch of n pairs, half same class, half different class
    """
    
