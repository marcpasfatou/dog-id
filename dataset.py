from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas as pd
import os
import numpy as np
import math, random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


execution_path = os.getcwd()
dataset_path = os.path.join(execution_path, "train")

tf.enable_eager_execution()

print(os.listdir())


train_label = pd.read_csv(os.path.join(execution_path,"labels.csv"))
classes = pd.read_csv(os.path.join(execution_path,"classes.csv"))
dataset_size, _ = train_label.shape
train_size = math.floor(float(dataset_size) * .8)
train_dataset = train_label.sample(n=train_size)
train_datagen = ImageDataGenerator(rescale=1./255)
#val_dataset = train_label[train_label["id"] != train_dataset["id"]]
train_dataset.reset_index(drop=True, inplace=True)
#val_dataset.reset_index(drop=True, inplace=True)
#print(train_dataset)

def get_batch(batch_size=32):

    left_images = []
    right_images = []
    target = []
    #we choose random classes
    categories = classes.sample(n=batch_size)

    #for the second pair we choose the same category for the first half, different for the other half
    part1 = categories[: batch_size // 2]
    part2 = classes.sample(n=batch_size // 2)
    categories_2 = pd.concat([part1, part2])
    categories.reset_index(drop=True, inplace=True)
    categories_2.reset_index(drop=True, inplace=True)
    categories = pd.concat([categories, categories_2], axis=1)



    #the label is created and store in target 1 when the pair is similar 0 when the pair is different
    target.extend(np.ones((batch_size //2), dtype=int))
    target.extend(np.zeros((batch_size //2), dtype=int))
    print(target)

    #we retrieve paths to images according to the category
    for index, row in categories.iterrows():
        left_images.append(os.path.join(dataset_path,row.iloc[0],train_dataset[train_dataset["breed"] == row.iloc[0]].sample(n=1)["id"].to_string(index=False)[1:] + ".jpg"))
        right_images.append( os.path.join(dataset_path, row.iloc[1],train_dataset[train_dataset["breed"] == row.iloc[1]].sample(n=1)["id"].to_string(index=False)[1:]+ ".jpg"))

    # the data is converted to a tensor
    filenames_l = tf.constant(left_images)
    filenames_r = tf.constant(right_images)
    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant(target)
    batch = tf.data.Dataset.from_tensor_slices((filenames_l, filenames_r, target))
    batch = batch.map(_parse_function)

    return batch



# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename_l, filename_r, label):
  image_string_l = tf.read_file(filename_l)
  image_decoded_l = tf.image.decode_jpeg(image_string_l)
  image_resized_l = tf.image.resize_images(image_decoded_l, [28, 28])

  image_string_r = tf.read_file(filename_r)
  image_decoded_r = tf.image.decode_jpeg(image_string_r)
  image_resized_r = tf.image.resize_images(image_decoded_r, [28, 28])
  return image_resized_l, image_resized_r, label





