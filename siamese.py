import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

input_shape = (100,100,1)

#we define the inputs
L_input = tf.keras.Input(input_shape)
R_input = tf.keras.Input(input_shape)


# CNN architecture
convnet = tf.keras.Sequential()

convnet.add(layers.Conv2D(64, (10,10), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
convnet.add(layers.MaxPooling2D())

convnet.add(layers.Conv2D(128, (7,7), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
convnet.add(layers.MaxPooling2D())

convnet.add(layers.Conv2D(128, (4,4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
convnet.add(layers.MaxPooling2D())

convnet.add(layers.Conv2D(256, (4,4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
convnet.add(layers.Flatten())

convnet.add(layers.Dense(4096, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

L_output = convnet(L_input)
R_output = convnet(R_input)

L1_layer = layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([L_output, R_output])

prediction = layers.Dense(1, activation='sigmoid')(L1_distance)

siamese_net = tf.keras.Model(inputs=[L_input, R_input], outputs=prediction)

