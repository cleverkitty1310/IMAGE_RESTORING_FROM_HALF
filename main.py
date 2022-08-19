import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Dense
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tqdm import tqdm

(x_train, y_train), (x_test, y_test) = load_data()

x_ds = []
y_ds = []
for img in tqdm(x_train):
    zeros = np.zeros((28, 14))
    left = img[:, :14] / 255.0
    right = img[:, 14:] / 255.0
    left = np.concatenate((left, zeros), axis = 1)
    right = np.concatenate((zeros, right), axis = 1)
    left = cv2.resize(left, (32, 32))
    right = cv2.resize(right, (32, 32))
    _img = cv2.resize(img, (32, 32))
    _img = _img / 255.0
    x_ds.append(left)
    x_ds.append(right)
    y_ds.append(_img)
    y_ds.append(_img)

x_ds = np.array(x_ds)
y_ds = np.array(y_ds)



def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample


def up(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.add(Dropout(0.2))
    upsample.add(keras.layers.LeakyReLU())
    return upsample

def model():
    inputs = layers.Input(shape= [32, 32, 1])
    d1 = down(128,(3,3),False)(inputs)
    d2 = down(256,(3,3),True)(d1)
    d3 = down(512,(3,3),True)(d2)
    d4 = down(512,(3,3),True)(d3)

    #upsampling
    u1 = up(512,(3,3),False)(d4)
    u1 = layers.concatenate([u1,d3])
    u2 = up(256,(3,3),True)(u1)
    u2 = layers.concatenate([u2,d2])
    u3 = up(128,(3,3),True)(u2)
    u3 = layers.concatenate([u3,d1])
    u4 = up(1, (3,3),True)(u3)
    u4 = layers.concatenate([u4, inputs])
    output = layers.Conv2D(1, (2,2), strides = 1, padding = 'same', activation='relu')(u4)
    return tf.keras.Model(inputs=inputs, outputs=output)


model = model()
model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# model.load_weights('./checkpoints/my_checkpoint')

model.fit(x_ds, y_ds, validation_split = 0.2, epochs = 2, batch_size = 32)

model.save_weights('./checkpoints/my_checkpoint')

rows = 10
cols = 8
fig = plt.figure(figsize=(10, 10))
for i in range(20):
    zeros = np.zeros((28, 14))
    tx_left = x_test[i][:, :14] / 255.0
    tx_right = x_test[i][:, 14:] / 255.0

    tx_left = np.concatenate((tx_left, zeros), axis = 1)
    tx_left = cv2.resize(tx_left, (32, 32))
    ty_left = model.predict(np.array([tx_left]))[0].reshape((32, 32))

    tx_right = np.concatenate((zeros, tx_right), axis = 1)
    tx_right = cv2.resize(tx_right, (32, 32))
    ty_right = model.predict(np.array([tx_right]))[0].reshape((32, 32))
    
    fig.add_subplot(rows, cols, (i // 2) * 8 + 1 + (i % 2) * 4)
    plt.imshow(tx_left, cmap = "gray")
    plt.axis('off')
    if i // 2 == 0:
        plt.title('Original')
    fig.add_subplot(rows, cols, (i // 2) * 8 + 2 + (i % 2) * 4)
    plt.imshow(ty_left, cmap = "gray")
    plt.axis('off')
    if i // 2 == 0:
        plt.title('Predict')

    fig.add_subplot(rows, cols, (i // 2) * 8 + 3 + (i % 2) * 4)
    plt.imshow(tx_right, cmap = "gray")
    plt.axis('off')
    if i // 2 == 0:
        plt.title('Original')
    fig.add_subplot(rows, cols, (i // 2) * 8 + 4 + (i % 2) * 4)
    plt.imshow(ty_right, cmap = "gray")
    plt.axis('off')
    if i // 2 == 0:
        plt.title('Predict')

plt.show()