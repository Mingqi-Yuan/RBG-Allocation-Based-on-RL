import tensorflow as tf
import pandas as pd
import numpy as np
import os

class Network:
    def __init__(self):
        pass

    def conv_block(self, input_tensor, num_filters, kernel_size=(3,3), strides=(1,1), padding='same'):
        x = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        return x

    def res_block(self, input_tensor, num_filters, kernel_size=(3,3), strides=(1,1), padding='same'):
        filters1, filters2, filters3 = num_filters
        x = self.conv_block(input_tensor,filters1, kernel_size, strides, padding)
        x = self.conv_block(x, filters2, kernel_size, strides, padding)
        x = self.conv_block(x, filters3, kernel_size, strides, padding)

        x = tf.keras.layers.add([x, input_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build(self, inputs):
        x = self.conv_block(inputs, 64)

        ''' Downsampling 1 '''
        x = tf.keras.layers.Conv2D(128, kernel_size=(1, 3), strides=(1, 3), padding='valid', name='DownSampling-1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = self.res_block(x, (64, 64, 128))

        ''' Downsampling 2 '''
        x = tf.keras.layers.Conv2D(256, kernel_size=(1, 3), strides=(1, 3), padding='same', name='DownSampling-2')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = self.res_block(x, (128, 128, 256))

        ''' Downsampling 3 '''
        x = tf.keras.layers.Conv2D(512, kernel_size=(1, 4), strides=(1, 3), padding='valid', name='DownSampling-3')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        return x

inputs = tf.keras.Input([None, 56, 1])
outputs = Network().build(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()