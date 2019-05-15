####################################
#                                  #
#           model define           #
#                                  #
####################################
import numpy as np
import tensorflow as tf
from contextlib import contextmanager


def gated_linear_layer(inputs, gates, name=None):
    activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)

    return activation


def instance_norm_layer(
        inputs,
        epsilon=1e-06,
        activation_fn=None,
        name=None):
    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs=inputs,
        epsilon=epsilon,
        activation_fn=activation_fn)

    return instance_norm_layer


def conv1d_layer(
        inputs,
        filters,
        kernel_size,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=None,
        name=None):
    conv_layer = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def conv2d_layer(
        inputs,
        filters,
        kernel_size,
        strides,
        padding='same',
        activation=None,
        kernel_initializer=None,
        name=None):
    conv_layer = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def residual1d_block(
        inputs,
        filters=1024,
        kernel_size=3,
        strides=1,
        name_prefix='residule_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs=h1_glu, filters=filters // 2, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs=h2, activation_fn=None, name=name_prefix + 'h2_norm')

    h3 = inputs + h2_norm

    return h3


def downsample1d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample1d_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def downsample2d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample2d_block_'):
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample1d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        shuffle_size=2,
        name_prefix='upsample1d_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs=h1, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs=h1_gates, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample2d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        shuffle_size=2,
        name_prefix='upsample1d_block_'):
    # h1 = tf.layers.conv2d_transpose(inputs, 256, [5, 5], strides=[2, 2], activation=None, padding="same")
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    # h1_shuffle = pixel_2Dshuffler(inputs=h1, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle')
    h1_shuffle = up_2Dsample(x=h1, scale_factor=shuffle_size)
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    # h1_shuffle_gates = pixel_2Dshuffler(inputs=h1_gates, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle_gates')
    h1_shuffle_gates = up_2Dsample(x=h1_gates, scale_factor=shuffle_size)
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample2d_block_withDeconv(
        inputs,
        filters,
        kernel_size,
        strides,
        shuffle_size=2,
        name_prefix='upsample1d_block_'):
    h1_shuffle = tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=[2, 2], activation=None,
                                            padding="same")
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_shuffle_gates = tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=[2, 2], activation=None,
                                                  padding="same")
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def up_2Dsample(x, scale_factor=2):
    # _, h, w, _ = x.get_shape().as_list()
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def pixel_shuffler(inputs, shuffle_size=2, name=None):
    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow, oc], name=name)

    return outputs


def pixel_2Dshuffler(inputs, shuffle_size=2, name=None):
    n = tf.shape(inputs)[0]
    w1 = tf.shape(inputs)[1]
    w2 = tf.shape(inputs)[2]
    c = inputs.get_shape().as_list()[3]

    oc = c // shuffle_size // shuffle_size
    ow1 = w1 * shuffle_size
    ow2 = w2 * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow1, ow2, oc], name=name)

    return outputs