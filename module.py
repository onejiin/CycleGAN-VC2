# -*- coding: utf-8 -*-

import tensorflow as tf
from ops import *

# ------------------------------------------------------------------------------------------------------------------- #
#                                                   CycleGAN-VC1                                                      #
def generator_gatedcnn(inputs, reuse = False, scope_name = 'generator_gatedcnn'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs = h1_glu, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs = d2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        # Upsample
        u1 = upsample1d_block(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample1d_block(inputs = u1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs = u2, filters = 24, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2


def discriminator(inputs, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv')
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block1_')
        d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block2_')
        d3 = downsample2d_block(inputs = d2, filters = 1024, kernel_size = [6, 3], strides = [1, 2], name_prefix = 'downsample2d_block3_')

        # Output
        o1 = tf.layers.dense(inputs = d3, units = 1, activation = tf.nn.sigmoid)

        return o1
#                                                   CycleGAN-VC1                                                      #
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
#                                                   CycleGAN-VC2                                                      #
def generator_gated2Dcnn(inputs, reuse = False, scope_name = 'generator_gated2Dcnn'):
    res_filter = 512   # 이거 512로 바꿔야 논문 format 임 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Check!!
    inputs = tf.transpose(inputs, perm=[0, 2, 1], name='input_transpose')
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5, 15], strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5, 15], strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_')
        d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_')

        # reshape : cyclegan-VC2
        d3 = tf.reshape(d2, shape=[tf.shape(d2)[0], -1, d2.get_shape()[3].value])
        # modification in paper - 2019.05.01
        d3 = conv1d_layer(inputs=d3, filters=res_filter//2, kernel_size = 1, strides = 1, activation = None, name = '1x1_down_conv1d')

        # Residual blocks
        r1 = residual1d_block(inputs = d3, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        # modification in paper
        r6 = conv1d_layer(r6, filters=res_filter, kernel_size = 1, strides = 1, activation = None, name = '1x1_up_conv1d')
        # reshape : cyclegan-VC2
        r6 = tf.reshape(r6, shape=[tf.shape(d2)[0], tf.shape(d2)[1], tf.shape(d2)[2], d2.get_shape()[3].value])

        # Upsample
        u1 = upsample2d_block(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample2d_block(inputs = u1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        # Output
        o1 = conv2d_layer(inputs = u2, filters = 1, kernel_size = [5, 15], strides = 1, activation = None, name = 'o1_conv')
        # o1 = tf.squeeze(o1)
        o1 = tf.reshape(o1, shape=[tf.shape(o1)[0], tf.shape(o1)[1], -1])
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2


# deconvolution addition
def generator_gated2Dcnn_withDeconv(inputs, reuse = False, scope_name = 'generator_gated2Dcnn'):
    res_filter = 512
    inputs = tf.transpose(inputs, perm=[0, 2, 1], name='input_transpose')
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5, 15], strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5, 15], strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_')
        d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_')

        # reshape : cyclegan-VC2
        d3 = tf.reshape(d2, shape=[tf.shape(d2)[0], -1, d2.get_shape()[3].value])
        # modification in paper - 2019.05.01
        d3 = conv1d_layer(inputs=d3, filters=res_filter//2, kernel_size = 1, strides = 1, activation = None, name = '1x1_down_conv1d')

        # Residual blocks
        r1 = residual1d_block(inputs = d3, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        # modification in paper - 2019.05.01
        r6 = conv1d_layer(r6, filters=res_filter, kernel_size = 1, strides = 1, activation = None, name = '1x1_up_conv1d')
        # reshape : cyclegan-VC2
        r6 = tf.reshape(r6, shape=[tf.shape(d2)[0], tf.shape(d2)[1], tf.shape(d2)[2], d2.get_shape()[3].value])

        # Upsample
        u1 = upsample2d_block_withDeconv(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample2d_block_withDeconv(inputs = u1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        # Output
        o1 = conv2d_layer(inputs = u2, filters = 1, kernel_size = [5, 15], strides = 1, activation = None, name = 'o1_conv')
        # o1 = tf.squeeze(o1)
        o1 = tf.reshape(o1, shape=[tf.shape(o1)[0], tf.shape(o1)[1], -1])
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2


def discriminator_2D(inputs, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 1], activation = None, name = 'h1_conv')
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 1], activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block1_')
        d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block2_')
        d3 = downsample2d_block(inputs = d2, filters = 1024, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block3_')
        d4 = downsample2d_block(inputs=d3, filters=1024, kernel_size=[1, 5], strides=[1, 1], name_prefix='downsample2d_block4_')


        # Output
        # o1 = tf.layers.dense(inputs = d3, units = 1, activation = tf.nn.sigmoid)
        o1 = conv2d_layer(inputs=d4, filters=1, kernel_size=[1, 3], strides=[1, 1], activation=tf.nn.sigmoid, name='out_1d_conv')

        return o1
#                                                   CycleGAN-VC2                                                      #
# ------------------------------------------------------------------------------------------------------------------- #
