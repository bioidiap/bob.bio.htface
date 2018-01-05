#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from bob.learn.tensorflow.network.InceptionResnetV2 import block35, block17, block8


def inception_resnet_v2_core(inputs,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV2',
                        mode=tf.estimator.ModeKeys.TRAIN,
                        trainable_variables=None,
                        **kwargs):
    """
    
    Core of the Inception Resnet V2 model.
    Here we consider the core from the layer `Mixed_6a`
   
    **Parameters**:
    
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      
      dropout_keep_prob: float, the fraction to keep before final layer.
      
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
        
      scope: Optional variable_scope.

      trainable_variables: list
        List of variables to be trainable=True
      
    **Returns**:
    
      logits: the logits outputs of the model.
    """
    
    net = inputs

    # 17 x 17 x 1024
    name = "Mixed_6a"
    with tf.variable_scope(name):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(
                net,
                384,
                3,
                stride=2,
                padding='VALID',
                scope='Conv2d_1a_3x3',
                trainable=False,
                reuse=reuse)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(
                net,
                256,
                1,
                scope='Conv2d_0a_1x1',
                trainable=False,
                reuse=reuse)
            tower_conv1_1 = slim.conv2d(
                tower_conv1_0,
                256,
                3,
                scope='Conv2d_0b_3x3',
                trainable=False,
                reuse=reuse)
            tower_conv1_2 = slim.conv2d(
                tower_conv1_1,
                384,
                3,
                stride=2,
                padding='VALID',
                scope='Conv2d_1a_3x3',
                trainable=False,
                reuse=reuse)
        with tf.variable_scope('Branch_2'):
            tower_pool = slim.max_pool2d(
                net,
                3,
                stride=2,
                padding='VALID',
                scope='MaxPool_1a_3x3')
        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

    # BLOCK 17
    name = "Block17"
    net = slim.repeat(
        net,
        20,
        block17,
        scale=0.10,
        trainable_variables=False,
        reuse=reuse)

    name = "Mixed_7a"
    with tf.variable_scope(name):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(
                net,
                256,
                1,
                scope='Conv2d_0a_1x1',
                trainable=False,
                reuse=reuse)
            tower_conv_1 = slim.conv2d(
                tower_conv,
                384,
                3,
                stride=2,
                padding='VALID',
                scope='Conv2d_1a_3x3',
                trainable=False,
                reuse=reuse)
        with tf.variable_scope('Branch_1'):
            tower_conv1 = slim.conv2d(
                net,
                256,
                1,
                scope='Conv2d_0a_1x1',
                trainable=False,
                reuse=reuse)
            tower_conv1_1 = slim.conv2d(
                tower_conv1,
                288,
                3,
                stride=2,
                padding='VALID',
                scope='Conv2d_1a_3x3',
                trainable=False,
                reuse=reuse)
        with tf.variable_scope('Branch_2'):
            tower_conv2 = slim.conv2d(
                net,
                256,
                1,
                scope='Conv2d_0a_1x1',
                trainable=False,
                reuse=reuse)
            tower_conv2_1 = slim.conv2d(
                tower_conv2,
                288,
                3,
                scope='Conv2d_0b_3x3',
                trainable=False,
                reuse=reuse)
            tower_conv2_2 = slim.conv2d(
                tower_conv2_1,
                320,
                3,
                stride=2,
                padding='VALID',
                scope='Conv2d_1a_3x3',
                trainable=False,
                reuse=reuse)
        with tf.variable_scope('Branch_3'):
            tower_pool = slim.max_pool2d(
                net,
                3,
                stride=2,
                padding='VALID',
                scope='MaxPool_1a_3x3')
        net = tf.concat([
            tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool
        ], 3)

    # Block 8
    name = "Block8"
    net = slim.repeat(
        net,
        9,
        block8,
        scale=0.20,
        trainable_variables=False,
        reuse=reuse)
    net = block8(
        net,
        activation_fn=None,
        trainable_variables=False,
        reuse=reuse)

    name = "Conv2d_7b_1x1"
    net = slim.conv2d(
        net, 1536, 1, scope=name, trainable=False, reuse=reuse)

    with tf.variable_scope('Logits'):
        # pylint: disable=no-member
        net = slim.avg_pool2d(
            net,
            net.get_shape()[1:3],
            padding='VALID',
            scope='AvgPool_1a_8x8')
        net = slim.flatten(net)

        net = slim.dropout(net, dropout_keep_prob, scope='Dropout', is_training=(mode == tf.estimator.ModeKeys.TRAIN))


    name = "Bottleneck"
    net = slim.fully_connected(
        net,
        bottleneck_layer_size,
        activation_fn=None,
        scope=name,
        reuse=reuse,
        trainable=False)

    return net
