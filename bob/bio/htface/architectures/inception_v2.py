#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Here we implement several structures crafted for Siamese networks where parts of the network
are shared and parts are not.

"""

import tensorflow as tf
from bob.learn.tensorflow.network.InceptionResnetV2 import block35, block17, block8
from .utils import inception_resnet_v2_core
import tensorflow.contrib.slim as slim


def compute_layer_name(name, is_left, is_siamese=True):
    """
    Compute the layer name for a siamese/triplet
    """
    if is_siamese:
        # Siamese is either left or right
        if is_left:
            name = name + "_left"
        else:
            name = name + "_right"
    else:
       # if is not siamese is triplet.
        if is_left:
            # Left is the anchor
            name = name + "_anchor"
        else:
            # now we need to decide if it is positive or negative
            name = name + "_positive-negative"
            
    return name


def is_trainable_variable(is_left, mode=tf.estimator.ModeKeys.TRAIN):
    """    
    Defining if it's trainable or not
    """

    # Left is never trainable    
    return mode == tf.estimator.ModeKeys.TRAIN and not is_left


def is_reusable_variable(is_siamese, is_left):
    """    
    Defining if is reusable or not
    """

    # Left is NEVER reusable
    if is_left:
        return False
    else:
        if is_siamese:
            # The right part of siamese is never reusable
            return False
        else:
            # If it's triplet and either posibe and negative branch is already declared,
            # it is reusable 
            for v in tf.global_variables():
                if "_positive-negative" in v.name:
                    return True

            return False


def inception_resnet_v2_adapt_first_head(inputs,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=None,
                                         scope='InceptionResnetV2',
                                         mode=tf.estimator.ModeKeys.TRAIN,
                                         trainable_variables=None,
                                         is_siamese=True,
                                         is_left = True,
                                         **kwargs):
    """Creates the Inception Resnet V2 model for the adaptation of the FIRST LAYER.
   
    **Parameters**:
    
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      
      dropout_keep_prob: float, the fraction to keep before final layer.
      
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
        
      scope: Optional variable_scope.

      trainable_variables: list
        List of variables to be trainable=True
      
      is_siamese: bool
        If True is Siamese, otherwise is triplet

      is_left: bool
        Is the left side of the Siamese?
              
    **Returns**:
    
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """

    end_points = dict()

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                             stride=1,
                             padding='SAME'):

            # Defining if the branches are reusable or not            
            is_trainable = is_trainable_variable(is_left, mode=tf.estimator.ModeKeys.TRAIN)
            is_reusable = is_reusable_variable(is_siamese, is_left)
            
            with slim.arg_scope([slim.dropout], is_training=(mode == tf.estimator.ModeKeys.TRAIN)):

                with slim.arg_scope([slim.batch_norm], trainable=is_trainable, is_training=is_trainable):
                    # CORE OF THE THE ADAPTATION                    
                    
                    # 149 x 149 x 32
                    name = "Conv2d_1a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        inputs,
                        32,
                        3,
                        stride=2,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                                                  
                    end_points[name] = net

            with slim.arg_scope([slim.batch_norm], trainable=False, is_training=False):

                    # 147 x 147 x 32
                    name = "Conv2d_2a_3x3"
                    net = slim.conv2d(
                        net,
                        32,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=False,
                        reuse=reuse)
                    end_points[name] = net

                    # 147 x 147 x 64
                    name = "Conv2d_2b_3x3"
                    net = slim.conv2d(
                        net, 64, 3, scope=name, trainable=False, reuse=reuse)
                    end_points[name] = net

                    # 73 x 73 x 64
                    net = slim.max_pool2d(
                        net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
                    end_points['MaxPool_3a_3x3'] = net

                    # 73 x 73 x 80
                    name = "Conv2d_3b_1x1"
                    net = slim.conv2d(
                        net,
                        80,
                        1,
                        padding='VALID',
                        scope=name,
                        trainable=False,
                        reuse=reuse)
                    end_points[name] = net

                    # 71 x 71 x 192
                    name = "Conv2d_4a_3x3"
                    net = slim.conv2d(
                        net,
                        192,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=False,
                        reuse=reuse)
                    end_points[name] = net

                    # 35 x 35 x 192
                    net = slim.max_pool2d(
                        net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
                    end_points['MaxPool_5a_3x3'] = net

                    # 35 x 35 x 320
                    name = "Mixed_5b"
                    with tf.variable_scope(name):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(
                                net,
                                96,
                                1,
                                scope='Conv2d_1x1',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(
                                net,
                                48,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=False,
                                reuse=reuse)
                            tower_conv1_1 = slim.conv2d(
                                tower_conv1_0,
                                64,
                                5,
                                scope='Conv2d_0b_5x5',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(
                                net,
                                64,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=False,
                                reuse=reuse)
                            tower_conv2_1 = slim.conv2d(
                                tower_conv2_0,
                                96,
                                3,
                                scope='Conv2d_0b_3x3',
                                trainable=False,
                                reuse=reuse)
                            tower_conv2_2 = slim.conv2d(
                                tower_conv2_1,
                                96,
                                3,
                                scope='Conv2d_0c_3x3',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(
                                net,
                                3,
                                stride=1,
                                padding='SAME',
                                scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(
                                tower_pool,
                                64,
                                1,
                                scope='Conv2d_0b_1x1',
                                trainable=False,
                                reuse=reuse)
                        net = tf.concat([
                            tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1
                        ], 3)
                    end_points[name] = net

                    # BLOCK 35
                    name = "Block35"
                    net = slim.repeat(
                        net,
                        10,
                        block35,
                        scale=0.17,
                        trainable_variables=False,
                        reuse=reuse)

                    net = inception_resnet_v2_core(
                                         net,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=reuse,
                                         scope='InceptionResnetV2',
                                         mode=mode,
                                         trainable_variables=None,
                                         **kwargs                    
                    )
                    
    return net, end_points


def inception_resnet_v2_adapt_layers_1_2_head(inputs,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=None,
                                         scope='InceptionResnetV2',
                                         mode=tf.estimator.ModeKeys.TRAIN,
                                         trainable_variables=None,
                                         is_siamese=True,
                                         is_left = True,
                                         **kwargs):
    """Creates the Inception Resnet V2 model for the adaptation of the
    FIRST AND SECOND LAYERS

    **Parameters**:

      inputs: a 4-D tensor of size [batch_size, height, width, 3].

      dropout_keep_prob: float, the fraction to keep before final layer.

      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.

      scope: Optional variable_scope.

      trainable_variables: list
        List of variables to be trainable=True

      is_siamese: bool
        If True is Siamese, otherwise is triplet

      is_left: bool
        Is the left side of the Siamese?

    **Returns**:

      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """


    end_points = dict()

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                             stride=1,
                             padding='SAME'):

            # Defining if the branches are reusable or not            
            is_trainable = is_trainable_variable(is_left, mode=tf.estimator.ModeKeys.TRAIN)
            is_reusable = is_reusable_variable(is_siamese, is_left)

            # ADAPTABLE PART
            with slim.arg_scope([slim.dropout], is_training=(mode == tf.estimator.ModeKeys.TRAIN)):

                with slim.arg_scope([slim.batch_norm], trainable=is_trainable, is_training=is_trainable):

                    # CORE OF THE THE ADAPTATION

                    # 149 x 149 x 32
                    name = "Conv2d_1a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        inputs,
                        32,
                        3,
                        stride=2,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 147 x 147 x 32
                    name = "Conv2d_2a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        32,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 147 x 147 x 64
                    name = "Conv2d_2b_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        64,
                        3,
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 73 x 73 x 64
                    net = slim.max_pool2d(
                        net,
                        3,
                        stride=2,
                        padding='VALID',
                        scope='MaxPool_3a_3x3')

                    end_points['MaxPool_3a_3x3'] = net

                    # 73 x 73 x 80
                    name = "Conv2d_3b_1x1"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        80,
                        1,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

            # NON ADAPTABLE PART
            with slim.arg_scope([slim.batch_norm], trainable=False, is_training=False):

                    # 71 x 71 x 192
                    name = "Conv2d_4a_3x3"
                    net = slim.conv2d(
                        net,
                        192,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=False,
                        reuse=reuse)
                    end_points[name] = net

                    # 35 x 35 x 192
                    net = slim.max_pool2d(
                        net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
                    end_points['MaxPool_5a_3x3'] = net

                    # 35 x 35 x 320
                    name = "Mixed_5b"
                    with tf.variable_scope(name):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(
                                net,
                                96,
                                1,
                                scope='Conv2d_1x1',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(
                                net,
                                48,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=False,
                                reuse=reuse)
                            tower_conv1_1 = slim.conv2d(
                                tower_conv1_0,
                                64,
                                5,
                                scope='Conv2d_0b_5x5',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(
                                net,
                                64,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=False,
                                reuse=reuse)
                            tower_conv2_1 = slim.conv2d(
                                tower_conv2_0,
                                96,
                                3,
                                scope='Conv2d_0b_3x3',
                                trainable=False,
                                reuse=reuse)
                            tower_conv2_2 = slim.conv2d(
                                tower_conv2_1,
                                96,
                                3,
                                scope='Conv2d_0c_3x3',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(
                                net,
                                3,
                                stride=1,
                                padding='SAME',
                                scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(
                                tower_pool,
                                64,
                                1,
                                scope='Conv2d_0b_1x1',
                                trainable=False,
                                reuse=reuse)
                        net = tf.concat([
                            tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1
                        ], 3)
                    end_points[name] = net

                    # BLOCK 35
                    name = "Block35"
                    net = slim.repeat(
                        net,
                        10,
                        block35,
                        scale=0.17,
                        trainable_variables=False,
                        reuse=reuse)

                    net = inception_resnet_v2_core(
                                         net,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=reuse,
                                         scope='InceptionResnetV2',
                                         mode=mode,
                                         trainable_variables=None,
                                         **kwargs
                    )

    return net, end_points


def inception_resnet_v2_adapt_layers_1_4_head(inputs,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=None,
                                         scope='InceptionResnetV2',
                                         mode=tf.estimator.ModeKeys.TRAIN,
                                         trainable_variables=None,
                                         is_siamese=True,
                                         is_left = True,
                                         **kwargs):
    """Creates the Inception Resnet V2 model for the adaptation of the
    FIRST AND FORTH LAYERS

    **Parameters**:

      inputs: a 4-D tensor of size [batch_size, height, width, 3].

      dropout_keep_prob: float, the fraction to keep before final layer.

      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.

      scope: Optional variable_scope.

      trainable_variables: list
        List of variables to be trainable=True

      is_siamese: bool
        If True is Siamese, otherwise is triplet

      is_left: bool
        Is the left side of the Siamese?

    **Returns**:

      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """

    end_points = dict()

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                             stride=1,
                             padding='SAME'):

            # Defining if the branches are reusable or not            
            is_trainable = is_trainable_variable(is_left, mode=tf.estimator.ModeKeys.TRAIN)
            is_reusable = is_reusable_variable(is_siamese, is_left)

            # ADAPTABLE PART
            with slim.arg_scope([slim.dropout], is_training=(mode == tf.estimator.ModeKeys.TRAIN)):


                with slim.arg_scope([slim.batch_norm], trainable=is_trainable, is_training=is_trainable):
                    # CORE OF THE THE ADAPTATION
                    # 149 x 149 x 32
                    name = "Conv2d_1a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        inputs,
                        32,
                        3,
                        stride=2,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 147 x 147 x 32
                    name = "Conv2d_2a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        32,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 147 x 147 x 64
                    name = "Conv2d_2b_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        64,
                        3,
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 73 x 73 x 64
                    net = slim.max_pool2d(
                        net,
                        3,
                        stride=2,
                        padding='VALID',
                        scope='MaxPool_3a_3x3')

                    end_points['MaxPool_3a_3x3'] = net

                    # 73 x 73 x 80
                    name = "Conv2d_3b_1x1"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        80,
                        1,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 71 x 71 x 192
                    name = "Conv2d_4a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        192,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net


            # NON ADAPTABLE PART
            with slim.arg_scope([slim.batch_norm], trainable=False, is_training=False):

                    # 35 x 35 x 192
                    net = slim.max_pool2d(
                        net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
                    end_points['MaxPool_5a_3x3'] = net

                    # 35 x 35 x 320
                    name = "Mixed_5b"
                    with tf.variable_scope(name):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(
                                net,
                                96,
                                1,
                                scope='Conv2d_1x1',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(
                                net,
                                48,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=False,
                                reuse=reuse)
                            tower_conv1_1 = slim.conv2d(
                                tower_conv1_0,
                                64,
                                5,
                                scope='Conv2d_0b_5x5',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(
                                net,
                                64,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=False,
                                reuse=reuse)
                            tower_conv2_1 = slim.conv2d(
                                tower_conv2_0,
                                96,
                                3,
                                scope='Conv2d_0b_3x3',
                                trainable=False,
                                reuse=reuse)
                            tower_conv2_2 = slim.conv2d(
                                tower_conv2_1,
                                96,
                                3,
                                scope='Conv2d_0c_3x3',
                                trainable=False,
                                reuse=reuse)
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(
                                net,
                                3,
                                stride=1,
                                padding='SAME',
                                scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(
                                tower_pool,
                                64,
                                1,
                                scope='Conv2d_0b_1x1',
                                trainable=False,
                                reuse=reuse)
                        net = tf.concat([
                            tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1
                        ], 3)
                    end_points[name] = net

                    # BLOCK 35
                    name = "Block35"
                    net = slim.repeat(
                        net,
                        10,
                        block35,
                        scale=0.17,
                        trainable_variables=False,
                        reuse=reuse)

                    net = inception_resnet_v2_core(
                                         net,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=reuse,
                                         scope='InceptionResnetV2',
                                         mode=mode,
                                         trainable_variables=None,
                                         **kwargs
                    )

    return net, end_points


def inception_resnet_v2_adapt_layers_1_5_head(inputs,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=None,
                                         scope='InceptionResnetV2',
                                         mode=tf.estimator.ModeKeys.TRAIN,
                                         trainable_variables=None,
                                         is_siamese=True,
                                         is_left = True,
                                         **kwargs):
    """Creates the Inception Resnet V2 model for the adaptation of the
    FIRST AND FIFTH LAYERS

    **Parameters**:

      inputs: a 4-D tensor of size [batch_size, height, width, 3].

      dropout_keep_prob: float, the fraction to keep before final layer.

      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.

      scope: Optional variable_scope.

      trainable_variables: list
        List of variables to be trainable=True

      is_siamese: bool
        If True is Siamese, otherwise is triplet

      is_left: bool
        Is the left side of the Siamese?

    **Returns**:

      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """

    end_points = dict()

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                             stride=1,
                             padding='SAME'):

            # Defining if the branches are reusable or not            
            is_trainable = is_trainable_variable(is_left, mode=tf.estimator.ModeKeys.TRAIN)
            is_reusable = is_reusable_variable(is_siamese, is_left)

            # ADAPTABLE PART
            with slim.arg_scope([slim.dropout], is_training=(mode == tf.estimator.ModeKeys.TRAIN)):

                # CORE OF THE THE ADAPTATION
                with slim.arg_scope([slim.batch_norm], trainable=is_trainable, is_training=is_trainable):
                    # 149 x 149 x 32
                    name = "Conv2d_1a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        inputs,
                        32,
                        3,
                        stride=2,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 147 x 147 x 32
                    name = "Conv2d_2a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        32,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 147 x 147 x 64
                    name = "Conv2d_2b_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        64,
                        3,
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 73 x 73 x 64
                    net = slim.max_pool2d(
                        net,
                        3,
                        stride=2,
                        padding='VALID',
                        scope='MaxPool_3a_3x3')

                    end_points['MaxPool_3a_3x3'] = net

                    # 73 x 73 x 80
                    name = "Conv2d_3b_1x1"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        80,
                        1,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 71 x 71 x 192
                    name = "Conv2d_4a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        192,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 35 x 35 x 192
                    net = slim.max_pool2d(
                        net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
                    end_points['MaxPool_5a_3x3'] = net

                    # 35 x 35 x 320
                    name = "Mixed_5b"
                    name = compute_layer_name(name, is_left, is_siamese)
                    with tf.variable_scope(name):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(
                                net,
                                96,
                                1,
                                scope='Conv2d_1x1',
                                trainable=is_trainable,
                                reuse=is_reusable)
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(
                                net,
                                48,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=is_trainable,
                                reuse=is_reusable)
                            tower_conv1_1 = slim.conv2d(
                                tower_conv1_0,
                                64,
                                5,
                                scope='Conv2d_0b_5x5',
                                trainable=is_trainable,
                                reuse=is_reusable)
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(
                                net,
                                64,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=is_trainable,
                                reuse=is_reusable)
                            tower_conv2_1 = slim.conv2d(
                                tower_conv2_0,
                                96,
                                3,
                                scope='Conv2d_0b_3x3',
                                trainable=is_trainable,
                                reuse=is_reusable)
                            tower_conv2_2 = slim.conv2d(
                                tower_conv2_1,
                                96,
                                3,
                                scope='Conv2d_0c_3x3',
                                trainable=is_trainable,
                                reuse=is_reusable)
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(
                                net,
                                3,
                                stride=1,
                                padding='SAME',
                                scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(
                                tower_pool,
                                64,
                                1,
                                scope='Conv2d_0b_1x1',
                                trainable=is_trainable,
                                reuse=is_reusable)
                        net = tf.concat([
                            tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1
                        ], 3)
                    end_points[name] = net


            # NON ADAPTABLE PART
            with slim.arg_scope([slim.batch_norm], trainable=False, is_training=False):

                    # BLOCK 35
                    name = "Block35"
                    net = slim.repeat(
                        net,
                        10,
                        block35,
                        scale=0.17,
                        trainable_variables=False,
                        reuse=reuse)

                    net = inception_resnet_v2_core(
                                         net,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=reuse,
                                         scope='InceptionResnetV2',
                                         mode=mode,
                                         trainable_variables=None,
                                         **kwargs
                    )

    return net, end_points


def inception_resnet_v2_adapt_layers_1_6_head(inputs,
                                              dropout_keep_prob=0.8,
                                              bottleneck_layer_size=128,
                                              reuse=None,
                                              scope='InceptionResnetV2',
                                              mode=tf.estimator.ModeKeys.TRAIN,
                                              trainable_variables=None,
                                              is_siamese=True,
                                              is_left=True,
                                              **kwargs):
    """Creates the Inception Resnet V2 model for the adaptation of the
    FIRST AND SIXTH LAYERS

    **Parameters**:

      inputs: a 4-D tensor of size [batch_size, height, width, 3].

      dropout_keep_prob: float, the fraction to keep before final layer.

      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.

      scope: Optional variable_scope.

      trainable_variables: list
        List of variables to be trainable=True

      is_siamese: bool
        If True is Siamese, otherwise is triplet

      is_left: bool
        Is the left side of the Siamese?

    **Returns**:

      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """

    end_points = dict()

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                             stride=1,
                             padding='SAME'):

            # Defining if the branches are reusable or not            
            is_trainable = is_trainable_variable(is_left, mode=tf.estimator.ModeKeys.TRAIN)
            is_reusable = is_reusable_variable(is_siamese, is_left)

            # ADAPTABLE PART
            with slim.arg_scope([slim.dropout], is_training=(mode == tf.estimator.ModeKeys.TRAIN)):

                # CORE OF THE THE ADAPTATION
                with slim.arg_scope([slim.batch_norm], trainable=is_trainable, is_training=is_trainable):
                    # 149 x 149 x 32
                    name = "Conv2d_1a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        inputs,
                        32,
                        3,
                        stride=2,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 147 x 147 x 32
                    name = "Conv2d_2a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        32,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 147 x 147 x 64
                    name = "Conv2d_2b_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        64,
                        3,
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 73 x 73 x 64
                    net = slim.max_pool2d(
                        net,
                        3,
                        stride=2,
                        padding='VALID',
                        scope='MaxPool_3a_3x3')

                    end_points['MaxPool_3a_3x3'] = net

                    # 73 x 73 x 80
                    name = "Conv2d_3b_1x1"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        80,
                        1,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 71 x 71 x 192
                    name = "Conv2d_4a_3x3"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.conv2d(
                        net,
                        192,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=is_trainable,
                        reuse=is_reusable)
                    end_points[name] = net

                    # 35 x 35 x 192
                    net = slim.max_pool2d(
                        net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
                    end_points['MaxPool_5a_3x3'] = net

                    # 35 x 35 x 320
                    name = "Mixed_5b"
                    name = compute_layer_name(name, is_left, is_siamese)
                    with tf.variable_scope(name):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(
                                net,
                                96,
                                1,
                                scope='Conv2d_1x1',
                                trainable=is_trainable,
                                reuse=is_reusable)
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(
                                net,
                                48,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=is_trainable,
                                reuse=is_reusable)
                            tower_conv1_1 = slim.conv2d(
                                tower_conv1_0,
                                64,
                                5,
                                scope='Conv2d_0b_5x5',
                                trainable=is_trainable,
                                reuse=is_reusable)
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(
                                net,
                                64,
                                1,
                                scope='Conv2d_0a_1x1',
                                trainable=is_trainable,
                                reuse=is_reusable)
                            tower_conv2_1 = slim.conv2d(
                                tower_conv2_0,
                                96,
                                3,
                                scope='Conv2d_0b_3x3',
                                trainable=is_trainable,
                                reuse=is_reusable)
                            tower_conv2_2 = slim.conv2d(
                                tower_conv2_1,
                                96,
                                3,
                                scope='Conv2d_0c_3x3',
                                trainable=is_trainable,
                                reuse=is_reusable)
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(
                                net,
                                3,
                                stride=1,
                                padding='SAME',
                                scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(
                                tower_pool,
                                64,
                                1,
                                scope='Conv2d_0b_1x1',
                                trainable=is_trainable,
                                reuse=is_reusable)
                        net = tf.concat([
                            tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1
                        ], 3)
                    end_points[name] = net


                    # BLOCK 35
                    name = "Block35"
                    name = compute_layer_name(name, is_left, is_siamese)
                    net = slim.repeat(
                        net,
                        10,
                        block35,
                        scale=0.17,
                        trainable_variables=is_trainable,
                        scope=name,
                        reuse=is_reusable
                    )

                # CORE OF THE THE ADAPTATION
                with slim.arg_scope([slim.batch_norm], trainable=False, is_training=False):

                    net = inception_resnet_v2_core(
                                         net,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=reuse,
                                         scope='InceptionResnetV2',
                                         mode=mode,
                                         trainable_variables=None,
                                         **kwargs)

    return net, end_points
