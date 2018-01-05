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


def compute_layer_name(name, is_left):
    if is_left:
        name = name + "_left"
    else:
        name = name + "_right"
        
    return name


def inception_resnet_v2_adapt_first_head(inputs,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=None,
                                         scope='InceptionResnetV2',
                                         mode=tf.estimator.ModeKeys.TRAIN,
                                         trainable_variables=None,
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
      
      is_left: bool
        Is the left side of the Siamese?
      
    **Returns**:
    
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
                                         
                                          
    end_points = dict()

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    #'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    
    weight_decay = 5e-5
    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                             stride=1,
                             padding='SAME'):
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                # Initializing             
                is_trainable = True
                if is_left:
                    # Shut down the batch normalization                
                    batch_norm_params.pop("variables_collections")
                    batch_norm_params['trainable'] =  False
                    is_trainable = False
            else:
                #LEFT is NON trainable and  right is TRAINABLE            
                is_trainable = False
                batch_norm_params.pop("variables_collections")
                batch_norm_params['trainable'] =  False
            
            #with slim.arg_scope(
            #    [slim.conv2d, slim.fully_connected],
            #        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            #        weights_regularizer=slim.l2_regularizer(weight_decay),
            #        normalizer_fn=slim.batch_norm,
            #        normalizer_params=batch_norm_params):
                                                                        
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=(mode == tf.estimator.ModeKeys.TRAIN)):

                # CORE OF THE THE ADAPTATION                    
                
                # 149 x 149 x 32
                name = "Conv2d_1a_3x3"
                name = compute_layer_name(name, is_left)
                net = slim.conv2d(
                    inputs,
                    32,
                    3,
                    stride=2,
                    padding='VALID',
                    scope=name,
                    trainable=is_trainable,
                    reuse=False)
                                              
                end_points[name] = net

            with slim.arg_scope([slim.batch_norm],is_training=False):

                    #with slim.arg_scope(
                    #    [slim.conv2d, slim.fully_connected],
                    #        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    #        weights_regularizer=slim.l2_regularizer(weight_decay),
                    #        normalizer_fn=slim.batch_norm,
                    #        normalizer_params=batch_norm_params,
                    #        trainable=False):
    
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


def inception_resnet_v2_adapt_layers_1_4_head(inputs,
                                         dropout_keep_prob=0.8,
                                         bottleneck_layer_size=128,
                                         reuse=None,
                                         scope='InceptionResnetV2',
                                         mode=tf.estimator.ModeKeys.TRAIN,
                                         trainable_variables=None,
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
      
      is_left: bool
        Is the left side of the Siamese?
      
    **Returns**:
    
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
                                         
                                          
    end_points = dict()

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    
    weight_decay = 5e-5
    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                             stride=1,
                             padding='SAME'):

            # Initializing 
            
            if is_left:
                # Shut down the batch normalization                
                batch_norm_params.pop("variables_collections")
                batch_norm_params['trainable'] =  False            
            
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
                                                                        
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=(mode == tf.estimator.ModeKeys.TRAIN)):

                    # CORE OF THE THE ADAPTATION                    
                    #LEFT is NON trainable and  right is TRAINABLE
                    
                    # 149 x 149 x 32
                    name = "Conv2d_1a_3x3"
                    name = compute_layer_name(name, is_left)
                    net = slim.conv2d(
                        inputs,
                        32,
                        3,
                        stride=2,
                        padding='VALID',
                        scope=name,
                        trainable=not is_left,
                        reuse=False)                                                  
                    end_points[name] = net
                    
                    # 147 x 147 x 32
                    name = "Conv2d_2a_3x3"
                    name = compute_layer_name(name, is_left)
                    net = slim.conv2d(
                        net,
                        32,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=not is_left,
                        reuse=False)
                    end_points[name] = net


                    # 147 x 147 x 64
                    name = "Conv2d_2b_3x3"
                    name = compute_layer_name(name, is_left)
                    net = slim.conv2d(
                        net, 64, 3,
                        scope=name,
                        trainable=not is_left,
                        reuse=False)
                    end_points[name] = net

                    # 73 x 73 x 64
                    net = slim.max_pool2d(
                        net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
                    end_points['MaxPool_3a_3x3'] = net

                    # 73 x 73 x 80
                    name = "Conv2d_3b_1x1"
                    name = compute_layer_name(name, is_left)                    
                    net = slim.conv2d(
                        net,
                        80,
                        1,
                        padding='VALID',
                        scope=name,
                        trainable=not is_left,
                        reuse=False)
                    end_points[name] = net

                    # 71 x 71 x 192
                    name = "Conv2d_4a_3x3"
                    name = compute_layer_name(name, is_left)                    
                    net = slim.conv2d(
                        net,
                        192,
                        3,
                        padding='VALID',
                        scope=name,
                        trainable=not is_left,
                        reuse=False)
                    end_points[name] = net

                    # 35 x 35 x 192
                    net = slim.max_pool2d(
                        net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
                    end_points['MaxPool_5a_3x3'] = net                    
                    

                # Shut down the batch normalization (if is left, the thing was already shut)
                if not is_left:
                    batch_norm_params.pop("variables_collections")
                    batch_norm_params['trainable'] =  False
                
                #, slim.dropout
                with slim.arg_scope([slim.batch_norm],is_training=False):

                    with slim.arg_scope(
                        [slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            trainable=False):
    

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
                                             mode=tf.estimator.ModeKeys.TRAIN,
                                             trainable_variables=None,
                                             **kwargs                    
                        )
                    
    return net, end_points



