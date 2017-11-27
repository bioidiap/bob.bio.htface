# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V2 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from .InceptionResnetV2 import block35, block17, block8, inference
  

def inception_resnet_v2(inputs, 
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV2',
                        mode = tf.estimator.ModeKeys.TRAIN,
                        trainable_variables=True,
                        **kwargs):
    """
    Creates the Inception Resnet V2 model.
    That switches one some layers
   
    Parameters
    ----------
    
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      
      dropout_keep_prob: float, the fraction to keep before final layer.
      
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
        
      scope: Optional variable_scope.
      
      mode:
      
      trainable_variables:
      
    Returns    
    -------    
      logits: the logits outputs of the model.

      end_points: the set of end_points from the inception model.
    """
    end_points = {}
    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=(mode == tf.estimator.ModeKeys.TRAIN)):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3', trainable=trainable_variables, reuse=reuse)
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3', trainable=trainable_variables, reuse=reuse)
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3', trainable=trainable_variables, reuse=reuse)
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1', trainable=trainable_variables, reuse=reuse)
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3', trainable=trainable_variables, reuse=reuse)
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 192
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_5a_3x3')
                end_points['MaxPool_5a_3x3'] = net
        
                ##### SHUTTING FROM HERE
                trainable_variables=False
        
                # 35 x 35 x 320
                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1', trainable=trainable_variables, reuse=reuse)
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1', trainable=trainable_variables, reuse=reuse)
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                    scope='Conv2d_0b_5x5', trainable=trainable_variables, reuse=reuse)
                    with tf.variable_scope('Branch_2'):
                        tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1', trainable=trainable_variables, reuse=reuse)
                        tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                    scope='Conv2d_0b_3x3', trainable=trainable_variables, reuse=reuse)
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                    scope='Conv2d_0c_3x3', trainable=trainable_variables, reuse=reuse)
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                                     scope='AvgPool_0a_3x3')
                        tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                                   scope='Conv2d_0b_1x1', trainable=trainable_variables, reuse=reuse)
                    net = tf.concat([tower_conv, tower_conv1_1,
                                        tower_conv2_2, tower_pool_1], 3)
        
                end_points['Mixed_5b'] = net
                net = slim.repeat(net, 10, block35, scale=0.17, trainable_variables=trainable_variables, reuse=reuse)
        
                # 17 x 17 x 1024
                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                                 scope='Conv2d_1a_3x3', trainable=trainable_variables, reuse=reuse)
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1', trainable=trainable_variables, reuse=reuse)
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                    scope='Conv2d_0b_3x3', trainable=trainable_variables, reuse=reuse)
                        tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                    stride=2, padding='VALID',
                                                    scope='Conv2d_1a_3x3', trainable=trainable_variables, reuse=reuse)
                    with tf.variable_scope('Branch_2'):
                        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
        
                end_points['Mixed_6a'] = net
                net = slim.repeat(net, 20, block17, scale=0.10,trainable_variables=trainable_variables, reuse=reuse)
        
                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1', trainable=trainable_variables, reuse=reuse)
                        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                                   padding='VALID', scope='Conv2d_1a_3x3', trainable=trainable_variables, reuse=reuse)
                    with tf.variable_scope('Branch_1'):
                        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1', trainable=trainable_variables, reuse=reuse)
                        tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3', trainable=trainable_variables, reuse=reuse)
                    with tf.variable_scope('Branch_2'):
                        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1', trainable=trainable_variables, reuse=reuse)
                        tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                    scope='Conv2d_0b_3x3', trainable=trainable_variables, reuse=reuse)
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3', trainable=trainable_variables, reuse=reuse)
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                    net = tf.concat([tower_conv_1, tower_conv1_1,
                                        tower_conv2_2, tower_pool], 3)
        
                end_points['Mixed_7a'] = net
        
                net = slim.repeat(net, 9, block8, scale=0.20,trainable_variables=trainable_variables, reuse=reuse)
                net = block8(net, activation_fn=None,trainable_variables=trainable_variables, reuse=reuse)
        
                net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1', trainable=trainable_variables, reuse=reuse)
                end_points['Conv2d_7b_1x1'] = net
        
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
          
                    net = slim.dropout(net, dropout_keep_prob,
                                       scope='Dropout')
          
                    end_points['PreLogitsFlatten'] = net
                
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=reuse, trainable=trainable_variables)
                end_points['Bottleneck'] = net

    return net, end_points
