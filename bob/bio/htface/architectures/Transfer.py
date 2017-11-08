#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


import tensorflow as tf
slim = tf.contrib.slim

def transfer_graph(inputs, reuse=False, non_linearity_outputs=64, bottleneck_outputs=128):

    with tf.variable_scope('Transfer', reuse=reuse):
        end_points={}
        prelogits = slim.fully_connected(inputs, non_linearity_outputs, activation_fn=tf.nn.relu, 
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                                         weights_regularizer=slim.l2_regularizer(0.2),
                                         scope='NonLinearity')
        end_points['NonLinearity1'] = prelogits

        #prelogits = slim.batch_norm(prelogits,
        #                           decay=0.95,
        #                           epsilon= 0.001
        #                           )

        prelogits = slim.fully_connected(prelogits, bottleneck_outputs, activation_fn=None, 
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                         scope='Bottleneck')
        end_points['Bottleneck2'] = prelogits

    return prelogits, end_points

