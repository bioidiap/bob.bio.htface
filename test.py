from bob.bio.htface.datashuffler import SiameseDiskHTFace
from bob.learn.tensorflow.loss import ContrastiveLoss
from bob.learn.tensorflow.trainers import SiameseTrainer, constant
from bob.learn.tensorflow.network import Chopra
from bob.learn.tensorflow.datashuffler import MeanOffset
import bob.io.base

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers.python.layers import layers as layers_lib
import os

directory = "./temp/inception"


def my_inception(inputs, scope='myInception', reuse=False, device="/cpu:0"):
    slim = tf.contrib.slim

    with variable_scope.variable_scope(scope, 'InceptionVx', [inputs], reuse=reuse):


        #initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=10)

        end_point = "conv1"
        graph = layers.conv2d(inputs, 64, [5, 5], stride=2, scope=end_point)

        end_point = "pool1"
        graph = layers_lib.max_pool2d(graph, [2, 2], stride=1, scope=end_point)
        #graph = slim.max_pool2d(graph, [2, 2], scope='pool1')

        #graph = slim.conv2d(graph, 10, [3, 3], activation_fn=tf.nn.relu,
        #                    stride=1,
        #                    weights_initializer=initializer,
        #                    scope='conv2', reuse=reuse)
        #graph = slim.max_pool2d(graph, [2, 2], scope='pool2')

        end_point = "conv2"
        graph = layers.conv2d(graph, 64, [5, 5], stride=2, scope=end_point)

        end_point = "pool2"
        graph = layers_lib.max_pool2d(graph, [2, 2], stride=1, scope=end_point)

        end_point = "conv3"
        graph = layers.conv2d(graph, 32, [5, 5], stride=2, scope=end_point)

        end_point = "pool3"
        graph = layers_lib.max_pool2d(graph, [2, 2], stride=2, scope=end_point)


        with variable_scope.variable_scope("Mixed2"):
            with variable_scope.variable_scope('Branch_0'):
                branch_0 = layers.conv2d(graph, 192, [1, 1], scope='Conv2d_0a_1x1')

            with variable_scope.variable_scope('Branch_1'):
                branch_1 = layers.conv2d(graph, 96, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = layers.conv2d(
                    branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')

            with variable_scope.variable_scope('Branch_2'):
                branch_2 = layers.conv2d(graph, 16, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = layers.conv2d(
                    branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')

            #with variable_scope.variable_scope('Branch_3'):
            #    branch_3 = layers_lib.max_pool2d(
            #        graph, [3, 3], scope='MaxPool_0a_3x3')
            #    branch_3 = layers.conv2d(
            #        branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')

            #graph = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            graph = array_ops.concat([branch_0, branch_1, branch_2], 3)

        with variable_scope.variable_scope('Logits'):
            graph = layers_lib.avg_pool2d( graph, [7, 7], stride=1, scope='MaxPool_0a_7x7')

            graph = layers_lib.dropout(graph, 0.4, scope='Dropout_0b')
            graph = layers.conv2d(
                graph,
                75, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope='Conv2d_0c_1x1')

            #if spatial_squeeze:
            #    logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')
        #import ipdb; ipdb.set_trace()
        graph = slim.flatten(graph, scope='flatten1')
        graph = layers_lib.softmax(graph)


    return graph


def create_architecture(placeholder):
    initializer = tf.contrib.layers.xavier_initializer(seed=10)  # Weights initializer

    slim = tf.contrib.slim
    graph = slim.conv2d(placeholder, 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1',
                        weights_initializer=initializer)
    graph = slim.flatten(graph, scope='flatten1')
    graph = slim.fully_connected(graph, 10, activation_fn=None, scope='fc1', weights_initializer=initializer)

    return graph

#import ipdb; ipdb.set_trace()

# Loading data
from bob.db.cuhk_cufs.query import Database
database = Database(original_directory="/Users/tiago.pereira/Documents/database/cuhk_cufs_process",
                    original_extension=".png",
                    arface_directory="", xm2vts_directory="")

train_data_shuffler = SiameseDiskHTFace(database=database, protocol="cuhk_p2s",
                                        batch_size=8,
                                        input_shape=[None, 224, 224, 1],
                                        normalizer=MeanOffset(bob.io.base.load("means.hdf5")))

# Loss for the softmax
loss = ContrastiveLoss()

# Creating inception model
inputs = train_data_shuffler("data", from_queue=False)

from tensorflow.contrib.slim.python.slim.nets import inception
graph = dict()
chopra = Chopra()
#graph['left'] = chopra(inputs['left'])
#graph['right'] = chopra(inputs['right'], reuse=True)

#graph['left'] = inception.inception_v1(inputs['left'])[0]
#graph['right'] = inception.inception_v1(inputs['right'], reuse=True)[0]

import ipdb;
ipdb.set_trace()

graph['left'] = my_inception(inputs['left'])
graph['right'] = my_inception(inputs['right'], reuse=True)


# One graph trainer
iterations = 100
trainer = SiameseTrainer(train_data_shuffler,
                         iterations=iterations,
                         analizer=None,
                         temp_dir=directory
                         )
trainer.create_network_from_scratch(graph=graph,
                                    loss=loss,
                                    learning_rate=constant(0.01, name="regular_lr"),
                                    optimizer=tf.train.GradientDescentOptimizer(0.01)
                                    )
trainer.train()
x = 0