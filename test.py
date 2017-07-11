from bob.bio.htface.datashuffler import SiameseDiskHTFace, MeanOffsetHT
from bob.learn.tensorflow.loss import ContrastiveLoss
from bob.learn.tensorflow.trainers import SiameseTrainer, constant
from bob.learn.tensorflow.network import Chopra
from bob.learn.tensorflow.datashuffler import MeanOffset, ImageAugmentation
import bob.io.base

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers.python.layers import layers as layers_lib
import os

directory = "./temp/inception"
database_path = "/Users/tiago.pereira/Documents/database/cuhk_cufs_process"
device = "/cpu:0"
l_rate = 0.01
iterations = 100000


def my_inception_2(inputs, scope='myInception', reuse=False, device="/cpu:0"):
    slim = tf.contrib.slim
    initializer = tf.contrib.layers.xavier_initializer(seed=10)  # Weights initializer

    with tf.device(device):

        with variable_scope.variable_scope(scope, 'InceptionVx', [inputs], reuse=reuse):


            # 224 x 224 x
            graph = slim.conv2d(inputs, 64, [5, 5], activation_fn=tf.nn.relu,
                                stride=2, scope='conv1',
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=initializer)

            # 112 x 112 x 64
            graph = slim.max_pool2d(graph, [3, 3], stride=2,
                                    padding="SAME", scope='pool1')

            # 56 x 56 x 64
            graph = slim.conv2d(graph, 96, [5, 5], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                stride=2, scope='conv2',
                                weights_initializer=initializer)

            # 28 x 28 x 64
            graph = slim.max_pool2d(graph, [3, 3], stride=2,
                                    padding="SAME", scope='pool2')

            # 14 x 14 x 64
            with variable_scope.variable_scope("Inception_1"):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(graph, 32, [1, 1], stride=1, normalizer_fn=slim.batch_norm,
                                           activation_fn=tf.nn.relu, scope='Branch_0_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(graph, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_1_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [3, 3], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_1_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(graph, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_2_1x1')
                    branch_2 = slim.conv2d(branch_2, 64, [5, 5], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_2_5x5')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(graph, [3, 3], padding="SAME", stride=1, scope='Branch_3_pool')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_3_5x5')


                graph = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)


            # 14 x 14 x 192 (32 + 64 + 64 + 32)
            with variable_scope.variable_scope("Inception_2"):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(graph, 32, [1, 1], stride=1, normalizer_fn=slim.batch_norm,
                                           activation_fn=tf.nn.relu, scope='Branch_0_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(graph, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_1_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [3, 3], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_1_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(graph, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_2_1x1')
                    branch_2 = slim.conv2d(branch_2, 64, [5, 5], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_2_5x5')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(graph, [3, 3], padding="SAME", stride=1, scope='Branch_3_pool')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_3_5x5')

                graph = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)


            with variable_scope.variable_scope("Inception_3"):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(graph, 32, [1, 1], stride=1, normalizer_fn=slim.batch_norm,
                                           activation_fn=tf.nn.relu, scope='Branch_0_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(graph, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_1_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [3, 3], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_1_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(graph, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_2_1x1')
                    branch_2 = slim.conv2d(branch_2, 64, [5, 5], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_2_5x5')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(graph, [3, 3], padding="SAME", stride=1, scope='Branch_3_pool')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Branch_3_5x5')

                graph = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)


            # 14 x 14 x 192 (32 + 64 + 64 + 32)
            graph = slim.flatten(graph, scope='flatten1')

            # N x 37.632
            graph = slim.dropout(graph, keep_prob=0.4)

            #graph = slim.fully_connected(graph, 808,
            #                             weights_initializer=initializer,
            #                             activation_fn=tf.nn.relu,
            #                             normalizer_fn=slim.batch_norm,
            #                             scope='fc1',
            #                             reuse=reuse)

            #graph = slim.dropout(graph, keep_prob=0.4)

            graph = slim.fully_connected(graph, 404,
                                         weights_initializer=initializer,
                                         activation_fn=None,
                                         scope='fcN',
                                         reuse=reuse)

            graph = tf.nn.l2_normalize(graph, 1, 1e-10, name="normalizer")

            #graph = slim.softmax(graph, 75, scope='Predictions')

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
database = Database(original_directory=database_path,
                    original_extension=".hdf5",
                    arface_directory="", xm2vts_directory="")

#normalizer = MeanOffsetHT(bob.io.base.load("means_visual.hdf5"), bob.io.base.load("means_sketch.hdf5"))
normalizer = MeanOffset(bob.io.base.load("means.hdf5"))

#train_data_shuffler = SiameseDiskHTFace(database=database, protocol="search_split1_p2s",
#                                        batch_size=8,
#                                        input_shape=[None, 112, 112, 1],
#                                        normalizer=MeanOffset(bob.io.base.load("means.hdf5")))


train_data_shuffler = SiameseDiskHTFace(database=database, protocol="search_split1_p2s",
                                        batch_size=8,
                                        input_shape=[None, 224, 224, 1],
                                        normalizer=normalizer,
                                        data_augmentation=ImageAugmentation())

validation_data_shuffler = SiameseDiskHTFace(database=database, protocol="search_split1_p2s",
                                             batch_size=32,
                                             input_shape=[None, 224, 224, 1],
                                             normalizer=normalizer,
                                             groups="dev",
                                             purposes=["enroll", "probe"])

# Loss for the softmax
loss = ContrastiveLoss()

# Creating inception model
inputs = train_data_shuffler("data", from_queue=False)

from tensorflow.contrib.slim.python.slim.nets import inception
graph = dict()
#chopra = Chopra()
#graph['left'] = chopra(inputs['left'])
#graph['right'] = chopra(inputs['right'], reuse=True)

#graph['left'] = inception.inception_v1(inputs['left'])[0]
#graph['right'] = inception.inception_v1(inputs['right'], reuse=True)[0]

#import ipdb;
#ipdb.set_trace()

graph['left'] = my_inception_2(inputs['left'], device=device)
graph['right'] = my_inception_2(inputs['right'], reuse=True, device=device)


# One graph trainer
trainer = SiameseTrainer(train_data_shuffler,
                         iterations=iterations,
                         analizer=None,
                         temp_dir=directory,
                         validation_snapshot=100
                         )
trainer.create_network_from_scratch(graph=graph,
                                    loss=loss,
                                    learning_rate=constant(l_rate, name="regular_lr"),
                                    optimizer=tf.train.GradientDescentOptimizer(l_rate)
                                    )
trainer.train(validation_data_shuffler)
x = 0
