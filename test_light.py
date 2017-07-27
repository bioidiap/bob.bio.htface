from bob.bio.htface.datashuffler import SiameseDiskHTFace, MeanOffsetHT
from bob.learn.tensorflow.loss import ContrastiveLoss, BaseLoss
from bob.learn.tensorflow.trainers import SiameseTrainer, constant
from bob.learn.tensorflow.network import Chopra, LightCNN9
from bob.learn.tensorflow.datashuffler import MeanOffset, ImageAugmentation, SiameseDisk, Disk
import bob.io.base

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers.python.layers import layers as layers_lib
import os
import numpy

directory = "./temp/inception"
#casia_path = "/Users/tiago.pereira/Documents/database/CASIA-CleanedCroped_gray"
#cuhk_cufs = "/Users/tiago.pereira/Documents/database/cuhk_cufs_process"
#device = "/cpu:0"
#l_rate = 0.1

casia_path = "/home/tiago-ttt/Documents/gitlab/database/CASIA-CleanedCroped_gray"
cuhk_cufs = "/home/tiago-ttt/Documents/gitlab/database/cuhk_cufs_process"
mobio = "/home/tiago-ttt/Documents/gitlab/database/mobio/mobio_cropped_gray"
#device = "/gpu:1"
device = "/cpu:0"
l_rate = 0.1

iterations = 300000

def map_labels(raw_labels):
    """
    Map the clients to 0 to 1
    """
    possible_labels = list(set(raw_labels))
    labels = numpy.zeros(len(raw_labels), dtype="float")
    raw_labels = numpy.array(raw_labels)

    #print len(set(labels))
    for i in range(len(possible_labels)):
      l = float(possible_labels[i])
      #print str(i) + " - " + str(len(numpy.where(labels==l)[0]))
      labels[numpy.where(raw_labels==l)[0]] = float(i)
    #print len(set(labels))
    return labels



# Loading data
normalizer = MeanOffset(bob.io.base.load("means_casia.hdf5"))


# CASIA
import bob.db.casia_webface
db_casia = bob.db.casia_webface.Database()

train_objects = sorted(db_casia.objects(groups="world"), key=lambda x: x.id)
train_labels = map_labels([int(o.client_id) for o in train_objects])
train_file_names = [o.make_path(directory=casia_path, extension=".png")
    for o in train_objects]


train_data_shuffler = Disk(train_file_names, train_labels,
                                  input_shape=[None, 224, 224, 1],
                                  batch_size=32,
                                  normalizer=normalizer,
                                  prefetch=True,
                                  prefetch_capacity=500,
                                  prefetch_threads=2)


# Loss for the softmax
loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

# Creating inception model
inputs = train_data_shuffler("data", from_queue=False)
n_classes = len(set([f.client_id for f in db_casia.objects(groups="world")]))

import ipdb; ipdb.set_trace()


lightcnn9 = LightCNN9(n_classes=n_classes)
lightcnn9 = lightcnn9(inputs)

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
trainer.train()
x = 0
