import numpy
import sys
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, Linear, TFRecord, PerImageStandarization
from bob.learn.tensorflow.network import Embedding, LightCNN9, inception_resnet_v2, Chopra
from bob.learn.tensorflow.loss import contrastive_loss
from bob.learn.tensorflow.trainers import Trainer, constant, SiameseTrainer
from bob.learn.tensorflow.utils import load_mnist
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import vgg
from bob.bio.htface.datashuffler import SiameseDiskHTFace
from bob.bio.htface.architectures import transfer_graph

import bob.io.base
import bob.io.image
import bob.io.base
import bob.io.image
import tensorflow as tf
import shutil
import os

batch_size = 8
validation_batch_size = 400
iterations = 2000000
seed = 10
slim = tf.contrib.slim

non_linearity_outputs=64


directory = "/idiap/temp/tpereira/casia_webface/polathermal/resnet_gray_64-to-128/"
checkpoint_filename = "/idiap/temp/tpereira/casia_webface/official_checkpoints/160x/resnet_inception_gray_centerloss/"


def build_graph(inputs, reuse=False):
    prelogits, _ = inception_resnet_v2(inputs, reuse=reuse, is_training=False)
    return prelogits
    
# Creating the tf record
#filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=100000, name="input")
#filename_queue = None

# Creating the CNN using the TFRecord as input
from bob.db.pola_thermal.query import Database
database = Database(original_directory="/idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/INITIAL_CHECKPOINT/split1/preprocessed/",
                    original_extension=".hdf5")

train_data_shuffler = SiameseDiskHTFace(database=database, protocol="VIS-polarimetric-overall-split1",
                                        batch_size=batch_size,
                                        input_shape=[None, 160, 160, 1],
                                        normalizer=PerImageStandarization())

#train_data_shuffler = SiameseDiskHTFace(database=database, protocol="VIS-polarimetric-overall-split1",
#                                        batch_size=batch_size,
#                                        input_shape=[None, 160, 160, 1],
#                                        normalizer=Linear())


from_checkpoint = False

tf.reset_default_graph()

# Loading the pretrained model
trainer = SiameseTrainer(train_data_shuffler,
                          iterations=iterations,
                          analizer=None,
                          temp_dir=directory)

#import ipdb; ipdb.set_trace();
from_checkpoint = False
if from_checkpoint:
    trainer.create_network_from_file(directory)
else:

    # Loss for the Siamese
    learning_rate=constant(0.01, name="regular_lr")

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdagradOptimizer(learning_rate)

    # Preparing the graph                              

    inputs = train_data_shuffler("data")
    labels = train_data_shuffler("label")
    graph = dict()
    #graph['left'], end_points = build_graph(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs['left'])]), get_embeddings=True)
    graph['left'] = build_graph(inputs['left'])
    graph['left'] = transfer_graph(graph['left'], non_linearity_outputs=non_linearity_outputs)

    #graph['right'], _ = build_graph(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs['right'])]), reuse=True, get_embeddings=True)
    graph['right'] = build_graph(inputs['right'], reuse=True)
    graph['right'] = transfer_graph(graph['right'], reuse=True, non_linearity_outputs=non_linearity_outputs)

    loss = contrastive_loss(graph['left'], graph['right'], labels, contrastive_margin=2.)

    # Bootstraping the siamease                        
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=optimizer)
     
    # Loading the pretrained variables
    var_list = ["InceptionResnetV2"]    
    trainer.load_variables_from_external_model(checkpoint_filename, var_list=var_list)

trainer.train()
#var
# model_snapshot16500.ckp

