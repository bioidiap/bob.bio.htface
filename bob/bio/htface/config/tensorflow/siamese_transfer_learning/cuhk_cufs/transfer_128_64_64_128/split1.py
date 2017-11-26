#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import os
import tensorflow as tf
from bob.bio.htface.dataset.siamese_htface import shuffle_data_and_labels_image_augmentation
from bob.learn.tensorflow.utils import reproducible
from bob.learn.tensorflow.network import inception_resnet_v2
from bob.learn.tensorflow.estimators import Siamese
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.loss import contrastive_loss
from bob.bio.htface.dataset.siamese_htface import shuffle_data_and_labels_image_augmentation
from bob.learn.tensorflow.utils import reproducible


from bob.bio.htface.architectures.Transfer import build_transfer_graph

#### SETTING SPLIT ####
model_dir = "/idiap/temp/tpereira/casia_webface/cuhk_cufs/siamese-transfer-128-64-64-128-idiap_inception_v2_gray--casia/split1"
protocol = "search_split1_p2s"



# Model for transfer
extra_checkpoint = {"checkpoint_path": "/idiap/temp/tpereira/casia_webface/new_tf_format/official_checkpoints/inception_resnet_v2_gray/centerloss_alpha-0.95_factor-0.02_lr-0.1/", 
                    "scopes": dict({"InceptionResnetV2/": "InceptionResnetV2/"}),
                    "is_trainable": False
                   }

learning_rate = 0.01
data_shape = (160, 160, 1)  # size of atnt images
output_shape = None
data_type = tf.uint8

batch_size = 16
validation_batch_size = 250
epochs = 100
embedding_validation = True


def build_graph(inputs, reuse=None, mode = tf.estimator.ModeKeys.TRAIN, trainable_variables=True):
    
    graph,_ = inception_resnet_v2(inputs, mode=mode, reuse=reuse, trainable_variables=False)
    graph, end_points = build_transfer_graph(graph, reuse=reuse, bottleneck_layers=[64, 64], outputs=128)

    return graph, end_points


def train_input_fn():
    from bob.db.cuhk_cufs.query import Database
    database = Database(original_directory="/idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split1/preprocessed/",
                        original_extension=".hdf5")


    return shuffle_data_and_labels_image_augmentation(database, protocol, data_shape, data_type,
                                                      batch_size, epochs=epochs, buffer_size=10**3,
                                                      gray_scale=False, 
                                                      output_shape=None,
                                                      random_flip=True, random_brightness=False,
                                                      random_contrast=False,
                                                      random_saturation=False,
                                                      per_image_normalization=True, 
                                                      groups="world", purposes="train",
                                                      extension="hdf5")

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(save_checkpoints_steps=500)

estimator = Siamese(model_dir=model_dir,
                    architecture=build_graph,
                    optimizer=tf.train.AdagradOptimizer(learning_rate),
                    validation_batch_size=validation_batch_size,
                    config=run_config,
                    loss_op=contrastive_loss,
                    extra_checkpoint=extra_checkpoint)

hooks = [LoggerHookEstimator(estimator, 16, 1),
         tf.train.SummarySaverHook(save_steps=5,
                                   output_dir=model_dir,
                                   scaffold=tf.train.Scaffold(),
                                   summary_writer=tf.summary.FileWriter(model_dir) )]


