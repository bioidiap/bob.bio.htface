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


# Setting the input source of the data
input_data_dir = "/idiap/temp/tpereira/HTFace/CUHK-CUFSF/idiap_inception_v2_gray--casia/split1/preprocessed/"
input_extension = ".hdf5"

from bob.db.cuhk_cufsf.query import Database
database = Database(original_directory=input_data_dir,
                    original_extension=input_extension)

# base Model for transfer learning
extra_checkpoint = {"checkpoint_path": "/idiap/temp/tpereira/casia_webface/new_tf_format/official_checkpoints/inception_resnet_v2_gray/centerloss_alpha-0.95_factor-0.02_lr-0.1/", 
                    "scopes": dict({"InceptionResnetV2/": "InceptionResnetV2/"}),
                   }

# Training varibles
learning_rate = 0.01
data_shape = (160, 160, 1)  # size of atnt images
output_shape = None
data_type = tf.uint8

batch_size = 16
validation_batch_size = 250
epochs = 200
embedding_validation = True


run_config = tf.estimator.RunConfig()
run_config = run_config.replace(save_checkpoints_steps=500)

