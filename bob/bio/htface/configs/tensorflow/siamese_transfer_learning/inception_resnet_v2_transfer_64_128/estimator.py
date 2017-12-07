#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

# Calling our base setup
from bob.bio.htface.configs.tensorflow.utils import transfer_128_64_128

import os
import tensorflow as tf
from bob.bio.htface.dataset.siamese_htface import shuffle_data_and_labels_image_augmentation
from bob.learn.tensorflow.utils import reproducible
from bob.learn.tensorflow.estimators import Siamese
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.loss import contrastive_loss
from bob.learn.tensorflow.utils import reproducible


# Training setup
learning_rate = 0.01
data_shape = (160, 160, 1)  # size of atnt images
output_shape = None
data_type = tf.uint8

batch_size = 16
validation_batch_size = 250
epochs = 100
embedding_validation = True
steps = 2000000


run_config = tf.estimator.RunConfig()
run_config = run_config.replace(save_checkpoints_steps=500)

# Preparing the prior
extra_checkpoint = {"checkpoint_path": inception_resnet_v2_casia_webface_gray, 
                    "scopes": dict({"InceptionResnetV2/": "InceptionResnetV2/"}),
                    "trainable_variables": []
                   }

model_dir = os.path.join(inception_resnet_v2_transfer_64_128[database.name], protocol)

def train_input_fn():
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

# Defining our estimator
estimator = Siamese(model_dir=model_dir,
                    architecture=transfer_128_64_128,
                    optimizer=tf.train.AdagradOptimizer(learning_rate),
                    validation_batch_size=validation_batch_size,
                    config=run_config,
                    loss_op=contrastive_loss,
                    extra_checkpoint=extra_checkpoint)

# Defining our hook mechanism
hooks = [LoggerHookEstimator(estimator, 16, 1),
         tf.train.SummarySaverHook(save_steps=5,
                                   output_dir=model_dir,
                                   scaffold=tf.train.Scaffold(),
                                   summary_writer=tf.summary.FileWriter(model_dir))]

