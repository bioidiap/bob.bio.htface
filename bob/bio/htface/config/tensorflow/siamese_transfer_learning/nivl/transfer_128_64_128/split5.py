#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


model_dir = "/idiap/temp/tpereira/casia_webface/nivl/siamese-transfer-128-64-128-idiap_inception_v2_gray--casia/split5"
protocol = "idiap-search_VIS-NIR_split5"

# Calling our base setup
from bob.bio.htface.config.tensorflow.siamese_transfer_learning.nivl.base_setup import *
from bob.bio.htface.config.tensorflow.siamese_transfer_learning.nivl.transfer_128_64_128.base_setup import *

# I know it's redundant, but I'm just making this explicit
architecture = transfer_architecture

import bob.db.nivl

# Sanity checks
assert extra_checkpoint["trainable_variables"] == []
assert isinstance(database, bob.db.nivl.Database)

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
                    architecture=architecture,
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

