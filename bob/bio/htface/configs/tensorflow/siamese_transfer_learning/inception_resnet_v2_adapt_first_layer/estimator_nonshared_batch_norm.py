#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

# Calling our base setup
from bob.bio.htface.architectures.inception_v2_batch_norm import inception_resnet_v2_adapt_first_head

import os
import tensorflow as tf
#from tensorflow.python import debug as tf_debug

from bob.bio.htface.dataset.siamese_htface import shuffle_data_and_labels_image_augmentation
from bob.learn.tensorflow.utils import reproducible
from bob.bio.htface.estimators import SiameseAdaptation
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.loss import contrastive_loss
from bob.learn.tensorflow.utils import reproducible
from bob.bio.htface.utils import get_cnn_model_name


architecture = inception_resnet_v2_adapt_first_head
model_name = "idiap_casia_inception_v2_gray_adapt_first_layer_nonshared_batch_norm"


# Training setup
learning_rate_values=[0.1, 0.01, 0.001]
learning_rate_boundaries=[400, 700, 700]

data_shape = (160, 160, 1)  # size of atnt images
output_shape = None
data_type = tf.uint8

batch_size = 90
validation_batch_size = 250
epochs = 200
embedding_validation = True
steps = 2000000


run_config = tf.estimator.RunConfig()
run_config = run_config.replace(save_checkpoints_steps=500)


#INCEPTION V2 LAYER ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
#             "Mixed_5b", "Block35", "Mixed_6a", "Block17", "Mixed_7a",
#             "Block8", "Conv2d_7b_1x1", "Bottleneck"]


# Preparing the checkpoint loading
left_scope = dict()
left_scope['InceptionResnetV2/Conv2d_1a_3x3/'] = "InceptionResnetV2/Conv2d_1a_3x3_left/"
left_scope['InceptionResnetV2/Conv2d_2a_3x3/'] = "InceptionResnetV2/Conv2d_2a_3x3/"
left_scope['InceptionResnetV2/Conv2d_2b_3x3/'] = "InceptionResnetV2/Conv2d_2b_3x3/"
left_scope['InceptionResnetV2/Conv2d_3b_1x1/'] = "InceptionResnetV2/Conv2d_3b_1x1/"
left_scope['InceptionResnetV2/Conv2d_4a_3x3/'] = "InceptionResnetV2/Conv2d_4a_3x3/"
left_scope['InceptionResnetV2/Repeat/'] = "InceptionResnetV2/Repeat/" # TF-SLIM ADD the prefix repeat unde each repeat
left_scope['InceptionResnetV2/Repeat_1/'] = "InceptionResnetV2/Repeat_1/" # TF-SLIM ADD the prefix repeat unde each repeat  
left_scope['InceptionResnetV2/Repeat_2/'] = "InceptionResnetV2/Repeat_2/" # TF-SLIM ADD the prefix repeat unde each repeat    

# JUst to be sure
left_scope['InceptionResnetV2/Mixed_5b/'] = "InceptionResnetV2/Mixed_5b/"
left_scope['InceptionResnetV2/Block35/'] = "InceptionResnetV2/Block35/"
left_scope['InceptionResnetV2/Mixed_6a/'] = "InceptionResnetV2/Mixed_6a/"
left_scope['InceptionResnetV2/Block17/'] = "InceptionResnetV2/Block17/"
left_scope['InceptionResnetV2/Mixed_7a/'] = "InceptionResnetV2/Mixed_7a/"
left_scope['InceptionResnetV2/Block8/'] = "InceptionResnetV2/Block8/"
left_scope['InceptionResnetV2/Conv2d_7b_1x1/'] = "InceptionResnetV2/Conv2d_7b_1x1/"
left_scope['InceptionResnetV2/Bottleneck/'] = "InceptionResnetV2/Bottleneck/"
left_scope['InceptionResnetV2/Logits/'] = "InceptionResnetV2/Logits/"

right_scope = dict()
right_scope['InceptionResnetV2/Conv2d_1a_3x3/'] = "InceptionResnetV2/Conv2d_1a_3x3_right/"


# Preparing the prior
extra_checkpoint = {"checkpoint_path": inception_resnet_v2_casia_webface_gray_batch_norm, 
                    "scopes": [left_scope, right_scope]
                   }

model_dir = get_cnn_model_name(temp_dir, model_name,
                               database.name, protocol)


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
optimizer = tf.train.AdagradOptimizer
estimator = SiameseAdaptation(model_dir=model_dir,
                              architecture=architecture,
                              optimizer=optimizer,
                              validation_batch_size=validation_batch_size,
                              config=run_config,
                              loss_op=contrastive_loss,
                              extra_checkpoint=extra_checkpoint,
                              learning_rate_values=learning_rate_values,
                              learning_rate_boundaries=learning_rate_boundaries,
                              )

# Defining our hook mechanism
hooks = [LoggerHookEstimator(estimator, 16, 1),
         tf.train.SummarySaverHook(save_steps=5,
                                   output_dir=model_dir,
                                   scaffold=tf.train.Scaffold(),
                                   summary_writer=tf.summary.FileWriter(model_dir))]

