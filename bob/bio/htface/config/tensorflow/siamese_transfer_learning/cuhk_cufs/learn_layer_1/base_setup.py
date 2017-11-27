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

# All layers
#shut_down = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
#             "Mixed_5b", "Block35", "Mixed_6a", "Block17", "Mixed_7a",
#             "Block8", "Conv2d_7b_1x1", "Bottleneck"]

trainable_variables = ["Conv2d_1a_3x3"]
extra_checkpoint = {"checkpoint_path": "/idiap/temp/tpereira/casia_webface/new_tf_format/official_checkpoints/inception_resnet_v2_gray/centerloss_alpha-0.95_factor-0.02_lr-0.1/", 
                    "scopes": dict({"InceptionResnetV2/": "InceptionResnetV2/"}),
                    "trainable_variables": trainable_variables
                   }

def learn_first_layer(inputs, reuse=None, mode = tf.estimator.ModeKeys.TRAIN, **kwargs):
    """
    """
    graph, end_points = inception_resnet_v2(inputs, mode=mode, reuse=reuse, **kwargs)

    return graph, end_points

