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

def transfer_128_64_128(inputs, reuse=None, mode = tf.estimator.ModeKeys.TRAIN, trainable_variables=True):
    """
    Build a joint encoder on top of the inception network
    """
    
    graph,_ = inception_resnet_v2(inputs, mode=mode, reuse=reuse, trainable_variables=False)
    graph, end_points = build_transfer_graph(graph, reuse=reuse, bottleneck_layers=[64], outputs=128)

    return graph, end_points

