#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from bob.learn.tensorflow.network import inception_resnet_v2
from bob.bio.htface.architectures.Transfer import build_transfer_graph


def transfer_128_64_128(inputs, reuse=None, mode = tf.estimator.ModeKeys.TRAIN, **kwargs):
    """
    Build a joint encoder on top of the inception network
    """
    graph,_ = inception_resnet_v2(inputs, mode=mode, reuse=reuse, **kwargs)
    graph, end_points = build_transfer_graph(graph, reuse=reuse, bottleneck_layers=[64], outputs=128)

    return graph, end_points

