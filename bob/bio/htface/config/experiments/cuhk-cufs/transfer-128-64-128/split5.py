#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
model_filename = "/idiap/temp/tpereira/casia_webface/cuhk_cufs/siamese-transfer-128-64-128-idiap_inception_v2_gray--casia/split5/"

import bob.ip.tensorflow_extractor
from bob.learn.tensorflow.network import inception_resnet_v2
from bob.bio.htface.extractor import TensorflowEmbedding
from bob.bio.htface.architectures.Transfer import build_transfer_graph


def build_graph(inputs, reuse=None, mode = tf.estimator.ModeKeys.TRAIN, trainable_variables=True):
    
    graph,_ = inception_resnet_v2(inputs, mode=mode, reuse=reuse, trainable_variables=False)
    graph, end_points = build_transfer_graph(graph, reuse=reuse, bottleneck_layers=[64], outputs=128)

    return graph



### Preparing the input of the extractor ####
inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))

# Taking the embedding
prelogits = build_graph(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]),
                                  mode=tf.estimator.ModeKeys.PREDICT)
embedding = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")


extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding))

