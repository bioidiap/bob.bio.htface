#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
model_filename = "/idiap/temp/tpereira/casia_webface/cuhk_cufs/siamese-transfer-firstlayer-idiap_inception_v2_gray--casia/split3/"

import bob.ip.tensorflow_extractor
from bob.learn.tensorflow.network import inception_resnet_v2
from bob.bio.htface.extractor import TensorflowEmbedding

### Preparing the input of the extractor ####
inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))

# Taking the embedding
prelogits,_ = inception_resnet_v2(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]),
                                  mode=tf.estimator.ModeKeys.PREDICT)
embedding = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")


extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding))

