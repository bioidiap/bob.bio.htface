#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.io.base
import bob.bio.face
from bob.bio.htface.extractor import TensorflowEmbedding
import os

import bob.ip.tensorflow_extractor
import tensorflow as tf
from bob.bio.htface.configs.tensorflow.utils import transfer_128_64_128
from bob.bio.htface.utils import get_cnn_model_name

# The model filename depends on the database and its protocol and those values are
# chain loaded via database.py
model_filename = get_cnn_model_name(temp_dir, "idiap_casia_inception_v2_gray_transfer_64_128",
                                    database.name, protocol)

inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))

# Taking the embedding
prelogits,_ = transfer_128_64_128(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]),
                                  mode=tf.estimator.ModeKeys.PREDICT)
embedding = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")

extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding))

