#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.io.base
import bob.bio.face
from bob.bio.htface.extractor import SiameseEmbedding
import os

import bob.ip.tensorflow_extractor
import tensorflow as tf
from bob.bio.htface.architectures.inception_v1_batch_norm import inception_resnet_v1_adapt_layers_1_2_head
from bob.bio.htface.utils import get_cnn_model_name


architecture = inception_resnet_v1_adapt_layers_1_2_head
model_name = "siamese_inceptionv1_adapt_1_2_betas_nonshared_batch_norm"


# The model filename depends on the database and its protocol and those values are
# chain loaded via database.py
model_filename = get_cnn_model_name(temp_dir, model_name,
                                    database.name, protocol)
#bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding)

#from bob.bio.htface.extractor import TensorflowEmbedding
#extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding))


extractor = SiameseEmbedding(model_filename, architecture, shape=(1, 160, 160, 1))

