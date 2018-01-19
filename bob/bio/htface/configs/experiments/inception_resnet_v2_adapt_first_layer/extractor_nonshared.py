#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.io.base
import bob.bio.face
from bob.bio.htface.extractor import SiameseEmbedding
import os

import bob.ip.tensorflow_extractor
import tensorflow as tf
from bob.bio.htface.architectures.inception_v2 import inception_resnet_v2_adapt_first_head
from bob.bio.htface.utils import get_cnn_model_name

# The model filename depends on the database and its protocol and those values are
# chain loaded via database.py
model_filename = get_cnn_model_name(temp_dir, "idiap_casia_inception_v2_gray_adapt_first_layer_nonshared",
                                    database.name, protocol)
#bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding)

#from bob.bio.htface.extractor import TensorflowEmbedding
#extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding))


extractor = SiameseEmbedding(model_filename, inception_resnet_v2_adapt_first_head, shape=(1, 160, 160, 1))

