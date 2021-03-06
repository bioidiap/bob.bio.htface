#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.htface.extractor import TripletEmbedding
import tensorflow as tf
from bob.bio.htface.architectures.inception_v2_batch_norm import inception_resnet_v2_adapt_layers_1_4_head
from bob.bio.htface.utils import get_cnn_model_name

architecture = inception_resnet_v2_adapt_layers_1_4_head
model_name = "triplet_inceptionv2_layers_1_4_nonshared_batch_norm"


# The model filename depends on the database and its protocol and those values are
# chain loaded via database.py
model_filename = get_cnn_model_name(temp_dir, model_name,
                                    database_name, protocol)

extractor = TripletEmbedding(model_filename, architecture, shape=(1, 160, 160, 1))

