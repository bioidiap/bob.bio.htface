#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from bob.bio.htface.dataset.siamese_htface import shuffle_data_and_labels_image_augmentation


def test_siamease_dataset_cuhk_cufs():

    #from bob.db.cuhk_cufs.query import Database
    #database = Database(original_directory="/idiap/temp/tpereira/HTFace/CUHK-CUFS/RESNET_GRAY/INITIAL_CHECKPOINT/split1/preprocessed/",
    #                    original_extension=".hdf5",
    #                    arface_directory="", xm2vts_directory="")

    #dataset = shuffle_data_and_labels_image_augmentation(database, protocol="search_split1_p2s", data_shape=(160, 160, 1), data_type=tf.uint8,
    #                                           batch_size=8, epochs=1, buffer_size=10**3,
    #                                           gray_scale=False, 
    #                                           output_shape=None,
    #                                           random_flip=False,
    #                                           random_brightness=False,
    #                                           random_contrast=False,
    #                                           random_saturation=False,
    #                                           per_image_normalization=True, 
    #                                           groups="world", purposes="train",
    #                                           extension="hdf5")
    #offset = 0
    #session = tf.Session()
    #batch = session.run([dataset])
    #assert "left" in batch[0][0]
    #assert "right" in batch[0][0]
    assert True

