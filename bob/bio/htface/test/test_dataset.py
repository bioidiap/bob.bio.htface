#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from bob.bio.htface.dataset.siamese_htface import shuffle_data_and_labels_image_augmentation
from bob.learn.tensorflow.utils import reproducible

def test_siamease_dataset_cuhk_cufs():
    """
    from bob.db.cuhk_cufs.query import Database
    database = Database(original_directory="/idiap/temp/tpereira/HTFace/CUHK-CUFS/RESNET_GRAY/INITIAL_CHECKPOINT/split1/preprocessed/",
                        original_extension=".hdf5",
                        arface_directory="", xm2vts_directory="")

    dataset = shuffle_data_and_labels_image_augmentation(database, protocol="search_split1_p2s", data_shape=(160, 160, 1), data_type=tf.uint8,
                                               batch_size=8, epochs=1, buffer_size=10**3,
                                               gray_scale=False, 
                                               output_shape=None,
                                               random_flip=False,
                                               random_brightness=False,
                                               random_contrast=False,
                                               random_saturation=False,
                                               per_image_normalization=False, 
                                               groups="world", purposes="train",
                                               extension="hdf5")
    offset = 0
    session = tf.Session()
    batch = session.run([dataset])
    assert "left" in batch[0][0]
    assert "right" in batch[0][0]
    
    import bob.io.base
    import bob.io.image
    #bob.io.base.save(batch[0][0]['right'][0,:,:,0].astype("uint8"), "0_right.png")
    #bob.io.base.save(batch[0][0]['left'][0,:,:,0].astype("uint8"), "0_left.png")    

    #bob.io.base.save(batch[0][0]['right'][3,:,:,0].astype("uint8"), "1_right.png")
    #bob.io.base.save(batch[0][0]['left'][3,:,:,0].astype("uint8"), "1_left.png")    

    print batch[0][1]
    """
    
    assert True


def test_siamease_dataset_polathermal():
    """
    from bob.db.pola_thermal.query import Database
    database = Database(original_directory="/idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/INITIAL_CHECKPOINT/split1/preprocessed/",
                        original_extension=".hdf5")

    dataset = shuffle_data_and_labels_image_augmentation(database, protocol="VIS-polarimetric-overall-split1", data_shape=(160, 160, 1), data_type=tf.uint8,
                                               batch_size=8, epochs=1, buffer_size=10**3,
                                               gray_scale=False, 
                                               output_shape=None,
                                               random_flip=False,
                                               random_brightness=False,
                                               random_contrast=False,
                                               random_saturation=False,
                                               per_image_normalization=True, 
                                               groups="world", purposes="train",
                                               extension="hdf5")
    offset = 0
    session = tf.Session()
    labels = [] 
    while True:
        try:
            batch = session.run([dataset])
            assert "left" in batch[0][0]
            assert "right" in batch[0][0]
            labels += list(batch[0][1])
        except:
            break
    import numpy
    print numpy.mean(batch[0][0]['left'])
    
    #import bob.io.base
    #import bob.io.image
    #bob.io.base.save(batch[0][0]['right'][0,:,:,0].astype("uint16"), "0_right.png")
    #bob.io.base.save(batch[0][0]['left'][0,:,:,0].astype("uint16"), "0_left.png")    

    #bob.io.base.save(batch[0][0]['right'][1,:,:,0].astype("uint16"), "1_right.png")
    #bob.io.base.save(batch[0][0]['left'][1,:,:,0].astype("uint16"), "1_left.png")    
    #print batch[0][1]
    """    
    assert True

