#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import numpy
import bob.core
from functools import partial
import tensorflow as tf
from bob.learn.tensorflow.dataset.siamese_image import image_augmentation_parser

numpy.random.seed(10)
logger = bob.core.log.setup("bob.learn.tensorflow")


def shuffle_data_and_labels_image_augmentation(database, protocol, data_shape, data_type,
                                               batch_size, epochs=None, buffer_size=10**3,
                                               gray_scale=False, 
                                               output_shape=None,
                                               random_flip=False,
                                               random_brightness=False,
                                               random_contrast=False,
                                               random_saturation=False,
                                               per_image_normalization=True, 
                                               groups="world", purposes="train",
                                               extension=None):
    """
    Dump random batches for siamese networks using heterogeneous face databases
    
    The batches returned with tf.Session.run() with be in the following format:
    **data** a dictionary containing the keys ['left', 'right'], each one representing 
    one element of the pair and **labels** which is [0, 1] where 0 is the genuine pair
    and 1 is the impostor pair.
    
    The left side will be with modality `0` and the right side will be with modality `1`

    **Parameters**

       database:
          List containing the path of the images
       
       protocol:
          List containing the labels (needs to be in EXACT same order as filenames)
          
       data_shape:
          Samples shape saved in the tf-record
          
       data_type:
          tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)
     
       batch_size:
          Size of the batch
          
       epochs:
           Number of epochs to be batched
       
       buffer_size:
            Size of the shuffle bucket

       gray_scale:
          Convert to gray scale?
          
       output_shape:
          If set, will randomly crop the image given the output shape

       random_flip:
          Randomly flip an image horizontally  (https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right)

       random_brightness:
           Adjust the brightness of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_brightness)

       random_contrast:
           Adjust the contrast of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_contrast)

       random_saturation:
           Adjust the saturation of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_saturation)

       per_image_normalization:
           Linearly scales image to have zero mean and unit norm.            
           
       groups:
       
       purposes:
       
       extension:
           If None, will load files using `tf.image.decode..` if set to `hdf5`, will load with `bob.io.base.load`

    """    
    parser = partial(image_augmentation_parser,
                 data_shape=data_shape,
                 data_type=data_type,
                 gray_scale=gray_scale, 
                 output_shape=output_shape,
                 random_flip=random_flip,
                 random_brightness=random_brightness,
                 random_contrast=random_contrast,
                 random_saturation=random_saturation,
                 per_image_normalization=per_image_normalization,
                 extension=extension)

    left_data, right_data, siamese_labels = siamese_htface_generator(database, protocol, groups, purposes)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((left_data, right_data, siamese_labels))
    dataset = dataset.map(parser)

    # Shuffling
    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat(epochs)
    #dataset = dataset.prefetch(1) TODO: FOR THE r1.4
    data, labels = dataset.make_one_shot_iterator().get_next()
    return data, labels


def siamese_htface_generator(database, protocol, groups="world", purposes="train"):
                 
    left_data = []
    right_data = []
    labels = []
    def append(left, right, label):
        """
        Just appending one element in each list
        """
        left_data.append(left)
        right_data.append(right)
        labels.append(label)
                           
    client_ids = database.model_ids_with_protocol(protocol=protocol, groups=groups)
                                              
    # List of samples from modality A
    samples_A = database.objects(protocol=protocol,
                                 groups=groups,
                                 purposes=purposes,
                                 modality=database.modalities[0])

    # Samples from modality B sorted by identiy
    samples_B = list_2_dict(database.objects(protocol=protocol,
                            groups=groups,
                            purposes=purposes,
                            modality=database.modalities[1]))

    genuine = True
    for o in samples_A:

        reference_identity = o.client_id
        left = o.make_path(database.original_directory, database.original_extension)
        if genuine:
            # Loading genuine pair
            right_object = samples_B[reference_identity].get_object()
            label = 0
        else:
            # Loading impostor pair
            label = 1
            while True:
                index = numpy.random.randint(len(samples_B.keys()))
                if samples_B.keys()[index] != o.client_id:
                    right_object = samples_B[samples_B.keys()[index]].get_object()
                    break    
                                              
        right = right_object.make_path(database.original_directory, database.original_extension)
        genuine = not genuine

        #yield left, right, label
        append(left, right, label)

    return left_data, right_data, labels



