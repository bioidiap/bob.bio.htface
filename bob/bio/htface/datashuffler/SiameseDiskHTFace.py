#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import bob.core
logger = bob.core.log.setup("bob.learn.tensorflow")

from bob.learn.tensorflow.datashuffler import SiameseDisk
from bob.learn.tensorflow.datashuffler.Normalizer import Linear


class SiameseDiskHTFace(SiameseDisk):
    """
     This :py:class:`bob.learn.tensorflow.datashuffler.Siamese` datashuffler deal with databases that are stored in the disk.
     The data is loaded on the fly,.

     **Parameters**

     database:
       Database package
       
     protocol:
       Protocol name

     input_shape:
       The shape of the inputs

     input_dtype:
       The type of the data,

     batch_size:
       Batch size

     seed:
       The seed of the random number generator

     data_augmentation:
       The algorithm used for data augmentation. Look :py:class:`bob.learn.tensorflow.datashuffler.DataAugmentation`

     normalizer:
       The algorithm used for feature scaling. Look :py:class:`bob.learn.tensorflow.datashuffler.ScaleFactor`, :py:class:`bob.learn.tensorflow.datashuffler.Linear` and :py:class:`bob.learn.tensorflow.datashuffler.MeanOffset`

    """
    def __init__(self, database,
                 protocol,
                 input_shape,
                 input_dtype="float32",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear()):

        #if isinstance(data, list):
        #    data = numpy.array(data)

        #if isinstance(labels, list):
        #    labels = numpy.array(labels)

        numpy.random.seed(seed)
        self.database = database

        #self.db_objects = self.database.objects(self.database.original_directory,
        #                                        self.database.original_extension)

        self.client_ids = self.database.model_ids(protocol=protocol, groups="world")
        self.protocol = protocol

        super(SiameseDisk, self).__init__(
            data=[],
            labels=[],
            input_shape=input_shape,
            input_dtype=input_dtype,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation,
            normalizer=normalizer
        )
        # Seting the seed
        numpy.random.seed(seed)

        # TODO: very bad solution to deal with bob.shape images an tf shape images
        self.bob_shape = tuple([input_shape[3]] + list(input_shape[1:3]))

    def load_data_per_identity_modality(self, identity, modality):

        # Fetching genuine modality A
        objects = self.database.objects(protocol=self.protocol,
                                        groups=["world"],
                                        purposes=["train"],
                                        model_ids=[identity],
                                        modality=[modality])
        numpy.random.shuffle(objects)
        file_name = objects[0].make_path(self.database.original_directory, self.database.original_extension)
        return self.normalize_sample(self.load_from_file(str(file_name)))

    def get_batch(self):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        shape = [self.batch_size] + list(self.input_shape[1:])

        sample_l = numpy.zeros(shape=shape, dtype=self.input_dtype)
        sample_r = numpy.zeros(shape=shape, dtype=self.input_dtype)
        labels_siamese = numpy.zeros(shape=shape[0], dtype=self.input_dtype)

        genuine = True

        for i in range(shape[0]):

            # Shuffling client ids
            numpy.random.shuffle(self.client_ids)
            genuine_index = self.client_ids[0]

            if genuine:
                l = self.load_data_per_identity_modality(genuine_index, self.database.modalities[0])
                r = self.load_data_per_identity_modality(genuine_index, self.database.modalities[1])

            else:
                impostor_index = self.client_ids[1]
                l = self.load_data_per_identity_modality(genuine_index, self.database.modalities[0])
                r = self.load_data_per_identity_modality(impostor_index, self.database.modalities[1])

            sample_l[i, ...] = l
            sample_r[i, ...] = r
            labels_siamese[i] = not genuine

            genuine = not genuine

        return sample_l, sample_r, labels_siamese
