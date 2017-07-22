#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
numpy.random.seed(10)
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
                 normalizer=Linear(),
                 groups="world",
                 purposes=["train"]):

        #if isinstance(data, list):
        #    data = numpy.array(data)

        #if isinstance(labels, list):
        #    labels = numpy.array(labels)

        self.groups = groups
        self.purposes = purposes

        numpy.random.seed(seed)
        self.database = database

        #self.db_objects = self.database.objects(self.database.original_directory,
        #                                        self.database.original_extension)

        self.client_ids = self.database.model_ids(protocol=protocol, groups=self.groups)
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

    def list_2_dict(self, list_objects):

        dict_objects = dict()
        for l in list_objects:
            if l.client_id not in dict_objects:
                dict_objects[l.client_id] = Counter()
            dict_objects[l.client_id].objects.append(l)

        for d in dict_objects:
            numpy.random.shuffle(dict_objects[d].objects)

        return dict_objects

    def _fetch_batch(self):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        # List of samples from modality A
        samples_A = self.database.objects(protocol=self.protocol,
                                          groups=[self.groups],
                                          purposes=self.purposes,
                                          modality=[self.database.modalities[0]])

        # Samples from modality B sorted by identiy
        samples_B = self.list_2_dict(self.database.objects(protocol=self.protocol,
                                          groups=[self.groups],
                                          purposes=self.purposes,
                                          modality=[self.database.modalities[1]]))
        genuine = True
        for o in samples_A:

            reference_identity = o.client_id
            left = self.load_from_file(o.make_path(self.database.original_directory, self.database.original_extension))
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

            right = self.load_from_file(right_object.make_path(self.database.original_directory, self.database.original_extension))

            # Applying the data augmentation
            if self.data_augmentation is not None:
                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(left)))
                left = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(right)))
                right = d

            left = self.normalize_sample(left)
            right = self.normalize_sample(right)

            genuine = not genuine

            yield left, right, label


class Counter(object):
    """
    Class that holds a list of objects of certain identity and a counter
    """

    def __init__(self):
        self.objects = []
        self.offset = 0

    def get_object(self):

        o = self.objects[self.offset]
        self.offset += 1
        if self.offset == len(self.objects):
            self.offset = 0

        return o
