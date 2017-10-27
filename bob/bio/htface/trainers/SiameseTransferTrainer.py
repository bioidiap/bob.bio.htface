#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from .Trainer import Trainer, SiameseTrainer
from ..analyzers import SoftmaxAnalizer
from .learning_rate import constant
import os
import logging
from bob.learn.tensorflow.utils.session import Session
import bob.core
logger = logging.getLogger("bob.learn")


class SiameseTransferTrainer(SiameseTrainer):
    """
    Trainer for siamese networks:

    Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a similarity metric discriminatively, with application to
    face verification." 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05). Vol. 1. IEEE, 2005.


    **Parameters**

    train_data_shuffler:
      The data shuffler used for batching data for training

    iterations:
      Maximum number of iterations

    snapshot:
      Will take a snapshot of the network at every `n` iterations

    validation_snapshot:
      Test with validation each `n` iterations

    analizer:
      Neural network analizer :py:mod:`bob.learn.tensorflow.analyzers`

    temp_dir: str
      The output directory

    verbosity_level:

    """

    def __init__(self,
                 train_data_shuffler,
                 validation_data_shuffler=None,

                 ###### training options ##########
                 iterations=5000,
                 snapshot=500,
                 validation_snapshot=100,
                 keep_checkpoint_every_n_hours=2,

                 ## Analizer
                 analizer=SoftmaxAnalizer(),

                 # Temporatu dir
                 temp_dir="siamese_cnn",

                 verbosity_level=2
                 ):

        self.train_data_shuffler = train_data_shuffler

        self.temp_dir = temp_dir

        self.iterations = iterations
        self.snapshot = snapshot
        self.validation_snapshot = validation_snapshot
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

        # Training variables used in the fit
        self.summaries_train = None
        self.train_summary_writter = None
        self.thread_pool = None

        # Validation data
        self.validation_summary_writter = None
        self.summaries_validation = None
        self.validation_data_shuffler = validation_data_shuffler

        # Analizer
        self.analizer = analizer
        self.global_step = None

        self.session = None

        self.graph = None
        self.validation_graph = None
                
        self.loss = None
        
        self.predictor = None
        self.validation_predictor = None        
        
        self.optimizer_class = None
        self.learning_rate = None

        # Training variables used in the fit
        self.optimizer = None
        
        self.data_ph = None
        self.label_ph = None
        
        self.validation_data_ph = None
        self.validation_label_ph = None
        
        self.saver = None

        bob.core.log.set_verbosity_level(logger, verbosity_level)

        # Creating the session
        self.session = Session.instance(new=True).session
        self.from_scratch = True

    def   
     
