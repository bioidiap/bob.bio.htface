#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.bio.htface.baselines import Baseline
import pkg_resources


class LightCNN(Baseline):
    """
    Light CNN Baseline
    """

    def __init__(self):
        self.baseline_type     = "Standard FaceRec"
        self.name              = "lightcnn"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/lightcnn_extractor.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/lightcnn_preprocessor.py")
        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None

