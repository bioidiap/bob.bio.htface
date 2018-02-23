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


class LightCNNPolathermal(Baseline):
    """
    Light CNN Baseline with special preprocessor for polathermal DB
    """

    def __init__(self):
        self.baseline_type     = "Standard FaceRec"
        self.name              = "lightcnn_polathermal"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/lightcnn_extractor.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/lightcnn_preprocessor_polathermal.py")

        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None


class Facenet(Baseline):
    """
    Facenet CNN Baseline
    """

    def __init__(self):
        self.baseline_type     = "Standard FaceRec"
        self.name              = "facenet"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/facenet_extractor.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/facenet_preprocessor.py")
        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None


class VGG16(Baseline):
    """
    VGG16 CNN Baseline
    """

    def __init__(self):
        self.baseline_type     = "Standard FaceRec"
        self.name              = "vgg16"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/vgg16_extractor.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/vgg16_preprocessor.py")
        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None


class Inceptionv1_gray(Baseline):
    """
    VGG16 CNN Baseline
    """

    def __init__(self):
        self.baseline_type     = "Standard FaceRec"
        self.name              = "idiap_casia_inception_v1_gray"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/inception_resnet_v1_gray_extractor.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/facenet_preprocessor.py")
        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None
