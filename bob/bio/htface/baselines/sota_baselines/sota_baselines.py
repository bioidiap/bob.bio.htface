#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.bio.htface.baselines import Baseline
import pkg_resources


class ISV(Baseline):
    """
    Baseline from:
    
    Freitas Pereira, Tiago, and Sébastien Marcel. "Heterogeneous Face Recognition using Inter-Session Variability Modelling." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2016.    
    
    """

    def __init__(self):
        super(ISV, self).__init__()
    
        self.baseline_type     = "SOTA baselines"
        self.name              = "isv-512g-u50"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/isv_extractor.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/isv_preprocessor.py")
        self.algorithm         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/isv_algorithm.py")

        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None


class MLBPHS(Baseline):
    """
    Baseline from:
    
    MLBP
    
    """



    def __init__(self):
        super(MLBPHS, self).__init__()

        self.baseline_type     = "SOTA baselines"
        self.name              = "mlbphs-r1357"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/brendan_extractor.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/brendan_preprocessor.py")
        self.algorithm         = "distance-euclidean"

        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None

