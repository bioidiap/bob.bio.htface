#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.bio.base.baseline import Baseline
import pkg_resources


class ISV(Baseline):
    """
    Baseline from:
    
    Freitas Pereira, Tiago, and Sébastien Marcel. "Heterogeneous Face Recognition using Inter-Session Variability Modelling." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2016.    
    
    """

    def __init__(self, **kwargs):
    
        name              = "isv-512g-u50"
        extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/isv_extractor.py")
        preprocessors     = {"default": pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/isv_preprocessor.py")}
        algorithm         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/isv_algorithm.py")

        self.baseline_type     = "SOTA baselines"
        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None
        
        super(ISV, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)
        

htface_isv = ISV()

class MLBPHS(Baseline):
    """
    Baseline from:
    
    MLBP
    
    """

    def __init__(self, **kwargs):

        name              = "mlbphs-r1357"
        extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/brendan_extractor.py")
        preprocessors     = {"default": pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/brendan_preprocessor.py")}
        algorithm         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/brendan_algorithm.py")

        self.baseline_type     = "SOTA baselines"
        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None
        
        super(MLBPHS, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)

htface_mlbphs = MLBPHS()

class GFKGabor(Baseline):
    """
    Baseline from:
    
    MLBP
    
    """

    def __init__(self, **kwargs):

        name              = "gfk_gabor"
        extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/gfk_extractor.py")
        preprocessors     = {"default": pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/gfk_preprocessor.py")}
        algorithm         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/sota_baselines/gfk_algorithm.py")

        self.baseline_type     = "SOTA baselines"
        self.reuse_extractor   = True

        # train cnn
        self.estimator         = None
        self.preprocessed_data = None

        super(GFKGabor, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)
        
htface_gfkgabor = GFKGabor()
