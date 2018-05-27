#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.bio.htface.baselines import Baseline
import pkg_resources


class SiameseAdaptLayers1_6_BatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the 1_6 layers
    """

    def __init__(self, **kwargs):

        name              = "siamese_inceptionv1_adapt_1_6_nonshared_batch_norm"
        extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_layers_1_6/extractor_nonshared_batch_norm.py")
        preprocessors     = {"default": pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")}
        algorithm       = "distance-cosine"

        self.baseline_type     = "Siamese BN"
        self.reuse_extractor   = False        

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v1_adapt_layers_1_6/estimator_nonshared_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")

        super(SiameseAdaptLayers1_6_BatchNorm, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)


class SiameseAdaptLayers1_6_BetasBatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the 1_6 layers
    """

    def __init__(self, **kwargs):

        name              = "siamese_inceptionv1_adapt_1_6_betas_nonshared_batch_norm"
        extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_layers_1_6/extractor_nonshared_betas_batch_norm.py")
        preprocessors     = {"default": pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")} # Same as v2
        algorithm       = "distance-cosine"

        self.baseline_type     = "Siamese BN adapt betas"
        self.reuse_extractor   = False

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v1_adapt_layers_1_6/estimator_nonshared_betas_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/") # Same as v2
        
        super(SiameseAdaptLayers1_6_BetasBatchNorm, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)
        
        
class TripletAdaptLayers1_6_BatchNorm(Baseline):
    """
    - Adapt the 1_6 layers
    """

    def __init__(self, **kwargs):
    
        name              = "triplet_inceptionv1_layers_1_6_nonshared_batch_norm"
        extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_layers_1_6/triplet_extractor_nonshared_batch_norm.py")
        preprocessors     = {"default" :pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")}
        algorithm       = "distance-cosine"

        self.baseline_type     = "Triplet BN"
        self.reuse_extractor   = False        

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/triplet_transfer_learning/inceptionv1_layers_1_6/estimator_nonshared_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")
        
        super(TripletAdaptLayers1_6_BatchNorm, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)        
        

class TripletAdaptLayers1_6_BetasBatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the 1_6 layers
      - ADAPT ONLY THE BETAS
    """

    def __init__(self, **kwargs):

        name              = "triplet_inceptionv1_layers_1_6_betas_nonshared_batch_norm"
        extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_layers_1_6/triplet_extractor_nonshared_betas_batch_norm.py")
        
        preprocessors     = {"default": pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")}
        algorithm         = "distance-cosine"
            
        self.baseline_type     = "Triplet BN adapt betas"
        self.reuse_extractor   = False

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/triplet_transfer_learning/inceptionv1_layers_1_6/estimator_nonshared_betas_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")        

        super(TripletAdaptLayers1_6_BetasBatchNorm, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)
        
# Entry points
inception_resnet_v1_siamese_adapt_1_6 = SiameseAdaptLayers1_6_BatchNorm()
inception_resnet_v1_siamese_adapt_1_6_betas = SiameseAdaptLayers1_6_BetasBatchNorm()

inception_resnet_v1_triplet_adapt_1_6 = TripletAdaptLayers1_6_BatchNorm()
inception_resnet_v1_triplet_adapt_1_6_betas = TripletAdaptLayers1_6_BetasBatchNorm()        
