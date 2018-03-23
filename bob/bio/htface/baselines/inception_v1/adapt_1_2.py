#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.bio.htface.baselines import Baseline
import pkg_resources


class SiameseAdaptLayers1_2_BatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the 1-2 layers
    """

    def __init__(self):
        super(SiameseAdaptLayers1_2_BatchNorm, self).__init__()

        self.baseline_type     = "Siamese BN"
        self.name              = "siamese_inceptionv1_adapt_1_2_nonshared_batch_norm"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_layers_1_2/extractor_nonshared_batch_norm.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
        self.reuse_extractor   = False        

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v1_adapt_layers_1_2/estimator_nonshared_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")


class SiameseAdaptLayers1_2_BetasBatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the first layer only
    """

    def __init__(self):
        super(SiameseAdaptLayers1_2_BetasBatchNorm, self).__init__()

        self.baseline_type     = "Siamese BN adapt betas"
        self.name              = "siamese_inceptionv1_adapt_1_2_betas_nonshared_batch_norm"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_layers_1_2/extractor_nonshared_betas_batch_norm.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py") # Same as v2
        self.reuse_extractor   = False

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v1_adapt_layers_1_2/estimator_nonshared_betas_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/") # Same as v2
        
        
class TripletAdaptLayers1_2_BatchNorm(Baseline):

    def __init__(self):
        super(TripletAdaptLayers1_2_BatchNorm, self).__init__()
    
        self.baseline_type     = "Triplet BN"
        self.name              = "triplet_inceptionv1_layers_1_2_nonshared_batch_norm"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_layers_1_2/triplet_extractor_nonshared_batch_norm.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
        self.reuse_extractor   = False        

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/triplet_transfer_learning/inceptionv1_layers_1_2/estimator_nonshared_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")
        

class TripletAdaptLayers1_2_BetasBatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the first layer only
      - ADAPT ONLY THE BETAS
    """

    def __init__(self):
        super(TripletAdaptLayers1_2_BetasBatchNorm, self).__init__()
            
        self.baseline_type     = "Triplet BN adapt betas"
        self.name              = "triplet_inceptionv1_layers_1_2_betas_nonshared_batch_norm"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_layers_1_2/triplet_extractor_nonshared_betas_batch_norm.py")
        
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
        self.reuse_extractor   = False

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/triplet_transfer_learning/inceptionv1_layers_1_2/estimator_nonshared_betas_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")        



