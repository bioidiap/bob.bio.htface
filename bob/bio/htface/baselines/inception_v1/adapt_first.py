#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.bio.base.baseline import Baseline
import pkg_resources


class SiameseAdaptFirstBatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the first layer only
    """

    def __init__(self):
        super(SiameseAdaptFirstBatchNorm, self).__init__()

        self.baseline_type     = "Siamese BN"
        self.name              = "siamese_inceptionv1_first_layer_nonshared_batch_norm"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_first_layer/extractor_nonshared_batch_norm.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py") # Same as v2
        self.reuse_extractor   = False        

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v1_adapt_first_layer/estimator_nonshared_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/") # Same as v2


class SiameseAdaptFirstBetasBatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the first layer only
    """

    def __init__(self):
        super(SiameseAdaptFirstBetasBatchNorm, self).__init__()

        self.baseline_type     = "Siamese BN adapt betas"
        self.name              = "siamese_inceptionv1_first_layer_betas_nonshared_batch_norm"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_first_layer/extractor_nonshared_betas_batch_norm.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py") # Same as v2
        self.reuse_extractor   = False

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v1_adapt_first_layer/estimator_nonshared_betas_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/") # Same as v2
        
        
class TripletAdaptFirstBatchNorm(Baseline):

    def __init__(self):
        super(TripletAdaptFirstBatchNorm, self).__init__()
    
        self.baseline_type     = "Triplet BN"
        self.name              = "triplet_inceptionv1_first_layer_nonshared_batch_norm"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_first_layer/triplet_extractor_nonshared_batch_norm.py")
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
        self.reuse_extractor   = False        

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/triplet_transfer_learning/inceptionv1_first_layer/estimator_nonshared_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")


class TripletAdaptFirstBetasBatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the first layer only
      - ADAPT ONLY THE BETAS
    """

    def __init__(self):
        super(TripletAdaptFirstBetasBatchNorm, self).__init__()
    
        self.baseline_type     = "Triplet BN adapt betas"
        self.name              = "triplet_inceptionv1_first_layer_betas_nonshared_batch_norm"
        self.extractor         = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v1_adapt_first_layer/triplet_extractor_nonshared_betas_batch_norm.py")
                
        self.preprocessor      = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
        self.reuse_extractor   = False

        # train cnn
        self.estimator         = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/triplet_transfer_learning/inceptionv1_first_layer/estimator_nonshared_betas_batch_norm.py")
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")
        
