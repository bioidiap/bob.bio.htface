#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from tensorflow.python.estimator import estimator
from bob.learn.tensorflow.utils import predict_using_tensors
from bob.learn.tensorflow.estimators import check_features, get_trainable_variables

import logging

logger = logging.getLogger("bob.learn")


class TripletAdaptation(estimator.Estimator):
    """
    NN estimator for Triplet networks.
    This particular structure admits that parts of the Siamese are non shared

    The **architecture** function should follow the following pattern:

      def my_beautiful_function(placeholder):

          end_points = dict()
          graph = convXX(placeholder)
          end_points['conv'] = graph
          ....
          return graph, end_points

    The **loss** function should follow the following pattern:

    def my_beautiful_loss(logits, labels):
       return loss_set_of_ops(logits, labels)


        extra_checkpoint = {"checkpoint_path":model_dir,
                            "scopes": dict({"Dummy/": "Dummy/"}),
                            "trainable_variables": [<LIST OF VARIABLES OR SCOPES THAT YOU WANT TO TRAIN>]
                           }




    **Parameters**
      architecture:
         Pointer to a function that builds the graph.

      optimizer:
         One of the tensorflow solvers (https://www.tensorflow.org/api_guides/python/train)
         - tf.train.GradientDescentOptimizer
         - tf.train.AdagradOptimizer
         - ....

      config:

      loss_op:
         Pointer to a function that computes the loss.

      embedding_validation:
         Run the validation using embeddings?? [default: False]

      model_dir:
        Model path

      validation_batch_size:
        Size of the batch for validation. This value is used when the
        validation with embeddings is used. This is a hack.


      params:
        Extra params for the model function
        (please see https://www.tensorflow.org/extend/estimators for more info)
        
      extra_checkpoint: dict
        In case you want to use other model to initialize some variables.
        This argument should be in the following format
        extra_checkpoint = {
            "checkpoint_path": <YOUR_CHECKPOINT>,
            "scopes": dict({"<SOURCE_SCOPE>/": "<TARGET_SCOPE>/"}),
            "trainable_variables": [<LIST OF VARIABLES OR SCOPES THAT YOU WANT TO TRAIN>],
            "non_reusable_variables": [<LIST OF VARIABLES OR SCOPES THAT YOU WANT TO TRAIN>]
        }
        
    """

    def __init__(self,
                 architecture=None,
                 optimizer=None,
                 config=None,
                 loss_op=None,
                 model_dir="",
                 validation_batch_size=None,
                 params=None,
                 extra_checkpoint=None):

        self.architecture = architecture
        self.optimizer = optimizer
        self.loss_op = loss_op
        self.loss = None
        self.extra_checkpoint = extra_checkpoint

        if self.architecture is None:
            raise ValueError(
                "Please specify a function to build the architecture !!")

        if self.optimizer is None:
            raise ValueError(
                "Please specify a optimizer (https://www.tensorflow.org/api_guides/python/train) !!"
            )

        if self.loss_op is None:
            raise ValueError("Please specify a function to build the loss !!")

        def _model_fn(features, labels, mode, params, config):

            # The input function needs to have dictionary of triplets whose keys are the `anchor`, `positive` and `negative`
            if 'anchor' not in features.keys() or \
                            'positive' not in features.keys() or \
                            'negative' not in features.keys():
                raise ValueError(
                    "The input function needs to contain a dictionary with the "
                    "keys `anchor`, `positive` and `negative` ")
            

            # For this part, nothing is trainable
            prelogits_anchor,_ = self.architecture(
                features['anchor'],
                mode=mode,
                reuse=False,
                trainable_variables=[],
                is_siamese=False,
                is_left = True)

            prelogits_positive,_= self.architecture(
                features['positive'],
                reuse=True,
                mode=mode,
                trainable_variables=[],
                is_siamese=False,
                is_left = False
                )
                
            prelogits_negative,_= self.architecture(
                features['negative'],
                reuse=True,
                mode=mode,
                trainable_variables=[],
                is_siamese=False,
                is_left = False
                )                

            for v in tf.all_variables():
                if "anchor" in v.name or "positive-negative" in v.name:
                    tf.summary.histogram(v.name, v)
                
            if mode == tf.estimator.ModeKeys.TRAIN:

                if self.extra_checkpoint is not None:                                
                
                    left_scope = self.extra_checkpoint["scopes"][0]
                    right_scope = self.extra_checkpoint["scopes"][1]
                
                    tf.contrib.framework.init_from_checkpoint(
                        self.extra_checkpoint["checkpoint_path"],
                        left_scope
                        )
                        
                    tf.contrib.framework.init_from_checkpoint(
                        self.extra_checkpoint["checkpoint_path"],
                        right_scope
                        )
                        

                # Compute Loss (for both TRAIN and EVAL modes)
                self.loss = self.loss_op(prelogits_anchor, prelogits_positive,
                                         prelogits_negative)

                # Configure the Training Op (for TRAIN mode)
                global_step = tf.train.get_or_create_global_step()
                
                train_op = self.optimizer.minimize(
                    self.loss, global_step=global_step)

                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, train_op=train_op)

            # Compute the embeddings
            embeddings_anchor = tf.nn.l2_normalize(prelogits_anchor, 1)
            embeddings_positive = tf.nn.l2_normalize(prelogits_positive, 1)
            embeddings_negative = tf.nn.l2_normalize(prelogits_negative, 1)
            
            predictions = {"embeddings_anchor": embeddings_anchor,
                           "embeddings_positive": embeddings_positive,
                           "embeddings_negative": embeddings_negative}

            # Prediction mode return the embeddings
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predictions)

            # IF validation raises an ex
            raise NotImplemented("Validation not implemented")


        super(TripletAdaptation, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            params=params,
            config=config)
