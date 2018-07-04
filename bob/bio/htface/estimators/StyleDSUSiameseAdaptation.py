#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from tensorflow.python.estimator import estimator
from bob.learn.tensorflow.utils import predict_using_tensors
from bob.learn.tensorflow.estimators import check_features, get_trainable_variables
from bob.bio.htface.loss import style_dsu_contrastive_loss, style_gram_dsu_contrastive_loss
import tensorflow.contrib.slim as slim
import logging

logger = logging.getLogger("bob.learn")


class StyleDSUSiameseAdaptation(estimator.Estimator):
    """
    NN estimator for Siamese networks.
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

      force_weights_shutdown: bool
        If True will shutdown the weights no matter the weights are set in trainable_variables.
        Default **False**        

        
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
                 model_dir="",
                 validation_batch_size=None,
                 params=None,
                 learning_rate_values=[0.1],
                 learning_rate_boundaries=[],
                 force_weights_shutdown=False,
                 extra_checkpoint=None,
                 loss_left_end_points=None,
                 loss_right_end_points=None
                 ):

        self.architecture = architecture
        self.optimizer = optimizer
        self.loss = None
        self.extra_checkpoint = extra_checkpoint
        self.learning_rate_boundaries = learning_rate_boundaries
        self.learning_rate_values = learning_rate_values
        self.force_weights_shutdown = force_weights_shutdown

        if self.architecture is None:
            raise ValueError(
                "Please specify a function to build the architecture !!")

        if self.optimizer is None:
            raise ValueError(
                "Please specify a optimizer (https://www.tensorflow.org/api_guides/python/train) !!"
            )

        def _model_fn(features, labels, mode, params, config):

            # Building one graph, by default everything is trainable
            # The input function needs to have dictionary pair with the `left` and `right` keys
            if 'left' not in features.keys(
            ) or 'right' not in features.keys():
                raise ValueError(
                    "The input function needs to contain a dictionary with the keys `left` and `right` "
                )

            # For this part, nothing is trainable
            prelogits_xs, end_points_xs = self.architecture(
                features['left'],
                mode=mode,
                reuse=False,
                trainable_variables=[],
                is_left = True,
                is_siamese=False,
                force_weights_shutdown=self.force_weights_shutdown)


            prelogits_xt0, end_points_xt0 = self.architecture(
                features['right'],
                mode=mode,
                reuse=True,
                trainable_variables=[],
                is_left = True,
                is_siamese=False,
                force_weights_shutdown=self.force_weights_shutdown)

            prelogits_xt, end_points_xt = self.architecture(
                features['right'],
                reuse=True,
                mode=mode,
                trainable_variables=[],
                is_left = False,
                is_siamese=False,
                force_weights_shutdown=self.force_weights_shutdown
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

                global_step = tf.train.get_or_create_global_step()

                # Compute the moving average of all individual losses and the total loss.
                variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
                variable_averages_op = variable_averages.apply(tf.trainable_variables())

                with tf.control_dependencies([variable_averages_op]):

                    #tf.summary.image("input-source", features["left"], 1)
                    #tf.summary.image("input-target", features["right"], 1)

                    #graph = end_points_xs[loss_left_end_points[0]]
                    #tf.summary.image("1-4-source", tf.transpose(graph[0:1, :, :, :], perm=[3,1,2,0]), max_outputs=graph.get_shape().as_list()[3])

                    #graph = end_points_xt[loss_right_end_points[0]]
                    #tf.summary.image("1-4-target", tf.transpose(graph[0:1, :, :, :], perm=[3,1,2,0]), max_outputs=graph.get_shape().as_list()[3])
                    source_layers = []
                    target_layers = []
                    for layer_source, layer_target in zip(loss_left_end_points, loss_right_end_points):
                        source_layers.append(end_points_xs[layer_source])
                        target_layers.append(end_points_xt[layer_target])
                       
                    self.loss = style_gram_dsu_contrastive_loss(prelogits_xt,
                                                           prelogits_xt0,

                                                           source_layers,
                                                           target_layers
                                                           )


                    # Compute the moving average of all individual losses and the total loss.
                    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
                    loss_averages_op = loss_averages.apply(tf.get_collection(tf.GraphKeys.LOSSES))
                    
                    # Defining the learning rate
                    learning_rate = tf.train.piecewise_constant(global_step,
                                                                self.learning_rate_boundaries,
                                                                self.learning_rate_values,
                                                                name="learning_rate")
                    # Bootstraping the solver
                    optimizer = self.optimizer(learning_rate)
                    tf.summary.scalar('learning_rate', learning_rate)
                    train_op = tf.group(optimizer.minimize(self.loss, global_step=global_step),
                                        variable_averages_op,
                                        loss_averages_op)


                    return tf.estimator.EstimatorSpec(
                        mode=mode, loss=self.loss, train_op=train_op)

            # Compute the embeddings
            embeddings_left = tf.nn.l2_normalize(prelogits_left, 1)
            embeddings_right = tf.nn.l2_normalize(prelogits_right, 1)
            
            predictions = {"embeddings_left": embeddings_left,
                           "embeddings_right": embeddings_right}

            # Prediction mode return the embeddings
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predictions)

            # IF validation raises an ex
            raise NotImplemented("Validation not implemented")


        super(StyleDSUSiameseAdaptation, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            params=params,
            config=config)

