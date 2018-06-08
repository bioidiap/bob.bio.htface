#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf


def kl_loss(left_embedding,
            right_embedding,
            contrastive_margin=2.0):
    """
    Compute the KL
 

    **Parameters**

    left_feature:
      First element of the pair

    right_feature:
      Second element of the pair

    margin:
      Contrastive margin

    """

    with tf.name_scope("kl_loss"):

        left_embedding = tf.maximum(tf.reduce_mean(tf.nn.l2_normalize(left_embedding, 1), axis=0), 1e-5)
        right_embedding = tf.maximum(tf.reduce_mean(tf.nn.l2_normalize(right_embedding, 1), axis=0), 1e-5)

        kl = (left_embedding - right_embedding) * tf.log(left_embedding/right_embedding)

        loss = tf.reduce_sum(kl, name="kl_loss")
        tf.summary.scalar('kl_loss', loss)

        return loss

