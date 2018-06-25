#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf


def fdsu_contrastive_loss(left_embedding,
                          right_embedding,
                          contrastive_margin=2.0):
    """
    Compute the FDSU Siamese loss
 

    **Parameters**

    left_feature:
      First element of the pair

    right_feature:
      Second element of the pair

    margin:
      Contrastive margin

    """

    with tf.name_scope("fsdu_contranstive_loss"):

        losses = []
        for l,r in zip(left_embedding, right_embedding):

            _, height, width, number = map(lambda i: i.value, l.get_shape())
            size = height * width * number
        
            # reshaping per channel
            l = tf.reshape(l, (-1, number))
            r = tf.reshape(r, (-1, number))

            # gram 
            left_gram = tf.matmul(tf.transpose(l), l) / size
            right_gram = tf.matmul(tf.transpose(r), r) / size

            losses.append(tf.nn.l2_loss(left_gram - right_gram))

        loss = reduce(tf.add, losses)

        return loss

