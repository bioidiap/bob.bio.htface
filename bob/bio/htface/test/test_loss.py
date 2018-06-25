#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
import numpy
from bob.bio.htface.loss import fdsu_contrastive_loss


def test_fsdu_contrastive_loss():
    numpy.random.seed(10)
    left_embedding  = numpy.random.normal(loc=0, size=(2,10,10,2)).astype("float32")
    right_embedding = numpy.random.normal(loc=2,size=(2,10,10,2)).astype("float32")

    sess = tf.Session()
    #xuxa = sess.run(tf.maximum(tf.nn.relu(tf.convert_to_tensor(left_embedding)),1e-5))

    left_embedding = tf.convert_to_tensor(left_embedding)
    right_embedding = tf.convert_to_tensor(right_embedding)
    loss = sess.run(fdsu_contrastive_loss([left_embedding, left_embedding], [right_embedding, right_embedding]))

    assert loss >0 
