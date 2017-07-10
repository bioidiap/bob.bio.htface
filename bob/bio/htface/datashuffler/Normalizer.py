#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Mon 07 Nov 2016 09:39:36 CET

from bob.learn.tensorflow.datashuffler import MeanOffset


class MeanOffsetHT(MeanOffset):
    """
    Normalize a sample by a mean offset for two different image modalities
    
    """

    def __init__(self, offset_modality_a, offset_modality_b):
        self.offset_modality_a = offset_modality_a
        self.offset_modality_b = offset_modality_b
        self.is_ht = True

    def __call__(self, x, is_modality_a=True):
        for i in range(len(self.offset_modality_a)):

            if is_modality_a:
                x[:, :, i] = x[:, :, i] - self.offset_modality_a[i]
            else:
                x[:, :, i] = x[:, :, i] - self.offset_modality_b[i]

        return x
