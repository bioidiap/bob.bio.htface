.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>



Preamble
--------

From the experiments in :ref:`first insights section <first-insights>` it was possible to observe that a CNN model
trained with only visible images provided recognition rates far from being random, but still with very low
if compared with the state-of-the-art in our closed-set evaluation.

In this section we will explore strategies on how to use such prior and adapt our :math:`\phi` to some target image modality.
More preciselly, we'll use Siamese networks [Chopra2005]_. **VERY BAD TEXT**.

Siamese Neural Networks (**SNN**) learn the non-linear subspaces :math:`\phi` by repeatedly presenting
pairs of positive and negative examples (belonging to the same class or not).
To our particular task, the pairs of examples belong to clients sensed by different image modalities
(:math:`X_A` and :math:`X_B` ) as we can observe in the figure below.

.. image:: ../plots/transfer-learning/siamese.png
  :scale: 100 %

, where :math:`\mathcal{L}` is defined as :math:`\mathcal{L}(x_A, x_B) = || \phi_{\theta_1}(x_A) - \phi_{\theta_1}(x_B)||`.

.. note:: Some details about this chart and the following ones.

          - :math:`x_A` and :math:`x_B` corresponds to inputs from different image modalities :math:`A` and :math:`B`
          
          - :math:`\theta` corresponds to the latent variables. The superscript :math:`t` (:math:`\theta^t`) means that such
            :math:`\theta` is trainable (boxes filled with red). The superscript :math:`s` (:math:`\theta^s`) means that such 
            :math:`\theta` is **not** trainable (boxes filled in blue).


One commom wisdom about convolutional neural networks states (sorry for the lack of scientific rigor in this sentence, 
I have plenty of references to quote about this statement) that feature detectors from the first layers of the network are more
general.
Some researchers systematically noticed some tendencies in this level, such as, Gabor filters, color blobs, edge detector [Yosinski2014]_.
With the input signal going deeper into the architecture, the feature detectors tend to be more task (database) specific.

At this point we have some research questions here.

1. Are our prior :math:`\phi` (trained with visual light images) too modality specific? 
Can we train a modality specific embedding this prior?

This question is quite tempting. If our current feature detectors preserve some information
about the image modality, we could train a simple shallow network to model that.

2. If the first question is a negative, it suggests that information about the image modality carried in the
input signal is suppresed by the deep set of feature detectors. The question here is, can we retrain a 
subset of the first layers in order to preserve the modality information?

.. error:: Formulate better those questions


To approch those two questions we designed two set of experiments.


Adaptation of domain specific features
--------------------------------------

In this section we approach our first research question.
Such approach is summarized in the image below.

.. image:: ../plots/transfer-learning/siamese_new_embeddings.png
  :scale: 100 %


As can be observed, the input signals from two image modalities (:math:`x_A` and :math:`x_B`) are projected using
our prior :math:`\phi`.
The 128d embedding are feed into a 2-layer neural network (:math:`\phi'`) with 64 neurons with ReLU non activation (:math:`\theta_2`)
followed by another fully connected layer with 128 neurons with linear activation (:math:`\theta_3`).


.. literalinclude:: ../../bob/bio/htface/architectures/Transfer.py
   :language: python
   :pyobject: build_transfer_graph
   :caption: "Transfer.py"

Next subsections we describe recognition rate results in the same fashion as in :ref:`first insights section <first-insights>`.


POLA THERMAL
############

 +------------+--------------+--------+-------------------+
 | Image size | ML           | Feat.  | Rank-1            |
 +============+==============+========+===================+
 | 160 x 160  | Resnet-Gray  |        | 11.798% (1.556)   |
 +------------+--------------+--------+-------------------+
 | 160 x 160  | 128-64-128   |        | 6.964% (1.37)     |
 +------------+--------------+--------+-------------------+


CUHK-CUFS
#########

 +------------+--------------+--------+-------------------+
 | Image size | ML           | Feat.  | Rank-1            |
 +============+==============+========+===================+
 | 160 x 160  | Resnet-Gray  |        | 64.158% (3.424)   |
 +------------+--------------+--------+-------------------+
 | 160 x 160  | 128-64-128   |        | 22.178% (5.534)   |
 +------------+--------------+--------+-------------------+


CUHK-CUFSF
##########



CASIA NIR-VIS
#############


NIVL
####


Final Discussions
#################




Adaptation of the first layers
------------------------------

In this section we approach our second research question.

Just to wrap up, in this section we hyphotesize that we can adapt :math:`\phi` to a target modality by
specializing the first :math:`n` layers of :math:`\phi`.


.. image:: ../plots/transfer-learning/siamese.png
  :scale: 100 %





Adapting layer by layer
-----------------------

