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
Can we train a modality specific embedding on top of this prior?

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

  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the thermograms database.

 +----------------+------------------+---------+-------------------+
 | Image size     | ML               | Feat.   | Rank-1            |
 +================+==================+=========+===================+
 | 160 x 160      | Resnet-Gray      |         | 11.798% (1.556)   |
 +----------------+------------------+---------+-------------------+
 | **160 x 160**  | **128-64-128**   |         | **6.964% (1.37)** |
 +----------------+-------------------+--------+-------------------+

It's possible to observe a decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings of our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases pola_thermal
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases pola_thermal



CUHK-CUFS
#########

  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the viewable sketch database .

 +----------------+------------------+--------+-----------------------+
 | Image size     | ML               | Feat.  | Rank-1                |
 +================+==================+========+=======================+
 | 160 x 160      | Resnet-Gray      |        | 64.158% (3.424)       |
 +----------------+------------------+--------+-----------------------+
 | **160 x 160**  | **128-64-128**   |        | **22.178% (5.534)**   |
 +----------------+------------------+--------+-----------------------+

It's possible to observe a severe decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings or our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufs
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufs



CUHK-CUFSF
##########

  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the viewable sketch database .

 +----------------+------------------+--------+------------------+
 | Image size     | ML               | Feat.  | Rank-1           |
 +================+==================+========+==================+
 | 160 x 160      | Resnet-Gray      |        | 16.518%(1.394)   |
 +----------------+------------------+--------+------------------+
 | **160 x 160**  | **128-64-128**   |        | **7.085(0.64)**  |
 +----------------+------------------+--------+------------------+

It's possible to observe a severe decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings or our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufsf
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufsf




CASIA NIR-VIS
#############

  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the NIR baseline.

 +----------------+------------------+--------+------------------+
 | Image size     | ML               | Feat.  | Rank-1           |
 +================+==================+========+==================+
 | 160 x 160      | Resnet-Gray      |        | 44.031%(0.999)   |
 +----------------+------------------+--------+------------------+
 | **160 x 160**  | **128-64-128**   |        | **30.716(0.8)**  |
 +----------------+------------------+--------+------------------+

It's possible to observe a severe decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings or our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases casia_nir_vis
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases casia_nir_vis



NIVL
####


  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the NIR baseline.

 +----------------+------------------+--------+-----------------------+
 | Image size     | ML               | Feat.  | Rank-1                |
 +================+==================+========+=======================+
 | 160 x 160      | Resnet-Gray      |        | 60.009% (2.518)       |
 +----------------+------------------+--------+-----------------------+
 | **160 x 160**  | **128-64-128**   |        | **31.772 (1.061)**    |
 +----------------+------------------+--------+-----------------------+

It's possible to observe a severe decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings or our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufsf
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufsf



Final Discussions
#################

In this sections we tried to answer the question if it is possible to model modality specific embeddings on top of 
our prior :math:`\phi`.
The results, in terms of recognition rate, were very conclusive showing that :math:`\phi` degrades the modality input signal
in such a way that is not possible to do a modality map on top of the embedding **VERY BAD TEXT**.

.. Todo:: Can we visually inspect that??

.. Todo:: Shall I try different classifiers on top of it?



Adaptation of the first layers
------------------------------

In this section we approach our second research question.

Just to wrap up, in this section we hyphotesize that we can adapt :math:`\phi` to a target modality by
specializing the first :math:`n` layers of :math:`\phi`.


.. image:: ../plots/transfer-learning/siamese.png
  :scale: 100 %



Adapting layer by layer
-----------------------


POLA THERMAL
############


 +----------------+------------------+-------------------+
 | Image size     | ML               | Rank-1            |
 +================+==================+===================+
 | 160 x 160      | Resnet-Gray      | 11.798% (1.556)   |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt first**  | **12.738% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-2**    | **17.143% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-4**    | **29.286% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-5**    | **33.095% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-6**    | **32.583(3.409)** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt All**    | **-    % (    )** |
 +----------------+------------------+-------------------+



CUHK-CUFS
#########

 +----------------+------------------+-------------------+
 | Image size     | ML               | Rank-1            |
 +================+==================+===================+
 | 160 x 160      | Resnet-Gray      | 64.158% (3.424)   |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt first**  | **63.366% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-2**    | **64.851% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-4**    | **78.713% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-5**    | **83.168% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt All**    | **23.762% (    )**|
 +----------------+------------------+-------------------+



CUHK-CUFSF
##########


 +----------------+------------------+-------------------+
 | Image size     | ML               | Rank-1            |
 +================+==================+===================+
 | 160 x 160      | Resnet-Gray      | 16.518%(1.394)    |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt first**  | **15.182% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-2**    | **18.623% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-4**    | **36.235% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-5**    | **41.093% (    )**|
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt All**    | **-    % (    )** |
 +----------------+------------------+-------------------+
 

CASIA NIR-VIS
#############

 +----------------+------------------+-------------------+
 | Image size     | ML               | Rank-1            |
 +================+==================+===================+
 | 160 x 160      | Resnet-Gray      | 44.031%(1.394)    |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt first**  | **40.206% (   )**|
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-2**    | **54.832% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-4**    | **77.835% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-5**    | **61.405% (    )**|
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt All**    | **-    % (    )** |
 +----------------+------------------+-------------------+




NIVL
####

 86.386 1-4

 +----------------+------------------+-------------------+
 | Image size     | ML               | Rank-1            |
 +================+==================+===================+
 | 160 x 160      | Resnet-Gray      | 60.009%(2.518)    |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt first**  | **69.796% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-2**    | **82.74% (    )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-4**    | **85.123% (   )** |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt 1-5**    | **76.713% (    )**|
 +----------------+------------------+-------------------+
 | **160 x 160**  | **Adapt All**    | **-    % (    )** |
 +----------------+------------------+-------------------+
 





