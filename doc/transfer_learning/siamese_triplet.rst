.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


Preamble
--------

From the experiments in :ref:`first insights section <first-insights>` it was possible to observe that a CNN model
trained with only visible images provided recognition rates far from being random, but still with very low
if compared with the state-of-the-art in our closed-set evaluation.

In this section we will explore strategies on how to use such prior and adapt our :math:`\phi` to some target image modality.

.. note:: Some details about the charts that will follow.

          - :math:`x_A` and :math:`x_B` corresponds to inputs from different image modalities :math:`A` and :math:`B`
          
          - :math:`\theta` corresponds to the latent variables. The superscript :math:`t` (:math:`\theta^t`) means that such
            :math:`\theta` is trainable (boxes filled with red). The superscript :math:`s` (:math:`\theta^s`) means that such 
            :math:`\theta` is **not** trainable (boxes filled in blue).

One commom wisdom about convolutional neural networks states that feature detectors from the first layers of the network are more
general (edge detectors, color blobs, etc..), and upper layers handle high level entities (such as eyes, nose, etc..).
Some researchers systematically noticed some tendencies in this level, such as, Gabor filters, color blobs, edge detector [Yosinski2014]_.

Our task handles essentially with faces sensed in different image modalities.
Assuming that faces and macro elements of the face, such as, eyes, nose and mouth, are modeled in higher layers of a deep neural network, it's
reasonable to hypothesize that a joint modeling between different image modalities would take place in the first layers rather than in the last layers.
To develop such hypothesis, we designed two research questions:

1. Are our prior :math:`\phi` (trained with visual light images) too modality specific? 
Can we train a modality specific embedding on top of this prior?

If our current feature detectors preserve some information about the image modality, 
it invalidates our base hypothesis and we could approach the taks by just training a shallow
network on top of this prior (or other classifier).


2. If the first question is a negative, it suggests that information about the image modality carried in the
input signal is suppressed by the deep set of feature detectors. The question here is, can we retrain a
subset of the first layers in order to preserve the modality information?

To approach those two questions we designed two major groups of experiments.
The first one will joint training two image modalities using Siamese Networks and the second one will use Triplet networks.


Siamese Networks
----------------

Siamese Neural Networks (**SNN**) [Chopra2005]_ learn the non-linear subspaces :math:`\phi` by repeatedly presenting
pairs of positive and negative examples (belonging to the same class or not).
To our particular task, the pairs of examples belong to clients sensed by different image modalities
(:math:`x_A` and :math:`x_B` ) as we can observe in the figure below.


.. image:: ../plots/transfer-learning/siamese.png
  :scale: 100 %

, where :math:`\mathcal{L}` is defined as :math:`\mathcal{L}(x_A, x_B) = || \phi_{\theta_1}(x_A) - \phi_{\theta_1}(x_B)||`.

The next two subsections we approach our two research questions using this network arrangement.


Adaptation of our prior embeddings using Siamese networks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

 +----------------+------------------+-------------------+
 | Image size     | CNN              | Rank-1            |
 +================+==================+===================+
 | 160 x 160      | Inception V2     | 11.798% (1.556)   |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **128-64-128**   | **6.964% (1.37)** |
 +----------------+------------------+-------------------+

It's possible to observe a decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings of our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases pola_thermal
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases pola_thermal



CUHK-CUFS
#########

  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the viewable sketch database .

 +----------------+------------------+-----------------------+
 | Image size     | CNN              | Rank-1                |
 +================+==================+=======================+
 | 160 x 160      | Inception V2     | 64.158% (3.424)       |
 +----------------+------------------+-----------------------+
 | **160 x 160**  | **128-64-128**   | **22.178% (5.534)**   |
 +----------------+------------------+-----------------------+

It's possible to observe a severe decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings or our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufs
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufs



CUHK-CUFSF
##########

  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the viewable sketch database .

 +----------------+------------------+------------------+
 | Image size     | CNN              | Rank-1           |
 +================+==================+==================+
 | 160 x 160      | Inception V2     | 16.518%(1.394)   |
 +----------------+------------------+------------------+
 | **160 x 160**  | **128-64-128**   | **7.085(0.64)**  |
 +----------------+------------------+------------------+

It's possible to observe a severe decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings or our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufsf
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases cuhk_cufsf




CASIA NIR-VIS
#############

  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the NIR baseline.

 +----------------+------------------+------------------+
 | Image size     | CNN              | Rank-1           |
 +================+==================+==================+
 | 160 x 160      | Inception V2     | 44.031%(0.999)   |
 +----------------+------------------+------------------+
 | **160 x 160**  | **128-64-128**   | **30.716(0.8)**  |
 +----------------+------------------+------------------+

It's possible to observe a severe decrease in the recognition rate using this assumption suggesting that it's
not possible to learn a modality map using the embeddings or our prior :math:`\phi`.
 
The following steps train and evaluate such CNN::

 $ bob_htface_train_cnn.py  --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases casia_nir_vis
 $ bob_htface_cnn_baselines.py --baselines idiap_casia_inception_v2_gray_transfer_64_128 --databases casia_nir_vis



NIVL
####


  Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
  the NIR baseline.

 +----------------+------------------+-------------------+
 | Image size     | CNN              | Rank-1            |
 +================+==================+===================+
 | 160 x 160      | Inception V2     | 60.009% (2.518)   |
 +----------------+------------------+-------------------+
 | **160 x 160**  | **128-64-128**   | **31.772 (1.061)**|
 +----------------+------------------+-------------------+

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
in such a way that is not possible to do a modality map on top of those embeddings.

.. Todo:: Can we visually inspect that??

.. Todo:: Shall I try different classifiers on top of it?



Adaptation of the first :math:`n` layers of our prior
+++++++++++++++++++++++++++++++++++++++++++++++++++++

In this section we approach our second research question.

Just to wrap up, in this section we hyphotesize that we can joint model two image modalities by backpropagating the :math:`\mathcal{L}` w.r.t :math:`\theta` only for
the :math:`n` first layers of :math:`\phi`.

Such adaptation will take place in 6 different points of the CNN (the rest of the network is kept intact):

 - Adapt first layer of :math:`\phi`: error backpropagated only in **Conv2d_1a_3x3**
 - Adapt layers 1-2 of :math:`\phi`: error backpropagated in **Conv2d_1a_3x3**, **Conv2d_2a_3x3** and **"Conv2d_2b_3x3"**
 - Adapt layers 1-4 of :math:`\phi`: error backpropagated in **Conv2d_1a_3x3**, **Conv2d_2a_3x3**, **"Conv2d_2b_3x3"**,  **Conv2d_3b_1x1** and **Conv2d_4a_3x3**
 - Adapt layers 1-5 of :math:`\phi`: error backpropagated in **Conv2d_1a_3x3**, **Conv2d_2a_3x3**, **"Conv2d_2b_3x3"**, **Conv2d_3b_1x1**, **Conv2d_4a_3x3** and **Mixed_5b** 
 - Adapt layers 1-6 of :math:`\phi`: error backpropagated in **Conv2d_1a_3x3**, **Conv2d_2a_3x3**, **"Conv2d_2b_3x3"**, **Conv2d_3b_1x1**, **Conv2d_4a_3x3** and **Block35**
 - Adapt the whole :math:`\phi`



POLA THERMAL
############

Follow bellow the results in terms of Rank-1 recognition rate and the CMC plots under this assumption using 
the thermograms database.

 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | Image size     | CNN              | Rank-1 - siamese joint    | Rank 1 - siamese single   | Rank 1 - Triplet          |
 +================+==================+===========================+===========================+===========================+
 | 160 x 160      | Inception V2     | 25.774(2.193)             | 25.774(2.193)             | 25.774(2.193)             |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt first**  | **13.952 (2.104)**        | 31.048(2.124)             | 25.94(1.711)              |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-2**    | **22.964 (4.181)**        |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-4**    | **27.917 (2.825)**        |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-5**    | **29.381 (4.002)**        | 37.536(1.341)             | 24.595(5.016)             |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-6**    | **32.583 (3.409)**        |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt All**    | **8.917 (1.409)**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 
.. image:: ../plots/transfer-learning/pola_thermal/siamese_cmc.png
 :scale: 100 %

 

It's possible to observe that the recognition rate increases according with the amount of layers we adapt which corroborates with our assumptions.
However, such error rate drops drastically when the whole network is adapted.
This is understandable, since the amount of parameters to be learnt are several times bigger than the amount of data available to train this task.

The sequence of commands below run the amount of experiments necessary to run this plot::

  $ # Adapting first layer
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases pola_thermal
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases pola_thermal
  
  $ # Adapting layers 1-2  
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases pola_thermal  
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases pola_thermal  
     
  $ # Adapting layers 1-4
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases pola_thermal  
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases pola_thermal  

  $ # Adapting layers 1-5
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases pola_thermal  
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases pola_thermal  

  $ # Adapting layers 1-6
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases pola_thermal  
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases pola_thermal  

  $ # Adapting all layers
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases pola_thermal
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases pola_thermal  



CUHK-CUFS
#########

Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
the viewed sketch database.

 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | Image size     | CNN              | Rank-1 - siamese joint    | Rank 1 - siamese single   | Rank 1 - Triplet          |
 +================+==================+===========================+===========================+===========================+
 | 160 x 160      | Inception V2     | 67.03(2.314)              | 67.03(2.314)              | 67.03(2.314)              |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt first**  | **69.208 (3.941)**        |  74.653(3.52)             | 74.356(1.94)              |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-2**    | **74.752 (5.575)**        |  84.109% (3.051)          |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-4**    | **79.604 (3.275)**        |  89.987% (3.123)          |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-5**    | **81.98 (2.271)**         |  92.376(0.243)            |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-6**    | **82.921 (1.422)**        |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt All**    | **28.218 (3.86)**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+


.. image:: ../plots/transfer-learning/cuhk_cufs/siamese_cmc.png
 :scale: 100 %


The same trends as before can be observed.
The recognition rate increases according with the amount of layers we adapt.
Adapting the whole networks leads to a heavy overfitting, making the error rates drop to ~20%.

The sequence of commands below run the amount of experiments necessary to run this plot::

  $ # Adapting first layer
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases cuhk_cufs
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases cuhk_cufs
  
  $ # Adapting layers 1-2  
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases cuhk_cufs
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases cuhk_cufs
     
  $ # Adapting layers 1-4
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases cuhk_cufs
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases cuhk_cufs

  $ # Adapting layers 1-5
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases cuhk_cufs
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases cuhk_cufs

  $ # Adapting layers 1-6
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases cuhk_cufs
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases cuhk_cufs

  $ # Adapting all layers
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases cuhk_cufs
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases cuhk_cufs


CUHK-CUFSF
##########

Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
the viewed sketch database.

 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | Image size     | CNN              | Rank-1 - siamese joint    | Rank 1 - siamese single   + Rank 1 - Triplet          |
 +================+==================+===========================+===========================+===========================+
 | 160 x 160      | Inception V2     | 16.559(0.717)             | 16.559(0.717)             | 16.559(0.717)             |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt first**  | **20.081%(1.327)**        | 24.008(1.164)             |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-2**    | **30.769%(1.235)**        |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-4**    | **38.178%(2.11)**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-5**    | **41.417(0.731)**         | 62.186(0.198)             | 62.186(0.198)             |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-6**    | **41.903(0.479)**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt All**    | **6.64(1.956)**           |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+


.. image:: ../plots/transfer-learning/cuhk_cufsf/siamese_cmc.png
 :scale: 100 %
 

The same trends as before can be observed.
The recognition rate increases according with the amount of layers we adapt.
Adapting the whole networks leads to a heavy overfitting, making the error rates drop to ~20%.

The sequence of commands below run the amount of experiments necessary to run this plot::

  $ # Adapting first layer
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases cuhk_cufsf
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases cuhk_cufsf
  
  $ # Adapting layers 1-2  
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases cuhk_cufsf
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases cuhk_cufsf
     
  $ # Adapting layers 1-4
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases cuhk_cufsf
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases cuhk_cufsf

  $ # Adapting layers 1-5
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases cuhk_cufsf
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases cuhk_cufsf

  $ # Adapting layers 1-6
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases cuhk_cufsf
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases cuhk_cufsf

  $ # Adapting all layers
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases cuhk_cufsf
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases cuhk_cufsf


CASIA NIR-VIS
#############

Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
a NIR database.


 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | Image size     | CNN              | Rank-1 - siamese joint    | Rank 1 - siamese single   | Rank 1 - Triplet          |
 +================+==================+===========================+===========================+===========================+
 | 160 x 160      | Inception V2     | 73.803(1.226)             | 73.803(1.226)             |  74.066(0.191)            |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt first**  | **42.729 (0.961)**        | 80.589(1.049)             |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-2**    | **62.694(1.466)**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-4**    | **74.733(0.795)**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-5**    | **45.117(5.414)**         | 94.043(0.39)              | 90.166(0.129)             |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-6**    | **7.739(0.579)**          |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt All**    | **10.928(0.247)**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 


The same trends as before can be observed.
The recognition rate increases according with the amount of layers we adapt, but with on particular difference.
Unlike the other modalites, for this one the error rate starts to decay quite "early".
A drop can be observed when the adaptation takes place from layers 1-5 (**Conv2d_1a_3x3**, **Conv2d_2a_3x3**, **"Conv2d_2b_3x3"**, **Conv2d_3b_1x1**, **Conv2d_4a_3x3** and **Mixed_5b**).
The sequence of commands below run the amount of experiments necessary to run this plot::

  $ # Adapting first layer
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases casia_nir_vis
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases casia_nir_vis
  
  $ # Adapting layers 1-2  
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases casia_nir_vis
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases casia_nir_vis
     
  $ # Adapting layers 1-4
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases casia_nir_vis
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases casia_nir_vis

  $ # Adapting layers 1-5
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases casia_nir_vis
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases casia_nir_vis

  $ # Adapting layers 1-6
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases casia_nir_vis
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases casia_nir_vis

  $ # Adapting all layers
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases casia_nir_vis
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases casia_nir_vis



NIVL
####

Follow bellow the results in terms of Rank-1 recognition rate under this assumption using 
a NIR database.

 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | Image size     | CNN              | Rank-1 - siamese joint    | Rank 1 - siamese single   | Rank 1 - Triplet          |
 +================+==================+===========================+===========================+===========================+
 | 160 x 160      | Inception V2     | 60.009%(2.518)            | 88.283(0.399)             |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt first**  | **69.393%(1.847)**        | 90.546(0.066)             |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-2**    | **85.19%(1.906)**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-4**    | **86.777%(1.908)**        |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-5**    | **73.968%(1.574)**        | 95.026(0.367)             |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt 1-6**    | **77.964 (     )**        |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 | **160 x 160**  | **Adapt All**    | **-    % (    )**         |                           |                           |
 +----------------+------------------+---------------------------+---------------------------+---------------------------+
 

The exact same trend observed in CASIA-NIR-VIS can be observerved for this database. 
The recognition rate increases according with the amount of layers we adapt and starts to decay in the exact same point.

The sequence of commands below run the amount of experiments necessary to run this plot::

  $ # Adapting first layer
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases nivl
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_first_layer --databases nivl
  
  $ # Adapting layers 1-2  
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases nivl
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_2 --databases nivl
     
  $ # Adapting layers 1-4
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases nivl
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_4 --databases nivl

  $ # Adapting layers 1-5
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases nivl
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_5 --databases nivl

  $ # Adapting layers 1-6
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases nivl
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_layers_1_6 --databases nivl

  $ # Adapting all layers
  $ bob_htface_train_cnn.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases nivl
  $ bob_htface_baselines.py --baselines idiap_casia_inception_v2_gray_adapt_all_layers --databases nivl


Final Discussions
#################

In this section we tried to joint model 2 image modalities by adapting layer by layer of our prior :math:`\phi`.
At this point we observed two major trends.

The first one is for all tested image modalities, we observed an increase of recognition rate once we go deeper with adapations.
As a second trend we observed points of saturation in the adaptations.
More interesting than that is that such points of saturation are shared between image modalities of the same type.
For the thermogram database, we could observe the saturation from layer 6 (**Adapt 1-6**).
For the sketch databases, points of saturation were obseved from layer 5 (**Adapt 1-5**).
For the NIR databases, points of saturation were obseved from layer 4 (**Adapt 1-4**).

**What can we hypothesize about the last observation?**
With such observations, we could hypothesize that for different image modalities, high level elements cannot be detected at certain stages of the network, vanishing the input signal.
Using an analogy we could interpret this as the following; imagine that the first layer of :math:`\phi` contains a set of high pass filters (edge detectors) that are very suitable for VIS images.
Providing themograms to this network (which are poor in high frequency elements) the signal is vanished right in the begining of the network.
The same is not true for sketch database, which is very rich in high frequency components and this may adding lots of noise into signal.
We clearlly need to readapt those feature detectors, in order to deliver better low level features to the following layers.

One possible exercise that we can do to observe such phenomena is to observe the signal of :math:`x_A` and :math:`x_B` in terms of Fourier decomposition after every convolution with our
**orignal** :math:`\phi` and the :math:`\phi` **after the adaptation** as can be observed in the figure below for the CUHK_CUFS database.

|pic1| |pic2|

|pic3| |pic4|

.. |pic1| image:: ../plots/transfer-learning/cuhk_cufs/adaptation_pictures/cuhk_cufs_photo_idiap_casia_inception_v2_gray.png
   :width: 45%
   
.. |pic2| image:: ../plots/transfer-learning/cuhk_cufs/adaptation_pictures/cuhk_cufs_photo_idiap_casia_inception_v2_gray_adapt_layers_1_6.png
   :width: 45%
   
.. |pic3| image:: ../plots/transfer-learning/cuhk_cufs/adaptation_pictures/cuhk_cufs_sketch_idiap_casia_inception_v2_gray.png
   :width: 45%
   
.. |pic4| image:: ../plots/transfer-learning/cuhk_cufs/adaptation_pictures/cuhk_cufs_sketch_idiap_casia_inception_v2_gray_adapt_layers_1_6.png
   :width: 45%

We can observe a boosting for some convolved signals in the adapted network (for this particular pair of signals) in comparison with the non adapted ones.

Let's do the same exploratory analysis for the CASIA NIR VIS database.

|pic5| |pic6|

|pic7| |pic8|
   
.. |pic5| image:: ../plots/transfer-learning/casia_nir_vis/adaptation_pictures/casia_nir_vis_VIS_idiap_casia_inception_v2_gray.png
   :width: 45%
   
.. |pic6| image:: ../plots/transfer-learning/casia_nir_vis/adaptation_pictures/casia_nir_vis_VIS_idiap_casia_inception_v2_gray_adapt_layers_1_6.png
   :width: 45%

.. |pic7| image:: ../plots/transfer-learning/casia_nir_vis/adaptation_pictures/casia_nir_vis_NIR_idiap_casia_inception_v2_gray.png
   :width: 45%
   
.. |pic8| image:: ../plots/transfer-learning/casia_nir_vis/adaptation_pictures/casia_nir_vis_NIR_idiap_casia_inception_v2_gray_adapt_layers_1_6.png
   :width: 45%


Now the same analysis for the PolaThermal.

Let's do the same exploratory analysis for the CASIA NIR VIS database.

|pic5| |pic6|

|pic7| |pic8|
   
.. |pic9| image:: ../plots/transfer-learning/pola_thermal/adaptation_pictures/pola_thermal_VIS_idiap_casia_inception_v2_gray.png
   :width: 45%
   
.. |pic10| image:: ../plots/transfer-learning/pola_thermal/adaptation_pictures/pola_thermal_VIS_idiap_casia_inception_v2_gray_adapt_layers_1_6.png
   :width: 45%

.. |pic11| image:: ../plots/transfer-learning/pola_thermal/adaptation_pictures/pola_thermal_THERMAL_idiap_casia_inception_v2_gray.png
   :width: 45%
   
.. |pic12| image:: ../plots/transfer-learning/pola_thermal/adaptation_pictures/pola_thermal_THERMAL_idiap_casia_inception_v2_gray_adapt_layers_1_6.png
   :width: 45%



In order to make a more summarized analysis of this phenomena in the whole dataset, we represent every convolved signal by the sum of its FFTs.
Hence, every image showed above is represented by a scalar which is accumulated for whole dataset.
The next pdfs shows this summarized representation for each dataset.

 - :download:`CUHK CUFS <../plots/transfer-learning/cuhk_cufs/adaptation_pictures/resnet_0-6_cufs_norm.pdf>`
 - :download:`Polathermal <../plots/transfer-learning/pola_thermal/adaptation_pictures/resnet_0-6_pola_norm.pdf>`
 - :download:`CUHK CUFSF <../plots/transfer-learning/cuhk_cufsf/adaptation_pictures/0-6_cufsf_norm.pdf>`


          
Triplet Networks
----------------

.. Todo:: To be done


