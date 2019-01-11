#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST
#
# Copyright (C) Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This file contains the python (distutils/setuptools) instructions so your
# package can be installed on **any** host system. It defines some basic
# information like the package name for instance, or its homepage.
#
# It also defines which other packages this python package depends on and that
# are required for this package's operation. The python subsystem will make
# sure all dependent packages are installed or will install them for you upon
# the installation of this package.
#
# The 'buildout' system we use here will go further and wrap this package in
# such a way to create an isolated python working environment. Buildout will
# make sure that dependencies which are not yet installed do get installed, but
# **without** requiring administrative privileges on the host system. This
# allows you to test your package with new python dependencies w/o requiring
# administrative interventions.

from setuptools import setup, dist

dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages

install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name='bob.bio.htface',
    version=open("version.txt").read().rstrip(),
    description='Tools for running heterogeneous face recognition experiments',

    url='https://gitlab.idiap.ch/bob/bob.bio.htface',
    license='BSD',
    author='Tiago de Freitas Pereira',
    author_email='tiago.pereira@idiap.ch',
    keywords='bob, biometric recognition, evaluation',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires=install_requires,

    # Your project should be called something like 'bob.<foo>' or
    # 'bob.<foo>.<bar>'. To implement this correctly and still get all your
    # packages to be imported w/o problems, you need to implement namespaces
    # on the various levels of the package and declare them here. See more
    # about this here:
    # http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
    #
    # Our database packages are good examples of namespace implementations
    # using several layers. You can check them out here:
    # https://gitlab.idiap.ch/bob/bob/wikis/Packages


    # This entry defines which scripts you will have inside the 'bin' directory
    # once you install the package (or run 'bin/buildout'). The order of each
    # entry under 'console_scripts' is like this:
    #   script-name-at-bin-directory = module.at.your.library:function
    #
    # The module.at.your.library is the python file within your library, using
    # the python syntax for directories (i.e., a '.' instead of '/' or '\').
    # This syntax also omits the '.py' extension of the filename. So, a file
    # installed under 'example/foo.py' that contains a function which
    # implements the 'main()' function of particular script you want to have
    # should be referred as 'example.foo:main'.
    #
    # In this simple example we will create a single program that will print
    # the version of bob.
    entry_points={

        'bob.bio.database': [
            'cuhk-cufs    = bob.bio.htface.configs.databases.cuhk_cufs:database',
            'cuhk-cufsf   = bob.bio.htface.configs.databases.cuhk_cufsf:database',
            'nivl         = bob.bio.htface.configs.databases.nivl:database',
            'pola_thermal = bob.bio.htface.configs.databases.pola_thermal:database',
            'thermal = bob.bio.htface.configs.databases.thermal:database',
            'casia-nir-vis-2 = bob.bio.htface.configs.databases.casia_nir_vis:database',
            'fargo = bob.bio.htface.configs.databases.fargo:database',
            'fargo_depth = bob.bio.htface.configs.databases.fargo_depth:database',
            'ldhf = bob.bio.htface.configs.databases.ldhf:database',
            'eprip = bob.bio.htface.configs.databases.eprip:database',
        ],
        
        'console_scripts' : [
            'gmm_responsibility_map.py = bob.bio.htface.script.gmm_responsibility_map:main',
            'plot_covariate_shift.py = bob.bio.htface.script.plot_covariate_shift:main',
            'grassmann_test.py = bob.bio.htface.script.grassmann_test:main',
            'bob_htface_evaluate_and_squash.py = bob.bio.htface.script.evaluate_and_squash:main',
            'bob_htface_convolve_and_view.py = bob.bio.htface.script.convolve_and_view:main',
            'bob_htface_recrate_vs_nparameters.py = bob.bio.htface.script.recrate_vs_nparameters:main', 
            'bob_htface_recrate_vs_nparameters_by_hand.py = bob.bio.htface.script.recrate_vs_nparameters_by_hand:main',

            'bob_htface_plot_tsne_modality_database.py = bob.bio.htface.script.plot_tsne_modality_database:main',
        ],
        
      'bob.bio.baseline':[

           # FACE REC BASELINES
          'htface_vgg16 = bob.bio.htface.baselines.standard_facerec:htface_vgg16',
          'htface_facenet = bob.bio.htface.baselines.standard_facerec:htface_facenet',
          'htface_lightcnn = bob.bio.htface.baselines.standard_facerec:htface_lightcnn',


          'htface_idiap_msceleb_inception_v1_centerloss_gray = bob.bio.htface.baselines.standard_facerec:htface_idiap_msceleb_inception_v1_centerloss_gray',
          'htface_idiap_msceleb_inception_v1_centerloss_rgb = bob.bio.htface.baselines.standard_facerec:htface_idiap_msceleb_inception_v1_centerloss_rgb',

          'htface_idiap_msceleb_inception_v2_centerloss_gray = bob.bio.htface.baselines.standard_facerec:htface_idiap_msceleb_inception_v2_centerloss_gray',
          'htface_idiap_msceleb_inception_v2_centerloss_rgb = bob.bio.htface.baselines.standard_facerec:htface_idiap_msceleb_inception_v2_centerloss_rgb',



          #'htface_idiap_casia_inception_v2_centerloss_gray = bob.bio.htface.baselines.standard_facerec:htface_idiap_casia_inception_v2_centerloss_gray',
          #'htface_idiap_casia_inception_v2_centerloss_rgb = bob.bio.htface.baselines.standard_facerec:htface_idiap_casia_inception_v2_centerloss_rgb',

          #'htface_idiap_casia_inception_v1_centerloss_gray = bob.bio.htface.baselines.standard_facerec:htface_idiap_casia_inception_v1_centerloss_gray',
          #'htface_idiap_casia_inception_v1_centerloss_rgb = bob.bio.face_ongoing.baselines.idiap_inception_v1:htface_idiap_casia_inception_v1_centerloss_rgb',
 
 
          # ISV
          'isv_g1024_u50 = bob.bio.htface.baselines.sota_baselines.isv_baselines:isv_g512_u1024',
          'isv_g512_u50 = bob.bio.htface.baselines.sota_baselines.isv_baselines:isv_g512_u50',
          'isv_g256_u50 = bob.bio.htface.baselines.sota_baselines.isv_baselines:isv_g256_u50',
          'isv_g128_u50 = bob.bio.htface.baselines.sota_baselines.isv_baselines:isv_g128_u50',
          'isv_g64_u50 = bob.bio.htface.baselines.sota_baselines.isv_baselines:isv_g64_u50',
 

          'isv_g512_u50_LBP = bob.bio.htface.baselines.sota_baselines.isv_lbp_baselines:isv_g512_u50_LBP',
          'isv_g256_u50_LBP = bob.bio.htface.baselines.sota_baselines.isv_lbp_baselines:isv_g256_u50_LBP',
          'isv_g128_u50_LBP = bob.bio.htface.baselines.sota_baselines.isv_lbp_baselines:isv_g128_u50_LBP',
          'isv_g64_u50_LBP = bob.bio.htface.baselines.sota_baselines.isv_lbp_baselines:isv_g64_u50_LBP',

 
          # SOTA BASELINES
          'htface_mlbphs = bob.bio.htface.baselines.sota_baselines:htface_mlbphs',
          'htface_classic_lbp = bob.bio.htface.baselines.sota_baselines:htface_classic_lbp',
          'htface_multiscale_features = bob.bio.htface.baselines.sota_baselines:htface_multiscale_features',
          'htface_gfkgabor = bob.bio.htface.baselines.sota_baselines:htface_gfkgabor',

          # DSU - HTFACE-BASELINE
          
          # V2
          
          # ADAPT FIRST
          'siamese_inceptionv2_first_layer_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_first:inception_resnet_v2_siamese_adapt_first',
          'siamese_inceptionv2_first_layer_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_first:inception_resnet_v2_siamese_adapt_first_betas',
          'triplet_inceptionv2_first_layer_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_first:inception_resnet_v2_triplet_adapt_first',
          'triplet_inceptionv2_first_layer_nonshared_betas_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_first:inception_resnet_v2_triplet_adapt_first_betas',
          
          # ADAPT 1-2
          'siamese_inceptionv2_adapt_1_2_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_2:inception_resnet_v2_siamese_adapt_1_2',
          'siamese_inceptionv2_adapt_1_2_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_2:inception_resnet_v2_siamese_adapt_1_2_betas',
          'triplet_inceptionv2_layers_1_2_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_2:inception_resnet_v2_triplet_adapt_1_2',
          'triplet_inceptionv2_layers_1_2_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_2:inception_resnet_v2_triplet_adapt_1_2_betas',

          # ADAPT 1-4
          'siamese_inceptionv2_adapt_1_4_nonshared_batch_norm_plda = bob.bio.htface.baselines.inception_v2.adapt_1_4:inception_resnet_v2_siamese_adapt_1_4_plda',
          'siamese_inceptionv2_adapt_1_4_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_4:inception_resnet_v2_siamese_adapt_1_4',
          'siamese_inceptionv2_adapt_1_4_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_4:inception_resnet_v2_siamese_adapt_1_4_betas',
          'triplet_inceptionv2_layers_1_4_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_4:inception_resnet_v2_triplet_adapt_1_4',
          'triplet_inceptionv2_layers_1_4_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_4:inception_resnet_v2_triplet_adapt_1_4_betas',

          # ADAPT 1-5
          'siamese_inceptionv2_adapt_1_5_nonshared_batch_norm_plda = bob.bio.htface.baselines.inception_v2.adapt_1_5:inception_resnet_v2_siamese_adapt_1_5_plda',
          'siamese_inceptionv2_adapt_1_5_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_5:inception_resnet_v2_siamese_adapt_1_5',
          'siamese_inceptionv2_adapt_1_5_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_5:inception_resnet_v2_siamese_adapt_1_5_betas',
          'triplet_inceptionv2_layers_1_5_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_5:inception_resnet_v2_triplet_adapt_1_5',
          'triplet_inceptionv2_layers_1_5_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_5:inception_resnet_v2_triplet_adapt_1_5_betas',

          # ADAPT 1-6
          'siamese_inceptionv2_adapt_1_6_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_6:inception_resnet_v2_siamese_adapt_1_6',
          'siamese_inceptionv2_adapt_1_6_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_6:inception_resnet_v2_siamese_adapt_1_6_betas',
          'triplet_inceptionv2_layers_1_6_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_6:inception_resnet_v2_triplet_adapt_1_6',
          'triplet_inceptionv2_layers_1_6_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v2.adapt_1_6:inception_resnet_v2_triplet_adapt_1_6_betas',
          

          # V1
          
          # ADAPT FIRST
          'siamese_inceptionv1_first_layer_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_first:inception_resnet_v1_siamese_adapt_first',
          'siamese_inceptionv1_first_layer_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_first:inception_resnet_v1_siamese_adapt_first_betas',
          'triplet_inceptionv1_first_layer_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_first:inception_resnet_v1_triplet_adapt_first',
          'triplet_inceptionv1_first_layer_nonshared_betas_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_first:inception_resnet_v1_triplet_adapt_first_betas',
          
          # ADAPT 1-2
          'siamese_inceptionv1_adapt_1_2_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_2:inception_resnet_v1_siamese_adapt_1_2',
          'siamese_inceptionv1_adapt_1_2_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_2:inception_resnet_v1_siamese_adapt_1_2_betas',
          'triplet_inceptionv1_layers_1_2_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_2:inception_resnet_v1_triplet_adapt_1_2',
          'triplet_inceptionv1_layers_1_2_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_2:inception_resnet_v1_triplet_adapt_1_2_betas',

          # ADAPT 1-4
          'siamese_inceptionv1_adapt_1_4_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_4:inception_resnet_v1_siamese_adapt_1_4',
          'siamese_inceptionv1_adapt_1_4_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_4:inception_resnet_v1_siamese_adapt_1_4_betas',
          'triplet_inceptionv1_layers_1_4_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_4:inception_resnet_v1_triplet_adapt_1_4',
          'triplet_inceptionv1_layers_1_4_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_4:inception_resnet_v1_triplet_adapt_1_4_betas',

          # ADAPT 1-5
          'siamese_inceptionv1_adapt_1_5_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_5:inception_resnet_v1_siamese_adapt_1_5',
          'siamese_inceptionv1_adapt_1_5_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_5:inception_resnet_v1_siamese_adapt_1_5_betas',
          'triplet_inceptionv1_layers_1_5_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_5:inception_resnet_v1_triplet_adapt_1_5',
          'triplet_inceptionv1_layers_1_5_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_5:inception_resnet_v1_triplet_adapt_1_5_betas',

          # ADAPT 1-6
          'siamese_inceptionv1_adapt_1_6_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_6:inception_resnet_v1_siamese_adapt_1_6',
          'siamese_inceptionv1_adapt_1_6_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_6:inception_resnet_v1_siamese_adapt_1_6_betas',
          'triplet_inceptionv1_layers_1_6_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_6:inception_resnet_v1_triplet_adapt_1_6',
          'triplet_inceptionv1_layers_1_6_betas_nonshared_batch_norm = bob.bio.htface.baselines.inception_v1.adapt_1_6:inception_resnet_v1_triplet_adapt_1_6_betas',


         ###### FREE DSU

          #'fdsu_inception_resnet_v2_siamese_adapt_first = bob.bio.htface.baselines.free_domain_specific_units.inception_v2.adapt_first:fdsu_inception_resnet_v2_siamese_adapt_first',
          
          #'fdsu_siamese_inceptionv2_adapt_1_4_nonshared_batch_norm = bob.bio.htface.baselines.free_domain_specific_units.inception_v2.adapt_1_4:fdsu_inception_resnet_v2_siamese_adapt_1_4',

          #'siamese_inceptionv2_adapt_1_4_nonshared_batch_norm_same_modality = bob.bio.htface.baselines.inception_v2.adapt_1_4:inception_resnet_v2_siamese_adapt_1_4_same_modality',
 
          #'siamese_inceptionv2_adapt_1_4_nonshared_batch_norm_random_pairs = bob.bio.htface.baselines.inception_v2.adapt_1_4:inception_resnet_v2_siamese_adapt_1_4_random_pairs',
 
          #'siamese_inceptionv2_adapt_1_4_nonshared_batch_norm_euclidean_loss = bob.bio.htface.baselines.inception_v2.adapt_1_4:inception_resnet_v2_siamese_adapt_1_4_euclidean_loss',


         ####### STYLE DSU
         
         #'styledsu_inception_resnet_v2_siamese_adapt_first = bob.bio.htface.baselines.style_domain_specific_units.inception_v2.adapt_first:styledsu_inception_resnet_v2_siamese_adapt_first',


         #'styledsu_siamese_inceptionv2_adapt_1_4_nonshared_batch_norm = bob.bio.htface.baselines.style_domain_specific_units.inception_v2.adapt_1_4:styledsu_siamese_inceptionv2_adapt_1_4_nonshared_batch_norm',
         
         #'styledsu_siamese_inceptionv2_adapt_1_4_betas_nonshared_batch_norm = bob.bio.htface.baselines.style_domain_specific_units.inception_v2.adapt_1_4:styledsu_siamese_inceptionv2_adapt_1_4_betas_nonshared_batch_norm',


        ### ADAPT 1-5
         #'styledsu_siamese_inceptionv2_adapt_1_5_nonshared_batch_norm = bob.bio.htface.baselines.style_domain_specific_units.inception_v2.adapt_1_5:styledsu_siamese_inceptionv2_adapt_1_5_nonshared_batch_norm',


        ### ADAPT 1-6
         #'styledsu_siamese_inceptionv2_adapt_1_6_nonshared_batch_norm = bob.bio.htface.baselines.style_domain_specific_units.inception_v2.adapt_1_6:styledsu_siamese_inceptionv2_adapt_1_6_nonshared_batch_norm',

         #'style_transfer_inception_v2 = bob.bio.htface.baselines.style_transfer.inception_v2:style_transfer_inception_v2',
 
      ],

      # bob bio scripts
      'bob.bio.cli': [
         'htface  = bob.bio.htface.script.htface:htface',
      ],


      # bob bio scripts
      'bob.bio.htface.cli': [
        'baseline  = bob.bio.htface.script.baseline:htface_baseline',
        'train_dsu = bob.bio.htface.script.domain_specic_units:htface_train_dsu',
        'filter_ldhf = bob.bio.htface.script.filter_ldhf:filter_ldhf',
        'fft_analysis = bob.bio.htface.script.fft_analysis:fft_analysis',
        'create_block_image = bob.bio.htface.script.create_block_image:create_block_image',
        'isv_intuition = bob.bio.htface.script.isv_intuition:isv_intuition',
        'evaluate_and_squash = bob.bio.htface.script.evaluate_and_squash:evaluate_and_squash',
      ],
        

    },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers=[
      'Framework :: Bob',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
