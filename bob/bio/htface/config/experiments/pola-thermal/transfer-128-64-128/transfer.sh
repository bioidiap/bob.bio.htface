#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

for split in 'split1.py' 'split2.py' 'split3.py' 'split4.py' \
           'split5.py'
do
 
  command_string="./bin/verify.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/pola-thermal/$split ./src/bob.bio.htface/bob/bio/htface/config/experiments/pola-thermal/transfer-128-64-128/base_setup.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/pola-thermal/transfer-128-64-128/$split "
  command_string+=" --temp-directory /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/ "
  command_string+=" --result-directory /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/ "
  command_string+=" -vvv "
#  command_string+=" -g demanding "
  command_string+=" --environment \"LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0/bin\""\
  command_string+=" --preprocessed-directory /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/INITIAL_CHECKPOINT/split1/preprocessed/ "
  
 $command_string  
 
done


