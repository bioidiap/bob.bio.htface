#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

for split in 'split1.py' 'split2.py' 'split3.py' 'split4.py' \
           'split5.py'
do

  command_string="./bin/verify.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/cuhk-cufs/$split ./src/bob.bio.htface/bob/bio/htface/config/experiments/cuhk-cufs/transfer_first_layer/base_setup.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/cuhk-cufs/transfer_first_layer/$split "
  command_string+=" --temp-directory /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-regular-transfer-first-layer/idiap_inception_v2_gray--casia/ "
  command_string+=" --result-directory /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-regular-transfer-first-layer/idiap_inception_v2_gray--casia/ " 
  command_string+=" -vvv "
#  command_string+=" -g demanding "
  command_string+=" --environment \"LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0/bin\""\
  command_string+=" --preprocessed-directory /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split1/preprocessed "
  
 $command_string  
 
done


#for step in 'preprocessing' 'extraction'
#do

  # Just to have the training set done
#  command_string="./bin/verify.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/resnet-inception-v2-gray--casia.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/cuhk-cufs/split1.py "
#  command_string+=" -a pca "
#  command_string+=" --temp-directory /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/ "
#  command_string+=" --result-directory /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/ "
#  command_string+=" -vvv "
  #  command_string+=" -g demanding "
#  command_string+=" --environment \"LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0/bin#\""\
#  command_string+=" --preprocessed-directory /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split1/preprocessed "
#  command_string+=" --extracted-directory /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split1/extracted "
#  command_string+=" -o $step "

#  $command_string

#done
