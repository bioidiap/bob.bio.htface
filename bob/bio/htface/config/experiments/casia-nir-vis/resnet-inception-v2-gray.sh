#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

#for split in 'split1.py'
#

#for split in 'split1.py' 'split2.py' 'split3.py' 'split4.py' \
#           'split5.py' 'split6.py' 'split7.py' 'split8.py' 'split9.py' 'split10.py'
#do 
#  command_string="./bin/verify.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/resnet-inception-v2-gray--casia.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/casia-nir-vis/$split ./src/bob.bio.htface/bob/bio/htface/config/experiments/casia-nir-vis/resnet-inception-v2-gray.py "
#  command_string+=" -g demanding --groups eval "
#  command_string+=" -vvv "

#  $command_string 
 
#done



for split in 'split1.py' 'split2.py' 'split3.py' 'split4.py' \
           'split5.py' 'split6.py' 'split7.py' 'split8.py' 'split9.py' 'split10.py'
do
  command_string="./bin/verify.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/resnet-inception-v2-gray--casia.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/casia-nir-vis/$split ./src/bob.bio.htface/bob/bio/htface/config/experiments/casia-nir-vis/resnet-inception-v2-gray.py "
  #command_string+=" -g demanding "
  command_string+=" -vvv -a pca -o preprocessing"

  $command_string 
 
done

