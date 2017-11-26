#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

#for split in 'split1.py' 'split2.py' 'split3.py' 'split4.py' \
#           'split5.py'
#do

#  command_string="./bin/verify.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/resnet-inception-v2-gray--casia.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/nivl/$split ./src/bob.bio.htface/bob/bio/htface/config/experiments/nivl/resnet-inception-v2-gray.py "
#  
#  command_string+=" --groups dev "
  #command_string+=" -g demanding "  
#  command_string+=" -vvv "

#  $command_string
 
#done



#for split in 'split1.py'
#do

#  command_string="./bin/verify.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/resnet-inception-v2-gray--casia.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/nivl/$split ./src/bob.bio.htface/bob/bio/htface/config/experiments/nivl/resnet-inception-v2-gray.py "
  
#  command_string+=" --groups dev "
  #command_string+=" -g demanding "  
#  command_string+=" -vvv "
#  command_string+=" -a pca "
#  command_string+=" -o preprocessing "

#  $command_string
#done

for split in 'split1.py'
do

  command_string="./bin/verify.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/resnet-inception-v2-gray--casia.py ./src/bob.bio.htface/bob/bio/htface/config/experiments/nivl/$split ./src/bob.bio.htface/bob/bio/htface/config/experiments/nivl/resnet-inception-v2-gray.py "
  
  command_string+=" --groups dev "
  #command_string+=" -g demanding "  
  command_string+=" -vvv "
  command_string+=" -a pca "
  command_string+=" -o extraction "

  $command_string
done
