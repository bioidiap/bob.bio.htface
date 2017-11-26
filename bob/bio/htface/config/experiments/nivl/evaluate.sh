#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


./bin/evaluate_and_squash.py \
 --dev-files \
   /idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/split1/idiap-search_VIS-NIR_split1/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/split2/idiap-search_VIS-NIR_split2/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/split3/idiap-search_VIS-NIR_split3/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/split4/idiap-search_VIS-NIR_split4/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/split5/idiap-search_VIS-NIR_split5/nonorm/scores-dev \
  --legends resnet \
  --title title --xmin 0 --xmax 100 \
  --colors red \
  --rr \
  --report-name /idiap/user/tpereira/gitlab/workspace_HTFace/src/bob.bio.htface/bob/bio/htface/config/experiments/nivl/results.pdf

