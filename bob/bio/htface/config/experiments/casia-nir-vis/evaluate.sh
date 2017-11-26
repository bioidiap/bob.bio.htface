#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


./bin/evaluate_and_squash.py \
 --dev-files \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split1/view2_1/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split2/view2_2/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split3/view2_3/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split4/view2_4/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split5/view2_5/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split6/view2_6/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split7/view2_7/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split8/view2_8/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split9/view2_9/nonorm/scores-eval \
   /idiap/temp/tpereira/HTFace/CBSR_NIR_VIS_2/idiap_inception_v2_gray--casia/split10/view2_10/nonorm/scores-eval \
  --legends resnet \
  --title title --xmin 0 --xmax 100 \
  --colors red \
  --rr
  --report-name /idiap/user/tpereira/gitlab/workspace_HTFace/src/bob.bio.htface/bob/bio/htface/config/experiments/casia-nir-vis/results.pdf

