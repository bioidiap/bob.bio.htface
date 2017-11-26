#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


./bin/evaluate_and_squash.py \
 --dev-files \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split1/search_split1_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split2/search_split2_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split3/search_split3_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split4/search_split4_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split5/search_split5_p2s/nonorm/scores-dev \
   \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-regular-transfer/idiap_inception_v2_gray--casia/split1/search_split1_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-regular-transfer/idiap_inception_v2_gray--casia/split2/search_split2_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-regular-transfer/idiap_inception_v2_gray--casia/split3/search_split3_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-regular-transfer/idiap_inception_v2_gray--casia/split4/search_split4_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-regular-transfer/idiap_inception_v2_gray--casia/split5/search_split5_p2s/nonorm/scores-dev \
   \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split1/search_split1_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split2/search_split2_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split3/search_split3_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split4/search_split4_p2s/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/CUHK-CUFS/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split5/search_split5_p2s/nonorm/scores-dev \
   \
  --legends resnet siamese-resnet siamese-128-64-128-resnet \
  --title title --xmin 0 --xmax 100 \
  --colors red blue green \
  --rr \
  --report-name /idiap/user/tpereira/gitlab/workspace_HTFace/src/bob.bio.htface/bob/bio/htface/config/experiments/cuhk-cufs/results.pdf

