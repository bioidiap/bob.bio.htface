#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


./bin/evaluate_and_squash.py \
 --dev-files \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/INITIAL_CHECKPOINT/split1/VIS-polarimetric-overall-split1/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/INITIAL_CHECKPOINT/split2/VIS-polarimetric-overall-split2/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/INITIAL_CHECKPOINT/split3/VIS-polarimetric-overall-split3/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/INITIAL_CHECKPOINT/split4/VIS-polarimetric-overall-split4/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/INITIAL_CHECKPOINT/split5/VIS-polarimetric-overall-split5/nonorm/scores-dev \
   \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split1/VIS-polarimetric-overall-split1/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split2/VIS-polarimetric-overall-split2/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split3/VIS-polarimetric-overall-split3/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split4/VIS-polarimetric-overall-split4/nonorm/scores-dev \
   /idiap/temp/tpereira/HTFace/POLA_THERMAL/RESNET_GRAY/CASIA_WEBFACE/siamese-transfer-128-64-128/idiap_inception_v2_gray--casia/split5/VIS-polarimetric-overall-split5/nonorm/scores-dev \
   \
  --legends resnet siamese-resnet-128-64-128 \
  --title title --xmin 0 --xmax 100 \
  --colors red blue\
  --rr \
  --report-name /idiap/user/tpereira/gitlab/workspace_HTFace/src/bob.bio.htface/bob/bio/htface/config/experiments/pola-thermal/results.pdf

