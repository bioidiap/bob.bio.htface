############ NEW SCRIPTS


#./bin/jman submit -q sgpu --name CHUK-TR-1 --environment="LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0" -- \
# ./bin/bob_tf_train_generic ./src/bob.bio.htface/bob/bio/htface/config/tensorflow/siamese_transfer_learning/cuhk_cufs/learn_first_layer/split1.py

./bin/jman submit -q sgpu --name CHUK-TR-2 --environment="LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0" -- \
 ./bin/bob_tf_train_generic ./src/bob.bio.htface/bob/bio/htface/config/tensorflow/siamese_transfer_learning/cuhk_cufs/learn_first_layer/split2.py


./bin/jman submit -q sgpu --name CHUK-TR-3 --environment="LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0" -- \
 ./bin/bob_tf_train_generic ./src/bob.bio.htface/bob/bio/htface/config/tensorflow/siamese_transfer_learning/cuhk_cufs/learn_first_layer/split3.py


./bin/jman submit -q sgpu --name CHUK-TR-4 --environment="LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0" -- \
 ./bin/bob_tf_train_generic ./src/bob.bio.htface/bob/bio/htface/config/tensorflow/siamese_transfer_learning/cuhk_cufs/learn_first_layer/split4.py


./bin/jman submit -q sgpu --name CHUK-TR-5 --environment="LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0" -- \
 ./bin/bob_tf_train_generic ./src/bob.bio.htface/bob/bio/htface/config/tensorflow/siamese_transfer_learning/cuhk_cufs/learn_first_layer/split5.py

