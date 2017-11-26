#-q sgpu

#./bin/jman submit --name TRANSF -q q1d --io-big \

./bin/jman submit --name TRANSF -q sgpu \
 ./bin/bob_tf_train_generic \
 ./src/bob.bio.htface/bob/bio/htface/config/tensorflow/siamese_transfer_learning/regular_transfer/POLATHERMAL_CASIA_inception_resnet_v2_center_loss_GRAY.py
  
