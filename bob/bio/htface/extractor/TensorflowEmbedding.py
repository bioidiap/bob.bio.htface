import numpy
from bob.bio.base.extractor import Extractor
from bob.learn.tensorflow.trainers import SiameseTrainer
from bob.learn.tensorflow.network import Embedding


class TensorflowEmbedding(Extractor):

    """
    """

    def __init__(
            self,
            path,
            normalizer=None
    ):
        Extractor.__init__(self, skip_extractor_training=True)

        # block parameters
        # initialize this when called for the first time
        # since caffe may not work if it is compiled to run with gpu
        self.new_feature = None
        self.feature_layer = ""

        trainer = SiameseTrainer(None,
                                 iterations=0,
                                 analizer=None,
                                 temp_dir=None)

        trainer.create_network_from_file(path)
        self.embedding = Embedding(trainer.data_ph['left'], trainer.graph['left'], normalizer=normalizer)

    def __call__(self, image):
        """__call__(image) -> feature

        Extract features

        **Parameters:**

        image : 3D :py:class:`numpy.ndarray` (floats)
          The image to extract the features from.

        **Returns:**

        feature : 2D :py:class:`numpy.ndarray` (floats)
          The extracted features
        """

        data = numpy.zeros(shape=(1, image.shape[0], image.shape[1], 1))
        data[0, :, :, 0] = image

        feature = self.embedding(data)[0]

        return feature

    # re-define the train function to get it non-documented
    def train(*args, **kwargs): raise NotImplementedError("This function is not implemented and should not be called.")

    def load(*args, **kwargs): pass
