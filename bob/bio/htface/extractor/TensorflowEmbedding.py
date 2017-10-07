import numpy
from bob.bio.base.extractor import Extractor
import bob.ip.color

def bob2skimage(bob_image):
    """
    Convert bob color image to the skcit image
    """
    skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], bob_image.shape[0]))
    skimage[:,:,2] = bob_image[0, :, :]
    skimage[:,:,1] = bob_image[1, :, :]
    skimage[:,:,0] = bob_image[2, :, :]        
    
    #for i in range(bob_image.shape[0]):
    #    skimage[:, :, i] = bob_image[i, :, :]
    
    return skimage

class TensorflowEmbedding(Extractor):

    """
    """

    def __init__(
            self,
            tf_extractor
    ):
        Extractor.__init__(self, skip_extractor_training=True)
        self.tf_extractor = tf_extractor

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

        if image.ndim>2:
            image = bob2skimage(image)
            image = numpy.reshape(image, tuple([1] + list(image.shape)) )
            image = image.astype("float32")            
        else:
            image = numpy.reshape(image, tuple([1] + list(image.shape) + [1]) )
        
        features = self.tf_extractor(image)

        return features[0]

    # re-define the train function to get it non-documented
    def train(*args, **kwargs): raise NotImplementedError("This function is not implemented and should not be called.")

    def load(*args, **kwargs): pass
