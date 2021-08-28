import cv2
import numpy as np

from PIL import Image as PILImage


class Augmentation:
    """
    Base class for augmentations that can be added to an Image object
    and will be applied on image content when loaded
    """

    def apply_to_image_parameters(self, image):
        """
        Updates Image parameters when added to the Image object,
        mainly annotation bounding boxes
        """

        pass

    def apply_to_cv2_img(self, img):
        """
        Transformation over an image in the cv2 format
        """

        return img

    def apply_to_pil_img(self, img):
        """
        Transformation over an image in the pil format
        """

        return img

class ScaleBrightness(Augmentation):
    """
    Simple pixel value scaling with a given coefficient
    """

    def __init__(self, coefficient):
        """
        Initialize the coefficient
        """
        super().__init__()

        self.coefficient = coefficient


    def apply_to_cv2_img(self, img):
        """
        Scaling brightness of an image in the cv2 format
        """

        return (img * self.coefficient).astype(np.uint8)

    def apply_to_pil_img(self, img):
        """
        Scaling brightness of an image in the pil format
        """

        return PILImage.eval(img, lambda p: p * self.coefficient)

class FlipImage(Augmentation):
    """
    Image horizontal flip
    """

    def apply_to_image_parameters(self, image):
        """
        Updates image annotation after the flip
        """

        for annotation in image.annotations:
            annotation.x = image.width - annotation.x - annotation.w

    def apply_to_cv2_img(self, img):
        """
        Image flipping of an image in the cv2 format
        """

        return cv2.flip(img, 1)

    def apply_to_pil_img(self, img):
        """
        Image flipping of an image in the pil format
        """

        return img.transpose(method=PILImage.FLIP_LEFT_RIGHT)
