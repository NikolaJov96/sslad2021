import cv2
import numpy as np
from copy import deepcopy
from imgaug import augmenters as iaa
from math import sqrt
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


class AddNoise(Augmentation):
    """
    Add noise to the image
    """

    def __init__(self, scale=0.1):
        """
        Initialize parameters
        """
        super().__init__()

        self.scale = scale * 255

    def apply_to_cv2_img(self, img):
        """
        Add noise to the image in the cv2 format
        """

        # Use pil to generate the image
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img = self.apply_to_pil_img(pil_img)

        return cv2.cvtColor(np.asarray(pil_img), cv2.cv2.COLOR_RGB2BGR)

    def apply_to_pil_img(self, img):
        """
        Add noise to the image in the pil format
        """

        im_arr = np.asarray(img)

        # Gaussian noise
        aug = iaa.AdditiveGaussianNoise(loc=0, scale=self.scale)
        # Poisson noise
        # aug = iaa.AdditivePoissonNoise(lam=10.0, per_channel=True)
        # Salt and pepper noise
        # aug = iaa.SaltAndPepper(p=0.05)

        im_arr = aug.augment_image(im_arr)

        return PILImage.fromarray(im_arr).convert('RGB')


def main():
    """
    Show augmentations on one training set image
    """

    from structures.sslad_2d.image import Image
    from structures.sslad_2d.sslad_dataset import SSLADDataset

    dataset = SSLADDataset()
    dataset.load()
    image = dataset.training_images[0]

    augmentations = [
        Augmentation(),
        ScaleBrightness(0.5),
        FlipImage(),
        AddNoise(0.05),
        AddNoise(0.3),
    ]

    window_name = 'Augmentations'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for augmentation in augmentations:

        augmented_image = deepcopy(image)
        augmented_image.add_augmentation(augmentation)
        augmented_img = augmented_image.draw_annotations()

        cv2.imshow(window_name, Image.resize_to_width(augmented_img, 1000))

        # Exit on esc
        if cv2.waitKey(0) == 27:
            exit(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
