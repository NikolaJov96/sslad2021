import cv2
import numpy as np

from structures.sslad_2d.definitions import SSLADDatasetTypes
from structures.sslad_2d.sslad_dataset import SSLADDataset
from structures.sslad_2d.image import Image


if __name__ == '__main__':
    """
    Displays lighting information about images in the training set
    """

    dataset = SSLADDataset()
    dataset.load()

    window_name = 'Images'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for training_image in dataset.get_subset(SSLADDatasetTypes.TRAINING):

        img = training_image.get_cv2_img()

        average_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        print('\raverage brightness: {}'.format(average_brightness), end='')

        resized_img = Image.resize_to_width(img, 1000)
        cv2.imshow(window_name, Image.resize_to_width(img, 1000))

        # Exit on esc
        if cv2.waitKey(0) == 27:
            exit(0)

    cv2.destroyAllWindows()
