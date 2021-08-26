import cv2
import sys

from structures.sslad_2d.definitions import SSLADDatasetTypes
from structures.sslad_2d.sslad_dataset import SSLADDataset
from structures.sslad_2d.image import Image


if __name__ == '__main__':
    """
    Cycles through training set images and displays them with drawn annotations
    """

    dataset_type = SSLADDatasetTypes.TRAINING
    if len(sys.argv) == 2:
        if sys.argv[1] == 'validation':
            dataset_type = SSLADDatasetTypes.VALIDATION
        elif sys.argv[1] == 'testing':
            dataset_type = SSLADDatasetTypes.TESTING
        elif sys.argv[1] == 'unlabeled':
            dataset_type = SSLADDatasetTypes.UNLABELED

    dataset = SSLADDataset()
    dataset.load()

    window_name = 'Annotated images'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for training_image in dataset.get_subset(dataset_type):

        img = training_image.draw_annotations()

        resized_img = Image.resize_to_width(img, 1000)

        cv2.imshow(window_name, resized_img)
        # Exit on esc
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()
