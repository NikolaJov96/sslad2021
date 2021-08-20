import cv2

from structures.sslad_2d.sslad_dataset import SSLADDataset
from structures.sslad_2d.image import Image


if __name__ == '__main__':
    """
    Cycles through training set images and displays them with drawn annotations
    """

    dataset = SSLADDataset()
    dataset.load()

    window_name = 'Annotated images'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for training_image in dataset.training_images:

        img = training_image.draw_annotations()

        resized_img = Image.resize_to_width(img, 1000)

        cv2.imshow(window_name, resized_img)
        # Exit on esc
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()
