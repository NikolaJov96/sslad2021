import cv2
import sys

from structures.sslad_2d.definitions import SSLADDatasetTypes
from structures.sslad_2d.sslad_dataset import SSLADDataset
from structures.sslad_2d.image import Image


if __name__ == '__main__':
    """
    Cycles through unlabeled images and displays their annotaitons from a given save file
    """

    unlabeled_data_file = sys.argv[1]
    starting_image = int(sys.argv[2])

    dataset = SSLADDataset()
    dataset.load(unlabeled_data_file=unlabeled_data_file)

    unlabeled_images = dataset.get_subset(SSLADDatasetTypes.UNLABELED)

    total_annotated = len([image for image in unlabeled_images if image.is_annotated])
    print('total annotated unlabeled images: {}'.format(total_annotated))

    window_name = 'Annotated images'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for i, image in enumerate(unlabeled_images[starting_image:]):

        print('\rimage {}'.format(starting_image + i), end='')

        img = image.draw_annotations()

        resized_img = Image.resize_to_width(img, 1000)

        cv2.imshow(window_name, resized_img)
        # Exit on esc
        if cv2.waitKey(0) == 27:
            break
    print()

    cv2.destroyAllWindows()
