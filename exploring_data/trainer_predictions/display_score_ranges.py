import cv2
import sys
from math import ceil

from structures.sslad_2d.definitions import SSLADDatasetTypes
from structures.sslad_2d.sslad_dataset import SSLADDataset
from structures.sslad_2d.image import Image


if __name__ == '__main__':
    """
    Cycles through unlabeled images and displays their annotaitons from a given save file
    Annotations from one score range are shown at the time
    """

    unlabeled_data_file = sys.argv[1]
    starting_image = int(sys.argv[2])
    range_step = float(sys.argv[3])

    # Set num of digits equal to the entered range step
    num_digits = len(sys.argv[3]) - 2

    dataset = SSLADDataset()
    dataset.load(unlabeled_data_file=unlabeled_data_file)

    unlabeled_images = dataset.get_subset(SSLADDatasetTypes.UNLABELED)

    window_name = 'Annotated images'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for i, image in enumerate(unlabeled_images[starting_image:]):

        for j in range(ceil(1.0 / range_step)):

            range_max = round(1.0 - j * range_step, num_digits)
            range_min = round(max(0.0, 1.0 - (j + 1) * range_step), num_digits)

            range_annotations = [
                annotation for annotation in image.annotations
                if annotation.score <= range_max and annotation.score > range_min
            ]

            print('{}\rimage {}, annotation score range {} - {}, annotations {}/{}'.format(
                    ' ' * 10,
                    starting_image + i,
                    range_max,
                    range_min,
                    len(range_annotations),
                    len(image.annotations)
                ),
                end=''
            )

            img = image.draw_annotations(annotations=range_annotations)

            resized_img = Image.resize_to_width(img, 1000)

            cv2.imshow(window_name, resized_img)
            # Exit on esc
            if cv2.waitKey(0) == 27:
                print()
                exit(0)
    print()

    cv2.destroyAllWindows()
