import cv2
import sys

from structures.sslad_2d.definitions import SSLADDatasetTypes
from structures.sslad_2d.sslad_dataset import SSLADDataset
from structures.sslad_2d.image import Image


if __name__ == '__main__':
    """
    Displays all images with no annotations assigned
    """

    if sys.argv[1] not in ['training', 'validation']:
        print('usage: python3 display_no_annotations_images.py "training"/"validation"')
        exit(1)

    dataset_type = SSLADDatasetTypes.TRAINING if sys.argv[1] == 'training' else SSLADDatasetTypes.VALIDATION

    dataset = SSLADDataset()
    dataset.load(filter_no_annotations=False)

    whole_subset = dataset.get_subset(dataset_type)
    num_whole_subset = len(whole_subset)
    filtered_images = [image for image in whole_subset if len(image.annotations) == 0]
    num_filtered_images = len(filtered_images)
    print('{} images with no annotations: {}/{}'.format(sys.argv[1], num_filtered_images, num_whole_subset))

    window_name = 'Images with no annotations'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for i, image in enumerate(filtered_images):

        print('\rimage {}/{}'.format(i + 1, num_filtered_images), end='')

        resized_img = Image.resize_to_width(image.get_cv2_img(), 1000)

        cv2.imshow(window_name, resized_img)
        # Exit on esc
        if cv2.waitKey(0) == 27:
            break
    print()

    cv2.destroyAllWindows()
