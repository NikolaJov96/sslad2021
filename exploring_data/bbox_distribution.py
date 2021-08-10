import cv2
import numpy as np

from structures.data_set import DataSet
from structures.image import Image


if __name__ == '__main__':
    """
    Visualize bbox image coverage per category
    """

    data_set = DataSet()
    data_set.load()

    # Initialize bbox count per pixel for each category
    bbox_distributions = {category_id: {
        'coverage': np.zeros((data_set.image_height, data_set.image_width)),
        'occurrences': 0
    } for category_id in data_set.categories}

    # Count bbox coverage
    for image_id in data_set.training_images:
        training_image: Image = data_set.training_images[image_id]
        for annotation_id in training_image.annotations:
            annotation = training_image.annotations[annotation_id]
            dist = bbox_distributions[annotation.category.category_id]
            p1 = (annotation.bbox[0], annotation.bbox[1])
            p2 = (annotation.bbox[0] + annotation.bbox[2], annotation.bbox[1] + annotation.bbox[3])
            dist['coverage'][p1[1]:p2[1], p1[0]:p2[0]] += 1
            dist['occurrences'] += 1

    # Occurrences per category
    for category_id in bbox_distributions:
        dist = bbox_distributions[category_id]
        print(category_id, data_set.categories[category_id].name, dist['occurrences'])

    # Visualize distributions
    window_name = 'Bbox distribution'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for category_id in bbox_distributions:
        dist = bbox_distributions[category_id]

        # Create the gray distribution image
        distribution_image = np.zeros((data_set.image_height, data_set.image_width, 1), np.uint8)
        distribution_image[:, :, 0] = dist['coverage'][:] * 255 / np.max(dist['coverage'])

        rgb_image = cv2.cvtColor(distribution_image, cv2.COLOR_GRAY2RGB)

        # Add the text category label
        cv2.putText(rgb_image,
                    data_set.categories[category_id].name,
                    (10, 50),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (0, 0, 255))

        # Show the image
        rgb_image = Image.resize_to_width(rgb_image, 1000)
        cv2.imshow(window_name, rgb_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
