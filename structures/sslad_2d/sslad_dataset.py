import json
import os

from structures.sslad_2d.annotation import Annotation
from structures.sslad_2d.category import Category
from structures.sslad_2d.definitions import SSLADDatasetTypes
from structures.sslad_2d.image import Image

"""
Annotation file structure:

"images": {
        "file_name": <str> -- File name.
        "id": <int>  -- Unique image id.
        "height": <float>  -- Height of the image.
        "width": <float> -- Width of the image.
        "city": <str> -- City tag.
        "location": <str> -- Location tag.
        "period": <str> -- Period tag.
        "weather": <str> -- Weather tag.}

"annotations": {
        "image_id": <int> -- The image id for this annotation.
        "category_id": <int> -- The category id.
        "bbox": <list> -- Coordinate of boundingbox [x, y, w, h].
        "area": <float> -- Area of this annotation (w * h).
        "id": <int> -- Unique annotation id.
        "iscrowd": <int> -- Whether this annotation is crowd. Note that iscrowd is always 0.}

"categories": {
        "name": <str> -- Unique category name.
        "id": <int> Unique category id.
        "supercategory": <str> The supercategory for this category.}
"""


class SSLADDataset:
    """
    Class wrapping the whole dataset
    """

    LABELED_DATA_PATH = os.path.join('data', 'SSLAD-2D', 'labeled')
    ANNOTATIONS_PATH = os.path.join(LABELED_DATA_PATH, 'annotations')
    TRAINING_IMAGES_PATH = os.path.join(LABELED_DATA_PATH, 'train')
    VALIDATION_IMAGES_PATH = os.path.join(LABELED_DATA_PATH, 'val')
    TESTING_IMAGES_PATH = os.path.join(LABELED_DATA_PATH, 'test')

    def __init__(self):
        """
        Prepare data storage members
        """

        self.categories = {}
        self.training_images = []
        self.validation_images = []
        self.testing_images = []
        self.image_width = 0
        self.image_height = 0

    def load(self):
        """
        Load dataset json descriptor files and store their data in structured from
        """

        with open(os.path.join(SSLADDataset.ANNOTATIONS_PATH, 'instance_train.json')) as in_file:
            training_data = json.load(in_file)
        with open(os.path.join(SSLADDataset.ANNOTATIONS_PATH, 'instance_val.json')) as in_file:
            validation_data = json.load(in_file)
        with open(os.path.join(SSLADDataset.ANNOTATIONS_PATH, 'instance_test.json')) as in_file:
            testing_data = json.load(in_file)

        # Initialize categories
        for category in training_data['categories']:
            self.categories[category['id']] = Category(category['name'], category['id'])

        # Define temporary maps
        training_images_map = {}
        validation_images_map = {}
        testing_images_map = {}

        # Initialize training images
        for image in training_data['images']:
            training_images_map[image['id']] = Image(
                SSLADDataset.TRAINING_IMAGES_PATH, image, SSLADDatasetTypes.TRAINING)

        # Initialize training annotations
        for annotation in training_data['annotations']:
            training_images_map[annotation['image_id']].add_annotation(Annotation(
                self.categories[annotation['category_id']], annotation))

        # Initialize validation images
        for image in validation_data['images']:
            validation_images_map[image['id']] = Image(
                SSLADDataset.VALIDATION_IMAGES_PATH, image, SSLADDatasetTypes.VALIDATION)

        # Initialize validation annotations
        for annotation in validation_data['annotations']:
            validation_images_map[annotation['image_id']].add_annotation(Annotation(
                self.categories[annotation['category_id']], annotation))

        # Initialize testing images
        for image in testing_data['images']:
            testing_images_map[image['id']] = Image(
                SSLADDataset.TESTING_IMAGES_PATH, image, SSLADDatasetTypes.TESTING)

        # Initialize testing annotations
        for annotation in testing_data['annotations']:
            testing_images_map[annotation['image_id']].add_annotation(Annotation(
                self.categories[annotation['category_id']], annotation))

        # Move values from temporary maps to lists
        self.training_images = list(training_images_map.values())
        self.validation_images = list(validation_images_map.values())
        self.testing_images = list(testing_images_map.values())

        # Set image width and height parameters
        image = self.training_images[0]
        self.image_width = image.width
        self.image_height = image.height


if __name__ == '__main__':
    """
    Run quick tests to check if dataset is being loaded correctly
    """

    data_set = SSLADDataset()
    data_set.load()

    print('Categories')
    categories = [
        (category_id, data_set.categories[category_id].name, data_set.categories[category_id].get_color())
        for category_id in data_set.categories]
    print(categories)

    training_annotation_counts = [len(image.annotations) for image in data_set.training_images]
    validation_annotation_counts = [len(image.annotations) for image in data_set.validation_images]
    testing_annotation_counts = [len(image.annotations) for image in data_set.testing_images]
    print('Training images: {}'.format(len(data_set.training_images)))
    print('Average annotations per image: {}'.format(sum(training_annotation_counts) / len(training_annotation_counts)))
    print('Validation images: {}'.format(len(data_set.validation_images)))
    print('Average annotations per image: {}'.format(
        sum(validation_annotation_counts) / len(validation_annotation_counts)))
    print('Testing images: {}'.format(len(data_set.testing_images)))
    print('Average annotations per image: {}'.format(sum(testing_annotation_counts) / len(testing_annotation_counts)))

    assert len(categories) > 0
    assert len(data_set.training_images) > 0
    assert len(training_annotation_counts) > 0
    assert len(data_set.validation_images) > 0
    assert len(validation_annotation_counts) > 0
    assert len(data_set.testing_images) > 0
    assert sum(testing_annotation_counts) == 0

    print('Tests passed')
