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

    DATA_PATH = 'data'
    SSLAD_DATA_PATH = os.path.join(DATA_PATH, 'SSLAD-2D')

    LABELED_DATA_PATH = os.path.join(SSLAD_DATA_PATH, 'labeled')
    ANNOTATIONS_PATH = os.path.join(LABELED_DATA_PATH, 'annotations')
    TRAINING_ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_PATH, 'instance_train.json')
    VALIDATION_ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_PATH, 'instance_val.json')
    TESTING_ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_PATH, 'instance_test.json')
    TRAINING_IMAGES_PATH = os.path.join(LABELED_DATA_PATH, 'train')
    VALIDATION_IMAGES_PATH = os.path.join(LABELED_DATA_PATH, 'val')
    TESTING_IMAGES_PATH = os.path.join(LABELED_DATA_PATH, 'test')

    UNLABELED_DATA_PATH = os.path.join(SSLAD_DATA_PATH, 'unlabel')
    UNLABELED_DESCRIPTOR_FILE = os.path.join(UNLABELED_DATA_PATH, 'annotations', 'instance_unlabel_{}.json')

    def __init__(self):
        """
        Prepare data storage members
        """

        self.categories = {}
        self.training_images = []
        self.validation_images = []
        self.testing_images = []
        self.unlabeled_images = []
        self.image_width = 0
        self.image_height = 0

    def load(self):
        """
        Load dataset json descriptor files and store their data in structured from
        """

        with open(SSLADDataset.TRAINING_ANNOTATIONS_FILE) as in_file:
            training_data = json.load(in_file)
        with open(SSLADDataset.VALIDATION_ANNOTATIONS_FILE) as in_file:
            validation_data = json.load(in_file)
        with open(SSLADDataset.TESTING_ANNOTATIONS_FILE) as in_file:
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
            training_images_map[annotation['image_id']].add_annotation(Annotation.from_annotation_data(
                self.categories[annotation['category_id']], annotation))

        # Initialize validation images
        for image in validation_data['images']:
            validation_images_map[image['id']] = Image(
                SSLADDataset.VALIDATION_IMAGES_PATH, image, SSLADDatasetTypes.VALIDATION)

        # Initialize validation annotations
        for annotation in validation_data['annotations']:
            validation_images_map[annotation['image_id']].add_annotation(Annotation.from_annotation_data(
                self.categories[annotation['category_id']], annotation))

        # Move values from temporary maps to lists
        # Skip any images with no annotations in training and validation sets
        self.training_images = [image for image in training_images_map.values() if len(image.annotations) > 0]
        self.validation_images = [image for image in validation_images_map.values() if len(image.annotations) > 0]
        self.testing_images = list(testing_images_map.values())

        # Initialize testing images
        for image in testing_data['images']:
            self.testing_images.append(Image(
                SSLADDataset.TESTING_IMAGES_PATH, image, SSLADDatasetTypes.TESTING))

        # Initialize unlabeled images
        # Try to load all 10 unlabeled image batches
        for b in range(10):
            descriptor_file = SSLADDataset.UNLABELED_DESCRIPTOR_FILE.format(b)
            if os.path.exists(descriptor_file):
                with open(descriptor_file) as in_file:
                    unlabeled_data = json.load(in_file)
                for image in unlabeled_data:
                    self.unlabeled_images.append(Image(SSLADDataset.DATA_PATH, image, SSLADDatasetTypes.UNLABELED))

        # Set image width and height parameters
        image = self.training_images[0]
        self.image_width = image.width
        self.image_height = image.height

    def get_subset(self, dataset_type):
        """
        Returns a part of the dataset depending on the requested type
        """

        if dataset_type == SSLADDatasetTypes.TRAINING:
            return self.training_images
        elif dataset_type == SSLADDatasetTypes.VALIDATION:
            return self.validation_images
        elif dataset_type == SSLADDatasetTypes.TESTING:
            return self.testing_images
        elif dataset_type == SSLADDatasetTypes.UNLABELED:
            return self.unlabeled_images
        else:
            return []


if __name__ == '__main__':
    """
    Run quick tests to check if dataset is being loaded correctly
    """

    dataset = SSLADDataset()
    dataset.load()

    print('Categories')
    categories = [
        (category_id, dataset.categories[category_id].name, dataset.categories[category_id].get_color())
        for category_id in dataset.categories]
    print(categories)

    training_annotation_counts = [len(image.annotations) for image in dataset.training_images]
    validation_annotation_counts = [len(image.annotations) for image in dataset.validation_images]
    testing_annotation_counts = [len(image.annotations) for image in dataset.testing_images]
    print('Training images: {}'.format(len(dataset.training_images)))
    print('Average annotations per image: {}'.format(sum(training_annotation_counts) / len(training_annotation_counts)))
    print('Validation images: {}'.format(len(dataset.validation_images)))
    print('Average annotations per image: {}'.format(
        sum(validation_annotation_counts) / len(validation_annotation_counts)))
    print('Testing images: {}'.format(len(dataset.testing_images)))
    print('Unlabeled images: {}'.format(len(dataset.unlabeled_images)))

    assert len(categories) > 0
    assert len(dataset.training_images) > 0
    assert len(training_annotation_counts) > 0
    assert len(dataset.validation_images) > 0
    assert len(validation_annotation_counts) > 0
    assert len(dataset.testing_images) > 0
    assert sum(testing_annotation_counts) == 0

    # No images witout any annotations in training and validations sets
    assert len([image for image in dataset.training_images if len(image.annotations) == 0]) == 0
    assert len([image for image in dataset.validation_images if len(image.annotations) == 0]) == 0

    print('Tests passed')
