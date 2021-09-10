from abc import abstractproperty
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

Prediction output file structure:

{
        "image_id": <int> -- The image id for this annotation.
        "category_id": <int> -- The category id.
        "bbox": <list> -- Coordinate of boundingbox [x, y, w, h].
        "score": <float> -- Prediction confidence
}
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

    def load(
        self,
        validation_data_file=None,
        test_data_file=None,
        unlabeled_data_file=None,
        min_annotation_score=0.0,
        filter_no_annotations=True):
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
        unlabeled_images_map = {}

        # Initialize training images
        for image in training_data['images']:
            training_images_map[image['id']] = Image(
                SSLADDataset.TRAINING_IMAGES_PATH, image, SSLADDatasetTypes.TRAINING
            )

        # Initialize training annotations
        for annotation in training_data['annotations']:
            training_images_map[annotation['image_id']].add_annotation(Annotation.from_annotation_data(
                self.categories[annotation['category_id']], annotation
            ))

        # Initialize validation images
        for image in validation_data['images']:
            validation_images_map[image['id']] = Image(
                SSLADDataset.VALIDATION_IMAGES_PATH, image, SSLADDatasetTypes.VALIDATION
            )

        # Initialize validation annotations
        if validation_data_file is None:
            # Keep original validation image annotations
            for annotation in validation_data['annotations']:
                validation_images_map[annotation['image_id']].add_annotation(Annotation.from_annotation_data(
                    self.categories[annotation['category_id']], annotation
                ))
        else:
            # Load validation images annotations from the provided file
            self.load_predictions(validation_images_map, validation_data_file, min_annotation_score)

        # Initialize testing images
        for image in testing_data['images']:
            testing_images_map[image['id']] = Image(
                SSLADDataset.TESTING_IMAGES_PATH, image, SSLADDatasetTypes.TESTING
            )

        # Load testing images annotations if load file provided
        if test_data_file is not None:
            self.load_predictions(testing_images_map, test_data_file, min_annotation_score)

        # Initialize unlabeled images
        # Try to load all 10 unlabeled image batches
        for b in range(10):
            descriptor_file = SSLADDataset.UNLABELED_DESCRIPTOR_FILE.format(b)
            if os.path.exists(descriptor_file):
                with open(descriptor_file) as in_file:
                    unlabeled_data = json.load(in_file)
                for image in unlabeled_data:
                    unlabeled_images_map[image['id']] = Image(
                        SSLADDataset.DATA_PATH, image, SSLADDatasetTypes.UNLABELED
                    )

        # Load unlabeled images annotations if load file provided
        if unlabeled_data_file is not None:
            self.load_predictions(unlabeled_images_map, unlabeled_data_file, min_annotation_score)

        # Move values from temporary maps to lists
        # Skip any images with no annotations in training and validation sets
        self.training_images = [
            image for image in training_images_map.values() if not filter_no_annotations or len(image.annotations) > 0
        ]
        self.validation_images = [
            image for image in validation_images_map.values() if not filter_no_annotations or len(image.annotations) > 0
        ]
        self.testing_images = list(testing_images_map.values())
        self.unlabeled_images = list(unlabeled_images_map.values())

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

    def save_validation_predictions(self, validation_data_file):
        """
        Saves annotations assigned to validation images,
        assuming original annotations were manually cleared
        """

        SSLADDataset.save_predictions(self.validation_images, validation_data_file)

    def save_test_predictions(self, test_data_file):
        """
        Saves annotations assigned to testing images
        """

        SSLADDataset.save_predictions(self.testing_images, test_data_file)

    def save_unlabeled_predictions(self, unlabeled_data_file):
        """
        Saves annotations assigned to unlabeled images
        """

        SSLADDataset.save_predictions(self.unlabeled_images, unlabeled_data_file)

    @staticmethod
    def save_predictions(images, data_file):
        """
        Saves annotations assigned to received images
        """

        save_data = []
        for image in images:
            if image.is_annotated:
                for annotation in image.annotations:
                    save_data.append({
                        'image_id': image.image_id,
                        'category_id': annotation.category.category_id,
                        'bbox': annotation.get_array(),
                        'score': annotation.score
                    })

        with open(data_file, 'w') as out_file:
            json.dump(save_data, out_file)

    def load_predictions(self, images_map, data_file, min_annotation_score):
        """
        Loads saved predicted annotations for testing or unlabeled subsets
        """

        if data_file is not None:
            # Load predictions file
            with open(data_file, 'r') as in_file:
                testing_prediction_data = json.load(in_file)
            # Add annotations to images
            for annotation in testing_prediction_data:
                if annotation['score'] >= min_annotation_score:
                    images_map[annotation['image_id']].add_annotation(Annotation(
                        self.categories[annotation['category_id']],
                        annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][2], annotation['bbox'][3],
                        annotation['score']
                    ))


if __name__ == '__main__':
    """
    Run quick tests to check if dataset is being loaded correctly
    """

    dataset = SSLADDataset()
    dataset.load()

    print('categories')
    categories = [
        (category_id, dataset.categories[category_id].name, dataset.categories[category_id].get_color())
        for category_id in dataset.categories
    ]
    print(categories)

    training_annotation_counts = [len(image.annotations) for image in dataset.training_images]
    validation_annotation_counts = [len(image.annotations) for image in dataset.validation_images]
    testing_annotation_counts = [len(image.annotations) for image in dataset.testing_images]
    print('training images: {}'.format(len(dataset.training_images)))
    print('average annotations per image: {}'.format(sum(training_annotation_counts) / len(training_annotation_counts)))
    print('validation images: {}'.format(len(dataset.validation_images)))
    print('average annotations per image: {}'.format(
        sum(validation_annotation_counts) / len(validation_annotation_counts)))
    print('testing images: {}'.format(len(dataset.testing_images)))
    print('unlabeled images: {}'.format(len(dataset.unlabeled_images)))

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

    # Test prediction saving and loading
    import pathlib

    # Add one annotation
    dataset.testing_images[0].add_annotation(Annotation(
        dataset.categories[1], 10, 20, 30, 40, 0.1
    ))

    # Save annotations
    save_folder = os.path.join(pathlib.Path(__file__).parent.resolve(), 'test_saves')
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    test_data_file = os.path.join(save_folder, 'test_predictions.json')
    dataset.save_test_predictions(test_data_file)

    # Same procedure for unlabeled data if it exists
    unlabeled_data_file = None
    unlabeled_data_exists = len(dataset.unlabeled_images) > 0
    if unlabeled_data_exists:
        dataset.unlabeled_images[1].add_annotation(Annotation(
            dataset.categories[2], 100, 200, 300, 400, 0.9
        ))
        unlabeled_data_file = os.path.join(save_folder, 'unlabeled_predictions.json')
        dataset.save_unlabeled_predictions(unlabeled_data_file)

    # Reload saved annotations
    dataset = SSLADDataset()
    dataset.load(test_data_file=test_data_file, unlabeled_data_file=unlabeled_data_file)

    assert len(dataset.testing_images[0].annotations) == 1
    print('saved test annotation')
    print(dataset.testing_images[0].annotations[0])

    if unlabeled_data_exists:
        assert len(dataset.unlabeled_images[1].annotations) == 1
        print('saved unlabeled annotation')
        print(dataset.unlabeled_images[1].annotations[0])

    print('tests passed')
