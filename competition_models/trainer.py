import copy
import json
import os
import pathlib
import random
import time

from competition_models.augmentations import AddNoise, FlipImage, ScaleBrightness
from competition_models.dataset_wrapper import DatasetWrapper
from competition_models.evaluator import Evaluator
from competition_models.faster_rcnn import FasterRCNN
from structures.sslad_2d.definitions import SSLADDatasetTypes
from structures.sslad_2d.sslad_dataset import SSLADDataset

"""
Algorithm:
- Train the initial model_0 using the training set

- Predict annotations for unlabeled data batches 0, 1 and 2
- Save dataset state as annotations_1
- Train model_0 with unlabeled batch 0, then with training set to get model_1_0
- Train model_0 with unlabeled batch 1, then with training set to get model_1_1
- Train model_0 with unlabeled batch 2, then with training set to get model_1_2
- Evaluate all 3 on the validation set
- The best performing model, corresponding to the batch x becomes model_1

- Predict annotations for unlabeled data batches 3, 4 and 5
- Save dataset state as annotations_2
- Train model_1 with batch x, then batch 3, then with training set to get model_2_0
- Train model_1 with batch x, then batch 4, then with training set to get model_2_1
- Train model_1 with batch x, then batch 5, then with training set to get model_2_2
- Evaluate all 3 on the validation set
- The best performing model, corresponding to the batch y becomes model_2

Log file content:
{
    "unlabeled_base": <list> -- Unlabeled batch ids used for all training
    "unlabeled_tested":
    [{
        "unlabeled_id": <int> -- Id of additional unlabeled training batch
        "training_time": <float> -- Training time in seconds
        "evaluation_time": <float> -- Evaluation time in seconds
        "evaluation": <list> -- Evaluator output
    }]
}
"""


class Trainer:
    """
    Class responsible for running the algorithm explained above
    """

    LOG_SUBDIR = 'log'
    MODELS_SUBDIR = 'models'
    ANNOTATIONS_SUBDIR = 'annotations'

    UNLABELED_BATCH = 3000
    BATCHES_IN_ITERATION = 2
    ORIGINAL_TRAINING_SET = 5000
    LIMIT_EVALUATION_TO_100 = False
    MIN_ANNOTATION_SCORE = 0.5
    USE_VALIDATION_FOR_TRAINING = True

    def __init__(self, session_id):
        """
        Prepare working direcotries for the desired session id
        Load current progress if any exists
        """

        self.session_id = session_id

        # Prepare work directories

        self.work_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), 'session_{}'.format(self.session_id))
        self.model_dir = os.path.join(self.work_dir, Trainer.MODELS_SUBDIR)
        self.annotation_dir = os.path.join(self.work_dir, Trainer.ANNOTATIONS_SUBDIR)
        self.log_dir = os.path.join(self.work_dir, Trainer.LOG_SUBDIR)

        dirs = [self.work_dir, self.annotation_dir, self.model_dir, self.log_dir]
        for directory in dirs:
            if not os.path.isdir(directory):
                os.mkdir(directory)

        # Prepare evaluator member, leave it uninitialized until the first training run
        self.evaluator = None

        # Load current session progress
        self.initial_log = {}
        if os.path.exists(self.initial_log_path()):
            with open(self.initial_log_path(), 'r') as in_file:
                self.initial_log = json.load(in_file)

        num_log_files = len(list(filter(lambda d: d.startswith('log'), os.listdir(self.log_dir))))
        log_files = [self.log_path(i + 1) for i in range(num_log_files)]
        self.iteration_logs = []
        for log_file in log_files:
            with open(os.path.join(self.log_dir, log_file), 'r') as in_file:
                self.iteration_logs.append(json.load(in_file))

    def get_current_iteration(self):
        """
        Returns the current training iteration
        """

        return len(self.iteration_logs) + 1

    def get_evaluator(self):
        """
        Initializes the evaluator if not yet initialized and returns it
        """

        if self.evaluator is None:
            dataset = SSLADDataset()
            # Evaluator unfortunately fails if it receives images with no annotations,
            # using only images with annotations for validation should give close enough estimate
            dataset.load(filter_no_annotations=True)
            validation_data = dataset.get_subset(SSLADDatasetTypes.VALIDATION)
            if Trainer.LIMIT_EVALUATION_TO_100:
                validation_data = validation_data[:100]
            validation_wrapper = DatasetWrapper(validation_data)
            self.evaluator = Evaluator(validation_wrapper)

        return self.evaluator

    def train_iteration(self):
        """
        Main iteration training method, calls submethods as needed
        """

        if os.path.exists(self.initial_log_path()):
            # Train standard iteration, depending on previously trained models
            self.train_standard_iteration()
        else:
            # Absence of initial log implies initial model training is not done
            self.train_initial_model()

    def train_initial_model(self):
        """
        Method that executes the initial model training using the original training set only
        """

        print('training initial model')

        # Prepare initial competition training set
        dataset = SSLADDataset()
        dataset.load()
        dataset_wrapper = DatasetWrapper(images=dataset.training_images[:Trainer.ORIGINAL_TRAINING_SET])

        model = FasterRCNN()

        # Train the model pretrained on ImageNet
        print('training initial model')
        train_start = time.time()
        model.train(dataset=dataset_wrapper, num_epochs=10)
        train_end = time.time()

        # Save the trained model
        model.save_model(self.model_path(0))

        # Evaluate the initial model for later reference
        print('evaluating initial model')
        evaluation_start = time.time()
        result, _ = self.get_evaluator().evaluate(model, verbose=True)
        evaluation_end = time.time()

        # Save the training log
        self.initial_log = {
            "training_time": train_end - train_start,
            "evaluation_time": evaluation_end - evaluation_start,
            "evaluation": result
        }
        with open(self.initial_log_path(), 'w') as out_file:
            json.dump(self.initial_log, out_file)

    def train_standard_iteration(self):
        """
        Method that executes every training iteration but the initial one
        Uses the bes previous model to predict labels for unlabeled images
        and uses them to improve future training
        """

        iteration = self.get_current_iteration()

        print('\nrunning iteration {}'.format(iteration))

        # Get the dataset from the last iteration with additional annotations
        # predictied using the model saved by the previous iteration
        dataset = self.execute_iteration_prediction(iteration)

        base_unlabeled_batches = self.get_base_unlabeled_training_data(iteration)
        print('previous unlabeled batches used for training', base_unlabeled_batches)

        new_unlabeled_batches = [
            (iteration - 1) * Trainer.BATCHES_IN_ITERATION + i for i in range(Trainer.BATCHES_IN_ITERATION)
        ]
        print('new unlabeled batch ids', new_unlabeled_batches)

        log = {
            "unlabeled_base": base_unlabeled_batches,
            "unlabeled_tested": []
        }

        for i, new_unlabeled_batch in enumerate(new_unlabeled_batches):

            # Load latest model
            model = self.load_best_model(iteration - 1)

            # Prepare images of the current batch of unlabeled data
            # Optional: Revisit some of the unlabeled data batches already used for training
            unlabeled_images = dataset.unlabeled_images[
                new_unlabeled_batch * Trainer.UNLABELED_BATCH: (new_unlabeled_batch + 1) * Trainer.UNLABELED_BATCH
            ]
            unlabeled_images = [image for image in unlabeled_images if len(image.annotations) > 0]
            print('unlabeled images after filtering', len(unlabeled_images))
            unlabeled_images += Trainer.copy_and_augment_tricycles(unlabeled_images)
            print('unlabeled images after augmenting tricycles', len(unlabeled_images))

            # Prepare images of the original training data
            training_images = Trainer.copy_and_add_augmentations(
                dataset.training_images[:Trainer.ORIGINAL_TRAINING_SET]
            )

            # Prepare images of the original validation data
            if Trainer.USE_VALIDATION_FOR_TRAINING:
                training_images += Trainer.copy_and_add_augmentations(dataset.validation_images)

            # Create combined unlabeled and training dataset wrapper
            dataset_wrapper = DatasetWrapper(images=unlabeled_images + training_images)

            # Train the model saved by the last iteration
            print('iteration {}, batch {}, training'.format(iteration, i))
            train_start = time.time()
            model.train(dataset=dataset_wrapper, num_epochs=5)
            train_end = time.time()

            # Save the trained model
            model.save_model(self.model_path(iteration, sub_model=i))

            # Evaluate the model for comparison with other models in this iteration
            print('iteration {}, batch {}, evaluation'.format(iteration, i))
            # Get and potentially initialize the evaluator before starting to count the time
            evaluator = self.get_evaluator()
            evaluation_start = time.time()
            result, _ = evaluator.evaluate(model, verbose=True)
            evaluation_end = time.time()

            # Add results of this batch to the iteration log
            log["unlabeled_tested"].append({
                "unlabeled_id": (iteration - 1) * Trainer.BATCHES_IN_ITERATION + i,
                "training_time": train_end - train_start,
                "evaluation_time": evaluation_end - evaluation_start,
                "evaluation": result
            })

        # Save the iteration log
        self.iteration_logs.append(log)
        with open(self.log_path(iteration), 'w') as out_file:
            json.dump(log, out_file)


    def execute_iteration_prediction(self, iteration):
        """
        Executes predictions on a dedicated part of the unlabeled data
        and stores it for future reference
        """

        print('predicting iteration {}'.format(iteration))

        dataset = SSLADDataset()

        # Check if current iteration has saved prediction data before by trying to load it
        try:
            dataset.load(
                unlabeled_data_file=self.annotation_path(iteration),
                min_annotation_score=Trainer.MIN_ANNOTATION_SCORE
            )
            print('found saved annotations file for this iteration')
        except FileNotFoundError:
            # No file found
            if iteration == 1:
                # For the first iteration there is no history data
                dataset.load()
            else:
                # For later iterations, load previous iteration annotations
                dataset.load(
                    unlabeled_data_file=self.annotation_path(iteration - 1),
                    min_annotation_score=Trainer.MIN_ANNOTATION_SCORE
                )

            for unlabeled_batch in range(Trainer.BATCHES_IN_ITERATION):

                # Load latest model
                model = self.load_best_model(iteration - 1)

                unlabeled_image_ids = [
                    (iteration - 1) * Trainer.BATCHES_IN_ITERATION * Trainer.UNLABELED_BATCH +
                        unlabeled_batch * Trainer.UNLABELED_BATCH + i
                    for i in range(Trainer.UNLABELED_BATCH)
                ]

                for id, unlabeled_image_id in enumerate(unlabeled_image_ids):
                    if (id * 100) % Trainer.UNLABELED_BATCH == 0:
                        print('\rprediction unlabeled batch {}, image {}/{}'.format(
                                unlabeled_batch, id, Trainer.UNLABELED_BATCH
                            ),
                            end=''
                        )
                    predictions = model.eval([dataset.unlabeled_images[unlabeled_image_id].get_pil_img()])
                    annotations = DatasetWrapper.prediction_to_annotations(dataset, predictions)[0]
                    dataset.unlabeled_images[unlabeled_image_id].add_annotations(annotations)
                print()

            dataset.save_unlabeled_predictions(self.annotation_path(iteration))

        return dataset

    def get_base_unlabeled_training_data(self, iteration):
        """
        Returns all annotated unlabeled data subset ids previously used in training
        (used for logging purposes)
        """

        if iteration == 1:
            return []
        else:
            previous_iteration_log = self.iteration_logs[iteration - 2]
            base_unlabeled_training_data = previous_iteration_log['unlabeled_base']
            best_sub_model_id = self.get_best_sub_model_id(iteration - 1)
            base_unlabeled_training_data.append(
                previous_iteration_log['unlabeled_tested'][best_sub_model_id]['unlabeled_id']
            )
            return base_unlabeled_training_data

    def load_best_model(self, iteration):
        """
        Loads and returns one of models from a previous iteration with a best validation score
        """

        model = FasterRCNN()

        if iteration == 0:
            # Load initial model
            model.load_model(self.model_path(0))
        else:
            # Load the best model from the last iteration
            best_sub_model_id = self.get_best_sub_model_id(iteration)
            model_path = self.model_path(iteration, best_sub_model_id)
            print('iteration {} best model id: {} at {}'.format(iteration, best_sub_model_id, model_path))
            model.load_model(model_path)

        return model

    def get_best_sub_model_id(self, iteration):
        """
        Returns the id of the model with the best validation score from a previous iteration
        """

        assert iteration > 0

        latest_log = self.iteration_logs[iteration - 1]
        results = [x['evaluation'][0] for x in latest_log['unlabeled_tested']]
        print('iteration {} results'.format(iteration), results)
        return results.index(max(results))

    def initial_log_path(self):
        """
        Returns the path to the log file of the 0-th iteration
        """

        return os.path.join(self.log_dir, 'initial_log.json')

    def log_path(self, iteration):
        """
        Returns the path to the log file of the specified iteration
        """

        return os.path.join(self.log_dir, 'log_{}.json'.format(iteration))

    def model_path(self, iteration, sub_model=None):
        """
        Returns the path to the model file of the specified model of the specified iteration
        """

        if sub_model is None:
            return os.path.join(self.model_dir, 'model_{}.pt'.format(iteration))
        else:
            return os.path.join(self.model_dir, 'model_{}_{}.pt'.format(iteration, sub_model))

    def annotation_path(self, iteration):
        """
        Returns the path to the annotations file generated by the specified iteration
        """

        return os.path.join(self.annotation_dir, 'annotation_{}.json'.format(iteration))

    def output_prediction_path(self, prediction_set):
        """
        Returns the path to the validation or testing predictions output file
        """

        return os.path.join(self.annotation_dir, '{}.json'.format(prediction_set))

    @staticmethod
    def copy_and_add_augmentations(images):
        """
        Takes a list of image objects, copies it and adds augmentations
        """

        new_images = [copy.deepcopy(image) for image in images]

        # Flip some images
        num_images = int(len(new_images) * 0.5)
        for image in random.sample(new_images, num_images):
            image.add_augmentation(FlipImage())

        # Scale brightness of some images
        num_images = int(len(new_images) * 0.3)
        for image in random.sample(new_images, num_images):
            random_scale_factor = 0.2 + random.random() * 0.6
            image.add_augmentation(ScaleBrightness(random_scale_factor))

        # Add noise to some images
        num_images = int(len(new_images) * 0.3)
        for image in random.sample(new_images, num_images):
            random_scale = 0.5 + random.random() * 2.0
            image.add_augmentation(AddNoise(random_scale))

        return new_images

    @staticmethod
    def copy_and_augment_tricycles(images):
        """
        Takes a list of image objects, copies each image containing tricycles
        and augments them with all available augmentations
        """

        tricycle_category = 6
        new_images = [
            copy.deepcopy(image) for image in images
            if any([annotation.category.category_id == tricycle_category for annotation in image.annotations])
        ]
        num_new_images = len(new_images)
        print('Tricycle images: {}/{}'.format(num_new_images, len(images)))
        new_images = new_images * 3

        for i in range(num_new_images):
            # Flip the tricycle image
            new_images[i].add_augmentation(FlipImage())

            # Scale brightness of the tricycle image
            random_scale_factor = 0.1 + random.random() * 0.5
            new_images[i + num_new_images].add_augmentation(ScaleBrightness(random_scale_factor))

            # Add noise to the tricycle image
            random_scale = 0.5 + random.random() * 2.0
            new_images[i + num_new_images * 2].add_augmentation(AddNoise(random_scale))

        return new_images

    def generate_output_predictions(self, prediction_set):
        """
        Finds a best model from a previous iteration and generates output predictions
        for validation or tresting set
        """

        if prediction_set not in ['validation', 'testing']:
            print('invalid prediction_set: {}'.format(prediction_set))
            return

        print('predicting {}'.format(prediction_set))

        is_validation = prediction_set == 'validation'

        dataset = SSLADDataset()
        dataset.load(filter_no_annotations=False)
        images = dataset.validation_images if is_validation else dataset.testing_images
        num_images = len(images)
        total_annotations = 0

        # Load latest model
        model = self.load_best_model(self.get_current_iteration() - 1)

        for id, image in enumerate(images):

            if (id * 100) % num_images == 0:
                print('\rimage {}/{}'.format(id, num_images), end='')

            # Clear potential previous annotations
            image.annotations = []

            # Generate new annotations
            predictions = model.eval([image.get_pil_img()])
            annotations = DatasetWrapper.prediction_to_annotations(dataset, predictions)[0]
            annotations = [annotation for annotation in annotations if annotation.score > 0.5]
            total_annotations += len(annotations)
            image.add_annotations(annotations)
        print()

        print('average annotations per image {} (on the training set it is around 8)'.format(total_annotations / num_images))

        # Save generated predictions
        save_file = self.output_prediction_path(prediction_set)
        if is_validation:
            dataset.save_validation_predictions(save_file)
        else:
            dataset.save_test_predictions(save_file)


def main():
    """
    Runs training for the specified training session
    """

    import sys

    if len(sys.argv) != 2:
        print('usage: python3 trainer.py session_id')
    training_session = sys.argv[1]

    trainer = Trainer(training_session)

    for _ in range(1000):
        trainer.train_iteration()


if __name__ == '__main__':
    main()
