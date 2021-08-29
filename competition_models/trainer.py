import json
import os
import time

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
    {
        "unlabeled_id": <int> -- Id of additional unlabeled training batch
        "training_time": <float> -- Training time in seconds
        "evaluation_time": <float> -- Evaluation time in seconds 
        "evaluation": <list> -- Evaluator output
    }
}
"""


class Trainer:
    """
    """

    LOG_SUBDIR = 'log'
    MODELS_SUBDIR = 'models'
    ANNOTATIONS_SUBDIR = 'annotations'

    UNLABELED_BATCH = 3000
    BATCHES_IN_ITERATION = 3

    def __init__(self, work_dir):
        """
        """

        # Prepare work directories
        self.work_dir = work_dir
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
        log_files = filter(lambda d: d.startswith('log'), os.listdir(self.log_dir))
        self.iteration_logs = []
        for log_file in log_files:
            with open(log_file, 'r') as in_file:
                self.iteration_logs.append(json.load(in_file))

    def get_evaluator(self):
        """
        """

        if self.evaluator is None:
            validation_dataset = SSLADDataset()
            validation_dataset.load()
            validation_wrapper = DatasetWrapper(
                validation_dataset.get_subset(SSLADDatasetTypes.VALIDATION)
            )
            self.evaluator = Evaluator(validation_wrapper)

        return self.evaluator

    def train_iteration(self):
        """
        """

        if os.path.exists(self.initial_log_path()):
            self.train_standard_iteration()
        else:
            self.train_initial_model()

    def train_initial_model(self):
        """
        """

        dataset = SSLADDataset()
        dataset.load()

        model = FasterRCNN()

        dataset_wrapper = DatasetWrapper(images=dataset.training_images)

        train_start = time.time()
        model.train(dataset=dataset_wrapper, num_epochs=10)
        train_end = time.time()

        model.save_model(self.model_path(0))

        evaluation_start = time.time()
        result = self.evaluator.evaluate(model)
        evaluation_end = time.time()

        log = {
            "training_time": train_end - train_start,
            "evaluation_time": evaluation_end - evaluation_start,
            "result": result
        }
        with open(self.initial_log_path(), 'w') as out_file:
            json.dump(log, out_file)

    def train_standard_iteration(self):
        """
        """

        iteration = len(self.iteration_logs) + 1

        dataset = SSLADDataset()
        if iteration > 1:
            dataset.load(unlabeled_data_file=self.annotation_path(iteration - 1))
        else:
            dataset.load()

        for unlabeled_batch in range(Trainer.BATCHES_IN_ITERATION):

            # Load latest model
            model = self.load_best_model(iteration - 1)

            unlabeled_image_ids = [
                (iteration - 1) * Trainer.BATCHES_IN_ITERATION * Trainer.UNLABELED_BATCH + i
                for i in range(Trainer.BATCHES_IN_ITERATION)
            ]

            for unlabeled_image_id in unlabeled_image_ids:
                predictions = model.eval(dataset.unlabeled_images[unlabeled_image_id].get_pil_img())
                annotations = DatasetWrapper.prediction_to_annotations(dataset, predictions)[0]
                dataset.unlabeled_images[unlabeled_image_id].add_annotations(annotations)

        dataset.save_unlabeled_predictions(self.annotation_path(iteration))

        base_unlabeled_batches = self.get_base_unlabeled_training_data(iteration)

        new_unlabeled_batches = [
            iteration * Trainer.BATCHES_IN_ITERATION * Trainer.UNLABELED_BATCH + i
            for i in range(Trainer.BATCHES_IN_ITERATION)
        ]
        print('new unlabeled batch ids')
        print(new_unlabeled_batches)

        log = {
            "unlabeled_tested": [dict() for _ in range(Trainer.BATCHES_IN_ITERATION)]
        }

        for new_unlabeled_batch in new_unlabeled_batches:

            # Load latest model
            model = self.load_best_model(iteration - 1)


    def get_base_unlabeled_training_data(self, iteration):
        """
        """

        if iteration == 0:
            return []
        else:
            base_unlabeled_training_data = self.iteration_logs[iteration - 1]['unlabeled_base']
            best_sub_model_id = self.get_best_sub_model_id(iteration)
            base_unlabeled_training_data.append(
                self.iteration_logs[iteration - 1]['unlabeled_tested'][best_sub_model_id]['unlabeled_id']
            )
            return base_unlabeled_training_data

    def load_best_model(self, iteration):
        """
        """

        model = FasterRCNN()

        if iteration == 0:
            # Load initial model
            model.load_model(self.model_path(0))
        else:
            # Load the best model from the last iteration
            best_sub_model_id = self.get_best_sub_model_id(iteration)
            print('best model id: {}'.format(best_sub_model_id))

            model.load_model(self.model_path(iteration, best_sub_model_id))

        return model

    def get_best_sub_model_id(self, iteration):
        """
        """

        assert iteration > 0

        latest_log = self.iteration_logs[iteration - 1]
        results = [x['evaluation'][0] for x in latest_log['unlabeled_tested']]
        print('last iteration results')
        print(results)
        return results.index(max(results))


    def initial_log_path(self):
        """
        """

        return os.path.join(self.log_dir, 'initial_log.json')

    def log_path(self, iteration):
        """
        """

        return os.path.join(self.log_dir, 'log_{}.json'.format(iteration))

    def model_path(self, iteration, sub_model=None):
        """
        """

        if sub_model is None:
            return os.path.join(self.model_dir, 'model_{}.pt'.format(iteration))
        else:
            return os.path.join(self.model_dir, 'model_{}_{}.pt'.format(iteration, sub_model))

    def annotation_path(self, iteration):
        """
        """

        return os.path.join(self.model_dir, 'annotation_{}'.format(iteration))


def main():
    """
    """

    import pathlib
    import sys

    if len(sys.argv) != 2:
        print('usage: python3 trainer.py session_id')
    training_session = sys.argv[1]

    work_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), 'session_{}'.format(training_session))
    trainer = Trainer(work_dir)


if __name__ == '__main__':
    main()
