import numpy as np
import os
import pathlib
import time

import torch

from competition_models.coco.coco_eval import CocoEvaluator
from competition_models.coco.coco_utils import convert_to_coco_api
from competition_models.dataset_wrapper import DatasetWrapper
from structures.sslad_2d.sslad_dataset import SSLADDataset


class Evaluator:
    """
    Class for evaluating a model agains the provided validation data
    """

    def __init__(self, validation_set_wrapper):
        """
        Initialize the evaluator by providing dataset wrapper containing validation images
        """

        self.validation_set_wrapper = validation_set_wrapper

        # Prepare data loader
        self.batch_size = 10
        self.total_batches = len(self.validation_set_wrapper) // self.batch_size
        self.validation_data_loader = torch.utils.data.DataLoader(
            self.validation_set_wrapper,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=DatasetWrapper.collate_fn
        )

        # Execute the time-consuming dataset conversion
        self.coco = convert_to_coco_api(self.validation_data_loader.dataset)

        # Per category IoUs used in the competition
        self.perCatIoUs = {
            # Pedestrian
            1: 0.5,
            # Cyclist
            2: 0.5,
            # Car
            3: 0.7,
            # Truck
            4: 0.7,
            # Tram
            5: 0.7,
            # Tricycle
            6: 0.5
        }

    @torch.no_grad()
    def evaluate(self, model, device, verbose=False):
        """
        Execute the evaluation and return the APs and mAP as used in the competition
        """

        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device('cpu')

        # Put model into evaluation mode
        model.eval()

        # Initialize the coco evaluator
        iou_types = ['bbox']
        coco_evaluator = CocoEvaluator(self.coco, iou_types)

        for batch, (images, targets) in enumerate(self.validation_data_loader):
            if verbose and batch % (self.total_batches // 100 + 1) == 0:
                print('batch {} / {}\r'.format(batch, self.total_batches), end='')

            # Get predictions from images
            images = list(img.to(device) for img in images)
            outputs = model(images)

            # Convert predictions to coco evaluator structure
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)
        if verbose:
            print()

        coco_evaluator.synchronize_between_processes()

        # Accumulate predictions from all images and batches
        coco_evaluator.accumulate()
        torch.set_num_threads(n_threads)

        """
        Coco evaluator content:
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        counts      = T,R,K,A,M
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))
        """

        # Calculate competition-specific metrics

        # Get the parematers object descripting the structure of the 5D precision array
        parameters = coco_evaluator.coco_eval['bbox'].eval['params']
        # Get the counts array representing the dimensions of the precions parray
        counts = coco_evaluator.coco_eval['bbox'].eval['counts']
        # Get the precision array
        precision = coco_evaluator.coco_eval['bbox'].eval['precision']

        # Get the id of 'all' area class
        a = parameters.areaRngLbl.index('all')

        # Get the id for maximum detections equals 100
        m = parameters.maxDets.index(100)

        # Evaluation results per category
        results = [-1.0] * 7

        # For each encountered category
        for k_i in range(counts[2]):

            # Get the category id
            k = parameters.catIds[k_i]

            # Get the category specific IoU
            t = np.where(self.perCatIoUs[k] == parameters.iouThrs)[0][0]

            # Get prediction for
            # - specific IoU,
            # - specific category
            # - bbox size class 'all'
            # - maximum number of detections 100
            # for all 101 recall recall values
            per_recall_precision = precision[t, :, k_i, a, m]

            # Get the mean for all recall values
            ap = np.mean(per_recall_precision[per_recall_precision > -1])

            results[k] = ap

        # Calculate the mAP if all categories have valid APs
        if all(ap > -1 for ap in results[1:]):
            results[0] = sum(results[1:]) / 6

        return results, coco_evaluator


def main():
    """
    Try out the evaluator on the default model saved after running faster_rcnn.py
    and the default validation set
    """

    from competition_models.faster_rcnn import FasterRCNN

    # Load all data descriptors
    dataset = SSLADDataset()
    dataset.load()

    # Wrap the data to be used for validation
    validation_set_length = 100
    validation_set_wrapper = DatasetWrapper(
        images=dataset.validation_images[:validation_set_length]
    )
    print('validation set size: {}'.format(len(validation_set_wrapper)))

    # Prepare the Evaluator object
    evaluator = Evaluator(validation_set_wrapper)

    # Create the model
    model = FasterRCNN()

    # Evaluate untrained model
    evaluate_untrained = False
    if evaluate_untrained:
        print('Untrained model results')
        print(evaluator.evaluate(model.model, model.device)[0])
        print()

    # Model save file
    save_file = os.path.join(pathlib.Path(__file__).parent.resolve(), 'model_save', 'torch_model.pt')

    # Try to load saved model
    if not model.load_model(save_file):
        print('saved model not found')
        exit(1)

    # Evaluate trained model
    start_time = time.time()
    results, coco_evaluator = evaluator.evaluate(model.model, model.device, verbose=True)
    end_time = time.time()
    print('Trained model results')
    print(results)
    print()

    # Check evaluation time
    print('evaluation time')
    print(end_time - start_time)
    print()

    parameters = coco_evaluator.coco_eval['bbox'].eval['params']
    counts = coco_evaluator.coco_eval['bbox'].eval['counts']
    precision = coco_evaluator.coco_eval['bbox'].eval['precision']

    print('count of values for T (iouThrs), R (recThrs), K (catIds), A (areaRng), M (maxDets)')
    print(counts)
    print()

    print('iouThrs - IoU thresholds')
    print(parameters.iouThrs)
    print()

    print('recThrs - recall thresholds')
    print(parameters.recThrs[:5])
    print()

    print('encountered categotry ids')
    print(parameters.catIds)
    print()

    print('bbox area class ranges')
    print(parameters.areaRng)
    print()

    print('maximum detections cap')
    print(parameters.maxDets)
    print()

    a_ind = [i for i, aRng in enumerate(parameters.areaRngLbl) if aRng == 'all']
    print('"all" area class id')
    print(a_ind)
    print()

    m_ind = [i for i, mDet in enumerate(parameters.maxDets) if mDet == 100]
    print('100 maximum detections id')
    print(m_ind)
    print()

    t = np.where(0.7 == parameters.iouThrs)[0]
    print('0.7 IoU threshold id')
    print(t)
    print()

    print('encountered category ids')
    print(parameters.catIds)
    print()

    print('AP per category')
    for k_i in range(counts[2]):
        k = parameters.catIds[k_i]
        s = precision[t, :, k - 1, a_ind, m_ind]
        mean_s = np.mean(s[s > -1])
        print(dataset.categories[k].name, mean_s)


if __name__ == '__main__':
    main()
