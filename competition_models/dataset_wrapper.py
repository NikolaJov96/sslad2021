import torch
from torchvision.transforms import ToTensor

from structures.sslad_2d.annotation import Annotation


class DatasetWrapper():
    """
    Wraps a list of Image objects with annotations for use with PyTorch models
    """

    def __init__(self, images):
        """
        Initialize the images list
        """

        self.images = images


    def __len__(self):
        """
        Return the length of the images list
        """

        return len(self.images)

    def __getitem__(self, idx):
        """
        Return the PyTorch compatible data sample with the given index
        """

        image_obj = self.images[idx]

        # Load image
        img = image_obj.get_pil_img()

        # Convert bboxes to the right format
        boxes = torch.as_tensor(
            [
                DatasetWrapper.sslad_to_pytorch(annotation.get_array())
                for annotation in image_obj.annotations
            ],
            dtype=torch.float32
        )

        # Array with a label for each bbox
        labels = [annotation.category.category_id for annotation in image_obj.annotations]

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Suppose all instances are not crowd
        target["iscrowd"] = torch.zeros((len(image_obj.annotations),), dtype=torch.int64)

        return ToTensor()(img), target

    @staticmethod
    def prediction_to_annotations(dataset, predictions):
        """
        Converts a list of PyTorch predictions to the list of Annotation objects
        Needs an access to a dataset instance to find the appropriate category object
        """

        annotations = []

        for prediction in predictions:

            bboxes = prediction['boxes'].tolist()
            labels = prediction['labels'].tolist()

            annotations.append([
                Annotation(
                    dataset.categories[labels[i]],
                    *DatasetWrapper.pytorch_to_sslad(bboxes[i])
                ) for i in range(len(bboxes))
            ])

        return annotations

    @staticmethod
    def sslad_to_pytorch(sslad_bbox):
        """
        Converts 4-element array bbox representation from sslad to pytorch
        """

        return [
            sslad_bbox[0],
            sslad_bbox[1],
            sslad_bbox[0] + sslad_bbox[2],
            sslad_bbox[1] + sslad_bbox[3]
        ]

    @staticmethod
    def pytorch_to_sslad(pytorch_bbox):
        """
        Converts 4-element array bbox representation from pytorch to sslad
        """

        return [
            pytorch_bbox[0],
            pytorch_bbox[1],
            pytorch_bbox[2] - pytorch_bbox[0],
            pytorch_bbox[3] - pytorch_bbox[1]
        ]
