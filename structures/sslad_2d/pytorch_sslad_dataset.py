import torch
from torchvision.transforms import ToTensor

from structures.sslad_2d.sslad_dataset import SSLADDataset


class PyTorchSSLADDataset():
    """
    Class wrapping the SSLAD-2D dataset loader for use with PyTorch models
    """

    def __init__(self, train):
        """
        Initialize the underlining dataset
        """

        self.dataset = SSLADDataset()
        self.dataset.load()

        self.train = train
        self.images = []
        if self.train:
            self.images = self.dataset.training_images
        else:
            self.images = self.dataset.validation_images

    def __len__(self):
        """
        Return the len of selected part of the dataset
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
                PyTorchSSLADDataset.sslad_to_pytorch(annotation.get_array())
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
