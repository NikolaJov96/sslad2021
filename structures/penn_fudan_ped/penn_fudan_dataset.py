# Revised sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import numpy as np
import os
import torch
from PIL import Image

import pytorchscripts.transforms as Transforms


class PennFudanDataset(object):
    """
    Class wrapping the PennFudan dataset
    """

    DATA_PATH = os.path.join('data', 'PennFudanPed')
    PNG_IMAGES_PATH = os.path.join(DATA_PATH, "PNGImages")
    PED_MASKS_PATH = os.path.join(DATA_PATH, "PedMasks")

    def __init__(self, train):
        """
        Load all image file paths, sorting them to ensure that they are aligned
        """

        self.transforms = PennFudanDataset.get_transform(train)

        self.image_paths = list(sorted(os.listdir(PennFudanDataset.PNG_IMAGES_PATH)))
        self.mask_paths = list(sorted(os.listdir(PennFudanDataset.PED_MASKS_PATH)))

    def __getitem__(self, idx):
        """
        Return pytorch compatible image-annotations pair given the index
        """

        # Load images and masks
        img = Image.open(self.get_image_path(idx)).convert("RGB")
        mask = np.array(Image.open(os.path.join(PennFudanDataset.PED_MASKS_PATH, self.mask_paths[idx])))

        # Instances are encoded as different colors
        obj_ids = np.unique(mask)
        # First id is the background, so remove it
        obj_ids = obj_ids[1:]

        # Split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # Get bounding box coordinates for each mask
        num_objects = len(obj_ids)
        boxes = []
        for i in range(num_objects):
            pos = np.where(masks[i])
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])
            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # There is only one class
        labels = torch.ones((num_objects,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Suppose all instances are not crowd
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """
        Return the number of images in the dataset
        """

        return len(self.image_paths)

    def get_image_path(self, idx):
        """
        Provide the path to the original image given the index
        """

        return os.path.join(PennFudanDataset.PNG_IMAGES_PATH, self.image_paths[idx])

    @staticmethod
    def get_transform(train):
        """
        Add horizontal flip transformation in case of training
        """

        transforms = []
        transforms.append(Transforms.ToTensor2())
        if train:
            transforms.append(Transforms.RandomHorizontalFlip2(0.5))
        return Transforms.Compose(transforms)
