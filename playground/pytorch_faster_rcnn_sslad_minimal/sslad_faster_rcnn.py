# Revised sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import cv2
import os
import pathlib
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms as T

import pytorchscripts.utils as utils
from pytorchscripts.engine import train_one_epoch, evaluate

from structures.sslad_2d.annotation import Annotation
from structures.sslad_2d.image import Image
from structures.sslad_2d.pytorch_sslad_dataset import SSLADDataset, PyTorchSSLADDataset


class SSLADFasterRCNN():
    """
    Class wrapping the PyTorch FasterRCNN pretrained on COCO train2017 applied to the SSLAD-2D dataset
    Intended as minimal working implementation
    """

    # Background + 6 object classes = 7 classes
    NUM_CLASSES = 7
    DEFAULT_NUM_EPOCHS = 10

    def __init__(self, num_epochs=DEFAULT_NUM_EPOCHS):
        """
        Initialize the model
        """

        self.num_epochs = num_epochs

        self.save_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), 'model_save')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.save_file = os.path.join(self.save_dir, 'torch_model.pt')

        # Use the GPU if available, avoid cluttering the main system GPU by using another one if available
        self.device = torch.device('cuda:{}'.format(torch.cuda.device_count() - 1)) \
            if torch.cuda.is_available() else torch.device('cpu')

        self.model = SSLADFasterRCNN.get_model_instance_segmentation()


    def get_model_instance_segmentation():
        """
        Construct the rcnn model
        """

        # Load an instance segmentation model pre-trained only on ImageNet
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, SSLADFasterRCNN.NUM_CLASSES)

        return model

    def train(self):
        """
        Train the final NN layers, evaluate and save the model
        """

        # Use our dataset and defined transformations
        SSLAD_dataset = PyTorchSSLADDataset(train=True)
        SSLAD_dataset_test = PyTorchSSLADDataset(train=False)

        # Define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            SSLAD_dataset, batch_size=2, shuffle=True, num_workers=1,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            SSLAD_dataset_test, batch_size=1, shuffle=False, num_workers=1,
            collate_fn=utils.collate_fn)

        # Move model to the right device
        self.model.to(self.device)

        # Construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        # And a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(self.num_epochs):
            # Train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, data_loader, self.device, epoch, print_freq=10)
            # Update the learning rate
            lr_scheduler.step()

        # Evaluate on the test dataset
        evaluate(self.model, data_loader_test, device=self.device)

    def eval(self, x):
        """
        Wrap the PyTorch model evaluation
        """

        x = [i.to(device=self.device) for i in x]
        self.model.eval()
        return self.model(x)

    def save_model(self):
        """
        Saves the trained model to the disk
        """

        torch.save(self.model.state_dict(), self.save_file)

    def load_model(self):
        """
        Load trained model from file
        """

        if os.path.exists(self.save_file):
            self.model.load_state_dict(torch.load(self.save_file))
            self.model.to(device=self.device)
            return True
        else:
            return False


if __name__ == '__main__':
    """
    Test the model on a few test images with no label provided
    """

    model = SSLADFasterRCNN()

    print('trying to load the model')
    if model.load_model():
        print('model loaded')
    else:
        print('trining the model')
        model.train()
        print('model trained, saving the model')
        model.save_model()
        print('model saved')

    dataset = SSLADDataset()
    dataset.load()

    window_name = 'Predictions'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for image in dataset.testing_images:

        prediction = model.eval([T.ToTensor()(image.get_pil_img())])[0]
        bboxes = prediction['boxes'].tolist()
        labels = prediction['labels'].tolist()

        cv2_image = image.get_cv2_img().copy()

        annotations = [
            Annotation(dataset.categories[labels[i]], *PyTorchSSLADDataset.pytorch_to_sslad(bboxes[i]))
            for i in range(len(prediction['boxes']))
        ]
        bbox_image = image.draw_annotations(annotations=annotations)
        bbox_image = Image.resize_to_width(bbox_image, 1000)

        cv2.imshow(window_name, bbox_image)
        # Exit on esc
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()
