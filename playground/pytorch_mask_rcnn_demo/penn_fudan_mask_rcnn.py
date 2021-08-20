# Revised sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import cv2
import os
import pathlib
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import pytorchscripts.utils as utils
from pytorchscripts.engine import train_one_epoch, evaluate

from structures.penn_fudan_ped.penn_fudan_dataset import PennFudanDataset


class PennFudanMaskRCNN():
    """
    Class wrapping the pytorch MaskRCNN
    """

    # Our dataset has two classes only - background and person
    NUM_CLASSES = 2
    MASK_PREDICTOR_HIDDEN_LAYERS = 256
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

        self.model = PennFudanMaskRCNN.get_model_instance_segmentation()


    def get_model_instance_segmentation():
        """
        Construct the rcnn model
        """

        # Load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # Get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, PennFudanMaskRCNN.NUM_CLASSES)

        # Now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # And replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           PennFudanMaskRCNN.MASK_PREDICTOR_HIDDEN_LAYERS,
                                                           PennFudanMaskRCNN.NUM_CLASSES)

        return model

    def train(self):
        """
        Train the final NN layers, evaluate and save the model
        """

        # Use our dataset and defined transformations
        PFD_dataset = PennFudanDataset(train=True)
        PFD_dataset_test = PennFudanDataset(train=False)

        # Split the dataset in train and test set
        indices = torch.randperm(len(PFD_dataset)).tolist()
        dataset = torch.utils.data.Subset(PFD_dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(PFD_dataset_test, indices[-50:])

        # Define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=1,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=1,
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
        Wrap the pytorch model evaluation
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

    def draw_boxes(self, predictions, images):
        """
        Draw prediction boxes over provided images
        """

        if len(predictions) != len(images):
            return

        for i in range(len(predictions)):
            for j in range(len(predictions[i]['boxes'])):
                x1, x2, x3, x4 = map(int, predictions[i]['boxes'][j].tolist())
                images[i] = cv2.rectangle(images[i], (x1, x2), (x3, x4), (255, 0, 0), 1)


if __name__ == '__main__':
    """
    Test the model on a few images
    """

    model = PennFudanMaskRCNN()

    print('trying to load the model')
    if model.load_model():
        print('model loaded')
    else:
        print('trining the model')
        model.train()
        print('model trained, saving the model')
        model.save_model()
        print('model saved')

    dataset = PennFudanDataset(train=False)
    dataset_size = len(dataset)
    print('loaded dataset with {} images'.format(dataset_size))

    window_name = 'Predictions'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for i in range(dataset_size):

        # Predict one by one to avoid exceeding GPU memory
        x = [dataset[i][0]]
        prediction = model.eval(x)[0]

        image = cv2.imread(dataset.get_image_path(i), cv2.COLOR_BGR2RGB)

        model.draw_boxes([prediction], [image])

        cv2.imshow(window_name, image)
        # Exit on esc
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()
