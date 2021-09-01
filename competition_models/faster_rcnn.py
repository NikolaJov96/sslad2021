import cv2
import math
import os
import pathlib
import sys

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms

from competition_models.dataset_wrapper import DatasetWrapper
from structures.sslad_2d.image import Image
from structures.sslad_2d.sslad_dataset import SSLADDataset


class FasterRCNN:
    """
    General model class wrapping the PyTorch Faster RCNN
    """

    # Background + 6 object classes = 7 classes
    NUM_CLASSES = 7
    BATCH_SIZE = 2

    def __init__(self):
        """
        Initialize the model and detect the optimal device
        """

        self.device = torch.device('cuda:{}'.format(torch.cuda.device_count() - 1)) \
            if torch.cuda.is_available() else torch.device('cpu')

        self.model = self.initialize_model()

    def initialize_model(self):
        """
        Create the model and replace the predictor with desired number of classes
        """

        # Initialize the faster RCNN model with ResNet50 with the backbone pretrained on ImageNet
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=True)

        # Get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, FasterRCNN.NUM_CLASSES)

        # Move the model to the optimal device
        model.to(self.device)

        return model

    def train(self, dataset, num_epochs):
        """
        Execute a number of training epochs on a provided dataset wrapper
        """

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FasterRCNN.BATCH_SIZE,
            shuffle=True,
            num_workers=1,
            collate_fn=DatasetWrapper.collate_fn
        )

        # Construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # And a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Put the model into the training mode
        self.model.train()

        # Loop through the epochs
        for epoch in range(num_epochs):

            # Loop through the batches
            for batch, (images, targets) in enumerate(data_loader):

                # Move images and targets to the device
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Calculate the losses for each batch
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                loss_value = losses.item()
                print('\repoch: {}/{}, batch: {}/{}, loss: {}'.format(
                        epoch + 1, num_epochs, batch + 1, len(dataset) // FasterRCNN.BATCH_SIZE, loss_value
                    ),
                    end=''
                )

                if not math.isfinite(loss_value):
                    print("loss is {}, stopping training".format(loss_value))
                    print(loss_dict)
                    sys.exit(1)

                # Reset the gradients used in forward pass
                optimizer.zero_grad()

                # Calculate backward pass gradients
                losses.backward()

                # Update the model weights
                optimizer.step()

            print()

            # Update the learning rate
            lr_scheduler.step()

    def eval(self, x):
        """
        Wrap the PyTorch model evaluation
        """

        x = [transforms.ToTensor()(img).to(device=self.device) for img in x]
        self.model.eval()
        return self.model(x)

    def save_model(self, save_file):
        """
        Saves the trained model to the disk
        """

        torch.save(self.model.state_dict(), save_file)

    def load_model(self, save_file):
        """
        Load trained model from file
        """

        if os.path.exists(save_file):
            self.model.load_state_dict(torch.load(save_file))
            self.model.to(device=self.device)
            return True
        else:
            return False


def main():
    """
    Execute the basic model training using the training set
    Show results on a few testing images
    """

    dataset = SSLADDataset()
    dataset.load()

    model = FasterRCNN()

    # Model save file
    save_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), 'model_save')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, 'torch_model.pt')

    # Try to load saved model
    if not model.load_model(save_file):
        # If save file not found, train the model and save it
        dataset_wrapper = DatasetWrapper(
            images=dataset.training_images
        )

        model.train(
            dataset=dataset_wrapper,
            num_epochs=10
        )

        model.save_model(save_file)

    # Show test set predictions
    window_name = 'Predictions'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for image in dataset.testing_images:

        predictions = model.eval([image.get_pil_img()])

        annotations = DatasetWrapper.prediction_to_annotations(dataset, predictions)[0]
        bbox_image = image.draw_annotations(annotations=annotations)
        bbox_image = Image.resize_to_width(bbox_image, 1000)

        cv2.imshow(window_name, bbox_image)
        # Exit on esc
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
