# Warm-up experiments

Experiments unrelated to the competition, used to get familiar with the topic. Mostly utilizing datasets or pretrained models not allowed by the competition rules.

These models make use of open-source PyTorch util scripts for easier managing of PyTorch models, taken from [here](https://github.com/pytorch/vision/tree/master/references/detection) and placed in `pytorchscripts/`.

Some of the following experiments utilize the Penn-Fudan pedestrian dataset. Download it from [here](https://www.cis.upenn.edu/~jshi/ped_html/), and place the extracted data into `data/PennFudanPed/` directory. You can find the dataset loader in `structures/penn_fudan_ped/`.

## PyTorch Mask RCNN demo

Modified PyTorch Mask RCNN demo using the Penn-Fudan pedestrian dataset and a pretrained model. Original code and explanations can be found [here](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). Modified version lives in `playground/pytorch_mask_rcnn_demo/`. The model is Mask RCNN with ResNet50 pretrained on COCO dataset.

Run the additional training of the pretrained PyTorch MaskRCNN model and display testing set prediction results by running:

```
python3 playground/pytorch_mask_rcnn_demo/penn_fudan_mask_rcnn.py
```

## PyTorch Faster RCNN demo

Equivalent to the Mask RCNN demo, with Mask RCNN swapped with Faster RCNN with ResNet50 and pretrained only on ImageNet dataset, valid for use in the competition. Results are waker then the Mask RCNN because of the simpler pretraining dataset. To run it on the Penn-Fudan dataset run:

```
python3 playground/pytorch_faster_rcnn_demo/penn_fudan_faster_rcnn.py
```

## SSLAD-2D minimal PyTorch Faster RCNN

Equivalent to previous PyTorch implementations but adapted to the SSLAD-2D dataset. The model is Faster RCNN with ResNet50 pretrained on COCO dataset. Results are decent out of the box.

```
python3 playground/pytorch_faster_rcnn_sslad_minimal/sslad_faster_rcnn.py
```
