# sslad2021

Entry for ICCV 2021 Workshop: Self-supervised Learning for Next-Generation Industry-level Autonomous Driving competition -  [sslad2021](https://sslad2021.github.io/pages/challenge.html).

## Getting the data

Data sets used for all experiments are placed in subfolders under `data/`. Data set loader classes are placed in subfolders under `structures/` for each data set.

Download the provided main competition data set from [here](https://soda-2d.github.io/documentation.html#data_collection), and place the extracted data into `data/SSLAD-2D/` directory, so it can be found by the data set loader.

## Exploring the data

Explore the data set by running scripts inside the `exploring_data/` directory.

```
python3 exploring_data/display_training_bbox.py
```

```
python3 exploring_data/bbox_distribution.py
```

## Competition models

Open-source PyTorch util scripts used for easier managing of PyTorch models are taken from [here](https://github.com/pytorch/vision/tree/master/references/detection) and placed in `pytorchscripts/`.

## Additional experiments

Experiments using different datasets or non-ImageNet pretraining and do not satisfy the competition rules. These models are placed in `playground/`.

### Pytorch Mask RCNN demo

Modified PyTorch Mask RCNN demo using the Penn-Fudan pedestrian dataset and a pretrained model. Original code and explanations can be found [here](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). Modified version lives in `playground/pytorch_maskrcnn_demo/` and the data set loader is in `structures/penn_fudan_ped/`.

Download the Penn-Fudan pedestrian data set for this demo from [here](https://www.cis.upenn.edu/~jshi/ped_html/), and place the extracted data into `data/PennFudanPed/` directory.

Run the additional training of the pretrained PyTorch MaskRCNN model and display testing set prediction results by running:

```
python3 playground/pytorch_maskrcnn_demo\penn_fudan_mask_rcnn.py
```
