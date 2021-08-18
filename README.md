# sslad2021
Entry for ICCV 2021 Workshop: Self-supervised Learning for Next-Generation Industry-level Autonomous Driving competition -  [sslad2021](https://sslad2021.github.io/pages/challenge.html).

## Getting the data
Download provided data set from [here](https://soda-2d.github.io/documentation.html#data_collection), and place the extracted data into `data/SSLAD-2D/` directory, so it can be found by the data set loader.

## Exploring the data
Explore the data set by running scripts inside the `exploring_data/` directory.

## Additional experiments

### Pytorch RCNN demo

Pytorch util modules are gathered from [here](https://github.com/pytorch/vision/tree/master/references/detection) and placed in `pytorchscripts/`. Pytorch demo code is downloaded from [here](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), slightly modified and placed in `playground/pytorch_RCNN_minimal_demo`.

Download the Penn-Fudan pedestrian data set for this demo from [here](https://www.cis.upenn.edu/~jshi/ped_html/), and place the extracted data into `data/PennFudanPed/` directory.

Additionally train the pretrained pytorch MaskRCNN model and display testing set prediction results by running `playground/pytorch_RCNN_minimal_demo\penn_fudan_mask_cnn.py`.
