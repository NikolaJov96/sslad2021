# sslad2021

Entry for ICCV 2021 Workshop: Self-supervised Learning for Next-Generation Industry-level Autonomous Driving competition - [sslad2021](https://sslad2021.github.io/pages/challenge.html).

## Getting the data

Datasets used for all experiments are placed in subfolders under `data/`. Dataset loader classes are placed in subfolders under `structures/` for each dataset.

Download the provided main competition dataset from [here](https://soda-2d.github.io/documentation.html), and place the extracted data into `data/SSLAD-2D/` directory, so it can be found by the dataset loader.

## Exploring the data

Explore the dataset by running scripts inside the `exploring_data/` directory.

```
python3 exploring_data/display_training_bbox.py
```

```
python3 exploring_data/bbox_distribution.py
```

## Warm-up experiments

Warm-up experiments used to get familiar with the competition topic can be found in `playground/` along with a README with more information on them.

## Competition model structure

Competition related scripts are placed in `competition_models/`.

### Faster RCNN model

Faster RCNN model, with backbone pretrained on ImageNet, can be trained with default parameters by running:

```
python3 competition_models/faster_rcnn.py
```

Prediction examples on the validation set are displayed after the training.

### Evaluation

Evaluation script can be tested on the model created by the script above by running:

```
python3 competition_models/evaluator.py
```

As per competition rules, evaluator is calculating average precisions (AP) using the COCO evaluation API for every category, as well as the mean average precision(mAP). With different categories have specific IoU thresholds. Results are returned as a list [mAP, AP1, ..., AP6].
