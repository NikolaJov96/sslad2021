# sslad2021

Entry for ICCV 2021 Workshop: Self-supervised Learning for Next-Generation Industry-level Autonomous Driving competition - [sslad2021](https://sslad2021.github.io/pages/challenge.html).

## Getting the data

Datasets used for all experiments are placed in subfolders under `data/`.

Download the provided main competition dataset from [here](https://soda-2d.github.io/documentation.html), and place the extracted data into `data/SSLAD-2D/` directory, so it can be found by the dataset loader.

Dataset loader classes are placed in subfolders under `structures/` for each dataset. SSLAD dataset loader assumes training, testing and validation datasets are downloaded (~6GB), but does not require any unlabeled data to work correctly.

To test if the SSLAD dataset loader loads the downloaded data correctly, run:

```
python3 structures/sslad_2d/sslad_dataset.py
```

## Exploring the data

Explore the dataset by running scripts inside the `exploring_data/` directory.

```
python3 exploring_data/display_training_bbox.py
python3 exploring_data/bbox_distribution.py
python3 exploring_data/lighting.py
```

During training, Trainer executes predictions on unlabeled data and saves them to annotation files. Following scripts are specific to analyzing prediction outputs from the trainer.

To display all predicted annotations run `display_annotations.py`, providing the annotation save file and a starting image id:

```
python3 exploring_data/trainer_predictions/display_annotations.py ./competition_models/<session>/annotations/annotation_1.json 0
```

To display predicted annotations with their confidence scores, one by one, run `display_annotation_scores.py`, providing the annotation save file and a starting image id:

```
python3 exploring_data/trainer_predictions/display_annotation_scores.py ./competition_models/<session>/annotations/annotation_1.json 0
```

To display predicted annotations in different score confidence ranges, run `display_score_ranges.py`, providing the annotation save file, a starting image id and a score range step:

```
python3 exploring_data/trainer_predictions/display_score_ranges.py /media/nikola/DataDrive/sslad2021/competition_models/<session>/annotations/annotation_1.json 0 0.1
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

As per competition rules, evaluator is calculating average precisions (AP) using the COCO evaluation API for every category, as well as the mean average precision(mAP). Calculation of mAP is custom, as different categories have different IoU thresholds. Results are returned as a list [mAP, AP1, ..., AP6].

### Data Augmentation

Different augmentations can be applied to image instances. Those can found in `competition_models/augmentations.py`. To preview them, run:

```
python3 competition_models/augmentations.py
```

### Trainer algorithm
