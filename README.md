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
