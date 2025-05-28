# Assignment 4 – Road Segmentation Challenge

## Overview

This repository contains the submission for **Assignment 4** of the Computer Vision course. This assignment is structured as a **practical competition**, where students are required to improve a baseline semantic segmentation model to achieve the best possible trade-off between **accuracy** and **computational efficiency** on a **road segmentation** dataset.

---

## Task Description

We chose **Competition 3: Road Segmentation**, which involves:

* **Dataset**: 150 training images, each sized approximately 800 × 256 pixels.
* **Classes**: 19 semantic classes.
* **Objective**: Improve the performance of a baseline convolutional neural network (CNN) on this dataset.

---

## Baseline Summary

The baseline model provided includes the following components:

* Data loading and preprocessing using `torchvision`.
* Basic data augmentation strategies.
* A simple CNN architecture.
* Cross-entropy loss function.
* Training and evaluation routines.
* Output predictions for visual inspection.

---

## Our Improvements

We introduced several changes to enhance both **accuracy** and **efficiency**:

### Model & Training Enhancements

* Replaced baseline CNN with a lightweight **U-Net** variant.
* Integrated **Batch Normalisation**, **Dropout**, and **Residual** layers.
* Applied **pretrained encoder** *(Transfer Learning from CityScapes Dataset)*.
* Switched optimiser from SGD to **AdamW** with cyclical learning rate.
* Custom loss: **Cross-Entropy + Dice Loss** hybrid.

### Data Augmentation

* Added **Random Horizontal Flip**, **Color Jitter**, and **Random Crop/Resize** to name a few.

### ⚙️ Efficiency-Oriented Modifications

* Pruned redundant layers.
* Tracked computational cost (GFLOPs) using `ptflops`.
* Balanced accuracy vs computational cost for higher **accuracy-to-FLOP ratio**.

---

## Results Summary

| Metric                        | Value          |
| ----------------------------- | -------------- |
| **Baseline Accuracy**         |  28% *(0.28)*  |
| **Improved Accuracy**         | 42% *(0.4119)* |
| **Baseline GFLOPs**           |  67 *(66.97)*  |
| **Improved Model GFLOPs**     | 227 *(227.69)*  |
| **Efficiency (Acc / GFLOPs)** | 0.00184 *(bigger is better)*  |

*Detailed results, analysis, and limitations are documented in the report.*

---

## Submission Contents

* `assignment4_road_segmentation.ipynb`: Modified Jupyter notebook.
* `Assignment4_Report.pdf`: 2-page report outlining improvements, methodology, efficiency calculations, and reflections.
* `models/`: Folder with model architecture and training visulisation results.

---

## Evaluation Breakdown

| Component               | Weight   |
| ----------------------- | -------- |
| Accuracy Improvement    | 10 marks |
| Efficiency (Acc/GFLOPs) | 10 marks |
| Report                  | 30 marks |

---

## Technologies Used

* Python, PyTorch, NumPy.
* OpenCV for image preprocessing.
* `torchvision`, `albumentations` for augmentations.
* `ptflops` for model complexity analysis.

---

## Authors

* Maxwell Busato
* Liam Hennig

---
