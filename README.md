# Semi-Supervised Building Damage Classification

## Overview

This project implements a semi-supervised deep learning pipeline for assessing building damage from images taken after earthquakes. Leveraging transfer learning with a VGG16 convolutional neural network and a large set of unlabeled images, the model improves classification accuracy for post-disaster damage assessment.

The approach combines labeled data with pseudo-labeling techniques to effectively utilize unlabeled images, resulting in more robust damage state predictions at both scene and structural levels.

## Features

- **Transfer Learning:** Uses pretrained VGG16 as a fixed feature extractor with added fully connected layers for classification.  
- **Semi-Supervised Learning:** Employs dynamic pseudo-labeling to incorporate 4,000+ unlabeled images during training, enhancing model generalization.  
- **Multi-Task Capability:** Supports both scene-level (object, pixel, structure) and damage state classification (damaged, undamaged).  
- **Comprehensive Evaluation:** Performance metrics include accuracy, F1-score, ROC curves, confusion matrices, and t-SNE embeddings for feature visualization.

## Dataset

- Publicly available post-earthquake building image datasets, divided into labeled tasks and unlabeled sets.  
- Preprocessing includes resizing images to 224x224, normalization, and data augmentation for labeled training samples.

## Project Structure

* `dataloader.py`: Loads labeled and unlabeled datasets.
* `model.py`: Defines the VGG16-based CNN architecture.
* `train.py`: Contains training routines for supervised and semi-supervised learning.
* `pseudolabeling.py`: Implements pseudo-label generation and training callbacks.
* `evaluate.py`: Functions for plotting ROC curves, confusion matrices, and calculating metrics.
* `main.py`: Main script to run training and evaluation with configurable options.
* `processdata.py`: Data splitting and augmentation utilities.

## Results

* Achieved up to **89% accuracy** on labeled test sets.
* Semi-supervised training improved F1-score by approximately **6%** compared to supervised baseline.
* Visualizations confirm better class separation and confidence in predictions.
