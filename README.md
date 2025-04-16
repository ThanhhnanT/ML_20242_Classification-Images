# Animal and Plant Classification

This repository contains a machine learning project aimed at classifying animals and plants into their respective categories using deep learning techniques. The project uses a pretrained model (ResNet) to classify 90 animal species and 40 plant species based on images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview

This project focuses on using deep learning techniques to classify images into two main categories:
- **Animals**: 90 species, with around 60 images per species.
- **Plants**: 40 species, with around 1000 images per species.

We use a pretrained ResNet model for transfer learning and fine-tune it on the specific dataset. The model is capable of classifying images with high accuracy, and it can be further improved with more training and hyperparameter tuning.

## Dataset

The dataset contains two categories:
1. **Animal Dataset**: Includes 90 species, each with around 60 images.
2. **Plant Dataset**: Includes 40 species, each with around 1000 images.

The images have been preprocessed and divided into training and testing datasets to allow for model evaluation and validation.

### Dataset Structure
