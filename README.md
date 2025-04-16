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


## Model

We use a **ResNet** model, pretrained on ImageNet, for this classification task. Transfer learning is employed by fine-tuning the model's last few layers for our specific dataset.

### Steps Involved:
1. Load the pretrained ResNet model.
2. Freeze the initial layers and train the last few layers with the dataset.
3. Apply image augmentation and normalization to improve the model’s generalization.
4. Train the model using an appropriate optimizer and loss function.
5. Evaluate the model on a test set.

## Installation

To get started with this project, follow these steps to set up the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/animal-plant-classification.git
2. Download checkpoint [here]([url](https://drive.google.com/drive/folders/1gJBBemxc0vCnqhe69FtZ4jUpNTWXLXPT?usp=sharing))
   
