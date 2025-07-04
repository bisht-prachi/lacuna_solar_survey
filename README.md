# Solar Panel Count Prediction

Predict the number of solar panel "boil" and "pan" units from aerial images and metadata using deep learning (EfficientNetV2 with metadata fusion).

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Albumentations](https://img.shields.io/badge/Augmentation-Albumentations-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This repository was built for the **Lacuna Solar Survey Challenge** on Kaggle. It uses a hybrid deep learning architecture combining image features and structured metadata to predict:

- `boil_nbr`: Number of boiling units
- `pan_nbr`: Number of panel units

---

## Directory Structure

  solar-panel-prediction
  -- images/  
  -- notebooks/
  -- Train.csv/
  -- Test.csv/
  -- main.py/
  -- README.md
  -- submission_*.csv
  
---

## Setting up

### 1. Clone the repo

git clone https://github.com/yourusername/solar-panel-prediction.git
cd solar-panel-prediction

### 2. Set up environment
bash
Copy
Edit
pip install -r requirements.txt
Make sure you have access to a CUDA-compatible GPU and your data paths are correctly set in the script.

## Training
To train the model across 3 folds:
python main.py
The best model per fold will be saved as:
best_model_fold0.pth
best_model_fold1.pth
best_model_fold2.pth

## Inference & Submission
Predictions on the test set will be saved as:

submission_original.csv (with float predictions)

submission_integer.csv (with rounded integers)

## Model Architecture
Backbone: tf_efficientnetv2_b3 from timm

Metadata Processor: Fully connected + LayerNorm + Dropout

Fusion: Image and metadata features are concatenated after attention

Regressor: 2-head count predictor with Softplus output

## Tools Used
PyTorch + AMP

Albumentations for augmentation

K-Fold Cross Validation

Huber Loss + CosineAnnealing Scheduler

Multihead Attention for metadata embedding

## Evaluation Metric
Mean Absolute Error (MAE) on the boil_nbr and pan_nbr predictions.

## License
This project is licensed under the MIT License.

### Acknowledgements
Inspired by the Lacuna Solar Survey Challenge on Kaggle.

EfficientNetV2 and pretrained models provided by timm.
