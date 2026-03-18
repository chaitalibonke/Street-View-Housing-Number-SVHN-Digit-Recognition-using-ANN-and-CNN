# Street View Housing Number (SVHN) Digit Recognition using ANN and CNN

## Overview
This project applies deep learning techniques to recognize digits from
street-level photos using the Street View House Numbers (SVHN) dataset.
For this project, the original color images were converted to grayscale
to reduce computational complexity while retaining structural features
for digit recognition. Both Artificial Neural Networks (ANN) and
Convolutional Neural Networks (CNN) were built, trained, and evaluated
to identify the best-performing model for digit classification across
10 classes (0-9).

---

## Objective
- Preprocess and normalize the SVHN grayscale image dataset
- Build and evaluate two ANN models of increasing complexity
- Build and evaluate two CNN models of increasing complexity
- Compare model performance using accuracy, classification report,
  and confusion matrix
- Identify the best model for digit recognition

---

## Dataset

- **Source:** SVHN (Street View House Numbers) — Stanford University
- **Original Format:** .mat files (train_32x32.mat, test_32x32.mat)
- **Working Format:** Converted to .h5 (grayscale preprocessed)
- **Subset used:** 42,000 training images and 18,000 testing images
  (full dataset contains 73,257 training and 26,032 testing images)
- **Image size:** 32x32 pixels (grayscale)
- **Classes:** 10 digit classes (0 through 9)
- **Official dataset link:** http://ufldl.stanford.edu/housenumbers/
- **Kaggle mirror:** https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers

> Note: The dataset is not included in this repository.
> Download train_32x32.mat and test_32x32.mat from Kaggle,
> convert to .h5 format using the conversion script below,
> and upload to Google Drive before running the notebook.

---

## Dataset Conversion Script

Since the notebook requires an H5 file, use the following script
to convert the downloaded .mat files to the required format:
```python
import scipy.io
import h5py
import numpy as np

# Load the .mat files
train_data = scipy.io.loadmat('/content/drive/MyDrive/train_32x32.mat')
test_data = scipy.io.loadmat('/content/drive/MyDrive/test_32x32.mat')

# Extract images and labels
X_train_raw = train_data['X']  # shape (32, 32, 3, N)
y_train_raw = train_data['y'].flatten()
X_test_raw = test_data['X']
y_test_raw = test_data['y'].flatten()

# Transpose to (N, 32, 32, 3)
X_train_raw = X_train_raw.transpose(3, 0, 1, 2)
X_test_raw = X_test_raw.transpose(3, 0, 1, 2)

# Convert RGB to grayscale
# Grayscale conversion reduces computational complexity while
# retaining structural features needed for digit recognition
def rgb_to_grey(images):
    return np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])

X_train_grey = rgb_to_grey(X_train_raw)
X_test_grey = rgb_to_grey(X_test_raw)

# Fix labels — SVHN uses 10 to represent digit 0
y_train_raw[y_train_raw == 10] = 0
y_test_raw[y_test_raw == 10] = 0

# Save as H5 file
with h5py.File('/content/drive/MyDrive/SVHN_single_grey1.h5', 'w') as hf:
    hf.create_dataset('X_train', data=X_train_grey)
    hf.create_dataset('y_train', data=y_train_raw)
    hf.create_dataset('X_test', data=X_test_grey)
    hf.create_dataset('y_test', data=y_test_raw)

print("H5 file created successfully")
print("X_train shape:", X_train_grey.shape)
print("X_test shape:", X_test_grey.shape)
```

---

## Methodology

### 1. Data Preprocessing
- Load dataset from .h5 file stored in Google Drive
- Reshape images to 4D array (samples, 32, 32, 1) for CNN input
- Normalize pixel values from 0-255 to 0-1 by dividing by 255
- One-hot encode target labels using to_categorical()

### 2. ANN Model 1 — Base Fully Connected Network
- Input shape: (1024,) — flattened 32x32 image
- Hidden Layer 1: Dense(64, relu)
- Hidden Layer 2: Dense(32, relu)
- Output Layer: Dense(10, softmax)
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Epochs: 20, Batch size: 128, Validation split: 20%

### 3. ANN Model 2 — Enhanced Fully Connected Network
- Input shape: (1024,)
- Hidden Layer 1: Dense(256, relu)
- Hidden Layer 2: Dense(128, relu)
- Dropout(0.2)
- Hidden Layer 3: Dense(64, relu)
- Hidden Layer 4: Dense(64, relu)
- Hidden Layer 5: Dense(32, relu)
- BatchNormalization
- Output Layer: Dense(10, softmax)
- Optimizer: Adam (lr=0.0005)
- Loss: Categorical Crossentropy
- Epochs: 30, Batch size: 128, Validation split: 20%

### 4. CNN Model 1 — Base Convolutional Network
- Conv2D(16, 3x3, same) → LeakyReLU(0.1)
- Conv2D(32, 3x3, same) → LeakyReLU(0.1)
- MaxPooling2D(2x2)
- Flatten
- Dense(32) → LeakyReLU(0.1)
- Output: Dense(10, softmax)
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Epochs: 20, Batch size: 32, Validation split: 20%

### 5. CNN Model 2 — Enhanced Convolutional Network
- Conv2D(16, 3x3, same) → LeakyReLU(0.1)
- Conv2D(32, 3x3, same) → LeakyReLU(0.1)
- MaxPooling2D(2x2) → BatchNormalization
- Conv2D(32, 3x3, same) → LeakyReLU(0.1)
- Conv2D(64, 3x3, same) → LeakyReLU(0.1)
- MaxPooling2D(2x2) → BatchNormalization
- Flatten → Dense(32) → LeakyReLU(0.1) → Dropout(0.5)
- Output: Dense(10, softmax)
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Epochs: 30, Batch size: 128, Validation split: 20%

---

## Model Architectures Summary

### ANN Model 1
```
Input (1024)
    ↓
Dense(64, relu)
    ↓
Dense(32, relu)
    ↓
Dense(10, softmax)
```

### ANN Model 2
```
Input (1024)
    ↓
Dense(256, relu) → Dense(128, relu) → Dropout(0.2)
    ↓
Dense(64, relu) → Dense(64, relu) → Dense(32, relu)
    ↓
BatchNormalization → Dense(10, softmax)
```

### CNN Model 1
```
Input (32x32x1)
    ↓
Conv2D(16) → LeakyReLU → Conv2D(32) → LeakyReLU
    ↓
MaxPooling2D → Flatten → Dense(32) → LeakyReLU
    ↓
Dense(10, softmax)
```

### CNN Model 2
```
Input (32x32x1)
    ↓
Conv2D(16) → LeakyReLU → Conv2D(32) → LeakyReLU
    ↓
MaxPooling2D → BatchNormalization
    ↓
Conv2D(32) → LeakyReLU → Conv2D(64) → LeakyReLU
    ↓
MaxPooling2D → BatchNormalization
    ↓
Flatten → Dense(32) → LeakyReLU → Dropout(0.5)
    ↓
Dense(10, softmax)
```

---

## Results

### ANN Model 1 (20 Epochs)
| Metric               | Value  |
|----------------------|--------|
| Final Train Accuracy | 67.70% |
| Final Val Accuracy   | 67.45% |

### ANN Model 2 (30 Epochs)
| Metric               | Value  |
|----------------------|--------|
| Final Train Accuracy | 76.25% |
| Final Val Accuracy   | 76.29% |

### CNN Model 1 (20 Epochs)
| Metric               | Value  |
|----------------------|--------|
| Final Train Accuracy | 97.56% |
| Final Val Accuracy   | 87.01% |

### CNN Model 2 (30 Epochs)
| Metric               | Value  |
|----------------------|--------|
| Final Train Accuracy | 94.94% |
| Final Val Accuracy   | 91.32% |

> Best model: CNN Model 2, with 91.32% validation accuracy.

---

## Key Findings
- CNN models significantly outperformed ANN models for digit
  recognition because they better preserve and learn spatial features
- ANN Model 2 improved over ANN Model 1 by approximately 9 percentage
  points in validation accuracy through added depth and regularization
- CNN Model 1 exhibited overfitting, as indicated by a large gap
  between training accuracy (97.56%) and validation accuracy (87.01%)
- CNN Model 2 achieved the best generalization with 91.32%
  validation accuracy through BatchNormalization and Dropout
- LeakyReLU activation helped improve stability in the deeper
  CNN models
- CNNs achieved substantially better performance than ANNs on this
  task due to parameter sharing and stronger spatial feature extraction
- This result aligns with established deep learning theory, as CNNs
  leverage local receptive fields and parameter sharing, making them
  more effective for image-based tasks than fully connected architectures

---

## Tech Stack
- Python
- TensorFlow 2.12
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- h5py
- SciPy
- Google Colab

---

## How to Run

> This notebook is designed for Google Colab.

**Step 1 — Download the dataset from Kaggle:**
```
https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers
```
Download: train_32x32.mat and test_32x32.mat

**Step 2 — Upload both .mat files to Google Drive**

**Step 3 — Run the conversion script** provided above in the
Dataset Conversion Script section to generate SVHN_single_grey1.h5

**Step 4 — Open the notebook in Google Colab**

**Step 5 — Mount Google Drive and run all cells:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 6 — Install dependencies if needed:**
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python h5py scipy
```

---

## Limitations
- A subset of the full SVHN dataset was used to reduce computation time
- CNN Model 1 exhibited overfitting suggesting more regularization
  or data augmentation would improve generalization
- Models were trained on grayscale images only — color information
  was not utilized
- No data augmentation was applied which could further improve
  model robustness

---

## Future Work
- Apply data augmentation (rotation, zoom, shift) to improve
  model generalization
- Experiment with deeper CNN architectures such as VGG or ResNet
- Train on the full SVHN dataset for improved accuracy
- Add early stopping and learning rate scheduling callbacks
- Deploy the best model as a real-time digit recognition application

---

## Disclaimer
The SVHN dataset is publicly available at
http://ufldl.stanford.edu/housenumbers/ and on Kaggle at
https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers.
The notebook is designed to run on Google Colab and requires
Google Drive access for dataset loading.
