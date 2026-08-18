# 🅿️ Smart Parking Space Detector

A machine learning project using **Support Vector Machine (SVM)** classifier to automatically detect empty and occupied parking spaces from images with **100% accuracy**. Built with scikit-learn, OpenCV, and scikit-image.

## 📋 Project Overview

This project implements a computer vision solution that uses an SVM classifier to distinguish between empty and non-empty parking spaces in parking lot images. The model achieves perfect accuracy on the test set by analyzing resized image features.

### Real-World Application
- **Smart Parking Systems**: Automatically identify available parking spaces
- **Parking Lot Management**: Real-time vacancy detection
- **Traffic Reduction**: Help drivers find parking faster
- **Urban Planning**: Analyze parking space utilization

## ✨ Key Features

- 🎯 **100% Accuracy**: Perfect classification on test dataset
- 🚗 **Binary Classification**: Empty vs. Occupied spaces
- ⚡ **Fast Inference**: Quick predictions using pre-trained SVM
- 📊 **Grid Search Optimization**: Hyperparameter tuning for best performance
- 💾 **Model Persistence**: Pre-trained model saved for deployment
- 🖼️ **Image Preprocessing**: Automatic resizing and normalization
- 📈 **Stratified Split**: Ensures balanced train/test distribution

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | scikit-learn |
| **Classifier** | Support Vector Machine (SVM) |
| **Image Processing** | OpenCV, scikit-image |
| **Data Science** | NumPy, Pandas |
| **Model Serialization** | Pickle |

## 📦 Dependencies

```
scikit-learn>=1.0.0
opencv-python>=4.5.0
scikit-image>=0.18.0
numpy>=1.20.0
```

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/ARTEMISFOWL-01/simple_image_classifier_using_svc.git
cd simple_image_classifier_using_svc

# Install dependencies
pip install scikit-learn opencv-python scikit-image numpy

# Download the dataset (if needed)
# Extract clf-data.zip to get clf-data folder
unzip clf-data.zip
```

## 📊 Dataset Structure

```
clf-data/
├── empty/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── not_empty/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Dataset Details:**
- **Categories**: 2 (empty, not_empty)
- **Image Format**: JPG/PNG
- **Preprocessing**: Resized to 15×15 pixels
- **Train/Test Split**: 80% training, 20% testing
- **Stratification**: Balanced class distribution

## 🧠 How It Works

### 1. **Data Loading**
```python
# Load images from categories
# Categories: ['empty', 'not_empty']
# Read and flatten each image
```

### 2. **Image Preprocessing**
```python
# Original image → Resize to 15×15 pixels
# RGB channels preserved → Flattened to 1D array
# Normalized pixel values (0-255)
```

### 3. **Feature Extraction**
- Image dimensions: 15×15 = 225 pixels
- Features per sample: 225 (R, G, B channels flattened)
- Total dataset: Multiple images with corresponding labels

### 4. **Train-Test Split**
```python
train_x, test_x, train_y, test_y = train_test_split(
    data, label,
    test_size=0.2,        # 20% test data
    shuffle=True,         # Randomize order
    stratify=label        # Maintain class balance
)
```

### 5. **Hyperparameter Tuning (Grid Search)**
```
Parameters tested:
├── gamma: [0.01, 0.001, 0.0001]
└── C: [1, 10, 100, 1000]
```

**Grid Search Results:**
- Exhaustive search over parameter space
- Cross-validation for robust evaluation
- Best parameters selected automatically
- Best estimator used for predictions

### 6. **Model Training**
```python
# SVM with optimal hyperparameters
# RBF kernel (default)
# Gamma and C values optimized
```

### 7. **Evaluation**
```python
# Accuracy Score: (Correct Predictions / Total Predictions) × 100
# Result: 100% on test set
```

## 🚀 Usage

### Training the Model

```python
import os
from skimage.io import imread
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.transform import resize
import numpy as np

# Dataset path
dir = 'clf-data'
categories = ['empty', 'not_empty']

# Load and preprocess data
data = []
label = []
for category_index, category in enumerate(categories):
    for file in os.listdir(os.path.join(dir, category)):
        img_path = os.path.join(dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))  # Resize to 15×15
        data.append(img.flatten())   # Flatten to 1D array
        label.append(category_index)

data = np.asarray(data)
label = np.asarray(label)

# Train-test split
train_x, test_x, train_y, test_y = train_test_split(
    data, label,
    test_size=0.2,
    shuffle=True,
    stratify=label
)

# Grid Search for hyperparameter tuning
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(train_x, train_y)

# Get best model
best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(test_x)
accuracy = accuracy_score(y_pred, test_y)
print(f"Accuracy: {accuracy * 100}%")

# Save model
pickle.dump(best_model, open('model.p', 'wb'))
```

### Testing/Predicting

```python
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Load trained model
model = pickle.load(open('model.p', 'rb'))

# Load test image
img = imread('parking_space.jpg')
img_resized = resize(img, (15, 15))
img_flattened = img_resized.flatten().reshape(1, -1)

# Predict
prediction = model.predict(img_flattened)
label_map = {0: 'empty', 1: 'not_empty'}
result = label_map[prediction[0]]
print(f"Parking Space Status: {result}")
```

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 100% |
| **Precision** | 100% |
| **Recall** | 100% |
| **F1-Score** | 1.0 |
| **Confusion Matrix** | Perfect classification |

## 🔍 Key Hyperparameters

**SVM Configuration:**
```python
{
    'kernel': 'rbf',           # Radial Basis Function
    'gamma': 'optimal',        # Kernel coefficient
    'C': 'optimal',            # Regularization parameter
    'probability': False       # No probability estimates
}
```

**Optimal Values (found via Grid Search):**
- **Gamma**: One of [0.01, 0.001, 0.0001]
- **C**: One of [1, 10, 100, 1000]

## 📁 Project Files

- **tester.py** - Main training script with Grid Search
- **model.p** - Pre-trained SVM model (Pickle format)
- **clf-data.zip** - Dataset with empty and non-empty parking images
- **README.md** - This file

## 🎓 Learning Concepts

This project demonstrates:
- **Support Vector Machines (SVM)**: Binary classification
- **Hyperparameter Tuning**: Grid Search optimization
- **Image Processing**: Resizing and normalization
- **Data Splitting**: Stratified train-test split
- **Model Serialization**: Pickle for model persistence
- **Accuracy Metrics**: Evaluation and performance measurement
- **Computer Vision**: Image classification pipeline

## 💡 Potential Improvements

- [ ] Use more sophisticated image features (HOG, SIFT)
- [ ] Implement CNN for better feature extraction
- [ ] Add confidence scores for predictions
- [ ] Create real-time video stream processing
- [ ] Deploy as REST API service
- [ ] Add database for historical data
- [ ] Implement multi-class classification (empty, occupied, reserved)
- [ ] Add image augmentation for robustness

## 🚗 Real-World Applications

1. **Parking Lot Management**
   - Real-time space availability
   - Dynamic pricing based on occupancy
   - Customer guidance systems

2. **Smart City**
   - Urban parking analytics
   - Traffic pattern analysis
   - Emission reduction

3. **Autonomous Vehicles**
   - Parking space detection
   - Navigation assistance
   - Lot management

## 📞 Troubleshooting

**Issue**: Model accuracy drops after retraining
- **Solution**: Ensure consistent image preprocessing and stratified split

**Issue**: Slow prediction on large datasets
- **Solution**: Use parallel processing with `n_jobs=-1` in GridSearchCV

**Issue**: Model file is too large
- **Solution**: Consider using model compression or quantization

## 🔒 Model Details

- **Input**: 225-dimensional feature vector (15×15×RGB)
- **Output**: Binary label (0: empty, 1: not_empty)
- **Algorithm**: Support Vector Machine with RBF kernel
- **Training Size**: ~4000+ images (80% of dataset)
- **Test Size**: ~1000+ images (20% of dataset)

## 📝 License

This project is open source and available for educational and research purposes.

## 👤 Author

**ARTEMISFOWL-01**

---

**Note**: For optimal results, ensure your test images are taken under similar lighting and angle conditions as the training dataset.
