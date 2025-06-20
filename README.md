🧠 Brain Tumor Detection using CNN
This project implements a Convolutional Neural Network (CNN) for the classification and detection of brain tumors from MRI images. It leverages deep learning techniques to distinguish between tumorous and non-tumorous images, assisting in early diagnosis and treatment planning.

🎯 Objective
To build a binary image classifier using CNN that can:

Accurately detect presence of brain tumors in MRI images.

Be trained on real-world brain MRI data.

Be used as a prototype for AI-assisted diagnosis.

🛠️ Technologies Used
Python 🐍

TensorFlow / Keras

OpenCV

NumPy, Matplotlib

Jupyter Notebook

📊 Dataset
Source: Kaggle Brain MRI Images for Brain Tumor Detection

Categories: yes (with tumor), no (without tumor)

Format: PNG/JPG MRI images

🧪 Model Architecture
A basic CNN model with:

Convolution layers for feature extraction

MaxPooling layers for dimensionality reduction

Dropout for regularization

Fully Connected Dense layers for final classification

Example:

python
```bash
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```
📈 Training
Loss Function: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy

Epochs: ~10–20 (customizable)

📌 How to Run
Clone the repository:

```bash

git clone https://github.com/sbhavya28/Brain-Tumour-Prediction-Using-CNN.git
cd brain-tumor-detection-cnn
```


🔍 Run the notebook:
```bash
jupyter notebook Brain_Tumour_CNN.ipynb
```
📊 Results
Achieved accuracy: ~95% (may vary based on train/test split)

Training/validation graphs for loss and accuracy included

Sample predictions and confusion matrix available in notebook

📌 Future Improvements
Use of Transfer Learning (e.g., VGG16, ResNet)

Integration with Flask for web-based MRI image diagnosis

Model quantization for mobile deployment

📚 References
Kaggle Datasets

TensorFlow & Keras documentation

Research papers on Brain Tumor Detection using Deep Learning
