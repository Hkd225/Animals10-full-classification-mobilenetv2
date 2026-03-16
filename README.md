# Animals-10 Image Classification (Full 10 Classes)
## End-to-End Transfer Learning Pipeline & TFLite Deployment
### By Muhammad Auffa Hakim Aditya

This project presents a complete Deep Learning pipeline designed to classify 10 different species of animals. Utilizing the Animals-10 dataset, the project implements a two-stage Transfer Learning approach with a MobileNetV2 backbone. A unique feature of this pipeline is its built-in Italian-to-English translation mapping, addressing the dataset's native Italian folder structure to output universally understandable predictions.

The project was developed by Muhammad Auffa Hakim Aditya to demonstrate structured Machine Learning workflows, from automated dataset downloading and multi-lingual label handling to fine-tuning and packaging the model into mobile-ready deployment artifacts.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Automated Data Engineering: Download the Animals-10 dataset directly via KaggleHub and safely extract all 10 animal classes.
2. Cross-Language Label Mapping: Implement a programmatic dictionary to translate the native Italian class names (e.g., "cavallo", "ragno") into English ("horse", "spider") for user-facing inference.
3. Robust Data Splitting: Utilize `split-folders` to automatically partition the dataset into Training (80%), Validation (10%), and Testing (10%) sets.
4. Two-Stage Transfer Learning:
   - Stage 1: Feature Extraction (Training a custom top-layer while freezing the MobileNetV2 base).
   - Stage 2: Fine-Tuning (Unfreezing the top 100 layers of the base model with a reduced learning rate to boost accuracy).
5. Dynamic Callbacks: Implement a custom `StopAtAccuracy` callback to automatically halt training once validation accuracy reaches the 90% threshold, saving computational resources.
6. Automated Zipping & Export: Package the final `.keras` model, the optimized `.tflite` model for mobile edge devices, and training metadata into a clean, downloadable `.zip` archive.

------------------------------------------------------------

DATASET INFORMATION

Source          : Kaggle (alessiocorrado99/animals10)
Classes Used    : All 10 Classes
- cane (dog)
- cavallo (horse)
- elefante (elephant)
- farfalla (butterfly)
- gallina (chicken)
- gatto (cat)
- mucca (cow)
- pecora (sheep)
- ragno (spider)
- scoiattolo (squirrel)

------------------------------------------------------------

MACHINE LEARNING PIPELINE ARCHITECTURE

1. Data Augmentation:
   - Random Flip, Rotation (15%), Zoom (15%), Contrast (10%), and Translation (10%) to ensure generalization.

2. Model Architecture:
   - Base Model: MobileNetV2 (pre-trained on ImageNet, without the top layer).
   - Custom Head:
     - BatchNormalization
     - Dense (256 units, ReLU, with L2 Regularization to prevent overfitting)
     - Dropout (0.5)
     - Output Dense (10 classes, Softmax)

------------------------------------------------------------
```md
## DEPLOYMENT ARTIFACTS & SUBMISSION STRUCTURE

The script automatically generates a `submission/` folder and compresses it into a ready-to-deploy `.zip` archive containing:

```text
submission/
├── tflite/
│   ├── model.tflite                  # Optimized for Android/iOS
│   └── label.txt
├── klasifikasi-hewan-10.keras        # Native Keras format
├── training_config.json              # Hyperparameters and evaluation metadata
├── README.md
└── requirements.txt
------------------------------------------------------------

INSTALLATION

Install the required dependencies:

pip install tensorflow kagglehub split-folders scikit-learn seaborn

------------------------------------------------------------

HOW TO RUN

1. Clone this repository:
   git clone https://github.com/YOUR_USERNAME/animals10-full-classification.git

2. Run the Python script or Google Colab Notebook. The script will handle everything from downloading the dataset to generating the ZIP file.
3. Live Inference: At the end of the script, you can upload any animal photo. The model will output the prediction in both Italian and English along with its confidence score.

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

This project was developed as an exploration of:
- Deep Learning & Computer Vision
- Transfer Learning & Fine-Tuning (MobileNetV2)
- Multi-Language Label Handling
- TensorFlow Lite (TFLite) Mobile Deployment
- Automated MLOps Pipeline Packaging

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Image Classification
- Transfer Learning MobileNetV2
- Animals-10 Dataset
- TensorFlow Lite TFLite
- Deep Learning Portfolio
