
---

# **Truth-Lie Detection Model**

This repository contains a deep learning-based image classification model designed to detect whether a subject is telling the **truth** or **lying** based on visual input. Built using **TensorFlow** and **Keras**, the model employs a **Convolutional Neural Network (CNN)** to classify images into two categories: **Lie** and **Truth**.

---

## 📚 Table of Contents

* [Project Overview](#project-overview)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Evaluation Metrics](#evaluation-metrics)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [License](#license)

---

## 📌 Project Overview

This project aims to develop a deep learning model capable of distinguishing between truthful and deceptive visual expressions based on labeled image data.

The dataset is composed of two classes:

* **Lie** – images labeled as deceptive
* **Truth** – images labeled as honest

The model leverages a **CNN architecture** to extract features and learn the underlying visual patterns associated with each category.

### 🔑 Key Features

* **End-to-End Pipeline**: From image preprocessing to training and evaluation.
* **CNN-based Classification**: Utilizes Conv2D and MaxPooling2D layers for feature extraction.
* **Metrics-Driven Evaluation**: Incorporates precision, recall, F1-score, and confusion matrix alongside accuracy.
* **Hyperparameter Optimization**: Integrates **Keras Tuner** for automated hyperparameter tuning.

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/truth-lie-detection.git
   cd truth-lie-detection
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. **Prepare your dataset**
   Structure your image dataset as:

   ```
   dataset/
     ├── Lie/
     └── Truth/
   ```

2. **Run training script**

   ```bash
   python train.py
   ```

3. **Evaluate the model**

   ```bash
   python evaluate.py
   ```

4. **Hyperparameter tuning (optional)**

   ```bash
   python tune_hyperparameters.py
   ```

---

## 🧠 Model Architecture

A simplified CNN architecture is used:

* `Conv2D` → `ReLU` activation
* `MaxPooling2D`
* `Flatten`
* `Dense` (fully connected)
* `Softmax` output for binary classification

> Customization of the model is possible to improve accuracy using deeper layers, dropout, batch normalization, etc.

---

## 📊 Evaluation Metrics

Model performance is assessed using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

> Evaluation is performed on a separate validation/test set to prevent overfitting.

---

## 🔧 Hyperparameter Tuning

The project includes support for **Keras Tuner** to explore optimal values for:

* Number of convolutional filters
* Kernel size
* Dropout rate
* Learning rate
* Optimizer selection

Run the tuning script to automatically search for the best configuration.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

Let me know if you'd like help writing the actual training code, evaluation script, or Keras Tuner integration!
