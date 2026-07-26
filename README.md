# ☀️ Solar Panel Defect Classification

A deep learning project to classify solar panel surface conditions into 6 categories using Convolutional Neural Networks (CNN) and Transfer Learning, served through a FastAPI backend with a Streamlit frontend, and containerized with Docker.

## 🔍 Problem Statement

Solar panels lose efficiency due to various surface issues like dust accumulation, bird droppings, snow cover, and physical/electrical damage. Manually inspecting large solar farms is time-consuming. This project builds an image classification system that automatically detects the condition of a solar panel from an image.

## 🗂️ Classes

The model classifies images into 6 categories:

- Clean
- Bird-drop
- Dusty
- Electrical-damage
- Physical-Damage
- Snow-Covered

## 🧠 Approach

The project was built step by step, starting simple and moving toward more advanced techniques:

1. **Baseline CNN (from scratch)** — Started with a custom CNN built from the ground up (Conv2D + MaxPooling blocks) to understand core CNN concepts before jumping into transfer learning.
2. **Transfer Learning — MobileNetV2 & ResNet50** — Tried multiple pretrained backbones (feature extraction + fine-tuning) to compare how different architectures perform on this dataset. Both were trained and fine-tuned, but the accuracy wasn't satisfactory enough to finalize.
3. **Transfer Learning — EfficientNetB0** — This backbone gave noticeably better and more consistent accuracy than the CNN, MobileNetV2, and ResNet50 attempts, so it was selected as the final model used in the deployed app.

Throughout the project, hyperparameters (learning rate, dropout, number of unfrozen layers, epochs, etc.) were tuned manually through iterative hit-and-trial experimentation rather than a fixed formula — adjusting one thing at a time and observing the effect on training/validation behavior.

All experiments (CNN, MobileNetV2, ResNet50, EfficientNetB0) are kept as separate notebooks in this repo for transparency, so the full comparison process is visible rather than just the final chosen model.

## ⚙️ Tech Stack

- **TensorFlow / Keras** — model building and training
- **EfficientNetB0** — final transfer learning backbone (fine-tuned)
- **Google Colab** — training environment (GPU)
- **FastAPI** — backend REST API serving the model
- **Streamlit** — frontend web app for interactive predictions
- **Docker** — containerized deployment

## 🏗️ Pipeline Overview

- Dataset organized into class-wise folders and split into train/validation/test sets
- Images resized to 224×224 and batched using `tf.data`
- Data augmentation (random flip, rotation, zoom, contrast) applied to improve generalization on a limited dataset
- Two-phase transfer learning:
  - **Phase 1:** Base model frozen, only a custom classification head trained
  - **Phase 2:** Top layers of the base model unfrozen and fine-tuned with a low learning rate (BatchNorm layers kept frozen to preserve pretrained statistics)
- Best model checkpointed based on validation loss, with early stopping to avoid overfitting

## 🏛️ Architecture

The app is split into two independent services:

- **FastAPI (`main.py`)** — loads the trained model and exposes a `/predict` endpoint that accepts an image and returns the predicted class, confidence score, and full class-wise probability breakdown as JSON.
- **Streamlit (`app.py`)** — a lightweight frontend where the user uploads an image; it sends the image to the FastAPI `/predict` endpoint and displays the result.

This separation keeps the model-serving logic independent from the UI, making the system easier to scale, test, or swap the frontend for another client later.

## 🖥️ App Features

- Upload a solar panel image and get:
  - Predicted defect class
  - Confidence score
  - Full class-wise probability breakdown
- Interactive Swagger docs available at `/docs` on the FastAPI service for direct API testing

**Live Demo:** _[link to be added]_

## 🚀 Running Locally

```bash
git clone https://github.com/rameezulhasan/solar-panel-defect-classification.git
cd solar-panel-defect-classification
pip install -r requirements.txt
```

**Terminal 1 — start the FastAPI backend:**
```bash
fastapi dev main.py --reload
```

**Terminal 2 — start the Streamlit frontend:**
```bash
streamlit run app.py
```

## 🐳 Running with Docker

```bash
docker build -t solar-panel-api .
docker run -p 8000:8000 solar-panel-api
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

## 📁 Repository Structure

```
solar-panel-defect-classification/
│
├── notebooks/
│   ├── with_cnn.ipynb                         # Baseline CNN from scratch
│   ├── with_mobilenet.ipynb                   # Transfer learning - MobileNetV2
│   ├── with_resnet.ipynb                      # Transfer learning - ResNet50
│   └── with_efficientnet.ipynb                # Transfer learning - EfficientNetB0 (final)
│
├── app.py                                     # Streamlit frontend
├── main.py                                    # FastAPI backend
├── Dockerfile                                 # Container setup for the API
├── best_model_efficientnetb0_finetuned.keras  # Final trained model
├── class_names.json                           # Class label mapping
├── requirements.txt                           # Dependencies
└── README.md
```

## 📌 Notes

This project was built primarily as a hands-on learning exercise — covering the full pipeline from raw image data, through CNNs and transfer learning, to a deployed application with a proper API/frontend split and Docker containerization.