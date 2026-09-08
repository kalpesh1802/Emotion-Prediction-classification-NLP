# 🧠 Emotion Classification with Deep Learning

> **An NLP-based emotion classification system that uses RNN, LSTM, GRU, and BiGRU architectures to classify text into six different emotions, with FastAPI-based deployment for real-time prediction.**

---

## 📌 Project Overview

Developed an end-to-end **NLP Emotion Classification System** that processes text and predicts the emotion expressed by the user. The project uses the **Dair-AI Emotion Dataset** and experiments with multiple recurrent neural network architectures including Simple RNN, LSTM, GRU, and Bidirectional GRU.

The text is preprocessed using tokenization, sequence conversion, and padding before being passed to Deep Learning models. Different architectures are evaluated and compared to identify the best-performing model.

The trained model and tokenizer are saved as reusable artifacts and integrated with **FastAPI** for real-time emotion prediction.

---

## 🚀 Key Features

* **📝 Emotion Classification:** Classifies text into six emotions — Sadness, Joy, Love, Anger, Fear, and Surprise.
* **🔤 NLP Preprocessing:** Converts raw text into numerical sequences using tokenization, padding, and truncation.
* **🤖 Multiple Deep Learning Models:** Experiments with Simple RNN, LSTM, GRU, and Bidirectional GRU architectures.
* **📊 Model Comparison:** Evaluates different architectures using test accuracy and test loss.
* **🏆 Best Model:** LSTM achieved **90.15% test accuracy** among the completed experiments.
* **⚡ FastAPI Deployment:** Provides an API endpoint for real-time emotion prediction.
* **📖 Swagger Documentation:** FastAPI `/docs` interface allows easy API testing.
* **💾 Model Artifacts:** Stores the trained model and tokenizer for inference without retraining.

---

## 🛠️ Tech Stack & Architecture

| Layer | Technology | Usage / Function |
| :--- | :--- | :--- |
| **Programming** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core programming language |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) | Model development and inference |
| **Framework** | ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) | Neural network implementation |
| **NLP** | ![NLP](https://img.shields.io/badge/NLP-Text_Processing-blue?style=for-the-badge) | Tokenization and text preprocessing |
| **Models** | ![RNN](https://img.shields.io/badge/RNN-Recurrent_Neural_Network-purple?style=for-the-badge) | Sequential text modeling |
| **Models** | ![LSTM](https://img.shields.io/badge/LSTM-Deep_Learning-orange?style=for-the-badge) | Long-term sequence learning |
| **Models** | ![GRU](https://img.shields.io/badge/GRU-Deep_Learning-green?style=for-the-badge) | Gated sequence modeling |
| **API** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) | Real-time model serving |
| **Server** | ![Uvicorn](https://img.shields.io/badge/Uvicorn-Server-blue?style=for-the-badge) | ASGI server |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | Data manipulation |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge) | Data visualization |
| **Visualization** | ![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-blue?style=for-the-badge) | EDA and visualization |
| **Dataset** | ![Hugging Face](https://img.shields.io/badge/HuggingFace-Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) | Emotion dataset |

---

## 📊 Model Performance

| Model | Test Accuracy | Test Loss |
| :--- | ---: | ---: |
| Simple RNN | 34.40% | ~1.7474 |
| **LSTM** | **90.15%** | **0.3202** |
| GRU | 11.45% | 1.7717 |
| BiGRU | Pending final evaluation | — |

### 🏆 Best Confirmed Model

**LSTM — 90.15% Test Accuracy**

Based on the completed experiments, LSTM achieved the highest confirmed test accuracy.

> ⚠️ The BiGRU training output currently available in the project is incomplete, so its final test accuracy is not reported.

---

## 🔄 Application Workflow

```text
                    ┌─────────────────┐
                    │    User Text    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  FastAPI API    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    Tokenizer    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Sequence +      │
                    │ Padding (50)    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Trained Model   │
                    │ LSTM / BiGRU    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Prediction    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Emotion +       │
                    │ Confidence      │
                    └─────────────────┘
---

## 📁 Project Structure
Emotion-Classification/
│
├── Artifacts/
│   ├── BiGRU_Model.keras
│   └── tokenizer.pkl
│
├── static/
│
├── final_clean.ipynb
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
---

## 🧠 NLP Preprocessing

The project uses the following preprocessing pipeline:

Raw Text
    ↓
Tokenization
    ↓
Integer Sequence
    ↓
Padding / Truncation
    ↓
Fixed Length Sequence
    ↓
Embedding Layer
    ↓
Deep Learning Model
---

## Configuration
Maximum vocabulary: 10,000 words
Maximum sequence length: 50 tokens
Padding: Post
Truncation: Post
OOV token: <unk>
---

## 🎯 Emotion Classes
The model predicts one of the following six emotions:

0 → Sadness
1 → Joy
2 → Love
3 → Anger
4 → Fear
5 → Surprise
---

## 📈 Key Learnings
Text preprocessing for Deep Learning
Tokenization and sequence generation
Padding and truncation
Word embeddings
Recurrent Neural Networks
LSTM architecture
GRU architecture
Bidirectional GRU
Model evaluation and comparison
Model serialization
FastAPI model deployment
REST API development
---

## ⚙️ Application Setup
Prerequisites

Make sure the following are installed:

Python 3.14.7
Git
VS Code or any Python IDE
FastAPI
Uvicorn
TensorFlow
---

## 🚀 Installation & Running Locally
Create Virtual Environment
python -m venv venv
Activate Virtual Environment
venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Start FastAPI
uvicorn main:app --reload
Open the Application
API: http://127.0.0.1:8000
Swagger API Documentation: http://127.0.0.1:8000/docs
---

👨‍💻 Author
Kalpesh Patil
Data Science | Machine Learning | Deep Learning | NLP | AI/ML
⭐ If you find this project useful, consider giving it a star.
