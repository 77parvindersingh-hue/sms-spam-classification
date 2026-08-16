# 📱 SMS Spam Classification using NLP and Machine Learning

A Machine Learning project that classifies SMS messages as **Spam** or **Ham** using Natural Language Processing (NLP) and supervised Machine Learning techniques.

The project explores different text representations and classification algorithms and evaluates their performance using multiple metrics.

---

## 📌 Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, misleading information, or suspicious links.

The objective of this project is to build a Machine Learning system that can automatically classify an SMS message as either:

- **Ham** → Legitimate message
- **Spam** → Unwanted or suspicious message

The project follows a complete Machine Learning workflow, from data exploration and preprocessing to model evaluation and error analysis.

---

## 🎯 Objectives

- Understand and explore the SMS Spam Collection dataset.
- Perform exploratory data analysis (EDA).
- Clean and preprocess SMS text.
- Analyze duplicate records and class distribution.
- Apply Natural Language Processing techniques.
- Convert text into numerical features using:
  - Bag of Words (BoW)
  - TF-IDF
- Train multiple Machine Learning classification models.
- Compare model performance using different evaluation metrics.
- Perform hyperparameter tuning.
- Analyze classification errors.
- Use the final model to classify new SMS messages.

---

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset**.

The dataset contains labeled SMS messages belonging to two classes:

| Label | Description |
|---|---|
| `ham` | Legitimate SMS |
| `spam` | Spam SMS |

The dataset was obtained from the **UCI Machine Learning Repository**.

---

## 🔄 Project Workflow

```text
Dataset Collection
        ↓
Data Loading
        ↓
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Duplicate Analysis
        ↓
Text Preprocessing
        ↓
Bag of Words
        ↓
TF-IDF
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Final Model Selection
        ↓
New SMS Prediction
        ↓
Error Analysis
