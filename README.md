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
## 🧹 Text Preprocessing

The SMS messages were cleaned using the following steps:

- Converted text to lowercase
- Removed punctuation
- Removed numerical characters
- Removed extra whitespace

**Example preprocessing flow:**

```
Original Message
      ↓
Lowercase
      ↓
Remove Punctuation
      ↓
Remove Numbers
      ↓
Remove Extra Whitespace
      ↓
Cleaned Message
```

Stopwords were also analyzed to understand their proportion in the dataset. Approximately 40.07% of the words were stopwords. However, stopword removal was not applied to the final model because the existing TF-IDF + Linear SVM pipeline already produced strong results.

## 🔢 Feature Extraction

Two different text-vectorization techniques were compared.

### 1. Bag of Words

Bag of Words represents each SMS using word-frequency information. The resulting feature matrix contained:

- **Training Shape:** (4135, 7555)
- **Testing Shape:** (1034, 7555)
- **Vocabulary Size:** 7555

### 2. TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) represents words according to their importance within individual messages and across the dataset. The resulting feature matrix also contained:

- **Training Shape:** (4135, 7555)
- **Testing Shape:** (1034, 7555)
- **Vocabulary Size:** 7555

Both feature representations were evaluated with the same classification algorithms.

## 🤖 Machine Learning Models

Three classification algorithms were compared:

- **Multinomial Naive Bayes** — A probabilistic classifier that is particularly suitable for word-count-based text classification.
- **Logistic Regression** — A linear classification algorithm commonly used for binary classification and high-dimensional text data.
- **Linear SVM** — A linear Support Vector Machine that works effectively with sparse, high-dimensional text features.

## 📈 Model Comparison

The models were evaluated using Training Accuracy, Testing Accuracy, Precision, Recall, and F1-score. For Precision, Recall, and F1-score, Spam was treated as the positive class.

| # | Vectorizer | Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---|-----------|-------|-----------------|-----------------|-----------|--------|----------|
| 1 | BoW | Naive Bayes | 96.86% | 96.13% | 100.00% | 69.47% | 81.98% |
| 2 | BoW | Logistic Regression | 91.29% | 89.17% | 54.15% | 94.66% | 68.89% |
| 3 | BoW | Linear SVM | 95.70% | 91.88% | 61.24% | 97.71% | 75.29% |
| 4 | TF-IDF | Naive Bayes | 96.06% | 94.78% | 100.00% | 58.78% | 74.04% |
| 5 | TF-IDF | Logistic Regression | 96.95% | 96.03% | 95.92% | 71.76% | 82.10% |
| 🏆 6 | TF-IDF | Linear SVM | 99.90% | 97.87% | 95.80% | 87.02% | 91.20% |

## 🏆 Final Model

The best-performing model was: **TF-IDF + Linear SVM**

**Final Test Performance**

| Metric | Score |
|--------|-------|
| Accuracy | 97.87% |
| Precision | 95.80% |
| Recall | 87.02% |
| F1 Score | 91.20% |

The model achieved a strong balance between detecting Spam messages and avoiding incorrectly labeling legitimate messages as Spam.

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed on the TF-IDF + Linear SVM model by testing different values of the `C` parameter. The tested values were:

- 0.01
- 0.10
- 1
- 10
- 100

The tuning experiment produced an interesting result. The tuned model achieved:

- **Training Accuracy:** 100.00%
- **Training F1 Score:** 100.00%
- **Test Accuracy:** 97.68%
- **Precision:** 92.80%
- **Recall:** 88.55%
- **F1 Score:** 90.63%

The original model performed better on the unseen test set:

- **Original F1 Score:** 91.20%
- **Tuned F1 Score:** 90.63%

Therefore, the original TF-IDF + Linear SVM configuration was retained as the final model.

> **Key Learning:** Hyperparameter tuning does not guarantee better performance. A model can fit the training data more closely while performing slightly worse on unseen data. Therefore, tuning results should always be compared against the original baseline using test-set performance.

## 🧪 Testing on New SMS Messages

The final model was also tested on new SMS messages that were not part of the original test set.

**Example 1**
> Congratulations! You have won a free prize. Click now to claim.

Prediction: 🔴 Spam

**Example 2**
> Hey, are we still meeting for lunch today?

Prediction: 🟢 Ham

**Example 3**
> URGENT! You have won 50000 rupees. Call now to claim your reward.

Prediction: 🔴 Spam

The model correctly classified all three example messages.

## 🔎 Error Analysis

The final model was evaluated on 1,034 test messages. It produced:

| Metric | Count |
|--------|-------|
| False Positives | 5 |
| False Negatives | 17 |
| Total Errors | 22 |

**False Positives** — Legitimate Ham messages incorrectly classified as Spam.

**False Negatives** — Spam messages incorrectly classified as Ham.

The False Negative messages included promotional, service-related, entertainment, and conversational Spam messages. This shows that some Spam messages can resemble ordinary conversations and may not contain the common vocabulary strongly associated with Spam in the training data.

> **Key Observation:** TF-IDF + Linear SVM learns statistical patterns from text rather than understanding the actual semantic meaning of a message. Therefore, unusual or conversational Spam messages can sometimes be difficult to classify.

## 🧠 Key Learnings

This project helped me understand:

- How to inspect and clean a real-world text dataset
- How duplicate records can affect a dataset
- Why class distribution matters
- How to perform basic NLP preprocessing
- How Bag of Words represents text
- How TF-IDF represents word importance
- How different Machine Learning algorithms perform on text data
- Why Accuracy alone is not enough for classification problems
- The importance of Precision, Recall, and F1-score
- How to compare multiple models systematically
- Why hyperparameter tuning does not always improve a model
- How to analyze False Positives and False Negatives
- How to use a trained model to classify new text

## 🛠️ Technologies Used

- Python
- JupyterLab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Joblib
