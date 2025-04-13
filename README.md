# SM-Technology-AI-Developer-Task

# Offensive Language Classification -

## 📌 Project Overview

This project addresses a real-world NLP classification task for identifying toxic content in online feedback. The main objective is to build machine learning models that can detect whether a comment is **toxic** or not, leveraging multi-label annotations (toxic, abusive, vulgar, menace, offense, bigotry) provided in the training set to improve model learning.

The dataset consists of feedback text with binary labels representing different forms of toxicity. While the model is trained using all six labels, final evaluation is based **only on the 'toxic' label**.

I developed and evaluated:
- **Model 1 (Baseline )**: Logistic Regression 
- **Model 2 (Transformer)**: Fine-tuned multilingual BERT

---

## 📂 Dataset Description

**Files Provided:**
- `train.csv`: Contains labeled data with 6 toxicity categories
- `test.csv`: Contains unlabeled data for prediction (binary toxic classification)

**Columns in `train.csv`:**
- `id`: Unique identifier
- `feedback_text`: Raw user feedback
- `toxic`: Target label (1 if toxic)
- `abusive`, `vulgar`, `menace`, `offense`, `bigotry`: Auxiliary labels to enrich model learning

---

## ⚙️ Model Implementation Details

### ✅ Model 1 - Logistic Regression (in `model1_implementation.ipynb`)
- **Text Preprocessing:** Tokenization, lowercasing, stopword removal, lemmatization
- **Feature Extraction:**
  - Logistic Regression: TF-IDF
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix

### 🚀 Model 2 - (in `model2_implementation.ipynb`)
- **Text Preprocessing:** Minimal, handled by tokenizer
- **Model:** Hugging Face Transformers `xlm-roberta-base`
- **Training:** Fine-tuned for binary classification
- **Handling Multilingual Data:** XLM-R's multilingual capability ensures better generalization on non-English text

---
I have completed Model 1, which achieved a good accuracy of 77%. However, I was unable to complete Model 2.
