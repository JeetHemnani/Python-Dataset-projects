# 🫀 Heart Disease Prediction using Machine Learning

This project aims to **predict the presence of heart disease** in a patient using various machine learning algorithms. It explores different models, improves them using hyperparameter tuning (GridSearchCV), and evaluates performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

---

## 📌 Project Objectives

- Load and understand the heart disease dataset
- Preprocess and analyze the data
- Train multiple classification models
- Tune the best-performing model
- Compare all models and finalize the most accurate one

---

## 📊 Dataset Information

- **Dataset Source:** [Kaggle Heart Disease UCI Dataset](https://www.kaggle.com/datasets)
- **Rows:** 303
- **Columns:** 14 (including target)

### 🔑 Key Features:
| Feature       | Description                                |
|---------------|--------------------------------------------|
| age           | Age of the patient                         |
| sex           | Gender (1 = male, 0 = female)              |
| cp            | Chest pain type (4 values)                 |
| trestbps      | Resting blood pressure                     |
| chol          | Serum cholesterol in mg/dl                 |
| fbs           | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg       | Resting electrocardiographic results       |
| thalach       | Max heart rate achieved                    |
| exang         | Exercise induced angina (1 = yes, 0 = no)  |
| oldpeak       | ST depression induced by exercise          |
| slope         | Slope of the peak exercise ST segment      |
| ca            | Number of major vessels (0–3)              |
| thal          | Thalassemia (0 = normal; 1 = fixed defect) |
| condition     | Target (1 = heart disease, 0 = no disease) |

---

## 🧪 Machine Learning Models Used

| Model                    | Accuracy |
|--------------------------|----------|
| Logistic Regression      | 73%      |
| Random Forest Classifier | 77% (after tuning) |
| K-Nearest Neighbors      | 72%      |
| Support Vector Machine   | 73%      |
| XGBoost                  | 68%      |

✅ **Best model**: Random Forest Classifier after tuning using `GridSearchCV`

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## ⚙️ Key Techniques

- Data preprocessing and cleaning
- Train-test split
- Model evaluation using:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
- Hyperparameter tuning using:
  - `GridSearchCV` for Random Forest

---

## 📈 Final Evaluation (Best Model - Tuned Random Forest)

- **Accuracy:** 77%
- **Precision (Class 0):** 0.78
- **Recall (Class 0):** 0.78
- **Precision (Class 1):** 0.75
- **Recall (Class 1):** 0.75
- **Confusion Matrix:**
[[25 7]
[ 7 21]]
