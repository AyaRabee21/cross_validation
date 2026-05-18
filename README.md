# 📊 Cross-Validation in Machine Learning: A Comparative Study

This project aims to demonstrate the practical importance of evaluating machine learning models using **Cross-Validation** compared to the traditional **Train/Test Split**. These concepts are applied to a medical dataset (`heart.csv`) to predict heart disease using four different classification algorithms.

---

## 🧠 What is Cross-Validation?
Cross-Validation is a statistical technique used to evaluate machine learning models more accurately and realistically. Instead of splitting the data into a training set and a testing set just once, this technique divides the dataset into multiple parts or "folds" (e.g., 5 folds). In each iteration, one fold is held out for testing while the remaining folds are used for training. This process is repeated until every fold has been used as the test set exactly once. The final evaluation is the average performance across all iterations.

<img width="1524" height="1162" alt="K-Fold-Cross-Validation" src="https://github.com/user-attachments/assets/f65fd7e9-513a-461e-8340-6ca79b760d40" />

## ❓ Why Use Cross-Validation?
* **More Accurate Evaluation:** A standard Train/Test split can yield misleading results based on pure "luck" in how the data was divided. Cross-validation ensures the model is tested across different data subsets.
* **Detecting Overfitting:** It helps verify whether the model is simply memorizing the training data (overfitting) or if it can genuinely generalize to new, unseen data.
* **Maximizing Data Utility:** It guarantees that every single row in your dataset is used for both training and testing.

## ⏱️ When to Use It?
* **Small Datasets:** When available data is limited, holding out a fixed test set might deprive the model of valuable training information.
* **Model Comparison:** When comparing multiple algorithms to choose the best one fairly (which is the core of this project).
* **Hyperparameter Tuning:** To ensure that the parameters you select are the best for the overall dataset, not just for one specific test sample.

## ⚖️ Pros and Cons
**Pros:**
* Provides a reliable and robust estimate of model performance on unseen data.
* Significantly reduces bias and variance in the evaluation process.

**Cons:**
* **Computational Cost and Time:** Since the model is trained multiple times (e.g., 5 times for a 5-Fold CV), it consumes more time and computational resources compared to a simple split, especially with large datasets or complex models.

---

## 🛠️ Project Overview

In this project, we built a complete data processing and modeling pipeline to uncover the difference between standard evaluation and Cross-Validation.

### 1. Data Preprocessing
We used the `heart.csv` dataset. The data was prepared using `Pipeline` and `ColumnTransformer` to ensure clean data and prevent data leakage:
* **Numerical Data:** Missing values were handled using `SimpleImputer` (strategy="median"), and the features were scaled using `StandardScaler`.
* **Categorical Data:** Missing values were handled using `SimpleImputer` (strategy="most_frequent"), and the features were encoded using `OneHotEncoder`.

### 2. Models Used
We trained and evaluated four classic machine learning classifiers:
1. `LogisticRegression(max_iter=1000)`
2. `SVC(kernel='linear')`
3. `KNeighborsClassifier()`
4. `RandomForestClassifier()`

### 3. Results & Comparison
The project practically proves why Cross-Validation is crucial. 

* **Using Standard Split (`train_test_split`):**
  The initial evaluation showed that the **K-Nearest Neighbors (KNN)** model performed the best, achieving an accuracy of **~80.3%**.
* **Using Cross-Validation (`cv=5`):**
  The *true* and stable performance of the models was revealed. The accuracy of **KNN** dropped drastically to **~64.4%**, proving it was overfitting to the specific train/test split. On the other hand, **Logistic Regression**, **SVC**, and **Random Forest** demonstrated their robustness, achieving stable accuracies between **81.8% and 82.8%**.

**Conclusion:** This discrepancy perfectly illustrates why relying on a single `train_test_split` can lead to selecting the wrong model!

---
**Technologies & Libraries Used:** `Python`, `Pandas`, `NumPy`, `Scikit-Learn`.
