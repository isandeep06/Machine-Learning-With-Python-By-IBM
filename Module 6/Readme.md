# 🎓 Machine Learning with Python – Module 6 (Final Project & Course Summary)

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)](https://www.python.org/)
[![Library: Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Course: IBM ML](https://img.shields.io/badge/Course-IBM%20Machine%20Learning%20with%20Python-lightblue?logo=ibm)](https://www.coursera.org/learn/machine-learning-with-python/home/module/6)
[![Status: Completed](https://img.shields.io/badge/Status-Completed-success)](https://www.coursera.org/learn/machine-learning-with-python)

> ✅ Final module of **IBM’s Machine Learning with Python (Coursera)** — combining all concepts learned to build and evaluate end-to-end classification models using pipelines, cross-validation, and hyperparameter tuning.

---

## 🧠 Overview

This repository includes my **final projects** and **notes** from **Module 6: Final Project & Course Summary**.  
In this module, I applied the complete machine learning workflow — from data cleaning to model evaluation — on two real-world datasets:  
the **Titanic Survival Prediction** (practice) and the **Australian Weather Forecast** (final project).

---

## 🎯 Learning Objectives

- Apply **end-to-end ML pipelines** for supervised classification  
- Perform **feature engineering**, **data cleaning**, and **transformation**  
- Compare models like **Random Forest** and **Logistic Regression**  
- Optimize models using **GridSearchCV** and **StratifiedKFold Cross-Validation**  
- Interpret **feature importance**, coefficients, and evaluation metrics  
- Demonstrate a full understanding of **data-driven model development**  

---

## 💻 Projects Completed

| 🧾 Project | 📊 Description | 🧮 Key Techniques |
|-------------|----------------|------------------|
| `Titanic_Survival_Prediction.ipynb` | Practice project predicting passenger survival | Pipelines, Logistic Regression, Random Forest, Cross-validation |
| `FinalProject_AUSWeather.ipynb` | Final project predicting rainfall using Australian weather data | Feature engineering, RandomForest, Logistic Regression, GridSearchCV |

---

## 📂 Repository Contents

| File | Description |
|------|--------------|
| `Machine_Learning_Notes_M6.pdf` | My detailed notes covering project workflow, data prep, model tuning, and final insights. |
| `Titanic_Survival_Prediction.ipynb` | Classification project predicting survival outcomes on Titanic dataset. |
| `FinalProject_AUSWeather.ipynb` | Rainfall prediction classifier with Random Forest and Logistic Regression comparison. |

---

## ⚙️ Tools & Libraries

- **Language:** Python 🐍  
- **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  
- **Methods Used:**  
  - Pipelines (`Pipeline`, `ColumnTransformer`)  
  - Imputation (`SimpleImputer`), Scaling (`StandardScaler`), One-Hot Encoding  
  - Model Selection (`GridSearchCV`, `StratifiedKFold`)  
  - Evaluation Metrics (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ConfusionMatrix`)  
  - Feature Engineering (Season extraction, categorical encoding)  

---

## 📊 Key Insights

### 🧩 Titanic Survival Prediction
- Achieved ~83% accuracy using both **Random Forest** and **Logistic Regression**  
- Key predictors: `sex_female`, `class_Third`, `age`, `fare`, `who_man`  
- Pipelines simplified preprocessing and hyperparameter tuning  
- Logistic Regression offered interpretability, Random Forest captured nonlinear patterns  

### 🌦️ Rainfall Prediction (AUSWeather)
- Built rainfall prediction model for **Melbourne region** using weather features  
- Used **RandomForestClassifier** and **LogisticRegression** pipelines  
- Engineered **seasonal features** from dates  
- Best accuracy: ~84% (Random Forest)  
- Feature importance showed `Humidity3pm`, `RainYesterday`, and `WindGustSpeed` as top predictors  

---

## 🧩 Concepts Reinforced
- Data preprocessing and transformation  
- Model selection and hyperparameter tuning  
- Evaluation metrics beyond accuracy  
- Importance of **cross-validation** and **data stratification**  
- Avoiding **data leakage** using well-designed pipelines  

---

## 🧠 Course Summary

This final module concludes the **IBM Machine Learning with Python** course, integrating all six modules:  
1. **Introduction to ML & Tools**  
2. **Regression Techniques**  
3. **Classification & Decision Trees**  
4. **Clustering & PCA**  
5. **Model Evaluation & Regularization**  
6. **End-to-End ML Projects**

> 🏁 These projects demonstrate proficiency in applying ML workflows, model comparison, and pipeline automation.

---

## 👨‍💻 Author

**Sandeep Maurya**  
🎓Aspiring Machine Learning Engineer  
📧 [isandeeep06@gmail.com](mailto:isandeeep06@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/sandeepmaurya-datascientist/) 

---

## 🧩 Acknowledgement

- **Course:** [Machine Learning with Python – IBM (Coursera)](https://www.coursera.org/learn/machine-learning-with-python/home/module/6)  
- **Instructor:** IBM Skills Network  

---

## 🌟 Support

If you found this repository helpful:  
⭐ **Star this repo** — it motivates me to share more.  
📢 **Share it** with other learners.  
💬 **Feedback or suggestions?** Open an issue anytime!  

---

> _“The true power of machine learning lies not just in algorithms, but in building a pipeline that delivers consistent, data-driven insights.”_
