# Podcast_listening_time_prediction
## 🎧 Podcast Listen Time Prediction (Kaggle Competition)

This project predicts how long users will listen to podcasts using advanced machine learning techniques. It is built using `XGBoost` and includes cross-validation, feature engineering, and hyperparameter tuning.

## 📊 Problem Statement
The goal is to predict **listening time (in seconds)** based on various features such as user behavior, podcast content, and playback speed.

This is part of a Kaggle competition: [https://www.kaggle.com/code/abhiii97/podcast-listen-time-prediction]

---

## 🔧 Tools & Libraries Used
- Python
- XGBoost
- Scikit-learn
- Pandas & NumPy
- Seaborn & Matplotlib
- Jupyter Notebook

---

## 🚀 Workflow

### 1. Data Cleaning & Preprocessing
- Handled missing values
- Converted date/time features
- Processed categorical variables

### 2. Feature Engineering
- Extracted day, month, hour from timestamps
- Calculated playback duration
- Created interaction features (e.g. `speed × duration`)

### 3. Exploratory Data Analysis
- Visualized distributions
- Checked correlation with listening time using heatmaps

### 4. Modeling with XGBoost
- Implemented `XGBRegressor` with advanced tuning
- Used GPU acceleration (`tree_method='gpu_hist'`)
- Applied `RandomizedSearchCV` with 5-fold cross-validation

### 5. Evaluation
- Scoring metric: **Root Mean Squared Error (RMSE)**
- Achieved RMSE: **~12.89** (aiming for <12)


