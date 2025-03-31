# AI-Powered-Heart-Disease-Prediction
AI-Powered Heart Disease Analysis: ML Evaluation &amp; Comparison

This repository implements an AI-powered approach to predict heart disease using various machine learning models. The project is built using Python and leverages popular libraries such as scikit-learn, XGBoost, Pandas, and NumPy.

Overview
The code in this repository performs the following steps:

Data Loading: Reads the heart disease dataset from a CSV file stored on Google Drive.

Data Preprocessing: Renames columns, removes records with invalid cholesterol values, and cleans outliers using Z-Score.

Feature Engineering: Applies one-hot encoding to categorical features and scales numeric features using MinMaxScaler.

Modeling: Defines multiple machine learning models including Logistic Regression, RandomForest, SVM, KNN, Decision Tree, ExtraTrees, and more.

Evaluation: Uses 5-fold cross-validation to evaluate model performance (accuracy, F1 score, and recall) and prints test set results.

Visualization: Displays the confusion matrix for the best performing model.

Requirements

Python 3.x

scikit-learn==1.3.2

xgboost

pandas==2.2.2

numpy==1.26.0

seaborn

matplotlib

scipy

Note: After installing/upgrading packages, restart the runtime in Google Colab if necessary.

How to Use
Clone the repository.

Open the notebook in Google Colab.

Ensure your Google Drive contains the dataset at drive/My Drive/heart_statlog_cleveland_hungary_final.csv.

Run the notebook cells sequentially to perform data preprocessing, modeling, and evaluation.

Results
The notebook displays evaluation metrics (accuracy, F1, and recall) for each model and a confusion matrix for the best performing model (ExtraTrees), allowing for a comprehensive comparison of different approaches in heart disease prediction.

License
This project is licensed under the MIT License.
