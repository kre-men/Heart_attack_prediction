Heart Attack Prediction Using Machine Learning Models


This project aims to predict the likelihood of a heart attack using patient health data. By employing multiple machine learning models, the project provides insights into the risk of heart disease, aiding early detection and prevention. The dataset is preprocessed to remove missing, duplicated values, and scaled for optimal model performance.

Key Features:
Data Preprocessing:

Handle missing and duplicate data.
Scale features using StandardScaler and MinMaxScaler for improved model performance.
Explore the dataset using descriptive statistics and visualizations (e.g., histograms).
Machine Learning Models:

Logistic Regression (basic and tuned versions using GridSearchCV).
Random Forest Classifier for ensemble-based predictions.
XGBoost for gradient-boosting classification.
Multi-layer Perceptron (MLP) for neural network-based predictions.
Support Vector Machines (SVM) for robust classification.
Model Comparison:

Evaluate models using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
Compare model performances using visualizations (e.g., bar plots of accuracy scores).
Performance Insights:

Logistic Regression achieved ~89% accuracy after hyperparameter tuning.
Random Forest and XGBoost provided competitive performance with robust feature handling.
MLP and SVM demonstrated strong classification capabilities for the dataset.
Technologies and Libraries:
Python: The core language for implementation.
Pandas and NumPy: For data manipulation and analysis.
Matplotlib and Seaborn: For data visualization and plotting.
Scikit-learn: For preprocessing, model training, and evaluation.
XGBoost: For gradient boosting.
Keras: For MLP implementation.
Applications:
This project can be applied in healthcare analytics to:

Predict heart attack risks based on patient data.
Support medical professionals in early diagnosis and personalized treatment plans.
Enhance preventive healthcare strategies using data-driven insights.


Future Improvements:
Use a larger and more diverse dataset for improved generalization.
Explore deep learning approaches for feature extraction and classification.
Integrate the model into a web application for real-time predictions.
