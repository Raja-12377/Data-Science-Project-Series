# Data-Science-Project-Series
Nexus info data science Intern projects 

Project 1 - Documentation : **Stock Market Prediction**

Introduction:

This project focuses on predicting stock prices using machine learning techniques. The dataset contains historical stock data with features like open, high, low, close, and volume. The goal is to build a regression model to forecast future stock values.

Exploratory Data Analysis (EDA) : 

The dataset is loaded and essential columns ('date', 'close', 'high', 'low', 'open', 'volume') are extracted.
Duplicates and null values are checked, and descriptive statistics are calculated.
Correlation between variables is visualized using a heatmap and pair plots.Box plots and line plots also are used to analyze the distribution of stock prices.

![SP corr](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/293be024-a649-42af-9676-6849cfe3bc86)


![SP line](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/02b22c67-8d5b-41de-ae8d-fdb5b63e18ff)


Data Preparation:

The dataset is split into independent variables (x) and dependent variable (y).
The data is further divided into training and testing sets (80% training, 20% testing) using train_test_split.

Predictive Modeling:

Linear Regression model is trained on the training data.
The model is used to make predictions on the test data.
Model coefficients, intercept, and confidence score are calculated.
Evaluation metrics like Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error are computed.

Model Evaluation:

The Mean Absolute Error is 0.315, Mean Squared Error is 1.201, and Root Mean Squared Error is approximately 0.561.
The model achieves an accuracy of 98.51%.

![SP pred](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/1b73c5c8-f64c-40a1-9558-29d1f456b048)

Conclusion:

The Linear Regression model demonstrates high accuracy in predicting stock prices based on the historical data. The project successfully showcases the application of machine learning in stock market prediction.

This documentation provides a clear overview of the project's approach, methodologies, and insights gained from analyzing the stock market dataset and building a predictive model for stock price forecasting.

--------------------------------------------------

Project 2 - Documentation: **Breast Cancer Prediction**

Introduction:

This project focuses on predicting breast cancer diagnosis using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to preprocess the data, select relevant features, and implement a Support Vector Machine (SVM) model for classifying tumors as malignant or benign.

Data Preprocessing:

The dataset is loaded and checked for missing values.
Categorical values in the 'diagnosis' column are encoded using LabelEncoder to convert them to numerical values.
Correlation analysis and visualization are performed to understand the relationships between features.

![Bp_corr](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/f28729c4-9958-495a-9d9d-df4001d068f7)

![Bp_pair1](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/020fd0a5-fcca-49b8-919e-9ddeab2dee6d)
![bp_pair2](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/55dbb83e-f597-46a0-a430-b15b9e81c45a)

Feature Selection and Engineering:

Relevant features are selected for breast cancer prediction.
StandardScaler is used to scale the feature data for better model performance.

Machine Learning Model (SVM):

A Support Vector Machine (SVM) model with a linear kernel is implemented for tumor classification.
The model is trained and evaluated on the Breast Cancer dataset.

Challenges Faced:

Error1: Could not convert string to float: 'M'
Solution: Categorical values are encoded using LabelEncoder to resolve the error.
Error2: Unknown label type: 'unknown'
Solution: LabelEncoder is used to transform the target labels to numerical values.

Model Evaluation:

The SVM model achieves an accuracy of 96.49% in predicting breast cancer diagnosis.
The classification report shows high precision, recall, and F1-score for both malignant and benign tumor predictions.

![bp_ac](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/defee1e3-db2e-4e42-bde6-2fb297c721f4)

Conclusion:

The SVM model demonstrates strong performance in classifying breast tumors as malignant or benign based on the dataset features. The project successfully showcases the application of machine learning in breast cancer prediction with high accuracy.

This documentation provides a clear overview of the project's approach, methodologies, challenges faced, model implementation, and performance evaluation in predicting breast cancer diagnosis using machine learning techniques.


--------------------------------------------------

Project 3 - Documentation: **Sentiment Analysis **

Data Loading and Exploration
Successfully read the dataset using different encodings.
Displayed basic information about the dataset structure and features.

Key Variables Identification
   
Identified key variables such as text content and sentiment labels.
Displayed the first few rows of the dataset to understand its structure.

Text Preprocessing
   
Implemented text preprocessing tasks including lowercasing, special character removal, tokenization, stop words removal, and lemmatization.
Created a function to preprocess text data and applied it to the 'text' column.

Exploratory Data Analysis (EDA)
   
Analyzed the distribution of sentiment labels in the dataset.
Visualized the sentiment label distribution using count plot and pie chart.

![sentiment](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/681d5672-12cc-4dae-a30d-a52a22916128)



Text Vectorization (TF-IDF)
    
Utilized TF-IDF Vectorizer to convert preprocessed text into numerical vectors.
Displayed the first few rows of the TF-IDF DataFrame for visualization.Text Vectorization (TF-IDF)
    
Utilized TF-IDF Vectorizer to convert preprocessed text into numerical vectors.
Displayed the first few rows of the TF-IDF DataFrame for visualization.

Text Vectorization (TF-IDF)
    
Utilized TF-IDF Vectorizer to convert preprocessed text into numerical vectors.
Displayed the first few rows of the TF-IDF DataFrame for visualization.

These points cover the data loading, preprocessing, exploratory analysis, and text vectorization stages of the Sentiment Analysis project.

Model Training and Evaluation:

Model Used: Support Vector Machine (SVM) with a linear kernel.
Accuracy: 70.17%
Precision: 71.16%
Recall: 70.17%
F1 Score: 70.11%

Classification Report:

Precision, recall, F1-score, and support for each class.

Cross-Validation Scores:

Mean Cross-Validation Score: 69.36%
Cross-Validation Scores: [68.98%, 69.39%, 70.34%, 68.82%, 69.29%]

Feature Importance:

Top 10 Features for Each Class:
Negative: ['aaaah']
Neutral: ['aaaah']
Positive: ['aaaah']

Confusion Matrix:

[[ 928  543   91]
 [ 248 1719  263]
 [  63  432 1210]]

Precision and Recall for Each Class:

Class 'Negative': Precision = 74.90%, Recall = 59.41%
Class 'Neutral': Precision = 63.81%, Recall = 77.09%
Class 'Positive': Precision = 77.37%, Recall = 70.97%

Future Deployment:

Future Deployment: Plan to deploy the project on a website and host it.
GitHub Update: Will update the deployment on GitHub for further access and development.

Conclusion:

Summarize the project outcomes, model performance, and potential future enhancements.
Highlight the significance of the sentiment analysis project and its potential real-world applications.

