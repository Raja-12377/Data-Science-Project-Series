# Data-Science-Project-Series
Nexus info data science Intern projects 

Project 1 - Documentation : **Stock Market Prediction**

Introduction

This project focuses on predicting stock prices using machine learning techniques. The dataset contains historical stock data with features like open, high, low, close, and volume. The goal is to build a regression model to forecast future stock values.

Exploratory Data Analysis (EDA) : 

The dataset is loaded and essential columns ('date', 'close', 'high', 'low', 'open', 'volume') are extracted.
Duplicates and null values are checked, and descriptive statistics are calculated.
Correlation between variables is visualized using a heatmap and pair plots.
![SP corr](https://github.com/Raja-12377/Data-Science-Project-Series/assets/93259031/293be024-a649-42af-9676-6849cfe3bc86)

Box plots are used to analyze the distribution of stock prices.

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

Conclusion:

The Linear Regression model demonstrates high accuracy in predicting stock prices based on the historical data. The project successfully showcases the application of machine learning in stock market prediction.
This documentation provides a clear overview of the project's approach, methodologies, and insights gained from analyzing the stock market dataset and building a predictive model for stock price forecasting.
