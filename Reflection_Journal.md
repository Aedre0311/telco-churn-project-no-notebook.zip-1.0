# Reflection Journal

**Name:** Andre Ellis
**Course:** ITAI1371
**Project:** Telco Customer Churn: Exploratory Data Analysis and Data Pre-processing (solo project)

## Overview

For this project I took the Telco Customer Churn dataset from Kaggle (7,043 customers, 21 columns) and prepared it for a machine learning problem: predicting which customers will churn. Before this project I had done the labs one step at a time. Doing the whole pipeline on one dataset showed me how the steps depend on each other, and that the order I do them in matters as much as the techniques themselves.

## What I learned from the EDA

The first thing the EDA showed was that the target is imbalanced. About 73.5% of customers stayed and 26.5% churned, so a model that always predicts "No churn" would be right nearly three quarters of the time without learning anything. That changed how I think about evaluation: accuracy alone would be misleading for this dataset.

The category plots were the most useful part. Customers on month-to-month contracts churned at 42.9%, compared with 10.9% for one-year and 3.0% for two-year contracts. Customers paying by electronic check churned at 45.6%, fiber optic customers at 41.8%, and customers without tech support at 41.9%. Among the numeric features, tenure had the strongest relationship with churn (r = -0.34): newer customers leave far more often. I also saw that TotalCharges and tenure are strongly correlated (r = 0.83), which makes sense because total charges are roughly tenure multiplied by monthly charges.

## Handling missing values

When I first checked for nulls, pandas reported none. The problem was only visible once I looked at the data types: TotalCharges was stored as text, because 11 rows contained a blank space instead of a number. These were hidden missing values. All 11 belonged to customers with a tenure of zero, meaning new customers who had not been billed yet. I converted the column with `pd.to_numeric(errors="coerce")`, which turned the blanks into real NaNs, and filled them with the median of the training set.

I used the median because TotalCharges is right-skewed and the median is not pulled around by large values. I also learned to calculate that median from the training data only. If I had used the whole dataset, information from the test set would have leaked into my preprocessing.

## Feature engineering, encoding, normalization and scaling

I created six new features: tenure groups, average monthly spend, the change between current and average monthly charges, the number of subscribed services, a flag for customers who have both online security and tech support, and a flag for automatic payment methods. I chose them because the EDA suggested churn depends on tenure, on how many services a customer uses, and on how they pay.

For encoding I used three approaches depending on the column. Yes/No columns became 0/1. Contract and tenure group have a natural order, so I used ordinal encoding to keep it. Nominal columns such as internet service and payment method got one-hot encoding, because giving them numbers like 1, 2, 3 would imply an order that does not exist. After encoding, the dataset had 45 numeric columns and no text columns.

TotalCharges had a skew of 0.95, so I applied a log transform. This brought the absolute skew down to 0.73, although the result leans slightly the other way, so it is improved but not perfectly symmetric. I then scaled the numeric columns with StandardScaler, fitted on the training data and applied to the test data, so every numeric feature has a mean near 0 and a standard deviation of 1. This matters for distance-based and gradient-based models.

## Splitting first and avoiding data leakage

The most important lesson of the project was to split the data before doing anything else. I split 70/30 in code, stratified on the target so both sets kept the same 26.5% churn rate. The median, the scaler parameters and the class balancing all used only the training set. The test set was only transformed, never fitted on, and I did not balance it. If I had balanced the test set, my final metrics would describe an artificial population instead of the real mix of customers.

## Class balancing

The training set started with 3,622 customers who stayed and 1,308 who churned. I used SMOTE, which creates new minority-class examples by interpolating between a churner and one of its nearest churner neighbors, so the two classes ended at 3,622 each. The benefit is that models cannot succeed by ignoring the minority class. The risk is that the synthetic customers are not real people, and a model can overfit to them. That is why I will judge models on the untouched test set.

## Challenges

The hardest part was not a statistical technique but getting the notebook to run reliably. I ran into errors from file paths, missing packages and a modified setup cell, and because every cell depends on the ones before it, one failure early on made every later cell show an error. I learned to look for the first error rather than the last, and to make the notebook look for the data file in several places and install missing packages itself. I also learned to be careful about letting tools change code, because an edit I did not review caused a problem.

## Plans for the finals

For the finals I will train and compare Logistic Regression, Random Forest and a gradient boosting model, and possibly KNN or an SVM, since the scaled features suit them. I will tune them with cross-validation on the training set, applying the balancing step inside each fold so there is no leakage. I will evaluate on the untouched test set using precision, recall, F1 for the churn class and ROC-AUC, with a confusion matrix. Recall on churners matters most, because missing a customer who is about to leave costs more than offering a discount to someone who would have stayed. Finally, I will look at feature importance to check that the models rely on the things the EDA pointed to, such as contract type and tenure.
