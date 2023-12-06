<img src="Resources/Images/element5-digital-OyCl7Y4y0Bk-unsplash.png" align="right"/>

# `Columbia Engineering FinTech Bootcamp 2023-06 Cohort`

## `Project: Module 12 Supervised-Learning-Challenge`

## Background
> Credit risk poses a classification problem that is inherently imbalanced, as healthy loans easily outnumber risky loans.

## Purpose
> In this Challenge, we’ll use various techniques to train and evaluate models that are characterized by imbalanced classes. We'll use an historical dataset of lending activity, sourced from a peer-to-peer lending services company, to build two models designed to predict the creditworthiness of borrowers:
1. A logistic regression model using original, unaltered raw source data; and
2. An alternative logistic regression model using the original source data but resampled to mitigate imbalanced loan health classification labels observed in the original dataset.

## Table of Contents
* [Background](#background)
* [Purpose](#purpose)
* [Overview of the Analysis](#overview-of-the-analysis)
* [Results](#results)
* [Summary](#summary)
* [Technologies Used](#technologies-used)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->

## Overview of the Analysis

- ### Purpose
    The purpose of this analysis was to construct, evaluate, and recommend an appropriate model for the prediction of borrower creditworthiness in peer-to-peer lending.  
    
- ### The Financial Data
    We used an historical dataset of peer-to-peer lending activity, sourced from a lending services company, to build two alternative logistic regression models designed to predict the creditworthiness of borrowers in the peer-to-peer lending domain.

    Our historical financial dataset consisted of 77,536 individual loans, each of which had been assigned a binary classification loan health status (`loan_status`) of either 'healthy' or 'high risk of default', numerically encoded within our dataset as 0 or 1, respectively.  Loan health status represented our target, or the dependent variable that we sought to predict, or explain, as proxy for borrower creditworthiness.
    
    The peer-to-peer lending dataset contained 7 features, or independent variables, to investigate as potential explanatory, or predictor, variables for loan health.  Like the target variable, all predictor variables were either numerical in origin or numerically pre-encoded by the peer-to-peer lending company prior to our receipt of the dataset.

    The 7 predictor variables included the following (listed below as 'descriptor: `feature variable name`' pairs):
    - Loan Size ($): `loan_size`
    - Loan Interest Rate (%): `interest_rate`
    - Borrower's Annual Income ($): `borrower_income`
    - Debt-to-Income Ratio: `debt_to_income` (the only dataset feature that may be considered standardized)
    - Number of Accounts: `num_of_accounts`
    - Number of Loan Derogatory Marks: `derogatory_marks`
    - Total Debt: `total_debt`

    As a supplementary bonus to our analysis, we not only evaluate the power of our models and the explanatory variables, collectively, to predict loan health status, we also attempt to compare the power of *each* explanatory variable to *individually* explain, or predict, loan health.

    Although all of our raw data were numerically provided, and therefore did not require encoding, such as One-Hot or Label encoding routines, we nevertheless recommend exploring, in a future pre-processing exercise, standardization of the numerical feature fields, using such routines as StandardScalar(), for roughly normally distributed fields, MinMaxScaler, for non-normally distributed fields, and MaxAbsScaler(), for sparse, non-normally distributed fields.

    When evaluating our peer-to-peer lending dataset, we observed that the vast majority of loans were considered healthy (96.8%; n = 75,036), while a slim minority were labeled at high risk of default (3.2%, n = 2500).  In practice, such observation of an imbalanced target class is common in financial lending, credit card fraud detection, and email spam detection, which are characterized by a high frequency of observations in the majority label (e.g., performing loans), and relatively few observations in the minority label (e.g., defaulted loans).  Unless accounted and corrected for, this imbalance will lead to bias in our models, and potentially inaccurate conclusions, reflecting overwhelming influence of the majority class label.  Imbalance is particularly concerning when the minority label is what we seek to predict, such as in our case, where we are attempting to predict loans at high risk of default (minority label '1', or 'True' in our classification schema).

- ### The Models
    We built and assessed two alternative logistic regression models to predict the creditworthiness of borrowers in the peer-to-peer lending domain.
    - Model 1: Logistic regression model fit to the originally provided dataset as-is.
    - Model 2: Logistic regression model fit to a resampled version of our original dataset, with the intention of correcting for an imbalanced dataset.

    According to Pindyck and Rubinfeld, while a traditional linear regression model may be applied to datasets where the features, or explanatory variables, are binary, application of linear regression is more complex when the target, or dependent, variable is binary, as in our case, where the features are either numerically continuous or binary, and the target variable, loan health, that we seek to predict is binary, either healthy ('0') or high risk of default status ('1') (1998).

    Generally, there are three *linear* binary outcome models that are available to us in our situation: the linear probability model, the probit model, and the logit, or logistic model.
    These models each assume a linear relationship exists between the features or a transformation of the features and the binary target variable, i.e., loan health in a high risk of default state ('1') rather than the alternative healthy state ('0').  The three models subsequently differ based on their assumptions about the probabilistic nature of the binary classification or decision process.
    - The regression equation for the linear probability model describes the *probability* that the binary outcome, such as a high risk of default status, will be observed based on the explanatory features.  However, the problem with the linear probability model for prediction is that while it is consistent and unbiased, it does not guarantee that predictions will lie between 0 and 1, or probabilities between 0 and 100%.
    - To address the primary limitation of the linear probability model, the model can be transformed so that predictions lie in the 0 to 1 interval for all features and all combinations of feature values.  The transformation translates all features, which may range in value over the entire number line, to a probability that ranges from 0 to 1.  This transformation is the application of the *cumulative* probability function to the regression problem.  While many cumulative probability functions may be used, in practice two common such functions are the normal cumulative distribution function and the logistic cumulative distribution function, which define respectively the probit probability model and the logit, or logistic, probability model.

- Reference:
    - Pindyck, Robert S. and Daniel L. Rubinfeld, "Econometric Models and Economic Forecasts", 1998, 4th Ed, McGraw-Hill, pp 298-333.

- ### The Modeling

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results of the Analysis

- ### Machine Learning Model 1

- ### Machine Learning Model 2

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

- ## Summary of the Analysis

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

## Technologies Used
- Python Version 3.10.12
- Jupyter Notebook within VS Code IDE
- README markdown

## Screenshots

## Setup
- GitHub Repository
    - name: 'supervised-learning-challenge'
    - location: uploaded to Bootcamp homework submission online portal and available publicly at:
        - [GitHub Repository](https://github.com/boz-tcm/supervised-learning-challenge.git)
- Python Standard Library Version 3.10.12
- Python Libraries and Modules:
    - holoviews
    - hvplot.pandas
    - imblearn.over_sampling and RandomOverSampler
    - matplotlib and pyplot
    - numpy
    - pandas
    - pathlib and Path
    - sklearn.linear_model and LogisticRegression
    - sklearn.metrics accuracy_score, balanced_accuracy_score, classification_report, and confusion_matrix
    - sklearn.model_selection and train_test_split
    - sklearn.preprocessing and StandardScaler
        > *Note that in this activity we did not end up having to preprocess, scale, or encode data in our original and resampled datasets.
 
    
- Jupyter Notebook(s):
    - [credit_risk_resampling.ipynb](credit_risk_resampling.ipynb)
- Data
    - [lending_data.csv](Resources/Data/lending_data.csv)
- Images
    - location: [Images](Images)
- References:
    - `Academic paper describing IBA measure:` ["Index of Balanced Accuracy: A Performance Measure for Skewed Class Distributions," Garcia et al. ](Resources/References/61392839.pdf)

## Usage
The scripts are run in the project's Jupyter Notebook cited in [Setup](#setup), located within the 'supervised-learning-challenge' repository's root directory, executed using the Notebook environments' command 'Run All Cells...'.

## Project Status
Project is: _complete

## Room for Improvement
Room for improvement: _

To do: _

## Acknowledgements

## Contact
Created by Todd C. Meier, tmeier@bozcompany.com - feel free to contact me!

<!-- ## License --> All rights reserved.