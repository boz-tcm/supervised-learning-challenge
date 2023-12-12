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

    When evaluating our peer-to-peer lending dataset, we observed that the vast majority of loans were considered healthy (normalized and raw value_counts, respectively: 96.8%; n = 75,036), while a slim minority were labeled at high risk of default (3.2%, n = 2500).  In practice, such observation of an imbalanced target class is common in financial lending, credit card fraud detection, and email spam detection, which are characterized by a high frequency of observations in the majority label (e.g., performing loans), and relatively few observations in the minority label (e.g., defaulted loans).  Left unaccounted and uncorrected, this degree of imbalance will lead to bias in our models, and potentially inaccurate conclusions, reflecting overwhelming influence of the majority class label.  Imbalance is particularly concerning when the minority label is what we seek to predict, such as in our case, where we are attempting to predict loans at high risk of default (minority label '1', or 'True' in our classification schema).

- ### The Models
    We built and assessed two alternative logistic regression models to predict the creditworthiness of borrowers in the peer-to-peer lending domain.
    - Model 1: Logistic regression model fit to the originally provided dataset as-is.
    - Model 2: Logistic regression model fit to a resampled version of our original dataset, with the intention of correcting for an imbalanced dataset.

    According to Pindyck and Rubinfeld, while a traditional linear regression model may be applied to datasets where the features, or explanatory variables, are binary, application of linear regression is more complex when the target, or dependent, variable is binary, as in our case, where the features are either numerically continuous or binary, and the target variable, loan health, that we seek to predict is binary, either healthy ('0') or high risk of default status ('1') (1998).

    Generally, there are three *linear* binary outcome models that are available to us in our situation: the linear probability model, the probit model, and the logit, or logistic model.
    These models each assume a linear relationship exists between the features or a transformation of the features and the binary target variable, i.e., loan health in a high risk of default state ('1') rather than the alternative healthy state ('0').  The three models subsequently differ based on their assumptions about the probabilistic nature of the binary classification or decision process.
    - The regression equation for the linear probability model describes the *probability* that the binary outcome, such as a high risk of default status, will be observed based on the explanatory features.  However, the problem with the linear probability model for prediction is that while it is consistent and unbiased, it does not guarantee that predictions will lie between 0 and 1, or probabilities between 0 and 100%.
    - To address the primary limitation of the linear probability model, the model can be transformed so that predictions lie in the 0 to 1 interval for all features and all combinations of feature values.  The transformation translates all features, which may range in value over the entire number line, to a probability that ranges from 0 to 1.  This transformation is the application of the *cumulative* probability function to the regression problem.  While many cumulative probability functions may be used, in practice two common such functions are the normal cumulative distribution function and the logistic cumulative distribution function, which define respectively the probit probability model and the logit, or logistic, probability model.
        - Pindyck and Rubinfeld argue that "[W]hile the probit model is more appealing than the linear probability model, it generally involves nonlinear maximum-likelihood estimation,"given the assumption of a normal distribution for the cumulative probability function transform.  "In addition, the theoretical justification for employing the probit model is somewhat limited" (1998).
        - In contrast to the probit model, the logit model assumes the logistic distribution for the cumulative probability distribution function transform.  Pindyck and Rubinfeld note that the logit and probit formulations are quite similar with respect to their cumulative probability functions, however the logistic distribution produces slightly fatter tails.  Given their similarities and because the logit model is easier to use computationally, presumably because its cumulative distribution function takes on a convenient algebraic form (specifically in the form of log-likelihood) when applying maximum likelihood estimation (MLE) to solve for the model's regression parameters, the logit model is often used in place of the probit model.
    - Our peer-to-peer borrower creditworthiness analysis allowed us to illustrate the practical benefits of applying logistic regression to model and predict our binomial outcome variable loan health status.  We describe our logistic modeling in the following section.

- ### The Modeling
Two logistic regression models were fit to two version of our peer-to-peer lending dataset.  Logistic Regression Model #1 was fit to the original, unaltered lending dataset, while Logistic Regression Model #2 was fit to a modified version of the lending dataset, where we accounted for a material class imbalance observed in the original dataset between healthy (majority class label, weight 96.8%) and high risk of default (minority class label, weight 3.2%) loans by randomly oversampling the minority class, resulting in a resampled, *balanced* dataset. For resampling, we relied on [scikit-learn's](https://imbalanced-learn.org/stable/) `imblearn.over_sampling` library function [RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler).

Alternative sampling options available to us were random and synthetic sampling, and within each type of sampling, over and undersampling the minority and majority classes, respectively.  Our chosen resampling method, *random oversampling*, randomly samples from the minority class, *duplicating* instances of the minority class until the number of minority class data points equal the number of majority class data points from the original dataset.  Note that this random oversampling methodology was applied to only the training segment portion (75%) of our original dataset, while the original testing portion (25%) was retained unaltered for prediction purposes in both models.  Training for Model #2 was thus based on 56,277 observations for both labels.  Because we do not create *new* instances in the minority label, rather only duplicates of the minority label when randomly oversampling, the methodology leads to underestimating the true variation in the minority label.  This resampling technique may therefore lead to overfitting the training data, which could be detected, all else equal, by observing better prediction performance on training data relative to testing.

For modeling purposes, our peer-to-peer lending dataset was split between train and test data using [scikit-learn's](https://scikit-learn.org/stable/) `sklearn.model_selection` library function [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), using the default 75%/25% train/test split and the stratify argument set to 'y' to conserve label proportion between both train and test datasets (moreover, data are shuffled, by default, prior to splitting).

Following the splitting of the original dataset into train and test datasets, scikit-learn's `sklearn.linear_model` library function [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) was used to fit logistic regression models to both the original train dataset and a randomly oversampled version of the train dataset to correct for material imbalance in the original dataset.  Because our peer-to-peer borrower creditworthiness analysis entails simple binomial classification, the logistic regression function's solver parameter was specified as 'liblinear' ("Library for Large Linear Classification"), as opposed to the function's default solver 'lbfgs' ("Limited-memory Broyden–Fletcher–Goldfarb–Shanno"), which is more flexible and appropriate for multinomial class problems, but not necessary in this case.

Once the models were fit to the two sets of training data, prediction performance and accuracy were evaluated based on the testing dataset.

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results of the Analysis

The Logistic Regression models were fit to two different training versions of the peer-to-peer lending dataset: the original training dataset as-is for Model #1 and a resampled version of the original training dataset, randomly oversampled on the minority 'high-risk of default' loan health label to yield a balanced training dataset for Model #2.

Balanced accuracy scores, along with precision and recall scores, were measured for the two logistic regression models to evaluate accuracy and performance in the context of a common testing dataset. 

- ### Machine Learning Model 1
    - Traditional model accuracy is the ratio of true predictions relative to all predictions made, or the ratio of true-positive plus true-negative predictions to total predictions.  Traditional accuracy for Model #1 was 0.993, or 99.3% accuracy.  However, a more accurate depiction of model accuracy for materially imbalanced testing datasets, in particular when the negative label dominates, as in our case, i.e., healthy loans, is scikit-learn's implementation of the balanced accuracy score, which seeks to correct bias in the traditional accuracy score for imbalanced datasets by calculating the *macro average* of the model's true positive rate and the true negative rate, or [(TPR + TNR) / 2], which is also [(sensitivity + specificity) / 2] (c.f. https://www.statology.org/balanced-accuracy).  A more accurate depiction of Model 1's accuracy is therefore the balanced accuracy score of 0.948, or 94.8%, which reveals a material overstatement in Model 1's accuracy had we relied on the traditional accuracy measure.
    - Precision Score for our Model #1 specification was 0.874, or 87.4%, as derived from our confusion matrix: 563 / (563 + 81) = True-Positive / (True-Positive + False-Positive).  Precision represents the proportion of true-positives predicted to all positives predicted (out of all positives predicted, what proportion was truly positive).  In other words, whenever Model #1 predicted a high-risk loan, how often was it correct?  In this case 87.4% of the time, where only 81 healthy loans, from a large pool of healthy loans, were incorrectly predicted, or classified, as at high risk of default. 
    - Recall Score for our Model #1 specification was 0.901, or 90.1%, as derived from our confusion matrix: 563 / (563 + 62) = True Positive Rate = Sensitivity = True-Positive / (True-Positive + False-Negative).  Recall represents the proportion of all true cases correctly predicted as true.  In other words, what proportion of high-risk loans were correctly predicted, or detected, as high risk?  In this case 90.1% of high-risk loans were correctly predicted by Model #1, with only 62 of the high-risk loans not detected by the model and incorrectly classified as healthy.  Given that we are more concerned about identifying high-risk loans than incorrectly classifying healthy loans and the economic cost of not detecting high-risk loans, recall in this case is more important to us than precision.
- ### Machine Learning Model 2
    - Balanced accuracy score for Model #2 was 0.996, or 99.6% = (TPR + TNR / 2) = [(623 / (623 + 2)) + (18668 / (18668 + 91))] / 2 = (0.9968 + 0.9951) / 2.  Traditional accuracy score of 0.995, or 99.5%, is nearly equivalent to balanced accuracy score, calculated as ((623 + 18668) / (623 + 18668 + 2 + 91)) = true predictions / all predictions.
    - Precision Score for Model #2 was 0.873, or 87.3% = 623 / (623 + 91).
    - Recall Score for Model #2 was 0.997, or 99.7% = 623 / (623 + 2).

    Note: Scores derived above from each model's confusion matrix reconcile to the respective classification report.

- ## Summary of the Analysis

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

- ## References
  
    - Breen, Richard, Kristian Karlson, Anders Holm (July 2018), "Interpreting and Understanding Logits, Probits, and Other Nonlinear Probability Models," *Annual Review of Sociology*, https://www.annualreviews.org/doi/10.1146/annurev-soc-073117-041429.
    - Brownlee, Jason (October 28, 2019), "A Gentle Introduction to Logistic Regression With Maximum Likelihood Estimation," https://machinelearningmastery.com/logistic-regression-with-maximum-likelihood-estimation.
    - Perraillon, Marcelo (2019), "Week 12: Linear Probability Models, Logistic and Probit", https://clas.ucdenver.edu/marcelo-perraillon/sites/default/files/attached-files/week_12_lpn_logit_0.pdf
    - Pindyck, Robert S. and Daniel L. Rubinfeld (1998), *Econometric Models and Economic Forecasts*, 4th Ed, McGraw-Hill, pp 298-333.

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