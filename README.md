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

- ### The Financial Data

- ### The Models

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
        - [GitHub Repository](git@github.com:boz-tcm/supervised-learning-challenge.git)
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