<img src="Resources/Images/element5-digital-OyCl7Y4y0Bk-unsplash.png" align="right"/>

# `Columbia Engineering FinTech Bootcamp 2023-06 Cohort`

## `Project: Module 12 Supervised-Learning-Challenge`

## Background
> Credit risk poses a classification problem that is inherently imbalanced, as healthy loans easily outnumber risky loans.

## Purpose
> In this Challenge, we’ll use various techniques to train and evaluate models that are characterized by imbalanced classes. We'll use an historical dataset of lending activity, sourced from a peer-to-peer lending services company, to build two models designed to predict the creditworthiness of borrowers:
1. A logistic regression model using the original source data unaltered; and
2. An alternative logistic regression using source data resampled to mitigate imbalanced loan health class labels.

## Table of Contents
* [Background](#background)
* [Purpose](#purpose)
* [Overview of the Analysis](#overview-of-the-analysis)
    * [Results](#results)
    * [Summary](#summary)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

## Technologies Used
- Python Version 3.10.12
- Jupyter Notebook within VS Code IDE
- README markdown

## Features

## Screenshots
- [Refer to General Information Section](#general-information)

## Setup
- GitHub Repository
    - name: 'unsupervised-learning-challenge'
    - location: uploaded to Bootcamp homework submission online portal and available publicly at:
        - [GitHub Repository](git@github.com:boz-tcm/unsupervised-learning-challenge.git)
- Python Standard Library (Version 3.10.12)
- Python Libraries and Modules:
    - holoviews
    - hvplot.pandas
    - IPython.display and Image
    - matplotlib and pyplot
    - pandas
    - pandas.tseries.offsets and DateOffset
    - pathlib and Path
    - phantomjs
    - pillow
    - pydotplus
    - selenium
    - sklearn.ensemble and AdaBoostClassifier
    - sklearn.metrics and classification_report
    - sklearn.preprocessing and StandardScaler
    - sklearn.svm and SVC
    - sklearn and tree
    - tensorflow
    - tensorflow-metal
    - Installation packages reference for virtual environment .venv on Mac Silicon (M1), including tensorflow-metal:
        - https://developer.apple.com/metal/tensorflow-plugin/
    
- Jupyter Notebook(s):
    - [machine_learning_trading_bot_baseline.ipynb](machine_learning_trading_bot_baseline.ipynb)
    - [machine_learning_trading_bot_modified_training_period.ipynb](machine_learning_trading_bot_modified_training_period.ipynb)
    - [machine_learning_trading_bot_modified_strategy_windows.ipynb](machine_learning_trading_bot_modified_strategy_windows.ipynb)
    - [machine_learning_trading_bot_modified_parameters_combined.ipynb](machine_learning_trading_bot_modified_parameters_combined.ipynb)
- Data
    - [emerging_markets_ohlcv.csv](Resources/emerging_markets_ohlcv.csv)
- Images
    - location: [Images](Images)

## Usage
The scripts are run in the project's four Jupyter Notebooks cited in both [General Instructions](#general-instructions) and [Setup](#setup), located within the 'algorithmic-trading-challenge' repository's root directory, executed using the Notebook environments' command 'Run All Cells...'.

## Project Status
Project is: _complete

## Room for Improvement
Room for improvement: _

To do: _

## Acknowledgements

## Contact
Created by Todd C. Meier, tmeier@bozcompany.com - feel free to contact me!

<!-- ## License --> All rights reserved.