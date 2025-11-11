# AIFUL-IIT Mandi AiHack 2025 - 1st Place Solution

This repository contains the 1st place winning solution for the AiHack 2025 machine learning hackathon, hosted by AIFUL Corporation and IIT Mandi.

### Our Team
* Gaurav (Lead)
* Arman
* Jatin
* Anirudh

---

## Result

* **Rank:** 1st Place
* **Final AUC Score:** 0.6787960

## Project Overview

Credit scoring is a method used by financial institutions to evaluate the creditworthiness of a borrower. It predicts the likelihood that a customer will repay a loan on time, helping lenders make informed decisions.

Credit scoring is important in finance because it reduces the risk of loan defaults, ensures better allocation of credit, and helps banks and lenders maintain financial stability while serving responsible borrowers efficiently. Especially in unsecured lending, where no collateral is taken from the customer.

## The Goal

The goal of this hackathon was to build a predictive model using the provided unsecured loan dataset. The goal was to predict the probability of default for the customer, enabling smarter lending decisions and effective risk management.

## Dataset

The dataset for this competition was provided by AIFUL Corporation. Due to a confidentiality affidavit we signed, the data cannot be shared and is not included in this repository.

---

## Our Solution

Our winning strategy was based on two key pillars: (1) deep feature engineering and (2) using a single, powerful, and explainable CatBoost model. We did not use an ensemble.

### 1. Initial Model Exploration
We first considered several baseline models, including Logistic Regression, Random Forest, XGBoost, and LightGBM. We wanted to see which model family would handle the data best.

### 2. Why a Single CatBoost Model?
We quickly saw that the dataset was full of important categorical features (like `JIS Address Code`, `Industry Type`, `Company Size Category`). We chose **CatBoost** as our *only* model for a few key reasons:
* **Native Categorical Handling:** It handles categorical data natively and very effectively, which saved us a lot of time on preprocessing.
* **Robustness:** It is less prone to overfitting on a dataset of this size compared to other gradient boosting models.
* **Built-in Explainability:** It has SHAP integration built-in, which was the key to our "business solution" goal.

### 3. The Real Win: Feature Engineering
Our 1st place finish was mainly due to deep feature engineering. We built over 100 new features. Our best ones were:
* **Financial Ratios:** We created many ratios, like `LOAN_TO_INCOME_RATIO` and `ACTUAL_DEBT_TO_INCOME_RATIO`.
* **"Honesty Checks":** We built features like `DEBT_DISCREPANCY_AMOUNT` to flag when an applicant's *declared* debt didn't match their *actual* debt.
* **Peer Group Analysis:** This was one of our most powerful ideas. We created features like `INCOME_RELATIVE_to_Industry Type` to measure an applicant's income *against the average* for their specific peer group.
* **New Categoricals:** We also created new categorical flags like `DEBT_LIAR_FLAG` and `AGE_BUCKET` to feed directly into CatBoost.

### 4. Hyperparameter Tuning
To get the final 1st place score, we didn't just use a default model. We used the **Optuna** library to run a Bayesian hyperparameter search. This helped us find the optimal settings for CatBoost (like `learning_rate`, `depth`, `l2_leaf_reg`) that maximized the AUC score.

### 5. The "Green Box" Model: A Business Solution
A high AUC score isn't enough. For a finance company like AIFUL, a "black box" model is a risk. Our solution was a **"green box" model** -one that is highly accurate *and* fully explainable.

We used CatBoost's built-in **SHAP integration** to:
* **Identify Key Risk Drivers:** We could see *exactly* which features (like our `DEBT_DISCREPANCY_AMOUNT`) were pushing a customer's risk score up or down.
* **Justify Decisions:** This allows the business to not only trust the model but also to explain a decision to a loan officer or even a regulator.
* **Find New Insights:** The SHAP values showed us new, non-obvious patterns in the data that AIFUL can use to improve its business logic.
