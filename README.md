# Zillow-Houses
This repository contains full machine learning pipeline of the Zillow Houses competition on Kaggle platform.


Pipeline is consists of 5 general steps
1) Exploratory Data Analysis (Univariate, Bivariate, Hypothesis testing, Confident Interals)
2) Missing values (different advanced and not strategies to impute: MICE algo with the using of gradient boosting, lightgbm etc.)
3) Duplicate checking
4) Advanced Anomaly Detection (models such as KNN, Isolation Forests, and final detector witch aggregates results from base models - SUOD)
5) Multicollinearity problem solving
6) Feature Engineering
7) Feature Transformation of some features with hypothesis testing on it (fitting distributions with some statistical tests)
8) Advanced Feature Selection and not - Recursive Feature Elimination with cross-validation on different tree-based models such as Gradient Boosting, Random Forests etc) and of course Lasso with L1-norm, Feature Importances of trees and combine them into one algorithm witch takes in account all the above method
9) Modeling (different regression models, fine-tuning, learning curves, validation curves, Residuals Analysis etc.). Later, i wan't to use some stacking stategies on boosted trees and some NN models
10) Results analysis: best model selection with the using of confident intervals and different non-parametric statistical tests etc.

This solution also contains custom preprocessing pipeline witch automaticly can do 2-8 steps ( all in :) )
