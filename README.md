# Cervical Cancer Prediction and Risk Stratification

* Authors: Samson Bakos, Morris Chan, Waiel Tinwala, Revathy Ponnambalam

A Data Analysis/ Machine Learning project developed for DSCI 522 in the UBC Master of Data Science Program

## Introduction

Our primary research question is: can we diagnose or stratify risk for cervical cancer based on lifestyle factors, sexual history, and comorbidities using a machine learning model? This question is naturally subdivided into risk level and diagnosis questions depending on the relative strengths of the models that prove effective for this data. Classifiers with hard predictions (i.e. k-NN) will be effective for diagnosis, while classifiers with probabilistic predictions (i.e. logistic regression) will be effective for predicting level of risk depending on computed prediction probabilities. 

This research question is important because of the importance of early diagnoses in cancer patients, where the timing of diagnosis can mean the difference between treatable and terminal illness, often with a small margin of error. Successful implementation of this project will not only add an additional, non invasive diagnostic tool (based solely upon medical records and subjective self-reported patient data rather than in-person examination), it will also help identify at-risk individuals who can be more closely monitored for cervical cancer before it develops and progresses.

This data is sourced from 'Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.'(2017) by Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. It was accessed through the UCI Machine Learning Repository, found here: https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29.
The data is composed of survey results and medical records for 858 female patients from 'Hospital Universitario de Caracas' in Caracas, Venezuela, alongside the results of four traditional diagnosis tests (i.e. biopsy). 

Our plan to analyze this data is to deploy and evaluate a number of classification models to establish which type of model will be most effective in classifying this data. Data will first be split into training and test sets. Initial EDA on training data will then include investigating the distribution of missing values (due to missing/ 'prefer not to say' survey responses), as well as visualizing distributions of each potential feature for both the positive and negative diagnosis class using bar charts and histograms to identify intially promising classification characteristics. Simple tables will also be prepared to investigate class imbalance in the dataset. 

Proposed models to be tested include Decision Trees, kNN, SVC, Naive Bayes, and Logistic Regression, though this scope may increase as the analysis proceeds depending on results. Models will be fit to training data and evaluated via cross-validation. Promising models will then be further tuned and evaluated by cross-validation, with a focus on maximizing recall/ minimizing type II error due to the dangers posed by false negatives in this diagnosis/ risk identification context, though still valuing f1/ precision to avoid an unacceptably excessive false positive rate. Successful models will be fit to the entire training set and scored against test data. Final results will be presented in the form of confusion matrices, PR Curves and ROC curves, with focus again being placed on recall and f1 score. with promising models being deemed fit for deployment. 

## License 

The materials here are licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license.

Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

## References

1. Fernandes, K., Cardoso, J.S., Fernandes, J. (2017). Transfer Learning with Partial Observability Applied to Cervical Cancer Screening. In: Alexandre, L., Salvador Sánchez, J., Rodrigues, J. (eds) Pattern Recognition and Image Analysis. IbPRIA 2017. Lecture Notes in Computer Science(), vol 10255. Springer, Cham. https://doi.org/10.1007/978-3-319-58838-4_27







