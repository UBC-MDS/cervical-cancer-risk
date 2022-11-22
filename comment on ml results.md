# Some Short Comments on Machine Learning Results

## Feature Selection and Preprocessing

When both binary and numeric columns for the same feature exist, only use 1 of them.

Only 1 STD feature (STDs:condylomatosis) is included as others are too imbalanced.

For numeric features, missing values are imputed with their medians. For binary features, as Waiel suggested, missing values will be coded as a new category (thus one-hot encoder is used).

All the selections right now are tentative and subject to change.

### Model Performance

While 13.49% of the observations are labelled as positive, a minimum precision of 14% should be achieved.

KNN: Very poor performance. Decision function is not accessible. No need to further explore this.

SVC: Not bad.

RFC: Not as good as SVC.

Naive Bayes: Not bad.

Logistic Regression: Similar to naive Bayes.

While it is hard to determine the sweet point on the precision-recall curve, maybe we can fix a required precision score so that we can compare the model's recall score on the same basis? For example, using 28% precision as the minimum precision so that when a subject is predicted as with cervical cancer, their posterior probability of having cancer is doubled (from 14% to 28%).

_I am not sure whether the logic above is valid in the case of machine learning as probability can be misleading in this context._