"""This script takes the trained models and returns the performance of them on the testing data set.

Usage:
model_testing.py --data_path=<data_path> --output_path=<output_path>

Options:
--data_path=<data_path>     Path to the data file (including the file name).
--output_path=<output_path> Desired path for the perfornace results returned.

Example:
python src/model_testing.py --data_path='data/processed/test.csv' --output_path='results'
"""

from docopt import docopt
import numpy as np
import pandas as pd
import os
from joblib import dump, load
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    PowerTransformer,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

opt = docopt(__doc__)


def better_metrics(y_test, y_hat):
    precision = precision_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat)
    f1 = f1_score(y_test, y_hat)
    auc = roc_auc_score(y_test, y_hat)
    metrics_dict = {"precision": precision, "recall": recall, "f1": f1, "auc": auc}
    return metrics_dict


def main(data_path, output_path):
    data_full = pd.read_csv(data_path)

    numeric_features = [
        "Age",
        "Number of sexual partners",
        "First sexual intercourse",
        "Num of pregnancies",
        "Hormonal Contraceptives (years)",
        "IUD (years)",
        "STDs (number)",
    ]

    binary_features = ["STDs:condylomatosis", "Smokes", "Dx:Cancer", "Dx:CIN", "Dx:HPV"]

    columns_tbc = numeric_features + binary_features

    X = data_full[columns_tbc]
    y = data_full["risk"]

    test_results = {}

    # Thresholds ---
    thresholds = pd.read_csv("src/thresholds-used.csv", index_col=0)
    thld_rfc = float(thresholds.loc["RFC"])
    thld_nb = float(thresholds.loc["NB"])
    thld_lsvc = float(thresholds.loc["LinearSVC"])

    # RFC ---
    pipe_rfc_opt = load("binary_files/pipe_rfc_opt.joblib")

    def rfc_with_threshold(pipe_rfc, X_test, threshold):
        proba = pipe_rfc.predict_proba(X_test)[:, 1]
        y_hat = proba > threshold
        return y_hat

    y_hat_rfc_opt = rfc_with_threshold(pipe_rfc_opt, X, thld_rfc)
    test_results["RFC_opt"] = better_metrics(y, y_hat_rfc_opt)

    print("Random Forest Classifier: Done.")

    # Naive Bayes ---
    pipe_nb = load("binary_files/pipe_nb.joblib")

    def nb_with_threshold(pipe_nb, X_test, threshold):
        proba = pipe_nb.predict_proba(X_test)[:, 1]
        y_hat = proba > threshold
        return y_hat

    y_hat_nb = nb_with_threshold(pipe_nb, X, thld_nb)
    test_results["NB_opt"] = better_metrics(y, y_hat_nb)

    print("Gaussian Naive Bayes Classifier: Done.")

    # Linear SVC ---
    pipe_lsvc_opt = load("binary_files/pipe_lsvc_opt.joblib")

    def lsvc_with_threshold(pipe_lsvc, X_test, threshold):
        proba = pipe_lsvc.decision_function(X_test)
        y_hat = proba > threshold
        return y_hat

    y_hat_lsvc_opt = lsvc_with_threshold(pipe_lsvc_opt, X, thld_lsvc)
    test_results["LinearSVC_opt"] = better_metrics(y, y_hat_lsvc_opt)

    print("Linear Support Vector Classifier: Done.")

    # All models

    all_test_results = pd.DataFrame(test_results)
    try:
        all_test_results.to_csv(f"{output_path}/test-results.csv")
    except:
        os.makedirs(os.path.dirname('results/'))
        all_test_results.to_csv(f"{output_path}/test-results.csv")

if __name__ == "__main__":
    main(opt["--data_path"], opt["--output_path"])
