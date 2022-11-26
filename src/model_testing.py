'''This script takes the trained models and returns the performance of them on the testing data set.

Usage:
model_testing.py --data_path=<data_path> --output_path=<output_path>

Options:
--data_path=<data_path>     Path to the data file (including the file name).
--output_path=<output_path> Desired path for the perfornace results returned.

Example:
python model_testing.py --data_path='../data/processed/test.csv' --output_path='../results'
'''

from docopt import docopt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score

import altair as alt
alt.renderers.enable('mimetype')
import vl_convert as vlc
# alt.data_transformers.enable('data_server')

opt = docopt(__doc__)

def better_confusion_matrix( y_test, y_hat, labels = [ 0, 1]):
    df = pd.DataFrame( confusion_matrix( y_test, y_hat, labels = labels))
    df.columns = labels
    df = pd.concat( [ df], axis = 1, keys = ['Predicted'])
    df.index = labels
    df = pd.concat( [df], axis = 0, keys = ['Actual'])
    return df

def better_confusion_matrix( y_test, y_hat, labels = [ 0, 1]):
    df = pd.DataFrame( confusion_matrix( y_test, y_hat, labels = labels))
    df.columns = labels
    df = pd.concat( [ df], axis = 1, keys = ['Predicted'])
    df.index = labels
    df = pd.concat( [df], axis = 0, keys = ['Actual'])
    return df

def better_metrics( y_test, y_hat):
    precision = precision_score( y_test, y_hat)
    recall = recall_score( y_test, y_hat)
    f1 = f1_score( y_test, y_hat)
    auc = roc_auc_score( y_test, y_hat)
    metrics_dict = {
        'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
    return metrics_dict

def pr_curve( precision, recall):
    plot_df = pd.DataFrame( {
        'precision': precision,
        'recall': recall
    })

    chart = alt.Chart( plot_df).mark_line().encode(
        x = 'precision',
        y = 'recall'
    ).properties( height = 300, width = 300)
    return chart

def save_chart(chart, filename, scale_factor=1):
    '''
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")
# Function by Joel Ostblom

def main( data_path, output_path):
    data_full = pd.read_csv( data_path)

    numeric_features = [ 'Age', 'Number of sexual partners', 'First sexual intercourse',
        'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)']

    binary_features = [ 'STDs:condylomatosis', 'Smokes', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV']

    columns_tbc = numeric_features+binary_features

    X = data_full[ columns_tbc]
    y = data_full[ 'risk']

    # KNN ---

    pipe_knn_opt = load( 'pipe_knn_opt.joblib')

    y_hat_knn_opt = pipe_knn_opt.predict( X)

    test_results = {}
    test_results[ 'KNN_opt'] = better_metrics( y, y_hat_knn_opt)

    print( 'KNN: Done.')

    # SVC ---

    pipe_svc_opt = load( 'pipe_svc_opt.joblib')

    y_hat_svc_opt = pipe_svc_opt.predict( X)
    test_results[ 'SVC_opt'] = better_metrics( y, y_hat_svc_opt)

    precision_svc, recall_svc, thresholds_svc = precision_recall_curve( y, pipe_svc_opt.decision_function( X))

    pr_curve_svc = pr_curve( precision_svc, recall_svc)
    save_chart( pr_curve_svc, f'{output_path}/pr_curve_svc.png')

    print( 'Support Vector Classifier: Done.')

    # RFC ---
    pipe_rfc_opt = load( 'pipe_rfc_opt.joblib')

    y_hat_rfc_opt = pipe_rfc_opt.predict( X)
    test_results[ 'RFC_opt'] = better_metrics( y, y_hat_rfc_opt)

    precision_rfc, recall_rfc, thresholds_rfc = precision_recall_curve( y, pipe_rfc_opt.predict_proba( X)[ :, 1])

    pr_curve_rfc = pr_curve( precision_rfc, recall_rfc)
    save_chart( pr_curve_rfc, f'{output_path}/pr_curve_rfc.png')

    print( 'Random Forest Classifier: Done.')

    # Naive Bayes ---
    pipe_nb = load( 'pipe_nb.joblib')

    y_hat_nb = pipe_nb.predict( X)
    test_results[ 'NB_opt'] = better_metrics( y, y_hat_nb)

    precision_nb, recall_nb, thresholds_nb = precision_recall_curve( y, pipe_nb.predict_proba( X)[ :, 1])

    pr_curve_nb = pr_curve( precision_nb, recall_nb)
    save_chart( pr_curve_nb, f'{output_path}/pr_curve_nb.png')

    print( 'Gaussian Naive Bayes Classifier: Done.')

    # Logistic Regression ---
    pipe_logreg_opt = load( 'pipe_logreg_opt.joblib')

    y_hat_logreg_opt = pipe_logreg_opt.predict( X)
    test_results[ 'LogReg_opt'] = better_metrics( y, y_hat_logreg_opt)

    precision_logreg, recall_logreg, thresholds_logreg = precision_recall_curve( y, pipe_logreg_opt.predict_proba( X)[ :, 1])

    pr_curve_logreg = pr_curve( precision_logreg, recall_logreg)
    save_chart( pr_curve_logreg, f'{output_path}/pr_curve_logreg.png')

    print( 'Logistic Regression Classifier: Done.')

    # All models

    all_test_results = pd.DataFrame( test_results)
    all_test_results.to_csv( 'test-results.csv')


if __name__ == "__main__":
  main(opt["--data_path"], opt["--output_path"])
