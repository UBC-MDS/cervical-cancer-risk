'''This script takes the training data set and returns the optimized models to be tested in .joblib formats.
The models will be exported in the directory where this script is found.
Tables for cross-validation results will be created in the specified directory.

Usage:
model_training.py --data_path=<data_path> --output_path_cv=<output_path_cv>

Options:
--data_path=<data_path>     Path to the data file (including the file name).
--output_path_cv=<output_path_cv> Desired path for the performance results returned.

Example:
python model_training.py --data_path='../data/processed/train.csv' --output_path_cv='../results'
'''

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
from sklearn.exceptions import ConvergenceWarning

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

import shutup

def main( data_path, output_path):
    data_full = pd.read_csv( data_path)

    numeric_features = [ 'Age', 'Number of sexual partners', 'First sexual intercourse',
        'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)']

    binary_features = [ 'STDs:condylomatosis', 'Smokes', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV']

    columns_tbc = numeric_features+binary_features

    column_transformer = make_column_transformer(
        ( make_pipeline( SimpleImputer( strategy = 'median'), StandardScaler()), numeric_features),
        ( make_pipeline( SimpleImputer( strategy = 'constant', fill_value = 99), OneHotEncoder( handle_unknown = 'ignore')), binary_features)
    )

    X = data_full[ columns_tbc]
    y = data_full[ 'risk']

    # KNN ---

    cv_result_dict = {}
    scoring_metrics = [ 'precision', 'recall', 'f1']

    pipe_knn = make_pipeline( column_transformer, KNeighborsClassifier()) # No class weight in KNN

    cv_result_knn = cross_validate( pipe_knn, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'KNN'] = pd.DataFrame( cv_result_knn).agg( [ 'mean', 'std']).T

    param_grid_knn = {
        "kneighborsclassifier__n_neighbors": list( range( 5, 35, 5))
    }

    grid_search_knn = GridSearchCV(
        pipe_knn, param_grid_knn, cv = 5, scoring = 'recall', n_jobs=-1, return_train_score = True
    )

    grid_search_knn.fit( X, y)

    best_params_knn = grid_search_knn.best_params_

    pipe_knn_opt = make_pipeline( column_transformer, KNeighborsClassifier( n_neighbors = best_params_knn['kneighborsclassifier__n_neighbors'])) # No class weight in KNN

    cv_result_knn_opt = cross_validate( pipe_knn_opt, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'KNN_opt'] = pd.DataFrame( cv_result_knn_opt).agg( [ 'mean', 'std']).T

    pipe_knn_opt.fit( X, y)
    dump( pipe_knn_opt, 'pipe_knn_opt.joblib')
    print( 'KNN: Done')

    # SVC ---

    pipe_svc = make_pipeline( column_transformer, SVC( class_weight = 'balanced'))
    cv_result_svc = cross_validate( pipe_svc, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'SVC'] = pd.DataFrame( cv_result_svc).agg( [ 'mean', 'std']).T

    param_dist_svc = {
        'svc__C': [ 10**x for x in range( -2, 5)],
        'svc__gamma': [ 10**x for x in range( -2, 5)]
    }

    random_search_svc = RandomizedSearchCV(
        pipe_svc, param_dist_svc, n_iter = 30, cv = 5, scoring = 'recall', n_jobs=-1, return_train_score = True
    )

    random_search_svc.fit( X, y)

    best_params_svc = random_search_svc.best_params_

    pipe_svc_opt = make_pipeline( column_transformer, SVC( gamma = best_params_svc[ 'svc__gamma'], 
                                                        C = best_params_svc[ 'svc__C'], class_weight = 'balanced'))

    cv_result_svc_opt = cross_validate( pipe_svc_opt, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'SVC_opt'] = pd.DataFrame( cv_result_svc_opt).agg( [ 'mean', 'std']).T

    pipe_svc_opt.fit( X, y)
    dump( pipe_svc_opt, 'pipe_svc_opt.joblib')
    print( 'Support Vector Classifier: Done')

    # RFC ---

    pipe_rfc = make_pipeline( column_transformer, RandomForestClassifier( class_weight = 'balanced', random_state = 123))
    cv_result_rfc = cross_validate( pipe_rfc, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'RFC'] = pd.DataFrame( cv_result_rfc).agg( [ 'mean', 'std']).T

    param_dist_rfc = {
        'randomforestclassifier__n_estimators': [ 100*x for x in range( 1, 11)],
        'randomforestclassifier__max_depth': [ 10*x for x in range( 1, 11)],
        'randomforestclassifier__max_features': [ 'sqrt', 'log2'],
        'randomforestclassifier__criterion': [ 'gini', 'entropy', 'log_loss'],
        'randomforestclassifier__bootstrap': [ True, False]
    }

    random_search_rfc = RandomizedSearchCV(
        pipe_rfc, param_dist_rfc, n_iter = 30, cv = 5, scoring = 'recall', n_jobs=-1, return_train_score = True, random_state = 123
    )

    random_search_rfc.fit( X, y)

    best_params_rfc = random_search_rfc.best_params_

    pipe_rfc_opt = make_pipeline( column_transformer,
                                RandomForestClassifier( n_estimators = best_params_rfc[ 'randomforestclassifier__n_estimators'],
                                                        max_features = best_params_rfc[ 'randomforestclassifier__max_features'],
                                                        max_depth = best_params_rfc[ 'randomforestclassifier__max_depth'],
                                                        criterion = best_params_rfc[ 'randomforestclassifier__criterion'],
                                                        bootstrap = best_params_rfc[ 'randomforestclassifier__bootstrap'],
                                                        class_weight = 'balanced', random_state = 123))

    cv_result_rfc_opt = cross_validate( pipe_rfc_opt, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'RFC_opt'] = pd.DataFrame( cv_result_rfc_opt).agg( [ 'mean', 'std']).T

    pipe_rfc_opt.fit( X, y)
    dump( pipe_rfc_opt, 'pipe_rfc_opt.joblib')
    print( 'Random Forest Classifier: Done')

    # Naive Bayes ---

    pipe_nb = make_pipeline( column_transformer, GaussianNB())
    cv_result_nb = cross_validate( pipe_nb, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'GaussianNB'] = pd.DataFrame( cv_result_nb).agg( [ 'mean', 'std']).T

    pipe_nb.fit( X, y) # As no hyperparameter optimization for Gaussian naive Bayes

    pipe_nb.fit( X, y)
    dump( pipe_nb, 'pipe_nb.joblib')
    print( 'Gaussian Naive Bayes Classifier: Done')

    # Logistic Regression ---

    pipe_logreg = make_pipeline( column_transformer, LogisticRegression( max_iter = 1000, solver = 'saga', class_weight = 'balanced', random_state = 123))
    cv_result_logreg = cross_validate( pipe_logreg, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'LogReg'] = pd.DataFrame( cv_result_logreg).agg( [ 'mean', 'std']).T

    param_dist_logreg = {
        'logisticregression__C': [ 10**x for x in range( -2, 5)],
        'logisticregression__penalty': [ 'none', 'elasticnet'],
        'logisticregression__l1_ratio': [ 0, 0.25, 0.5, 0.75, 1]
    }

    grid_search_logreg = GridSearchCV(
        pipe_logreg, param_dist_logreg, cv = 5, scoring = 'precision', n_jobs=-1, return_train_score = True
    )

    grid_search_logreg.fit( X, y)

    best_params_logreg = grid_search_logreg.best_params_

    pipe_logreg_opt = make_pipeline( column_transformer, LogisticRegression( max_iter = 1000, solver = 'saga',
                                                                        C = best_params_logreg[ 'logisticregression__C'],
                                                                        penalty = best_params_logreg[ 'logisticregression__penalty'],
                                                                        l1_ratio = best_params_logreg[ 'logisticregression__l1_ratio'],
                                                                        class_weight = 'balanced', random_state = 123))

    cv_result_logreg_opt = cross_validate( pipe_logreg_opt, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)
    cv_result_dict[ 'LogReg_opt'] = pd.DataFrame( cv_result_logreg_opt).agg( [ 'mean', 'std']).T

    pipe_logreg_opt.fit( X, y)
    dump( pipe_logreg_opt, 'pipe_logreg_opt.joblib')
    print( 'Logistic Regression Classifier: Done')

    # Cross-validation results of all models

    all_cv_results = pd.concat( cv_result_dict, axis = 1)
    all_cv_results.to_csv( f'{output_path}/cv-results.csv')

    all_cv_results_mean = all_cv_results.T.xs( 'mean', level = 1).T # Probably not the most elegant way to do that.
    all_cv_results_mean.to_csv( f'{output_path}/mean-cv-results.csv')

if __name__ == "__main__":
  main(opt["--data_path"], opt["--output_path_cv"])