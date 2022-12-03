# Author: Morris Chan
# Date 2022-12-02

'''This script takes the training data set and returns the optimized models to be tested in .joblib formats.
The models will be exported in the directory where this script is found.
Tables for cross-validation results will be created in the specified directory.

Usage:
model_training.py --data_path=<data_path> --output_path_cv=<output_path_cv>

Options:
--data_path=<data_path>     Path to the data file (including the file name).
--output_path_cv=<output_path_cv> Desired path for the performance results returned.

Example:
python src/model_training.py --data_path='data/processed/train.csv' --output_path_cv='results'
'''

from docopt import docopt
import numpy as np
import pandas as pd
import os
from joblib import dump, load
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

import altair as alt
alt.renderers.enable('mimetype')
import vl_convert as vlc
# alt.data_transformers.enable('data_server')

import shutup
shutup.please()

opt = docopt(__doc__)

def pr_curve( model, X_train, X_test, y_train, y_test):
    model.fit( X_train, y_train)
    try:
        proba = model.predict_proba( X_test)[ :, 1]
    except:
        proba = model.decision_function( X_test)
    precision, recall, thresholds = precision_recall_curve( y_test, proba)
    thresholds = np.append( thresholds, 1)
    
    plot_df = pd.DataFrame( {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    })
    
    chart = alt.Chart( plot_df).mark_point().encode(
        x = 'precision',
        y = 'recall',
        tooltip = 'thresholds'
    ).properties( height = 300, width = 300)
    return plot_df, chart

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

def knn_cv( X, y, column_transformer, scoring_metrics):
    pipe_knn = make_pipeline( column_transformer, KNeighborsClassifier()) # No class weight in KNN
    cv_result_knn = cross_validate( pipe_knn, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)

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

    pipe_knn_opt.fit( X, y)
    try:
    	dump( pipe_knn_opt, 'binary_files/pipe_knn_opt.joblib')
    except:
    	os.makedirs(os.path.dirname('binary_files/'))
    	dump( pipe_knn_opt, 'binary_files/pipe_knn_opt.joblib')

    return cv_result_knn, cv_result_knn_opt

def svc_cv( X, y, column_transformer, scoring_metrics, output_path):
    pipe_svc = make_pipeline( column_transformer, SVC( class_weight = 'balanced', random_state = 123))
    cv_result_svc = cross_validate( pipe_svc, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)

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

    pipe_svc_opt.fit( X, y)
    dump( pipe_svc_opt, 'binary_files/pipe_svc_opt.joblib')

    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.5, stratify = y, random_state = 123)
    pr_df_svc, pr_curve_svc = pr_curve( pipe_svc_opt, X_train, X_validation, y_train, y_validation)
    save_chart( pr_curve_svc, f'{output_path}/pr_curve_svc.png')
    pr_df_svc.to_csv( f'{output_path}/threshold_svc.csv')

    return cv_result_svc, cv_result_svc_opt

def rfc_cv( X, y, column_transformer, scoring_metrics, output_path):
    
    pipe_rfc = make_pipeline( column_transformer, RandomForestClassifier( class_weight = 'balanced', random_state = 123))
    cv_result_rfc = cross_validate( pipe_rfc, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)

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

    pipe_rfc_opt.fit( X, y)
    dump( pipe_rfc_opt, 'binary_files/pipe_rfc_opt.joblib')

    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.5, stratify = y, random_state = 123)
    pr_df_rfc, pr_curve_svc = pr_curve( pipe_rfc_opt, X_train, X_validation, y_train, y_validation)
    try:
        save_chart( pr_curve_svc, f'{output_path}/pr_curve_rfc.png')
    except:
        os.makedirs(os.path.dirname('results/'))
        save_chart( pr_curve_svc, f'{output_path}/pr_curve_rfc.png')
    pr_df_rfc.to_csv( f'{output_path}/threshold_rfc.csv')

    return cv_result_rfc, cv_result_rfc_opt

def nb_cv( X, y, column_transformer, scoring_metrics, output_path):
    
    pipe_nb = make_pipeline( column_transformer, GaussianNB())
    cv_result_nb = cross_validate( pipe_nb, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)

    pipe_nb.fit( X, y) # As no hyperparameter optimization for Gaussian naive Bayes

    dump( pipe_nb, 'binary_files/pipe_nb.joblib')
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.5, stratify = y, random_state = 123)
    pr_df_nb, pr_curve_nb = pr_curve( pipe_nb, X_train, X_validation, y_train, y_validation)
    save_chart( pr_curve_nb, f'{output_path}/pr_curve_nb.png')
    pr_df_nb.to_csv( f'{output_path}/threshold_nb.csv')

    return cv_result_nb

def logreg_cv( X, y, column_transformer, scoring_metrics, output_path):
    pipe_logreg = make_pipeline( column_transformer, LogisticRegression( max_iter = 1000, solver = 'saga', class_weight = 'balanced', random_state = 123))
    cv_result_logreg = cross_validate( pipe_logreg, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)

    param_dist_logreg = {
        'logisticregression__C': [ 10**x for x in range( -2, 5)],
        'logisticregression__penalty': [ 'l1', 'l2', 'elasticnet']
    }

    grid_search_logreg = GridSearchCV(
        pipe_logreg, param_dist_logreg, cv = 5, scoring = 'precision', n_jobs=-1, return_train_score = True
    )

    grid_search_logreg.fit( X, y)

    best_params_logreg = grid_search_logreg.best_params_

    pipe_logreg_opt = make_pipeline( column_transformer, LogisticRegression( max_iter = 1000, solver = 'saga',
                                                                        C = best_params_logreg[ 'logisticregression__C'],
                                                                        penalty = best_params_logreg[ 'logisticregression__penalty'],
                                                                        class_weight = 'balanced', random_state = 123))

    cv_result_logreg_opt = cross_validate( pipe_logreg_opt, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)

    pipe_logreg_opt.fit( X, y)
    dump( pipe_logreg_opt, 'binary_files/pipe_logreg_opt.joblib')
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.5, stratify = y, random_state = 123)
    pr_df_logreg, pr_curve_logreg = pr_curve( pipe_logreg_opt, X_train, X_validation, y_train, y_validation)
    save_chart( pr_curve_logreg, f'{output_path}/pr_curve_logreg.png')
    pr_df_logreg.to_csv( f'{output_path}/threshold_logreg.csv')

    return cv_result_logreg, cv_result_logreg_opt

def lsvc_cv( X, y, column_transformer, scoring_metrics, output_path):
    pipe_lsvc = make_pipeline( column_transformer, LinearSVC( dual = False, random_state = 123))
    cv_result_lsvc = cross_validate( pipe_lsvc, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)

    param_dist = {
    'linearsvc__C': [ 10**x for x in range( -2, 5)]
    }

    grid_search_lsvc = GridSearchCV(
        pipe_lsvc, param_dist, cv = 5, scoring = 'precision', n_jobs=-1, return_train_score = True
    )

    grid_search_lsvc.fit( X, y)
    best_params_rfc = grid_search_lsvc.best_params_

    pipe_lsvc_opt = make_pipeline( column_transformer, LinearSVC( dual = False, random_state = 123,
                                C = best_params_rfc[ 'linearsvc__C']))
    
    cv_result_lsvc_opt = cross_validate( pipe_lsvc_opt, X, y, cv = 5, return_train_score = True, scoring = scoring_metrics)

    pipe_lsvc_opt.fit( X, y)
    dump( pipe_lsvc_opt, 'binary_files/pipe_lsvc_opt.joblib')
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.5, stratify = y, random_state = 123)
    pr_df_lsvc, pr_curve_lsvc = pr_curve( pipe_lsvc_opt, X_train, X_validation, y_train, y_validation)
    save_chart( pr_curve_lsvc, f'{output_path}/pr_curve_lsvc.png')
    pr_df_lsvc.to_csv( f'{output_path}/threshold_lsvc.csv')
    return cv_result_lsvc, cv_result_lsvc_opt

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

    cv_result_knn, cv_result_knn_opt = knn_cv( X, y, column_transformer, scoring_metrics)
    cv_result_dict[ 'KNN'] = pd.DataFrame( cv_result_knn).agg( [ 'mean', 'std']).T
    cv_result_dict[ 'KNN_opt'] = pd.DataFrame( cv_result_knn_opt).agg( [ 'mean', 'std']).T

    print( 'KNN: Done')

    # SVC ---

    cv_result_svc, cv_result_svc_opt = svc_cv( X, y, column_transformer, scoring_metrics, output_path)
    cv_result_dict[ 'SVC'] = pd.DataFrame( cv_result_svc).agg( [ 'mean', 'std']).T
    cv_result_dict[ 'SVC_opt'] = pd.DataFrame( cv_result_svc_opt).agg( [ 'mean', 'std']).T

    print( 'Support Vector Classifier: Done')

    # RFC ---

    cv_result_rfc, cv_result_rfc_opt = rfc_cv( X, y, column_transformer, scoring_metrics, output_path)
    cv_result_dict[ 'RFC'] = pd.DataFrame( cv_result_rfc).agg( [ 'mean', 'std']).T
    cv_result_dict[ 'RFC_opt'] = pd.DataFrame( cv_result_rfc_opt).agg( [ 'mean', 'std']).T

    print( 'Random Forest Classifier: Done')

    # Naive Bayes ---
    
    cv_result_nb = nb_cv( X, y, column_transformer, scoring_metrics, output_path)
    cv_result_dict[ 'GaussianNB'] = pd.DataFrame( cv_result_nb).agg( [ 'mean', 'std']).T

    print( 'Gaussian Naive Bayes Classifier: Done')

    # Logistic Regression ---

    cv_result_logreg, cv_result_logreg_opt = logreg_cv( X, y, column_transformer, scoring_metrics, output_path)
    cv_result_dict[ 'LogReg'] = pd.DataFrame( cv_result_logreg).agg( [ 'mean', 'std']).T
    cv_result_dict[ 'LogReg_opt'] = pd.DataFrame( cv_result_logreg_opt).agg( [ 'mean', 'std']).T

    print( 'Logistic Regression Classifier: Done')

    # Linear SVC
    
    cv_result_lsvc, cv_result_lsvc_opt = lsvc_cv( X, y, column_transformer, scoring_metrics, output_path)
    cv_result_dict[ 'LinearSVC'] = pd.DataFrame( cv_result_lsvc).agg( [ 'mean', 'std']).T
    cv_result_dict[ 'LinearSVC_opt'] = pd.DataFrame( cv_result_lsvc_opt).agg( [ 'mean', 'std']).T
    print( 'Linear Support Vector Classifier: Done')
    
    # Cross-validation results of all models

    all_cv_results = pd.concat( cv_result_dict, axis = 1)
    
    all_cv_results.to_csv( f'{output_path}/cv-results.csv')

    all_cv_results_mean = all_cv_results.T.xs( 'mean', level = 1).T # Probably not the most elegant way to do that.
    all_cv_results_mean.to_csv( f'{output_path}/mean-cv-results.csv')

if __name__ == "__main__":
  main(opt["--data_path"], opt["--output_path_cv"])
