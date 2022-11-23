# author: Cervical Cancer group 7
# date: 2022-11-21
"""
code adapted from Lecture 3 - "DSCI 522 Data Science Workflows"
source file: "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"


"Cleans, splits and pre-processes the Cervical Cancer data (from https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv).

Writes the training and test data to separate csv files.
Usage: src/preprocess_cervical_cancer.py --input_file=<input_file> --out_dir=<out_dir>

Example script to run in terminal: 

python preprocess_cervical_cancer.py --input_file="data/raw/risk_factors_cervical_cancer.csv" --out_dir="data/processed"

  
Options:
--input_file=<input_file>       Path (including filename) to raw data (csv file)
--out_dir=<out_dir>   Path to directory where the processed data should be written
"
"""

# import required libraries
from docopt import docopt
import numpy as np
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split, StratifiedKFold


dc = docopt(__doc__)
def main(input_file, out_dir):
    
  # read data and convert 
  cervical_raw = pd.read_csv(input_file) 
    
  # create target variable 'risk'
  cervical_raw[ 'risk'] = data_full[ 'Hinselmann'] | data_full[ 'Schiller'] | data_full[ 'Citology'] | data_full[  
      'Biopsy']
    
 
  # drop the previous target variables
  cervical_modified = cervical_raw.drop(columns=['Hinselmann', 'Schiller', 'Citology', 'Biopsy'])


  # replace the ? values with NaN
  cervical_clean = cervical_modified.replace('?', np.nan)

  # convert columns to relevant data types
  for col_name in cervical_clean.columns:
    if cervical_clean[col_name].dtype == 'object':
        cervical_clean[col_name] = cervical_clean[col_name].astype(float)
        
 # there are three similar columns and one is chosen while the others are dropped      
  cervical_clean.drop(['STDs:cervical condylomatosis','STDs:vaginal condylomatosis'])

  # split data into training and test sets
  train_df, test_df = train_test_split(cervical_clean, test_size=0.2, random_state=123)

 # defining the numeric features
  numeric_features = [ 'Age', 'Number of sexual partners', 'First sexual intercourse',
        'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)']
    
  #defining the binary features in list  
  binary_features = [ 'STDs:condylomatosis', 'Smokes', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Hinselmann']
    
  column_transformer = make_column_transformer(
    ( make_pipeline( SimpleImputer( strategy = 'median'), StandardScaler()), numeric_features),
    ( make_pipeline( SimpleImputer( strategy = 'constant', fill_value = '99'), OneHotEncoder( handle_unknown =  
                                                                                'ignore')), binary_features))
   
    
  # write training and test data to csv files
  train_df.to_csv(out_dir, index=False)
  test_df.to_csv("data/processed/test.csv", index=False)


if __name__ == "__main__":
    main(dc["--input_file"], dc["--out_dir"])
