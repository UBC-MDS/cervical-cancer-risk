# author: Cervical Cancer group 7
# date: 2022-11-21
"""
code adapted from Lecture 3 - "DSCI 522 Data Science Workflows"
source file: "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"


"Cleans, splits and pre-processes the Cervical Cancer data (from https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv).

Writes the training and test data to separate csv files.
Usage: src/preprocess_cervical_cancer.py --input=<input> --out_dir=<out_dir>

Example script to run in terminal: 

python preprocess_cervical_cancer.py --input=data/raw/rfcc.csv --out_dir=data/processed

  
Options:
--input=<input>       Path (including filename) to raw data (csv file)
--out_dir=<out_dir>   Path to directory where the processed data should be written
"
"""

# import required libraries
from docopt import docopt
import numpy as np
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split, StratifiedKFold


dc = docopt(doc)
def main(input, out_dir):
    
  # read data and convert 
  cervical_raw <- pd.read_csv(input) 
  # create target variable 'risk'
  risk = []
  for row in range(len(cervical_raw)):
    risk.append(
        cervical_raw.loc[cervical_raw.index[row], 'Hinselmann'] or
        cervical_raw.loc[cervical_raw.index[row], 'Schiller'] or
        cervical_raw.loc[cervical_raw.index[row], 'Citology'] or
        cervical_raw.loc[cervical_raw.index[row], 'Biopsy']         
    )
  cervical_modified = cervical_raw.copy()
  cervical_modified['risk'] = risk

# drop the previous target variables
  cervical_modified = cervical_modified.drop(columns=['Hinselmann', 'Schiller', 'Citology', 'Biopsy'])


# replace the ? values with NaN
  cervical_clean = cervical_modified.replace('?', np.nan)

# convert columns to relevant data types
  for col_name in cervical_clean.columns:
    if cervical_clean[col_name].dtype == 'object':
        cervical_clean[col_name] = cervical_clean[col_name].astype(float)

# split data into training and test sets
  train_df, test_df = train_test_split(cervical_clean, test_size=0.2, random_state=123)

  
    
  # write training and test data to csv files
  

if __name__ == "__main__":
    main(dc["--input"], dc["--out_dir"])
