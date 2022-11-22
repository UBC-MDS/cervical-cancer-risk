# author: Cervical Cancer group 7
# date: 2022-11-21
"""
code adapted from Lecture 3 - "DSCI 522 Data Science Workflows"
source file: "https://github.com/ttimbers/breast_cancer_predictor/blob/master/src/pre_process_wisc.r"


"Cleans, splits and pre-processes (scales) the Wisconsin breast cancer data (from https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv).

Writes the training and test data to separate feather files.
Usage: src/preprocess_Cervical_cancer.py --input=<input> --out_dir=<out_dir>
  
Options:
--input=<input>       Path (including filename) to raw data (feather file)
--out_dir=<out_dir>   Path to directory where the processed data should be written
" -> doc
"""
library(feather)
library(tidyverse)
library(caret)
library(docopt)
set.seed(2020)

dc = docopt(doc)
def main(url, output_file):
    
  # read data and convert 
  cervical_raw <- read_feather(input) 
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

# create dataframe with counts of each class
class_counts = pd.DataFrame(cervical_modified['risk'].value_counts()).rename(index={0:'No risk of cervical cancer',
                                                                                    1:'Risk of cervical cancer'},
                                                                             columns={'risk':'Target'})
# set caption for Table 1                                                                             
class_counts.style.set_caption('Table 1. Counts of observation for each class')

# replace the ? values with NaN
cervical_clean = cervical_modified.replace('?', np.nan)

# convert columns to relevant data types
for col_name in cervical_clean.columns:
    if cervical_clean[col_name].dtype == 'object':
        cervical_clean[col_name] = cervical_clean[col_name].astype(float)

# split data into training and test sets
train_df, test_df = train_test_split(cervical_clean, test_size=0.2, random_state=123)

# create dataframe with counts of each class and for both train and test set
train_class_counts = pd.DataFrame(train_df['risk'].value_counts())
test_class_counts = pd.DataFrame(test_df['risk'].value_counts())

train_test_class_counts = pd.concat([train_class_counts, test_class_counts], axis=1).rename(
    index={0:'No risk of cervical cancer',
           1:'Risk of cervical cancer'}
)
train_test_class_counts.columns = ['Train', 'Test']

# set caption for Table 2
train_test_class_counts.style.set_caption('Table 2. Counts of observations for each class and partition')






  
  # write scale factor to a file
  try({
    dir.create(out_dir)
  })
  saveRDS(pre_process_scaler, file = paste0(out_dir, "/scale_factor.rds"))
  
  # write training and test data to feather files
  write_feather(training_scaled, paste0(out_dir, "/training.feather"))
  write_feather(test_scaled, paste0(out_dir, "/test.feather"))
}

main(opt[["--input"]], opt[["--out_dir"]])
Footer