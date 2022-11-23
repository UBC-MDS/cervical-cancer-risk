"""
code adapted from Lecture 2 - "DSCI 522 Data Science Workflows"
source file: "https://github.ubc.ca/MDS-2022-23/DSCI_522_dsci-workflows_students/blob/master/src/download_data.py"

Downloads csv data from the given input website to a local filepath as a csv.

Usage: download_data_script.py --url=<url> --output_file=<output_file> 
 
Options:

--url=<url>                   The URL from where the data needs to be downloaded (csv format)
--output_file=<output_file>   path in your local system where the the csv to be saved and desired name of the csv file

Example script to run in terminal: 

python download_data_script.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv" --output_file="risk_factors_cervical_cancer.csv"

"""

#import necessary libraries
import os
import pandas as pd
from docopt import docopt

# initialize docopt
dc = docopt(__doc__)

# function to read the csv and download the csv to the local system

def main(url, output_file):
    
    data = pd.read_csv(url, header=None)
    data.to_csv(f'data/raw/{output_file}', index=False)


if __name__ == "__main__":
    main(dc["--url"], dc["--output_file"])
