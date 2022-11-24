# author: Waiel Tinwala
# date: 2022-11-22

"""Creates exploratory data analysis figures from the preprocessed training data of the cervical cancer (risk factors) dataset (from https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29)
Saves the figures to the provided folder as .png files.

Usage: src/eda_figures.py --train_data=<train_data> --out_file=<out_file>

Example:

python eda_figures.py --train_data='../data/processed/train.csv' --out_file='../results'

Options:
--train_data=<train_data>   Path (including filename) of where the data is stored
--out_file=<out_file>       Path to directory where the figures will be saved
"""

# import required packages
from docopt import docopt
import os
import numpy as np
import pandas as pd
import altair as alt
from altair_saver import save

opt = docopt(__doc__)

def hist(data, feat = None, feat_list = None, repeat = False):
    """Returns a grid of altair histogram charts from a dataframe repeated on every column and faceted on one binary column

    Args:
        data (pandas.dataframe): A pandas dataframe with at least one numerical column and one binary column
        feat (str, optional): Name of column with binary values. Defaults to None.
        feat_list (list, optional): List of numeric feature names to repeat histogram for. Defaults to None.
        repeat (bool, optional): Boolean stating whether or not repeated charts are being created. Defaults to False.

    Returns:
        altair.vegalite.v4.api.VConcatChart: Repeated charts created from the given dataframe
    """

    # code if single chart is required
    if repeat == False:
        chart = alt.Chart(data).mark_bar().encode(
            alt.X( 'Age', type='quantitative'),
            alt.Y( 'count()', stack=False, title=''),
            alt.Color( 'risk', type='ordinal', scale=alt.Scale(scheme='category10'))
        ).properties(
            height=100,
            width=150
        ).facet( 'risk', columns = 1)
        return chart

    # code if repeated charts are required
    if repeat == True:

        # create empty list to store charts 
        chart_list_0 = []
        chart_list_1 = []
        chart_list_concat = []

        # create chart for every column in feat_list 
        for feat in feat_list:
            
            # repeated chart for binary 0
            chart_tmp_0 = alt.Chart(data.query('risk==0')).mark_bar().encode(
                alt.X(feat, type='quantitative', scale = alt.Scale(domain = (0, data[ feat].max()+1))),
                alt.Y('count()', stack=False, title=''),
                alt.Color('risk', type='ordinal', scale=alt.Scale(scheme='category10'))
            ).properties(
                height=100,
                width=150
            )

            # repeated chart for binary 1
            chart_tmp_1 = alt.Chart(data.query('risk==1')).mark_bar().encode(
                alt.X(feat, type='quantitative', scale = alt.Scale(domain = ( 0, data[ feat].max()+1))),
                alt.Y('count()', stack=False, title=''),
                alt.Color('risk', type='ordinal', scale=alt.Scale(scheme='category10'))
            ).properties(
                height=100,
                width=150
            )

            # add each chart to list and return concatenated chart
            chart_list_0.append(chart_tmp_0)
            chart_list_1.append(chart_tmp_1)
            chart_concat = chart_tmp_0 | chart_tmp_1
            chart_list_concat.append(chart_concat)

        return alt.vconcat(*chart_list_concat).properties(title='Distribution of Numeric Features').configure_title(anchor='middle')


def main(data, out_file):
    """Function to create EDA figures for DSCI_522 group project and save to provided directory

    Args:
        data (str): Path to cleaned training data
        out_file (str): Path where figures will be stored at
    """

    # create pandas dataframe of preprocessed train data
    train_data = pd.read_csv(data)

    # create list of binary features
    binary_features = ['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis',
                    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
                    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
                    'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
                    'STDs:Hepatitis B', 'STDs:HPV', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx']

    # create list of numeric features
    numeric_features = ['Age', 'Smokes (years)', 'Smokes (packs/year)', 'Number of sexual partners', 'First sexual intercourse',
                        'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD (years)',
                        'STDs (number)', 'STDs: Number of diagnosis', 'STDs: Time since first diagnosis',
                        'STDs: Time since last diagnosis']

    # create EDA figure for binary data
    cat_chart = alt.Chart(train_data).mark_bar().encode(
        alt.X(alt.repeat(), type='ordinal'),
        alt.Y('count()', title=''),
        alt.Color('risk', type='ordinal', scale=alt.Scale(scheme='category10'), title='Risk')
    ).properties(
        height=150,
        width=75
    ).repeat(
        binary_features,
        columns=4
    )

    cat_chart = cat_chart.properties(
        title='Distribution of Binary Features'
    ).configure_title(
        anchor='middle'
    )

    # create EDA figure for numeric data
    num_chart = hist(train_data, feat_list=numeric_features, repeat=True)

    # save charts to designated folders
    save(num_chart, f'{out_file}/numeric_feat.png', scale_factor=4.0)
    save(cat_chart, f'{out_file}/binary_feat.png', scale_factor=4.0)


if __name__ == "__main__":
    main(opt['--train_data'], opt['--out_file'])