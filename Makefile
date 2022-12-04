# Makefile
#Group number: 7
# Date: 2022-11-29
# Makefile written by Revathy P, week of Nov 28th, 2022

#all the major output files
all :  data/raw/risk_factors_cervical_cancer.csv data/processed/train.csv results/binary_feat.png results/cv-results.csv results/pr_curve_logreg.png Analysis_Docs/Analysis.html

#download the data from the website, Written by Revathy P, week of Nov 21, 2022
data/raw/risk_factors_cervical_cancer.csv:	src/download_data_script.py	
	python src/download_data_script.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv" --output_file="data/raw/risk_factors_cervical_cancer.csv"
    
#preprocess the downloaded data, Written by Waiel H and Revathy P, week of Nov 21, 2022
data/processed/train.csv data/processed/test.csv:	src/preprocess_cervical_cancer.py data/raw/risk_factors_cervical_cancer.csv
	python src/preprocess_cervical_cancer.py --input_file="data/raw/risk_factors_cervical_cancer.csv" --out_dir="data/processed"

#Exploratory Data Analysis (EDA) of training data, Written by Waiel H, week of Nov 21, 2022
results/binary_feat.png results/numeric_feat.png results/smoke_corr.csv results/std_corr.csv:	src/eda_figures.py data/processed/train.csv
	python src/eda_figures.py --train_data='data/processed/train.csv' --out_file='results'

#create optimised model for the training data, Written by Morris C, week of Nov 21, 2022
results/cv-results.csv results/mean-cv-results.csv binary_files/pipe_rfc_opt.joblib binary_files/pipe_nb.joblib binary_files/pipe_lsvc_opt.joblib binary_files/pipe_knn_opt.joblib binary_files/pipe_logreg_opt.joblib binary_files/pipe_svc_opt.joblib: src/model_training.py data/processed/train.csv
	python src/model_training.py --data_path='data/processed/train.csv' --output_path_cv='results'

#tests the model for the test data, Written by Morris C, week of Nov 21, 2022
results/pr_curve_logreg.png results/pr_curve_nb.png results/pr_curve_rfc.png results/pr_curve_svc.png results/test-results.csv : src/model_testing.py data/processed/test.csv binary_files/pipe_nb.joblib binary_files/pipe_lsvc_opt.joblib binary_files/pipe_rfc_opt.joblib
	python src/model_testing.py --data_path='data/processed/test.csv' --output_path='results'

# render report, Report written by Samson B, week of Nov 21, 2022
Analysis_Docs/Analysis.html: Analysis_Docs/Analysis.Rmd Analysis_Docs/ref.bib results/binary_feat.png results/test-results.csv results/pr_curve_logreg.png 
	Rscript -e "rmarkdown::render('Analysis_Docs/Analysis.Rmd')"


clean: 
	rm -rf data
	rm -rf results
	rm -rf Analysis_Docs/Analysis.html
