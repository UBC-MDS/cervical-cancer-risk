# Written by Samson Bakos, Dec 2022

# Base architecture
FROM continuumio/miniconda3:4.12.0

# Update Python
RUN conda install -y python=3.10

# Add conda-forge channel 
RUN conda config --append channels conda-forge

# Conda Installs
RUN conda install -y\
    ipykernel=6.17.1 \
    scikit-learn>=1.1.3 \
    altair=4.2.0 \
    altair_saver=0.1.0 \
    matplotlib=3.6.2\ 
    pandas=1.4.4 \
    pandoc>=1.12.3

# Install Pip   
RUN apt-get update && apt-get install -y pip

# Pip Installs
RUN pip install \
    docopt==0.6.2 \
    joblib==1.1.0 \
    selenium==4.2.0 \
    vl-convert-python==0.5.0 \
    shutup==0.2.0

# Install R
RUN apt-get install r-base r-base-dev -y

# Install non R tidyverse dependencies
RUN apt-get update && apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev

# R Installs 
RUN R -q -e 'install.packages("tidyverse")'
RUN R -q -e 'install.packages("rmarkdown")'

# Install Make
RUN apt update && apt install -y make
