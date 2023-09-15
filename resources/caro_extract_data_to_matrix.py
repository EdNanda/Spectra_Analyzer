# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:28:45 2022

@author: HYDN02
"""

import pandas as pd
import numpy as np
import seaborn as sns

## Setup
raw_file= "C:\\Data\\caro\\full-integration_CR_01.dat.txt"
final_file="C:\\Data\\caro\\CR_01_matrix.csv"

## Open file
# colnames = ["imagenum","twotheta","twotheta_cuka","dspacing","qvalue","intensity","frame_number","izero","date","time"]
data = pd.read_csv(raw_file, header= 0, decimal=",",delim_whitespace=True)

## Convert columns to matrix
matrix = data.pivot_table(index="qvalue",columns="frame_number",values="intensity")

sns.heatmap(matrix) ##To test plotting (it works)

# Save matrix to file 
# matrix.to_csv(final_file)