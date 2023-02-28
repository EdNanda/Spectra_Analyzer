# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:44:06 2022

@author: HYDN02
"""
from lmfit import Parameters,minimize, fit_report, Model
from lmfit.models import LinearModel,PolynomialModel
from lmfit.models import ExponentialModel,GaussianModel,LorentzianModel,VoigtModel
from lmfit.models import PseudoVoigtModel,ExponentialGaussianModel,SkewedGaussianModel,SkewedVoigtModel
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import sys

## Folder addresses
main_folder = "C:\\Users\\HYDN02\\Seafile\\carofiles\\"
folder = main_folder+"FittingIndividualPeaks_CR05\\"
raw_file = main_folder + "CR_05_matrix.csv"

peaks = glob(folder+"*\\*xlsx")


## Load raw file
raw_data = pd.read_csv(raw_file,header=0,index_col=0)
raw_bckd = raw_data.iloc[:,0:5].mean(axis=1) ##to remove background
x = raw_data.index.values ##needed for fitting and plotting

## sample name
sample = folder.split("_")[-1].split("\\")[0]

##Load all fitting files
repeat_track = True
for c,p in enumerate(peaks):
    name = p.split("\\")[-2].split("_")[-1] ##get sample's name
    
    data = pd.read_excel(p,header=0,index_col=0)
            
    data = data.drop(columns=["r-squared"]) ##drop it to avoid problems
    
    if c == 0:
        fit_data = data
    elif data.keys()[0] in fit_data.keys():
        ## Merge columns with same data
        if len(data)>len(fit_data):
            fit_data = fit_data.reindex(index=range(fit_data.index[0],len(data))) ##Make empty lines so they can be filled
        fit_data.update(data,overwrite=False) ##Update empty lines with new values

    else:
        ## add new data to dataframe
        fit_data = pd.concat([fit_data, data],axis=1,sort=False)
    
fit_data=fit_data.fillna(0) ##fill all NaN with 0


### Unifying fit data 

##find fitted curves
params = Parameters()
par_list = []
for i in fit_data.keys():
    par = i.rsplit("_",1)[0]
    if par not in par_list:
        par_list.append(par)

##make parameters for each fitted curve
for c,m in enumerate(par_list):
    if c == 0:
        gmodel = PseudoVoigtModel(prefix=m+"_")
        params = gmodel.make_params()
    else:
        nmodel = PseudoVoigtModel(prefix=m+"_")
        params.update(nmodel.make_params())
        gmodel = gmodel+nmodel
        

##refit unified data with fit information
vals_dict = {}
r_vals = []
len_curves = fit_data.T.shape[1]
for n in range(len_curves):
    if n%50==0:
        print(str(round(n/len_curves*100,1))+"% lines fitted")

    for c,i in enumerate(fit_data.keys()):
        if i == "r-squared":
            pass
        else:
            vals_dict[i] = fit_data.T[n+fit_data.index[0]][i]
            params.add(i,value=vals_dict[i],vary=False)
            
       
    dataplot = raw_data.iloc[:,n].values-raw_bckd
    
    ## lmfit fitting part
    init = gmodel.eval(params, x=x)
    out = gmodel.fit(dataplot, params, x=x)
    ## calculate r-square value
    r_sq = 1 - out.redchi / np.var(dataplot, ddof=2)
    if r_sq < 0:
        r_sq = 0
    r_vals.append(r_sq)

## add r-squared value to main array
fit_data["r-squared"]=r_vals

## save new unified fitting data to file
fit_data.to_excel(main_folder+sample+"_all_fitting.xlsx")


## Plot data per type
dims = []
for f in fit_data.keys():
    di = f.rsplit("_",1)[-1]
    if di not in dims:
        dims.append(di)
        
for d in dims:
    for ke in fit_data.keys():
        peak_name = ke.rsplit("_",1)[0]
        if d in ke:
            plt.title(sample+"_"+d)
            plt.plot(fit_data[ke],label=peak_name)
            plt.legend(prop={'size': 6})
    plt.savefig(main_folder+sample+"_"+d+".png")
    plt.cla()
    plt.close()
            
