# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:33:53 2022

@author: HYDN02
"""

import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")

root = "C:\\Data\\caro\\"
samples = glob(root+"*\\")

time_file = "C:\\Data\\caro\\CR_01\\TimeScales.xlsx"
time_data = pd.read_excel(time_file, skiprows= 5, header= 0, index_col=0)

## Path where result images will be saved
save_path = "C:\\Data\\caro\\"

## To clean PL data with dummy columns
def remove_dummy_columns(mdata):
    non_floats = []
    for col in mdata.columns:
        try:
            float(col)
        except:
            non_floats.append(col)
    mdata = mdata.drop(columns=non_floats)
    mdata = mdata.drop(columns=mdata.columns[-1], axis=1) #remove last also
    return mdata

def find_first_data(dataset):
    ## Make groups to find states at 0 rpm
    grouped = dataset.groupby((dataset["Spin Motor"].shift() != dataset["Spin Motor"]).cumsum())
    
    for k,v in grouped:
        ## find when spincoat process starts (last 0 rpm value)
        if k == 2:
            pos = 0
            start_indx = v.index[pos]
            start_time = v.iloc[pos]["Time (s)"]
        ## find when spincoat process ends (first 0 rpm value afterwards)
        elif k == len(grouped):
            end_indx = v.index[0]
            end_time = v.iloc[0]["Time (s)"]
        else:
            pass
            
    return start_indx,start_time,end_indx,end_time

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


for s in samples:
    name = s.split("\\")[-2] ##get sample name from folderpath
    code = name.replace("_","")  
    print(name)
    
    ## list of files
    try:
        xrd_file = glob(s+"*matrix.csv")[0]
        xrd_plot = True
        
        ## Load files
        xrd_data = pd.read_csv(xrd_file, header= 0, index_col=0)
        
        ## Get data from timescales file
        xrd_time = time_data["Average Time per Frame"][code]
        new_xrd_time = (xrd_data.keys().astype(int)-int(xrd_data.keys()[0]))*xrd_time
        
        ## time matching
        xrd_max_time = new_xrd_time[-1]
        
    except:
        xrd_plot = False
        xrd_file = pd.DataFrame({'A' : [np.nan]})
        
    try:
        ple_file = glob(s+"*PL_measurement.csv")[0]
        ple_plot = True
        
        ## Load files
        ple_data = pd.read_csv(ple_file, skiprows= 22, header= 0, index_col=0)
        ple_data = remove_dummy_columns(ple_data)
        
        ## Get data from timescales file
        ple_offset = time_data["tstart for PL"][code]
        new_ple_time = (ple_data.keys().astype(float)-float(ple_data.keys()[0]))
        
        ## time matching
        ple_max_time = new_ple_time[-1]
        ple_start = find_nearest(new_ple_time, ple_offset)
        
    except:
        ple_plot = False
        ple_file = pd.DataFrame({'A' : [np.nan]})
        
        
    try:
        sep_file = glob(s+"*separated_data.csv")[0]
        sep_plot = True
        
        ## Load files
        sep_data = pd.read_csv(sep_file, header= 0, index_col=None)
        
        ## Get data from timescales file
        si,st, ei,et = find_first_data(sep_data)
        
        ## time matching
        new_sep_time = sep_data["Time (s)"][si:ei]-st
        sep_max_time = new_sep_time.values[-1]
        
    except:
        sep_plot = False
        sep_file = pd.DataFrame({'A' : [np.nan]})
 
    
    if xrd_plot and ple_plot and sep_plot:
        all_max_time = np.max([xrd_max_time,ple_max_time,sep_max_time])
        # p_top = xrd_max_time / all_max_time
        # p_mid = ple_max_time / all_max_time
        # p_bot = sep_max_time / all_max_time 
    elif xrd_plot and ple_plot and not sep_plot:
        all_max_time = np.max([xrd_max_time,ple_max_time])
    elif xrd_plot and not ple_plot and sep_plot:
        all_max_time = np.max([xrd_max_time,sep_max_time])
    elif not xrd_plot and ple_plot and sep_plot:
        all_max_time = np.max([ple_max_time,sep_max_time])
    elif xrd_plot and not ple_plot and not sep_plot:
        all_max_time = np.max([xrd_max_time])
    elif not xrd_plot and not ple_plot and sep_plot:
        all_max_time = np.max([sep_max_time])
    elif not xrd_plot and ple_plot and not sep_plot:
        all_max_time = np.max([ple_max_time])
    else:
        print("Check the filepath to the samples")
        sys.exit()

    
    final_time = 801 #seconds
    time_ticks = np.arange(0,final_time,200) ## ticks are places every 200 s
    tim = len(time_ticks)
    # tim = find_nearest(time_ticks,final_time)
    # print(len(time_ticks),tim, time_ticks)
    
    
    xrd_ticks = []
    ple_ticks = []
    for tt in time_ticks:
        if xrd_plot and ple_plot:
            xrd_ticks.append(find_nearest(new_xrd_time,tt))
            ple_ticks.append(find_nearest(new_ple_time,tt))
        elif xrd_plot and not ple_plot:
            xrd_ticks.append(find_nearest(new_xrd_time,tt))
        elif not xrd_plot and ple_plot:
            ple_ticks.append(find_nearest(new_ple_time,tt))


    ## General config plot
    fig, axs = plt.subplots(3,figsize=(6, 11))
    fig.suptitle(name)
    
    if xrd_plot:
        ## Get plots ranges
        xrd_s = find_nearest(xrd_data.index,0.45)
        xrd_f = find_nearest(xrd_data.index,2.5)
        xrd_tend = find_nearest(new_xrd_time, final_time)
        
        ## 1st plot data
        pc1 = axs[0].pcolormesh(xrd_data.iloc[xrd_s:xrd_f,:xrd_tend],cmap="viridis")
        ## 1st plot config
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Q value')
        axs[0].grid(which='major', axis='x', linestyle='--')
        axs[0].set_xticks(xrd_ticks[:tim])
        axs[0].set_xticklabels(time_ticks)
        axs[0].set_yticks(np.linspace(0,xrd_f-xrd_s,7))
        axs[0].set_yticklabels(np.around(np.linspace(xrd_data.index[xrd_s], xrd_data.index[xrd_f], 7),decimals=2))
        
        # Add the colorbars outside
        box1 = axs[0].get_position()
        pad1, width1 = 0.02, 0.02
        cax1 = fig.add_axes([box1.xmax - pad1*3/4, box1.ymin+0.05, width1, box1.height])
        fig.colorbar(pc1, cax=cax1)
        
    if ple_plot:
        ## Get plots ranges
        ple_s = find_nearest(ple_data.index,450)
        ple_f = find_nearest(ple_data.index,900)
        ple_start = find_nearest(new_ple_time, ple_offset)
        ple_tend = find_nearest(new_ple_time, final_time+ple_offset)
        
        ## 2st plot data
        pc2 = axs[1].pcolormesh(ple_data.iloc[ple_s:ple_f,ple_start:ple_tend],cmap="plasma")
        ## 2st plot config
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Wavelength (nm)')
        axs[1].grid(which='major', axis='x', linestyle='--')
        axs[1].set_xticks(ple_ticks[:tim])
        axs[1].set_xticklabels(time_ticks)
        axs[1].set_yticks(np.linspace(0,ple_f-ple_s,7))
        axs[1].set_yticklabels(np.around(np.linspace(ple_data.index[ple_s], ple_data.index[ple_f], 7),decimals=1))
        # Add the colorbars outside
        box2 = axs[1].get_position()
        pad2, width2 = 0.02, 0.02
        cax2 = fig.add_axes([box2.xmax - pad2*3/4, box2.ymin, width2, box2.height])
        fig.colorbar(pc2, cax=cax2)
    
    if sep_plot:
        ## Get plots ranges
        sep_tend = find_nearest(new_sep_time.values, final_time)
        
        ## 3rd plot data
        ax2 = axs[2].twinx()
        pc3 = axs[2].plot(new_sep_time[si:sep_tend], sep_data["Pyrometer"][si:sep_tend], 'g-')
        ax2.plot(new_sep_time[si:sep_tend], sep_data["Spin Motor"][si:sep_tend], 'b-')
        axs[2].set_xlim(left=0,right=new_sep_time[sep_tend])
        axs[2].grid(which='major', axis='both', linestyle='--')
        # plt.gca().xaxis.grid(True)
        ## 3rd plot config
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Temperature (° C)', color='g')
        ax2.set_ylabel('Speed (rpm)', color='b')
     
    fig.tight_layout()
    
    plt.savefig(save_path+"all-plots_"+name+".png")
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    