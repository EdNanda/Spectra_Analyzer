# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:13:45 2022

@author: HYDN02
"""
import matplotlib.pyplot as plt
import pandas as pd

## Setup
raw_file   ="C:\\Data\\caro\\CR01_PureBr_NoAS_2 001524.txt"
final_file ="C:\\Data\\caro\\CR_01_separated_data.csv"
final_image="C:\\Data\\caro\\CR_01_separated_data.png"

## Open file
data = pd.read_csv(raw_file, header=0, skiprows = 16, sep="\t")

## Separate specific data
summary = data[["Time (s)","Pyrometer","Spin Motor"]]



## Plot setup
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(summary["Time (s)"], summary["Pyrometer"], 'g-')
ax2.plot(summary["Time (s)"], summary["Spin Motor"], 'b-')

## Plot data
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (° C)', color='g')
ax2.set_ylabel('Speed (rpm)', color='b')


## Save plot
fig.savefig(final_image)

# Save matrix to file 
summary.to_csv(final_file,index=False)
