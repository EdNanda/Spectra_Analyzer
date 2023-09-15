import pandas as pd
import numpy as np
from lmfit import Model
from glob import glob
from matplotlib import pyplot as plt

def equation(t, k0, Ea, to, n):
    R = 8.314  # J mol^-1 K^-1
    T = 300  # K

    return 1 - np.exp(-k0 * np.exp(-Ea / (R * T)) * (t - to)) ** n


mainfolder = "C:\\Users\\HYDN02\\Seafile\\Code\\DataExamples\\berkleydata\\"

folders = glob(mainfolder+"*\\*fitting*.csv")


all_files = pd.DataFrame()
for i, fo in enumerate(folders):
    file = pd.read_csv(fo, delimiter=";", header=0,index_col=0, decimal = ",")
    suffix = f"_f{i+1}"
    file = file.add_suffix(suffix)
    all_files = pd.concat((all_files, file), axis=1)

cols_to_drop = [col for col in all_files.columns if 'amplitude' not in col]
all_files = all_files.drop(columns=cols_to_drop)

# non_numbers = all_files.applymap(lambda x: not np.isreal(x))
# print(all_files[non_numbers])

# print(all_files.head())
#
df_norm = all_files.apply(lambda x: (x) / x.max())

df_inter = df_norm.iloc[:240]

#df_inter.to_excel(mainfolder+"normalized.xlsx")

for ti in df_inter.columns[:1]:
    print(ti)
    t_array = df_inter[ti].values
    c_array = df_inter.index.values
    try:
        last_zero = np.where(t_array > 0.01)[0][0]-1
    except:
        print("    ###"+ti)
        pass

    max_values = np.sort(t_array)[-20:]
    average_max = np.mean(max_values)
    val_max = np.where(t_array > average_max)[0][0]
    # print(val_max)

    #print(last_zero,t_array[last_zero], val_max, t_array[val_max])

    t_array = t_array[last_zero:val_max-50]
    c_array = c_array[last_zero:val_max-50]
    distanc = val_max-last_zero

    # print(len(t_array), len(c_array),distanc)
    # print(c_array)
    print(t_array)
    # print(distanc)
    # plt.plot(c_array,t_array)
    # plt.show()

    model = Model(equation)
    params = model.make_params(k0=0.04, Ea=1e-7, to=9.45, n=2)
    params['k0'].set(min=0.001, max=0.9, vary=True)
    params['Ea'].set(min=0, max=10000, vary=True)
    params['to'].set(min=0, max=distanc, vary=True)
    params['n'].set(min=1, max=3, vary=True)
    result = model.fit(c_array, params, t=t_array, method='differential_evolution')
    rsqrd = 1 - result.redchi / np.var(t_array, ddof=2)

    # Print the results
    print(result.fit_report())
    print(rsqrd)

    plt.plot(c_array, t_array,".")
    plt.plot(c_array, result.best_fit, '-', label='best fit')
    plt.show()