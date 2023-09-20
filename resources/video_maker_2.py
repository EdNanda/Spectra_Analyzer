import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
#from lmfit import Parameters, minimize, fit_report, Model
from lmfit.models import LinearModel, PolynomialModel
from lmfit.models import ExponentialModel, GaussianModel, LorentzianModel, VoigtModel
from glob import glob

main_folder = "D:\\Seafile\\Code\\spectra_analyzer\\Data_examples\\GIWAXS_open_folder_multiple_files\\"
raw_file = glob(main_folder + "*matrix.csv")[0]
fit_file = glob(main_folder + "Fitting\\*fitting_parameters.xlsx")[0]

raw_data = pd.read_csv(raw_file, header=0, index_col=0, skiprows=0)
fit_data = pd.read_excel(fit_file, header=0, index_col=0)

models = []
for ke in fit_data.keys():
    models.append(ke.rsplit("_",1)[0])

models = np.unique(models)[:-1]

unique_mods = len(models)

xvals = raw_data.index.values

color_list = ["orange", "skyblue", "green", "purple", "brown",
              "turquoise", "pink", "gray", "olive", "cyan",
              "crimson", "violet", "lawngreen", "orchid", "gold"]


fig, ax = plt.subplots()

ax.set_xlabel("Wavelength (nm)")  # todo let choose names
ax.set_ylabel("Intensity (a.u.)")
graph_raw, = ax.plot([], [], color="blue", lw=2, markersize=1.5, label='Raw')
graph_fit, = ax.plot([], [], color="red", linestyle="dashed", lw=2, markersize=1.5, label='Fit')

graphs = []
for cc, um in enumerate(range(unique_mods)):
    graphs.append(ax.plot([], [], color=color_list[cc], linestyle='dotted', lw=2, markersize=1.5, label=models[cc]))
    # graph_g1, = ax.plot([], [], color="purple", linestyle='dotted', lw=2, markersize=1.5, label='G1')
    # graph_g2, = ax.plot([], [], color="deeppink", linestyle='dotted', lw=2, markersize=1.5, label='G2')
label = ax.text(.05, .95, '0', ha='left', va='top', transform=ax.transAxes, fontsize=16)
legend = plt.legend(loc=1)  # Define legend objects

allplots = [graph_raw, graph_fit].append(graphs)




def init():
    ax.set_xlim(raw_data.index.min(), raw_data.index.max())
    ax.set_ylim(raw_data.min().min(), raw_data.max().max())
    return allplots


def animate(i):
    graph_raw.set_data(raw_data.index.values, raw_data.iloc[:, i].values)
    label.set_text(str(i))
    try:
        out, comps, parlist = data_collect(fit_data, i)
        graph_fit.set_data(xvals, out.best_fit)
        for cc, ga in enumerate(graphs):
            ga[0].set_data(xvals, comps[parlist[cc]])
        print(i)

        # graph_g1.set_data(xvals, comps[parlist[0] + "_"])
        # graph_g2.set_data(xvals, comps[parlist[1] + "_"])
        # print(i)
    except:
        graph_fit.set_data([], [])
        graph_raw.set_data([], [])
        for ch, gp in enumerate(graphs):
            gp[0].set_data([], [])
        # graph_g1.set_data([], [])
        # graph_g2.set_data([], [])
    # if i == raw_data.shape[1]-1:
    #     print("finished: " + str(i))
    #     plt.close()

    # legend.get_texts()[0].set_text(i) #Update label each at frame

    return allplots


def data_collect(fitting_data, i):
    # find fitted curves

    x = xvals  # raw_data.index.values  ##needed for fitting and plotting
    fit_pars = fitting_data.loc[[i]]
    # print("passed "+str(i))

    # make parameters for each fitted curve

    for c, m in enumerate(models):
        if c == 0:
            gmodel = GaussianModel(prefix=m)
            # gmodel = GaussianModel(prefix=dummy[c])
            params = gmodel.make_params()
        else:
            # nmodel = GaussianModel(prefix=dummy[c])
            nmodel = GaussianModel(prefix=m)
            params.update(nmodel.make_params())
            gmodel = gmodel + nmodel
    # refit unified data with fit information
    vals_dict = {}

    for c, k in enumerate(fitting_data.keys()):
        if k == "r-squared":
            pass
        else:
            # vals_dict[k] = fit_data.T[n+fit_data.index[0]][k]
            # print(k, fit_pars[k].values[0])
            vals_dict[k] = float(fit_pars[k].values[0])
            params.add(k, value=vals_dict[k], vary=False)
    dataplot = raw_data.iloc[:, i].values


    # colors
    # col = cm.get_cmap('plasma', len(par_list))
    # co = [matplotlib.colors.rgb2hex(col(j)) for j in range(col.N)]

    # lmfit fitting part
    out = gmodel.fit(dataplot, params, x=x)
    # print(out.result)

    comps = out.eval_components(x=x)

    return out, comps, models

frames = np.arange(170,raw_data.shape[1])
print(frames)

ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, interval=10, repeat=False)
#plt.show()

writergif = animation.PillowWriter(fps=1)
# FFwriter = animation.FFMpegWriter(fps=10)

print("save")
ani.save(main_folder + "00_animation2.gif", writer=writergif)
