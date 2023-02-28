import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from lmfit import Parameters,minimize, fit_report, Model
from lmfit.models import LinearModel,PolynomialModel
from lmfit.models import ExponentialModel,GaussianModel,LorentzianModel,VoigtModel
from glob import glob

fig,ax = plt.subplots()

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity (a.u.)")
graph_raw, = ax.plot([], [], color="gold",lw=2,markersize=1.5,label='Raw')
graph_fit, = ax.plot([], [], color="green",linestyle="dashed",lw=2,markersize=1.5,label='Fit')
graph_g1, = ax.plot([], [], color="purple",linestyle='dotted',lw=2,markersize=1.5,label='G1')
graph_g2, = ax.plot([], [], color="deeppink",linestyle='dotted',lw=2,markersize=1.5,label='G2')
label = ax.text(.05, .95, '0', ha='left', va='top', transform=ax.transAxes, fontsize=16)
legend=plt.legend(loc=1) #Define legend objects

allplots = [graph_raw,graph_fit,graph_g1,graph_g2]


main_folder = "D:\\Seafile\\Code\\DataExamples\\PL_video_test\\"
raw_file = glob(main_folder + "*PL_measurement.csv")[0]
fit_file = glob(main_folder+"*\\*fitting_parameters.xlsx")[0]

raw_data = pd.read_csv(raw_file,header=0,index_col=0,skiprows=22)
fit_data = pd.read_excel(fit_file,header=0,index_col=0)

xvals = raw_data.index.values


def init():
    ax.set_xlim(raw_data.index.min(), raw_data.index.max())
    ax.set_ylim(raw_data.min().min(), raw_data.max().max())
    return allplots

def animate(i):
    graph_raw.set_data(raw_data.index.values,raw_data.iloc[:,i].values)
    label.set_text(str(i))
    try:
        out, comps, parlist = data_collect(fit_data, i)
        graph_fit.set_data(xvals, out.best_fit)
        graph_g1.set_data(xvals, comps[parlist[0]+"_"])
        graph_g2.set_data(xvals, comps[parlist[1]+"_"])
        # print(i)
    except:
        graph_fit.set_data([], [])
        graph_g1.set_data([], [])
        graph_g2.set_data([], [])

    # legend.get_texts()[0].set_text(i) #Update label each at frame


    return allplots


def data_collect(fit_data, i):
    ##find fitted curves

    x = xvals#raw_data.index.values  ##needed for fitting and plotting
    fit_pars = fit_data.loc[[i]]
    # plot_array = []

    lencurves = fit_data.loc[[159]].shape[1]

    loopcounter = int(lencurves / 15) ##This is for the counter

    par_list = []
    for ii in fit_data.keys():
        par = ii.rsplit("_", 1)[0]
        if par not in par_list:
            par_list.append(par)
    par_list = par_list[:-1]##remove r-squared

    ##make parameters for each fitted curve
    for c, m in enumerate(par_list):
        if c == 0:
            gmodel = GaussianModel(prefix=m + "_")
            params = gmodel.make_params()
        else:
            nmodel = GaussianModel(prefix=m + "_")
            params.update(nmodel.make_params())
            gmodel = gmodel + nmodel

    ##refit unified data with fit information
    vals_dict = {}

    for c, k in enumerate(fit_data.keys()):
        if k == "r-squared":
            pass
        else:
            # vals_dict[k] = fit_data.T[n+fit_data.index[0]][k]
            vals_dict[k] = float(fit_pars[k])
            params.add(k, value=vals_dict[k], vary=False)
    dataplot = raw_data.iloc[:, i].values

    ## colors
    # col = cm.get_cmap('plasma', len(par_list))
    # co = [matplotlib.colors.rgb2hex(col(j)) for j in range(col.N)]

    ## lmfit fitting part
    out = gmodel.fit(dataplot, params, x=x)
    # print(out.result)

    comps = out.eval_components(x=x)

    return out,comps, par_list

ani = animation.FuncAnimation(fig,animate,frames=np.arange(raw_data.shape[1]),init_func=init,interval=10)

plt.show()

writergif = animation.PillowWriter(fps=30)
FFwriter = animation.FFMpegWriter(fps=10)
plt.close()
ani.save(main_folder+"animation.gif", writer=writergif)

