import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from lmfit.models import Model, LinearModel, PolynomialModel
from lmfit.models import ExponentialModel, GaussianModel, LorentzianModel, VoigtModel
from lmfit.models import PseudoVoigtModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel
# from spectra_analyzer.SpectraAnalyzer import MainWindow
# from glob import glob


class VideoMaker:
    def __init__(self, paths, lists, values, gui):
        self.raw_file, self.fit_file, self.save_file = paths
        self.bool_basic, self.names_basic, self.bool_list, self.model_list, self.names_list = lists
        self.name_xaxis, self.name_yaxis, self.srt_frame, self.end_frame, self.fps = values
        self.gui = gui

        self.raw_data = pd.read_excel(self.raw_file, header=0, index_col=0, skiprows=0)
        self.fit_data = pd.read_excel(self.fit_file, header=0, index_col=0)
        # self.fit_data = self.fit_data.reset_index(drop=True)

        models = []
        self.allplots = []
        for ke in self.fit_data.keys():
            models.append(ke.rsplit("_", 1)[0])
        models = np.unique(models)[:-1]
        unique_mods = len(models)

        self.xvals = self.raw_data.index.values
        color_list = ["orange", "skyblue", "green", "purple", "brown",
                      "turquoise", "pink", "gray", "olive", "cyan",
                      "crimson", "violet", "lawngreen", "orchid", "gold"]

        fig, self.ax = plt.subplots()
        self.ax.set_xlabel(self.name_xaxis)
        self.ax.set_ylabel(self.name_yaxis)
        if self.bool_basic[0]:
            self.graph_raw, = self.ax.plot([], [], color="blue", lw=2, markersize=1.5, label=self.names_basic[0])
            self.allplots.append(self.graph_raw)
        if self.bool_basic[1]:
            self.graph_fit, = self.ax.plot([], [], color="red", linestyle="dashed", lw=2, markersize=1.5,
                                           label=self.names_basic[1])
            self.allplots.append(self.graph_fit)
        self.graphs = []
        for cc, um in enumerate(self.bool_list):
            if um:
                self.graphs.append(self.ax.plot([], [], color=color_list[cc], linestyle='dotted', lw=2,
                                                markersize=1.5, label=self.names_list[cc]))
        self.allplots.append(self.graphs)
        self.label = self.ax.text(.05, .95, '0', ha='left', va='top', transform=self.ax.transAxes, fontsize=16)
        self.legend = plt.legend(loc=1)  # Define legend objects


        frames = np.arange(self.srt_frame, self.end_frame)

        ani = animation.FuncAnimation(fig, self.animate, frames=frames, init_func=self.init_gr,
                                      interval=10, repeat=False)

        writergif = animation.PillowWriter(fps=self.fps)

        ani.save(self.save_file, writer=writergif)

    def init_gr(self):
        self.ax.set_xlim(self.raw_data.index.min(), self.raw_data.index.max())
        self.ax.set_ylim(self.raw_data.min().min(), self.raw_data.max().max())
        return self.allplots

    def animate(self, i):
        out, comps, parlist = self.data_collect(self.fit_data, i)
        if self.bool_basic[0]:
            self.graph_raw.set_data(self.raw_data.index.values, self.raw_data.iloc[:, i].values)
        if self.bool_basic[1]:
            self.graph_fit.set_data(self.xvals, out.best_fit)

        self.label.set_text(str(i))

        for cc, ga in enumerate(self.graphs):
            ga[0].set_data(self.xvals, comps[parlist[cc]])

        total_frames = self.end_frame - self.srt_frame

        count = i-self.srt_frame + 1
        if count == total_frames:
            update = "Finished"
        else:
            update = f"Frame {count} of {total_frames} ({count/total_frames*100:.1f})%"
        self.gui.popup_ani_update_label(update)  # TODO this is not working in real-time
        print(update)

        return self.allplots

    def trivial(self, x):
        return 0

    def data_collect(self, fitting_data, i):
        # print("data collect " + str(i))
        x = self.xvals  # raw_data.index.values  ##needed for fitting and plotting
        fit_pars = fitting_data.loc[[i]]

        # model_classes = {
        #     "Linear": LinearModel,
        #     "Polynomial": PolynomialModel,
        #     "Exponential": ExponentialModel,
        #     "Gaussian": GaussianModel,
        #     "Lorentzian": LorentzianModel,
        #     "Voigt": VoigtModel,
        #     "PseudoVoigt": PseudoVoigtModel,
        #     "ExpGaussian": ExponentialGaussianModel,
        #     "SkewedGaussian": SkewedGaussianModel,
        #     "SkewedVoigt": SkewedVoigtModel,
        # }
        #
        # gmodel = None  # Initialize nmodel outside the loop
        #
        # for c, m in enumerate(self.bool_list):
        #     if m:
        #         print(c, self.model_list[c], self.names_list[c])
        #         model_name = self.model_list[c]
        #
        #         if model_name not in model_classes:
        #             raise Exception(f"Model '{model_name}' does not exist")
        #
        #         model_class = model_classes[model_name]
        #         namemodel = model_class(prefix=self.names_list[c])
        #
        #         if c == 0:
        #             params = namemodel.make_params()
        #             gmodel = namemodel
        #         else:
        #             params.update(namemodel.make_params())
        #             gmodel += namemodel
        nmodel = Model(self.trivial)
        gmodel = Model(self.trivial)
        params = None
        # print(self.names_list)
        # print(self.bool_list)
        for c, m in enumerate(self.bool_list):
            if m:
                # print(c,self.model_list[c], self.names_list[c])
                if self.model_list[c] == "Linear":
                    nmodel = LinearModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "Polynomial":
                    nmodel = PolynomialModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "Exponential":
                    nmodel = ExponentialModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "Gaussian":
                    nmodel = GaussianModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "Lorentzian":
                    nmodel = LorentzianModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "Voigt":
                    nmodel = VoigtModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "PseudoVoigt":
                    nmodel = PseudoVoigtModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "ExpGaussian":
                    nmodel = ExponentialGaussianModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "SkewedGaussian":
                    nmodel = SkewedGaussianModel(prefix=self.names_list[c]+"_")
                elif self.model_list[c] == "SkewedVoigt":
                    nmodel = SkewedVoigtModel(prefix=self.names_list[c]+"_")
                else:
                    raise Exception("Model does not exist")

                if c == 0:
                    gmodel = nmodel
                    params = gmodel.make_params()
                else:
                    params.update(nmodel.make_params())
                    gmodel = gmodel + nmodel

        # Select data with the boolean list
        unwanted = ["center", "amplitude", "sigma", "fwhm", "height", "decay", "slope", "intercept",
                    "gamma", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "r-squared"]
        just_names = []

        for fi in fitting_data.keys():
            for un in unwanted:
                if "r-squared" in fi:
                    pass
                elif un in fi:
                    just_names.append(fi.replace("_" + un, ""))
                else:
                    pass

        unique_names = np.unique(just_names)

        # refit unified data with fit information
        if len(unique_names) == len(self.bool_list):
            vals_dict = {}
            models = []
            for cc, na in enumerate(self.names_list):
                if self.bool_list[cc]:
                    models.append(na+"_")
                for ce, ka in enumerate(fit_pars.keys()):
                    if na in ka and self.bool_list[cc]:
                        new_ka = ka.replace(unique_names[cc], na)
                        vals_dict[new_ka] = float(fit_pars[ka].values[0])
                        params.add(new_ka, value=vals_dict[new_ka], vary=False)
                    else:
                        pass

        else:
            raise Exception("Remove background models from fit file and try again")


        dataplot = self.raw_data.iloc[:, i].values
        out = gmodel.fit(dataplot, params, x=x)
        comps = out.eval_components(x=x)

        return out, comps, models
