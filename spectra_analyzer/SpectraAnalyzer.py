__author__ = "Edgar Nandayapa"
__version__ = "1.15 (2022)"

import sys
import os
import csv
import traceback
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import openpyxl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from collections import OrderedDict
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QScrollBar, QToolButton, QLabel, QComboBox, QLineEdit, QMenu
from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QAction, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRunnable, pyqtSlot, QThreadPool
from PyQt5.QtGui import QFont, QIcon
from qtrangeslider import QRangeSlider
from qtrangeslider .qtcompat import QtCore
from qtrangeslider .qtcompat import QtWidgets as QtW
from lmfit.models import Model, LinearModel, PolynomialModel
from lmfit.models import ExponentialModel, GaussianModel, LorentzianModel, VoigtModel
from lmfit.models import PseudoVoigtModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel
from time import time
from datetime import datetime
from glob import glob
from functools import partial
from matplotlib import rcParams

matplotlib.use('Qt5Agg')
rcParams.update({'figure.autolayout': True})
cmaps = OrderedDict()


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class MplCanvas_heatplot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figh = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figh.add_subplot(111)
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Wavelength (nm)')

        super(MplCanvas_heatplot, self).__init__(self.figh)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_xlabel('Wavelength (nm)')
        self.axes.set_ylabel('Intensity (a.u.)')
        self.axes.grid(True, linestyle='--')

        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Create a placeholder widget to hold our toolbar and canvas.
        self.grid_count = 0
        self.mnm = 14  # Max number of models
        self.combo_mod = []
        self.plots = []
        self.models = ["", "Linear", "Polynomial", "Exponential", "Gaussian",
                       "Lorentzian", "Voigt", "PseudoVoigt", "SkewedVoigt",
                       "ExpGaussian", "SkewedGaussian", ]
        self.is_giwaxs = False
        self.is_pero_peak = False
        self.is_file_selected = False

        self.constraints = []

        self.GUI_menubar_setup()
        self.GUI_widgets()
        self.add_fit_setup()
        self.GUI_combobox()
        self.button_actions()

    def GUI_menu_builder(self, actions_list, menu):
        actions = []
        for action_info in actions_list:
            action = QAction(action_info["name"], self)
            action.setShortcut(action_info["shortcut"])
            action.triggered.connect(action_info["callback"])
            menu.addAction(action)
            # actions.append(action)

    def GUI_menubar_setup(self):
        mainMenu = self.menuBar()
        self.infoMenu = QAction("&About", self)

        file_PL_options = [
            {"name": "Open single file (&Automatic)", "shortcut": "Ctrl+O", "callback": self.menu_load_single_matrix},
            {"name": "Open folder (Multiple &files)", "shortcut": "Ctrl+U", "callback": self.menu_load_PL_folder},
        ]
        file_XRD_options = [
            {"name": "Open single file (&Manual)", "shortcut": "Ctrl+M", "callback": self.menu_load_single_manual},
            {"name": "Open folder w/ &Log files", "shortcut": "Ctrl+P", "callback": self.menu_load_giwaxs_w_log},
            {"name": "Open folder (Multiple f&iles)", "shortcut": "", "callback": self.menu_load_xrd_separated},

        ]
        fileMenu = mainMenu.addMenu("&File")
        # a = fileMenu.addAction("Photoluminescence")
        # a.setDisabled(True)
        self.GUI_menu_builder(file_PL_options, fileMenu)
        fileMenu.addSeparator()
        # b = fileMenu.addAction("GIWAXS")
        # b.setDisabled(True)
        special = fileMenu.addMenu("Special")
        self.GUI_menu_builder(file_XRD_options, special)

        fit_set_options = [
            {"name": "Add &model line", "shortcut": "Ctrl+S", "callback": self.model_row_add},
            {"name": "&Save fit parameters", "shortcut": "Ctrl+L", "callback": self.get_all_fit_fields},
            {"name": "&Load fit parameters", "shortcut": "Ctrl+A", "callback": self.populate_fit_fields},
        ]
        fit_pro_options = [
            {"name": "Fit &current spectra", "shortcut": "Ctrl+D", "callback": self.fitmodel_process},
            {"name": "&Fit selected range", "shortcut": "Ctrl+Alt+F", "callback": self.start_parallel_calculation},
        ]
        fitMenu = mainMenu.addMenu("Fi&t")
        self.GUI_menu_builder(fit_set_options, fitMenu)
        fitMenu.addSeparator()
        self.GUI_menu_builder(fit_pro_options, fitMenu)

        functions_1 = [
            {"name": "Convert to &Energy (eV)", "shortcut": "", "callback": self.convert_to_eV},
        ]
        functions_2 = [
            {"name": "Clean &dead Pixel (831nm)", "shortcut": "", "callback": self.clean_dead_pixel},
            {"name": "Subtract &background", "shortcut": "", "callback": self.popup_subtract_bkgd},
            {"name": "Set heatplot &color range", "shortcut": "", "callback": self.popup_heatplot_color_range},
        ]
        functions_3 = [
            {"name": "Save &fitting curves only", "shortcut": "", "callback": self.save_snapshot_data},
            {"name": "Save &current matrix dataset", "shortcut": "", "callback": self.save_current_matrix_state},
            {"name": "Save &initial matrix dataset", "shortcut": "", "callback": self.save_data_2DMatrix},
            {"name": "Save &heatplot as png", "shortcut": "", "callback": self.save_heatplot_giwaxs},
            {"name": "Rename plots axis", "shortcut": "", "callback": self.rename_plot_axis},
        ]
        otherMenu = mainMenu.addMenu("&Other")
        self.GUI_menu_builder(functions_1, otherMenu)
        otherMenu.addSeparator()
        self.GUI_menu_builder(functions_2, otherMenu)
        otherMenu.addSeparator()
        self.GUI_menu_builder(functions_3, otherMenu)

        mainMenu.addAction(self.infoMenu)

    def GUI_widgets(self):
        Lmain = QHBoxLayout()

        self.setWindowTitle("Spectra Analyzer")
        self.setWindowIcon(QIcon("../resources/graph.ico"))

        self.L1fit = QHBoxLayout()
        self.LGfit = QGridLayout()
        self.Badd = QToolButton()
        self.Badd.setText("+")
        self.Badd.setToolTip("Add calculation model")
        self.Bsubtract = QToolButton()
        self.Bsubtract.setText("-")
        self.Bsubtract.setToolTip("Remove calculation model")
        self.Bfit = QToolButton()
        self.Bfit.setText("Fit")
        self.Bfit.setToolTip("Fit single curve")
        self.LR = QLabel()
        self.Lvalue = QLabel()
        self.L1 = QLabel("Start:")
        self.L1.setAlignment(Qt.AlignRight)
        self.L1.setFixedWidth(60)
        self.L2 = QLabel("End:")
        self.L2.setFixedWidth(60)
        self.L2.setAlignment(Qt.AlignRight)
        self.LEstart = QLineEdit()
        self.LEend = QLineEdit()
        self.LEstart.setFixedWidth(60)
        self.LEend.setFixedWidth(60)
        self.BMulti = QToolButton()
        self.BMulti.setText("Fit range")
        self.BMulti.setToolTip("Fit the whole selected range")
        self.BMulti.setFixedWidth(120)
        self.Btest = QToolButton()
        self.Btest.setText("test")
        self.Btest.setToolTip("Click if you dare...")

        verticalSpacer = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        horizontSpacer = QSpacerItem(75, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.L1fit.addWidget(self.Badd)
        self.L1fit.addWidget(self.Bsubtract)
        self.L1fit.addWidget(self.LR)
        self.L1fit.addWidget(self.Lvalue)
        self.L1fit.addItem(horizontSpacer)
        self.L1fit.addWidget(self.Bfit)

        Lmulti = QGridLayout()
        Lmulti.addWidget(self.L1, 0, 0)
        Lmulti.addWidget(self.LEstart, 0, 1)
        Lmulti.addWidget(self.L2, 0, 2)
        Lmulti.addWidget(self.LEend, 0, 3)
        Lmulti.addWidget(self.BMulti, 1, 1, 1, 2)
        Lmulti.addWidget(self.Btest, 2, 0)

        Lend = QHBoxLayout()
        Lend.addLayout(Lmulti)
        Lend.addItem(horizontSpacer)

        self.maximum_label = QLabel("")
        self.LGfit.addWidget(self.maximum_label, 100, 0,1,4)

        Lfit = QVBoxLayout()
        Lfit.addLayout(self.L1fit)
        Lfit.addLayout(self.LGfit)
        Lfit.addItem(verticalSpacer)
        Lfit.addItem(Lend)
        # Create the maptlotlib FigureCanvas object
        self.canvas = MplCanvas(self)
        self.savnac = MplCanvas_heatplot(self)

        self.threadpool = QThreadPool.globalInstance()

        self.ScrollbarTime = QScrollBar()
        self.ScrollbarTime.setOrientation(Qt.Horizontal)
        self.ScrollbarTime.setMaximum(0)
        self.ScrollbarTime.setStyleSheet("background : gray;")

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)
        Lgraph = QVBoxLayout()
        LgrTop = QVBoxLayout()
        self.canvas.setMinimumWidth(500)  # Fix width so it doesn't change
        self.range_slider = QRangeSlider(QtCore.Qt.Vertical)
        self.range_slider.setValue((0, 100))
        LgrTop.addWidget(toolbar)
        LgrTop.addWidget(self.canvas, 10)
        LgrTop.addWidget(self.ScrollbarTime, 1)

        Lspacer = QHBoxLayout()
        Lspacer.addItem(horizontSpacer)
        Lspacer.addLayout(LgrTop, 5)
        LHgra = QHBoxLayout()
        LHgra.addWidget(self.range_slider)
        LHgra.addWidget(self.savnac)
        Lgraph.addLayout(Lspacer, 7)
        Lgraph.addLayout(LHgra, 5)

        Lmain.addLayout(Lfit, 3)
        Lmain.addLayout(Lgraph, 6)
        widget = QtWidgets.QWidget()
        widget.setLayout(Lmain)
        self.statusBar().showMessage("", 100)
        self.setCentralWidget(widget)
        self.show()

    def GUI_combobox(self):
        for i in range(self.mnm):
            self.constraints.append([])
        self.fw = 46  # width of QLineEdit fields
        for nn, cb in enumerate(self.combo_mod):
            try:
                cb[1].currentTextChanged.connect(partial(self.make_ComboBox_fields, cb, nn))
                cb[1].setFixedWidth(100)
            except:
                pass

    def button_actions(self):
        self.Bfit.pressed.connect(self.fitmodel_process)
        self.BMulti.pressed.connect(self.start_parallel_calculation)
        self.ScrollbarTime.valueChanged.connect(self.scrollbar_action)
        self.range_slider.valueChanged.connect(self.slider_action)
        self.Badd.pressed.connect(self.model_row_add)
        self.Bsubtract.pressed.connect(self.model_row_remove)
        self.infoMenu.triggered.connect(self.popup_info)
        # self.Btest.pressed.connect(self.clean_dead_pixel)

    def menu_load_single_matrix(self):
        self.select_file()
        self.is_giwaxs = False
        if self.is_file_selected:
            self.load_single_matrix_file()
            self.create_mod_data()
            self.extract_data_for_axis()
            self.menu_load_successful()
        else:
            self.statusBar().showMessage("File not selected", 5000)

    def menu_load_single_manual(self):
        self.select_file()
        self.is_giwaxs = False
        if self.is_file_selected:
            self.popup_read_file()
            self.popup_test_file_slow()
            self.create_mod_data()
            try:
                self.extract_data_for_axis()
                self.menu_load_successful()
            except:
                self.statusBar().showMessage("ERROR: All column names should be numbers in manual mode!!", 5000)
                print(self.init_data)
        else:
            self.statusBar().showMessage("File not selected", 5000)

    def menu_load_PL_folder(self):
        self.select_folder()
        self.is_giwaxs = False
        if self.is_file_selected:
            self.pl_folder_gather_data()
            self.create_mod_data()
            self.extract_data_for_axis()
            self.menu_load_successful()
        else:
            self.statusBar().showMessage("Folder not selected", 5000)

    def menu_load_giwaxs_w_log(self):
        self.is_giwaxs = True
        self.select_folder()
        if self.is_file_selected:
            self.popup_giwaxs_w_log()
            self.create_mod_data()
            self.extract_data_for_axis()
            self.menu_load_successful()
        else:
            self.statusBar().showMessage("Folder not selected", 5000)

    def menu_load_xrd_separated(self):
        self.select_folder()
        self.is_giwaxs = False
        if self.is_file_selected:
            self.separate_xrd_gather_data()
            self.create_mod_data()
            self.extract_data_for_axis()
            self.menu_load_successful()
        else:
            self.statusBar().showMessage("Folder not selected", 5000)

    def menu_load_successful(self):
        self.statusBar().showMessage("Loading files, please wait...")
        self.plot_setup()
        self.set_default_fitting_range()
        self.ScrollbarTime.setMaximum(self.xsize)
        self.bar_update_plots(0)
        self.statusBar().showMessage("")

    def save_data_2DMatrix(self):
        fi, le = self.dummy_folderpath_file.rsplit("/", 1) # TODO remove dummy...
        self.init_data.to_excel(fi + "/0_collected_" + le + ".xlsx")

    def clean_dead_pixel(self):
        self.mod_data.iloc[1421] = self.mod_data.iloc[1419:1421].mean()
        self.mod_data.iloc[1423] = self.mod_data.iloc[1425:1427].mean()
        self.mod_data.iloc[1422] = self.mod_data.iloc[np.r_[1419:1421, 1425:1427]].mean()
        self.scrollbar_action()

    def save_snapshot_data(self):
        self.fitmodel_process()
        bar = int(self.ScrollbarTime.value())  # Read scrollbar value

        x_data = np.array(self.yarray)
        y_data = np.array(self.mod_data.iloc[:, [bar]].T.values[0])

        comps = self.result.eval_components(x=x_data)

        # Build dataframe
        snapshot = pd.DataFrame()
        snapshot['x_data'] = x_data
        snapshot['raw data'] = y_data
        snapshot['Best fit'] = self.result.best_fit
        for cc, mc in enumerate(self.model_mix.components):
            snapshot[mc.prefix[:-1]] = comps[mc.prefix]
        snapshot.set_index('x_data', inplace=True)

        folder = self.file_path.rsplit("/", 1)[0]
        snapshot.to_excel(folder + "/snapshot_" + str(bar) + ".xlsx")
        self.statusBar().showMessage("Snapshot saved", 5000)

    def model_row_add(self):
        if self.grid_count < 14:
            for gc in self.combo_mod[self.grid_count][1:]:
                gc.setVisible(True)
            self.grid_count += 1
        else:
            self.maximum_label.setText("Reached Maximum")

    def model_row_remove(self):
        self.maximum_label.setText("")
        if self.grid_count > 0:
            self.grid_count -= 1
            self.combo_mod[self.grid_count][1].setCurrentIndex(0)
            for gc in self.combo_mod[self.grid_count][1:]:
                gc.setVisible(False)

            self.clean_all_fit_fields()
        else:
            pass

    def create_mod_data(self):
        self.mod_data = self.init_data.copy()

    def clean_all_fit_fields(self):
        rows = list(range(self.LGfit.rowCount()))[1:]
        x_arr = [1, 2, 4, 6, 8]  # usable grid coordinates
        y_arr = [h for h in rows if h % 2 == 0]  # odd rows

        for yd in y_arr:
            for xd in x_arr:
                testmod = self.LGfit.itemAtPosition(yd, xd)
                try:  # if widget found
                    fieldwid = testmod.widget()
                    if isinstance(fieldwid, QLabel):
                        fieldwid.setText("")  # remove text
                except:
                    pass
        self.LR.setText("")
        self.Lvalue.setText("")

    def get_all_fit_fields(self):
        x_arr = [1, 2, 4, 6, 8]  # usable grid coordinates
        rows = list(range(self.LGfit.rowCount()))
        y_arr = [h for h in rows if h % 2 == 1]

        all_mods = []
        for yd in y_arr:
            row_mod = []
            for xd in x_arr:
                testmod = self.LGfit.itemAtPosition(yd, 2)

                testwid = testmod.widget()
                if len(testwid.currentText()):
                    field = self.LGfit.itemAtPosition(yd, xd)
                    try:
                        fieldwid = field.widget()
                        try:
                            row_mod.append(fieldwid.text())
                        except:
                            row_mod.append(fieldwid.currentText())
                    except:
                        pass
            all_mods.append(row_mod)

        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Fit Parameters File', "", "fit (*.fit)")

        if filename[0] != "":
            textfile = open(filename[0], "w")
            for col_dat in all_mods:
                for row_dat in col_dat:
                    textfile.write(row_dat + "\t")
                textfile.write("\n")
            textfile.close()
        else:
            self.statusBar().showMessage("Fit parameter file not saved", 5000)
            # raise Exception("Fit parameter file not saved")
            # return

    def populate_fit_fields(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, "Select file with fitting parameters?", "",
                                                         "fit (*.fit)")

        if filename[0] != "":
            x_arr = [1, 2, 4, 6, 8]  # usable grid coordinates
            with open(filename[0], "r") as fd:
                reader = csv.reader(fd, delimiter="\t")
                for cr, row in enumerate(reader):
                    if len(row) > 1:
                        self.grid_count += 1
                        for gc in self.combo_mod[cr][1:]:
                            gc.setVisible(True)
                        for ce, ele in enumerate(row[:-1]):
                            field = self.LGfit.itemAtPosition(cr * 2 + 1, x_arr[ce])
                            try:
                                fieldwid = field.widget()
                                try:
                                    fieldwid.setText(ele)
                                except:
                                    fieldwid.setCurrentText(ele)
                            except:
                                pass
        else:
            self.statusBar().showMessage("Fit parameter file not loaded", 5000)
            # raise Exception("Fit parameter file not loaded")
            # return

    def save_heatplot_giwaxs(self):
        # TODO check that this works for any curve, not just giwaxs
        fi, le = self.dummy_folderpath_file.rsplit("/", 1)

        ticks_n = 10

        try:
            sets = len(self.separated)

            if "Time" in self.separated[0].keys():
                total_size = self.separated[-1]["Time"].iloc[-1]
            else:
                total_size = self.separated[-1]["eta"].iloc[-1]

            ind_sizes = []
            last_time = 0

            mnT = 100  # dummy values to find the temperature range
            mxT = 0
            step = 0
            for cg, gs in enumerate(self.separated):
                if cg == 0:
                    step = gs["Time"].iloc[1]
                    dist = gs["Time"].iloc[-1] + step
                    total_size += step
                    last_time = dist
                else:
                    dist = gs["Time"].iloc[-1] + step - last_time
                    # print(gs["Time"].iloc[0],gs["Time"].iloc[-1],dist)
                    last_time = dist

                ind_sizes.append(dist / total_size)  # To find the ratio of the frames

                if mnT > np.min(gs.DegC):
                    mnT = np.min(gs.DegC)

                if mxT < np.max(gs.DegC):
                    mxT = np.max(gs.DegC)

            fig, axs = plt.subplots(1, sets, figsize=(12, 9), gridspec_kw={'width_ratios': ind_sizes, 'wspace': 0.02,
                                                                           'hspace': 0.02}, sharex=False, sharey=True)
            fig.set_tight_layout(False)

            srt = 0
            for c, ks in enumerate(self.separated):
                end = srt + ks.shape[0] - 1
                mat = self.mod_data.iloc[:, srt:end]
                leng = ks.shape[0]
                temp = ks.DegC

                axs[c].pcolorfast(mat, vmin=min(self.mod_data.min()) * 0.9, vmax=max(self.mod_data.max()) * 1.1)
                axt = axs[c].twinx()
                axt.plot(temp, "--m")
                axt.set_ylim([mnT * 0.9, mxT * 1.05])

                tn = int(ticks_n * ind_sizes[c])
                if tn <= 1:
                    tn = 2
                axs[c].set_xticks(np.linspace(0, leng - 1, tn))
                axs[c].set_xticklabels(
                    np.around(np.linspace(self.xtime[srt], self.xtime[end], tn), decimals=1).astype(int))

                if c == 0:
                    axs[0].set_ylabel(r"2$\theta$ (Degree)")
                    axs[0].set_yticks(np.linspace(0, len(self.yarray), 8))
                    axs[0].set_yticklabels(np.linspace(self.yarray[0], self.yarray[-1], 8).astype(int))

                else:
                    pass

                if c % 2 == 0:
                    axs[c].xaxis.tick_bottom()
                else:
                    axs[c].xaxis.tick_top()

                if c == sets - 1:
                    # this uses the secondary axis (axt)
                    axt.set_ylabel("Temperature (°C)", color="m")
                    axt.tick_params(axis='y', colors='m')

                else:
                    axt.yaxis.set_major_locator(plt.NullLocator())

                srt = srt + leng

            if "eta" in self.gname:
                fig.text(0.5, 0.08, 'Eta (degrees)', ha='center')
            else:
                fig.text(0.5, 0.08, 'Time (seconds)', ha='center')

            fig.savefig(fi + "/0_heatplot_" + le + ".png", dpi=300)
            plt.close()
        except:
            self.savnac.figh.savefig(fi + "/0_heatplot_" + le + ".png", dpi=300)


    def select_file(self):
        # print("  select_file")
        old_folder = "C:\\Data\\test\\"

        if not old_folder:  # If empty, go to default
            old_folder = "C:\\Data\\"

        # Select directory from selection
        directory = QtWidgets.QFileDialog.getOpenFileName(self, "Select a file", old_folder)

        if directory[0] != "":  # if cancelled, keep the old one
            self.file_path = directory[0]
            file_text = directory[0].rsplit("/", 1)
            self.folder_path = file_text[0] + "/"
            # print(self.folder_path)
            self.sample_name = file_text[1].split(".")[0]
            # print(self.sample_name)

            self.is_file_selected = True
        else:
            self.is_file_selected = False
            self.statusBar().showMessage("File not selected", 5000)

    def select_folder(self):
        # print("  select_folder")
        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a directory')

        if self.folder_path != "":  # If folder selected, then
            self.is_file_selected = True
            self.folder_path = self.folder_path + "/"
            # print(self.folder_path)
            file_list = glob(self.folder_path)  # Get all files
            self.sample_name = self.folder_path.split("/")[-2]
            # print(self.sample_name)
            input_file = []

            for p in file_list:  # Keep only log files (for giwaxs)
                if "." not in p[-5:] and "Fitting" not in p:
                    input_file.append(p)
                else:
                    pass

            self.giw_names = []
            for f in input_file:
                self.giw_names.append(f.split("\\")[-1])


        else:
            self.is_file_selected = False

    def popup_giwaxs_w_log(self):
        self.dgiw = QDialog()
        Lopt = QVBoxLayout()
        Lopt.setAlignment(Qt.AlignCenter)

        Tdats = QLabel("The following datasets were found,\nplease select one number:\n")
        Lopt.addWidget(Tdats)

        for cs, ds in enumerate(self.giw_names):
            Tlab = QLabel("\t" + str(cs + 1) + ":\t" + ds)
            Lopt.addWidget(Tlab)

        Tempt = QLabel("\n")
        Lopt.addWidget(Tempt)

        self.sel_ds = QLineEdit()
        self.sel_ds.setFixedWidth(50)
        self.sel_ds.setAlignment(Qt.AlignCenter)
        self.sel_ds.setText("1")
        Lopt.addWidget(self.sel_ds)

        Bok = QDialogButtonBox(QDialogButtonBox.Ok)
        Lopt.addWidget(Bok)
        Bok.accepted.connect(self.popup_giwaxs_w_log_ok)

        self.dgiw.setLayout(Lopt)
        self.dgiw.setWindowTitle("Select data")
        self.dgiw.setWindowModality(Qt.ApplicationModal)
        self.dgiw.exec_()

    def popup_giwaxs_w_log_ok(self):
        self.dgiw.close()
        self.giwaxs_gather_data()

    def separate_xrd_gather_data(self):
        self.statusBar().showMessage("Loading files, please be patient...")
        pl_files = sorted(glob(self.folder_path + "*.dat"))

        self.dummy_folderpath_file = self.folder_path + self.sample_name

        # List to store all dataframes
        dataframes = []

        for counter, file in enumerate(pl_files):
            data = pd.read_csv(file, delimiter="\t", skiprows=4, header=None, names=["2Theta", counter],
                               index_col=False)
            data.set_index("2Theta", inplace=True)
            dataframes.append(data)

        # Concatenate all dataframes along the columns axis
        self.init_data = pd.concat(dataframes, axis=1)

        self.statusBar().showMessage("")

    def pl_folder_gather_data(self):
        pl_files = sorted(glob(self.folder_path + "\\*.txt"), key=os.path.getmtime)

        self.dummy_folderpath_file = self.folder_path + "/" + self.folder_path.split("/")[-1]
        self.file_path = self.dummy_folderpath_file

        for counter, file in enumerate(pl_files):
            data = pd.read_csv(file, delimiter="\t", skiprows=14, header=None, names=["Wavelength", counter],
                               index_col=False)
            meta = pd.read_csv(file, delimiter=": ", skiprows=2, nrows=10, index_col=0, header=None, engine="python")
            time = str(meta.T.Date.values[0])
            time = time.replace("CEST ", "")
            delta = datetime.strptime(time, '%a %b %d %H:%M:%S %Y')

            if counter == 0:
                start_t = delta
                self.init_data = data

            else:
                curr_t = delta - start_t
                curr_t = curr_t.total_seconds()
                self.init_data = self.init_data.join(data.set_index("Wavelength"), on="Wavelength")
                self.init_data.rename(columns={counter: curr_t})

        self.init_data.set_index("Wavelength", inplace=True)

    def giwaxs_gather_data(self):
        # This part pre-reads files to find where data start and end

        # read number from popup and fix it if its not there
        fnum = self.sel_ds.text()
        if int(fnum) - 1 in range(len(self.giw_names)):
            fnum = int(fnum) - 1
        else:
            fnum = 0

        self.gname = self.giw_names[fnum]
        self.dummy_folderpath_file = self.folder_path + "/" + self.gname
        input1 = open(self.dummy_folderpath_file, 'rb')
        with input1 as f:
            lines = f.readlines()

        # Analyze file and gather positions of relevant information (time, eta, degC)
        count = 0
        elapsed = 0
        data1 = False
        datetime_object = 0
        combined = 0

        times = []
        starts = []
        ends = []
        line = 1
        # datasets = 0
        for line in lines:
            if "#S" in str(line):
                data1 = True

            if "#D" in str(line) and data1:
                time = str(line[3:-1])[2:-1]
                delta = datetime.strptime(time, '%a %b %d %H:%M:%S %Y')
                if datetime_object == 0:
                    elapsed = 0
                    delta_start = delta
                else:
                    elapsed = delta - delta_start
                    elapsed = elapsed.total_seconds()
                datetime_object = delta
                times.append(elapsed)

            if "#L" in str(line) and data1:
                head = str(line[3:-1])[2:-1].split("  ")
                starts.append(count)

            if "#C results" in str(line) and data1:
                ends.append(count)

            count += 1
        input1.close()

        # Gather relevant information from step above with pandas
        end_times = []
        self.separated = []
        for c, t in enumerate(times):
            data = pd.read_csv(self.dummy_folderpath_file, delimiter=" ", skiprows=starts[c] + 1, header=None,
                               nrows=ends[c] - starts[c] - 1)
            data.columns = head

            if c == 0:
                combined = data
            else:
                if "eta" in self.dummy_folderpath_file:
                    pass
                else:
                    end_times.append(data.Time.iloc[-1])
                    data.Time = data.Time + t

                    combined = pd.concat([combined, data])
            # print(combined)
            self.separated.append(data)
        combined = combined.reset_index(drop=True)

        # Gather measurement data using pandas
        pd.set_option('mode.chained_assignment', None)  # ignores an error message
        for counter, gf in enumerate(glob(self.dummy_folderpath_file + "_[0-9]*.dat")):
            # Read data from file
            Mdata = pd.read_csv(gf, index_col=None, skiprows=15, header=None, delimiter="\t")
            raw_dat = Mdata[[0, 1]]
            raw_dat.rename(columns={0: "TTh", 1: "m_" + str(counter)}, inplace=True)

            if counter == 0:
                self.init_data = raw_dat
            else:
                self.init_data = self.init_data.join(raw_dat.set_index("TTh"), on="TTh")

        self.init_data = self.init_data.set_index("TTh")
        # print(self.mdata)

        if "eta" in self.dummy_folderpath_file:
            self.init_data.columns = [combined.eta, combined.DegC]
            self.comb_data = combined[["eta", "DegC"]]
        else:
            self.init_data.columns = [combined.Time, combined.DegC]
            self.comb_data = combined[["Time", "DegC"]]

    def extract_data_for_axis(self):
        try:
            self.xtime = [ik[0] for ik in self.mod_data.keys()]
        except:
            self.xtime = self.mod_data.keys().values.astype(float)
        self.xsize = len(self.xtime) - 1

        self.max_int = self.mod_data.to_numpy().max()
        self.min_int = self.mod_data.to_numpy().min()

        self.yarray = self.mod_data.index
        self.ysize = len(self.yarray)

        self.range_slider.setMaximum(self.ysize)
        self.range_slider.setValue((0, self.ysize))
        self.set_default_fitting_range()

    # def extract_data_for_axis(self):
    #     # Extract relevant data
    #     self.xtime = self.mod_data.keys().astype(float)
    #     self.xsize = len(self.xtime) - 1
    #
    #     fix_arr = np.ma.masked_invalid(self.mod_data.to_numpy())
    #     self.max_int = fix_arr.max()
    #     self.min_int = fix_arr.min()
    #
    #     self.wave = self.mod_data.index
    #     self.ysize = len(self.wave)
    #     self.range_slider.setMaximum(self.ysize)
    #     self.range_slider.setValue((0, self.ysize))

    def popup_info(self):
        dinf = QDialog()
        Ltext = QVBoxLayout()

        Tlibra = QLabel("Fitting is done using \nthe python library \"lmfit\"")
        Tlibra.setAlignment(Qt.AlignCenter)
        Tlibra.setFont(QFont('Arial', 12))

        Tmodel = QLabel("More information about the models can be found at")
        Tmodel.setAlignment(Qt.AlignCenter)
        Tmodel.setFont(QFont('Arial', 12))
        Tmodel.setOpenExternalLinks(True)

        urlLink = "<a href=\"https://lmfit.github.io/lmfit-py/builtin_models.html\">lmfit.github.io</a>"
        Tlink = QLabel()
        Tlink.setOpenExternalLinks(True)
        Tlink.setText(urlLink)
        Tlink.setAlignment(Qt.AlignCenter)
        Tlink.setFont(QFont('Arial', 12))

        Tempty = QLabel("")
        Tversion = QLabel("Current version: " + __version__)
        Tversion.setAlignment(Qt.AlignCenter)
        Tversion.setFont(QFont('Arial', 12))

        Tauthor = QLabel("Program created by Edgar Nandayapa (2021)\nHelmholtz-Zentrum Berlin")
        Tauthor.setAlignment(Qt.AlignCenter)
        Tauthor.setFont(QFont('Arial', 8))

        Ltext.addWidget(Tlibra)
        Ltext.addWidget(Tempty)
        Ltext.addWidget(Tmodel)
        Ltext.addWidget(Tlink)
        Ltext.addWidget(Tempty)
        Ltext.addWidget(Tversion)
        Ltext.addWidget(Tempty)
        Ltext.addWidget(Tauthor)

        dinf.setLayout(Ltext)
        dinf.setWindowTitle("About")
        dinf.setWindowModality(Qt.ApplicationModal)
        dinf.exec_()

    def popup_read_file(self):
        # Optimize the building of this popup
        self.dlg = QDialog()
        self.clean = QCheckBox()

        self.ind_col = QLineEdit("0")
        self.skiprow = QLineEdit("22")
        self.headers = QLineEdit("0")
        self.remove = QLineEdit("None")
        self.decimal = QLineEdit(".")
        self.delimit = QLineEdit("," if "csv" in self.file_path[-4:] else "\\t")

        bok = QDialogButtonBox(QDialogButtonBox.Ok)
        btest = QToolButton()
        btest.setText("Test")

        self.QFL = QFormLayout()

        widgets_info = [
            {"label": "Skip rows", "widget": self.skiprow,
             "tooltip": "Number of rows to skip\n   e.g.where metadata is\n   None if no not needed"},
            {"label": "Position of header", "widget": self.headers,
             "tooltip": "Row where header is\n   (Remember first row is 0)"},
            {"label": "Index column", "widget": self.ind_col,
             "tooltip": "Number of column where index is, usually Wavelength\n   (Remember first column is 0)"},
            {"label": "Delimiting symbol", "widget": self.delimit,
             "tooltip": "e.g. tab = \\t, comma = ,"},
            {"label": "Remove columns", "widget": self.remove,
             "tooltip": "Separated by a comma\n   or None if not needed\n   e.g. 1,2,3"},
        ]

        for widget_info in widgets_info:
            label = QLabel(widget_info["label"])
            label.setToolTip(widget_info["tooltip"])
            widget_info["widget"].setToolTip(widget_info["tooltip"])
            self.QFL.addRow(label, widget_info["widget"])

        self.QFL.addRow(btest, bok)

        bok.accepted.connect(self.popup_ok)
        btest.clicked.connect(self.popup_test_file_slow)

        self.dlg.setLayout(self.QFL)
        self.dlg.setWindowTitle("Setup up file characteristics")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.dlg.exec_()

    def load_single_matrix_file(self):
        self.statusBar().showMessage("Loading file, please be patient...")
        if "xlsx" in self.file_path[-5:]:
            self.init_data = pd.read_excel(self.file_path, index_col=0, header=0)
        else:
            try:  # load standard matrix file
                self.init_data = pd.read_csv(self.file_path, index_col=0, skiprows=None, header=0,
                                             delimiter="\t", engine="python")
                if not self.init_data.shape[1] != 0:
                    try:
                        self.init_data = pd.read_csv(self.file_path, index_col=0, skiprows=21, header=0,
                                                     delimiter=",", engine="python")
                    except:
                        self.init_data = pd.read_csv(self.file_path, index_col=0, skiprows=22, header=0,
                                                     delimiter=",", engine="python")
                    # When Dark&Bright, do the math to display the raw data properly
                    if self.init_data.keys()[1] == "Bright spectra":
                        sd = self.init_data.iloc[:, 2:].subtract(self.init_data["Dark spectra"], axis="index")
                        bd = self.init_data["Bright spectra"] - self.init_data["Dark spectra"]
                        fn = 1 - sd.divide(bd, axis="index")

                        # filter extreme values
                        val = 10  # This is the extreme value
                        fn.values[fn.values > val] = val
                        fn.values[fn.values < -val] = -val
                        self.init_data = fn
                    elif self.init_data.keys()[0] == "Dark spectra":
                        sd = self.init_data.iloc[:, 2:].subtract(self.init_data["Dark spectra"], axis="index")
                        self.init_data = sd
                    else:
                        pass

            except:
                self.popup_read_file()
            self.statusBar().showMessage("")

    def set_default_fitting_range(self):
        self.LEstart.setText("0")
        self.LEend.setText(str(self.mod_data.shape[1] - 1))

    def read_fitting_range(self):
        self.start = int(self.LEstart.text())
        self.end = int(self.LEend.text())

    def remove_dummy_columns(self):
        non_floats = []
        for col in self.mod_data.columns:
            try:
                float(col)
            except:
                non_floats.append(col)
        self.mod_data = self.mod_data.drop(columns=non_floats)
        self.mod_data = self.mod_data.drop(columns=self.mod_data.columns[-1], axis=1)  # remove last also
        # self.mdata = self.mdata.reindex(sorted(self.mdata.columns), axis=1)
        # print(non_floats)
        # print(self.mdata)

    def popup_test_file_slow(self):
        self.success = False
        h = int(self.headers.text())
        l = self.delimit.text()
        # dc = self.decimal.text()
        rem = self.remove.text().split(",")
        # cleanf = self.clean.
        remove = False
        if "None" not in rem:
            rem = [int(r) - 1 for r in rem]
            remove = True
        if self.skiprow.text() == "None":
            sr = None
        else:
            sr = int(self.skiprow.text())

        if self.ind_col.text() == "None":
            ic = None
        else:
            ic = int(self.ind_col.text())

        try:
            try:
                self.init_data = pd.read_csv(self.file_path, index_col=ic, skiprows=sr, header=h, delimiter=l,
                                             engine="python")
            except:
                self.init_data = pd.read_excel(self.file_path, index_col=ic, skiprows=sr, header=h)

            if remove:
                self.init_data.drop(self.init_data.columns[rem], axis=1, inplace=True)

            self.remove_dummy_columns()

            self.QFL.addRow(QLabel("Headers:"), QLabel(str("  ".join(self.init_data.keys().values[:5]))))
            self.QFL.addRow(QLabel("First line:"), QLabel(str("  ".join(self.init_data.head(1).values[0][:5].astype(str)))))
            self.success = True
        except:
            self.QFL.addRow(QLabel("Something went wrong, please try again."))
            self.success = False

    def popup_ok(self):
        self.popup_test_file_slow()
        if self.success:
            self.dlg.close()
            self.extract_data_for_axis()
        else:
            self.QFL.addRow(QLabel("Something went wrong, please try again."))

    def add_fit_setup(self):
        for ii in range(self.mnm):
            combobool = False

            combobox = QComboBox()
            combobox.addItems(self.models)
            combobox.setVisible(False)

            comboName = QLineEdit()
            comboName.setFixedWidth(80)
            comboName.setVisible(False)

            comboNumber = QLabel(str(ii + 1))
            comboNumber.setFixedWidth(14)
            comboNumber.setVisible(False)

            if ii == 0:
                self.LGfit.addWidget(QLabel("Name"), 0, 1)
                self.LGfit.addWidget(QLabel("Model"), 0, 2)
                self.LGfit.addWidget(QLabel("Parameters"), 0, 3)
                self.LGfit.addWidget(QLabel("  fix\ncenter"), 0, 9)
                self.LGfit.addWidget(QLabel("  neg."), 0, 10)

            self.LGfit.addWidget(comboNumber, ii * 2 + 1, 0)
            self.LGfit.addWidget(comboName, ii * 2 + 1, 1)
            self.LGfit.addWidget(combobox, ii * 2 + 1, 2)
            self.combo_mod.append([combobool, combobox, comboName, comboNumber])

    def make_ComboBox_fields(self, cb, ii):
        single_cnst = []
        if cb[0]:
            for i in reversed(range(1, self.mnm)):
                try:
                    self.LGfit.itemAtPosition(ii * 2 + 1, i + 2).widget().deleteLater()
                except:
                    pass
            cb[0] = False
        else:
            pass

        if cb[1].currentText() == "":
            cb[0] = False
            QLE_array = [QLabel("0"), QLabel("0"), QLabel("0")]

        elif cb[1].currentText() == "Linear":
            slope = QLineEdit()
            inter = QLineEdit()

            QLE_array = [slope, inter]
            for ql in QLE_array:
                ql.setFixedWidth(self.fw)
            single_cnst.append(QLE_array)

            self.LGfit.addWidget(QLabel("Slope:"), ii * 2 + 1, 3)
            self.LGfit.addWidget(slope, ii * 2 + 1, 4)
            self.LGfit.addWidget(QLabel("Y-int:"), ii * 2 + 1, 5)
            self.LGfit.addWidget(inter, ii * 2 + 1, 6)
            cb[0] = True

        elif cb[1].currentText() == "Polynomial":
            degree = QLineEdit()

            QLE_array = [degree]
            for ql in QLE_array:
                ql.setFixedWidth(self.fw)
                ql.setText("7")
            single_cnst.append(QLE_array)

            self.LGfit.addWidget(QLabel("Degree:"), ii * 2 + 1, 3)
            self.LGfit.addWidget(degree, ii * 2 + 1, 4)
            cb[0] = True

        elif cb[1].currentText() == "Exponential":
            amp = QLineEdit()
            exp = QLineEdit()

            QLE_array = [amp, exp]
            for ql in QLE_array:
                ql.setFixedWidth(self.fw)
            single_cnst.append(QLE_array)

            self.LGfit.addWidget(QLabel("Amplitude:"), ii * 2 + 1, 3)
            self.LGfit.addWidget(amp, ii * 2 + 1, 4)
            self.LGfit.addWidget(QLabel("Exponent:"), ii * 2 + 1, 5)
            self.LGfit.addWidget(exp, ii * 2 + 1, 6)
            cb[0] = True

        else:
            amp = QLineEdit()
            center = QLineEdit()
            sigma = QLineEdit()
            fix_button = QCheckBox()
            neg_button = QCheckBox()

            QLE_array = [amp, center, sigma]
            for ql in QLE_array:
                ql.setFixedWidth(self.fw)
            single_cnst.append(QLE_array)

            self.LGfit.addWidget(QLabel("Amplitude:"), ii * 2 + 1, 3)
            self.LGfit.addWidget(amp, ii * 2 + 1, 4)
            self.LGfit.addWidget(QLabel("Center:"), ii * 2 + 1, 5)
            self.LGfit.addWidget(center, ii * 2 + 1, 6)
            self.LGfit.addWidget(QLabel("Sigma:"), ii * 2 + 1, 7)
            self.LGfit.addWidget(sigma, ii * 2 + 1, 8)
            self.LGfit.addWidget(fix_button, ii * 2 + 1, 9)
            self.LGfit.addWidget(neg_button, ii * 2 + 1, 10)
            cb[0] = True

        self.constraints[ii] = single_cnst

    def start_parallel_calculation(self):
        # self.fitting_parameters_to_plot()
        self.read_fitting_range()
        self.fitmodel_setup()
        self.start_time = time()
        try:
            del self.res_df
        except:
            pass

        self.calc_length = self.end - self.start + 1

        self.threadpool.clear()
        self.statusBar().showMessage(
            "Fitting multiprocess started with " + str(self.threadpool.maxThreadCount()) + " threads...")
        for ww in range(self.start, self.end + 1):
            self.send_to_Qthread(ww)

    def send_to_Qthread(self, w):
        # Create a worker object and send function to it
        self.worker = Worker(self.parallel_calculation, w)

        # Whenever signal exists, send it to plot
        self.worker.signals.progress.connect(self.fitting_progress)

        self.threadpool.start(self.worker)
        self.threadpool.releaseThread()  # I think this makes it faster over time

    def fitting_end(self):
        self.res_df = self.res_df.reindex(sorted(self.res_df.columns), axis=1)
        # self.res_df["Best Fit"] = self.result.best_fit
        # print(self.res_df)
        self.save_fitting_data()

    def fitting_progress(self, res):
        try:
            self.res_df = pd.concat([self.res_df, res], axis=1, join="inner")
        except:
            self.res_df = res

        current = self.res_df.shape[1]
        total = self.calc_length

        perc = current / total * 100

        self.statusBar().showMessage(f"{perc:0.1f}% completed ({current:2d})")
        if current == total:
            self.statusBar().showMessage("Finished in " + str(round(time() - self.start_time, 1)) + " seconds")
            self.fitting_end()

    def get_peak_ratios(self):
        df = self.res_df.T

        Akeys = []
        sk = "amplitude"
        for k in df.keys():
            if sk in k:
                Akeys.append(k)
            if "center" in k:
                if np.mean(df[k]) - 14.2 < 0.2:
                    the_key = k
                    # print(the_key)
                    self.is_pero_peak = True
        m_key = the_key.rsplit("_", 1)[0]

        self.norm_df = None
        for ak in Akeys:
            if m_key in ak:
                # print(m_key)
                continue
            else:
                # print(ak)
                norm = pd.DataFrame({ak: df[ak] / df[m_key + "_" + sk].values})
                try:
                    self.norm_df = pd.concat([self.norm_df, norm], axis=1, join="inner")
                except:
                    self.norm_df = norm

    def save_fitting_data(self):
        # Get folder names
        if self.is_giwaxs:
            fi, le = self.dummy_folderpath_file.rsplit("/", 1)
            if self.start != 0 or self.end != self.xsize:
                folder = self.dummy_folderpath_file.rsplit("/", 1)[0] + "/Fitting_" + le + "[" + str(
                    self.start) + "-" + str(
                    self.end) + "]/"
            else:
                folder = fi + "/Fitting_" + le + "/"
            name = le
        else:
            if self.start != 0 or self.end != self.xsize:
                folder = self.file_path.rsplit("/", 1)[0] + "/Fitting[" + str(self.start) + "-" + str(self.end) + "]/"
            else:
                folder = self.file_path.rsplit("/", 1)[0] + "/Fitting/"
            name = self.file_path.rsplit("/", 2)[1]
        # print(folder)

        if not os.path.exists(folder):
            os.makedirs(folder)

        # Start an excel file
        writer = pd.ExcelWriter(folder + "0_" + name + "_fitting_parameters.xlsx")
        dataF = self.res_df.T

        # Add data to excel file, making a new worksheet per dataset
        if self.is_giwaxs:
            self.get_peak_ratios()
            normF = pd.concat([self.comb_data, self.norm_df], axis=1, join="inner")
            dataF = pd.concat([self.comb_data, dataF], axis=1, join="inner")

            normF.to_excel(writer, index=True, sheet_name="Normalized")

        dataF.to_excel(writer, index=True, sheet_name="Fitting")

        # writer.save()
        writer.close()
        self.plot_fitting_previews(folder)

    def plot_fitting_previews(self, folder):
        plt.ioff()
        df = self.res_df.T
        variables = []
        # This part cleans the model labels so it can use them generally
        for ke in df.keys():
            for mn in self.mod_names:
                if mn in ke:
                    variables.append(ke.replace(mn, ""))
        variables = list(set(variables))

        for va in variables:
            for ke in df.keys():
                name = ke.replace(va, "")
                plt.title(va)
                if va in ke:
                    plt.plot(self.xtime[self.start:self.end + 1], df[ke], label=name[:-1])
            self.plot_preview_fitting(folder, va)

        plt.plot(self.xtime[self.start:self.end + 1], df["r-squared"], label="R²")
        plt.title("R-squared")
        self.plot_preview_fitting(folder, "r-squared")

        if self.is_pero_peak:
            try:
                for ndf in self.norm_df.keys():
                    plt.plot(self.xtime[self.start:self.end + 1], self.norm_df[ndf], label=ndf.rsplit("_", 1)[0])
                plt.title("Amplitude ratio with Perovskite peak")
                self.plot_preview_fitting(folder, "ratio")
            except:
                pass

    def plot_preview_fitting(self, folder, fn):
        plt.legend(bbox_to_anchor=(1, 1), loc="best")
        plt.xlabel("Time (seconds)")
        plt.ylabel("")
        plt.grid(True, linestyle='--')
        if fn == "r-squared":
            plt.savefig(folder + "0_preview_fit_0_" + fn + ".png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(folder + "0_preview_fit_" + fn + ".png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    def parallel_calculation(self, w, progress_callback):

        # try:
        ydata = np.array(self.mod_data.iloc[:, w].values)
        xdata = np.array(self.mod_data.index.values)
        # except:
        #     xdata = np.array(self.wave)
        #     ydata = np.array(self.init_data.iloc[:, w].values)

        result = self.model_mix.fit(ydata, self.pars, x=xdata)
        rsqrd = 1 - result.redchi / np.var(ydata, ddof=2)

        res = pd.DataFrame.from_dict(result.values, orient="index", columns=[w])

        rsq = pd.DataFrame([rsqrd], columns=[w], index=["r-squared"])

        new = pd.concat([res, rsq])

        progress_callback.emit(new)

    def fitmodel_process(self):
        self.clean_all_fit_fields()

        self.fitmodel_setup()
        if self.fit_model_bool:
            self.fitmodel_plot()
        else:
            pass

    def trivial(self, x):
        return 0

    def fix_model_name(self, name, model):
        model_dict = {'Linear': 'li', 'Polynomial': 'po', 'Exponential': 'ex', 'Gaussian': 'ga',
                'Lorentzian': 'lo', 'Voigt': 'vo', 'PseudoVoigt': 'pv', 'SkewedVoigt': 'sv',
                'ExpGaussian': 'eg', 'SkewedGaussian': 'sg', }

        new_name = model_dict[model] + "_" + name

        return new_name


    def fitmodel_setup(self):  # FITTING PART #TODO make class, when gui created, make instance of class
        self.is_pero_peak = False  # Reset value to False
        bar = int(self.ScrollbarTime.value())  # Read scrollbar value

        # try:
        y_data = np.array(self.mod_data.iloc[:, [bar]].T.values[0])
        x_data = np.array(self.mod_data.index.values)
        # x_data = np.array(self.wave)
        # except:
        #     y_data = np.array(self.init_data.iloc[:, [bar]].T.values[0])
        #     x_data = np.array(self.wave)

        # mod_number = 0
        self.model_mix = Model(self.trivial)
        self.pars = {}
        self.mod_names = []
        self.fit_vals = []
        model_type = None

        for nn, list_name in enumerate(self.combo_mod):
            if nn == 0:
                try:
                    del self.model_mix
                    del self.pars
                except:
                    pass
            else:
                pass

            list_name = list_name[1]
            if list_name.currentText() == "":
                pass

            elif list_name.currentText() == "Linear":
                if len(self.combo_mod[nn][2].text()) > 0:
                    model_type = self.combo_mod[nn][2].text() + "_"
                else:
                    model_type = "Linear_" + str(nn + 1) + "_"

                first_model = LinearModel(prefix=model_type)

                try:
                    self.model_mix = self.model_mix + first_model
                    self.pars.update(first_model.make_params())

                except:
                    self.model_mix = first_model
                    self.pars = first_model.guess(y_data, x=x_data)

                slope = self.constraints[nn][0][0].text().replace(",", ".")
                interc = self.constraints[nn][0][1].text().replace(",", ".")

                if len(slope) >= 1:
                    self.pars[model_type + "slope"].set(value=float(slope))
                else:
                    pass
                if len(interc) >= 1:
                    self.pars[model_type + "intercept"].set(value=float(interc))
                else:
                    pass

            elif list_name.currentText() == "Polynomial":
                if len(self.combo_mod[nn][2].text()) > 0:
                    model_type = self.combo_mod[nn][2].text() + "_"
                else:
                    model_type = "Polynomial_" + str(nn + 1) + "_"

                # mod_name = cur_name+"_"+str(nn+1)+"_"

                deg = self.constraints[nn][0][0].text()
                if int(deg) > 7:
                    deg = 7
                    self.constraints[nn][0][0].setText("7")

                first_model = PolynomialModel(prefix=model_type, degree=int(deg))

                try:
                    self.model_mix = self.model_mix + first_model
                    self.pars.update(first_model.make_params())
                except:
                    self.model_mix = first_model
                    self.pars = first_model.guess(y_data, x=x_data)



            elif list_name.currentText() == "Exponential":
                if len(self.combo_mod[nn][2].text()) > 0:
                    model_type = self.combo_mod[nn][2].text() + "_"
                else:
                    model_type = "Exponential_" + str(nn + 1) + "_"

                # mod_name = cur_name+"_"+str(nn+1)+"_"
                first_model = ExponentialModel(prefix=model_type)

                try:
                    self.model_mix = self.model_mix + first_model
                    self.pars.update(first_model.make_params())
                except:
                    self.model_mix = first_model
                    self.pars = first_model.guess(y_data, x=x_data)

                amp = self.constraints[nn][0][0].text().replace(",", ".")
                dec = self.constraints[nn][0][1].text().replace(",", ".")

                if len(amp) >= 1:
                    self.pars[model_type + "amplitude"].set(value=float(amp))
                else:
                    pass
                if len(dec) >= 1:
                    self.pars[model_type + "decay"].set(value=float(dec))
                else:
                    pass

            else:
                if len(self.combo_mod[nn][2].text()) > 0:
                    model_type = self.combo_mod[nn][2].text() + "_"
                    print("combo_mod " + self.combo_mod[nn][2].text())
                else:
                    model_type = list_name.currentText() + "_" + str(nn + 1) + "_"


                if "Lorentzian" in list_name.currentText():
                    first_model = LorentzianModel(prefix=model_type)
                elif "PseudoVoigt" in list_name.currentText():
                    first_model = PseudoVoigtModel(prefix=model_type)
                elif "ExpGaussian" in list_name.currentText():
                    first_model = ExponentialGaussianModel(prefix=model_type)
                elif "SkewedGaussian" in list_name.currentText():
                    first_model = SkewedGaussianModel(prefix=model_type)
                elif "SkewedVoigt" in list_name.currentText():
                    first_model = SkewedVoigtModel(prefix=model_type)
                elif "Voigt" in list_name.currentText():
                    first_model = VoigtModel(prefix=model_type)
                elif "Gaussian" in list_name.currentText():
                    first_model = GaussianModel(prefix=model_type)
                else:
                    print("model error")

                try:
                    self.model_mix = self.model_mix + first_model
                    self.pars.update(first_model.make_params())
                except:
                    self.model_mix = first_model
                    self.pars = first_model.guess(y_data, x=x_data)

                amp = self.constraints[nn][0][0].text().replace(",", ".")
                cen = self.constraints[nn][0][1].text().replace(",", ".")
                sig = self.constraints[nn][0][2].text().replace(",", ".")

                if len(amp) >= 1:
                    va = float(amp)
                    if self.LGfit.itemAtPosition(nn * 2 + 1, 10).widget().isChecked():
                        self.pars[model_type + "amplitude"].set(value=va, max=0)
                    else:
                        self.pars[model_type + "amplitude"].set(value=va, min=0)
                else:
                    self.pars[model_type + "amplitude"].set(min=0)

                self.pars[model_type + "height"].set(max=self.max_int)

                if len(cen) >= 1:
                    vv = float(cen)
                    if self.LGfit.itemAtPosition(nn * 2 + 1, 9).widget().isChecked():
                        self.pars[model_type + "center"].set(value=vv, vary=False)
                    else:
                        self.pars[model_type + "center"].set(value=vv, min=vv / 3, max=vv * 3)
                else:
                    pass
                if len(sig) >= 1:
                    vs = float(sig)
                    self.pars[model_type + "sigma"].set(value=vs, min=vs / 3, max=vs * 3)
                else:
                    pass

            if model_type is not None:
                self.mod_names.append(model_type)
                self.fit_model_bool = True
            else:
                self.statusBar().showMessage("No fitting models selected", 5000)
                self.fit_model_bool = False
                # raise Exception("No fitting models")
                # return

    def fitmodel_plot(self):
        self.statusBar().showMessage("Fitting...   This might take some time")

        bar = int(self.ScrollbarTime.value())  # Read scrollbar value

        # try:
        y_data = np.array(self.mod_data.iloc[:, [bar]].T.values[0])
        x_data = np.array(self.mod_data.index.values)
        # except:
        #     y_data = np.array(self.init_data.iloc[:, [bar]].T.values[0])
        #     x_data = np.array(self.wave)

        try:
            self.result = self.model_mix.fit(y_data, self.pars, x=x_data)
            comps = self.result.eval_components(x=x_data)
        except ValueError:
            self.statusBar().showMessage("## One of the models shows an error ##", 10000)

        self.fit_vals = self.result.values
        self.add_fitting_data_to_gui()

        # This can be separated into new function (if needed)
        if self.plots:
            try:
                for sp in self.plots:
                    sp.pop(0).remove()
                self.best_fit.pop(0).remove()
                self.plots = []
            except:
                pass
        else:
            pass

        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20(np.linspace(0, 1, 10)))

        for cc, mc in enumerate(self.model_mix.components):
            plot = self.canvas.axes.plot(x_data, comps[mc.prefix], '--', label=mc.prefix[:-1])
            self.plots.append(plot)
        self.best_fit = self.canvas.axes.plot(x_data, self.result.best_fit, '-.b', label='Best fit')

        try:
            # self.LGfit.itemAtPosition(0,1).widget().deleteLater()
            # self.LGfit.itemAtPosition(0,2).widget().deleteLater()
            self.LR.setText("")
            self.Lvalue.setText("")
        except:
            pass

        self.rsquared = 1 - self.result.redchi / np.var(y_data, ddof=2)

        r2_label = str(np.round(self.rsquared, 4))

        # self.LGfit.addWidget(QLabel(" R² = "),0,1)
        # self.LGfit.addWidget(QLabel(r2_label),0,2)
        self.LR.setText(" R² = ")
        self.Lvalue.setText(r2_label)

        self.canvas.axes.legend(loc="best")
        self.canvas.draw_idle()
        self.statusBar().showMessage("Initial fitting is done", 5000)

    def convert_to_eV(self):
        # set variables
        hc = (4.135667696E-15) * (2.999792E8) * 1E9
        eV_conv = hc / self.mod_data.index

        # Make conversion of database and index
        ev_df = self.mod_data.multiply(self.mod_data.index.values ** 2, axis="index") / hc

        ev_df = ev_df.set_index(eV_conv)
        ev_df.index.names = ["Energy"]

        # This is for plotting later
        axis = np.around(np.linspace(self.yarray[0], self.yarray[-1], 8), decimals=1)
        self.eV_axis = np.round(hc / axis, 1)

        # Rename mdata (this is what is always plotted)
        self.mod_data = ev_df

        # Update plot
        self.extract_data_for_axis()
        self.plot_setup()
        self.bar_update_plots(0)
        # self.scrollbar_action()

    def popup_subtract_bkgd(self):
        self.dgiw = QDialog()
        Lopt = QVBoxLayout()
        Lopt.setAlignment(Qt.AlignCenter)

        Tdats = QLabel("Select the starting position\nand the number of spectra curves to average")
        Lopt.addWidget(Tdats)

        Tempt = QLabel("\n")
        Lopt.addWidget(Tempt)

        layout = QFormLayout()

        self.left_b = QLineEdit()
        self.left_b.setFixedWidth(50)
        self.left_b.setAlignment(Qt.AlignCenter)
        self.left_b.setText("0")
        self.len_range = QLineEdit()
        self.len_range.setFixedWidth(50)
        self.len_range.setAlignment(Qt.AlignCenter)
        self.len_range.setText("5")
        layout.addRow("Start pos.", self.left_b)
        layout.addRow("Mean length", self.len_range)

        Lopt.addLayout(layout)

        Bok = QDialogButtonBox(QDialogButtonBox.Ok)
        Lopt.addWidget(Bok)
        Bok.accepted.connect(self.subtract_background)

        self.dgiw.setLayout(Lopt)
        self.dgiw.setWindowTitle("Select range to subtract")
        self.dgiw.setWindowModality(Qt.ApplicationModal)
        self.dgiw.exec_()

    def subtract_background(self):
        left_b = int(self.left_b.text())
        right_b = left_b + int(self.len_range.text())

        # Calculate mean of selected range of columns
        col_mean = self.init_data.iloc[:, left_b:right_b].mean(axis=1)
        # Subtract mean to all dataset
        clean_data = self.init_data.subtract(col_mean, "index")
        # Rename mdata (this is what is always plotted)
        self.init_data = clean_data
        self.create_mod_data()
        # self.init_data = clean_data

        # Update plot
        self.extract_data_for_axis()
        self.plot_setup()
        self.bar_update_plots(0)
        # self.scrollbar_action()

    def plot_restart(self):
        self.savnac.axes.set_xlabel('Time (s)')
        self.savnac.axes.set_ylabel('Wavelength (nm)')

        self.canvas.axes.set_xlabel('Wavelength (nm)')
        self.canvas.axes.set_ylabel('Intensity (a.u.)')
        self.canvas.axes.grid(True, linestyle='--')

    def plot_setup(self):
        self.setWindowTitle("Spectra Analyzer (" + self.sample_name + ")")

        try:
            self.canvas.axes.cla()
            self.savnac.axes.cla()
            self.plot_restart()
            self.ax2.remove()
        except:
            pass

        # First plot
        self._plot_ref, = self.canvas.axes.plot(self.yarray, self.mod_data.iloc[:, [0]], 'r', label="Experiment")
        index_name = self.mod_data.index.name

        if "0.000" in index_name:
            axis_name = "Wavelength (nm)"
        elif "Wavelength" in index_name:
            axis_name = index_name + " (nm)"
        elif "Energy" in index_name:
            axis_name = index_name + " (eV)"
        elif "TTh" in index_name:
            axis_name = r"2$\theta$ (Degree)"
        else:
            axis_name = index_name

        if self.is_giwaxs:
            self.canvas.axes.set_xlabel(axis_name)
            if "eta" in self.gname:
                self.t_label = "Degree"
            else:
                self.t_label = "Time"
        else:

            self.canvas.axes.set_xlabel(axis_name)
            self.t_label = "Time"

        # Set text fields for time and position
        self.text_time = self.canvas.axes.text(0.4, 0.9, self.t_label + " 0.0",
                                               horizontalalignment='left', verticalalignment='center',
                                               transform=self.canvas.axes.transAxes)
        self.text_pos = self.canvas.axes.text(0.4, 0.83, "Position 0",
                                              horizontalalignment='left', verticalalignment='center',
                                              transform=self.canvas.axes.transAxes)

        self.canvas.axes.set_ylim([self.min_int * 0.9, self.max_int * 1.1])  # Set y-axis range
        self.canvas.axes.legend(loc="best")  # Position legend smartly

        # Second plot
        if self.is_giwaxs:
            self.ax2 = self.savnac.axes.twinx()

        self._plot_heat = self.savnac.axes.pcolorfast(self.mod_data)  # 2D heatplot
        self._plot_vline, = self.savnac.axes.plot([0, 0], [0, self.ysize], 'r')  # Vertical line (Time select)
        self._plot_hline1, = self.savnac.axes.plot([0, self.xsize], [0, 0], 'b')  # Horizontal line1 (Up boundary)
        self._plot_hline2, = self.savnac.axes.plot([0, self.xsize], [0, 0], 'b')  # Horizontal line2 (Down boundary)

        if self.is_giwaxs:
            if "eta" in self.gname:
                self.savnac.axes.set_xlabel("Eta (degrees)")
            else:
                self.savnac.axes.set_xlabel("Time (seconds)")
            self.savnac.axes.set_ylabel(axis_name)
            tempe = [ik[1] for ik in self.mod_data.keys()]
            self.ax2.plot(range(len(self.xtime)), tempe, "--m")
            self.ax2.set_ylabel("Temperature (°C)", color="m")  #
            self.ax2.set_ylim([min(tempe) * 0.9, max(tempe) * 1.1])
            self.ax2.tick_params(axis='y', colors='m')
        else:
            self.savnac.axes.set_xlabel("Time (seconds)")
            self.savnac.axes.set_ylabel(axis_name)

        # Reset ticks to match data
        # Y-axis
        if "Energy" in axis_name:
            self.savnac.axes.set_yticks(np.linspace(0, len(self.yarray), 8))
            self.savnac.axes.set_yticklabels(self.eV_axis)
        else:
            self.savnac.axes.set_yticks(np.linspace(0, len(self.yarray), 8))
            self.savnac.axes.set_yticklabels(np.around(np.linspace(self.yarray[0], self.yarray[-1], 8), decimals=1))
        # X-axis
        self.savnac.axes.set_xticks(np.linspace(0, len(self.xtime), 8))
        try:  # In case index is not made of numbers but strings
            if "eta" in self.gname:
                self.savnac.axes.set_xticklabels(np.around(np.linspace(0, self.xtime[-1], 8), decimals=1))
            else:
                self.savnac.axes.set_xticklabels(np.around(np.linspace(0, self.xtime[-1], 8), decimals=0).astype(int))
        except:
            pass

    def rename_plot_axis(self):
        self.dgiw = QDialog()
        layout = QVBoxLayout()

        form_lay = QFormLayout()

        axis_labels = [QLineEdit(),QLineEdit(),QLineEdit(),QLineEdit()]

        form_lay.addRow("Plot X-label",axis_labels[0])
        form_lay.addRow("Plot Y-label", axis_labels[1])
        form_lay.addRow("Heatmap X-label", axis_labels[2])
        form_lay.addRow("Heatmap Y-label", axis_labels[3])

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.rename_accept(axis_labels))
        buttons.rejected.connect(self.rename_close)

        layout.addLayout(form_lay)
        layout.addWidget(buttons)

        self.dgiw.setLayout(layout)
        self.dgiw.setWindowTitle('Rename axis')
        self.dgiw.setWindowModality(Qt.ApplicationModal)
        self.dgiw.exec_()

    def rename_close(self):
        self.dgiw.close()

    def rename_accept(self, labels):
        for cl, lab in enumerate(labels):
            if lab != "":
                if cl == 0:
                    self.canvas.axes.set_xlabel(labels[cl].text())
                elif cl == 1:
                    self.canvas.axes.set_ylabel(labels[cl].text())
                elif cl == 2:
                    self.savnac.axes.set_xlabel(labels[cl].text())
                else:
                    self.savnac.axes.set_ylabel(labels[cl].text())
        self.canvas.draw_idle()
        self.savnac.draw_idle()
        self.dgiw.close()

    def reject(self):
        return None

    def simplify_number(self, number):
        if number < 0:
            number = np.round(number, 4)
        elif number < 20:
            number = np.round(number, 2)
        else:
            number = int(number)

        return number

    # Allow to keep center fixed (with checkbox)
    def add_fitting_data_to_gui(self):
        fv = self.fit_vals
        ke = fv.keys()
        # print(ke)

        row = 1
        cou = 1
        col = 0
        extra = 0
        for cc, key in enumerate(ke):
            box_name = self.combo_mod[row - 1][1].currentText()
            # print(box_name)

            # This part sets the lenght of parameters and the number of skipped ones
            if box_name == "":
                mod = 1
                extra = 0
            elif box_name == "Polynomial":
                # print(self.constraints[row-1][0][0].text())
                mod = int(self.constraints[row - 1][0][0].text()) + 2
                extra = int(self.constraints[row - 1][0][0].text()) + 2
            elif box_name in ["Linear", "Exponential"]:
                mod = 3
                extra = 0
            elif box_name in ["Gaussian", "Lorentzian"]:
                mod = 6
                extra = 2
            elif box_name in ["PseudoVoigt", "ExpGaussian", "SkewedGaussian", "SkewedVoigt", "Voigt"]:
                mod = 7
                extra = 3
            else:
                pass

            if cou % mod == 0:
                col = 0
                cou = 1
                row += 1
                extra = 0
            else:
                pass

            try:  # To remove old fitting value on GUI
                self.LGfit.itemAtPosition(row * 2, col * 2 + 4).widget().deleteLater()
            except:
                pass

            if mod - col - 1 <= extra:
                pass
            else:
                val = str(self.simplify_number(fv[key]))
                labl = QLabel(val)
                labl.setAlignment(Qt.AlignCenter)
                self.LGfit.addWidget(labl, row * 2, col * 2 + 4)
            col += 1
            cou += 1

    def popup_heatplot_color_range(self):
        dgiw = QDialog()
        Lopt = QVBoxLayout()
        Lopt.setAlignment(Qt.AlignCenter)

        Tdats = QLabel("Select a new color range for the heaplot")
        Lopt.addWidget(Tdats)

        Tempt = QLabel("\n")
        Lopt.addWidget(Tempt)

        max_val = round(self.mod_data.to_numpy().max(), 2)
        min_val = round(self.mod_data.to_numpy().min(), 2)

        self.cb_max = QLineEdit()
        self.cb_max.setFixedWidth(100)
        self.cb_max.setAlignment(Qt.AlignCenter)
        self.cb_max.setText(str(max_val))
        self.cb_min = QLineEdit()
        self.cb_min.setFixedWidth(100)
        self.cb_min.setAlignment(Qt.AlignCenter)
        self.cb_min.setText(str(min_val))

        Lvalues = QFormLayout()
        Lvalues.addRow("Upper Boundary: ", self.cb_max)
        Lvalues.addRow("Lower Boundary: ", self.cb_min)

        Lopt.addLayout(Lvalues)
        Bok = QDialogButtonBox(QDialogButtonBox.Ok)
        Lopt.addWidget(Bok)
        Bok.accepted.connect(self.set_heaplot_color_range)

        dgiw.setLayout(Lopt)
        dgiw.setWindowTitle("Select boundaries")
        dgiw.setWindowModality(Qt.ApplicationModal)
        dgiw.exec_()

    def set_heaplot_color_range(self):
        min_val = float(self.cb_min.text())
        max_val = float(self.cb_max.text())
        self._plot_heat.set_clim(min_val, max_val)
        self.scrollbar_action()

    def slider_action(self):
        sli1, sli2 = self.range_slider.value()

        try:
            self._plot_hline1.set_ydata([sli1, sli1])
            self._plot_hline2.set_ydata([sli2, sli2])

            self.mod_data = self.mod_data.iloc[sli1:sli2 + 1]

            self.scrollbar_action()
        except:
            pass

    def save_current_matrix_state(self):
        self.statusBar().showMessage("Saving file, please wait...")
        try:
            fi, le = self.dummy_folderpath_file.rsplit("/", 1)
        except:
            fi, le = self.file_path.rsplit("/", 1)

        filename = \
            QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', directory=fi,
                                                  filter="CSV (*.csv) ;; Excel (*.xlsx)")[0]

        # try:
        if ".xlsx" in filename:
            self.mod_data.to_excel(filename)
        else:
            self.mod_data.to_csv(filename)
        # except:
        #     if ".xlsx" in filename:
        #         self.init_data.to_excel(filename)
        #     else:
        #         self.init_data.to_csv(filename)

        self.statusBar().showMessage("File saved!", 5000)

    def popup_error_msg(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("There are no changes")
        msg.setInformativeText('The dataset has not been modified')
        msg.setWindowTitle("Error")
        msg.exec_()

    def scrollbar_action(self):
        bar = int(self.ScrollbarTime.value())  # Read scrollbar value
        self.bar_update_plots(bar)

    def bar_update_plots(self, bar):
        # try:
        self._plot_ref.set_xdata(self.mod_data.index.values)
        self._plot_ref.set_ydata(self.mod_data.iloc[:, [bar]].T.values[0])
        # except:
        #     self._plot_ref.set_ydata(self.init_data.iloc[:, [bar]].T.values[0])
        try:
            time = str(round(float(self.xtime[bar]), 1))
        except:
            time = str(self.xtime[bar])
        self.text_time.set_text(self.t_label + " " + time)
        self.text_pos.set_text("Position " + str(bar))

        self._plot_vline.set_xdata([bar, bar])
        self.canvas.draw_idle()
        self.savnac.draw_idle()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
