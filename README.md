# Spectra Analyzer
Graphical interface that speeds up the analysis of multiple spectras, whether PL or XRD, that are usually collected during in-situ measurements

If you find this program useful, please cite it DOI: 10.5281/zenodo.10370284

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10370284.svg)](https://doi.org/10.5281/zenodo.10370284)


## Description
The purpose of this program is to simplify the visualization and analysis of PL or XRD data. 
It automatically identifies and loads a large variety of data file formats, but includes a "manual" mode in case the displayed data is not what was expected.
Additionally, it allows the editing of the data, by removing background defects, or in the case of PL data, convert from nm to eV.
Using the lmfit library, it allows a multiprocess fitting, speeding up the analysis of multiple spectra. 
Supported models are, for background: linear, polynomial, and exponential; for data peaks: Gaussian, Lorentzian, Voigt, Pseudo Voigt, Exponential Gaussian, Skewed Gaussian, and Skewed Voigt.
Examples of supported data are included in this repository.

## Installation
Other than the libraries in the requirements.txt, pyqt is needed

If using anaconda, use:

```bash
conda install -c anaconda pyqt=5.12.3
```

else:

```bash
pip install pyqt=5.12.3
```

## Usage
Files can be loaded under the File menu. Supported data formats are located in "Data_examples" on this repository, with the corresponding name formats.

Time steps can be selected with the scroll bar. Similarly, a range of interest (e.g. Wavelength) can be selected with the vertical blue scroll bar on the left of the heatplot.

Models can be added with the plus sign button, or by going on the top menu Fit/Add model Line. Once a correct model and an (optional) name is selected, the Fit button on top can be used to have a fitting. It is recommended that all the fields (e.g. for Gaussian: Amplitude, Center, and Sigma) are filled, so that the fitting process is faster and to reduce divergent results.

For in-situ measurements, sometimes peaks red- or -blue shift. The program automatically follows this peaks. However, if the there are multiple close-from-each-other peaks (as it is the case of XRD measurements), one can select the "fix center" button, to force the fitting at that point.

For repeating measurements, multiple model parameters can be saved under Fit/Save fit parameters and loaded under Fit/Load fit parameters.

For multiple fitting of measurements, select the range of desired positions on the Start/End fields at the bottom. By pressing "Fit range", a multiprocess will be started, using as many cores as the cpu contains, to speed up the fitting process within this range.

Once the multiprocess is done, a new folder is created containing all of the resulting fitting parameters, as well as plots of the major parameters, e.g. amplitude, center, sigma, for the selected fitted range.

## Support
For help, contact enandayapa@gmail.com

## Roadmap
The program is mostly on a finished state, albeit some small bugs that need fixing. 
Please contact us if you find any new issues.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors and acknowledgment
All programming was done by Edgar Nandayapa B.
Field testing has been done by C. Rehermann and F. Mathies.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Known issues
Fitting by using the Voigt model makes the program crash
