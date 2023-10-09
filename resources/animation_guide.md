# Animation Maker
_An Addendum to Spectra Analyzer for Building Animations_

## Guide to Building Animations
___

### 1. Selecting the raw data - _Raw data file(Matrix)_:
- **File Type**: Utilize a file containing a single matrix. Accepted formats include `.txt`, `.csv`, and `.xlsx`.
- **How to Load**: Simply press **Open** and navigate to the desired file.

### 2. Loading the fitting results - _Fit data file_:
- **Requirement**: This must be an `.xlsx` file, structured similarly to the fitting output from Spectra Analyzer.
- **Automatic Detection**: The application will autonomously search for **"Fitting"** folders, auto-populating the field if a valid file is found.

### 3. Specifying Axis Names:
- **Labels**: The x-axis and y-axis are labeled "Wavelength (nm)" and "Intensity (a.u.)" by default respectively.
- **Modification**: Ensure to adjust these labels as per your experimental parameters.

### 4. Curve Management - _Curves found_:
- **Visualization**: If the fit data file is read successfully (see Step 2), identifiable curves will be listed here.
- **Selection**: Deselect the checkbox for any curve you wish to exclude from the animation.

### 5. Setting Up Animation:
- **Frame Range**: Define the starting and ending frames for your animation.
- **Playback Speed**: Set the FPS (frames per second) to control the animation speed.

### 6. Animation Output:
- **Format**: All animations are exported as GIF files by default.

### 7. Initiating Animation Creation:
- **Action**: Click "Create" once all settings are configured to your liking.

### 8. Completion Notification:
- **Indicator**: After the process has completed, "Finished" will be displayed in the bottom-left corner.

> **Warning**
> A progress bar is not currently available. After starting the process, please patiently wait for the animation to be completed. If using the python code version of the app, progress can be monitored in the console.
