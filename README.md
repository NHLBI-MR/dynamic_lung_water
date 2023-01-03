# Dynamic lung water magnetic resonance imaging during exercise stress
This code is for the manuscript (in sub): Seemann, F., et al. Dynamic lung water magnetic resonance imaging during exercise stress

## This repository contains 
- Matlab pipeline code for inline reconstruction and analysis
- Python processing for lung segmentation for analysis
- Image registration package

# NHLBI TOOLBOX GADGETRON Installation
Repository with gadgetron image reconstruction code: https://github.com/NHLBI-MR/selfgated_noncartesian_reconstruction.git

% GADGETRON MATLAB
- Install application through matlab
- https://www.mathworks.com/matlabcentral/fileexchange/72715-gadgetron
- Make directory "+nhlbi" inside the +gadgetron class

# Requires python packages 
- numpy
- matplotlib
- torch
- torchvision
- scipy

pip install numpy matplotlib torch torchvision scipy
the trained neural network is named best_lung_seg_model.pkl and is saved in the python folder
