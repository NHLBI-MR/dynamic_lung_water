%% Sliding window moco recon

clc
close all
clear all

dirPath='C:\Users\User\Documents\folder\file.h5'; %full path to the h5-file to be reconstructed
data=ute_slwindow_moco_recon(filename); %the output data is a struct containing the images and lung water densities

disp('Done')

