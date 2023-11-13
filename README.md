# Detecting and Tracking Cells in Microscopic Images using Oriented Representations

This repository contains the code for the [Cell Tracking-by-detection using Elliptical Bounding Boxes (Lucas N. Kirsten and Cláudio R. Jung)](https://arxiv.org/abs/2310.04895) paper.

## Topics

- [CTC Execution](#ctc-execution)
- [UFRGS - Oriented Cell Dataset Execution](#ufrgs---oriented-cell-dataset-execution)
- [Visual Results](#visual-results)

## CTC Execution

You can execute the prediction on the [CTC](http://celltrackingchallenge.net/) images using the respective bash script with its name.

We also provide a [Google Colab notebook](https://github.com/LucasKirsten/Deep-Cell-Tracking-EBB/blob/master/ISBI_Cell_Tracking.ipynb) to execute the complete pipeline.

## UFRGS - Oriented Cell Dataset Execution

We provide the [weights](https://drive.google.com/drive/u/0/folders/13N4G9k1E6wO3-RWXQv0pXMI8-4bZeBKo) for the trained models, and a [bash script](https://github.com/LucasKirsten/Deep-Cell-Tracking-EBB/blob/master/RotationDetection/make_prediction.sh) for inference in the [UFRGS - Oriented Cell Dataset](https://drive.google.com/drive/folders/1Ua8OExEh12u4YiXfAErNpTPspOo6RQr2?usp=drive_link).

## Visual Results

Below is examples of the code execution for the [CTC](http://celltrackingchallenge.net/) datasets:

### GOWT1-01
![GOWT1-01](images/GOWT1-01.gif)

### U373-02
![U373-02](images/U373-02.gif)

### HeLa-02
![HeLa-02](images/HeLa-02.gif)
