# Detecting and Tracking Cells in Microscopic Images using Oriented Representations

This repository contains the code for the following papers:
- [Cell Tracking-by-detection using Elliptical Bounding Boxes (Lucas Kirsten and Cláudio Jung)](https://arxiv.org/abs/2310.04895)

-  [Oriented Cell Dataset: efficient imagery analyses using angular representation (Lucas Kirsten et al.)](https://www.biorxiv.org/content/10.1101/2024.04.05.588327v1)

## Topics

- [CTC Execution](#ctc-execution)
- [Oriented Cell Dataset (OCD)](#oriented-cell-dataset-ocd)
- [Visual Results](#visual-results)
- [Citation](#citation)

## CTC Execution

You can execute the prediction on the [CTC](http://celltrackingchallenge.net/) images using the respective bash script with its name.

We also provide a [Google Colab notebook](https://github.com/LucasKirsten/Deep-Cell-Tracking-EBB/blob/master/ISBI_Cell_Tracking.ipynb) to execute the complete pipeline.

## Oriented Cell Dataset (OCD)

*[LEGACY] We provide the [weights](https://drive.google.com/drive/u/0/folders/13N4G9k1E6wO3-RWXQv0pXMI8-4bZeBKo) for the trained models, and a [bash script](https://github.com/LucasKirsten/Deep-Cell-Tracking-EBB/blob/master/RotationDetection/make_prediction.sh) for inference in the [OCD](https://ieee-dataport.org/documents/oriented-cell-dataset-ocd).*

The OCD dataset is available in the following: [IEEE-DataPort](https://ieee-dataport.org/documents/oriented-cell-dataset-ocd), [GoogleDrive](https://drive.google.com/drive/folders/1vREKlRz9QkSWrUApkZamv_oUrw3tOFI3?usp=drive_link)

For updated code regarding training OCD with [MMRotate](https://github.com/open-mmlab/mmrotate) please refer to [here](https://github.com/jhlmarques/OCDDataset).

## Visual Results

Below are examples of the code execution for the [CTC](http://celltrackingchallenge.net/) datasets:

### GOWT1-01
![GOWT1-01](images/GOWT1-01.gif)

### U373-02
![U373-02](images/U373-02.gif)

### HeLa-02
![HeLa-02](images/HeLa-02.gif)

## Citation

If you find this repository useful for your research, please consider citing.

For the Cell Tracking-by-Detection algorithm:
```
@article{kirten2023cell,
  title={Cell Tracking-by-detection using Elliptical Bounding Boxes},
  author={Kirten, Lucas N and Jung, Cl{\'a}udio R},
  journal={arXiv preprint arXiv:2310.04895},
  year={2023}
}
```

For the Oriented Cell dataset (OCD):

```
@article{kirten2023oriented,
  title={Oriented Cell Dataset: A Dataset and Benchmark for Oriented Cell Detection and Applications},
  author={Kirsten, Lucas and Angonezi, Angelo and Marques, Jose and Oliveira, Fernanda and Faccioni, Juliano and Cassel, Camila and Santos de Sousa, Débora and Vedovatto, Samlai and Lenz, Guido and Jung, Cl{\'a}udio},
  journal={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025)},
  year={2025}
}
```
