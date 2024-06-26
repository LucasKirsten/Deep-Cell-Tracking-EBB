# Detecting and Tracking Cells in Microscopic Images using Oriented Representations

This repository contains the code for the [Cell Tracking-by-detection using Elliptical Bounding Boxes (Lucas N. Kirsten and Cláudio R. Jung)](https://arxiv.org/abs/2310.04895) paper.

## Topics

- [CTC Execution](#ctc-execution)
- [UFRGS - Oriented Cell Dataset Execution](#ufrgs---oriented-cell-dataset-execution)
- [Visual Results](#visual-results)
- [Citation](#citation)

## CTC Execution

You can execute the prediction on the [CTC](http://celltrackingchallenge.net/) images using the respective bash script with its name.

We also provide a [Google Colab notebook](https://github.com/LucasKirsten/Deep-Cell-Tracking-EBB/blob/master/ISBI_Cell_Tracking.ipynb) to execute the complete pipeline.

## UFRGS - Oriented Cell Dataset Execution

We provide the [weights](https://drive.google.com/drive/u/0/folders/13N4G9k1E6wO3-RWXQv0pXMI8-4bZeBKo) for the trained models, and a [bash script](https://github.com/LucasKirsten/Deep-Cell-Tracking-EBB/blob/master/RotationDetection/make_prediction.sh) for inference in the [UFRGS - Oriented Cell Dataset](https://ieee-dataport.org/documents/oriented-cell-dataset-ocd).

## Visual Results

Below is examples of the code execution for the [CTC](http://celltrackingchallenge.net/) datasets:

### GOWT1-01
![GOWT1-01](images/GOWT1-01.gif)

### U373-02
![U373-02](images/U373-02.gif)

### HeLa-02
![HeLa-02](images/HeLa-02.gif)

## Citation

If you find this repository useful for your research, please use the following citations.

For the Cell Tracking-by-Detection algorithm:
```
@article{kirten2023cell,
  title={Cell Tracking-by-detection using Elliptical Bounding Boxes},
  author={Kirten, Lucas N and Jung, Cl{\'a}udio R},
  journal={arXiv preprint arXiv:2310.04895},
  year={2023}
}

@article{kirsten2023detecting,
  title={Detecting and tracking cells in microscopic images using oriented representations},
  author={Kirsten, Lucas Nedel},
  year={2023}
}
```

For the UFRGS - Oriented Cell dataset (OCD):

```
@article{kirten2023oriented,
  url = {https://doi.org/10.1101/2024.04.05.588327},
  title={Oriented Cell Dataset: efficient imagery analyses using angular representation},
  author={Kirsten, Lucas and Angonezi , Angelo and Oliveira , Fernanda and Faccioni , Juliano and Cassel , Camila and Santos de Sousa, Débora and Vedovatto , Samlai and Jung, Cláudio and Lenz, Guido},
  year = {2024},
	doi = {10.1101/2024.04.05.588327},
	publisher = {Cold Spring Harbor Laboratory},
  journal = {bioRxiv}
}

@data{ocd_dataset,
  doi = {10.21227/3rk2-b882},
  url = {https://dx.doi.org/10.21227/3rk2-b882},
  author = {Angonezi , Angelo and Oliveira , Fernanda and Faccioni , Juliano and Cassel , Camila and Santos de Sousa, Débora and Vedovatto , Samlai and Lenz, Guido and Jung, Cláudio and Kirsten, Lucas},
  publisher = {IEEE Dataport},
  title = {Oriented Cell Dataset (OCD)},
  year = {2024}
}
```
