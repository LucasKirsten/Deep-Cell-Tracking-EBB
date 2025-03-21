# Detecting and Tracking Cells in Microscopic Images using Oriented Representations

This repository contains the code for the following papers:
- [Cell Tracking-by-detection using Elliptical Bounding Boxes (Lucas Kirsten and Cláudio Jung)](https://www.sciencedirect.com/science/article/pii/S1047320325000392)

-  [Oriented Cell Dataset: efficient imagery analyses using angular representation (Lucas Kirsten et al.)](https://openaccess.thecvf.com/content/WACV2025/papers/Kirsten_Oriented_Cell_Dataset_A_Dataset_and_Benchmark_for_Oriented_Cell_WACV_2025_paper.pdf)

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
@article{KIRSTEN2025104425,
    title   = {Cell tracking-by-detection using elliptical bounding boxes},
    journal = {Journal of Visual Communication and Image Representation},
    pages   = {104425},
    year    = {2025},
    issn    = {1047-3203},
    doi     = {https://doi.org/10.1016/j.jvcir.2025.104425},
    url     = {https://www.sciencedirect.com/science/article/pii/S1047320325000392},
    author  = {Lucas N. Kirsten and Cláudio R. Jung},
}
```

For the Oriented Cell dataset (OCD):

```
@InProceedings{Kirsten_2025_WACV,
    author    = {Kirsten, Lucas and Angonezi, Angelo and Marques, Jose and Oliveira, Fernanda and Faccioni, Juliano and Cassel, Camila and de Sousa, D\'ebora and Vedovatto, Samlai and Lenz, Guido and Jung, Claudio},
    title     = {Oriented Cell Dataset: A Dataset and Benchmark for Oriented Cell Detection and Applications},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {3996-4005}
}
```
