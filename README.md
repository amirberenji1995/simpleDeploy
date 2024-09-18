# Serving a Pytorch model as a RESTful API by FastAPI
This repository is a demonstration on how you can develop a simple endpoint using FastAPI to serve a Pytorch model as an independent service. The process is almost the same for any other intelligent model.

## 1. Introduction

Intelligent methods are developed to be used and Jupyter/Colab notebooks are far from being ideal for the production stage. On the other hand, deployment through **Model as a Service** or [MasS](https://medium.com/@pearcenjordan/what-is-maas-unlock-the-power-of-model-as-a-service-27ebbcaefce6) is one of the favourite options. The main aim of this repository is to deploy a Pytorch model as an individual service, using FastAPI. We start from structurizing a benchmark bearing fault diagnosis dataset; then, we design, implement, train and evaluate a deep learning model to diagnose the bearing. Next is to develop a RESTful API using FastAPI to serve the model. Last but not least, we deploy the whole application on [render.com](https://render.com/).

## 2. Data

Data is the critical ingridient of every data-driven solution. I use the  Case Western Reverse University [bearing dataset](https://engineering.case.edu/bearingdatacenter); this dataset includes signals from both drive-end and fan-end bearings but we focus on the drive-end signals in this implementation. Similar to other benchmark datasets, the raw data is presented as super long signals (e.g. 122281-points long). Hence, the starting point is to divide these raw signals into 2048-points signals with a hop-lenght of 2048. Metadata (load, health state and fault severity) are also extracted in this stage. To achieve smoother convergence, time series need to be scaled; to do so, every signal is subtracted from its mean and divided by its standard deviation, as illustrated in the following equation:

$\widetilde{x} = \frac{x - \mu}{\sigma}$

where $\widetilde{x}$ is the scaled time serie, $x$ is the raw time serie, $\mu$ is the mean of the $x$ and $\sigma$ is its standard deviation. In the [scaler.py]() file, the implementation of the scaling operation can be found.

Next, is to split the dataset into train and test subsets, using a 70:30 ratio. It worth mentioning that as the scaling transorm in this implementation is instance-specific, it is safe to first scale the dataset and then split into train/test subsets.

## 3. Model

The problem we aim to solve is an example of time series classification and Convolutional Neural Networks (CNNs) are one of the favorite model types to engage such problem. I use a one-dimensional CNN consisting of a wide kernel convolutional layer, followed by an average pooling layer and a linear layer, as illustrated in the following figure.

![model architecture](/assets/architecture.jpg)

Although this model looks too simple at the first glance, it achieves 100% accuracy over the hold-out test set. You can see the training curves and the confusion matrix at the following figures.

<img src="assets/training_curves.jpg" height="250" width = 790/>

<img src="assets/confusion_matrix.jpg" height="250" width = 270/>

For details on how the training is done, you can check the [training notebook]().

## 4. API



## 5. Deployment