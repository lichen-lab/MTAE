# MTAE
Multi-task deep autoencoder improves the prediction for Alzheimer's disease progression using temporal DNA methylation data in peripheral blood


## Introduction
We develop two Multi-Task deep AutoEncoder based on Long Short-Term Memory AutoEncoder (MT-LSTMAE) and Convolutional AutoEncoder respectively, which take the advantage of capacity of LSTM and convolutional network for learning the dependency on temporal DNA methylation data, to predict AD progression using longitudinal DNA methylation data collected from peripheral blood. 

## Maintainer

Li Chen (li.chen1@ufl.edu)

## Requirements and Installation

MT-LSTMAE and MT-CAE is implemented by TensorFlow/Keras.

- Python 3.8
- Keras == 2.4.0
- TensorFlow == 2.3.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3
- seaborn >=0.9.0
- matplotlib >=3.1.0


Download MDeep:
```
git clone https://github.com/lichen-lab/MTAE
```


```

python main.py -h

usage: main.py [-h] [--hdf5 HDF5] [--comparetype COMPARETYPE]
               [--isnorm ISNORM] [--AEtype AETYPE] [--nrep NREP]
               [--methods METHODS [METHODS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --hdf5 HDF5           hdf5 file
  --comparetype COMPARETYPE
                        cn or mci
  --isnorm ISNORM       no or yes
  --AEtype AETYPE       define or custom
  --nrep NREP           number of experiments
  --methods METHODS [METHODS ...]
                        Methods used in comparison

```


## Example


```
python main.py \
    --hdf5 temporal500.h5  \
    --comparetype cn  \
    --isnorm no  \
    --nrep 10 \
    --methods LSTMAE CNNAE
```


