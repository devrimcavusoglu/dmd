#!/bin/bash

mkdir -p data
mkdir -p models

# Download Distillation dataset in HDF5 format
wget https://huggingface.co/datasets/Devrim/dmd_cifar10_edm_distillation_dataset/resolve/main/cifar.hdf5?download=true -O data/cifar.hdf5
wget https://huggingface.co/datasets/Devrim/dmd_cifar10_edm_distillation_dataset/resolve/main/cifar_toy.hdf5?download=true -O data/cifar_toy.hdf5

wget https://huggingface.co/Devrim/dmd-cifar-10-cond/resolve/main/model.pt?download=true -O models/dmd_cifar_10_cond.pt

# Download vanilla zip dataset
# wget https://huggingface.co/datasets/Devrim/dmd_cifar10_edm_distillation_dataset/resolve/main/distillation_dataset.zip?download=true -O data/distillation_dataset.zip
#cd data
#unzip -oq distillation_dataset.zip
#rm -rf distillation_dataset.zip

# By default the pipeline automatically downloads given URL, but to download the base model (EDM) to locale comment off the following line
# wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl -O models/edm-cifar10-32x32-cond-vp.pt
