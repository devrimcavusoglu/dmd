#!/bin/bash

mkdir -p data

# Download Distillation dataset in HDF5 format
wget https://huggingface.co/datasets/Devrim/dmd_cifar10_edm_distillation_dataset/resolve/main/cifar.hdf5?download=true -O data/cifar_toy.hdf5
wget https://huggingface.co/datasets/Devrim/dmd_cifar10_edm_distillation_dataset/resolve/main/cifar_toy.hdf5?download=true -O data/cifar_toy.hdf5


# Download vanilla zip dataset
# wget https://huggingface.co/datasets/Devrim/dmd_cifar10_edm_distillation_dataset/resolve/main/distillation_dataset.zip?download=true -O data/distillation_dataset.zip
#cd data
#unzip -oq distillation_dataset.zip
#rm -rf distillation_dataset.zip
