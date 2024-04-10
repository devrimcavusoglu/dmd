#!/bin/bash

mkdir -p data

# Download Distillation dataset
wget https://huggingface.co/datasets/Devrim/dmd_cifar10_edm_distillation_dataset/resolve/main/distillation_dataset.zip?download=true -O data/distillation_dataset.zip

cd data
unzip -oq distillation_dataset.zip
rm -rf distillation_dataset.zip
