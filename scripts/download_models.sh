#!/bin/bash

mkdir -p models

# Download CIFAR-10 Condition VP model
wget https://huggingface.co/Devrim/edm-cifar10-32x32-cond-vp/resolve/main/edm-cifar10-32x32-cond-vp.pt?download=true -O models/edm-cifar10-32x32-cond-vp.pt
