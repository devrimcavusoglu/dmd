# One-step Diffusion with Distribution Matching Distillation
<a href="https://paperswithcode.com/paper/one-step-diffusion-with-distribution-matching"><img src="https://img.shields.io/badge/DMD-temp?style=square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCIgdmlld0JveD0iMCAwIDUxMiA1MTIiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDUxMiA1MTI7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4gPHN0eWxlIHR5cGU9InRleHQvY3NzIj4gLnN0MHtmaWxsOiMyMUYwRjM7fSA8L3N0eWxlPiA8cGF0aCBjbGFzcz0ic3QwIiBkPSJNODgsMTI4aDQ4djI1Nkg4OFYxMjh6IE0yMzIsMTI4aDQ4djI1NmgtNDhWMTI4eiBNMTYwLDE0NGg0OHYyMjRoLTQ4VjE0NHogTTMwNCwxNDRoNDh2MjI0aC00OFYxNDR6IE0zNzYsMTI4IGg0OHYyNTZoLTQ4VjEyOHoiLz4gPHBhdGggY2xhc3M9InN0MCIgZD0iTTEwNCwxMDRWNTZIMTZ2NDAwaDg4di00OEg2NFYxMDRIMTA0eiBNNDA4LDU2djQ4aDQwdjMwNGgtNDB2NDhoODhWNTZINDA4eiIvPjwvc3ZnPg%3D%3D&label=paperswithcode&labelColor=%23555&color=%2321b3b6&link=https%3A%2F%2Fpaperswithcode.com%2Fpaper%2Fone-step-diffusion-with-distribution-matching" alt="DMD Implementation"></a>

A PyTorch implementation of the paper [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828). This 
project codebase is mostly based on the codebase of [EDM from NVLabs](https://github.com/NVlabs/edm) and built on top of it with according 
modifications.

Note that this is an unofficial reimplementation study for the paper, and in this codebase we focused on experimenting 
with CIFAR-10 dataset reproduce the results. However, the technique may be applicable to other datasets with minor 
adjustments.

## Setup

Create a conda environment with the configuration file, and activate the environment when necessary.

```shell
conda create env -f environment.yml
```

### Models

DMD method is an application of distillation, and thus requires a teacher model. The teacher diffusion model 
used in the paper was [EDM](https://github.com/NVlabs/edm) models. Specifically, for CIFAR-10 we will focus on 
a conditioned model. You can see pretrained EDM Models [here](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/).

### Dataset

Being Prepared...

## Training
WIP.

## Generation
WIP.

## License

Copyright © 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.\
Copyright © 2024, Devrim Cavusoglu & Ahmet Burak Yıldırım

This work contains the implementation of the methodology and study presented in the *One-step Diffusion with 
Distribution Matching Distillation* paper. Also as the building block of the codebase, [NVLabs/edm](https://github.com/NVlabs/edm) is 
used, modified and adapted accordingly when necessary. As the original license of the underlying framework (edm) 
dictates (ShareAlike), this derived work and all the source are licensed under the same license 
of [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
