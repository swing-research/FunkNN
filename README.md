# FunkNN: Neural Interpolation for Functional Generation


[![Paper](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2212.14042)
[![PWC](https://img.shields.io/badge/PWC-report-blue)](https://paperswithcode.com/paper/funknn-neural-interpolation-for-functional)

This repository is the official Pytorch implementation of "[FunkNN: Neural Interpolation for Functional Generation](https://arxiv.org/abs/2212.14042)".

| [**Project Page**]()  | 


<p float="center">
<img src="https://github.com/swing-research/FunkNN/blob/main/figures/network.png" width="1000">
</p>



## Requirements
(This code is tested with PyTorch 1.12.1, Python 3.8.3, CUDA 11.6 and cuDNN 7.)
- numpy
- scipy
- matplotlib
- odl
- imageio
- torch==1.12.1
- torchvision=0.13.1
- astra-toolbox

## Installation

Run the following code to install conda environment "environment.yml":
```sh
conda env create -f environment.yml
```

## Datasets
You can download the [CelebA-HQ](https://drive.switch.ch/index.php/s/pA6X3TY9x4jgcxb), [LoDoPaB-CT](https://drive.switch.ch/index.php/s/lQeYWmAIYcEEdlc) and [LSUN-bedroom](https://drive.switch.ch/index.php/s/d1MNcrUZkPpK0zx) validation datasets and split them into train and test sets and put them in the data folder. You should specify the data folder addresses in config_funknn.py and config_generative.py.

## Experiments
### Train FunkNN
All arguments for training FunkNN model are explained in config_funknn.py. After specifying your arguments, you can run the following command to train the model:
```sh
python3 train_funknn.py 
```

### Train generative autoencoder
All arguments for training generative autoencoder are explained in config_generative.py. After specifying your arguments, you can run the following command to train the model:
```sh
python3 train_generative.py
```


### Solving inverse problems
All arguments for solving inverse problem by combining FunkNN and generative autoencoder are explained in config_IP_solver.py. After specifying your arguments including the folder address of trained FunkNN and generator, you can run the following command to solve the inverse problem of your choice (CT or PDE):
```sh
python3 IP_solver.py
```

## Citation
If you find the code useful in your research, please consider citing the paper.

```
@article{khorashadizadeh2022funknn,
  title={FunkNN: Neural Interpolation for Functional Generation},
  author={Khorashadizadeh, AmirEhsan and Chaman, Anadi and Debarnot, Valentin and Dokmani{\'c}, Ivan},
  journal={arXiv preprint arXiv:2212.14042},
  year={2022}
}
```

