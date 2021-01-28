## Neuroscience Group Report

Paper: 

Simulating calcium signals from mice visual cortex with GANs and inferring the functional connectivity of the biological neurons with Spike Neural Network (SNN)

## What is this project?

This is the code base for CS280@ShanghaiTech University's Final Project. 

Our project lies in the intersection of neural science and computer science, called computational neuroscience. To uncover our magic brain, two urgent questions need to be answered: how to find more samples and how to apply appropriate mathematical tools to analyze those data.
In this paper, we proposed a method named "neuron-GAN" to produce more neuronal population signals and at the same time,we proposed implemented a Spike Neural Network(SNN) with explicit inhibitory neurons and learnable synaptic weights to infer neuronal  functional  connectivities.

## 2. Prerequisites 

### 2.1 SNN 


- Matlab 2019.a

### 2.2 Neuro-GAN 


- python 3.6
- [Elephant](https://github.com/NeuralEnsemble/elephant): for neuronal signal analysis
- Pytorch 1.7.0
- [OASIS](https://github.com/j-friedrich/OASIS): for fast online deconvolution
- Numpy, Seanborn, Sklearn

## 3. Code Structure



- dataset: contains data for training model
- images: result from visualization
- NeuroGAN: model for NeuroGAN part
- SNN: model for SNN part

## 4. Dataset description  


All data are collected in Guanlab@Shanghaitech SLST.

We only release a subset of the total recorded dataset in ```/dataset```  folder due to academic privacy reasons (yet enough for training a model). For educational purpose please contact Guanlab.

## 5. Experiments

### 5.1 How to Run SNN Experiments

1. Copy dataset to working directory: ```cp -r dataset/SNN/* ./SNN```
2. Run SNN.m to experiment with SNN
3. Run pre_inhibitory_SVM.m to classify excitatory neurons and inhibitory neurons with SVM

### 5.2 How to Run Neuro-GAN Experiments


1. copy dataset to working directory: ```cp -r dataset/GAN/* ./NeuroGAN ```

2. Train the Neuro-GAN model: ```python main.py -n_epochs -batch_size -lr -b1 -b2 -n_cpu -latent_dim - sample_interval ```

3. Train a SVM classifier based on recorded and generated data: ```python svm.py```

4. Compare the Mean Firing Rate among all the neuron populations: ```python mfr_comparison.py```

5. Visualize the inception score of batch size up to N: ```python inception.py N ```

   

## 6. Acknowledgement  

We would like to thank Prof.Jisong Guan, Dr.Guangyu Wang, Kaiyuan Liu, Chenhui Liu for providing the raw data and sharing their insightful opinions with us.

