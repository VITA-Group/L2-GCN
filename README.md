# L<sup>2</sup>-GCN: Layer-Wise and Learned Efficient Training of Graph Convolutional Networks

## Overview

Graph convolution networks (GCN) are increasingly popular in many applications, yet remain notoriously hard to train over large graph datasets. They need to compute node representations recursively from their neighbors. Current GCN training algorithms suffer from either high computational costs that grow exponentially with the number of layers, or high memory usage for loading the entire graph and node embeddings. In this paper, we propose a novel efficient layer-wise training framework for GCN (L-GCN), that disentangles feature aggregation and feature transformation during training, hence greatly reducing time and memory complexities. We present theoretical analysis for L-GCN under the graph isomorphism framework, that L-GCN leads to as powerful GCNs as the more costly conventional training algorithm does, under mild conditions. We further propose L<sup>2</sup>-GCN, which learns a controller for each layer that can automatically adjust the training epochs per layer in L-GCN. Our paper is available at [here](https://arxiv.org/abs/2003.13606).

## Citation

If you are use this code for you research, please cite our paper.

```
@inproceedings{you2020l2,
  title={L$^2$-GCN: Layer-Wise and Learned Efficient Training of Graph Convolutional Networks},
  author={You, Yuning and Chen, Tianlong and Wang, Zhangyang and Shen, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2127--2135},
  year={2020}
}
```

## Dependencies

* torch == 1.3.1
* numpy == 1.17.2
* scipy == 1.4.1

## Run

Download data from https://drive.google.com/file/d/1amUK4zWRxBsVyi3eqCHQFTtaWtmm67hW/view?usp=sharing.

**L-GCN**

```shell
python -m l2_gcn.main --dataset cora
python -m l2_gcn2.main_ppi --dataset ppi
```

**L<sup>2</sup>-GCN**

```shell
python -m l2_gcn.main_l2o --dataset cora
python -m l2_gcn2.main_l2o_ppi --dataset ppi
```


