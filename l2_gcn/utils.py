import torch
import torch.utils.data as d

import json
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler
import random
import sys
from fastgcn_utils.utils import *


class feeder(d.Dataset):

    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        data = self.feat[index]
        label = self.label[index]
        return data, label


class feeder_sample(d.Dataset):

    def __init__(self, feat, label, train, total_round, sample_node_num):
        self.feat = feat
        self.label = label
        self.train = train
        self.total_round = total_round
        self.sample_node_num = sample_node_num

    def __len__(self):
        return self.total_round

    def __getitem__(self, index):

        train_sample = random.sample(list(self.train), self.sample_node_num)
        data = np.array(self.feat[train_sample])
        label = self.label[train_sample]

        return data, label, train_sample, index


# dataset loader should return:
#   1. feature
#   2. label
#   3. Adj_hat matrix
#   4. dataset split: [train, val, test]


def cora_loader():

    dataset_str = 'cora'

    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    feat_data = features.todense()
    Adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) + sps.eye(2708).tocsr()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.where(labels)[1].astype(np.int64)
    # assert False

    dataset_split = {}
    dataset_split['test'] = np.array(test_idx_range.tolist())
    dataset_split['train'] = np.array(list(range(140)) + list(range(140 + 500, 1708)))
    dataset_split['val'] = np.array(range(140, 140 + 500))

    return feat_data, labels, Adj, dataset_split


def pubmed_loader():

    dataset_str = 'pubmed'

    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sps.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    feat_data = features.todense()
    Adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) + sps.eye(19717).tocsr()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.where(labels)[1].astype(np.int64)
    # assert False

    dataset_split = {}
    dataset_split['test'] = np.array(test_idx_range.tolist())
    dataset_split['train'] = np.array(list(range(60)) + list(range(60 + 500, 18217)))
    dataset_split['val'] = np.array(range(60, 60 + 500))

    return feat_data, labels, Adj, dataset_split

