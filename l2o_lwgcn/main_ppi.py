import torch
import torch.nn as nn

import numpy as np
import time
import random
from sklearn.metrics import f1_score
import os
import yaml
import argparse
import scipy.sparse as sps

import l2o_lwgcn.utils as utils
import l2o_lwgcn.net as net


def run(dataset_load_func, parse_args, seed):

    ###############################
    # config setting              #
    ###############################

    # dataset load
    feat_data, labels, Adj, dataset_split = dataset_load_func()
    print('Finished loading dataset.')

    # config parameter load
    config_file = parse_args['config_file']
    with open('./config/' + config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    for parg_key in parse_args.keys():
        if (not parg_key in args.keys()) or parse_args[parg_key] == None:
            continue
        args[parg_key] = parse_args[parg_key]
    print(args)

    # result file
    result_file = './result/' + config_file[:-5] + str(args['layer_num']) + '_layer_' + str(seed) + '.npy'
    result_loss_data = []

    ##############################################################
    ##############################################################

    ###############################
    # training initialization     #
    ###############################

    # train, val and test set
    train = dataset_split['train']
    train.sort()
    val = dataset_split['val']
    val.sort()
    test = dataset_split['test']
    test.sort()

    # feature and label generate
    num_nodes = args['node_num']
    feat_train = torch.FloatTensor(feat_data)[train, :]
    label_train = labels[train]
    feat_val = torch.FloatTensor(feat_data)[val, :]
    feat_test = torch.FloatTensor(feat_data)[test, :]

    # Adj matrix generate
    Adj_eye = sps.eye(num_nodes, dtype=np.float32).tocsr()
    Adj_train = Adj[train, :][:, train]
    D_train = Adj_train.sum(axis=0)
    Adj_train = Adj_train.multiply(1/D_train.transpose())
    Adj_train = Adj_train + Adj_eye[train, :][:, train]
    Adj_val = Adj[val, :][:, val]
    D_val = Adj_val.sum(axis=0)
    Adj_val = Adj_val.multiply(1/D_val.transpose())
    Adj_val = Adj_val + Adj_eye[val, :][:, val]
    Adj_test = Adj[test, :][:, test]
    D_test = Adj_test.sum(axis=0)
    Adj_test = Adj_test.multiply(1/D_test.transpose())
    Adj_test = Adj_test + Adj_eye[test, :][:, test]

    ##############################################################
    ##############################################################

    ###############################
    # start training              #
    ###############################

    # layerwise training
    times = 0
    epochs = 0

    net_lwgcn = net.net_lwgcn(in_channel=args['feat_dim'], hidden_channel=args['hidden_dim'], out_channel=args['class_num'], layer_num=args['layer_num'])
    optimizer = torch.optim.Adam(net_lwgcn.parameters(), lr=args['learning_rate'])
    loss_func = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    for l in range(args['layer_num']):

        print('layer ' + str(l+1) + ' training:')

        feat_train = feat_train.numpy()
        start_time = time.time()
        feat_train = Adj_train.dot(feat_train)
        end_time = time.time()
        times = times + ( end_time - start_time )
        feat_train = torch.FloatTensor(feat_train)

        feat_val = feat_val.numpy()
        feat_val = Adj_val.dot(feat_val)
        feat_val = torch.FloatTensor(feat_val) ###

        # create data loader
        feeder_train = utils.feeder(feat_train, label_train)
        dataset_train = torch.utils.data.DataLoader(dataset=feeder_train, batch_size=args['batch_size'], shuffle=True, drop_last=True)
        net_lwgcn = net_lwgcn.cuda()

        batch = 0
        while True:
            for x, x_label in dataset_train:

                x = x.cuda()
                x_label = x_label.cuda()

                start_time = time.time()
                optimizer.zero_grad()
                output = net_lwgcn(x, layer_index=l)
                output = sigmoid(output)
                loss = loss_func(output, x_label)
                loss.backward()
                optimizer.step()
                end_time = time.time()
                times = times + ( end_time - start_time )

            result_loss_data.append(loss.data.cpu().numpy())
            batch = batch + 1
            epochs = epochs + 1
            print('batch', batch, 'loss:', loss.data)

            if batch % 50 == 0: ###
                with torch.no_grad():
                    output = net_lwgcn(feat_val.cuda(), layer_index=l)
                    output = sigmoid(output)
                    print('val loss:', loss_func(output, torch.FloatTensor(labels[val]).cuda()).data)
        
            if batch == args['epoch_num'][l]:

                net_lwgcn = net_lwgcn.cpu()
                if l != args['layer_num'] - 1:
                    start_time = time.time()
                    feat_train = net_lwgcn(feat_train, layer_index=l, with_classifier=False).detach()
                    feat_val = net_lwgcn(feat_val, layer_index=l, with_classifier=False).detach() ###
                    end_time = time.time()
                    times = times + ( end_time - start_time )
                break

    os.system('nvidia-smi')
    np.save(result_file, np.array(result_loss_data))

    ##############################################################
    ##############################################################

    ###############################
    # val or test                 #
    ###############################

    # val or test
    with torch.no_grad():
        output_val = net_lwgcn.val_test(feat_val, Adj_val)
        output_val[output_val>0] = 1
        output_val[output_val<=0] = 0
        output_test = net_lwgcn.val_test(feat_test, Adj_test)
        output_test[output_test>0] = 1
        output_test[output_test<=0] = 0

    print("accuracy in val:", f1_score(labels[val], output_val.data.numpy(), average="micro"))
    print("accuracy in test:", f1_score(labels[test], output_test.data.numpy(), average="micro"))
    print("average epoch time:", times / epochs)
    print("total time:", times)

    return f1_score(labels[test], output_test.data.numpy(), average="micro"), times

    ##############################################################
    ##############################################################


def parser_loader():

    parser = argparse.ArgumentParser(description='L2O_LWGCN')
    parser.add_argument('--dataset', type=str, default='ppi')
    parser.add_argument('--config-file', type=str, default='ppi.yaml')
    parser.add_argument('--layer-num', type=int, default=None)
    parser.add_argument('--epoch-num', nargs='+', type=int, default=None)

    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = parser_loader()
    parse_args = vars(parser.parse_args())
    print(parse_args)

    acc = np.zeros(10)
    times = np.zeros(10)
    for seed in range(10):
        setup_seed(seed)
        acc[seed], times[seed] = run(utils.ppi_loader, parse_args, seed)

    print('')
    print(np.mean(acc), np.mean(times))
    print(np.std(acc), np.std(times))

