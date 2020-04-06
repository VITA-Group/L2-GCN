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


def run_l2o(dataset_load_func, parse_args, seed):

    #####################################
    # learn to optimize, config setting #
    #####################################

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

    ##############################################################
    ##############################################################

    ###############################
    # training initialization     #
    ###############################

    # train, val and test set index
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
    feat_test = torch.FloatTensor(feat_data)

    # Adj matrix generate
    Adj_eye = sps.eye(num_nodes, dtype=np.float32).tocsr()
    Adj_train = Adj[train, :][:, train]
    D_train = Adj_train.sum(axis=0)
    Adj_train = Adj_train.multiply(1/D_train.transpose())
    Adj_train = Adj_train + Adj_eye[train, :][:, train]
    Adj_test = Adj
    D_test = Adj_test.sum(axis=0)
    Adj_test = Adj_test.multiply(1/D_test.transpose())
    Adj_test = Adj_test + Adj_eye

    ##############################################################
    ##############################################################

    ###############################
    # l2o controller training     #
    ###############################

    # l2o controller training
    loss_func = nn.CrossEntropyLoss()

    controller_l2o = net.controller_l2o(args['layer_num'], args['controller_len']).cuda()
    optimizer_l2o = torch.optim.Adam(controller_l2o.parameters(), lr=args['l2o_learning_rate'])
    baseline = args['baseline_reward']

    # sample subgraphs
    feeder_train_sample = utils.feeder_sample(feat_data, labels, train, args['total_round'], args['sample_node_num'])
    dataset_train_sample = torch.utils.data.DataLoader(dataset=feeder_train_sample, batch_size=1)

    # warm up with predefine action
    predefine_action = np.zeros((5 * args['init_round'], args['controller_len']), dtype=np.int32)
    predefine_action[:, args['controller_len']-1] = 1
    for ir in range(args['init_round']):
        for ln in range(args['layer_num']-1):
            predefine_action[ir*5+4, int(args['controller_len']/args['layer_num']*(ln+1))] = 1
        for n in range(4):
            predefine_action[ir*5+n, random.sample(range(args['controller_len']-1), args['layer_num']-1)] = 1

    # start training
    for feat_train, label_train, train_sample, iround in dataset_train_sample:

        net_lwgcn = net.net_lwgcn(in_channel=args['feat_dim'], hidden_channel=args['hidden_dim'], out_channel=args['class_num'], layer_num=args['layer_num'])
        optimizer = torch.optim.Adam(net_lwgcn.parameters(), lr=args['learning_rate'])

        feat_train = feat_train.view(args['sample_node_num'], args['feat_dim'])
        label_train = label_train.view(-1)

        Adj_train = Adj[train_sample, :][:, train_sample]
        D = Adj_train.sum(axis=0)
        Adj_train = Adj_train.multiply(1/D.transpose())
        Adj_train = Adj_train + Adj_eye[train_sample, :][:, train_sample]

        epochs = 0

        print('')
        print('New round')
        print('')

        for l in range(args['layer_num']):

            print('layer ' + str(l+1) + ' training:')

            feat_train = feat_train.numpy()
            feat_train = Adj_train.dot(feat_train)
            feat_train = torch.FloatTensor(feat_train)

            x = feat_train.cuda()
            x_label = label_train.cuda()

            net_lwgcn = net_lwgcn.cuda()
            batch = 0

            while True:

                optimizer.zero_grad()
                output = net_lwgcn(x, layer_index=l)
                loss = loss_func(output, x_label)
                loss.backward()
                optimizer.step()

                if epochs % args['decision_step'] != 0:
                    epochs = epochs + 1
                    continue

                if epochs == 0:
                    loss_base = loss.detach()

                batch = batch + 1
                print('batch', batch, 'loss:', loss.data)

                input_l2o = torch.zeros((1, args['layer_num']+1)).cuda()
                input_l2o[0] = loss.detach() - loss_base + 1 ###
                input_l2o[0, l+1] = 1
                input_l2o = input_l2o * 0.1

                if epochs == 0:
                    action, hx, cx = controller_l2o(input_l2o, 0, 0, 0, 0)
                else:
                    action, hx, cx = controller_l2o(input_l2o, action, hx, cx, int(epochs/args['decision_step']))

                # predefine action
                if iround < args['init_round'] * 5:
                    action = predefine_action[iround, int(epochs/args['decision_step'])]
                epochs = epochs + 1

                # stop or not
                if action == 1:

                    if l != args['layer_num'] - 1:
                        net_lwgcn = net_lwgcn.cpu()
                        feat_train = net_lwgcn(feat_train, layer_index=l, with_classifier=False).detach()

                    break

        neg_rewards = loss.detach().cuda() + epochs * args['time_ratio']
        print('loss: ', neg_rewards)
        baseline = args['baseline_ratio'] * baseline + (1 - args['baseline_ratio']) * neg_rewards
        neg_rewards = neg_rewards - baseline
        neg_rewards = sum( controller_l2o.get_selected_log_probs() ) * neg_rewards

        optimizer_l2o.zero_grad()
        neg_rewards.backward()
        optimizer_l2o.step()

    ##############################################################
    ##############################################################

    ###############################
    # l2o-lwgcn  training         #
    ###############################

    print(args)

    result_file = './result/' + config_file[:-5] + '_l2o_' + str(args['layer_num']) + '_layer_' + str(seed) + '.npy'
    result_loss_data = []
    batch_each_layer = []

    # feature and label generate
    num_nodes = args['node_num']
    feat_train = torch.FloatTensor(feat_data)[train, :]
    label_train = labels[train]
    feat_test = torch.FloatTensor(feat_data)

    # Adj matrix generate
    Adj_eye = sps.eye(num_nodes, dtype=np.float32).tocsr()
    Adj_train = Adj[train, :][:, train]
    D_train = Adj_train.sum(axis=0)
    Adj_train = Adj_train.multiply(1/D_train.transpose())
    Adj_train = Adj_train + Adj_eye[train, :][:, train]
    Adj_test = Adj
    D_test = Adj_test.sum(axis=0)
    Adj_test = Adj_test.multiply(1/D_test.transpose())
    Adj_test = Adj_test + Adj_eye

    # l2o-layerwise training
    times = 0
    epochs = 0

    net_lwgcn = net.net_lwgcn(in_channel=args['feat_dim'], hidden_channel=args['hidden_dim'], out_channel=args['class_num'], layer_num=args['layer_num'])
    optimizer = torch.optim.Adam(net_lwgcn.parameters(), lr=args['learning_rate'])
    loss_func = nn.CrossEntropyLoss()

    for l in range(args['layer_num']):

        print('layer ' + str(l+1) + ' training:')

        feat_train = feat_train.numpy()
        start_time = time.time()
        feat_train = Adj_train.dot(feat_train)
        end_time = time.time()
        times = times + ( end_time - start_time )
        feat_train = torch.FloatTensor(feat_train)

        # create data loader
        feeder_train = utils.feeder(feat_train, label_train)
        dataset_train = torch.utils.data.DataLoader(dataset=feeder_train, batch_size=args['batch_size'], shuffle=True)
        net_lwgcn = net_lwgcn.cuda()

        batch = 0
        while True:
            for x, x_label in dataset_train:

                x = x.cuda()
                x_label = x_label.cuda()

                start_time = time.time()
                optimizer.zero_grad()
                output = net_lwgcn(x, layer_index=l)
                loss = loss_func(output, x_label)
                loss.backward()
                optimizer.step()
                end_time = time.time()
                times = times + ( end_time - start_time )

            result_loss_data.append(loss.data.cpu().numpy())
            batch = batch + 1
            print('batch', batch, 'loss:', loss.data)

            if epochs % args['decision_step'] != 0:
                epochs = epochs + 1
                continue

            if epochs == 0:
                loss_base = loss.detach()

            input_l2o = torch.zeros((1, args['layer_num']+1)).cuda()
            input_l2o[0] = loss.detach() - loss_base + 1
            input_l2o[0, l+1] = 1
            input_l2o = input_l2o * 0.1

            if epochs == 0:
                action, hx, cx = controller_l2o(input_l2o, 0, 0, 0, 0)
            else:
                action, hx, cx = controller_l2o(input_l2o, action, hx, cx, int(epochs/args['decision_step']))

            epochs = epochs + 1

            if (controller_l2o.get_stop_prob() >= args['stop_prob_threshold'] and batch > args['min_batch']) or batch > args['max_batch']:

                batch_each_layer.append(batch)
                net_lwgcn = net_lwgcn.cpu()
                if l != args['layer_num'] - 1:

                    start_time = time.time()
                    feat_train = net_lwgcn(feat_train, layer_index=l, with_classifier=False).detach()
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
        output = net_lwgcn.val_test(feat_test, Adj_test)
        output_val = output[val]
        output_test = output[test]

    print("accuracy in val:", f1_score(labels[val], output_val.data.numpy().argmax(axis=1), average="micro"))
    print("accuracy in test:", f1_score(labels[test], output_test.data.numpy().argmax(axis=1), average="micro"))
    print("average epoch time:", times / epochs)
    print("total time:", times)

    return f1_score(labels[test], output_test.data.numpy().argmax(axis=1), average="micro"), sum(batch_each_layer), times, np.array(batch_each_layer)

    ##############################################################
    ##############################################################


def parser_loader():

    parser = argparse.ArgumentParser(description='L2O_LWGCN')
    parser.add_argument('--dataset', type=str, default='reddit')
    parser.add_argument('--config-file', type=str, default='reddit.yaml')
    parser.add_argument('--layer-num', type=int, default=None)
    parser.add_argument('--epoch-num', nargs='+', type=int, default=None)
    parser.add_argument('--controller-len', type=int, default=None)

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

    if parse_args['dataset'] == 'cora':
        dataset_loader_func = utils.cora_loader
    elif parse_args['dataset'] == 'pubmed':
        dataset_loader_func = utils.pubmed_loader
    elif parse_args['dataset'] == 'reddit':
        dataset_loader_func = utils.reddit_loader
    elif parse_args['dataset'] == 'amazon_670k':
        dataset_loader_func = utils.amazon_670k_loader
    elif parse_args['dataset'] == 'amazon_3m':
        dataset_loader_func = utils.amazon_3m_loader

    acc = np.zeros(10)
    epoch_sum = np.zeros(10)
    times = np.zeros(10)
    epoch_array = np.zeros((parse_args['layer_num'], 10))
    for seed in range(10):
        setup_seed(seed)
        acc[seed], epoch_sum[seed], times[seed], epoch_array[:, seed] = run_l2o(dataset_loader_func, parse_args, seed)

    print('')
    print(np.mean(acc), np.mean(epoch_sum), np.mean(times), np.mean(epoch_array, axis=1))
    print(np.std(acc), np.std(epoch_sum), np.std(times), np.std(epoch_array, axis=1))

