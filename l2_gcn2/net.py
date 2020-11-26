import torch
import torch.nn as nn
import scipy.sparse as sps


class net_train(nn.Module):

    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()

        self.weight = nn.Parameter(
                torch.zeros((in_channel, hidden_channel), dtype=torch.float))
        nn.init.xavier_uniform_(self.weight)

        self.classifier = nn.Parameter(
                torch.zeros((hidden_channel, out_channel), dtype=torch.float))
        nn.init.xavier_uniform_(self.classifier)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = self.relu(x)
        x = torch.mm(x, self.classifier)
        return x

    def get_w(self):
        return self.weight

    def get_c(self):
        return self.classifier


class net_test(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, Adj, weight_list, classifier):

        for w in weight_list:
            x = Adj.dot(x.numpy())
            x = torch.FloatTensor(x)
            x = torch.mm(x, w)
            x = self.relu(x)

        x = torch.mm(x, classifier)

        return x


class controller_l2o(nn.Module):

    def __init__(self, layer_num, max_epoch):
        super().__init__()

        self.lstm = torch.nn.LSTMCell(8, 8)
        self.encoder = nn.Embedding(2 * max_epoch, 7 - layer_num)
        self.decoder = nn.ModuleList([nn.Linear(8, 2) for _ in range(max_epoch)])

        self.stop_prob = 0
        self.selected_log_probs = []

    def forward(self, x, action, hx, cx, epochs):

        x = torch.cat((x, self.encoder.weight[epochs * 2 + action, :].view((1, -1))), dim=1)
        x = x.view(1, -1)
        if epochs == 0:
            hx, cx = self.lstm(x)
        else:
            hx, cx = self.lstm(x, (hx, cx))

        logit = self.decoder[epochs](hx)
        # logit[0, 1] = logit[0, 1] * 0.01

        prob = torch.nn.functional.softmax(logit, dim=-1)
        self.stop_prob = prob[0, 1]
        action = prob.multinomial(1)
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)

        self.selected_log_probs.append(log_prob.gather(1, action.data))

        return action, hx, cx

    def get_selected_log_probs(self):
        selected_log_probs = self.selected_log_probs
        self.selected_log_probs = []
        return selected_log_probs

    def get_stop_prob(self):
        return self.stop_prob

