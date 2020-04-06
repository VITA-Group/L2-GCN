import torch
import torch.nn as nn
import scipy.sparse as sps


class net_lwgcn(nn.Module):

    def __init__(self, in_channel, hidden_channel, out_channel, layer_num):
        super().__init__()

        self.net_lw = nn.ModuleList()
        for ln in range(layer_num):
            if ln == 0:
                self.net_lw.append(nn.Linear(in_channel, hidden_channel, bias=False))
            else:
                self.net_lw.append(nn.Linear(hidden_channel, hidden_channel, bias=False))
        self.classifier = nn.Linear(hidden_channel, out_channel, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, layer_index, with_classifier=True):

        x = self.net_lw[layer_index](x)
        x = self.relu(x)
        if with_classifier:
            x = self.classifier(x)

        return x

    # remember to val or test in cpu
    def val_test(self, x, Adj):

        for net_lw in self.net_lw:
            x = Adj.dot(x.numpy())
            x = net_lw(torch.FloatTensor(x))
            x = self.relu(x)
        x = self.classifier(x)

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

        prob = torch.nn.functional.softmax(logit, dim=-1)
        self.stop_prob = prob[0, 1]
        print(prob.data)
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

