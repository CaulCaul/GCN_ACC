import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dgl.data import CoraGraphDataset

import scipy.io
import scipy.sparse


def load_cora_data(g):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    return features, labels, train_mask, test_mask


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GraphConv(500, 16)
        self.layer2 = GraphConv(16, 3)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g):
    print("training")
    net = Net()

    features, labels, train_mask, test_mask = load_cora_data(g)
    # Add edges between each node and itself to preserve old node representations
    g.add_edges(g.nodes(), g.nodes())
    optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
    dur = []

    for epoch in range(50):
        if epoch >= 3:
            t0 = time.time()

        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))

    paras = list(net.parameters())
    for p in paras:
        print(type(p), p.size())

    W0 = paras[0].detach().numpy()
    W1 = paras[2].detach().numpy()

    return W0, W1

