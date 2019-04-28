import torch
import torch.nn as nn
import torch.legacy.nn as luann
# import torch.nn.functional as F
import sys

from UtilLayer import Transpose, MaxPool, View

class CnnEncoder(nn.Module):
    def __init__(self, config, d_in, d_out, winsize = None, short_cut=False, padding=1, nonlinear='relu', normal_init=False):
        super(CnnEncoder, self).__init__()
        self.config = config
        self.d_in = d_in
        self.d_out = d_out
        self.short_cut = short_cut

        self.padding = padding

        if not winsize:
            self.winsize = config.winsize
        else:
            self.winsize = winsize
        self.normal_init = normal_init

        # self.model.add_module('transpose', Transpose())
        # self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        # self.model.add_module('emb', self.embed)
        self.model = nn.Sequential()

        # self.model.add_module('trans2', Transpose(1, 2))

        conv_nn = nn.Conv1d(self.d_in, self.d_out, self.winsize, padding=self.padding)
        if self.normal_init:
            print 'random_init conv weights: normal'
            conv_nn.weight.data.normal_(mean=0, std=0.1)
            conv_nn.bias.data.normal_(mean=0, std=0.1)
        self.model.add_module('conv', conv_nn)
        if nonlinear == 'relu':
            self.model.add_module('relu2', nn.ReLU())
        elif nonlinear == 'tanh':
            self.model.add_module('tanh2', nn.Tanh())
        if short_cut:
            assert self.d_in == self.d_out

    def forward(self, x):
        output = self.model.forward(x)
        if self.short_cut:
            output = output + x
        return output

class CNNModel(nn.Module):
    def __init__(self, vocab_size, num_labels, emb_size, w_hid_size, h_hid_size, win, batch_size, with_proj=False):
        super(CNNModel, self).__init__()

        self.model = nn.Sequential()
        self.model.add_module('transpose', Transpose())
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.model.add_module('emb', self.embed)
        if with_proj:
            self.model.add_module('view1', View(-1, emb_size))
            self.model.add_module('linear1', nn.Linear(emb_size, w_hid_size))
            # self.model.add_module('view2', View(batch_size, w_hid_size, -1))
            self.model.add_module('relu1', nn.ReLU())
        else:
            w_hid_size = emb_size
            # self.model.add_module('view2', View(batch_size, w_hid_size, -1))

        self.model.add_module('trans2', Transpose(1, 2))

        conv_nn = nn.Conv1d(w_hid_size, h_hid_size, win, padding=1)
        self.model.add_module('conv', conv_nn)
        self.model.add_module('relu2', nn.ReLU())

        # # self.model.add_module('view3', View(batch_size, -1, h_hid_size))
        # self.model.add_module('trans3', Transpose(1, 2))
        # self.model.add_module('max', MaxPool(1))

        # new implementation
        self.model.add_module('max', MaxPool(2))

        # old implementation
        # self.model.add_module('transpose2', Transpose(1, 2))
        # # self.model.add_module('view3', View(batch_size, -1, h_hid_size))
        # self.model.add_module('max', MaxPool(1))

        self.model.add_module('view4', View(batch_size, h_hid_size))
        self.model.add_module('linear2', nn.Linear(h_hid_size, num_labels))
        # m = nn.LogSoftmax()
        self.model.add_module('softmax', nn.LogSoftmax())
        # model:add(nn.Max(2))

        # model:add(nn.Linear(opt.numFilters, opt.hiddenDim))
        # model:add(nn.ReLU())
        # if opt.dropout > 0:
        #     model:add(nn.Dropout(opt.dropout))

        # self.model2 = nn.Sequential()
        # self.model2.add_module('linear2', nn.Linear(h_hid_size, num_labels))
        # self.model2.add_module('softmax', nn.LogSoftMax())
        # # Criterion
        # self.criterion = nn.ClassNLLCriterion()

    def forward(self, x):

        # output = self.lookupTable.forward(x)
        # output = x
        # for i in range(9):
        #     output = self.model[i].forward(output)
        #     print output.size()
        # sys.stdin.readline()
        # output = self.model[1].forward(output)
        # print output.size()
        # output = self.model[2].forward(output)
        # print output.size()
        # output = self.model[3].forward(output)
        # print output.size()
        # output = self.model[4].forward(output)
        # print output.size()
        # output = self.model[5].forward(output)
        # print output.size()

        output = self.model.forward(x)
        # output = torch.max(output, 1)[0]
        # output = self.model2.forward(output)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
