import torch
import torch.nn as nn
import torch.optim as OPT

# import data
from CNNModel import CnnEncoder
# from RnnModel import Encoder as RnnEncoder
from torch.autograd import Variable

from torchtext import data
import UtilFunctions
from UtilLayer import *
import sys
from copy import deepcopy


class MatchPair():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MatchingCnn(nn.Module):
    def __init__(self, config, w_hid_size, h_hid_size, num_tasks=4, pre_trained_emb=None, debug_mode=False, additional_proj=False, normal_init=False):
        super(MatchingCnn, self).__init__()
        self.config = config
        self.w_hid_size = w_hid_size
        self.h_hid_size = h_hid_size
        self.additional_proj = additional_proj
        self.normal_init = normal_init

        self.column_embed = nn.Sequential()
        self.column_embed.add_module('transpose1', Transpose())
        embed = nn.Embedding(num_embeddings=config.n_embed, embedding_dim=config.d_embed)
        #if normal_init:
        embed.weight.data.normal_(mean=0, std=0.1)
        if pre_trained_emb is not None:
            embed.weight.data = deepcopy(pre_trained_emb)
        self.column_embed.add_module('embed', embed)

        self.column_encoder = nn.Sequential()
        d_in = w_hid_size
        d_out = h_hid_size
        self.column_encoder.add_module('transpose2', Transpose(1, 2))
        if debug_mode:
            torch.manual_seed(12345678)
        self.column_encoder.add_module('cnn', CnnEncoder(config, d_in, d_out, normal_init=self.normal_init))
        if self.additional_proj:
            self.column_encoder.add_module('transform', CnnEncoder(config, d_in, h_hid_size, winsize=1, padding=0))
        self.column_encoder.add_module('max', MaxPool(2))
        self.column_encoder.add_module('view', View(-1, h_hid_size))

        self.match_classifier = MatchingLayer(nonlinear='softmax')

    def forward(self, input):
        emb = self.column_embed(input.x)
        #emb = Variable(emb.data)
        hidden = self.column_encoder(emb)

        y_emb = self.column_embed(input.y)
        #y_emb = Variable(y_emb.data)
        y_hidden = self.column_encoder(y_emb)

        output = self.match_classifier(MatchPair(hidden, y_hidden))
        return output
    
    def get_hidden(self, x):
        emb = self.column_embed(x)
        hidden = self.column_encoder(emb)
        return Variable(hidden.data)

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
