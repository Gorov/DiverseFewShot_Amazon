import torch
import torch.nn as nn

# import data
from MatchingCnn import MatchingCnn, MatchPair
from torch.autograd import Variable

class MatchPairStd(MatchPair):
    def __init__(self, x, y):
        #super(MatchPairStd, self).__init__(x, y)
        MatchPair.__init__(self, x, y)
        self.std = None

    def set_std(self, std):
        self.std = std

class MatchingLayerL2(nn.Module):
    def __init__(self, nonlinear='softmax', take_sqrt=True):
        super(MatchingLayerL2, self).__init__()
        self.nonlinear = nonlinear
        self.take_sqrt = take_sqrt
        if nonlinear == 'softmax':
            self.activation = nn.LogSoftmax()
        elif nonlinear == 'softmax_exp':
            self.activation = nn.Softmax()
        else:
            self.activation = None

    def forward(self, input):
        X = torch.mm(input.x, input.y.t())
        a = torch.pow(input.x, 2).sum(dim=1)
        b = torch.pow(input.y, 2).sum(dim=1)
        dist = a.expand(X.size(0), b.size(0)) + b.t().expand(a.size(0), X.size(1)) - 2.0 * X
        #dist = -2.0 * torch.mm(input.x, input.y.t()) + torch.mm(input.x, input.x.t()) + torch.mm(input.y, input.y.t())
        if self.take_sqrt:
            dist = torch.sqrt(dist)
        sim = 0.0 - dist
        if input.std is not None:
            #print input.std
            sim = sim / input.std.unsqueeze(0).expand_as(sim)
        if self.activation is not None:
            output = self.activation(sim)
        else:
            output = sim
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.nonlinear) + ')'


class MatchingCnnWithSuppPolicy(MatchingCnn):
    def __init__(self, config, w_hid_size, h_hid_size, num_tasks=4, pre_trained_emb=None, debug_mode=False,
                 additional_proj=False, normal_init=False):
        super(MatchingCnnWithSuppPolicy, self).__init__(config, w_hid_size, h_hid_size, num_tasks,
                                                        pre_trained_emb, debug_mode,
                                                        additional_proj, normal_init
                                                        )
        if config.sim_measure == 'L2':
            self.match_classifier = MatchingLayerL2(nonlinear='softmax', take_sqrt=config.take_sqrt)

    def forward(self, input, x_mode='words', y_mode='words', std=None):
        if x_mode == 'words':
            emb = self.column_embed(input.x)
            # emb = Variable(emb.data)
            hidden = self.column_encoder(emb)
        else:
            hidden = input.x

        if y_mode == 'words':
            y_emb = self.column_embed(input.y)
            # y_emb = Variable(y_emb.data)
            y_hidden = self.column_encoder(y_emb)
        else:
            y_hidden = input.y

        match_pair = MatchPairStd(hidden, y_hidden)
        if std is not None:
            match_pair.std = std
        output = self.match_classifier(match_pair)

        return output
