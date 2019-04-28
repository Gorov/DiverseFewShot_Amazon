import random
import torch
import numpy as np
from torch.autograd import Variable
from SupportSetManager import SupportSetManager

class SupportSetManagerLargeHybrid(SupportSetManager):
    FIXED_FIRST = 0
    RANDOM = 1
    RANDOM_SUB = 2
    def __init__(self, datasets, config, sample_per_class, sampling_proto=False):
        super(SupportSetManagerLargeHybrid, self).__init__(datasets, config, 0)
        self.sample_per_class = sample_per_class
        self.sampling_proto = sampling_proto

    def select_support_set(self, taskid, policy):
        if policy == self.FIXED_FIRST:
            supp_set = self.select_support_set_first(taskid)
        elif policy == self.RANDOM:
            supp_set = self.select_support_set_random(taskid)
        elif policy == self.RANDOM_SUB:
            supp_set = self.select_support_set_random_sub(taskid)
        return supp_set

    def select_support_set_random_sub(self, taskid):
        prototype_text = self.prototype_text_list[taskid]

        examples_text = []
        for lab_id in range(len(prototype_text)):
            if self.sample_per_class < len(prototype_text[lab_id]):
                rand_idx = random.randint(0, self.sample_per_class - 1)
            else:
                rand_idx = random.randint(0, len(prototype_text[lab_id]) - 1)
            examples_text.append(prototype_text[lab_id][rand_idx])

        prototype_matrix = self.TEXT.numericalize(
            self.TEXT.pad(x for x in examples_text),
            device=self.config.gpu, train=True)

        return prototype_matrix

    def get_average_as_support(self, taskid, mnet_model):
        prototype_text = self.prototype_text_list[taskid]

        prototype_emb_list = []
        for lab_id in range(len(prototype_text)):
            if self.sample_per_class >= 1 and self.sample_per_class < len(prototype_text[lab_id]):
                prototype_sent = self.TEXT.numericalize(
                    self.TEXT.pad(x for x in prototype_text[lab_id][:self.sample_per_class]),
                    device=self.config.gpu, train=True)
            else:
                prototype_sent = self.TEXT.numericalize(
                    self.TEXT.pad(x for x in prototype_text[lab_id]),
                    device=self.config.gpu, train=True)

            prototype_matrix = mnet_model.get_hidden(prototype_sent)
            prototype_emb_list.append(torch.mean(prototype_matrix, dim=0))
        #print prototype_emb_list
        #return torch.cat(prototype_emb_list, dim=0) #works for the new pytorch version
        return torch.cat(prototype_emb_list, 0)
