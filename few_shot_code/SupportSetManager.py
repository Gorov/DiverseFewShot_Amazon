import random
import torch
import numpy as np
from torch.autograd import Variable

class SupportSetManager(object):
    FIXED_FIRST = 0
    RANDOM = 1
    def __init__(self, datasets, config, sample_per_class):
        self.config = config
        (TEXT, LABEL, train, dev, test) = datasets[0]
        self.TEXT = TEXT
        self.sample_per_class = sample_per_class

        print 'Picking up prototypes'
        self.prototype_text_list = []

        for taskid, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
            prototype_text = []
            #print taskid, LABEL.vocab
            if not hasattr(LABEL, 'vocab'):
                self.prototype_text_list.append(prototype_text)
                continue
            for lab_id in range(len(LABEL.vocab.itos)):
                prototype_text.append([])
            for example in train.examples:
                lab_id = LABEL.vocab.stoi[example.label]
                if prototype_text[lab_id] is not None:
                    prototype_text[lab_id].append(example.text)
                else:
                    prototype_text[lab_id] = [example.text]

            for lab_id in range(len(LABEL.vocab.itos)):
                if len(prototype_text[lab_id]) == 0:
                    prototype_text[lab_id].append(['<pad>'])

                if self.sample_per_class >= 1 and self.sample_per_class < len(prototype_text[lab_id]):
                    prototype_text[lab_id] = prototype_text[lab_id][:self.sample_per_class]

            print 'Task %d: picked up %d prototypes'%(taskid, self.sample_per_class)
            self.prototype_text_list.append(prototype_text)

    def select_support_set(self, taskid, policy):
        if policy == self.FIXED_FIRST:
            supp_set = self.select_support_set_first(taskid)
        elif policy == self.RANDOM:
            supp_set = self.select_support_set_random(taskid)
        return supp_set

    def select_support_set_first(self, taskid):
        prototype_text = self.prototype_text_list[taskid]

        examples_text = []
        for lab_id in range(len(prototype_text)):
            examples_text.append(prototype_text[lab_id][0])

        prototype_matrix = self.TEXT.numericalize(
            self.TEXT.pad(x for x in examples_text),
            device=self.config.gpu, train=True)
        #if taskid == 0: #TODO test the consistency of the first example
        #    print examples_text
        #    print prototype_matrix

        return prototype_matrix

    def select_support_set_random(self, taskid):
        prototype_text = self.prototype_text_list[taskid]

        examples_text = []
        for lab_id in range(len(prototype_text)):
            rand_idx = random.randint(0, len(prototype_text[lab_id]) - 1)
            examples_text.append(prototype_text[lab_id][rand_idx])

        prototype_matrix = self.TEXT.numericalize(
            self.TEXT.pad(x for x in examples_text),
            device=self.config.gpu, train=True)
        #if taskid == 0: #TODO test the consistency of the first example
        #    print examples_text
        #    print prototype_matrix

        return prototype_matrix

    def get_average_as_support(self, taskid, mnet_model):
        prototype_text = self.prototype_text_list[taskid]

        prototype_emb_list = []
        for lab_id in range(len(prototype_text)):
            prototype_sent = self.TEXT.numericalize(
                self.TEXT.pad(x for x in prototype_text[lab_id]),
                device=self.config.gpu, train=True)

            prototype_matrix = mnet_model.get_hidden(prototype_sent)
            prototype_emb_list.append(torch.mean(prototype_matrix, dim=0))
        #print prototype_emb_list
        #return torch.cat(prototype_emb_list, dim=0) #works for the new pytorch version
        return torch.cat(prototype_emb_list, 0)

    def get_average_and_std_as_support(self, taskid, mnet_model):
        prototype_text = self.prototype_text_list[taskid]

        prototype_emb_list = []
        prototype_std_list = []
        for lab_id in range(len(prototype_text)):
            N = len(prototype_text[lab_id])
            prototype_sent = self.TEXT.numericalize(
                self.TEXT.pad(x for x in prototype_text[lab_id]),
                device=self.config.gpu, train=True)

            prototype_matrix = mnet_model.get_hidden(prototype_sent)
            mean_vec = torch.mean(prototype_matrix, dim=0)
            if N > 1:
                #std_val = torch.sqrt((torch.pow(prototype_matrix, 2).sum() - N * torch.pow(mean_vec, 2).sum()) / (N - 1))
                std_val = (torch.pow(prototype_matrix, 2).sum() - N * torch.pow(mean_vec, 2).sum()) / (N - 1)
                std_val = Variable(std_val.data)
            else:
                std_val = Variable(torch.from_numpy(np.array([1.0]).astype(np.float32))).cuda()
            prototype_emb_list.append(mean_vec)
            prototype_std_list.append(std_val)
        #print prototype_std_list
        return torch.cat(prototype_emb_list, 0), torch.cat(prototype_std_list, 0)

    def get_average_as_support_sample(self, taskid, mnet_model, sample_per_class):
        prototype_text = self.prototype_text_list[taskid]

        prototype_emb_list = []
        for lab_id in range(len(prototype_text)):
            if sample_per_class > len(prototype_text[lab_id]):
                prototype_sent = self.TEXT.numericalize(
                    self.TEXT.pad(x for x in prototype_text[lab_id]),
                    device=self.config.gpu, train=True)
            else:
                top_ind = range(len(prototype_text[lab_id]))
                random.shuffle(top_ind)
                top_ind = top_ind[:sample_per_class]
                prototype_text_sample = [prototype_text[lab_id][i] for i in top_ind]
                prototype_sent = self.TEXT.numericalize(
                    self.TEXT.pad(x for x in prototype_text_sample),
                    device=self.config.gpu, train=True)

            prototype_matrix = mnet_model.get_hidden(prototype_sent)
            prototype_emb_list.append(torch.mean(prototype_matrix, dim=0))
        return torch.cat(prototype_emb_list, 0)

    def get_average_as_support_large(self, taskid, mnet_model, batchsize):
        prototype_text = self.prototype_text_list[taskid]

        prototype_emb_list = []
        for lab_id in range(len(prototype_text)):
            num_batch = len(prototype_text[lab_id]) / batchsize
            if len(prototype_text[lab_id]) % batchsize != 0 and num_batch == 0:
                num_batch += 1
            lab_emb_sum = []
            for i in range(num_batch):
                #print i
                #print len(prototype_text[lab_id]), i*batchsize, (i+1) * batchsize
                batch_text = prototype_text[lab_id][i * batchsize : min((i+1) * batchsize, len(prototype_text[lab_id]))]
                #print batch_text
                len_text = len(batch_text)
                #print len_text
                batch_prototype_sent = self.TEXT.numericalize(
                    self.TEXT.pad(x for x in batch_text),
                    device=self.config.gpu, train=True)
                #print batch_prototype_sent
                prototype_matrix = mnet_model.get_hidden(batch_prototype_sent)
                prototype_matrix = Variable(prototype_matrix.data)
                
                #prototype_emb_list.append(torch.mean(prototype_matrix, dim=0))
                #prototype_emb_list.append(torch.sum(prototype_matrix, dim=0) / len_text)
                #break
                #TODO: the following three lines not equivalent to the two lines below
            #    lab_emb_sum.append(torch.sum(prototype_matrix, dim=0))
            #lab_emb_sum = torch.sum( torch.cat(lab_emb_sum, 0), dim=0 )
            #lab_emb_sum /= len(prototype_text[lab_id])
                lab_emb_sum.append(torch.mean(prototype_matrix, dim=0))
            lab_emb_sum = torch.mean( torch.cat(lab_emb_sum, 0), dim=0 )
            prototype_emb_list.append(lab_emb_sum)
        return torch.cat(prototype_emb_list, 0)


