import sys
import torch
import MatchingCnn

def get_length_avg_std(dataset):
    len_vec = [len(x.text) for x in dataset.examples]
    return torch.std(torch.Tensor(len_vec)), torch.mean(torch.Tensor(len_vec))

def get_maxlen_dataset(dataset):
    maxlen = 0
    for example in dataset.examples:
        if len(example.text) > maxlen:
            maxlen = len(example.text)
    return maxlen

def padding_dataset(dataset, maxlen=None):
    if not maxlen:
        maxlen = get_maxlen_dataset(dataset)
    print 'maxlen:', maxlen
    for example in dataset.examples:
        if len(example.text) > maxlen:
            example.text = example.text[:maxlen]
        elif len(example.text) < maxlen:
            for i in range(maxlen - len(example.text)):
                example.text.append('<pad>')


def eval_matching_model(model, criterion, dev_iter, test_iter, prototype_list, batch_size, pre_compute=True):
    model.eval();
    for set_idx, set_iter in enumerate([dev_iter, test_iter]):
        if not pre_compute:
            set_iter.init_epoch()
        n_dev_correct, tmp_loss = 0, 0
        n_dev_total = 0
        for dev_batch_idx, dev_batch in enumerate(set_iter):
            # print dev_batch.text.size()
            if not pre_compute and dev_batch.text.size()[1] != batch_size:
                continue
            # elif pre_compute and dev_batch.text.size()[0] != batch_size:
            #     continue
            # answer = model(dev_batch.x)
            answer = model(MatchingCnn.MatchPair(dev_batch.x, prototype_list))
            n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.y.size()).data == dev_batch.y.data).sum()
            n_dev_total += dev_batch.batch_size
            tmp_loss += criterion(answer, dev_batch.y)

        if set_idx == 0:
            dev_acc = 100. * n_dev_correct / n_dev_total
            dev_loss = tmp_loss
        else:
            test_acc = 100. * n_dev_correct / n_dev_total
            test_loss = tmp_loss
    return dev_acc, test_acc, dev_loss, test_loss

def eval_classification_model(model, criterion, dev_iter, test_iter, batch_size, pre_compute=False, selected_batch=False):
    model.eval();
    for set_idx, set_iter in enumerate([dev_iter, test_iter]):
        if not pre_compute and not selected_batch:
            set_iter.init_epoch()
        n_dev_correct, tmp_loss = 0, 0
        n_dev_total = 0
        for dev_batch_idx, dev_batch in enumerate(set_iter):
            # print dev_batch.text.size()
            #if not pre_compute and dev_batch.text.size()[1] != batch_size:
            #    continue
            if not pre_compute:
                answer = model(dev_batch.text)
                n_dev_correct += (
                torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                n_dev_total += dev_batch.batch_size
                tmp_loss += criterion(answer, dev_batch.label)
            else:
                answer = model(dev_batch.x)
                n_dev_correct += (
                torch.max(answer, 1)[1].view(dev_batch.y.size()).data == dev_batch.y.data).sum()
                n_dev_total += dev_batch.batch_size
                tmp_loss += criterion(answer, dev_batch.y)

        if set_idx == 0:
            dev_acc = 100. * n_dev_correct / n_dev_total
            dev_loss = tmp_loss
        else:
            test_acc = 100. * n_dev_correct / n_dev_total
            test_loss = tmp_loss

    return dev_acc, test_acc, dev_loss, test_loss
