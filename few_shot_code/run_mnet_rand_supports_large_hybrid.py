import sys, os, glob, random
import time

sys.path.append('/dccstor/yum-dbqa/pyTorch/text/build/lib')
sys.path.append('/dccstor/yum-dbqa/pyTorch/cleaned_mnet')

import torch
import torch.nn as nn
# from AdaAdam import AdaAdam
import torch.optim as OPT

# import data
from SupportSetManagerLargeHybrid import SupportSetManagerLargeHybrid
from MatchingCnn import MatchPair
import ArgumentProcessorMnet as args_mnet
from MatchingNetWithSupportPolicy import MatchingCnnWithSuppPolicy
from DataProcessing.NlcDatasetSingleFile import NlcDatasetSingleFile

from torchtext import data
from DataProcessing.MTLField import MTLField

parser = args_mnet.get_parser()
args = parser.parse_args()

print args
batch_size = args.batch_size
#args.epochs = 10
args.seed = 12345678

torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def load_train_test_files(listfilename, test_suffix='.test'):
    filein = open(listfilename, 'r')
    file_tuples = []
    for line in filein:
        array = line.strip().split('\t')
        line = array[0]
        trainfile = line + '.train'
        devfile = line + '.dev'
        testfile = line + test_suffix
        file_tuples.append((trainfile, devfile, testfile))
    filein.close()
    return file_tuples

datasets = []
list_dataset = []
file_tuples = load_train_test_files(args.filelist)
print file_tuples
TEXT = MTLField(lower=True)
for (trainfile, devfile, testfile) in file_tuples:
    print trainfile, devfile, testfile
    LABEL1 = data.Field(sequential=False)
    train1, dev1, test1 = NlcDatasetSingleFile.splits(
        TEXT, LABEL1, path=args.workingdir, train=trainfile,
        validation=devfile, test=testfile)
    datasets.append((TEXT, LABEL1, train1, dev1, test1))
    list_dataset.append(train1)
    list_dataset.append(dev1)
    list_dataset.append(test1)

dataset_iters = []
for (TEXT, LABEL, train, dev, test) in datasets:
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=batch_size, device=args.gpu)
    train_iter.repeat = False
    dataset_iters.append((train_iter, dev_iter, test_iter))

# print information about the data
num_batch_total = 0
for i, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
    #print 'DATASET%d'%(i+1)
    #print('train.fields', train.fields)
    #print('len(train)', len(train))
    #print('len(dev)', len(dev))
    #print('len(test)', len(test))
    #print('vars(train[0])', vars(train[0]))
    num_batch_total += len(train) / batch_size

TEXT.build_vocab(list_dataset, wv_type=args.emfilename, wv_dim=args.emsize, wv_dir=args.emfiledir)
# TEXT.build_vocab(list_dataset)

# build the vocabulary
for taskid, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
    LABEL.build_vocab(train, dev, test)
    LABEL.vocab.itos = LABEL.vocab.itos[1:]
    for k, v in LABEL.vocab.stoi.items():
        LABEL.vocab.stoi[k] = v - 1

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    # print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
    print('len(LABEL.vocab)', len(LABEL.vocab)),
    #print LABEL.vocab.itos
    print len(LABEL.vocab.itos)
    #if taskid == 0:
    #    print LABEL.vocab.stoi
    #print len(LABEL.vocab.stoi)

config = args
config.n_embed = len(TEXT.vocab)
config.d_embed = args.emsize
config.d_proj = 100
config.d_hidden = args.nhid
config.fixed_emb = True
config.projection = False

ss_manager = SupportSetManagerLargeHybrid(datasets, config, config.sample_per_class)

config.n_labels = []
for (TEXT, LABEL, train, dev, test) in datasets:
    config.n_labels.append(len(LABEL.vocab))
print config.n_labels

config.n_cells = 1
config.maxpool = True

config.num_tasks = len(config.n_labels)
print 'num_tasks',
print config.num_tasks,
print len(dataset_iters)
#model = MatchingCnn(config, args.emsize, config.d_hidden, num_tasks=config.num_tasks, pre_trained_emb=TEXT.vocab.vectors, normal_init=True)
#model = MatchingCnnWithSuppPolicy(config, args.emsize, config.d_hidden, num_tasks=config.num_tasks, pre_trained_emb=TEXT.vocab.vectors, normal_init=True)
model = MatchingCnnWithSuppPolicy(config, args.emsize, config.d_hidden, num_tasks=config.num_tasks, normal_init=True)

if args.gpu != -1:
    model.cuda()

print model

# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
opt = OPT.Adam(model.parameters(), lr=args.lr)
#opt = AdaAdam(model.parameters(), lr=args.lr)

for param in model.parameters():
    print(type(param.data), param.size())
# sys.exit(0)

iterations = 0
start = time.time()
best_dev_acc = -1
best_dev_epoch = -1
best_test_acc = -1
best_test_epoch = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))

overall_log_template = 'Best dev acc: {:>8.6f}, epoch: {:>5.0f}; Best test acc: {:>8.6f}, epoch: {:>5.0f}'
# os.makedirs(args.save_path, exist_ok=True)
print(header)
iter_per_sample = 1
n_correct, n_total = 0, 0
for t in range(num_batch_total * args.epochs / iter_per_sample):
    taskid = random.randint(0, config.num_tasks - 1)
    (train_iter, dev_iter, test_iter) = dataset_iters[taskid]
    train_iter.init_epoch()
    model.train()

    for num_iter in range(iter_per_sample):
        batch = next(iter(train_iter))
        sys.stdout.write('%d\r'%t)
        sys.stdout.flush()
        opt.zero_grad()

        if args.training_policy == 'first':
            supp_text = ss_manager.select_support_set(taskid, ss_manager.FIXED_FIRST)
        elif args.training_policy == 'random':
            supp_text = ss_manager.select_support_set(taskid, ss_manager.RANDOM)
        elif args.training_policy == 'hybrid':
            if t % 2 == 0:
                supp_text = ss_manager.select_support_set(taskid, ss_manager.RANDOM_SUB)
            else:
                supp_text = ss_manager.select_support_set(taskid, ss_manager.RANDOM)
        #print batch.text
        #print supp_text

        answer = model(MatchPair(batch.text, supp_text))
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total
        loss = criterion(answer, batch.label)
        loss.backward()
        opt.step()

    #if (t + 1) % num_batch_total == 0:
    if (t + 1) % min(10000 / iter_per_sample, num_batch_total) == 0:
        #if config.lrDecay > 0:
        #    cur_lr = opt.getScale()
        #    cur_lr = cur_lr * 0.9
        #    opt.setScale(cur_lr * config.lr > 0.0001 and cur_lr or 0.0001 / config.lr)
        #    print 'lr: ', opt.getScale() * config.lr

        print(log_template.format(time.time() - start,
                                  (t + 1) / num_batch_total, (t + 1), (t + 1), num_batch_total * args.epochs,
                                  100., loss.data[0], ' ' * 8,
                                  n_correct / n_total * 100, ' ' * 12))
        #continue

        model.eval();
        avg_dev_acc = 0.0
        avg_test_acc = 0.0
        for taskid, (train_iter, dev_iter, test_iter) in enumerate(dataset_iters):
            #print 'task ', taskid,
            n_dev_correct, dev_loss = 0, 0
            n_dev_total = 0
            n_test_correct, test_loss = 0, 0
            n_test_total = 0
            if args.testing_policy == 'first':
                supp_text = ss_manager.select_support_set(taskid, ss_manager.FIXED_FIRST)
            elif args.testing_policy == 'mean':
                supp_emb = ss_manager.get_average_as_support(taskid, model)
            elif args.testing_policy == 'mean_std':
                supp_emb, supp_std = ss_manager.get_average_and_std_as_support(taskid, model)
            for set_idx, set_iter in enumerate([dev_iter, test_iter]):
                set_iter.init_epoch()
                for dev_batch_idx, dev_batch in enumerate(set_iter):
                    if args.testing_policy == 'mean':
                        answer = model(MatchPair(dev_batch.text, supp_emb), y_mode = 'emb')
                    elif args.testing_policy == 'mean_std':
                        answer = model(MatchPair(dev_batch.text, supp_emb), y_mode='emb', std=supp_std)
                    else:
                        answer = model(MatchPair(dev_batch.text, supp_text))
                    if set_idx == 0:
                        n_dev_correct += (
                        torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                        n_dev_total += dev_batch.batch_size
                        dev_loss = criterion(answer, dev_batch.label)
                    else:
                        n_test_correct += (
                            torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                        n_test_total += dev_batch.batch_size
                        test_loss = criterion(answer, dev_batch.label)

            dev_acc = 100. * n_dev_correct / n_dev_total
            test_acc = 100. * n_test_correct / n_test_total
            avg_dev_acc += dev_acc
            avg_test_acc += test_acc
            #print('Dev:  ' + dev_log_template.format(time.time() - start,
            #                              (t + 1) / num_batch_total, (t + 1), (t + 1), num_batch_total * args.epochs,
            #                              100., loss.data[0], dev_loss.data[0],
            #                              train_acc, dev_acc))
            #print('Test: ' + dev_log_template.format(time.time() - start,
            #                              (t + 1) / num_batch_total, (t + 1), (t + 1), num_batch_total * args.epochs,
            #                              100., loss.data[0], test_loss.data[0],
            #                              train_acc, test_acc))

        avg_dev_acc /= config.num_tasks
        avg_test_acc /= config.num_tasks
        print 'Iteration Results:\t{:>4.2f}\t{:>4.2f}'.format(avg_dev_acc, avg_test_acc)

        if avg_dev_acc > best_dev_acc:
            best_dev_acc = avg_dev_acc
            best_dev_epoch = (t + 1) / num_batch_total
            # snapshot_prefix = os.path.join(args.save_path, args.model_prefix)
            # snapshot_path = snapshot_prefix + '_devacc_{}_iter_{}_model.pt'.format(best_dev_acc, iterations)
            # torch.save(model, snapshot_path)
            # for f in glob.glob(snapshot_prefix + '*'):
            #     if f != snapshot_path:
            #         os.remove(f)

        #if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            best_test_epoch = (t + 1) / num_batch_total

        print(overall_log_template.format(best_dev_acc, best_dev_epoch, best_test_acc, best_test_epoch))

