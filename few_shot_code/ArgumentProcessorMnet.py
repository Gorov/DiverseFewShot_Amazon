import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Matching Network')
    parser.add_argument('--workingdir', type=str, default='semeval',
                        help='working directory containing data')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='pretrained word embeddings')
    parser.add_argument('--emfilename', type=str, default='simple.token.vectors',
                        help='pretrained word embeddings file')
    parser.add_argument('--emfiledir', type=str, default='.',
                        help='directory of the pretrained word embeddings file')
    parser.add_argument('--cutlength', type=bool, default=False,
                        help='cut the sentences with mean + 2std')
    parser.add_argument('--task-split', type=int, default=0,
                        help='the task split used in experiment')
    parser.add_argument('--training-policy', type=str, default='first',
                        help='type of training sampling policies (first, random, hybrid)')
    parser.add_argument('--testing-policy', type=str, default='first',
                        help='type of training sampling policies (first, random, mean)')
    parser.add_argument('--sim-measure', type=str, default='inner',
                        help='type of similarity/distance measure (inner, cosine, L2)')
    parser.add_argument('--take-sqrt', type=bool, default=False,
                        help='whether to take square root of L2 distance')
    parser.add_argument('--sample-per-class', type=int, default=1000,
                        help='number of support samples per class during testing')
    parser.add_argument('--few-shot-num', type=int, default=5,
                        help='number of shots per class for testing tasks')
    parser.add_argument('--data', type=str, default='./data/penn',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='cnn',
                        help='type of encoder net (CNN, RNN_TANH, RNN_RELU, LSTM, GRU, BiLSTM, BiGRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--winsize', type=int, default=3,
                        help='size of conv windows')
    parser.add_argument('--nhid', type=int, default=200,
                        help='humber of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--lrDecay', type=float, default=0.8,
                        help='learning rate decay')
    parser.add_argument('--l2-weight', type=float, default=0.01,
                        help='weight for the L2-norm on gates')
    parser.add_argument('--l1-weight', type=float, default=0.01,
                    help='weight for the L1-norm on gates')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=20,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--gpu', type=int, default=0,
                        help='sequence length')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--fulllist', type=str, default='filelist.txt',
                        help='list of ids for multitask learning')
    parser.add_argument('--filelist', type=str, default='filelist.txt',
                        help='list of ids for multitask learning')
    parser.add_argument('--trainfilelist', type=str, default='trainfilelist.txt',
                        help='list of training tasks for few-shot learning')
    parser.add_argument('--trglist', type=str, default='filelist.txt',
                        help='list of target tasks for domain adaptation')
    parser.add_argument('--save-path', type=str, default='.',
                        help='model save path')
    parser.add_argument('--model-prefix', type=str, default='best_model',
                        help='list of ids for multitask learning')
    return parser
