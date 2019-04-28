import torch
import torch.nn as nn
import torch.functional as F


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    r"""Returns cosine similarity between x1 and x2, computed along dim.
    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero. Default: 1e-8
    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.
    # >>> input1 = autograd.Variable(torch.randn(100, 128))
    # >>> input2 = autograd.Variable(torch.randn(100, 128))
    # >>> output = F.cosine_similarity(input1, input2)
    # >>> print(output)
    """
    # w12 = torch.sum(x1 * x2, dim)
    w12 = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    # return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    return w12 / torch.mm(w1, w2.t()).clamp(min=eps)

class MaxPool(nn.Module):
    def __init__(self, dim=1):
        super(MaxPool, self).__init__()
        self.dim = dim

    def forward(self, input):
        return torch.max(input, self.dim)[0]

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'dim=' + str(self.dim) + ')'


class View(nn.Module):
    def __init__(self, *sizes):
        super(View, self).__init__()
        # self.
        self.sizes_list = sizes

    def forward(self, input):
        return input.view(*self.sizes_list)
        # return input.view(*self.sizes_list).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'sizes=' + str(self.sizes_list) + ')'


class Transpose(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        return input.transpose(self.dim1, self.dim2).contiguous()
        # return input.t()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'between=' + str(self.dim1) + ',' + str(self.dim2) + ')'


class LinearTransform(nn.Module):
    def __init__(self, n_in, n_out, nonlinear='softmax'):
        super(LinearTransform, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.classifier = nn.Sequential()
        self.classifier.add_module('linear', nn.Linear(n_in, n_out))
        if nonlinear == 'softmax':
            self.classifier.add_module('softmax', nn.LogSoftmax())
        elif nonlinear == 'relu':
            self.classifier.add_module('relu', nn.ReLU())
        elif nonlinear == 'softmax_exp':
            self.classifier.add_model('softmax_exp', nn.Softmax())

    def forward(self, input):
        output = self.classifier(input)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.n_in) + ',' + str(self.n_out) + ')'


class mul_scalar(torch.autograd.Function):
    """
    Customized autograd.Function of
    f(T,s) = s * T,
    where T is a fixed Tensor and s is a Variable
    """

    def forward(self, T, s_var):
        self.save_for_backward(T, s_var)
        return T.mul(s_var[0])

    def backward(self, grad_output):
        print 'here'
        T, s_var = self.saved_tensors
        return grad_output.mul(s_var[0]), grad_output.dot(T)


class MulScalar(nn.Module):
    def __init__(self, init_value=None):
        super(MulScalar, self).__init__()
        if init_value is None:
            self.weight = nn.Parameter(torch.randn(1))
        else:
            self.weight = nn.Parameter(torch.FloatTensor([init_value]))
        self.ms = mul_scalar()

    def forward(self, input):
        output = self.ms(input, self.weight)
        return output

    def __repr__(self):
        return self.__class__.__name__


class MatchingLayer(nn.Module):
    def __init__(self, nonlinear='softmax', use_cosine=False):
        super(MatchingLayer, self).__init__()
        self.nonlinear = nonlinear
        self.use_cosine = use_cosine
        if nonlinear == 'softmax':
            self.activation = nn.LogSoftmax()
        elif nonlinear == 'softmax_exp':
            self.activation = nn.Softmax()
        else:
            self.activation = None

    def forward(self, input):
        if self.use_cosine:
            sim = cosine_similarity(input.x, input.y, dim=1)
        else:
            sim = torch.mm(input.x, input.y.t())
        if self.activation is not None:
            output = self.activation(sim)
        else:
            output = sim
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.nonlinear) + ')'
