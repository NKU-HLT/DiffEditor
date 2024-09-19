import torch
from torch.nn import functional as F
from torch.nn import Dropout, Sequential, Linear, Softmax


class GradientReversalFunction(torch.autograd.Function):
    """Revert gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, l, c):
        # l=1
        # c=clip_boundary
        ctx.l = l
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)

        return ctx.l * grad_output.neg(), None, None
    
    # grad_output.neg()计算梯度的负值


class GradientClippingFunction(torch.autograd.Function):
    """Clip gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)
        return grad_output, None


class ReversalClassifier(torch.nn.Module):
    """Adversarial classifier (with two FC layers) with a gradient reversal layer.
    
    Arguments:
        input_dim -- size of the input layer (probably should match the output size of encoder)
        hidden_dim -- size of the hiden layer
        output_dim -- number of channels of the output (probably should match the number of speakers/languages)
        gradient_clipping_bound (float) -- maximal value of the gradient which flows from this module
    Keyword arguments:
        scale_factor (float, default: 1.0)-- scale multiplier of the reversed gradientts
    """

    def __init__(self, input_dim, hidden_dim, output_dim, gradient_clipping_bounds, scale_factor=1.0):
        super(ReversalClassifier, self).__init__()
        self._lambda = scale_factor
        self._clipping = gradient_clipping_bounds
        self._output_dim = output_dim
        self._classifier = Sequential(
            Linear(input_dim, hidden_dim),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, x):  
        x = GradientReversalFunction.apply(x, self._lambda, self._clipping)
        x = self._classifier(x)
        return x

    @staticmethod
    def loss(input_lengths, speakers, prediction, embeddings=None):
        # 取得一个batch里面的最长的长度ml
        # 将target设置为batchsize*ml的tensor，里面每一个值都是speaker_id
        # 对于target里面的对应于原始数据里没有信息的部分设置为ignore_index（也就是ml-true_length的部分）
        ignore_index = -100
        ml = torch.max(input_lengths)
        input_mask = torch.arange(ml, device=input_lengths.device)[None, :] < input_lengths[:, None]
        target = speakers.repeat(ml, 1).transpose(0,1)
        target[~input_mask] = ignore_index
        
        
        return F.cross_entropy(prediction.transpose(1,2), target, ignore_index=ignore_index)

