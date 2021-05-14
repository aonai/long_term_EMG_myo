import re
import math
import numpy as np
import collections

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.loss import _WeightedLoss
import torch.distributions.normal as NormalDistribution


class ReversalGradientLayerF(Function):
    @staticmethod
    def forward(ctx, input, lambda_hyper_parameter):
        ctx.lambda_hyper_parameter = lambda_hyper_parameter
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_hyper_parameter
        return output, None

    @staticmethod
    def grad_reverse(x, constant):
        return ReversalGradientLayerF.apply(x, constant)
