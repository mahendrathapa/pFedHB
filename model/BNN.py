import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal


class BNNLinear(nn.Module):
    def __init__(self, in_features, out_features, init_type='uniform'):
        super(BNNLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.q_weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.q_weight_sig = nn.Parameter(torch.empty(out_features, in_features))
        self.q_bias_mu = nn.Parameter(torch.empty(out_features))
        self.q_bias_sig = nn.Parameter(torch.empty(out_features))

        self.eps = 1e-6
        self.initialize_params(init_type)

    def initialize_params(self, init_type):

        if init_type == 'uniform':
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.q_weight_mu, -bound, bound)
            nn.init.constant_(self.q_bias_mu, 0.0)
        elif init_type == 'normal':
            nn.init.normal_(self.q_weight_mu, mean=0.0, std=1.0)
            nn.init.normal_(self.q_bias_mu, mean=0.0, std=1.0)

        nn.init.constant_(self.q_weight_sig, -4.0)
        nn.init.constant_(self.q_bias_sig, -4.0)

    def get_posterior_dist(self):
        q_weight = Normal(self.q_weight_mu, F.softplus(self.q_weight_sig))
        q_bias = Normal(self.q_bias_mu, F.softplus(self.q_bias_sig))
        return q_weight, q_bias

    def get_kl_divergence(self, p_weight, p_bias):
        q_weight, q_bias = self.get_posterior_dist()

        kl_weight = kl_divergence(q_weight, p_weight)
        kl_bias = kl_divergence(q_bias, p_bias)

        return kl_weight.sum() + kl_bias.sum()

    def get_reg_loss(self):
        mu_loss = self.q_weight_mu.square().sum() + self.q_bias_mu.square().sum()
        sig_loss = F.softplus(self.q_weight_sig).square().sum() + F.softplus(self.q_bias_sig).square().sum()

        return mu_loss, sig_loss

    def forward(self, x, sample_W):
        q_weight, q_bias = self.get_posterior_dist()

        if sample_W:
            weight, bias = q_weight.rsample(), q_bias.rsample()
        else:
            weight, bias = q_weight.mean, q_bias.mean

        return F.linear(x, weight, bias)


class BNNConv(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride, padding):
        super(BNNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        self.stride = stride 
        self.padding = padding

        self.q_weight_mu = nn.Parameter(torch.empty(out_features, in_features, *kernel))
        self.q_weight_sig = nn.Parameter(torch.empty(out_features, in_features, *kernel))
        self.q_bias_mu = nn.Parameter(torch.empty(out_features))
        self.q_bias_sig = nn.Parameter(torch.empty(out_features))

        self.eps = 1e-6
        self.initialize_params()

    def initialize_params(self):

        fan_out = self.out_features * self.kernel[0] * self.kernel[1] / (self.stride * self.stride)
        bound = math.sqrt(6 / fan_out)

        nn.init.uniform_(self.q_weight_mu, -bound, bound)
        nn.init.constant_(self.q_weight_sig, -4.0)
        nn.init.constant_(self.q_bias_mu, 0.0)
        nn.init.constant_(self.q_bias_sig, -4.0)

    def get_posterior_dist(self):
        q_weight = Normal(self.q_weight_mu, F.softplus(self.q_weight_sig))
        q_bias = Normal(self.q_bias_mu, F.softplus(self.q_bias_sig))
        return q_weight, q_bias

    def get_kl_divergence(self, p_weight, p_bias):
        q_weight, q_bias = self.get_posterior_dist()

        kl_weight = kl_divergence(q_weight, p_weight)
        kl_bias = kl_divergence(q_bias, p_bias)

        return kl_weight.sum() + kl_bias.sum()

    def get_reg_loss(self):
        mu_loss = self.q_weight_mu.square().sum() + self.q_bias_mu.square().sum()
        sig_loss = F.softplus(self.q_weight_sig).square().sum() + F.softplus(self.q_bias_sig).square().sum()

        return mu_loss, sig_loss

    def forward(self, x, sample_W):
        q_weight, q_bias = self.get_posterior_dist()

        if sample_W:
            weight, bias = q_weight.rsample(), q_bias.rsample()
        else:
            weight, bias = q_weight.mean, q_bias.mean

        return F.conv2d(x, weight, bias, padding=self.padding, stride=self.stride)
