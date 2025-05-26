import torch.nn as nn
import torch.nn.functional as F

from model.BNN import BNNLinear


class BNNFMNISTBase(nn.Module):

    def __init__(self):
        super(BNNFMNISTBase, self).__init__()
        self.fc = BNNLinear(784, 100)
        self.d_feature = 100
    
    def forward(self, x, sample_W):
        x = x.reshape(-1, 784)
        x = F.relu(self.fc(x, sample_W))
        return x
    
    def get_kl_divergence(self, server_model):

        server_fc_q_weight, server_fc_q_bias = server_model.fc.get_posterior_dist()
        kl_loss_fc = self.fc.get_kl_divergence(server_fc_q_weight, server_fc_q_bias)

        return kl_loss_fc

    def get_reg_loss(self):
        reg_loss_fc_mu, reg_loss_fc_sig = self.fc.get_reg_loss()

        reg_loss_mu = reg_loss_fc_mu
        reg_loss_sig = reg_loss_fc_sig

        return reg_loss_mu, reg_loss_sig
