import torch.nn as nn
import torch.nn.functional as F

from model.BNN import BNNConv, BNNLinear


class BNNCifarBase(nn.Module):
    def __init__(self):
        super(BNNCifarBase, self).__init__()
        self.conv1 = BNNConv(3, 16, kernel=(5, 5), stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = BNNConv(16, 32, kernel=(5, 5), stride=1, padding=0)
        self.fc1 = BNNLinear(32 * 5 * 5, 120)
        self.fc2 = BNNLinear(120, 84)
        self.d_feature = 84
    
    def forward(self, x, sample_W):
        x = self.pool(F.relu(self.conv1(x, sample_W)))
        x = self.pool(F.relu(self.conv2(x, sample_W)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x, sample_W))
        x = F.relu(self.fc2(x, sample_W))
        return x

    def get_kl_divergence(self, server_model):

        server_conv1_q_weight, server_conv1_q_bias = server_model.conv1.get_posterior_dist()
        kl_loss_conv1 = self.conv1.get_kl_divergence(server_conv1_q_weight, server_conv1_q_bias)

        server_conv2_q_weight, server_conv2_q_bias = server_model.conv2.get_posterior_dist()
        kl_loss_conv2 = self.conv2.get_kl_divergence(server_conv2_q_weight, server_conv2_q_bias)

        server_fc1_q_weight, server_fc1_q_bias = server_model.fc1.get_posterior_dist()
        kl_loss_fc1 = self.fc1.get_kl_divergence(server_fc1_q_weight, server_fc1_q_bias)

        server_fc2_q_weight, server_fc2_q_bias = server_model.fc2.get_posterior_dist()
        kl_loss_fc2 = self.fc2.get_kl_divergence(server_fc2_q_weight, server_fc2_q_bias)

        return kl_loss_conv1 + kl_loss_conv2 + kl_loss_fc1 + kl_loss_fc2

    def get_reg_loss(self):
        reg_loss_conv1_mu, reg_loss_conv1_sig = self.conv1.get_reg_loss()
        reg_loss_conv2_mu, reg_loss_conv2_sig = self.conv2.get_reg_loss()
        reg_loss_fc1_mu, reg_loss_fc1_sig = self.fc1.get_reg_loss()
        reg_loss_fc2_mu, reg_loss_fc2_sig = self.fc2.get_reg_loss()

        reg_loss_mu = reg_loss_conv1_mu + reg_loss_conv2_mu + reg_loss_fc1_mu + reg_loss_fc2_mu
        reg_loss_sig = reg_loss_conv1_sig + reg_loss_conv2_sig + reg_loss_fc1_sig + reg_loss_fc2_sig

        return reg_loss_mu, reg_loss_sig
