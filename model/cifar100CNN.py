import torch.nn as nn
import torch.nn.functional as F

from model.BNN import BNNConv


class BNNCifar100Base(nn.Module):
    def __init__(self):
        super(BNNCifar100Base, self).__init__()
        self.conv1 = BNNConv(3, 32, kernel=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = BNNConv(32, 64, kernel=(3, 3), stride=1, padding=1)
        self.conv3 = BNNConv(64, 128, kernel=(3, 3), stride=1, padding=1)
        self.conv4 = BNNConv(128, 256, kernel=(3, 3), stride=1, padding=1)
        self.conv5 = BNNConv(256, 100, kernel=(3, 3), stride=1, padding=1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.d_feature = 100
    
    def forward(self, x, sample_W):
        x = self.pool(F.relu(self.conv1(x, sample_W)))
        x = self.pool(F.relu(self.conv2(x, sample_W)))
        x = self.pool(F.relu(self.conv3(x, sample_W)))
        x = self.pool(F.relu(self.conv4(x, sample_W)))
        x = self.adaptive_pool(F.relu(self.conv5(x, sample_W)))

        return x.squeeze()

    def get_kl_divergence(self, server_model):
        server_conv1_q_weight, server_conv1_q_bias = server_model.conv1.get_posterior_dist()
        kl_loss_conv1 = self.conv1.get_kl_divergence(server_conv1_q_weight, server_conv1_q_bias)

        server_conv2_q_weight, server_conv2_q_bias = server_model.conv2.get_posterior_dist()
        kl_loss_conv2 = self.conv2.get_kl_divergence(server_conv2_q_weight, server_conv2_q_bias)

        server_conv3_q_weight, server_conv3_q_bias = server_model.conv3.get_posterior_dist()
        kl_loss_conv3 = self.conv3.get_kl_divergence(server_conv3_q_weight, server_conv3_q_bias)

        server_conv4_q_weight, server_conv4_q_bias = server_model.conv4.get_posterior_dist()
        kl_loss_conv4 = self.conv4.get_kl_divergence(server_conv4_q_weight, server_conv4_q_bias)

        server_conv5_q_weight, server_conv5_q_bias = server_model.conv5.get_posterior_dist()
        kl_loss_conv5 = self.conv5.get_kl_divergence(server_conv5_q_weight, server_conv5_q_bias)

        return kl_loss_conv1 + kl_loss_conv2 + kl_loss_conv3 + kl_loss_conv4 + kl_loss_conv5

    def get_reg_loss(self):
        reg_loss_conv1_mu, reg_loss_conv1_sig = self.conv1.get_reg_loss()
        reg_loss_conv2_mu, reg_loss_conv2_sig = self.conv2.get_reg_loss()
        reg_loss_conv3_mu, reg_loss_conv3_sig = self.conv3.get_reg_loss()
        reg_loss_conv4_mu, reg_loss_conv4_sig = self.conv4.get_reg_loss()
        reg_loss_conv5_mu, reg_loss_conv5_sig = self.conv5.get_reg_loss()

        reg_loss_mu = reg_loss_conv1_mu + reg_loss_conv2_mu + reg_loss_conv3_mu + reg_loss_conv4_mu + reg_loss_conv5_mu
        reg_loss_sig = reg_loss_conv1_sig + reg_loss_conv2_sig + reg_loss_conv3_sig + reg_loss_conv4_sig + reg_loss_conv5_sig

        return reg_loss_mu, reg_loss_sig
