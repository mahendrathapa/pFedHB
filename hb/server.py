import torch
import numpy as np
import torch.nn as nn


class HBServer:

    def __init__(self, model_base, model_head, device, args):
        self.clients = []
        self.num_data = 0
        self.model_base = model_base
        self.model_head = model_head
        self.device = device
        self.args = args

        self.opt_global_head = torch.optim.Adam(self.model_head.parameters(), lr=args['global_lr_head'])
        self.opt_global_base = torch.optim.Adam(self.model_base.parameters(), lr=args['global_lr_base'])

        self.sch_global_head = torch.optim.lr_scheduler.MultiStepLR(self.opt_global_head, milestones=[60, 80], gamma=0.1)
        self.sch_global_base = torch.optim.lr_scheduler.MultiStepLR(self.opt_global_base, milestones=[60, 80], gamma=0.1)

    def connect_clients(self, clients):
        self.clients = clients
        self.num_data = np.sum(len(x.train_loader.dataset) for x in self.clients)

    def update_head_model(self, sampled_clients):
        
        self.model_base.eval()
        self.model_head.train()

        sampled_data = np.sum(
            len(self.clients[x].train_loader.dataset)
            for x in sampled_clients
        )

        for epoch in range(self.args['global_head_epochs']):
            kl_loss = torch.tensor(0, dtype=torch.float).to(self.device)
            self.opt_global_head.zero_grad()

            server_q_weight, server_q_bias = self.model_head.get_posterior_dist()

            for client_idx in sampled_clients:
                client = self.clients[client_idx]
                client.model_head.eval()
                client.model_base.eval()

                coeff = len(self.clients[client_idx].train_loader.dataset)/sampled_data
                kl_loss += client.model_head.get_kl_divergence(server_q_weight, server_q_bias).mul(coeff)

            reg_loss_mu, reg_loss_sig = self.model_head.get_reg_loss()

            loss = kl_loss + self.args['mu_regularizer'] * reg_loss_mu + self.args['sigma_regularizer'] * reg_loss_sig

            loss.backward()
            self.opt_global_head.step()

        return kl_loss.item(), reg_loss_mu.item(), reg_loss_sig.item()

    def update_base_model(self, sampled_clients):

        self.model_base.train()
        self.model_head.eval()

        sampled_data = np.sum(
            len(self.clients[x].train_loader.dataset)
            for x in sampled_clients
        )

        for epoch in range(self.args['global_base_epochs']):

            kl_loss = torch.tensor(0, dtype=torch.float).to(self.device)
            self.opt_global_base.zero_grad()

            for client_idx in sampled_clients:
                client = self.clients[client_idx]
                client.model_head.eval()
                client.model_base.eval()

                coeff = len(self.clients[client_idx].train_loader.dataset)/sampled_data
                kl_loss += client.model_base.get_kl_divergence(self.model_base).mul(coeff)

            reg_loss_mu, reg_loss_sig = self.model_base.get_reg_loss()

            loss = kl_loss + self.args['mu_regularizer'] * reg_loss_mu + self.args['sigma_regularizer'] * reg_loss_sig

            loss.backward()
            self.opt_global_base.step()

        server_model_base = self.model_base.state_dict()
        for client in self.clients:
            client.model_base.load_state_dict(server_model_base)

        return kl_loss.item(), reg_loss_mu.item(), reg_loss_sig.item()

    def test(self, test_loaders, class_2_superclass, relabel):
        if isinstance(test_loaders, list) or isinstance(test_loaders, np.ndarray):
            test_loaders = test_loaders
        else:
            test_loaders = [test_loaders]

        self.model_base.eval()
        self.model_head.eval()

        test_loss = 0
        correct = 0
        preds_list = []
        targets_list = []
        criterion = nn.CrossEntropyLoss(reduction="none")
        with torch.no_grad():
            for test_loader in test_loaders:
                for data, target in test_loader:
                    if relabel:
                        target = torch.tensor([class_2_superclass[t.item()] for t in target],
                                              dtype=torch.int64)

                    data, target = data.to(self.device), target.to(self.device)
                    z = self.model_base(data, sample_W=False)
                    output = self.model_head(z, sample_W=False)
                    test_loss += criterion(input=output, target=target).sum().item()
                    output = torch.softmax(output, dim=1)
                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()
                    preds_list += output.cpu().tolist()
                    targets_list += target.squeeze().cpu().tolist()

        test_loss /= np.sum(len(ld.dataset) for ld in test_loaders)
        acc = correct / np.sum(len(ld.dataset) for ld in test_loaders)
        return test_loss, acc
