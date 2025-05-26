import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HBClient(nn.Module):
    def __init__(
            self,
            device,
            model_base,
            model_head,
            train_loader,
            client_labels,
            class_2_superclass,
            args):
        super(HBClient, self).__init__()
        self.device = device
        self.client_labels = client_labels
        self.train_loader = train_loader
        self.class_2_superclass = class_2_superclass

        self.model_base = model_base
        self.model_head = model_head 
        self.args = args

        self.train_use_mc = True
        self.test_use_mc = False

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def test(self, test_loaders, relabel):
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
        num = 0
        with torch.no_grad():
            for test_loader in test_loaders:
                for data, target in test_loader:
                    if relabel:
                        target = torch.tensor([self.class_2_superclass[t.item()] for t in target],
                                              dtype=torch.int64)
                    inds = [t.cpu().item() in self.client_labels for t in target]
                    data, target = data[inds].to(self.device), target[inds].to(self.device)

                    # Option 1: use mean of the variational approximation
                    if not self.test_use_mc:
                        z = self.model_base(data, sample_W=False)
                        output = self.model_head(z, sample_W=False)
                        test_loss += self.criterion(input=output, target=target).sum().item()
                        output = torch.softmax(output, dim=1)
                    # Option 2: use distribution of the variational approximation
                    else:
                        output = 0
                        for k in range(self.args['n_mc']):
                            z = self.model_base(data, sample_W=True)
                            output += torch.softmax(self.model_head(z, sample_W=True), dim=1)
                        output /= self.args['n_mc']
                        output = torch.log(output)
                        test_loss += F.nll_loss(output, target, reduction='sum').item()

                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    num += target.numel()
                    preds_list += output.cpu().tolist()
                    targets_list += target.squeeze().cpu().tolist()

        test_loss /= num
        acc = correct / num
        return test_loss, acc

    def train(self, server_model_base, server_model_head, relabel, local_lr_head, local_lr_base):

        opt_local = torch.optim.Adam(self.model_head.parameters(), local_lr_head)
        opt_base = torch.optim.Adam(self.model_base.parameters(), local_lr_base)

        server_model_base.eval()
        server_model_head.eval()
        self.model_base.train()
        self.model_head.train()

        for p in self.model_head.parameters():
            p.requires_grad = True
        for p in self.model_base.parameters():
            p.requires_grad = False

        head_train_loss = 0
        head_kl_loss = 0
        for epoch in range(self.args['local_head_epochs']):
            data, target = zip(*[(d, t) for d, t in self.train_loader])
            data = torch.cat(data, dim=0)
            target = torch.cat(target, dim=0)
            if relabel:
                target = torch.tensor([self.class_2_superclass[t.item()] for t in target],
                                      dtype=torch.int64)

            opt_local.zero_grad()
            data, target = data.to(self.device), target.to(self.device)

            # use multiple mc samples to estimate the expecation
            nll = torch.tensor(0, dtype=torch.float).to(self.device)
            for k in range(self.args['n_mc']):
                z = self.model_base(data, sample_W=True)
                output = self.model_head(z, sample_W=True)
                loss = self.criterion(input=output, target=target)
                head_train_loss += loss.sum().div(self.args['n_mc']).item()
                nll += loss.sum().div(self.args['n_mc'])

            with torch.no_grad():
                server_q_weight, server_q_bias = server_model_head.get_posterior_dist()

            kl = self.model_head.get_kl_divergence(server_q_weight, server_q_bias)
            loss = nll + self.args['kl_coeff_head'] * kl
            loss.backward()
            opt_local.step()

            head_kl_loss += kl.item()

        head_train_loss /= (self.args['local_head_epochs'] * len(self.train_loader.dataset))
        head_kl_loss /= (self.args['local_head_epochs'] * len(self.train_loader.dataset))

        for p in self.model_head.parameters():
            p.requires_grad = False
        for p in self.model_base.parameters():
            p.requires_grad = True

        base_train_loss = 0
        for epoch in range(self.args['local_base_epochs']):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if relabel:
                    target = torch.tensor([self.class_2_superclass[t.item()] for t in target],
                                          dtype=torch.int64)


                opt_base.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
            
                nll_loss = torch.tensor(0, dtype=torch.float).to(self.device)
                for _ in range(self.args['n_mc']):
                    z = self.model_base(data, sample_W=True)
                    output = self.model_head(z, sample_W=True)
                    nll_loss += self.criterion(input=output, target=target).mean().div(self.args['n_mc'])

                kl_loss = self.model_base.get_kl_divergence(server_model_base)

                loss = nll_loss + self.args['kl_coeff_base'] * kl_loss

                loss.backward()
                opt_base.step()

                base_train_loss += nll_loss.item()

        base_train_loss /= (self.args['local_base_epochs'] * len(self.train_loader))

        return head_train_loss, head_kl_loss, base_train_loss
