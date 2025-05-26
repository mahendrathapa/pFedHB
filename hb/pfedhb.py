import copy
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from model.BNN import BNNLinear
from model.fmnistLinear import BNNFMNISTBase
from model.cifarCNN import BNNCifarBase
from model.cifar100CNN import BNNCifar100Base
from hb.server import HBServer as Server
from hb.hb_client import HBClient as Client
from hb.client_sampling import BinomialSampler
from utils.utils import set_random
from utils.load_dataloaders import load_dataloaders


def train(args):

    run_id = datetime.datetime.now().strftime(("%Y-%m-%d %H:%M:%S"))
    run_id = "out/" + f"{str(run_id)}_{args['run_name']}"
    print(f"Run id: {run_id}")

    writer = SummaryWriter(run_id + "/logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random(args['seed'])

    # prepare the dataset
    train_loaders, test_loaders, local_classes, client_labels, \
    class_2_superclass, client_test_ind, relabel = load_dataloaders(
        dataset=args['dataset'],
        n_clients=args['n_clients'],
        n_labels=args['n_labels'],
        relabel=args['relabel'],
        device=device,
        batch_size=args['batch_size'],
        path_to_data=args['path_to_data'],
        max_data=args['max_data'],
        seed=args['seed']
    )

    # build up the model
    if args['model'] == 'CNNCifar':
        model_base = BNNCifarBase().to(device)
    elif args['model'] == 'CNNCifar100':
        model_base = BNNCifar100Base().to(device)
    elif args['model'] == 'LinearFMNIST':
        model_base = BNNFMNISTBase().to(device)

    if args['head_init_type'] == "uniform":
        model_head = BNNLinear(model_base.d_feature, local_classes, init_type='uniform').to(device)
    elif args['head_init_type'] == "normal":
        model_head = BNNLinear(model_base.d_feature, local_classes, init_type='normal').to(device)

    client_sampler = BinomialSampler(args['n_clients'], p=args['sampling_rate'])
    server = Server(model_base=model_base,
                    model_head=model_head,
                    device=device,
                    args=args)

    clients = [Client(
        device=device,
        model_base=copy.deepcopy(model_base),
        model_head=copy.deepcopy(model_head),
        train_loader=train_loaders[n],
        client_labels=client_labels[n],
        class_2_superclass=class_2_superclass,
        args=args
    ).to(device) for n in range(args['n_clients'])]

    server.connect_clients(clients)

    local_lr_head = args['local_lr_head']
    local_lr_base = args['local_lr_base']

    for round_idx in range(args['n_rounds']):
        head_nll_losses = []
        head_kl_losses = []
        base_nll_losses = []

        for client_idx in range(args['n_clients']):
            # personalization
            client = clients[client_idx]
            head_nll_loss, head_kl_loss, base_nll_loss = client.train(server_model_base=server.model_base,
                                                                      server_model_head=server.model_head,
                                                                      relabel=relabel,
                                                                      local_lr_head=local_lr_head,
                                                                      local_lr_base=local_lr_base)

            head_nll_losses.append(head_nll_loss * len(client.train_loader.dataset) / server.num_data)
            head_kl_losses.append(head_kl_loss * len(client.train_loader.dataset) / server.num_data)
            base_nll_losses.append(base_nll_loss * len(client.train_loader.dataset) / server.num_data)

        head_nll_losses_avg = sum(head_nll_losses)
        head_kl_losses_avg = sum(head_kl_losses)
        base_nll_losses_avg = sum(base_nll_losses)

        writer.add_scalar('train_client/head_nll', head_nll_losses_avg, round_idx + 1)
        writer.add_scalar('train_client/base_nll', base_nll_losses_avg, round_idx + 1)
        writer.add_scalar('train_client/head_kl_loss', head_kl_losses_avg, round_idx + 1)

        sampled_clients = client_sampler.sample()

        if round_idx in list(range(0, args['n_rounds'], 5)) + [args['n_rounds'] - 1]:
            test_losses = []
            test_accs = []
            # test personalized models
            for client_idx in range(args['n_clients']):
                client = clients[client_idx]
                test_loss, test_acc = client.test(test_loaders=[
                    test_loaders[idx] for idx in client_test_ind[client_idx]
                ], relabel=relabel)

                test_losses.append(test_loss)
                test_accs.append(test_acc)

            test_losses_avg = sum(test_losses) / len(test_losses)
            test_accs_avg = sum(test_accs) / len(test_accs)

            # update and evaluate the global model
            server_base_kl_loss, server_base_reg_loss_mu, server_base_reg_loss_sig = server.update_base_model(sampled_clients)
            server_head_kl_loss, server_head_reg_loss_mu, server_head_reg_loss_sig = server.update_head_model(sampled_clients)

            test_global_loss, test_global_acc = server.test(test_loaders=test_loaders,
                                                            relabel=relabel,
                                                            class_2_superclass=class_2_superclass)

            print(
                f"Round {round_idx+1}, " +
                f"PMs test loss = {test_losses_avg:.4f}, " +
                f"PMs test acc = {test_accs_avg:.4f}, " +
                f"GM test loss = {test_global_loss:.4f}, " +
                f"GM test acc = {test_global_acc:.4f}, "
            )

            writer.add_scalar('test/pm_loss', test_losses_avg, round_idx + 1)
            writer.add_scalar('test/pm_acc', test_accs_avg, round_idx + 1)
            writer.add_scalar('test/gm_loss', test_global_loss, round_idx + 1)
            writer.add_scalar('test/gm_acc', test_global_acc, round_idx + 1)

        else:
            # update the global model
            server_base_kl_loss, server_base_reg_loss_mu, server_base_reg_loss_sig = server.update_base_model(sampled_clients)
            server_head_kl_loss, server_head_reg_loss_mu, server_head_reg_loss_sig = server.update_head_model(sampled_clients)

            writer.add_scalar('train_server/base_kl_loss', server_base_kl_loss, round_idx + 1)
            writer.add_scalar('train_server/base_reg_loss_mu', server_base_reg_loss_mu, round_idx + 1)
            writer.add_scalar('train_server/base_reg_loss_sig', server_base_reg_loss_sig, round_idx + 1)

            writer.add_scalar('train_server/head_kl_loss', server_head_kl_loss, round_idx + 1)
            writer.add_scalar('train_server/head_reg_loss_mu', server_head_reg_loss_mu, round_idx + 1)
            writer.add_scalar('train_server/head_reg_loss_sig', server_head_reg_loss_sig, round_idx + 1)

        server.sch_global_head.step()
        server.sch_global_base.step()

        # print("---------------------------------------")
