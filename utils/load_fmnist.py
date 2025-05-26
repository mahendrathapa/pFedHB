import numpy as np
from utils.utils import *
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


def load_fmnist(n_clients,
                 path,
                 device,
                 max_data=None,
                 n_labels=2,
                 batch_size=20,
                 seed=42):

    def get_dataset(train):
        dataset = torchvision.datasets.FashionMNIST(root=path,
                                                    train=train,
                                                    download=True,
                                                    transform=fmnist_transform())

        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        data, targets = next(iter(dataloader))
        if max_data and train:
            data_per_class = max_data // 10
            ind = []
            for i in range(10):
                ind += np.arange(len(targets))[targets == i].tolist()[:data_per_class]
            data, targets = data[ind], targets[ind]
        return data, targets

    train_data, train_targets = get_dataset(True)
    test_data, test_targets = get_dataset(False)

    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)

    return load_data(train_data=train_data,
                     train_targets=train_targets,
                     test_data=test_data,
                     test_targets=test_targets,
                     n_classes=10,
                     n_clients=n_clients,
                     n_labels=n_labels,
                     batch_size=batch_size,
                     seed=seed)


def load_data(train_data,
              train_targets,
              test_data,
              test_targets,
              n_classes,
              n_clients,
              n_labels,
              batch_size,
              seed=1234):
    client_train_datasets, client_labels = get_split_datasets(data=train_data,
                                                              targets=train_targets,
                                                              n_clients=n_clients,
                                                              n_labels=n_labels,
                                                              n_classes=n_classes,
                                                              seed=seed)

    test_datasets = []
    test_targets_np = test_targets.cpu().numpy()
    for i in range(n_classes):
        cls_ind = np.arange(len(test_data))[test_targets_np == i]
        test_datasets.append(
            TensorDataset(test_data[cls_ind], test_targets[cls_ind])
        )

    train_loaders = np.array([
        DataLoader(d, batch_size=batch_size, shuffle=True) for d in client_train_datasets
    ])
    test_loaders = np.array([
        DataLoader(d, batch_size=500, shuffle=False) for d in test_datasets
    ])

    return train_loaders, test_loaders, client_labels


def get_split_datasets(
        data,
        targets,
        n_clients,
        n_labels,
        n_classes,
        seed,
):
    if n_classes % n_labels != 0:
        raise ValueError("# of classes must be exactly divided by # of local labels.")
    if n_clients % (n_classes // n_labels) != 0:
        raise ValueError("# of clients must be exactly divided by # of data block.")

    # Initialize clients' labels
    rng = np.random.default_rng(seed)
    client_labels = []
    for client in range(n_clients // (n_classes // n_labels)):
        classes = np.arange(n_classes)
        rng.shuffle(classes)
        client_labels += np.split(classes, n_classes // n_labels)

    split_strategy = RandomDataSplit(targets=targets,
                                     n_classes=n_classes,
                                     n_clients=n_clients,
                                     n_labels=n_labels,
                                     seed=seed)
    client_indices = split_strategy.assign(client_labels)

    client_datasets = []
    for client in range(n_clients):
        ind = client_indices[client]
        client_data = data[ind]
        client_targets = targets[ind]
        client_d = TensorDataset(client_data, client_targets)
        client_datasets.append(client_d)

    return client_datasets, client_labels


def fmnist_transform():
    return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,),
                                     (0.3530,)),
            ])
