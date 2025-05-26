# Harnessing Heterogeneous Statistical Strength for Personalized Federated Learning via Hierarchical Bayesian Inference

## Setup
```
conda create -n pfedhb python==3.8
conda activate pfedhb
pip install -r requirements.txt
```

## Run

For label distribution skew on FMNIST
```
python -m experiment.run_experiment --config experiment/configs/fmnist.json
```

For label distribution skew on CIFAR10
```
python -m experiment.run_experiment --config experiment/configs/cifar10.json
```

For label concept drift on CIFAR100
```
python -m experiment.run_experiment --config experiment/configs/cifar100.json
```
