from torch import optim
from adabelief_pytorch import AdaBelief


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    elif optimizer == "adabelief ":
        optimizer = AdaBelief(network.parameters(),
                                lr=learning_rate)
    return optimizer


def build_optimizers(networks, optimizer_type, learning_rate):
    optimizers = []
    for network in networks:
        if optimizer_type == "sgd":
            optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == "adam":
            optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer type")
        optimizers.append(optimizer)
    return optimizers


