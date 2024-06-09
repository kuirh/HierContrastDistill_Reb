import os
import pickle

import torch
import wandb
import numpy as np
import random
import argparse

from Build_Dataset import build_dataset
from Train_Epoch import train_epoch
from Build_Network import build_network
from Build_Optimizer import build_optimizer
from Test_Epoch import test_epoch,val_epoch
from Plot import plot_output_and_label
from adabelief_pytorch import AdaBelief

def set_random_seed(seed):
    """
    Set the seed for generating random numbers to ensure reproducibility.

    Parameters:
    - seed (int): The seed value.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch for CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch for CUDA operations
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
        torch.backends.cudnn.deterministic = True  # CUDNN operations
        torch.backends.cudnn.benchmark = False



sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        # Data parameters
        "batch_size": {"values": [32]},
        # Training parameters
        "optimizer": {"values": ["adam"]},
        "learning_rate": {"values": [0.01]},
        "teacher_epochs": {"values": [3500]},
        "student_epochs": {"values": [3500]},
        "loss": {"values": ["huber"]},


        # Additional parameters
        "use_cbam": {"values": [False]},
        "use_gating": {"values": [True]},  # Correct parameter based on function need

        "decay": {"values": [0.99]},
        "sigma": {"values": [0.5]},
        "temperature": {"values": [0.1]}
    },
}

def parse_tuple(string):
    try:
        return tuple(map(int, string.split(',')))
    except ValueError as e:
        # 可以在这里处理错误或者简单地抛出
        raise ValueError(f"Error parsing tuple from string '{string}': {e}")

def build_params_dict(config):
    # 直接从config提取并组织参数
    return {
        "MutiScaleModel_parameters": {
            "use_cbam": config.use_cbam,
            "use_gating": config.use_gating,
        }
    }

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Kimore")





def train(config=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    seed = random.randint(0, 655350)
    set_random_seed(seed)

    os.makedirs(str(seed), exist_ok=True)

    # Initialize a new wandb run
    #wandb.init(mode="offline")
    #with wandb.init(config=config):

    with wandb.init(dir=str(seed), name='ex5'):


        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        params_dict=build_params_dict(config)
        best_val_loss = float('inf')  # 设置一个很高的初始值

        no_improvement = 0  # Counter for early stopping
        patience = 4000  # Set the patience value in epochs

        train_loader, val_loader,test_loder,dataloder = build_dataset(device,'../DataProcess/Dataset_cTS/ex5', batch_size=config.batch_size)

        network = build_network(device,7,params_dict['MutiScaleModel_parameters'])

        optimizer = build_optimizer(network, 'adam', learning_rate=config.learning_rate)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10*1000, 35*1000], gamma=0.1)


        for epoch in range(config.teacher_epochs):
            lr = scheduler.get_last_lr()[0]
            train_l1loss, train_pred, train_truth=train_epoch(network, train_loader, optimizer,config.loss,device,epoch,config.sigma,config.temperature,scheduler)   # 训练一个epoch)

            val_loss, val_correlation, val_truth, val_pred = val_epoch(network, val_loader, dataloder, device)

            wandb.log({"train_L1Loss": train_l1loss*50, "val_loss": val_loss, "val_correlation": val_correlation[0], "epoch": epoch, "step": scheduler.last_epoch, "lr": lr})


            if val_loss < best_val_loss:
                plot_output_and_label(train_truth, train_pred, os.path.join(str(seed), f't_train'))  # 绘制输出和标签的散点图

            #if t_val_loss < t_best_val_loss :
                no_improvement = 0  # Reset patience counter
                print(f"Val loss decreased ({best_val_loss} --> {val_loss}). ")
                best_val_loss = val_loss  # 更新最佳验证损失
                # 保存模型
                if val_loss < 2:
                    plot_output_and_label(val_truth, val_pred, os.path.join(str(seed), f'{best_val_loss}_{val_correlation[0]}_val'))
                    torch.save(network.state_dict(), os.path.join(str(seed), 'best_model.pth'))
                wandb.log({"best_val_loss": best_val_loss})
            elif val_loss > 1.1:#训练炸了，给更多的容忍
                no_improvement = max(no_improvement - 30, 0)  #  patience counter
            else:
                no_improvement += 1
                if no_improvement > patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


        torch.save(network.state_dict(), os.path.join(str(seed), 'final_model.pth'))
        network.load_state_dict(torch.load(os.path.join(str(seed), 'best_model.pth')))
        test_loss, test_correlation,test_truth,test_pred,_,test_rmse,test_mape = test_epoch(network, test_loder,dataloder, device)
        print(f"Test loss: {test_loss}, Test correlation: {test_correlation[0]}, Test RMSE: {test_rmse}, Test MAPE: {test_mape}")
        wandb.log({"test_loss": test_loss, "test_correlation": test_correlation[0], "test_rmse": test_rmse, "test_mape": test_mape})







wandb.agent(sweep_id, train, count=20)