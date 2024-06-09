import torch
import wandb
from torch import nn


def getlossfn(loss):
    if loss == "huber":
        loss_fn = nn.SmoothL1Loss()
    elif loss == "l1":
        loss_fn = nn.L1Loss()
    elif loss == "l2":
        loss_fn = nn.MSELoss()
    return loss_fn


def get_bacth_distil_lloss(Xteacher, Yteacher, Xstudnet,sigma,temperature,distill=False ):
    # 使用广播计算 Y 中所有元素对之间的差的绝对值
    distances_y = torch.abs(Yteacher.view(-1, 1) - Yteacher.view(1, -1))

    # 计算高斯权重
    weights = torch.exp(-distances_y ** 2 / (2 * sigma ** 2))
    if not distill:
        weights.fill_diagonal_(0)
    # 扩展 Xtrain 和 X 以计算所有样本对的平方差


    similarity_matrix = torch.nn.functional.cosine_similarity(Xteacher.unsqueeze(1), Xstudnet.unsqueeze(0), dim=2)
    exp_sim = torch.exp(similarity_matrix / temperature)
    sum_exp_sim = exp_sim.sum(1, keepdim=True)

    log_prob = torch.log(exp_sim / sum_exp_sim)

    # 计算损失，只考虑正样本对
    total_loss = - (weights * log_prob).sum(1) / weights.sum(1)



    return total_loss.mean()*0.1

def train_epoch(network, loader, optimizer,lossfn,device,ep,sigma,temperature,scheduler):
    train_truth=[]
    train_pred=[]
    network.train()

    loss_fn=getlossfn(lossfn)


    for  i, batch  in enumerate(loader):
        sklpos,sklori, labels = batch
        sklpos,sklori, labels = sklpos.to(device).float(), sklori.to(device).float(),labels.to(device).float()
        optimizer.zero_grad()

        # ➡ Forward pass
        combined_input = torch.cat((sklpos, sklori), dim=3)

        cls_1,cls_2,cls_3,_,x1_feature,x3_feature = network(combined_input)


        train_pred.append(cls_3.detach())  # 收集预测值
        train_truth.append(labels.detach())  # 收集真实标签


        if ep<500:
            task_loss=0.0
            contrastloss=0.0
            headloss = loss_fn(cls_1, labels.float())
        elif 500<=ep<2000:
            task_loss = 0.5*loss_fn(cls_3, labels.float())
            contrastloss=0.3*get_bacth_distil_lloss(x1_feature, labels.float(), x3_feature, sigma,temperature,distill=True)
            headloss = 0.2*loss_fn(cls_1, labels.float())
        else:
            task_loss = loss_fn(cls_3, labels.float())
            contrastloss=0.0
            headloss = 0.0


        loss=task_loss+contrastloss+headloss

        # ⬅ Backward pass + weight update
        loss.backward()


        optimizer.step()
        scheduler.step()


        wandb.log({"task loss": task_loss,"toal epoch loss": loss,"contrast loss": contrastloss,"head loss": headloss})


    train_pred = torch.cat(train_pred, dim=0)  # 假设预测值是在第0维度上拼接
    train_truth = torch.cat(train_truth, dim=0)  # 同样假设真实标签是在第0维度上拼接
    train_l1loss=torch.abs(train_pred - train_truth).mean()        # 打印验证损失
    print("train L1Loss:", train_l1loss)







    return train_l1loss,train_pred,train_truth


