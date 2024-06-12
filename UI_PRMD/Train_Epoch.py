import torch
import wandb
from torch import nn

##定义邻接矩阵
# def contrastive_loss(features1,features2, scores, samemodelmargin, diffmodalmargin,scale=10):
#     l1=featrue_contrastive_loss(features1, features1, scores, margin=samemodelmargin, scale=scale)
#     l2=featrue_contrastive_loss(features2, features2, scores, margin=samemodelmargin, scale=scale)
#     l3=featrue_contrastive_loss(features1, features2, scores, margin=diffmodalmargin, scale=scale)
#     return l1+l2+l3



# def featrue_contrastive_loss(features1, features2, scores, margin, scale):
#     """
#     自定义损失函数，仅当实际特征距离与期望距离之间的差异超过margin时才产生损失。
#
#     参数:
#     - features1: Tensor, 第一组特征，形状为(batch_size, feature_dim)。
#     - features2: Tensor, 第二组特征，形状为(batch_size, feature_dim)。
#     - scores: Tensor, 对应的得分，形状为(batch_size,)。
#     - margin: float, 容忍的距离差异边界。
#     - scale: float, 缩放因子，用于调整分数差异对距离的影响。
#     """
#     batch_size = features1.size(0)
#     loss = 0.0
#     for i in range(batch_size):
#         for j in range(batch_size):
#             # 计算特征之间的欧几里得距离
#             feature_distance = (features1[i] - features2[j]).pow(2).sum().sqrt()
#             # 计算分数之间的绝对差异
#             score_diff = torch.abs(scores[i] - scores[j])
#             # 计算期望的特征距离
#             expected_distance = score_diff * scale
#             # 计算距离差异并应用边界
#             distance_diff = (feature_distance - expected_distance).abs()
#             # 当距离差异大于margin时，才计算损失
#             if distance_diff > margin:
#                 loss += distance_diff - margin  # 仅计算超过margin的部分
#
#     # 对所有样本对的损失求平均
#     loss /= (batch_size * batch_size)
#     return loss




def getlossfn(loss):
    if loss == "huber":
        loss_fn = nn.SmoothL1Loss()
    elif loss == "l1":
        loss_fn = nn.L1Loss()
    elif loss == "l2":
        loss_fn = nn.MSELoss()
    return loss_fn

def get_bacth_distil_lloss(Xteacher, Yteacher, Xstudnet,sigma):
    # 使用广播计算 Y 中所有元素对之间的差的绝对值
    distances_y = torch.abs(Yteacher.view(-1, 1) - Yteacher)

    # 计算高斯权重
    weights = torch.exp(-distances_y ** 2 / (2 * sigma ** 2))

    # 扩展 Xtrain 和 X 以计算所有样本对的平方差
    diff = Xstudnet.unsqueeze(1) - Xteacher
    squared_diff = diff ** 2

    # 沿最后一个轴求和以计算欧氏距离的平方
    squared_distances = torch.sum(squared_diff, axis=2)

    # 开方以得到实际的欧氏距离
    distances = torch.sqrt(squared_distances)

    # 使用广播将每行距离与对应的权重相乘
    weighted_distances = distances * weights

    # 计算总损失
    total_loss = torch.sum(weighted_distances)

    return total_loss



def train_epoch(network, loader, optimizer,lossfn,device,alpha,sigma,scheduler):
    train_truth=[]
    train_pred=[]
    network.train()

    loss_fn=getlossfn(lossfn)


    for  i, batch  in enumerate(loader):
        sklpos, labels = batch
        sklpos, labels = sklpos.to(device).float(),labels.to(device).float()
        optimizer.zero_grad()

        # ➡ Forward pass
        combined_input = sklpos

        cls_1,cls_2,cls_3,_,x1_feature,x3_feature = network(combined_input)


        train_pred.append(cls_3.detach())  # 收集预测值
        train_truth.append(labels.detach())  # 收集真实标签



        task_loss= loss_fn(cls_3, labels.float())

        distill_loss=get_bacth_distil_lloss(x1_feature, cls_1, x3_feature, sigma)*0.0001+loss_fn(cls_1, labels.float())+0.5*loss_fn(cls_2, labels.float())

        #c_loss=contrastive_loss(modalfeatures1, modalfeatures2, labels, samemodelmargin=samemodelmargin, diffmodalmargin=diffmodalmargin, scale=scale)
        loss=0.8*task_loss+0.2*distill_loss
        loss=task_loss


        # ⬅ Backward pass + weight update
        loss.backward()


        optimizer.step()
        scheduler.step()


        wandb.log({"task loss": task_loss.cpu().item(),"student epoch loss": loss.item(),"alpha": alpha})

        #wandb.log({"task loss": task_loss.cpu().item(),"distillation loss": distill_loss.item(),"student epoch loss": loss.item(),"alpha": alpha})

    train_pred = torch.cat(train_pred, dim=0)  # 假设预测值是在第0维度上拼接
    train_truth = torch.cat(train_truth, dim=0)  # 同样假设真实标签是在第0维度上拼接
    train_l1loss=torch.abs(train_pred - train_truth).mean()        # 打印验证损失
    print("train L1Loss:", train_l1loss)







    return train_l1loss,train_pred,train_truth


