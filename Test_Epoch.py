import torch
from scipy.stats import spearmanr

from data_processing import Data_Loader




def val_epoch(network, val_loader,dataloader,device):
    val_truth=[]
    val_pred=[]


    network.eval()

    with torch.no_grad():  # 在验证阶段不需要计算梯度
        for batch in val_loader:
            sklpos, sklori, labels = batch
            sklpos, sklori, labels = sklpos.to(device).float(), sklori.to(device).float(), labels.to(device).float()
            combined_input = torch.cat((sklpos, sklori), dim=3)
            _,_, predictions, _, _, _ = network(combined_input)



            predictions = torch.from_numpy(dataloader.sc3.inverse_transform(predictions.cpu().numpy()))
            labels = torch.from_numpy(dataloader.sc3.inverse_transform(labels.cpu().numpy()))


            val_pred.append(predictions)  # 收集预测值
            val_truth.append(labels)  # 收集真实标签



        val_pred = torch.cat(val_pred, dim=0)  # 假设预测值是在第0维度上拼接
        val_truth = torch.cat(val_truth, dim=0)  # 同样假设真实标签是在第0维度上拼接


        val_loss=torch.abs(val_pred - val_truth).mean()
        # 打印验证损失
        print("Validation L1Loss:", val_loss)

        val_correlation = spearmanr(val_pred.cpu().numpy(), val_truth.cpu().numpy())
        print("Validation_correlation:", val_correlation)


        return val_loss.item(),val_correlation,val_truth,val_pred


def test_epoch(network, val_loader,dataloader,device):
    val_truth=[]
    val_pred=[]


    network.eval()

    with torch.no_grad():  # 在验证阶段不需要计算梯度
        for batch in val_loader:
            sklpos, sklori, labels = batch
            sklpos, sklori, labels = sklpos.to(device).float(), sklori.to(device).float(), labels.to(device).float()
            combined_input = torch.cat((sklpos, sklori), dim=3)

            _, _,predictions, status, _, _ = network(combined_input)

            predictions = torch.from_numpy(dataloader.sc3.inverse_transform(predictions.cpu().numpy()))
            labels = torch.from_numpy(dataloader.sc3.inverse_transform(labels.cpu().numpy()))


            val_pred.append(predictions)  # 收集预测值
            val_truth.append(labels)  # 收集真实标签



        val_pred = torch.cat(val_pred, dim=0)  # 假设预测值是在第0维度上拼接
        val_truth = torch.cat(val_truth, dim=0)  # 同样假设真实标签是在第0维度上拼接


        ##MAD
        val_loss=torch.abs(val_pred - val_truth).mean()

        # 计算均方根误差（RMSE）
        squared_error = (val_pred - val_truth) ** 2
        mse = torch.mean(squared_error)
        rmse = torch.sqrt(mse)

        # 计算平均绝对百分比误差（MAPE）
        absolute_percentage_error = torch.abs((val_pred - val_truth) / val_truth)
        mape = torch.mean(absolute_percentage_error) * 100
        # 打印验证损失
        # 打印验证损失
        print("Test L1Loss:", val_loss)

        val_correlation = spearmanr(val_pred.cpu().numpy(), val_truth.cpu().numpy())
        print("Test_correlation:", val_correlation)


        return val_loss.item(),val_correlation,val_truth,val_pred,1,rmse,mape
