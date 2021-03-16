import torch
import numpy as np
from modelConfig import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def getCoef(outputs):
    x_mean,y_mean,x_std,y_std,cor=outputs[...,0],outputs[...,1],outputs[...,2],outputs[...,3],outputs[...,4]
    x_std=torch.exp(x_std)
    y_std=torch.exp(y_std)
    cor=torch.tanh(cor)
    return x_mean,y_mean,x_std,y_std,cor

def normal2d_sample(prev_o_t):
    """
    :param prev_o_t: (batch_size,ped_num,output_dim=5)
    :return: (batch_size,ped_num,2)
    """
    pred_pos=torch.zeros(prev_o_t.shape[0],prev_o_t.shape[1],2,device=device)
    x_mean, y_mean, x_std, y_std, cor=getCoef(prev_o_t)
    #x_mean=prev_o_t[:,:,0]
    #y_mean=prev_o_t[:,:,1]

    #x_std=prev_o_t[:,:,2]
    #y_std=prev_o_t[:,:,3]

    #cor=torch.tanh(prev_o_t[:,:,4])

    x_var = torch.square(x_std)  # var
    y_var = torch.square(y_std)  # var
    xy_cov = x_std * y_std * cor  # 协方差

    #cov=torch.tensor([[x_var,xy_cov],[xy_cov,y_var]])
    #cov=cov.numpy()
    for p in range(prev_o_t.shape[1]):
        for b in range(x_mean.shape[0]):
            mean=torch.tensor([x_mean[b,p],y_mean[b,p]])
            mean=mean.numpy()
            x_var_p=x_var[b,p]
            y_var_p=y_var[b,p]
            xy_cov_p=xy_cov[b,p]
            cov_p=torch.tensor([[x_var_p,xy_cov_p],[xy_cov_p,y_var_p]])
            cov_p=cov_p.numpy()
            pos=np.random.multivariate_normal(mean,cov_p,1)
            pos=torch.from_numpy(pos)
            pred_pos[b, p] = pos
        # mean=torch.tensor([x_mean[:,p],y_mean[:,p]])
        # mean=mean.numpy()
        # x_var_p=x_var[:,p]
        # y_var_p=y_var[:,p]
        # xy_cov_p=xy_cov[:,p]
        # cov_p = torch.tensor([[x_var_p, xy_cov_p], [xy_cov_p, y_var_p]])
        # cov_p = cov_p.numpy()
        # pos = np.random.multivariate_normal(mean, cov_p, 1)
        #while np.any(pos<0):
            #pos = np.random.multivariate_normal(mean, cov_p, 1)
        # pos = torch.from_numpy(pos)
        # pos = pos.squeeze(1)
        # pred_pos[:,p] = pos
    return pred_pos

def get_ade(y_target,x_pred):
    '''
    :param y_target: (batch_size,pred_len,ped_num,3)
    :param x_pred: (batch_size,pred_len,ped_num,3)
    :return: 
    '''
    #square_sum=torch.sum(torch.square(y_target-x_pred))
    pred_len=x_pred.shape[1]
    #square_sum=square_sum/pred_len
    for b in range(x_pred.shape[0]):
        ade_sum=torch.tensor(0.0,device=device)
        for p in range(x_pred.shape[2]):
            target_x=y_target[b,:,p,1]
            target_y=y_target[b,:,p,2]
            #print("target_x:{}".format(target_x))
            #print("target_y:{}".format(target_y))
            pred_x=x_pred[b,:,p,1]
            pred_y=x_pred[b,:,p,2]
            #print("pred_x:{}".format(pred_x))
            #print("pred_y:{}".format(pred_y))

            squ_dis=(pred_x-target_x)**2+(pred_y-target_y)**2
            squ_dis_sum=torch.sum(squ_dis)
            squ_dis_avg=squ_dis_sum/pred_len

            ade_sum+=squ_dis_avg
        ade=ade_sum/x_pred.shape[2]

    return ade

#def get_fde(y_target,x_pred):











