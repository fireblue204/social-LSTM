import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from calculate import getCoef
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class logLikelihood_loss(nn.Module):
    def __init__(self):
        super(logLikelihood_loss, self).__init__()
    def forward(self,y_target,o_pred):
        loss=compute_LogLikelihood_loss(y_target,o_pred)
        return loss

def compute_LogLikelihood_loss(y_target,o_pred):
    """
    :param y_target: (batch_size,pred_len,ped_num,3)
    :param o_pred: (batch_size,pred_len,ped_num,out_dim)
    :return: loss
    """
    epsilon = 1e-10
    mux,muy,sx,sy,corr=getCoef(o_pred)#shape:(batch_size,pred_len,ped_num)

    norm_x=(y_target[...,1]-mux)/sx
    norm_y=(y_target[...,2]-muy)/sy
    sigma_xy=sx*sy

    z=norm_x**2+norm_y**2-2*corr*norm_x*norm_y
    onemrho=1-corr**2

    #onemrho=torch.clamp(onemrho,min=epsilon)
    #onemrho+=epsilon

    #pdf1=torch.exp(-z/2*onemrho)/(2*np.pi*(sigma_xy*torch.sqrt(onemrho)))
    top=torch.exp(-z/(2*onemrho))
    bottom=2*np.pi*(sigma_xy*torch.sqrt(onemrho))

    pdf=top/bottom

    total_P=torch.tensor(0.0,device=device)
    for b in range(pdf.shape[0]):
        for p in range(pdf.shape[2]):
            P = torch.tensor(0.0, device=device)
            for t in range(pdf.shape[1]):
                p0=pdf[b,t,p]
                P+=-torch.log(torch.clamp(p0,min=epsilon))
            total_P+=P

    result=total_P/pdf.shape[2]
    #print(result.item())
    return result