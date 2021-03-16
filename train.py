import os
import torch
import numpy as np
from argparse import Namespace, ArgumentParser
from torch import optim
from dataProcess import *
from smodel import *
from loss import *
from modelConfig import modelConfig
from datetime import datetime
from calculate import *
from torch.utils.data import TensorDataset,DataLoader,Dataset
#from torch.nn.utils import clip_grad_value_


#root_dir="D:\\ImportantSoft\\VSworkspace\\social-LSTM\\torch_start\\"
#out_dir=root_dir+("output")
#config_dir=root_dir+"config.json"
def load_train_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\config.json")
    parser.add_argument("--out_root", type=str, default="results\\")
    return parser.parse_args()

def main():
    print(torch.cuda.is_available())
    global device
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running device is {device}\n")

    args = load_train_args()
    config = modelConfig.load_model_config(args.config)
    time_str=now_to_str()
    out_dir=os.path.join(args.out_root,"{}\\".format(time_str))
    #out_dir=out_dir.join("model.pt")

    #在这里遍历不同文件夹下的数据集，处理完的训练数据再进行拼接
    train_data, test_data = dataLoader(config).get_train_test_data()
    social_model = SocialModel(config)
    #train(out_dir,config,train_data,social_model)
    social_model.load_state_dict(torch.load(os.path.join(out_dir,"model.pt")))
    test(config,test_data,social_model)


def train(out_dir:str,config:modelConfig,train_data,social_model)->None:
    obs_train,pred_train = split_data(config.obs_len,config.pred_len,*train_data)
    x_obs_input,_=obs_train
    _,y_pred_target=pred_train
    x_obs_input=x_obs_input.float()
    y_pred_target=y_pred_target.float()
    x_obs_input=x_obs_input.resize_(x_obs_input.shape[0]-1,x_obs_input.shape[1],x_obs_input.shape[2],x_obs_input.shape[3])
    y_pred_target=y_pred_target.resize_(y_pred_target.shape[0]-1,y_pred_target.shape[1],y_pred_target.shape[2],y_pred_target.shape[3])

    # x_obs_input_bs=DataLoader(x_obs_input,batch_size=2,shuffle=True)
    # y_pred_target_bs=DataLoader(y_pred_target,batch_size=2,shuffle=True)

    #criterion=torch.nn.MSELoss(reduction="mean")
    criterion=logLikelihood_loss()
    optimizer=optim.RMSprop(social_model.parameters(),lr=0.003)
    print(x_obs_input.shape)
    batch_size=2

    for epoch in range(10):
        train_data_x = DataLoader(x_obs_input, batch_size=batch_size, shuffle=True)
        train_data_y = DataLoader(y_pred_target, batch_size=batch_size, shuffle=True)
        bs=0
        for data in zip(train_data_x,train_data_y):
            bs=bs+1
            x_input,y_target=data
            x_pred,o_pred=social_model(x_input)
            optimizer.zero_grad()
            loss=criterion(y_target,o_pred)
            ade=get_ade(y_target,x_pred)
            loss.backward()
            optimizer.step()
        # for bs in range(int(x_obs_input.shape[0]/batch_size)+1):
        #     if(bs==int(x_obs_input.shape[0]/batch_size)):
        #         x_obs_input_bs=x_obs_input[bs*batch_size:]
        #         y_pred_target_bs=y_pred_target[bs*batch_size:]
        #     x_obs_input_bs=x_obs_input[bs*batch_size:(bs+1)*batch_size]
        #     y_pred_target_bs=y_pred_target[bs*batch_size:(bs+1)*batch_size]
        #     #x_obs_input_bs=torch.unsqueeze(x_obs_input_bs,0)
        #     #y_pred_target_bs=torch.unsqueeze(y_pred_target_bs,0)
        #     x_pred, o_pred = social_model(x_obs_input_bs)
        #     x_pred.requires_grad=True
        #     loss=criterion(y_pred_target_bs,o_pred)
        #     ade=get_ade(y_pred_target_bs,x_pred)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     #clip_grad_value_(social_model.parameters(),5)
        #     #for name,parms in social_model.named_parameters():
        #         #print('-->name:',name,'-->grad_requires:',parms.requires_grad,'-->grad_value:',parms.grad)
        #     optimizer.step()
            print("Epoch:{} batch:{} loss:{} ade:{}".format(epoch, bs, loss.item(), ade.item()))

    os.makedirs(out_dir,exist_ok=True)
    torch.save(social_model,os.path.join(out_dir,"model.pt"))


def test(config:modelConfig,test_data,social_model)->None:
    social_model.eval()
    obs_test,pred_test=split_data(config.obs_len,config.pred_len,*test_data)
    x_test,_=obs_test
    _,y_target_test=pred_test
    #model=torch.load(os.path.join(out_dir,"model.pt"))
    x_test_pred,o_test_pred=social_model(modelConfig,x_test)
    predictions=x_test_pred
    return predictions

def visualize(predictions):
    

def now_to_str(format: str = "%Y%m%d%H%M%S") -> str:
    return datetime.now().strftime(format)

if __name__=="__main__":
    main()
