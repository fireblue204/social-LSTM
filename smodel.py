from torch import nn
from calculate import *
from modelConfig import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SocialModel(nn.Module):

    def __init__(self, config: modelConfig, dropout_prob=0.5)->None:
        super(SocialModel,self).__init__()
        self.input_dim = config.input_dim
        self.input_mid_dim = config.input_mid_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.grid_cell_size = config.grid_cell_size
        self.grid_size = config.grid_size
        self.ped_num = config.ped_num
        self.batch_size = config.batch_size
        self.width = config.img_width
        self.height = config.img_height
        self.obs_len=config.obs_len
        self.pred_len=config.pred_len
        self.seq_len=config.seq_len
        self.Phi = Phi(dropout_prob=dropout_prob)
        self.inputEmbedding = nn.Linear(self.input_dim, self.input_mid_dim).cuda()
        self.socialEmbedding = nn.Linear(self.grid_size**2*self.hidden_dim,self.hidden_dim).cuda()
        self.prevEmbedding = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        self.lstmLayer = nn.LSTMCell(self.hidden_dim+self.input_mid_dim,self.hidden_dim).cuda()
        self.outlayer = nn.Linear(self.hidden_dim,self.output_dim).cuda()


    #positions:(ped_num,3)
    #求某一帧的grid_mask
    def get_grid_mask(self,positions,width,height,t):#positions==x_t
        gridMask=torch.zeros(self.batch_size,self.ped_num,self.ped_num,self.grid_size*self.grid_size,device=device)
        for p in range(self.ped_num):
            #print(p)
            #print(positions[p,0])
            if positions[p,0] == 0:
                continue
            p_x,p_y=positions[p,1],positions[p,2]
            if torch.isnan(p_x) or torch.isnan(p_y):
                print(t)
                continue
            #R=self.grid_cell_size/2
            w_bound=self.grid_size*self.grid_cell_size/width
            h_bound=self.grid_size*self.grid_cell_size/height
            R_x=w_bound/2
            R_y=h_bound/2
            lx,rx=p_x-R_x,p_x+R_x
            ty,by=p_y+R_y,p_y-R_y

            for other_p in range(self.ped_num):
                if positions[other_p,0] == 0.:
                    continue
                if positions[other_p,0] == positions[p,0]:
                    continue
                other_x,other_y=positions[other_p,1],positions[other_p,2]
                if other_x>=rx or other_x<lx \
                    or other_y>=ty or other_y<by:
                    continue

                #如果当前的other_p是p的邻居,计算other_p在grid中的所处cell的坐标
                #print(other_x)
                #print(lx)
                cell_x=int(np.floor((other_x.cpu()-lx.cpu())/self.grid_cell_size))
                cell_y=int(np.floor((other_y.cpu()-by.cpu())/self.grid_cell_size))

                gridMask[:,p,other_p,cell_x+cell_y*self.grid_size]=1
        return gridMask

    def get_socialPooling(self,gridMask,prev_h_t):
        with torch.no_grad():
            H_t=torch.zeros(self.batch_size,self.ped_num,self.grid_size**2*self.hidden_dim,device=device)
            for i in range(self.ped_num):
                grid_it=gridMask[:,i,...]
                grid_it_T=grid_it.permute(0,2,1)#grid_it.t()#(batch_size,ped_num,grid_size**2)-->(batch_size,grid_size**2,ped_num)
                H_it=torch.matmul(grid_it_T,prev_h_t)#grid_it_T@prev_h_t#(batch_size,grid_size**2,ped_num)@(batch_size,ped_num,hidden_dim)-->(batch_size,grid_size**2,hidden_dim)
                H_it=H_it.reshape(self.batch_size,self.grid_size**2*self.hidden_dim)#(batch_size,grid_size**2,hidden_dim)-->(grid_size**2*hidden_dim)
                #H_t.append(H_it)
                H_t[:,i,...]=H_it
        return H_t

    #frame_data:(sep_len,ped_num,3)
    #利用所有行人的前8帧位置数据进行训练
    #在每一帧，对每个行人，输入其当前的坐标，预测出其下一个时间步的坐标，并计算出h_it以用于其他行人下一帧位置的预测
    def forward(self,x):
        output_obs=torch.zeros(self.batch_size,self.obs_len,self.ped_num,self.output_dim,device=device)
        h=torch.zeros(self.batch_size,self.obs_len,self.ped_num,self.hidden_dim,device=device)
        #c=torch.zeros(self.batch_size,self.obs_len,self.ped_num,self.hidden_dim,device=device)
        for t in range(self.obs_len):
            x_t = x[:,t, ...]  # x:(batch_size,obs_len,ped_num,3)
            h_t = torch.zeros(self.batch_size,self.ped_num, self.hidden_dim, device=device)
            o_t = torch.zeros(self.batch_size,self.ped_num, self.output_dim, device=device)

            if t == 0:
                prev_h_t = torch.zeros(self.batch_size,self.ped_num, self.hidden_dim, device=device)
                prev_h_t = self.Phi(self.prevEmbedding(prev_h_t))#relu+linear
            gridMask = self.get_grid_mask(x_t[0,...],self.width,self.height,t)#?
            H_t = self.get_socialPooling(gridMask, prev_h_t)#?

            for p in range(self.ped_num):
                posX_it = x_t[:,p, 1:]#x_t(batch_size,ped_num,2)
                e_it = self.Phi(self.inputEmbedding(posX_it))#e_it(batch_size,64)relu+linear
                H_it = H_t[:,p,...]
                a_it = self.Phi(self.socialEmbedding(H_it))#a_it(batch_size,128)relu+linear

                emb_it = torch.cat((e_it, a_it), dim=1)#concat
                #prev_states_it = [prev_h_t[p], prev_c_t[p]]

                sl_output, h_it = self.lstmLayer(emb_it)#lstm
                o_it = self.outlayer(sl_output)#linear
                h_t[:,p] = h_it
                #c_t[:,p] = c_it
                o_t[:,p] = o_it

            h[:,t]=h_t#pytorch可以这样赋值
            #c[:,t]=c_t
            output_obs[:,t]=o_t
            prev_h_t = h_t
            #prev_c_t = c_t


        x_obs_final=x[:,-1,...]#x_obs_final:(batch_size,ped_num,3)
        pid_obs_final=x_obs_final[:,:,0]#pid_obs_final:(batch_size,ped_num):最后一帧所有人的id
        pid_obs_final=torch.unsqueeze(pid_obs_final,dim=2)#pid_obs_final:(batch_size,ped_num,1)

        x_pred=torch.zeros(self.batch_size,self.pred_len,self.ped_num,3,device=device)
        o_pred=torch.zeros(self.batch_size,self.pred_len,self.ped_num,self.output_dim,device=device)
        for t in range(self.pred_len):
            if t==0:
                prev_o_t=output_obs[:,-1,:,:]#prev_o_t:(batch_size,ped_num,5)
            # if(torch.any(torch.isnan(prev_o_t))):
            #     print(t)
            pos_t_pred=normal2d_sample(prev_o_t)#pos_t_pred:(batch_size,ped_num,2)
            input_x_pred_t=torch.cat((pid_obs_final,pos_t_pred),dim=2)#input_x_pred_t(batch_size,ped_num,3)
            grid_t = self.get_grid_mask(input_x_pred_t[0],self.width,self.height,t)#计算mask

            h_t = torch.zeros(self.batch_size,self.ped_num,self.hidden_dim,device=device)
            o_t = torch.zeros(self.batch_size,self.ped_num,self.output_dim,device=device)

            H_t = self.get_socialPooling(grid_t,prev_h_t)#计算socialPooling

            for i in range(self.ped_num):
                #prev_o_it=prev_o_t[i,...]
                H_it = H_t[:,i,...]

                pos_it_pred = pos_t_pred[:,i,:]
                e_it = self.Phi(self.inputEmbedding(pos_it_pred))#linear
                a_it = self.Phi(self.socialEmbedding(H_it))#linear

                emb_it = torch.cat((e_it, a_it), dim=1)#连接操作
                #prev_states_it = [prev_h_t[p], prev_c_t[p]]
                sl_output, h_it = self.lstmLayer(emb_it)#lstm
                o_it = self.outlayer(sl_output)#linear

                h_t[:,i]=h_it
                #c_t[:,i]=c_it
                o_t[:,i]=o_it

            o_pred[:,t]=o_t
            #print(o_t.shape)
            #x_pred[:,t]=pos_it_pred
            x_pred[:,t]= input_x_pred_t

            prev_h_t=h_t
            #prev_c_t=c_t
            prev_o_t=o_t
        #print(o_pred.shape)
        return x_pred,o_pred




class Phi(nn.Module):
    ''' a non-linear layer'''

    def __init__(self, dropout_prob):
        super(Phi, self).__init__()
        self.dropout_prob = dropout_prob
        self.ReLU = nn.ReLU().cuda()
        self.Dropout = nn.Dropout(p=dropout_prob).cuda()

    def forward(self, x):
        return self.Dropout(self.ReLU(x))