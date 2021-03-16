from loadData import *
from functools import reduce
from pathlib import Path
import os
import pandas as pd
import numpy as np
import modelConfig
import torch

class dataLoader:
    def __init__(self,config:modelConfig):
        self.num_peds=config.ped_num
        self.seq_len=config.seq_len
        self.width=config.img_width
        self.height=config.img_height

    def preprocess_frame_data(self,data_dir):
        #data_dir = "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\hotel\\"
        homo_file = os.path.join(data_dir, "H.txt")
        obsmat_file = os.path.join(data_dir, "obsmat.txt")

        obs_columns = ["frame", "pid", "px","py"]
        obs_df = pd.DataFrame(np.genfromtxt(obsmat_file), columns=obs_columns)
        pos_df_raw = obs_df[["frame", "pid", "px", "py"]]

        xy = np.array(pos_df_raw[["px", "py"]])
        homo_file=Path(homo_file)
        if homo_file.exists():
            H = np.genfromtxt(homo_file)
            xy = world_to_image_space(xy, H)

        img_size=np.array([self.width,self.height])
        if data_dir=="D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\test\\eth\\":
            img_size=np.array([640,840])

        xy=xy/img_size

        pos_df_preprocessed = pd.DataFrame({
            "frame": pos_df_raw["frame"],
            "pid": pos_df_raw["pid"],
            "x": xy[:, 0],
            "y": xy[:, 1]
        })
        return pos_df_preprocessed

    def build_data(self,all_frames_data, seq_len):

        x_data = []
        y_data = []

        # 取出相邻两个序列中都存在的人
        for i in range(len(all_frames_data) - seq_len):
            cf_data = all_frames_data[i:i + seq_len, ...]
            nf_data = all_frames_data[i + 1:i + seq_len + 1, ...]

            ped_col_index = 0

            cf_ped_ids = reduce(set.intersection, [set(nf_ped_ids) for nf_ped_ids in cf_data[..., ped_col_index]])
            nf_ped_ids = reduce(set.intersection,
                                [set(nf_ped_ids) for nf_ped_ids in
                                 nf_data[..., ped_col_index]])

            ped_ids = list(cf_ped_ids & nf_ped_ids - {0})  # 相邻两条序列中都存在的行人的id
            if not ped_ids:
                continue

            x = np.zeros((seq_len, self.num_peds, 3))
            y = np.zeros((seq_len, self.num_peds, 3))

            # 逐帧对cf_data和nf_data中的数据进行处理：fi--索引，（cf,nf）--数据
            for fi, (cf, nf) in enumerate(zip(cf_data, nf_data)):
                for j, ped_id in enumerate(ped_ids):
                    cf_ped_row = cf[:, 0] == ped_id
                    nf_ped_row = nf[:, 0] == ped_id

                    if np.any(cf_ped_row):
                        x[fi, j, :] = cf[cf[:, 0] == ped_id]
                    if np.any(nf_ped_row):
                        y[fi, j, :] = nf[nf[:, 0] == ped_id]

            x_data.append(x)
            y_data.append(y)
        return x_data, y_data

    def world_to_image_space(self,world_xy, H):
        world_xy = np.array(world_xy)
        world_xy1 = np.concatenate([world_xy, np.ones((len(world_xy), 1))],
                                   axis=1)
        image_xy1 = np.linalg.inv(H).dot(world_xy1.T).T
        image_xy = image_xy1[:, :2] / np.expand_dims(image_xy1[:, 2], axis=1)
        return image_xy

    def get_train_test_data(self):
        data_dirs_train={
            "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\train\\hotel\\",
            "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\train\\students001\\",
            "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\train\\students003\\",
            "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\train\\univ\\",
            "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\train\\zara01\\",
            "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\train\\zara02\\",
            "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\test\\"
        }
        test_x_data = []
        test_y_data = []
        train_x_data = []
        train_y_data = []

        for dir in data_dirs_train:
            df=self.preprocess_frame_data(dir)
            all_frames=df["frame"].unique().tolist()
            num_frames=len(all_frames)
            all_frame_data=np.zeros((num_frames,self.num_peds,3),np.float64)

            for index,frame in enumerate(all_frames):
                peds_with_pos=np.array(df[df["frame"]==frame][["pid","x","y"]])
                n_peds=len(peds_with_pos)
                all_frame_data[index,0:n_peds,:]=peds_with_pos

            x_data,y_data=self.build_data(all_frame_data,self.seq_len)
            x_data=np.array(x_data)
            y_data=np.array(y_data)

            if dir=="D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\test\\":
                for i in range(x_data.shape[0]):
                    test_x_data.append(x_data[i])
                for i in range(y_data.shape[0]):
                    test_y_data.append(y_data[i])
            else:
                for i in range(x_data.shape[0]):
                    train_x_data.append(x_data[i])
                for i in range(y_data.shape[0]):
                    train_y_data.append(y_data[i])
        #for i in range(x_data.shape[0]):
            #if (i <= int(x_data.shape[0] * 0.8)):
                #train_x_data.append(x_data[i])
            #else:
                #test_x_data.append(x_data[i])

        #for i in range(y_data.shape[0]):
            #if (i <= int(y_data.shape[0] * 0.8)):
                #train_y_data.append(y_data[i])
            #else:
                #test_y_data.append(y_data[i])

        train_x_data = torch.Tensor(train_x_data).cuda() # (obs_len,ped_num,3)
        train_y_data = torch.Tensor(train_y_data).cuda()  # (pred_len,ped_num,3)
        test_x_data = torch.Tensor(test_x_data).cuda()
        test_y_data = torch.Tensor(test_y_data).cuda()

        train_data=(train_x_data,train_y_data)
        test_data=(test_x_data,test_y_data)

        return train_data,test_data

def split_data(obs_len, pred_len, *arrays):
    obs_len_arrays=[a[:, :obs_len, ...] for a in arrays]
    pred_len_arrays=[a[:, obs_len:obs_len+pred_len, ...] for a in arrays]
    return obs_len_arrays, pred_len_arrays
