import os
import numpy as np
import pandas as pd

def preprocess_frame_data():
    data_dir = "D:\\ImportSoft\\Workspaces\\torch_start_LSTM\\torch_start\\datasets\\eth\\hotel\\"
    homo_file=os.path.join(data_dir,"H.txt")
    obsmat_file=os.path.join(data_dir,"obsmat.txt")

    H=np.genfromtxt(homo_file)
    #print(H)
    obs_columns=["frame","pid","px","pz","py","vx","vz","vy"]
    obs_df=pd.DataFrame(np.genfromtxt(obsmat_file),columns=obs_columns)
    pos_df_raw=obs_df[["frame","pid","px","py"]]

    xy=np.array(pos_df_raw[["px","py"]])
    xy=world_to_image_space(xy,H)

    pos_df_preprocessed = pd.DataFrame({
        "frame": pos_df_raw["frame"],
        "pid": pos_df_raw["pid"],
        "x": xy[:, 0],
        "y": xy[:, 1]
    })
    return pos_df_preprocessed

def world_to_image_space(world_xy,H):
    world_xy=np.array(world_xy)
    world_xy1=np.concatenate([world_xy,np.ones((len(world_xy),1))],axis=1)
    image_xy1=np.linalg.inv(H).dot(world_xy1.T).T
    image_xy = image_xy1[:, :2] / np.expand_dims(image_xy1[:, 2], axis=1)
    return image_xy
