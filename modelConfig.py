import json

class modelConfig:
    def __init__(self, input_dim, hidden_dim, input_mid_dim, output_dim,
                 ped_num, grid_cell_size, grid_size, obs_len, pred_len,seq_len,img_width,img_height,batch_size,**kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_mid_dim = input_mid_dim
        self.output_dim = output_dim
        self.ped_num = ped_num
        self.grid_cell_size = grid_cell_size
        self.grid_size = grid_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size

    def load_model_config(config_file: str)->__init__:
        with open(config_file,"r") as f:
            config=json.load(f)

        return modelConfig(**config)





