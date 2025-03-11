import numpy as np
from sklearn.model_selection import train_test_split
from example_lib.ex_utility import get_curvedata


class DataLoader:
    def __init__(self, train_data, valid_data=None, transform=None):
        self.rawdata = train_data
        self.valid_rawdata = valid_data
        self.transform = transform

        print("<DataLoader> init finished")

        return


    def raw_to_np(self, channel_wise=0, channel_mode=1, interp_period=10000):
        self.train_data, self.exp_info, self.ch_info, self.cy_info, _, _ = get_curvedata(self.rawdata, channel_wise, channel_mode, interp_period)

        if self.valid_rawdata:
            self.valid_data, self.valid_info, self.valid_ch, self.valid_cy, _, _ = get_curvedata(self.valid_rawdata, channel_wise, channel_mode, interp_period)
        else:
            self.valid_data, self.valid_info, self.valid_ch, self.valid_cy = [np.array([]) for _ in range(4)]

        return


    def split_data(self, val_size=0.2, seed=42):
        indices = range(self.train_data.shape[0])

        self.x_test, self.y_test, self.test_index, self.test_info, self.test_ch, self.test_cy = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.x_train, self.y_train, self.train_index, self.train_info, self.train_ch, self.train_cy = self.train_data, self.train_data, indices, self.exp_info, self.ch_info, self.cy_info
        
        if not self.valid_rawdata:
            self.x_train, self.x_val, self.y_train, self.y_val, self.train_index, self.valid_index, self.train_info, self.valid_info, self.train_ch, self.valid_ch, self.train_cy, self.valid_cy =\
                train_test_split(self.x_train, self.y_train, self.train_index, self.train_info, self.train_ch, self.train_cy, test_size=val_size, random_state=seed)
        else:
            shf = np.arange(self.valid_data.shape[0])

            rng = np.random.default_rng()
            shf = rng.permutation(shf)

            self.valid_data = self.valid_data[shf]
            self.valid_info = self.valid_info[shf]
            self.valid_ch = self.valid_ch[shf]
            self.valid_cy = self.valid_cy[shf]

            self.x_val, self.y_val, self.valid_index = self.valid_data, self.valid_data, range(self.valid_data.shape[0])

        print(f"<split_data> train_shape: {self.x_train.shape}, {self.y_train.shape}")
        print(f"<split_data> val_shape: {self.x_val.shape}, {self.y_val.shape}")

        return


    def dl_ready(self, channel_wise=0, channel_mode=1, interp_period=10000, val_size=0.2, seed=42):
        print("<dl_ready> dl_ready begins")

        self.raw_to_np(channel_wise, channel_mode, interp_period)
        print("<dl_ready> raw_to_np done")

        self.split_data(val_size, seed)
        print("<dl_ready> split_data done")

        return
