import os
import datetime
import pickle
import pandas as pd


def pkl_load(path, name):
    if name is None:
        name = datetime.now().strftime("%Y%m%d%H%M")

    file_name = os.path.join(path, name + ".pkl")

    try:
        with open(file_name, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise Exception(f"<pkl_load> Something went wrong! {e}")

    print(f"<pkl_load> File {file_name} is successfully loaded from {path}")

    return data


def pkl_save(path, name, data_list):
    if name is None:
        name = datetime.now().strftime("%Y%m%d%H%M")

    if os.path.exists(path) == False:
        os.makedirs(path)

    file_name = os.path.join(path, name + ".pkl")

    try:
        with open(file_name, "wb") as f:
            pickle.dump(data_list, f)
    except Exception as e:
        raise Exception(f"<pkl_save> Something went wrong! {e}")

    print(f"<pkl_save> File {file_name} is successfully saved at {path}")

    return


def load(file_path, file_name):
    input_Filename = os.path.join(file_path, file_name)
    batch = pickle.load(open(input_Filename,'rb'))

    bat_dict = {**batch}

    return bat_dict
