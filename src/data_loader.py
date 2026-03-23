import kagglehub
import pandas as pd
import os

def load_walmart_data():
    path = kagglehub.dataset_download("yasserh/walmart-dataset")
    file_path = os.path.join(path, os.listdir(path)[0])
    return pd.read_csv(file_path)