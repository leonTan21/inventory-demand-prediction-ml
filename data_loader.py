import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("yasserh/walmart-dataset")
file_path = os.path.join(path, os.listdir(path)[0])
df = pd.read_csv(file_path)
