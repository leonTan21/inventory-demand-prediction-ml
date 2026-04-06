import os
import pandas as pd

_cache = os.path.expanduser("~/.cache/kagglehub/datasets/yasserh/walmart-dataset/versions/1/Walmart.csv")

if os.path.exists(_cache):
    file_path = _cache
else:
    import kagglehub
    path = kagglehub.dataset_download("yasserh/walmart-dataset")
    file_path = os.path.join(path, os.listdir(path)[0])

df = pd.read_csv(file_path)
