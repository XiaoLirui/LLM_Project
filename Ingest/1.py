from datasets import load_dataset
import pandas as pd

# First time run this code, and it will download a json file 
dataset = load_dataset("taddeusb90/finbro-v0.1.0", cache_dir="../data")
dataset = pd.DataFrame(dataset)
dataset = pd.json_normalize(dataset['train'])


print(dataset)

