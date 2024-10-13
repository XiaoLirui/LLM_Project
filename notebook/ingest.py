from datasets import load_dataset
import pandas as pd
import minsearch


def load_index():
# First time run this code, and it will download a json file 
    dataset = load_dataset("taddeusb90/finbro-v0.1.0", cache_dir="../data")
    dataset = pd.DataFrame(dataset)
    dataset = pd.json_normalize(dataset['train'])
    dataset = dataset.dropna(subset=['input', 'instruction','output'])

    dataset = dataset.head(10000)


    documents = dataset.to_dict(orient="records")

    index = minsearch.Index(
        text_fields=["input","instruction","output"],  
        keyword_fields=[],  
    )

    index.fit(documents)
    return index



