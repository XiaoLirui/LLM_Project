import pandas as pd
import requests

df = pd.read_csv("./data/question_sample.csv")


sample_row = df.sample(n=1).iloc[0]

question = sample_row['input']
instruction = sample_row['instruction']

print("Question: ", question)
print("Instruction: ", instruction)
url = "http://localhost:5000/question"

data = {"question": question, "instruction": instruction}

response = requests.post(url, json=data)

print(response.json())
