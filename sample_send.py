import pandas as pd
import requests

df = pd.read_csv("./data/question_sample.csv")

sample_row = df.sample(n=1).iloc[0]

question = sample_row['input']
instructions = sample_row['instructions']

print("Question: ", question)
print("Instruction: ", instructions)
url = "http://localhost:5000/question"

data = {"question": question, "instructions": instructions}

response = requests.post(url, json=data)

print(response.json())
