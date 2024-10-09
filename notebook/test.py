from datasets import load_dataset
import pandas as pd

# First time run this code, and it will download a json file 
dataset = load_dataset("taddeusb90/finbro-v0.1.0", cache_dir="../data")
dataset = pd.DataFrame(dataset)
dataset = pd.json_normalize(dataset['train'])
# dataset = dataset.head(10000)
dataset = dataset.dropna(subset=['input', 'instruction','output'])


import minsearch

index = minsearch.Index(
        text_fields=["input",'instruction','output'],  
        keyword_fields=[], 
)

documents = dataset.to_dict(orient="records")
# print(documents)
index.fit(documents)

from transformers import pipeline

llm = pipeline("question-answering", model="deepset/roberta-base-squad2")  
# llm = pipeline("text-generation", model="gpt2")  


def search(query):
    boost = {}
    
    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results

prompt_template = """
You're a financial analyst. Answer the QUESTION based on the CONTEXT from our finance database, and the output is for your reference.
Use only the facts from the CONTEXT when answering the QUESTION.
QUESTION: {question}
CONTEXT:{context}
""".strip()

entry_template = """
input: {input}
instruction: {instruction}
output: {output}
""".strip()

def build_prompt(query, search_results):
    context = ""
    
    for doc in search_results:
        context += entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    # print(prompt)
    return prompt

def rag(query):
    search_results = search(query)
    if not search_results:
        return "No relevant documents found."
    prompt = build_prompt(query, search_results)
    # print(llm(prompt,max_new_tokens=50,num_return_sequences=1,truncation=True, pad_token_id=50256))
    # response = llm(prompt,max_new_tokens=100,num_return_sequences=1,truncation=True, pad_token_id=50256)[0]['generated_text']
    
    response = llm(question=query, context=prompt)
    return response['answer']


question = 'What is APR?'
answer = rag(question)
print("Answer:",answer)