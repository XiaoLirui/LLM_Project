import json
from time import time
from openai import OpenAI
import ingest
from transformers import pipeline

client = OpenAI()
index = ingest.load_index()

def search(query):
    boost = {
        'input': 2.0,
        'instruction': 1.5,
        'output': 1.0,
        'source': 0.5
    }

    results = index.search(
        query=query, filter_dict={}, boost_dict=boost, num_results=10
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
    return prompt

llm = pipeline("question-answering", model="deepset/roberta-base-squad2")  

def rag(query):
    search_results = search(query)
    if not search_results:
        return "No relevant documents found."
    prompt = build_prompt(query, search_results)
    response = llm(question=query, context=prompt)
    return response['answer']


evaluation_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluate_relevance(question, answer):
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt, model="gpt-4o-mini")

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens


def rag(query, model="gpt-4o-mini"):
    t0 = time()

    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get("Explanation", "Failed to parse evaluation"),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
    }

    return answer_data
