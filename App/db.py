import os
from pymongo import MongoClient
from datetime import datetime
# from zoneinfo import ZoneInfo
import pytz


RUN_TIMEZONE_CHECK = os.getenv('RUN_TIMEZONE_CHECK', '1') == '1'
TZ_INFO = os.getenv("TZ", "Asia/Shanghai")
tz = pytz.timezone(TZ_INFO)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB", "finance_qa")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

def init_db():
    db.conversations.drop()
    db.feedback.drop()

    # 创建索引
    db.conversations.create_index("id", unique=True)
    db.feedback.create_index("conversation_id")

def save_conversation(conversation_id, question, answer_data, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)

    conversation_data = {
        "id": conversation_id,
        "question": question,
        "answer": answer_data["answer"],
        "model_used": answer_data["model_used"],
        "response_time": answer_data["response_time"],
        "relevance": answer_data["relevance"],
        "relevance_explanation": answer_data["relevance_explanation"],
        "prompt_tokens": answer_data["prompt_tokens"],
        "completion_tokens": answer_data["completion_tokens"],
        "total_tokens": answer_data["total_tokens"],
        "eval_prompt_tokens": answer_data["eval_prompt_tokens"],
        "eval_completion_tokens": answer_data["eval_completion_tokens"],
        "eval_total_tokens": answer_data["eval_total_tokens"],
        "timestamp": timestamp
    }

    db.conversations.insert_one(conversation_data)

def save_feedback(conversation_id, feedback, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)

    feedback_data = {
        "conversation_id": conversation_id,
        "feedback": feedback,
        "timestamp": timestamp
    }

    db.feedback.insert_one(feedback_data)

def get_recent_conversations(limit=5, relevance=None):
    query = {}
    if relevance:
        query["relevance"] = relevance

    return list(db.conversations.find(query).sort("timestamp", -1).limit(limit))

def get_feedback_stats():
    thumbs_up = db.feedback.count_documents({"feedback": {"$gt": 0}})
    thumbs_down = db.feedback.count_documents({"feedback": {"$lt": 0}})

    return {"thumbs_up": thumbs_up, "thumbs_down": thumbs_down}

def check_timezone():
    py_time = datetime.now(tz)
    print(f"Python current time: {py_time}")

    # 测试插入和选择
    test_data = {
        "id": "test",
        "question": "test question",
        "answer": "test answer",
        "model_used": "test model",
        "response_time": 0.0,
        "relevance": "0.0",
        "relevance_explanation": "test explanation",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "eval_prompt_tokens": 0,
        "eval_completion_tokens": 0,
        "eval_total_tokens": 0,
        "openai_cost": 0.0,
        "timestamp": py_time
    }

    db.conversations.insert_one(test_data)

    inserted_time = db.conversations.find_one({"id": "test"})["timestamp"]
    print(f"Inserted time ({TZ_INFO}): {inserted_time.astimezone(tz)}")

    db.conversations.delete_one({"id": "test"})

if RUN_TIMEZONE_CHECK:
    check_timezone()
