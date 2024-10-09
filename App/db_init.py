import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def init_db():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri)
    
    # Access the MongoDB database
    db_name = os.getenv("MONGO_DB_NAME", "course_assistant")
    db = client[db_name]

    # Initialize collections and any necessary indexes
    if "conversations" not in db.list_collection_names():
        print("Creating 'conversations' collection...")
        db.create_collection("conversations")
    
    # Create indexes if needed (example: creating index on 'conversation_id')
    db.conversations.create_index("conversation_id", unique=True)
    print("Database initialization complete.")

if __name__ == "__main__":
    print("Initializing MongoDB database...")
    init_db()
