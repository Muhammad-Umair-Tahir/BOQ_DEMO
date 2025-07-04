import os
from agno.memory.v2 import Memory
from agno.memory.v2.db.mongodb import MongoMemoryDb
from agno.storage.mongodb import MongoDbStorage
from agno.models.google import Gemini
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb
import uuid
import json
from agno.workflow import WorkflowRunResponseEvent
from typing import Iterator, Optional
import dotenv

dotenv.load_dotenv()

# def shared_memory():
#     memory = Memory(db=MongoMemoryDb(db_name=os.getenv("MONGODB_NAME"),collection_name="viab_shared_memories", db_url=os.getenv("MONGODB_URL")), model=Gemini(os.getenv("MEMORY_MODEL", "gemini-2.5-flash"), api_key=os.getenv("MEMORY_MODEL_API_KEY")))
#     return memory

# def shared_storage():
#     return MongoDbStorage(db_name=os.getenv("MONGODB_NAME"), collection_name="viab_shared_storage", db_url=os.getenv("MONGODB_URL"))

async def  shared_memory():
    """
    Returns a shared memory instance using SQLite as the backend.
    """
    return Memory(db=SqliteMemoryDb(table_name="shared_memory", db_file=os.getenv("MEMORY_DB_FILE", "data/memoryDB/shared_memory.db")), model=Gemini(os.getenv("MEMORY_MODEL", "gemini-2.0-flash"), api_key=os.getenv("MEMORY_MODEL_API_KEY")))

async def shared_storage():
    """
    Returns a shared storage instance using SQLite as the backend.
    """
    return SqliteStorage(table_name="shared_storage", db_file=os.getenv("STORAGE_DB_FILE", "data/storageDB/shared_storage.db"))



def generate_user_id():
    """
    Generates a shorter ID using only the first 8 characters of UUID4.
    """
    return str(uuid.uuid4())[:8]

def generate_session_id():
    """
    Generates a unique session ID based on the current environment variable.
    """
    return str(uuid.uuid4())[:8]


def get_current_user():
    return generate_user_id()

def get_current_session():
    return generate_session_id()





def stream_text_response(events: Iterator[WorkflowRunResponseEvent]) -> Iterator[str]:
    """
    Stream the raw content of RunResponseEvent exactly as the terminal version would show it.
    """
    for event in events:
        if hasattr(event, "content") and isinstance(event.content, str):
            yield event.content  # no extra newline
            
    
def user_preference(key: str, value: str) -> str:
    """Save user preferences in key-value pair format to a JSON file.
    
    Args:
        key (str): The preference key
        value (str): The preference value
        
    Returns:
        str: Confirmation message
    """
    import json
    import os
    
    # Your implementation here
    preferences_file = "user_preferences.json"
    
    # Load existing preferences or create new dict
    if os.path.exists(preferences_file):
        with open(preferences_file, 'r') as f:
            preferences = json.load(f)
    else:
        preferences = {}
    
    # Update preferences
    preferences[key] = value
    
    # Save to file
    with open(preferences_file, 'w') as f:
        json.dump(preferences, f, indent=2)
    
    return f"Saved preference: {key} = {value}"



def get_boq(boq_output: str) -> str:
    """
    Save the BOQ output to a JSON file with a unique timestamped key.

    Args:
        boq_output (str): The BOQ output string to save.

    Returns:
        str: Confirmation message.
    """
    import json
    import os
    import datetime

    preferences_file = "boq_preferences.json"
    # Load existing preferences or create new dict
    if os.path.exists(preferences_file):
        with open(preferences_file, 'r') as f:
            preferences = json.load(f)
    else:
        preferences = {}

    # Use a unique key for each BOQ entry
    key = f"boq_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    preferences[key] = boq_output

    # Save to file
    with open(preferences_file, 'w') as f:
        json.dump(preferences, f, indent=2)

    return f"Saved BOQ output with key: {key}"


def read_boq_json() -> dict:
    """
    Read and return the contents of the user_preferences.json file as a dictionary.

    Returns:
        dict: The contents of the BOQ JSON file. Returns an empty dict if file does not exist.
    """
    import json
    import os

    preferences_file = "boq_preferences.json"
    if os.path.exists(preferences_file):
        with open(preferences_file, 'r') as f:
            return json.load(f)
    else:
        return {}



