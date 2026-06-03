import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
DATA_SOURCE_PATH = os.getenv("DATA_SOURCE_PATH")

MODEL_NAME = "gemini-2.5-flash"