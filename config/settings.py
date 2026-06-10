import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "chroma_db"))
DATA_SOURCE_PATH = os.getenv("DATA_SOURCE_PATH", str(BASE_DIR / "data" / "laptop_kendari.csv"))

MODEL_NAME = "gemini-2.5-flash-lite"