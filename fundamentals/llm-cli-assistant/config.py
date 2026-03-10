import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL = os.getenv("MODEL", "qwen/qwen3.5-122b-a10b")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
