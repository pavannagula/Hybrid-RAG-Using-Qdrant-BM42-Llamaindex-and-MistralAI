from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient

# Load environmental variables from a .env file
load_dotenv()

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

qdrant_client = QdrantClient(
    url=Qdrant_URL,
    api_key=Qdrant_API_KEY
)