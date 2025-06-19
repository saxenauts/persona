import openai
from typing import List, Dict, Any
import json
from server.config import config
from server.logging_config import get_logger

logger = get_logger(__name__)

# openai.api_key = config.MACHINE_LEARNING.OPENAI_API_KEY
openai_client = openai.Client(api_key=config.MACHINE_LEARNING.OPENAI_API_KEY)

def generate_embeddings(texts, model="text-embedding-3-small"):
    try:
        # Takes in a list of strings and returns a list of embeddings
        response = openai_client.embeddings.create(input=texts, model=model, dimensions=1536)
        embeddings = [data.embedding for data in response.data]
        
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [None] * len(texts)  # Return a list of Nones to maintain alignment with input texts
