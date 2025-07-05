from typing import List
from .client_factory import get_embedding_client
from server.logging_config import get_logger

logger = get_logger(__name__)


def generate_embeddings(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using the configured LLM service.
    This is a synchronous function that wraps the async embedding generation.
    
    Args:
        texts: List of texts to generate embeddings for
        model: Model name (ignored, uses configured model)
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    try:
        import asyncio
        
        # Get the embedding client
        client = get_embedding_client()
        
        # Run the async embeddings function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to use a different approach
            # This is a fallback for sync usage in async contexts
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, client.embeddings(texts))
                return future.result()
        else:
            return asyncio.run(client.embeddings(texts))
            
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [None] * len(texts)  # Return a list of Nones to maintain alignment with input texts


async def generate_embeddings_async(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Async version of generate_embeddings.
    
    Args:
        texts: List of texts to generate embeddings for
        model: Model name (ignored, uses configured model)
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    try:
        client = get_embedding_client()
        return await client.embeddings(texts)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [None] * len(texts)  # Return a list of Nones to maintain alignment with input texts
