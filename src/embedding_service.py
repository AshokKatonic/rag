from langchain_openai import OpenAIEmbeddings
from . import settings

def create_embeddings_model(api_key):
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=api_key
    )