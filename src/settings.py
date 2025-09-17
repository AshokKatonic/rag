DATABASE_NAME = "vectordb"
CONTAINER_NAME = "documents"
METADATA_CONTAINER_NAME = "metadata"


CHUNK_SIZE = 512 
CHUNK_OVERLAP = 80

EMBEDDING_MODEL = "text-embedding-ada-002" 
LLM_MODEL = "gpt-4-turbo-preview"
TEMPERATURE = 0.0

TOP_K_CHUNKS = 3

RAG_TEMPLATE = """Use the following context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer that question."

Context: {context}

Question: {question}

Answer:"""