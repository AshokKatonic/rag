import os
import numpy as np
from azure.cosmos import CosmosClient, PartitionKey
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv
from . import settings

load_dotenv()

COSMOS_ENDPOINT = os.getenv("AZURE_COSMO_URL")
COSMOS_KEY = os.getenv("AZURE_COSMO_KEY")
DATABASE_NAME = settings.DATABASE_NAME
CONTAINER_NAME = settings.CONTAINER_NAME

if not COSMOS_ENDPOINT:
    raise ValueError("AZURE_COSMO_URL environment variable is required")
if not COSMOS_KEY:
    raise ValueError("AZURE_COSMO_KEY environment variable is required")

_cosmos_client = None
_cosmos_database = None
_cosmos_container = None

def _initialize_cosmos_client():
    global _cosmos_client, _cosmos_database, _cosmos_container
    
    try:
        _cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        print("Connected to Cosmos DB successfully")
        
        try:
            _cosmos_database = _cosmos_client.create_database_if_not_exists(id=DATABASE_NAME)
            print(f"Database '{DATABASE_NAME}' ready")
        except Exception as db_error:
            print(f"Database error: {db_error}")
            _cosmos_database = _cosmos_client.get_database_client(DATABASE_NAME)
            print(f"Using existing database '{DATABASE_NAME}'")
        
        try:
            _cosmos_container = _cosmos_database.create_container_if_not_exists(
                id=CONTAINER_NAME,
                partition_key=PartitionKey(path="/id")
            )
            print(f"Container '{CONTAINER_NAME}' ready")
        except Exception as container_error:
            print(f"Container creation error: {container_error}")
            try:
                _cosmos_container = _cosmos_database.get_container_client(CONTAINER_NAME)
                print(f"Using existing container '{CONTAINER_NAME}'")
            except Exception as get_error:
                print(f"Error accessing container: {get_error}")
                raise Exception("Failed to initialize Cosmos DB container")
                
    except Exception as e:
        print(f"Error initializing Cosmos DB: {e}")
        raise Exception(f"Failed to initialize Cosmos DB: {e}")

def _get_embedding(text: str, embeddings_model):
    try:
        response = embeddings_model.embed_query(text)
        return response
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise e

def create_vector_store(documents, embeddings_model):
    global _cosmos_client, _cosmos_database, _cosmos_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    print(f"Storing {len(documents)} documents in Cosmos DB...")
    
    stored_count = 0
    for i, doc in enumerate(documents):
        try:
            embedding = _get_embedding(doc.page_content, embeddings_model)
            
            source = doc.metadata.get('source', 'unknown')
            chunk_index = doc.metadata.get('chunk_index', i)
            safe_source = source.replace('/', '_').replace(':', '').replace('.', '_').replace('?', '_').replace('&', '_').replace('=', '_')
            doc_id = f"{safe_source}_chunk_{chunk_index}"
            
            cosmos_doc = {
                "id": doc_id,
                "content": doc.page_content,
                "vector": embedding,
                "source": source,
                "chunk_index": chunk_index,
                "metadata": doc.metadata
            }
            
            try:
                existing_doc = _cosmos_container.read_item(item=doc_id, partition_key=doc_id)
                print(f"Document already exists: {doc_id}")
                _cosmos_container.upsert_item(cosmos_doc)
                print(f"Updated document: {doc_id}")
            except Exception:
                _cosmos_container.create_item(cosmos_doc)
                print(f"Stored document: {doc_id}")
            
            stored_count += 1
            
        except Exception as e:
            print(f"Error storing document {i}: {e}")
            continue
    
    print(f"Successfully stored {stored_count} documents in Cosmos DB")
    return True

def similarity_search(query: str, embeddings_model, k: int = 3):
    global _cosmos_client, _cosmos_database, _cosmos_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        query_embedding = _get_embedding(query, embeddings_model)
        
        query_sql = "SELECT * FROM c"
        results = list(_cosmos_container.query_items(query_sql, enable_cross_partition_query=True))
        
        if not results:
            return []
        
        similarities = []
        for result in results:
            if 'vector' in result:
                vec1 = np.array(query_embedding)
                vec2 = np.array(result['vector'])
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    similarities.append((similarity, result))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:k]
        
        documents = []
        for similarity, result in top_results:
            doc = Document(
                page_content=result['content'],
                metadata={
                    'source': result.get('source', 'unknown'),
                    'chunk_index': result.get('chunk_index', 0),
                    'similarity': similarity,
                    **result.get('metadata', {})
                }
            )
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []

def create_retriever(embeddings_model, search_kwargs = None):
    if search_kwargs is None:
        search_kwargs = {"k": settings.TOP_K_CHUNKS}
    
    class CosmosRetriever(BaseRetriever):
        embeddings_model: OpenAIEmbeddings
        search_kwargs: dict
        
        def __init__(self, embeddings_model, search_kwargs):
            super().__init__(
                embeddings_model=embeddings_model,
                search_kwargs=search_kwargs
            )
        
        def _get_relevant_documents(self, query: str):
            k = self.search_kwargs.get("k", 3)
            return similarity_search(query, self.embeddings_model, k)
    
    return CosmosRetriever(embeddings_model, search_kwargs)

def clear_vector_store():
    global _cosmos_client, _cosmos_database, _cosmos_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        query_sql = "SELECT c.id FROM c"
        results = list(_cosmos_container.query_items(query_sql, enable_cross_partition_query=True))
        
        deleted_count = 0
        for result in results:
            try:
                _cosmos_container.delete_item(item=result['id'], partition_key=result['id'])
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting document {result['id']}: {e}")
        
        print(f"Deleted {deleted_count} documents from Cosmos DB")
        return deleted_count
        
    except Exception as e:
        print(f"Error clearing vector store: {e}")
        return 0

def get_document_count():
    global _cosmos_client, _cosmos_database, _cosmos_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        query_sql = "SELECT VALUE COUNT(1) FROM c"
        result = list(_cosmos_container.query_items(query_sql, enable_cross_partition_query=True))
        return result[0] if result else 0
    except Exception as e:
        print(f"Error getting document count: {e}")
        return 0


def load_vector_store(embeddings_model: OpenAIEmbeddings):
    doc_count = get_document_count()
    
    if doc_count > 0:
        print(f"Loaded Cosmos DB vector store with {doc_count} documents")
        return True
    else:
        print("No documents found in Cosmos DB")
        return False
