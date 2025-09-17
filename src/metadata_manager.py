import os
import hashlib
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv
from . import settings

load_dotenv()

COSMOS_ENDPOINT = os.getenv("AZURE_COSMO_URL")
COSMOS_KEY = os.getenv("AZURE_COSMO_KEY")
DATABASE_NAME = settings.DATABASE_NAME
METADATA_CONTAINER_NAME = settings.METADATA_CONTAINER_NAME

_cosmos_client = None
_cosmos_database = None
_metadata_container = None

def _initialize_cosmos_client():
    global _cosmos_client, _cosmos_database, _metadata_container
    
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        raise ValueError("Azure Cosmos DB credentials not found in environment variables")
    
    _cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    _cosmos_database = _cosmos_client.create_database_if_not_exists(id=DATABASE_NAME)
    
    _metadata_container = _cosmos_database.create_container_if_not_exists(
        id=METADATA_CONTAINER_NAME,
        partition_key=PartitionKey(path="/chunk_id")
    )

def generate_chunk_id(source_url: str, chunk_index: int) -> str:
    doc_id = hashlib.md5(source_url.encode()).hexdigest()
    return f"{doc_id}_chunk_{chunk_index}"

def add_chunk(chunk_id: str, source_url: str, total_chunks: int, language: str = "en"):
    global _cosmos_client, _cosmos_database, _metadata_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    metadata_item = {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "source_url": source_url,
        "total_chunks": total_chunks,
        "language": language,
        "created_at": os.popen('date /t').read().strip() if os.name == 'nt' else os.popen('date').read().strip()
    }
    
    try:
        _metadata_container.upsert_item(metadata_item)
    except Exception as e:
        print(f"Error storing metadata for chunk {chunk_id}: {e}")

def get_chunk_metadata(chunk_id: str):
    global _cosmos_client, _cosmos_database, _metadata_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        item = _metadata_container.read_item(item=chunk_id, partition_key=chunk_id)
        return item
    except Exception as e:
        print(f"Error retrieving metadata for chunk {chunk_id}: {e}")
        return None

def get_all_metadata():
    global _cosmos_client, _cosmos_database, _metadata_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        items = list(_metadata_container.read_all_items())
        return items
    except Exception as e:
        print(f"Error retrieving all metadata: {e}")
        return []

def get_metadata_by_source_url(source_url: str):
    global _cosmos_client, _cosmos_database, _metadata_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        query = "SELECT * FROM c WHERE c.source_url = @source_url"
        parameters = [{"name": "@source_url", "value": source_url}]
        
        items = list(_metadata_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        return items
    except Exception as e:
        print(f"Error retrieving metadata for source URL {source_url}: {e}")
        return []

def clear_all_metadata():
    global _cosmos_client, _cosmos_database, _metadata_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        items = list(_metadata_container.read_all_items())
        deleted_count = 0
        
        for item in items:
            try:
                _metadata_container.delete_item(item=item['id'], partition_key=item['chunk_id'])
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting metadata item {item['id']}: {e}")
        
        print(f"Cleared {deleted_count} metadata entries from Cosmos DB")
        return deleted_count
    except Exception as e:
        print(f"Error clearing metadata: {e}")
        return 0

def get_metadata_count() -> int:
    global _cosmos_client, _cosmos_database, _metadata_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        items = list(_metadata_container.read_all_items())
        return len(items)
    except Exception as e:
        print(f"Error counting metadata: {e}")
        return 0

def delete_metadata_by_source_url(source_url: str) -> int:
    global _cosmos_client, _cosmos_database, _metadata_container
    
    if _cosmos_client is None:
        _initialize_cosmos_client()
    
    try:
        items = get_metadata_by_source_url(source_url)
        deleted_count = 0
        
        for item in items:
            try:
                _metadata_container.delete_item(item=item['id'], partition_key=item['chunk_id'])
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting metadata item {item['id']}: {e}")
        
        print(f"Deleted {deleted_count} metadata entries for source URL: {source_url}")
        return deleted_count
    except Exception as e:
        print(f"Error deleting metadata for source URL {source_url}: {e}")
        return 0